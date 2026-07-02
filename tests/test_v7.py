import importlib.util
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from macos_dualbox_aim.v7 import (
    AIMBOT_V7_VERSION,
    AimController,
    AimbotConfigV7,
    AimbotV7,
    IncrementalPid,
    PerlinNoise1D,
)
from macos_dualbox_aim.v7.tuner import TUNABLE_FIELDS, WebTuner, _HTML


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "main_v7.py"
    spec = importlib.util.spec_from_file_location("main_v7", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_auto_tune_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "auto_tune_v7.py"
    spec = importlib.util.spec_from_file_location("auto_tune_v7", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeKmbox:
    def __init__(self):
        self.moves = []

    def mouse_move(self, x: int, y: int) -> int:
        self.moves.append((x, y))
        return 0


class V7IncrementalPidRuntimeTests(unittest.TestCase):
    def test_v7_config_defaults_enable_strict_controller_replica(self):
        config = AimbotConfigV7()

        self.assertEqual(config.version, AIMBOT_V7_VERSION)
        self.assertEqual(config.output_max, config.max_speed)
        self.assertEqual(config.noise_amp, 0.0)
        self.assertFalse(config.pid_integral_gate_enabled)

    def test_perlin_noise_is_deterministic_and_bounded(self):
        first = PerlinNoise1D(seed=12345)
        second = PerlinNoise1D(seed=12345)

        samples = [first.noise(index * 0.25) for index in range(12)]

        self.assertEqual(samples, [second.noise(index * 0.25) for index in range(12)])
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in samples))

    def test_incremental_pid_matches_formula_deadzone_and_output_damping(self):
        pid = IncrementalPid(kp=1.0, ki=0.5, kd=0.25)

        self.assertEqual(pid.update(0.2), 0.0)

        first = pid.update(10.0, scale=0.5)
        second = pid.update(8.0, scale=1.0)

        self.assertAlmostEqual(first, 8.75)
        self.assertAlmostEqual(second, 7.75)

    def test_aim_controller_returns_debug_output_and_noise_can_be_disabled(self):
        controller = AimController()
        controller.configure_pid(1.0, 0.0, 0.0)

        first = controller.update(
            raw_x=12.0,
            raw_y=-6.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            init_scale=1.0,
            ramp_time=0.001,
            output_max=100.0,
            noise_amp=0.0,
        )
        second = controller.update(
            raw_x=10.0,
            raw_y=-5.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            init_scale=1.0,
            ramp_time=0.001,
            output_max=100.0,
            noise_amp=0.0,
        )

        self.assertEqual(first.predicted_x, 0.0)
        self.assertEqual(first.predicted_y, 0.0)
        self.assertAlmostEqual(first.fused_x, 12.0)
        self.assertAlmostEqual(first.fused_y, -6.0)
        self.assertGreater(first.curve_len, 0.0)
        self.assertLess(second.move_x, first.move_x)

    def test_aim_controller_resets_on_target_jump(self):
        controller = AimController()
        controller.configure_pid(1.0, 0.0, 0.0)

        controller.update(0.0, 0.0, 0.5, 0.5, 1.0, 0.001, 100.0, 0.0)
        before = controller.pid_x.output
        controller.update(100.0, 0.0, 0.5, 0.5, 1.0, 0.001, 100.0, 0.0)

        self.assertNotEqual(before, controller.pid_x.output)
        self.assertEqual(controller.pid_x.previous_previous_error, 0.0)

    def test_v7_aimbot_uses_crosshair_and_selected_class_chain(self):
        config = AimbotConfigV7(
            screen_width=100,
            screen_height=100,
            fov_width=40,
            fov_height=40,
            class_count=2,
            class_names=["enemy", "teammate"],
            selected_class_ids=[0],
            tracker_generate=1,
            tracker_terminate=8,
            aim_offset_x=0.0,
            aim_offset_y=0.0,
            aim_offset_dynamic=False,
            pid_kp=1.0,
            pid_ki=0.0,
            pid_kd=0.0,
            output_max=100.0,
            noise_amp=0.0,
            sensitivity=1.0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            crosshair_enabled=True,
            crosshair_use_hsv=False,
            crosshair_target_r=0,
            crosshair_target_g=255,
            crosshair_target_b=0,
            crosshair_color_tolerance=0.0,
            crosshair_search_radius=12,
            crosshair_min_pixels=1,
        )
        aimbot = AimbotV7(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[20, 20] = [0, 255, 0]

        ignored = aimbot.process_detection(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 1}],
            frame.shape,
            (30, 30),
            frame=frame,
        )
        accepted_first = aimbot.process_detection(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 0}],
            frame.shape,
            (30, 30),
            frame=frame,
        )
        accepted_second = aimbot.process_detection(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 0}],
            frame.shape,
            (30, 30),
            frame=frame,
        )
        self.assertIsNone(ignored)
        self.assertIsNone(accepted_first)
        self.assertIsNotNone(accepted_second)

    def test_v7_main_passes_detection_frame_to_aimbot_update(self):
        module = _load_script_module()
        config = AimbotConfigV7(enable_tuner=False)
        aimbot = Mock()
        engine = Mock()
        engine.crop_offset = (10, 20)
        engine.on_detection = None
        hotkey = Mock()
        result = Mock()
        result.detections = [{"bbox": [1, 2, 3, 4]}]
        result.latency_ms = {}
        result.frame = np.zeros((4, 4, 3), dtype=np.uint8)

        with (
            patch.object(module, "load_config", return_value=config),
            patch.object(module, "AimbotV7", return_value=aimbot),
            patch.object(module, "HotkeyMonitor", return_value=hotkey),
            patch.object(module, "build_engine", return_value=engine),
            patch("builtins.print"),
        ):
            aimbot.connect.return_value = True
            hotkey.connect.return_value = True
            engine.start.side_effect = KeyboardInterrupt
            module.main()

        assert engine.on_detection is not None
        engine.on_detection(result)
        aimbot.update.assert_called_with(
            result.detections,
            (config.fov_height, config.fov_width),
            engine.crop_offset,
            timing_ms=result.latency_ms,
            frame=result.frame,
        )

    def test_v7_tuner_exposes_noise_output_and_controller_debug(self):
        config = AimbotConfigV7(class_count=2, class_names=["enemy", "teammate"], selected_class_ids=[0])
        aimbot = Mock()
        aimbot.latest_aim_output = Mock(
            move_x=1.0,
            move_y=2.0,
            curve_len=3.0,
            predicted_x=4.0,
            predicted_y=5.0,
            fused_x=6.0,
            fused_y=7.0,
        )
        tuner = WebTuner(config, Path("/tmp/config_v7.json"), aimbot=aimbot)
        snapshot = tuner.snapshot()

        self.assertIn("noise_amp", TUNABLE_FIELDS)
        self.assertIn("output_max", TUNABLE_FIELDS)
        self.assertEqual(snapshot["config"]["noise_amp"], 0.0)
        self.assertEqual(snapshot["controller"]["predicted_x"], 4.0)
        self.assertIn("<title>Aimbot V7 Tuner</title>", _HTML)
        self.assertIn('id="noise_amp"', _HTML)

    def test_v7_tuner_hidden_aim_active_control_is_not_in_html(self):
        config = AimbotConfigV7()
        aimbot = Mock()
        aimbot.is_active.return_value = False
        tuner = WebTuner(config, Path("/tmp/config_v7.json"), aimbot=aimbot)

        activated = tuner.set_aim_active(True)
        deactivated = tuner.set_aim_active(False)

        aimbot.activate.assert_called_once_with()
        aimbot.deactivate.assert_called_once_with()
        self.assertIn("aim_active", activated)
        self.assertIn("aim_active", deactivated)
        self.assertNotIn("/api/aim/active", _HTML)

    def test_v7_tuner_aim_active_control_uses_hotkey_override_when_available(self):
        config = AimbotConfigV7()
        aimbot = Mock()
        hotkey = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v7.json"), hotkey=hotkey, aimbot=aimbot)

        tuner.set_aim_active(True)
        tuner.set_aim_active(False)

        self.assertEqual([call.args[0] for call in hotkey.set_override_active.call_args_list], [True, False])
        aimbot.activate.assert_not_called()
        aimbot.deactivate.assert_not_called()

    def test_v7_tuner_applies_max_speed_to_runtime_output_limit(self):
        config = AimbotConfigV7(max_speed=44.0, output_max=99.0)
        aimbot = Mock()
        aimbot.controller = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v7.json"), aimbot=aimbot)

        tuner.update_config({"max_speed": 55.0})

        kwargs = aimbot.controller.update_params.call_args.kwargs
        self.assertEqual(kwargs["max_speed"], 55.0)
        self.assertEqual(kwargs["output_max"], 55.0)

        tuner.update_config({"max_speed": 60.0, "output_max": 80.0})

        kwargs = aimbot.controller.update_params.call_args.kwargs
        self.assertEqual(kwargs["max_speed"], 60.0)
        self.assertEqual(kwargs["output_max"], 80.0)

    def test_v7_records_aim_metrics_for_target_and_misses(self):
        config = AimbotConfigV7(
            screen_width=100,
            screen_height=100,
            fov_width=40,
            fov_height=40,
            tracker_generate=1,
            tracker_terminate=8,
            aim_offset_x=0.0,
            aim_offset_y=0.0,
            aim_offset_dynamic=False,
            pid_kp=1.0,
            pid_ki=0.0,
            pid_kd=0.0,
            output_max=100.0,
            noise_amp=0.0,
            sensitivity=1.0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            crosshair_enabled=True,
            crosshair_use_hsv=False,
            crosshair_target_r=0,
            crosshair_target_g=255,
            crosshair_target_b=0,
            crosshair_color_tolerance=0.0,
            crosshair_search_radius=12,
            crosshair_min_pixels=1,
        )
        aimbot = AimbotV7(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        self.assertTrue(aimbot.aim_at_target(Mock(aim_x=20.0, aim_y=0.0, track_id=7)))
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[20, 20] = [0, 255, 0]
        aimbot.update([], frame.shape, (30, 30), frame=frame)

        metrics = aimbot.get_aim_metrics_snapshot()

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["target_found_samples"], 1)
        self.assertEqual(metrics["target_lost_samples"], 1)
        self.assertEqual(metrics["latest"]["track_id"], 7)
        self.assertAlmostEqual(metrics["mean_abs_error"], 20.0)
        self.assertAlmostEqual(metrics["mean_abs_x_error"], 20.0)
        self.assertAlmostEqual(metrics["mean_abs_y_error"], 0.0)
        self.assertIn("p99_abs_x_error", metrics)
        self.assertIn("x_center_dwell_ratio_1px", metrics)
        self.assertIn("x_center_dwell_ratio_2px", metrics)
        self.assertIn("x_crossing_count", metrics)
        self.assertIn("time_to_x_settle_ms", metrics)
        self.assertGreater(metrics["mean_move"], 0.0)

    def test_v7_tuner_exposes_and_resets_aim_metrics(self):
        config = AimbotConfigV7()
        aimbot = Mock()
        aimbot.get_aim_metrics_snapshot.return_value = {"available": True, "samples": 3}
        tuner = WebTuner(config, Path("/tmp/config_v7.json"), aimbot=aimbot)

        snapshot = tuner.snapshot()
        reset = tuner.reset_aim_metrics()

        self.assertEqual(snapshot["aim"], {"available": True, "samples": 3})
        aimbot.reset_aim_metrics.assert_called_once_with()
        self.assertEqual(reset["aim"], {"available": True, "samples": 3})

    def test_auto_tune_v7_fields_exclude_non_controller_surfaces(self):
        module = _load_auto_tune_module()
        excluded = {
            "selected_class_ids",
            "confidence_threshold",
            "iou_threshold",
            "aim_offset_x",
            "aim_offset_y",
            "aim_offset_dynamic",
            "tracker_generate",
            "tracker_terminate",
            "tracker_vx_noise",
            "tracker_vy_noise",
            "tracker_w_noise",
            "tracker_h_noise",
            "tracker_r_std",
            "trigger_button",
            "trigger_button_secondary",
            "crosshair_enabled",
            "crosshair_search_radius",
            "crosshair_min_pixels",
            "crosshair_use_hsv",
            "crosshair_h_min",
            "crosshair_h_max",
            "crosshair_s_min",
            "crosshair_s_max",
            "crosshair_v_min",
            "crosshair_v_max",
            "crosshair_target_r",
            "crosshair_target_g",
            "crosshair_target_b",
            "crosshair_color_tolerance",
            "output_max",
            "noise_amp",
        }

        self.assertFalse(excluded & set(module.AUTOTUNE_FIELDS))
        self.assertTrue({
            "max_speed",
            "pid_integral_gate_threshold",
            "pid_integral_gate_rate",
        } <= set(module.AUTOTUNE_FIELDS))

    def test_auto_tune_v7_score_prefers_strict_x_centering(self):
        module = _load_auto_tune_module()
        centered = {
            "available": True,
            "samples": 100,
            "target_lost_ratio": 0.0,
            "mean_abs_error": 4.0,
            "p95_abs_error": 8.0,
            "mean_abs_x_error": 1.0,
            "p95_abs_x_error": 2.0,
            "p99_abs_x_error": 3.0,
            "mean_signed_x_error": 0.2,
            "x_center_dwell_ratio_1px": 0.7,
            "x_center_dwell_ratio_2px": 0.9,
            "x_crossing_count": 1,
            "mean_abs_y_error": 3.0,
            "p95_abs_y_error": 6.0,
            "overshoot_count": 0,
            "oscillation_energy": 1.0,
            "mean_move": 2.0,
            "settled_ratio": 0.8,
            "time_to_x_settle_ms": 120.0,
        }
        horizontally_off_center = dict(centered)
        horizontally_off_center.update({
            "mean_abs_x_error": 8.0,
            "p95_abs_x_error": 14.0,
            "mean_abs_y_error": 1.0,
            "p95_abs_y_error": 2.0,
            "mean_signed_x_error": 4.0,
            "x_center_dwell_ratio_1px": 0.1,
            "x_center_dwell_ratio_2px": 0.25,
            "x_crossing_count": 8,
            "time_to_x_settle_ms": 600.0,
        })

        self.assertLess(module.score_metrics(centered), module.score_metrics(horizontally_off_center))

    def test_auto_tune_v7_client_posts_hidden_active_state(self):
        module = _load_auto_tune_module()
        client = module.TunerClient("http://example.invalid")

        with patch.object(client, "_request", return_value={"aim_active": True}) as request:
            result = client.set_aim_active(True)

        request.assert_called_once_with("POST", "/api/aim/active", {"active": True})
        self.assertEqual(result, {"aim_active": True})

    def test_auto_tune_v7_auto_trigger_enables_and_force_disables_aim(self):
        module = _load_auto_tune_module()

        class FakeClient:
            def __init__(self, _url, *, timeout_s):
                self.config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}
                self.active_calls = []
                self.updates = []

            def get_config(self):
                return {"config": dict(self.config)}

            def update_config(self, data):
                self.config.update(data)
                self.updates.append(dict(data))
                return {"config": dict(self.config)}

            def set_aim_active(self, active):
                self.active_calls.append(bool(active))
                return {"aim_active": bool(active)}

            def save_config(self):
                return {"ok": True}

        fake_client = FakeClient("http://example.invalid", timeout_s=1.0)
        args = Mock(
            url="http://example.invalid",
            timeout=1.0,
            warmup=0.0,
            duration=0.01,
            min_samples=1,
            out_dir="/tmp",
            passes=1,
            min_improvement=0.01,
            repeats=1,
            combo_trials=0,
            mixed_trials=0,
            mixed_shrink_every=4,
            auto_iterations=0,
            auto_trials_per_iteration=12,
            auto_max_trials=0,
            auto_patience=3,
            auto_strength=0.12,
            auto_min_strength=0.01,
            auto_max_strength=0.30,
            auto_shrink=0.55,
            auto_expand=1.08,
            combo_strength=0.12,
            combo_shrink=0.75,
            seed=7,
            fail_fast=True,
            restore_on_interrupt="original",
            auto_trigger=True,
            save_best=False,
        )

        with (
            patch.object(module, "TunerClient", return_value=fake_client),
            patch.object(module, "generate_candidates", return_value=[]),
            patch.object(module, "evaluate_trial", return_value={
                "score": 100.0,
                "metrics": {"available": True},
                "repeat_scores": [100.0],
                "repeat_metrics": [],
                "hard_failure": False,
            }),
            patch.object(module, "write_record"),
            patch.object(module, "_new_output_path", return_value=Path("/tmp/v7_auto_tune_test.jsonl")),
            patch("builtins.print"),
        ):
            result = module.run_search(args)

        self.assertEqual(fake_client.active_calls, [True, False])
        self.assertTrue(result["auto_trigger"])


if __name__ == "__main__":
    unittest.main()
