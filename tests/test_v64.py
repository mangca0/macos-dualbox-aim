import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from macos_dualbox_aim.v64 import AIMBOT_V64_VERSION, AimbotConfigV64, AimbotV64
from macos_dualbox_aim.core import HotkeyConfig, HotkeyMonitor
from macos_dualbox_aim.v64.controller import PIDController
from macos_dualbox_aim.v64.crosshair import CrosshairDetector
from macos_dualbox_aim.v64.tuner import TUNABLE_FIELDS, WebTuner, _HTML


def _load_script_module():
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "main_v64.py"
    spec = importlib.util.spec_from_file_location("main_v64", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_auto_tune_module():
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "auto_tune_v64.py"
    spec = importlib.util.spec_from_file_location("auto_tune_v64", script_path)
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


class V64ClassAwareTunerTests(unittest.TestCase):
    def test_v64_config_defaults_include_selectable_classes(self):
        config = AimbotConfigV64()

        self.assertEqual(config.version, AIMBOT_V64_VERSION)
        self.assertEqual(config.class_names, ["class_0", "class_1", "class_2", "class_3"])
        self.assertEqual(config.selected_class_ids, [0, 1, 2, 3])
        self.assertTrue(config.crosshair_enabled)
        self.assertTrue(config.stop_brake_enabled)

    def test_crosshair_detector_finds_green_centroid_in_center_crop(self):
        config = AimbotConfigV64(
            crosshair_enabled=True,
            crosshair_use_hsv=False,
            crosshair_target_r=0,
            crosshair_target_g=255,
            crosshair_target_b=0,
            crosshair_color_tolerance=0.0,
            crosshair_search_radius=12,
            crosshair_min_pixels=3,
        )
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[18, 24] = [0, 255, 0]
        image[19, 25] = [0, 255, 0]
        image[20, 26] = [0, 255, 0]

        result = CrosshairDetector(config).detect(image)

        self.assertTrue(result.found)
        self.assertAlmostEqual(result.crosshair_x, 25.0)
        self.assertAlmostEqual(result.crosshair_y, 19.0)
        self.assertAlmostEqual(result.offset_x, 5.0)
        self.assertAlmostEqual(result.offset_y, -1.0)

    def test_crosshair_detector_handles_default_radius_without_python_pixel_loop_cost(self):
        config = AimbotConfigV64(
            crosshair_enabled=True,
            crosshair_use_hsv=True,
            crosshair_search_radius=80,
            crosshair_min_pixels=3,
        )
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        image[155:158, 198:201] = [0, 255, 0]
        detector = CrosshairDetector(config)

        start = time.perf_counter()
        for _ in range(100):
            result = detector.detect(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertTrue(result.found)
        self.assertLess(elapsed_ms, 100.0)

    def test_v64_filters_out_unselected_detection_classes(self):
        config = AimbotConfigV64(
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
            max_speed=100.0,
            sensitivity=1.0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            pid_integral_gate_enabled=False,
            crosshair_enabled=True,
            crosshair_use_hsv=False,
            crosshair_target_r=0,
            crosshair_target_g=255,
            crosshair_target_b=0,
            crosshair_color_tolerance=0.0,
            crosshair_search_radius=12,
            crosshair_min_pixels=1,
        )
        aimbot = AimbotV64(config)
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

    def test_v64_allows_empty_selected_classes_and_returns_no_target(self):
        config = AimbotConfigV64(
            screen_width=100,
            screen_height=100,
            fov_width=40,
            fov_height=40,
            class_count=2,
            class_names=["enemy", "teammate"],
            selected_class_ids=[],
            tracker_generate=1,
            tracker_terminate=8,
            aim_offset_x=0.0,
            aim_offset_y=0.0,
            aim_offset_dynamic=False,
            crosshair_enabled=True,
            crosshair_use_hsv=False,
            crosshair_target_r=0,
            crosshair_target_g=255,
            crosshair_target_b=0,
            crosshair_color_tolerance=0.0,
            crosshair_search_radius=12,
            crosshair_min_pixels=1,
        )
        aimbot = AimbotV64(config)

        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[20, 20] = [0, 255, 0]

        target = aimbot.process_detection(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 0}],
            frame.shape,
            (30, 30),
            frame=frame,
        )

        self.assertEqual(config.selected_class_ids, [])
        self.assertEqual(aimbot._detections_to_objects(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 0}],
            40,
            40,
        ), [])
        self.assertIsNone(target)

    def test_v64_main_loads_class_info_from_model_inspection(self):
        module = _load_script_module()
        config = AimbotConfigV64(
            model_path="models/converted/cs2_fp16_fp16_fast.mlpackage",
            class_count=99,
            class_names=[f"class_{index}" for index in range(99)],
            selected_class_ids=[],
        )

        with (
            patch.object(module, "inspect_coreml_model_classes") as inspect_classes,
            patch.object(module, "RealtimeInferenceV64") as engine_cls,
        ):
            inspect_classes.return_value = Mock(
                class_count=2,
                class_names=("enemy", "teammate"),
            )
            module.build_engine(Path("/tmp/project"), config)

        self.assertEqual(config.class_count, 2)
        self.assertEqual(config.class_names, ["enemy", "teammate"])
        self.assertEqual(config.selected_class_ids, [])
        self.assertEqual(engine_cls.call_args.kwargs["class_count"], 2)

    def test_v64_main_passes_detection_frame_to_aimbot_update(self):
        module = _load_script_module()
        config = AimbotConfigV64(enable_tuner=False)
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
            patch.object(module, "AimbotV64", return_value=aimbot),
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

    def test_v64_tuner_exposes_class_selection_fields(self):
        config = AimbotConfigV64(class_count=2, class_names=["enemy", "teammate"], selected_class_ids=[0])
        tuner = WebTuner(config, Path("/tmp/config_v64.json"))
        snapshot = tuner.snapshot()

        self.assertIn("selected_class_ids", TUNABLE_FIELDS)
        self.assertEqual(snapshot["options"]["classes"][0]["name"], "enemy")
        self.assertEqual(snapshot["config"]["selected_class_ids"], [0])
        self.assertIn("<title>Aimbot V6.4 Tuner</title>", _HTML)
        self.assertIn('id="class-list"', _HTML)

    def test_v64_tuner_applies_selected_classes_to_runtime(self):
        config = AimbotConfigV64(class_count=2, class_names=["enemy", "teammate"], selected_class_ids=[0])
        aimbot = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v64.json"), aimbot=aimbot)

        tuner.update_config({"selected_class_ids": [1]})

        aimbot.update_selected_classes.assert_called_once_with([1])

    def test_v64_tuner_hot_updates_speed_and_integral_gate_params(self):
        config = AimbotConfigV64(
            max_speed=30.0,
            pid_integral_gate_threshold=50.0,
            pid_integral_gate_rate=0.025,
        )
        aimbot = Mock()
        aimbot.controller = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v64.json"), aimbot=aimbot)

        tuner.update_config({
            "max_speed": 123.0,
            "pid_integral_gate_threshold": 77.0,
            "pid_integral_gate_rate": 0.4,
        })

        kwargs = aimbot.controller.update_params.call_args.kwargs
        self.assertEqual(kwargs["max_speed"], 123.0)
        self.assertEqual(kwargs["pid_integral_gate_threshold"], 77.0)
        self.assertEqual(kwargs["pid_integral_gate_rate"], 0.4)

    def test_v64_tuner_hot_updates_stop_brake_params(self):
        config = AimbotConfigV64(
            stop_brake_radius=14.0,
            stop_brake_output_decay=0.4,
            stop_brake_pred_decay=0.25,
        )
        aimbot = Mock()
        aimbot.controller = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v64.json"), aimbot=aimbot)

        tuner.update_config({
            "stop_brake_enabled": False,
            "stop_brake_radius": 22.0,
            "stop_brake_output_decay": 0.2,
            "stop_brake_pred_decay": 0.1,
        })

        kwargs = aimbot.controller.update_params.call_args.kwargs
        self.assertFalse(kwargs["stop_brake_enabled"])
        self.assertEqual(kwargs["stop_brake_radius"], 22.0)
        self.assertEqual(kwargs["stop_brake_output_decay"], 0.2)
        self.assertEqual(kwargs["stop_brake_pred_decay"], 0.1)

    def test_v64_controller_brakes_high_output_when_target_suddenly_stops_near_crosshair(self):
        controller = PIDController(
            kp=0.45,
            ki=0.0,
            kd=0.0,
            max_speed=200.0,
            sensitivity=1.0,
            fov_radius=320,
            init_scale=1.0,
            ramp_time=0.001,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            target_jump_reset=0.0,
            pid_integral_gate_enabled=False,
            stop_brake_enabled=True,
            stop_brake_radius=18.0,
            stop_brake_output_decay=0.25,
            stop_brake_pred_decay=0.2,
            stop_brake_min_output=20.0,
        )

        controller.update(0.0, 0.0, 180.0, 0.0)
        fast_move, _ = controller.update(0.0, 0.0, 180.0, 0.0)
        stopped_move, _ = controller.update(0.0, 0.0, 8.0, 0.0)

        self.assertGreater(abs(fast_move), 70.0)
        self.assertLess(abs(stopped_move), 25.0)

    def test_v64_tuner_hidden_aim_active_control_is_not_in_html(self):
        config = AimbotConfigV64()
        aimbot = Mock()
        aimbot.is_active.return_value = False
        tuner = WebTuner(config, Path("/tmp/config_v64.json"), aimbot=aimbot)

        activated = tuner.set_aim_active(True)
        deactivated = tuner.set_aim_active(False)

        aimbot.activate.assert_called_once_with()
        aimbot.deactivate.assert_called_once_with()
        self.assertIn("aim_active", activated)
        self.assertIn("aim_active", deactivated)
        self.assertNotIn("/api/aim/active", _HTML)

    def test_v64_tuner_aim_active_control_uses_hotkey_override_when_available(self):
        config = AimbotConfigV64()
        aimbot = Mock()
        hotkey = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v64.json"), hotkey=hotkey, aimbot=aimbot)

        tuner.set_aim_active(True)
        tuner.set_aim_active(False)

        self.assertEqual([call.args[0] for call in hotkey.set_override_active.call_args_list], [True, False])
        aimbot.activate.assert_not_called()
        aimbot.deactivate.assert_not_called()

    def test_hotkey_override_keeps_aim_active_without_physical_trigger(self):
        aimbot = Mock()
        hotkey = HotkeyMonitor(HotkeyConfig(toggle_mode=False), aimbot=aimbot)

        hotkey.set_override_active(True)
        hotkey._check_trigger()
        hotkey.set_override_active(False)

        self.assertGreaterEqual(aimbot.on_activate.call_count, 1)
        aimbot.on_deactivate.assert_called_once_with()

    def test_auto_tune_v64_client_posts_hidden_active_state(self):
        module = _load_auto_tune_module()
        client = module.TunerClient("http://example.invalid")

        with patch.object(client, "_request", return_value={"aim_active": True}) as request:
            result = client.set_aim_active(True)

        request.assert_called_once_with("POST", "/api/aim/active", {"active": True})
        self.assertEqual(result, {"aim_active": True})

    def test_v64_records_aim_metrics_for_target_and_misses(self):
        config = AimbotConfigV64(
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
            max_speed=100.0,
            sensitivity=1.0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
            pid_integral_gate_enabled=False,
            crosshair_enabled=True,
            crosshair_use_hsv=False,
            crosshair_target_r=0,
            crosshair_target_g=255,
            crosshair_target_b=0,
            crosshair_color_tolerance=0.0,
            crosshair_search_radius=12,
            crosshair_min_pixels=1,
        )
        aimbot = AimbotV64(config)
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

    def test_v64_tuner_exposes_and_resets_aim_metrics(self):
        config = AimbotConfigV64()
        aimbot = Mock()
        aimbot.get_aim_metrics_snapshot.return_value = {"available": True, "samples": 3}
        tuner = WebTuner(config, Path("/tmp/config_v64.json"), aimbot=aimbot)

        snapshot = tuner.snapshot()
        reset = tuner.reset_aim_metrics()

        self.assertEqual(snapshot["aim"], {"available": True, "samples": 3})
        aimbot.reset_aim_metrics.assert_called_once_with()
        self.assertEqual(reset["aim"], {"available": True, "samples": 3})

    def test_auto_tune_v64_fields_exclude_non_controller_surfaces(self):
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
            "stop_brake_enabled",
        }

        self.assertFalse(excluded & set(module.AUTOTUNE_FIELDS))

    def test_auto_tune_v64_scores_penalize_loss_and_oscillation(self):
        module = _load_auto_tune_module()
        stable = {
            "available": True,
            "samples": 100,
            "target_lost_ratio": 0.0,
            "mean_abs_error": 4.0,
            "p95_abs_error": 8.0,
            "overshoot_count": 0,
            "oscillation_energy": 1.0,
            "mean_move": 2.0,
        }
        unstable = {
            "available": True,
            "samples": 100,
            "target_lost_ratio": 0.2,
            "mean_abs_error": 4.0,
            "p95_abs_error": 8.0,
            "overshoot_count": 5,
            "oscillation_energy": 10.0,
            "mean_move": 2.0,
        }

        self.assertLess(module.score_metrics(stable), module.score_metrics(unstable))

    def test_auto_tune_v64_score_prefers_strict_x_centering(self):
        module = _load_auto_tune_module()
        centered = {
            "available": True,
            "samples": 100,
            "target_lost_ratio": 0.0,
            "mean_abs_error": 4.0,
            "p95_abs_error": 8.0,
            "mean_abs_x_error": 1.0,
            "p95_abs_x_error": 2.0,
            "mean_abs_y_error": 3.0,
            "p95_abs_y_error": 6.0,
            "overshoot_count": 0,
            "oscillation_energy": 1.0,
            "mean_move": 2.0,
            "settled_ratio": 0.8,
        }
        horizontally_off_center = dict(centered)
        horizontally_off_center.update({
            "mean_abs_x_error": 8.0,
            "p95_abs_x_error": 14.0,
            "mean_abs_y_error": 1.0,
            "p95_abs_y_error": 2.0,
        })

        self.assertLess(module.score_metrics(centered), module.score_metrics(horizontally_off_center))

    def test_auto_tune_v64_score_prefers_x_dwell_and_low_bias(self):
        module = _load_auto_tune_module()
        stable_center = {
            "available": True,
            "samples": 100,
            "target_lost_ratio": 0.0,
            "mean_abs_error": 3.0,
            "p95_abs_error": 5.0,
            "mean_abs_x_error": 1.0,
            "p95_abs_x_error": 2.0,
            "p99_abs_x_error": 3.0,
            "mean_signed_x_error": 0.2,
            "x_center_dwell_ratio_1px": 0.7,
            "x_center_dwell_ratio_2px": 0.9,
            "x_crossing_count": 1,
            "mean_abs_y_error": 2.0,
            "p95_abs_y_error": 5.0,
            "overshoot_count": 0,
            "oscillation_energy": 1.0,
            "mean_move": 2.0,
            "settled_ratio": 0.8,
            "time_to_x_settle_ms": 120.0,
        }
        biased_and_twitchy = dict(stable_center)
        biased_and_twitchy.update({
            "mean_signed_x_error": 4.0,
            "x_center_dwell_ratio_1px": 0.1,
            "x_center_dwell_ratio_2px": 0.25,
            "x_crossing_count": 8,
            "time_to_x_settle_ms": 600.0,
        })

        self.assertLess(module.score_metrics(stable_center), module.score_metrics(biased_and_twitchy))

    def test_auto_tune_v64_combo_candidates_change_grouped_fields(self):
        module = _load_auto_tune_module()
        config = {
            "pid_kp": 0.6,
            "pid_ki": 0.02,
            "pid_kd": 0.2,
            "max_speed": 80.0,
            "sensitivity": 1.2,
            "init_scale": 0.4,
            "ramp_time": 0.15,
            "pred_weight_x": 0.4,
            "pred_weight_y": 0.3,
            "target_jump_reset": 90.0,
            "pid_integral_gate_threshold": 24.0,
            "pid_integral_gate_rate": 0.3,
            "stop_brake_radius": 18.0,
            "stop_brake_output_decay": 0.35,
            "stop_brake_pred_decay": 0.2,
            "stop_brake_min_output": 35.0,
        }

        candidates = module.generate_combo_candidates(
            config,
            trials=3,
            rng=module.random.Random(7),
            strength=0.12,
            shrink=0.75,
        )

        self.assertEqual([label for label, _candidate in candidates], ["pid", "response", "prediction"])
        for _label, candidate in candidates:
            self.assertGreater(len(candidate), 1)
            self.assertTrue(set(candidate) <= set(module.AUTOTUNE_FIELDS))
            for field, value in candidate.items():
                spec = module.FIELD_SPECS[field]
                self.assertGreaterEqual(value, spec.minimum)
                self.assertLessEqual(value, spec.maximum)

    def test_auto_tune_v64_mixed_combo_candidate_changes_cross_group_fields(self):
        module = _load_auto_tune_module()
        config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}

        candidate = module.generate_mixed_combo_candidate(
            config,
            rng=module.random.Random(63),
            strength=0.12,
            min_fields=5,
            max_fields=5,
        )

        touched_groups = {
            label
            for label, fields in module.COMBO_GROUPS
            if set(candidate) & set(fields)
        }
        self.assertGreaterEqual(len(candidate), 3)
        self.assertGreaterEqual(len(touched_groups), 2)
        self.assertTrue(set(candidate) <= set(module.AUTOTUNE_FIELDS))

    def test_auto_tune_v64_auto_iteration_candidates_mix_search_modes(self):
        module = _load_auto_tune_module()
        config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}

        candidates = module.generate_auto_iteration_candidates(
            config,
            rng=module.random.Random(63),
            strength=0.12,
            trials=6,
        )

        labels = [label for label, _candidate in candidates]
        self.assertIn("explore", labels)
        self.assertIn("exploit", labels)
        self.assertIn("refine", labels)
        for _label, candidate in candidates:
            self.assertTrue(set(candidate) <= set(module.AUTOTUNE_FIELDS))

    def test_auto_tune_v64_boundary_candidates_include_speed_and_integral_gate_fields(self):
        module = _load_auto_tune_module()
        config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}
        config.update({
            "max_speed": 200.0,
            "pid_integral_gate_threshold": 300.0,
            "pid_integral_gate_rate": 1.0,
        })

        max_speed_values = module.candidate_values(config, "max_speed")
        threshold_values = module.candidate_values(config, "pid_integral_gate_threshold")
        rate_values = module.candidate_values(config, "pid_integral_gate_rate")

        self.assertTrue(any(value < 200.0 for value in max_speed_values))
        self.assertTrue(any(value < 300.0 for value in threshold_values))
        self.assertTrue(any(value < 1.0 for value in rate_values))
        self.assertTrue({"max_speed", "pid_integral_gate_threshold", "pid_integral_gate_rate"} <= set(module.AUTOTUNE_FIELDS))

    def test_auto_tune_v64_evaluate_trial_uses_median_repeat_score(self):
        module = _load_auto_tune_module()
        args = Mock(warmup=0.0, duration=0.01, min_samples=1, repeats=3, fail_fast=True)

        with (
            patch.object(module, "run_trial", side_effect=[
                {"available": True, "samples": 10, "mean_abs_x_error": 9.0},
                {"available": True, "samples": 10, "mean_abs_x_error": 1.0},
                {"available": True, "samples": 10, "mean_abs_x_error": 5.0},
            ]),
            patch.object(module, "score_metrics", side_effect=[90.0, 10.0, 50.0]),
        ):
            trial = module.evaluate_trial(Mock(), args)

        self.assertEqual(trial["score"], 50.0)
        self.assertEqual(trial["repeat_scores"], [90.0, 10.0, 50.0])
        self.assertEqual(trial["metrics"]["mean_abs_x_error"], 5.0)

    def test_auto_tune_v64_combo_stage_accepts_whole_candidate_group(self):
        module = _load_auto_tune_module()

        class FakeClient:
            def __init__(self, _url, *, timeout_s):
                self.config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}
                self.updates = []
                self.saved = False

            def get_config(self):
                return {"config": dict(self.config)}

            def update_config(self, data):
                self.config.update(data)
                self.updates.append(dict(data))
                return {"config": dict(self.config)}

            def save_config(self):
                self.saved = True
                return {"ok": True}

        fake_client = FakeClient("http://example.invalid", timeout_s=1.0)
        seen_best_pid_kp = []

        def combo_candidate(config, _fields, *, rng, strength):
            seen_best_pid_kp.append(config["pid_kp"])
            if len(seen_best_pid_kp) == 1:
                return {"pid_kp": 0.7, "pid_kd": 0.25}
            return {"max_speed": 1.2, "sensitivity": 1.1}

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
            combo_trials=2,
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
            seed=63,
            fail_fast=True,
            restore_on_interrupt="original",
            auto_trigger=False,
            save_best=False,
        )

        with (
            patch.object(module, "TunerClient", return_value=fake_client),
            patch.object(module, "generate_candidates", return_value=[]),
            patch.object(module, "generate_group_combo_candidate", side_effect=combo_candidate),
            patch.object(module, "run_trial", side_effect=[{"available": True}, {"available": True}, {"available": True}]),
            patch.object(module, "score_metrics", side_effect=[100.0, 80.0, 75.0]),
            patch.object(module, "write_record"),
            patch.object(module, "_new_output_path", return_value=Path("/tmp/v64_auto_tune_test.jsonl")),
            patch("builtins.print"),
        ):
            result = module.run_search(args)

        self.assertEqual(result["best_config"]["pid_kp"], 0.7)
        self.assertEqual(result["best_config"]["pid_kd"], 0.25)
        self.assertEqual(result["best_config"]["max_speed"], 1.2)
        self.assertEqual(seen_best_pid_kp, [1.0, 0.7])
        self.assertIn({"pid_kp": 0.7, "pid_kd": 0.25}, fake_client.updates)
        self.assertEqual(fake_client.updates[-1]["pid_kp"], 0.7)

    def test_auto_tune_v64_adaptive_iterations_stop_after_plateau(self):
        module = _load_auto_tune_module()

        class FakeClient:
            def __init__(self):
                self.config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}
                self.updates = []

            def update_config(self, data):
                self.config.update(data)
                self.updates.append(dict(data))
                return {"config": dict(self.config)}

        args = Mock(
            min_improvement=0.01,
            seed=63,
            auto_iterations=5,
            auto_trials_per_iteration=2,
            auto_max_trials=0,
            auto_patience=2,
            auto_strength=0.12,
            auto_min_strength=0.01,
            auto_max_strength=0.30,
            auto_shrink=0.5,
            auto_expand=1.08,
        )
        candidates = [
            [("explore", {"pid_kp": 0.9}), ("refine", {"max_speed": 1.1})],
            [("explore", {"pid_kd": 0.8}), ("refine", {"sensitivity": 1.1})],
            [("explore", {"pid_ki": 0.1}), ("refine", {"ramp_time": 0.8})],
        ]

        with (
            patch.object(module, "generate_auto_iteration_candidates", side_effect=candidates) as generate,
            patch.object(module, "evaluate_candidate", side_effect=[
                {"accepted": False, "config": {}, "score": 101.0, "metrics": {}},
                {"accepted": False, "config": {}, "score": 102.0, "metrics": {}},
                {"accepted": False, "config": {}, "score": 103.0, "metrics": {}},
                {"accepted": False, "config": {}, "score": 104.0, "metrics": {}},
            ]),
            patch("builtins.print"),
        ):
            result = module.run_auto_iterations(
                FakeClient(),
                args,
                Path("/tmp/v64_auto_tune_test.jsonl"),
                {field: 1.0 for field in module.AUTOTUNE_FIELDS},
                100.0,
                {"available": True},
                original_config={field: 1.0 for field in module.AUTOTUNE_FIELDS},
                rng=module.random.Random(63),
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["best_score"], 100.0)

    def test_auto_tune_v64_adaptive_iterations_continue_from_accepted_best(self):
        module = _load_auto_tune_module()

        class FakeClient:
            def __init__(self):
                self.config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}
                self.updates = []

            def update_config(self, data):
                self.config.update(data)
                self.updates.append(dict(data))
                return {"config": dict(self.config)}

        args = Mock(
            min_improvement=0.01,
            seed=63,
            auto_iterations=2,
            auto_trials_per_iteration=1,
            auto_max_trials=0,
            auto_patience=2,
            auto_strength=0.12,
            auto_min_strength=0.01,
            auto_max_strength=0.30,
            auto_shrink=0.5,
            auto_expand=1.08,
        )
        seen_configs = []

        def candidates(config, *, rng, strength, trials):
            seen_configs.append(dict(config))
            return [("explore", {"pid_kp": 0.7 if len(seen_configs) == 1 else 0.6})]

        with (
            patch.object(module, "generate_auto_iteration_candidates", side_effect=candidates),
            patch.object(module, "evaluate_candidate", side_effect=[
                {"accepted": True, "config": {"pid_kp": 0.7}, "score": 80.0, "metrics": {"available": True}},
                {"accepted": True, "config": {"pid_kp": 0.6}, "score": 70.0, "metrics": {"available": True}},
            ]),
            patch("builtins.print"),
        ):
            result = module.run_auto_iterations(
                FakeClient(),
                args,
                Path("/tmp/v64_auto_tune_test.jsonl"),
                {field: 1.0 for field in module.AUTOTUNE_FIELDS},
                100.0,
                {"available": True},
                original_config={field: 1.0 for field in module.AUTOTUNE_FIELDS},
                rng=module.random.Random(63),
            )

        self.assertEqual(seen_configs[0]["pid_kp"], 1.0)
        self.assertEqual(seen_configs[1]["pid_kp"], 0.7)
        self.assertEqual(result["best_score"], 70.0)

    def test_auto_tune_v64_interrupt_restores_original_config(self):
        module = _load_auto_tune_module()

        class FakeClient:
            def __init__(self, _url, *, timeout_s):
                self.config = {field: 1.0 for field in module.AUTOTUNE_FIELDS}
                self.updates = []

            def get_config(self):
                return {"config": dict(self.config)}

            def update_config(self, data):
                self.config.update(data)
                self.updates.append(dict(data))
                return {"config": dict(self.config)}

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
            seed=63,
            fail_fast=True,
            restore_on_interrupt="original",
            auto_trigger=False,
            save_best=False,
        )

        with (
            patch.object(module, "TunerClient", return_value=fake_client),
            patch.object(module, "generate_candidates", return_value=[{"pid_kp": 0.7}]),
            patch.object(module, "evaluate_trial", side_effect=[
                {"score": 100.0, "metrics": {"available": True}, "repeat_scores": [100.0], "repeat_metrics": [], "hard_failure": False},
                KeyboardInterrupt,
            ]),
            patch.object(module, "write_record"),
            patch.object(module, "_new_output_path", return_value=Path("/tmp/v64_auto_tune_test.jsonl")),
            patch("builtins.print"),
        ):
            result = module.run_search(args)

        self.assertTrue(result["interrupted"])
        self.assertEqual(fake_client.updates[-1], {field: 1.0 for field in module.AUTOTUNE_FIELDS})

    def test_auto_tune_v64_auto_trigger_enables_and_force_disables_aim(self):
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
            seed=63,
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
            patch.object(module, "_new_output_path", return_value=Path("/tmp/v64_auto_tune_test.jsonl")),
            patch("builtins.print"),
        ):
            result = module.run_search(args)

        self.assertEqual(fake_client.active_calls, [True, False])
        self.assertTrue(result["auto_trigger"])

    def test_auto_tune_v64_auto_trigger_force_disables_after_interrupt(self):
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
            seed=63,
            fail_fast=True,
            restore_on_interrupt="original",
            auto_trigger=True,
            save_best=False,
        )

        with (
            patch.object(module, "TunerClient", return_value=fake_client),
            patch.object(module, "generate_candidates", return_value=[{"pid_kp": 0.7}]),
            patch.object(module, "evaluate_trial", side_effect=[
                {"score": 100.0, "metrics": {"available": True}, "repeat_scores": [100.0], "repeat_metrics": [], "hard_failure": False},
                KeyboardInterrupt,
            ]),
            patch.object(module, "write_record"),
            patch.object(module, "_new_output_path", return_value=Path("/tmp/v64_auto_tune_test.jsonl")),
            patch("builtins.print"),
        ):
            result = module.run_search(args)

        self.assertTrue(result["interrupted"])
        self.assertEqual(fake_client.active_calls, [True, False])


if __name__ == "__main__":
    unittest.main()
