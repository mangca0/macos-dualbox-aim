import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from macos_dualbox_aim.v63 import AIMBOT_V63_VERSION, AimbotConfigV63, AimbotV63
from macos_dualbox_aim.v63.crosshair import CrosshairDetector
from macos_dualbox_aim.v63.tuner import TUNABLE_FIELDS, WebTuner, _HTML


def _load_script_module():
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "main_v63.py"
    spec = importlib.util.spec_from_file_location("main_v63", script_path)
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


class V63ClassAwareTunerTests(unittest.TestCase):
    def test_v63_config_defaults_include_selectable_classes(self):
        config = AimbotConfigV63()

        self.assertEqual(config.version, AIMBOT_V63_VERSION)
        self.assertEqual(config.class_names, ["class_0", "class_1", "class_2", "class_3"])
        self.assertEqual(config.selected_class_ids, [0, 1, 2, 3])
        self.assertTrue(config.crosshair_enabled)

    def test_crosshair_detector_finds_green_centroid_in_center_crop(self):
        config = AimbotConfigV63(
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
        config = AimbotConfigV63(
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

    def test_v63_filters_out_unselected_detection_classes(self):
        config = AimbotConfigV63(
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
        aimbot = AimbotV63(config)
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

    def test_v63_allows_empty_selected_classes_and_returns_no_target(self):
        config = AimbotConfigV63(
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
        aimbot = AimbotV63(config)

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

    def test_v63_main_loads_class_info_from_model_inspection(self):
        module = _load_script_module()
        config = AimbotConfigV63(
            model_path="models/converted/cs2_fp16_fp16_fast.mlpackage",
            class_count=99,
            class_names=[f"class_{index}" for index in range(99)],
            selected_class_ids=[],
        )

        with (
            patch.object(module, "inspect_coreml_model_classes") as inspect_classes,
            patch.object(module, "RealtimeInferenceV63") as engine_cls,
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

    def test_v63_main_passes_detection_frame_to_aimbot_update(self):
        module = _load_script_module()
        config = AimbotConfigV63(enable_tuner=False)
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
            patch.object(module, "AimbotV63", return_value=aimbot),
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

    def test_v63_tuner_exposes_class_selection_fields(self):
        config = AimbotConfigV63(class_count=2, class_names=["enemy", "teammate"], selected_class_ids=[0])
        tuner = WebTuner(config, Path("/tmp/config_v63.json"))
        snapshot = tuner.snapshot()

        self.assertIn("selected_class_ids", TUNABLE_FIELDS)
        self.assertEqual(snapshot["options"]["classes"][0]["name"], "enemy")
        self.assertEqual(snapshot["config"]["selected_class_ids"], [0])
        self.assertIn("<title>Aimbot V6.3 Tuner</title>", _HTML)
        self.assertIn('id="class-list"', _HTML)

    def test_v63_tuner_applies_selected_classes_to_runtime(self):
        config = AimbotConfigV63(class_count=2, class_names=["enemy", "teammate"], selected_class_ids=[0])
        aimbot = Mock()
        tuner = WebTuner(config, Path("/tmp/config_v63.json"), aimbot=aimbot)

        tuner.update_config({"selected_class_ids": [1]})

        aimbot.update_selected_classes.assert_called_once_with([1])


if __name__ == "__main__":
    unittest.main()
