import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from macos_dualbox_aim.v62 import AIMBOT_V62_VERSION, AimbotConfigV62, AimbotV62
from macos_dualbox_aim.v62.crosshair import CrosshairDetector
from macos_dualbox_aim.v62.tuner import TUNABLE_FIELDS, WebTuner, _HTML


def _load_script_module():
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "main_v62.py"
    spec = importlib.util.spec_from_file_location("main_v62", script_path)
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


class V62CrosshairTests(unittest.TestCase):
    def test_v62_config_defaults_enable_crosshair_reference(self):
        config = AimbotConfigV62()

        self.assertEqual(config.version, AIMBOT_V62_VERSION)
        self.assertTrue(config.crosshair_enabled)
        self.assertTrue(config.crosshair_use_hsv)
        self.assertEqual(config.crosshair_search_radius, 80)
        self.assertEqual(config.crosshair_min_pixels, 3)

    def test_crosshair_detector_finds_green_centroid_in_center_crop(self):
        config = AimbotConfigV62(
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
        config = AimbotConfigV62(
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

    def test_v62_aim_uses_crosshair_reference_instead_of_screen_center(self):
        config = AimbotConfigV62(
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
        aimbot = AimbotV62(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[20, 25] = [0, 255, 0]
        first = aimbot.process_detection(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 1}],
            frame.shape,
            (30, 30),
            frame=frame,
        )
        target = aimbot.process_detection(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 1}],
            frame.shape,
            (30, 30),
            frame=frame,
        )

        self.assertIsNone(first)
        self.assertIsNotNone(target)
        assert target is not None
        self.assertAlmostEqual(target.aim_x, 1.0)
        self.assertAlmostEqual(target.aim_y, 1.0)
        self.assertTrue(aimbot.aim_at_target(target))
        self.assertEqual(aimbot.kmbox.moves, [(1, 1)])

    def test_v62_stops_when_crosshair_is_not_found(self):
        config = AimbotConfigV62(
            screen_width=100,
            screen_height=100,
            fov_width=40,
            fov_height=40,
            tracker_generate=1,
            tracker_terminate=8,
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
        aimbot = AimbotV62(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()
        frame = np.zeros((40, 40, 3), dtype=np.uint8)

        moved = aimbot.update(
            [{"bbox": [25, 20, 27, 22], "confidence": 0.9, "class_id": 1}],
            frame.shape,
            (30, 30),
            frame=frame,
        )

        self.assertFalse(moved)
        self.assertEqual(aimbot.kmbox.moves, [])

    def test_v62_main_passes_detection_frame_to_aimbot_update(self):
        module = _load_script_module()
        config = AimbotConfigV62(enable_tuner=False)
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
            patch.object(module, "AimbotV62", return_value=aimbot),
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

    def test_v62_tuner_exposes_crosshair_fields(self):
        config = AimbotConfigV62()
        tuner = WebTuner(config, Path("/tmp/config_v62.json"))
        snapshot = tuner.snapshot()

        self.assertIn("crosshair_search_radius", TUNABLE_FIELDS)
        self.assertIn("crosshair_min_pixels", snapshot["config"])
        self.assertIn("crosshair_h_min", snapshot["config"])
        self.assertIn("crosshair_target_g", snapshot["config"])
        self.assertIn("<title>Aimbot V6.2 Tuner</title>", _HTML)
        self.assertIn('data-field="crosshair_search_radius"', _HTML)
        self.assertIn('data-field="crosshair_color_tolerance"', _HTML)


if __name__ == "__main__":
    unittest.main()
