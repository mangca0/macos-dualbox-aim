import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "probe_v5_capture.py"
    spec = importlib.util.spec_from_file_location("probe_v5_capture", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V5CaptureProbeScriptTests(unittest.TestCase):
    def test_parse_args_defaults_to_v4_capture_and_fast_model(self):
        module = _load_script_module()

        args = module.parse_args([])

        self.assertEqual(args.model, Path("models/converted/cs2_fp16_fp16_fast.mlpackage"))
        self.assertEqual(args.check_model, Path("models/converted/cs2_fp16_fp32_check.mlpackage"))
        self.assertEqual(args.capture_device, 0)
        self.assertEqual(args.screen_width, 1920)
        self.assertEqual(args.screen_height, 1080)
        self.assertEqual(args.fov_width, 320)
        self.assertEqual(args.fov_height, 320)
        self.assertEqual(args.pixel_format, "MJPEG")

    def test_center_crop_uses_screen_center(self):
        module = _load_script_module()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[380, 800] = [255, 255, 255]

        crop, offset = module.center_crop(frame, crop_size=(320, 320))

        self.assertEqual(crop.shape, (320, 320, 3))
        self.assertEqual(offset, (800, 380))
        self.assertTrue(np.array_equal(crop[0, 0], [255, 255, 255]))

    def test_draw_detections_marks_normalized_box(self):
        module = _load_script_module()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        output = module.draw_detections(
            frame,
            [{"bbox": [0.5, 0.5, 0.4, 0.2], "confidence": 0.9, "class_id": 1}],
        )

        self.assertFalse(np.array_equal(output, frame))


if __name__ == "__main__":
    unittest.main()
