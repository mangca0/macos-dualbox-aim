import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from macos_dualbox_aim.v4 import AimbotConfigV4
from macos_dualbox_aim.v5 import ModelRuntimeConfigV5


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "main_v5.py"
    spec = importlib.util.spec_from_file_location("main_v5", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V5MainTests(unittest.TestCase):
    def test_sync_model_thresholds_updates_control_config(self):
        module = _load_script_module()
        control = AimbotConfigV4(detection_confidence_threshold=0.1, detection_iou_threshold=0.9)
        model = ModelRuntimeConfigV5(confidence_threshold=0.42, iou_threshold=0.12)

        module.sync_model_thresholds(control, model)

        self.assertEqual(control.detection_confidence_threshold, 0.42)
        self.assertEqual(control.detection_iou_threshold, 0.12)

    def test_build_engine_uses_v5_runtime_and_model_config(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            control = AimbotConfigV4(
                capture_device=2,
                target_fps=120,
                enable_display=True,
                fov_width=320,
                fov_height=320,
                screen_width=1920,
                screen_height=1080,
                pixel_format="MJPEG",
            )
            model = ModelRuntimeConfigV5(
                model_path="models/converted/cs2_fp16_fp16_fast.mlpackage",
                class_count=4,
                confidence_threshold=0.5,
                iou_threshold=0.2,
            )

            with patch.object(module, "RealtimeInferenceV5") as engine_cls:
                module.build_engine(project_root, control, model)

        engine_cls.assert_called_once()
        kwargs = engine_cls.call_args.kwargs
        self.assertEqual(kwargs["class_count"], 4)
        self.assertEqual(kwargs["confidence_threshold"], 0.5)
        self.assertEqual(kwargs["iou_threshold"], 0.2)
        self.assertEqual(kwargs["crop_size"], (320, 320))
        self.assertEqual(kwargs["capture_resolution"], (1920, 1080))


if __name__ == "__main__":
    unittest.main()
