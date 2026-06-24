import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from macos_dualbox_aim.v6.tracker import KalmanP
from macos_dualbox_aim.v6 import AIMBOT_V6_VERSION, AimbotConfigV6, AimbotV6
from macos_dualbox_aim.v6.tuner import TUNABLE_FIELDS, WebTuner, _HTML


class FakeKmbox:
    def __init__(self):
        self.moves = []

    def mouse_move(self, x: int, y: int) -> int:
        self.moves.append((x, y))
        return 0


class FakeEngine:
    def __init__(self):
        self.confidence_threshold = 0.0
        self.iou_threshold = 0.0

    def get_latency_snapshot(self):
        return {
            "available": True,
            "fps": 240.0,
            "window": 1,
            "current": {"program_total_ms": 8.0},
            "avg": {},
            "p95": {},
            "max": {},
            "counters": {},
        }


class FakeHotkey:
    def __init__(self):
        self.config = type("HotkeyConfig", (), {
            "trigger_button": "right",
            "trigger_button_secondary": "side1",
        })()
        self.checks = 0

    def _check_trigger(self):
        self.checks += 1


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "main_v6.py"
    spec = importlib.util.spec_from_file_location("main_v6", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V6Tests(unittest.TestCase):
    def test_v6_config_is_single_source_for_runtime_control_and_tracker_fields(self):
        config = AimbotConfigV6()

        self.assertEqual(config.version, AIMBOT_V6_VERSION)
        self.assertEqual(config.class_count, 4)
        self.assertEqual(config.confidence_threshold, 0.65)
        self.assertEqual(config.iou_threshold, 0.3)
        self.assertIsInstance(config.pid_kp, float)
        self.assertEqual(config.tracker_generate, 2)
        self.assertEqual(config.tracker_terminate, 8)

    def test_v6_config_save_writes_v6_metadata_and_tracker_fields(self):
        path = self._write_temp_config({})
        config = AimbotConfigV6()

        config.to_json(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["_version"], AIMBOT_V6_VERSION)
        self.assertIn("V6", data["_comment"])
        self.assertIn("class_count", data)
        self.assertIn("confidence_threshold", data)
        self.assertIn("iou_threshold", data)
        self.assertIn("adapter", data)
        self.assertIn("pid_kp", data)
        self.assertIn("aim_offset_y", data)
        self.assertIn("tracker_r_std", data)
        self.assertIn("tracker_vx_noise", data)

    def test_v6_uses_v3_tracker_for_target_selection_before_pid_control(self):
        config = AimbotConfigV6(
            screen_width=100,
            screen_height=100,
            fov_width=100,
            fov_height=100,
            tracker_generate=2,
            tracker_terminate=8,
            aim_offset_y=0.0,
            pid_kp=1.0,
            pid_ki=0.0,
            pid_kd=0.0,
            max_speed=100.0,
            sensitivity=1.0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
        )
        aimbot = AimbotV6(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        first = aimbot.process_detection(
            [
                {"bbox": [60, 49, 62, 51], "confidence": 0.9, "class_id": 1},
                {"bbox": [80, 49, 82, 51], "confidence": 0.8, "class_id": 1},
            ],
            (100, 100),
            (0, 0),
        )
        second = aimbot.process_detection(
            [
                {"bbox": [61, 49, 63, 51], "confidence": 0.9, "class_id": 1},
                {"bbox": [81, 49, 83, 51], "confidence": 0.8, "class_id": 1},
            ],
            (100, 100),
            (0, 0),
        )
        third = aimbot.process_detection(
            [
                {"bbox": [62, 49, 64, 51], "confidence": 0.9, "class_id": 1},
                {"bbox": [82, 49, 84, 51], "confidence": 0.8, "class_id": 1},
            ],
            (100, 100),
            (0, 0),
        )

        self.assertIsInstance(aimbot.tracker, KalmanP)
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(third)
        assert second is not None
        assert third is not None
        self.assertEqual(second.track_id, third.track_id)
        self.assertLess(second.aim_x, 20.0)
        self.assertTrue(aimbot.aim_at_target(second))
        self.assertTrue(aimbot.kmbox.moves)

    def test_v6_main_build_engine_uses_v6_config_only(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            control = AimbotConfigV6(
                model_path="models/converted/cs2_fp16_fp16_fast.mlpackage",
                class_count=4,
                confidence_threshold=0.5,
                iou_threshold=0.2,
                capture_device=2,
                target_fps=120,
                enable_display=True,
                fov_width=320,
                fov_height=320,
                screen_width=1920,
                screen_height=1080,
                pixel_format="MJPEG",
            )

            with patch.object(module, "RealtimeInferenceV6") as engine_cls:
                module.build_engine(project_root, control)

        engine_cls.assert_called_once()
        kwargs = engine_cls.call_args.kwargs
        self.assertEqual(kwargs["class_count"], 4)
        self.assertEqual(kwargs["confidence_threshold"], 0.5)
        self.assertEqual(kwargs["iou_threshold"], 0.2)
        self.assertEqual(kwargs["crop_size"], (320, 320))
        self.assertEqual(kwargs["capture_resolution"], (1920, 1080))

    def test_v6_package_does_not_import_v3_v4_or_v5_layers(self):
        v6_dir = Path(__file__).resolve().parent.parent / "src" / "macos_dualbox_aim" / "v6"
        offenders = {}
        for path in v6_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            bad_lines = [
                line
                for line in text.splitlines()
                if "from ..v3" in line or "from ..v4" in line or "from ..v5" in line
            ]
            if bad_lines:
                offenders[path.name] = bad_lines

        self.assertEqual(offenders, {})

    def test_v6_main_reads_only_v6_config(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            config_dir = project_root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "config_v6.json"
            AimbotConfigV6(confidence_threshold=0.44, iou_threshold=0.22).to_json(config_path)

            with patch.object(module, "project_root", project_root):
                config = module.load_config(config_path)

        self.assertEqual(config.confidence_threshold, 0.44)
        self.assertEqual(config.iou_threshold, 0.22)

    def test_v6_tuner_exposes_control_detection_and_tracker_fields(self):
        config = AimbotConfigV6()
        tuner = WebTuner(config, self._write_temp_config({}))
        snapshot = tuner.snapshot()

        self.assertEqual(set(snapshot["config"]), TUNABLE_FIELDS)
        self.assertIn("confidence_threshold", snapshot["config"])
        self.assertIn("iou_threshold", snapshot["config"])
        self.assertNotIn("detection_confidence_threshold", snapshot["config"])
        self.assertIn("tracker_generate", snapshot["config"])
        self.assertIn("tracker_r_std", snapshot["config"])

    def test_v6_tuner_applies_thresholds_controller_and_resets_tracker(self):
        config = AimbotConfigV6()
        engine = FakeEngine()
        hotkey = FakeHotkey()
        aimbot = AimbotV6(config)
        original_tracker = aimbot.tracker
        tuner = WebTuner(config, self._write_temp_config({}), engine=engine, hotkey=hotkey, aimbot=aimbot)

        snapshot = tuner.update_config({
            "pid_kp": 0.5,
            "max_speed": 45.0,
            "confidence_threshold": 0.42,
            "iou_threshold": 0.12,
            "tracker_generate": 3,
            "tracker_r_std": 7.0,
            "trigger_button": "left",
            "trigger_button_secondary": None,
        })

        self.assertTrue(snapshot["dirty"])
        self.assertEqual(config.pid_kp, 0.5)
        self.assertEqual(aimbot.controller._base_kp, 0.5)
        self.assertEqual(aimbot.controller.max_speed, 45.0)
        self.assertEqual(engine.confidence_threshold, 0.42)
        self.assertEqual(engine.iou_threshold, 0.12)
        self.assertEqual(config.tracker_generate, 3)
        self.assertEqual(config.tracker_r_std, 7.0)
        self.assertIsNot(aimbot.tracker, original_tracker)
        self.assertEqual(hotkey.config.trigger_button, "left")
        self.assertIsNone(hotkey.config.trigger_button_secondary)
        self.assertEqual(hotkey.checks, 1)

    def test_v6_tuner_html_is_labeled_v6(self):
        self.assertIn("<title>Aimbot V6 Tuner</title>", _HTML)
        self.assertIn("<h1>Aimbot V6 Tuner</h1>", _HTML)
        self.assertIn('data-field="confidence_threshold"', _HTML)
        self.assertIn('data-field="tracker_generate"', _HTML)
        self.assertIn('const integerFields = new Set(["fov_radius", "tracker_generate", "tracker_terminate"])', _HTML)
        self.assertNotIn('data-field="detection_confidence_threshold"', _HTML)

    def _write_temp_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
