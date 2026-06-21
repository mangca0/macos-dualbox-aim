import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from macos_dualbox_aim.v1.kmbox import SUCCESS
from macos_dualbox_aim.v4.config import AIMBOT_V4_VERSION, AimbotConfigV4
from macos_dualbox_aim.v4.controller import AimbotV4, PIDController
from macos_dualbox_aim.v4.tuner import TUNABLE_FIELDS, WebTuner, _HTML


class FakeKmbox:
    def __init__(self):
        self.moves = []

    def mouse_move(self, x: int, y: int) -> int:
        self.moves.append((x, y))
        return SUCCESS


class FakeEngine:
    def __init__(self):
        self.confidence_threshold = 0.0
        self.iou_threshold = 0.0
        self.latency = {
            "available": True,
            "fps": 240.0,
            "window": 2,
            "current": {"program_total_ms": 8.0},
            "avg": {},
            "p95": {},
            "max": {},
            "counters": {"frames_dropped": 1},
        }

    def get_latency_snapshot(self):
        return self.latency


class FakeHotkey:
    def __init__(self):
        self.config = type("HotkeyConfig", (), {
            "trigger_button": "right",
            "trigger_button_secondary": "side1",
        })()
        self.checks = 0

    def _check_trigger(self):
        self.checks += 1


class V4Tests(unittest.TestCase):
    def test_config_exposes_all_mpid_controller_parameters(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config_v4.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(data["_version"], AIMBOT_V4_VERSION)
        for field in (
            "enable_tuner",
            "tuner_host",
            "tuner_port",
            "pid_kp",
            "pid_ki",
            "pid_kd",
            "slew_limit",
            "max_speed",
            "sensitivity",
            "fov_radius",
            "init_scale",
            "ramp_time",
            "pred_weight_x",
            "pred_weight_y",
        ):
            self.assertIn(field, data)

    def test_config_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = AimbotConfigV4()
            config.to_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["pid_kpp"] = 9.0
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unknown config field: pid_kpp"):
                AimbotConfigV4.from_json(path)

    def test_learning_controller_uses_incremental_pid_output(self):
        controller = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            max_speed=100.0,
            sensitivity=1.0,
            fov_radius=0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
        )

        with patch("macos_dualbox_aim.v4.controller.time.monotonic", side_effect=[100.0, 100.01]):
            first = controller.update(0.0, 0.0, 10.0, -5.0)
            second = controller.update(0.0, 0.0, 10.0, -5.0)

        self.assertEqual(first, (10.0, -5.0))
        self.assertEqual(second, (10.0, -5.0))

    def test_aimbot_uses_first_detection_directly_and_sends_kmbox_move(self):
        config = AimbotConfigV4(
            screen_width=100,
            screen_height=100,
            fov_width=20,
            fov_height=20,
            aim_offset_x=0.0,
            aim_offset_y=-0.5,
            aim_offset_dynamic=True,
            pid_kp=1.0,
            pid_ki=0.0,
            pid_kd=0.0,
            max_speed=100.0,
            sensitivity=1.0,
            fov_radius=0,
            init_scale=1.0,
            pred_weight_x=0.0,
            pred_weight_y=0.0,
        )
        aimbot = AimbotV4(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        detections = [
            {"bbox": [10, 10, 14, 18], "confidence": 0.9, "class_id": 1},
            {"bbox": [0, 0, 20, 20], "confidence": 0.99, "class_id": 1},
        ]
        timing_ms = {}
        result = aimbot.update(detections, (20, 20), (40, 40), timing_ms=timing_ms)

        self.assertTrue(result)
        self.assertEqual(aimbot.kmbox.moves, [(2, 0)])
        self.assertIn("target_select_ms", timing_ms)
        self.assertIn("pid_ms", timing_ms)
        self.assertIn("kmbox_send_ack_ms", timing_ms)

    def test_aimbot_resets_controller_when_detection_is_missing(self):
        config = AimbotConfigV4()
        aimbot = AimbotV4(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        self.assertFalse(aimbot.update([], (20, 20), (0, 0)))
        self.assertEqual(aimbot.kmbox.moves, [])

    def test_v4_tuner_exposes_all_runtime_tunable_fields(self):
        config = AimbotConfigV4()
        tuner = WebTuner(config, self._write_temp_config({}))
        snapshot = tuner.snapshot()

        self.assertEqual(set(snapshot["config"]), TUNABLE_FIELDS)
        for field in (
            "pid_kp",
            "pid_ki",
            "pid_kd",
            "slew_limit",
            "max_speed",
            "sensitivity",
            "fov_radius",
            "init_scale",
            "ramp_time",
            "pred_weight_x",
            "pred_weight_y",
        ):
            self.assertIn(field, snapshot["config"])

    def test_v4_tuner_applies_runtime_updates_to_engine_hotkey_and_controller(self):
        config = AimbotConfigV4()
        engine = FakeEngine()
        hotkey = FakeHotkey()
        aimbot = AimbotV4(config)
        path = self._write_temp_config({})
        tuner = WebTuner(config, path, engine=engine, hotkey=hotkey, aimbot=aimbot)

        snapshot = tuner.update_config({
            "pid_kp": 0.5,
            "pid_ki": 0.01,
            "pid_kd": 0.02,
            "slew_limit": 12.0,
            "max_speed": 45.0,
            "sensitivity": 1.25,
            "fov_radius": 300,
            "init_scale": 0.7,
            "ramp_time": 0.25,
            "pred_weight_x": 0.2,
            "pred_weight_y": 0.3,
            "detection_confidence_threshold": 0.42,
            "detection_iou_threshold": 0.12,
            "aim_offset_y": -0.25,
            "trigger_button": "left",
            "trigger_button_secondary": None,
        })

        self.assertTrue(snapshot["dirty"])
        self.assertEqual(config.pid_kp, 0.5)
        self.assertEqual(aimbot.controller._base_kp, 0.5)
        self.assertEqual(aimbot.controller.slew_limit, 12.0)
        self.assertEqual(aimbot.controller.max_speed, 45.0)
        self.assertEqual(aimbot.controller.sensitivity, 1.25)
        self.assertEqual(aimbot.controller.fov_radius, 300)
        self.assertEqual(aimbot.controller.init_scale, 0.7)
        self.assertEqual(aimbot.controller.ramp_time, 0.25)
        self.assertEqual(aimbot.controller.pred_weight_x, 0.2)
        self.assertEqual(aimbot.controller.pred_weight_y, 0.3)
        self.assertEqual(engine.confidence_threshold, 0.42)
        self.assertEqual(engine.iou_threshold, 0.12)
        self.assertEqual(hotkey.config.trigger_button, "left")
        self.assertIsNone(hotkey.config.trigger_button_secondary)
        self.assertEqual(hotkey.checks, 1)
        self.assertEqual(snapshot["latency"]["current"]["program_total_ms"], 8.0)

        saved = tuner.save_config()
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(saved["dirty"])
        self.assertEqual(data["pid_kp"], 0.5)
        self.assertEqual(data["trigger_button_secondary"], None)

    def test_v4_tuner_limits_trigger_buttons_to_supported_runtime_inputs(self):
        tuner = WebTuner(AimbotConfigV4(), self._write_temp_config({}))

        with self.assertRaisesRegex(ValueError, "trigger_button"):
            tuner.update_config({"trigger_button": "middle"})

    def test_v4_tuner_web_uses_v3_layout_conventions(self):
        self.assertIn('<button id="reload" type="button">Reload</button>', _HTML)
        self.assertIn('id="latency-highlights"', _HTML)
        self.assertIn('class="metric-head"', _HTML)
        self.assertIn('data-number-for="pid_kp"', _HTML)
        self.assertIn('window.setInterval(refreshLatency, 500)', _HTML)

    def test_v4_tuner_web_only_truncates_declared_integer_fields(self):
        self.assertIn('const integerFields = new Set(["fov_radius"])', _HTML)
        self.assertNotIn("Number.isInteger(state.config[field])", _HTML)

    def _write_temp_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path
