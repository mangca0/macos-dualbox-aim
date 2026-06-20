import inspect
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from macos_dualbox_aim.v1.config import AIMBOT_V1_VERSION, AimbotConfigV1
from macos_dualbox_aim.v1.controller import AimbotV1, PIDFControllerV1
from macos_dualbox_aim.v1.hotkey import HotkeyConfig
from macos_dualbox_aim.v1.inference import CoreMLDetector, RealtimeInference
from macos_dualbox_aim.v1.kmbox import ERR_NET_RX_TIMEOUT, KmboxConfig, KmboxNet, SUCCESS
from macos_dualbox_aim.v1.tuner import WebTuner


class FakeKmbox:
    def __init__(self):
        self.moves = []

    def mouse_move(self, x: int, y: int) -> int:
        self.moves.append((x, y))
        return SUCCESS


class TimeoutSocket:
    def __init__(self):
        self.sent = []
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout: float):
        self.timeout = timeout

    def sendto(self, data: bytes, address):
        self.sent.append((data, address))

    def recvfrom(self, _size: int):
        raise socket.timeout

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self):
        self.confidence_threshold = 0.0
        self.iou_threshold = 0.0
        self.latency = {
            "available": True,
            "fps": 238.5,
            "window": 2,
            "current": {"program_total_ms": 8.0, "coreml_ms": 5.0},
            "avg": {"program_total_ms": 9.0, "coreml_ms": 5.5},
            "p95": {"program_total_ms": 9.9, "coreml_ms": 5.9},
            "max": {"program_total_ms": 10.0, "coreml_ms": 6.0},
            "counters": {
                "frames_captured": 12,
                "frames_inferred": 10,
                "frames_dropped": 2,
                "frame_queue_replaced": 1,
                "frame_queue_drained": 1,
            },
        }

    def get_latency_snapshot(self):
        return self.latency


class FakeHotkey:
    def __init__(self):
        self.config = HotkeyConfig()
        self.checks = 0

    def _check_trigger(self):
        self.checks += 1


class V1Tests(unittest.TestCase):
    def test_pidf_uses_standard_derivative_and_feedforward_terms(self):
        config = AimbotConfigV1(pid_kp=1.0, pid_ki=2.0, pid_kd=3.0, pid_kf=4.0)
        controller = PIDFControllerV1(config)

        first = controller.update(10.0, -5.0, 0.0, 0.0, 100.0)
        second = controller.update(8.0, -4.0, 50.0, -25.0, 100.1)

        self.assertEqual(first, (10.0, -5.0))
        self.assertAlmostEqual(second[0], 149.6)
        self.assertAlmostEqual(second[1], -74.8)

    def test_detection_to_mouse_uses_center_bbox_and_aim_offset(self):
        config = AimbotConfigV1(
            screen_width=100,
            screen_height=100,
            fov_width=20,
            fov_height=20,
            target_classes=[1],
            class_priority_weights={},
            aim_offset_x=0.0,
            aim_offset_y=-0.5,
            aim_offset_dynamic=True,
            pid_kp=1.0,
            pid_ki=0.0,
            pid_kd=0.0,
            pid_kf=0.0,
        )
        aimbot = AimbotV1(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        detections = [{"bbox": [10, 10, 14, 18], "confidence": 0.9, "class_id": 1}]
        timing_ms = {}
        result = aimbot.update(detections, (20, 20), (40, 40), timing_ms=timing_ms)

        self.assertTrue(result)
        self.assertEqual(aimbot.kmbox.moves, [(2, 0)])
        self.assertIn("target_select_ms", timing_ms)
        self.assertIn("pid_ms", timing_ms)
        self.assertIn("kmbox_send_ack_ms", timing_ms)

    def test_config_exposes_only_pidf_control_parameters(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config_v1.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual({"pid_kp", "pid_ki", "pid_kd", "pid_kf"} & set(data), {
            "pid_kp",
            "pid_ki",
            "pid_kd",
            "pid_kf",
        })
        for forbidden in (
            "pid_output_gain",
            "pid_micro_gain",
            "direct_chase_gain",
            "output_gain_x",
            "output_gain_y",
            "max_output_per_frame",
            "max_axis_output",
            "deadzone",
            "enable_kalman_filter",
            "enable_delay_compensation",
            "enable_sot_tracker",
            "aim_reference_mode",
            "crosshair_template_path",
        ):
            self.assertNotIn(forbidden, data)

    def test_config_rejects_unknown_fields(self):
        path = self._write_temp_config({"pid_kpp": 9.0})

        with self.assertRaisesRegex(ValueError, "Unknown config field: pid_kpp"):
            AimbotConfigV1.from_json(path)

    def test_config_rejects_invalid_field_types(self):
        path = self._write_temp_config({"pid_kp": "fast"})

        with self.assertRaisesRegex(ValueError, "pid_kp must be a number"):
            AimbotConfigV1.from_json(path)

    def test_config_accepts_json_object_class_weight_keys(self):
        path = self._write_temp_config({"class_priority_weights": {"1": 1.5}})

        config = AimbotConfigV1.from_json(path)

        self.assertEqual(config.class_priority_weights, {1: 1.5})

    def test_config_rejects_frame_queue_size(self):
        path = self._write_temp_config({"frame_queue_size": 1})

        with self.assertRaisesRegex(ValueError, "Unknown config field: frame_queue_size"):
            AimbotConfigV1.from_json(path)

    def test_v1_config_does_not_expose_frame_queue_size(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config_v1.json"

        data = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(data["_version"], AIMBOT_V1_VERSION)
        self.assertNotIn("frame_queue_size", data)

    def test_config_save_writes_current_version_metadata(self):
        path = self._write_temp_config({
            "_comment": "macos-dualbox-aim V1 - minimal PIDF runtime",
            "_version": "1.0.0",
            "_custom_note": "keep me",
        })
        config = AimbotConfigV1.from_json(path)

        config.to_json(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["_version"], AIMBOT_V1_VERSION)
        self.assertIn("V1.2.3", data["_comment"])
        self.assertEqual(data["_custom_note"], "keep me")
        self.assertNotIn("frame_queue_size", data)

    def test_realtime_inference_keeps_published_frame_queue_default(self):
        signature = inspect.signature(RealtimeInference)

        self.assertEqual(signature.parameters["frame_queue_size"].default, 3)

    def test_realtime_inference_uses_v1_0_capture_latency_shape(self):
        engine = RealtimeInference.__new__(RealtimeInference)
        engine.frames_captured = 0
        engine.frames_inferred = 0
        engine.frames_dropped = 0
        engine.frame_queue_replaced = 0
        engine.frame_queue_drained = 0
        engine.actual_inference_fps = 0.0
        engine.latency_samples = []
        engine.latency_lock = __import__("threading").RLock()

        engine._record_latency(7, {"capture_read_ms": 9.0})
        snapshot = engine.get_latency_snapshot()

        self.assertEqual(snapshot["current"]["capture_read_ms"], 9.0)
        self.assertNotIn("capture", snapshot)
        self.assertNotIn("capture_grab_failures", snapshot["counters"])
        self.assertNotIn("capture_retrieve_failures", snapshot["counters"])
        json.dumps(snapshot)

    def test_raw_postprocess_keeps_decode_before_threshold_filter(self):
        detector = CoreMLDetector.__new__(CoreMLDetector)
        detector.input_shape = (320, 320)
        detector.decode_cache = {}
        decode_calls = []

        def decode_grid(input_size, num_anchors):
            decode_calls.append((input_size, num_anchors))
            return (
                np.zeros(num_anchors, dtype=np.float32),
                np.zeros(num_anchors, dtype=np.float32),
                np.ones(num_anchors, dtype=np.float32),
            )

        detector._decode_grid = decode_grid
        predictions = {
            "output": np.array([[
                [0.0, 0.0, 0.0, 0.0, 0.10, 0.20, 0.30],
                [0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.10],
            ]], dtype=np.float32)
        }

        detections = detector._parse_raw_predictions(predictions, confidence_threshold=0.9, iou_threshold=0.3)

        self.assertEqual(detections, [])
        self.assertEqual(decode_calls, [(320, 2)])

    def test_web_tuner_applies_live_fields_and_saves_config(self):
        config = AimbotConfigV1()
        engine = FakeEngine()
        hotkey = FakeHotkey()
        path = self._write_temp_config({})
        tuner = WebTuner(config, path, engine=engine, hotkey=hotkey)

        snapshot = tuner.update_config({
            "pid_kp": 0.5,
            "detection_confidence_threshold": 0.42,
            "detection_iou_threshold": 0.12,
            "target_classes": [1, 2],
            "class_priority_weights": {"1": 1.75},
            "aim_offset_y": -0.25,
            "trigger_button": "left",
            "trigger_button_secondary": "side2",
        })

        self.assertTrue(snapshot["dirty"])
        self.assertEqual(config.pid_kp, 0.5)
        self.assertEqual(config.class_priority_weights, {1: 1.75})
        self.assertEqual(engine.confidence_threshold, 0.42)
        self.assertEqual(engine.iou_threshold, 0.12)
        self.assertEqual(hotkey.config.trigger_button, "left")
        self.assertEqual(hotkey.config.trigger_button_secondary, "side2")
        self.assertEqual(hotkey.checks, 1)
        self.assertEqual(snapshot["latency"]["current"]["program_total_ms"], 8.0)
        self.assertEqual(snapshot["latency"]["counters"]["frames_dropped"], 2)

        saved = tuner.save_config()
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(saved["dirty"])
        self.assertEqual(data["pid_kp"], 0.5)
        self.assertEqual(data["detection_confidence_threshold"], 0.42)
        self.assertEqual(data["trigger_button_secondary"], "side2")

    def test_web_tuner_limits_trigger_buttons_to_requested_mouse_inputs(self):
        tuner = WebTuner(AimbotConfigV1(), self._write_temp_config({}))

        with self.assertRaisesRegex(ValueError, "trigger_button"):
            tuner.update_config({"trigger_button": "middle"})

    def test_kmbox_init_retries_once_then_fails_on_two_timeouts(self):
        sockets = []

        def make_socket(*_args, **_kwargs):
            sock = TimeoutSocket()
            sockets.append(sock)
            return sock

        kmbox = KmboxNet(KmboxConfig(connect_attempts=2, socket_timeout=0.01))
        with patch("macos_dualbox_aim.v1.kmbox.socket.socket", side_effect=make_socket):
            result = kmbox.init()

        self.assertEqual(result, ERR_NET_RX_TIMEOUT)
        self.assertEqual(len(sockets), 2)
        self.assertEqual([len(sock.sent) for sock in sockets], [1, 1])
        self.assertTrue(all(sock.closed for sock in sockets))

    def _write_temp_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
