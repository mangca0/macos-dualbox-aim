import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from macos_dualbox_aim.v1.config import AimbotConfigV1
from macos_dualbox_aim.v2.config import AIMBOT_V2_VERSION, AimbotConfigV2
from macos_dualbox_aim.v2.controller import AimbotV2, KalmanFilter2D
from macos_dualbox_aim.v2.kmbox import SUCCESS
from macos_dualbox_aim.v2.tuner import WebTuner


class FakeKmbox:
    def __init__(self):
        self.moves = []

    def mouse_move(self, x: int, y: int) -> int:
        self.moves.append((x, y))
        return SUCCESS


class FakeAimbot:
    def __init__(self):
        self.resets = 0

    def reset_tracking(self):
        self.resets += 1


class V2Tests(unittest.TestCase):
    def test_v2_config_is_independent_and_exposes_kalman_fields(self):
        self.assertFalse(hasattr(AimbotConfigV1(), "enable_kalman_filter"))

        config = AimbotConfigV2()

        self.assertEqual(config.version, AIMBOT_V2_VERSION)
        self.assertTrue(config.enable_kalman_filter)
        self.assertGreater(config.kalman_measurement_noise, 0.0)
        self.assertGreater(config.kalman_process_noise, 0.0)

    def test_v2_config_save_writes_v2_metadata_and_kalman_fields(self):
        path = self._write_temp_config({})
        config = AimbotConfigV2()

        config.to_json(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["_version"], AIMBOT_V2_VERSION)
        self.assertIn("V2", data["_comment"])
        self.assertTrue(data["enable_kalman_filter"])
        self.assertIn("kalman_measurement_noise", data)
        self.assertIn("kalman_process_noise", data)

    def test_kalman_filter_smooths_measurements_and_estimates_velocity(self):
        tracker = KalmanFilter2D(process_noise=1.0, measurement_noise=100.0, initial_covariance=1.0)

        tracker.update(0.0, 0.0, 100.0)
        state = tracker.update(100.0, 0.0, 100.01)

        self.assertGreater(state.x, 0.0)
        self.assertLess(state.x, 100.0)
        self.assertGreater(state.vx, 0.0)
        self.assertAlmostEqual(state.y, 0.0)

    def test_aimbot_uses_filtered_position_before_pidf(self):
        config = AimbotConfigV2(
            screen_width=100,
            screen_height=100,
            fov_width=100,
            fov_height=100,
            target_classes=[1],
            class_priority_weights={},
            aim_offset_x=0.0,
            aim_offset_y=0.0,
            aim_offset_dynamic=False,
            pid_kp=1.0,
            pid_ki=0.0,
            pid_kd=0.0,
            pid_kf=0.0,
            enable_kalman_filter=True,
            kalman_measurement_noise=100.0,
            kalman_process_noise=1.0,
            kalman_initial_covariance=500.0,
        )
        aimbot = AimbotV2(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        first_detection = [{"bbox": [49, 49, 51, 51], "confidence": 0.9, "class_id": 1}]
        second_detection = [{"bbox": [69, 49, 71, 51], "confidence": 0.9, "class_id": 1}]

        with patch("macos_dualbox_aim.v2.controller.time.time", side_effect=[100.0, 100.0, 100.01, 100.01]):
            self.assertFalse(aimbot.update(first_detection, (100, 100), (0, 0)))
            self.assertTrue(aimbot.update(second_detection, (100, 100), (0, 0)))

        self.assertEqual(len(aimbot.kmbox.moves), 1)
        self.assertGreater(aimbot.kmbox.moves[0][0], 0)
        self.assertLess(aimbot.kmbox.moves[0][0], 20)

    def test_web_tuner_resets_tracking_when_kalman_parameters_change(self):
        config = AimbotConfigV2()
        aimbot = FakeAimbot()
        tuner = WebTuner(config, self._write_temp_config({}), aimbot=aimbot)

        tuner.update_config({"pid_kp": 0.25})
        self.assertEqual(aimbot.resets, 0)

        tuner.update_config({"kalman_measurement_noise": 80.0})
        self.assertEqual(aimbot.resets, 1)

        tuner.update_config({"enable_kalman_filter": False})
        self.assertEqual(aimbot.resets, 2)

    def _write_temp_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
