import json
import tempfile
import unittest
from pathlib import Path

from macos_dualbox_aim.v1.config import AimbotConfigV1
from macos_dualbox_aim.v3.config import AIMBOT_V3_VERSION, AimbotConfigV3
from macos_dualbox_aim.v3.controller import AimbotV3
from macos_dualbox_aim.v3.kmbox import SUCCESS
from macos_dualbox_aim.v3.tracker import BoundingBox, DetectionObject, KalmanP


class FakeKmbox:
    def __init__(self):
        self.moves = []

    def mouse_move(self, x: int, y: int) -> int:
        self.moves.append((x, y))
        return SUCCESS


class V3TrackerTests(unittest.TestCase):
    def test_v3_config_is_independent_and_exposes_tracker_fields(self):
        self.assertFalse(hasattr(AimbotConfigV1(), "tracker_generate"))

        config = AimbotConfigV3()

        self.assertEqual(config.version, AIMBOT_V3_VERSION)
        self.assertFalse(hasattr(config, "pid_kp"))
        self.assertFalse(hasattr(config, "aim_offset_y"))
        self.assertEqual(config.tracker_generate, 2)
        self.assertEqual(config.tracker_terminate, 8)

    def test_v3_config_save_writes_v3_metadata_and_tracker_fields(self):
        path = self._write_temp_config({})
        config = AimbotConfigV3()

        config.to_json(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["_version"], AIMBOT_V3_VERSION)
        self.assertIn("V3", data["_comment"])
        self.assertNotIn("pid_kp", data)
        self.assertNotIn("aim_offset_y", data)
        self.assertIn("tracker_r_std", data)
        self.assertIn("tracker_vx_noise", data)

    def test_tracker_confirms_after_generate_threshold_and_keeps_id(self):
        tracker = KalmanP(generate=2, terminate=8)

        first = tracker.predict([self._det(10.0, 20.0, 30.0, 40.0, label=1)])
        second = tracker.predict([self._det(12.0, 21.0, 30.0, 40.0, label=1)])
        third = tracker.predict([self._det(14.0, 22.0, 30.0, 40.0, label=1)])

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertEqual(len(third), 1)
        self.assertEqual(second[0].track_id, third[0].track_id)
        self.assertEqual(second[0].label, 1)

    def test_tracker_outputs_unmatched_prediction_until_terminate(self):
        tracker = KalmanP(generate=1, terminate=2)
        self.assertEqual(tracker.predict([self._det(100.0, 100.0, 20.0, 30.0)]), [])
        confirmed = tracker.predict([self._det(102.0, 100.0, 20.0, 30.0)])

        predicted = tracker.predict([])
        removed = tracker.predict([])

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0].track_id, confirmed[0].track_id)
        self.assertEqual(removed, [])

    def test_aimbot_v3_aims_with_tracker_output(self):
        config = AimbotConfigV3(
            screen_width=100,
            screen_height=100,
            fov_width=100,
            fov_height=100,
            tracker_generate=1,
            tracker_terminate=8,
        )
        aimbot = AimbotV3(config)
        aimbot.kmbox = FakeKmbox()
        aimbot.activate()

        self.assertFalse(aimbot.update(
            [{"bbox": [60, 49, 62, 51], "confidence": 0.9, "class_id": 1}],
            (100, 100),
            (0, 0),
        ))
        moved = aimbot.update(
            [{"bbox": [60, 49, 62, 51], "confidence": 0.9, "class_id": 1}],
            (100, 100),
            (0, 0),
        )

        self.assertTrue(moved)
        self.assertEqual(aimbot.kmbox.moves, [(11, 0)])

    def _det(self, x: float, y: float, w: float, h: float, label: int = 0) -> DetectionObject:
        return DetectionObject(BoundingBox(x, y, w, h), label=label, prob=0.9)

    def _write_temp_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
