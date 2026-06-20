import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from macos_dualbox_aim.v1.capture_probe import (
    CaptureMode,
    build_modes,
    format_results_markdown,
    parse_resolution,
    probe_capture_mode,
    write_results_jsonl,
)


class StepClock:
    def __init__(self, step: float = 0.001):
        self.value = 0.0
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


class FakeCapture:
    instances = []

    def __init__(self, device: int, backend=None):
        self.device = device
        self.backend = backend
        self.released = False
        self.set_calls = []
        self.retrieve_count = 0
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            cv2.CAP_PROP_FPS: 119.88,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
            cv2.CAP_PROP_BUFFERSIZE: 1.0,
        }
        type(self).instances.append(self)

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return self.properties.get(prop, 0.0)

    def getBackendName(self):
        return "FAKE"

    def grab(self):
        return True

    def retrieve(self):
        self.retrieve_count += 1
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class CaptureProbeTests(unittest.TestCase):
    def setUp(self):
        FakeCapture.instances = []

    def test_parse_resolution_requires_width_by_height(self):
        self.assertEqual(parse_resolution("1920x1080"), (1920, 1080))
        with self.assertRaisesRegex(ValueError, "resolution"):
            parse_resolution("1920")

    def test_build_modes_expands_format_fps_resolution_matrix(self):
        modes = build_modes(
            device=1,
            pixel_formats=["MJPEG", "YUY2"],
            fps_values=[120, 240],
            resolutions=[(1280, 720)],
            load="sleep",
            load_ms=9.0,
            load_placement="thread",
        )

        self.assertEqual([mode.pixel_format for mode in modes], ["MJPEG", "MJPEG", "YUY2", "YUY2"])
        self.assertEqual([mode.fps for mode in modes], [120, 240, 120, 240])
        self.assertEqual([mode.load for mode in modes], ["sleep", "sleep", "sleep", "sleep"])
        self.assertEqual([mode.load_ms for mode in modes], [9.0, 9.0, 9.0, 9.0])
        self.assertEqual([mode.load_placement for mode in modes], ["thread", "thread", "thread", "thread"])

    def test_probe_capture_mode_records_requested_actual_and_stage_timings(self):
        mode = CaptureMode(device=2, width=1280, height=720, fps=120, pixel_format="MJPEG", samples=3)

        result = probe_capture_mode(mode, capture_factory=FakeCapture, clock=StepClock())

        capture = FakeCapture.instances[0]
        self.assertTrue(capture.released)
        self.assertIn((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")), capture.set_calls)
        self.assertEqual(result.requested["device"], 2)
        self.assertEqual(result.requested["fourcc"], "MJPG")
        self.assertEqual(result.actual["backend"], "FAKE")
        self.assertEqual(result.actual["fourcc"], "MJPG")
        self.assertEqual(result.ok_frames, 3)
        self.assertEqual(result.grab_failures, 0)
        self.assertEqual(result.retrieve_failures, 0)
        self.assertGreater(result.avg_grab_ms, 0.0)
        self.assertGreater(result.avg_retrieve_ms, 0.0)
        self.assertGreater(result.avg_frame_interval_ms, 0.0)
        json.dumps(result.to_record())

    def test_probe_capture_mode_runs_sleep_load_per_successful_frame(self):
        slept = []
        mode = CaptureMode(
            device=2,
            width=1280,
            height=720,
            fps=120,
            pixel_format="MJPEG",
            samples=2,
            warmup=1,
            load="sleep",
            load_ms=9.0,
        )

        result = probe_capture_mode(
            mode,
            capture_factory=FakeCapture,
            clock=StepClock(),
            sleeper=lambda seconds: slept.append(seconds),
        )

        self.assertEqual(slept, [0.009, 0.009, 0.009])
        self.assertEqual(result.requested["load"], "sleep")
        self.assertEqual(result.requested["load_ms"], 9.0)
        self.assertEqual(result.requested["load_placement"], "inline")
        self.assertGreater(result.avg_load_ms, 0.0)
        self.assertEqual(result.load_iterations, 3)

    def test_probe_capture_mode_records_threaded_load_stats(self):
        class FakeWorker:
            def stop(self):
                return {
                    "iterations": 4,
                    "durations_ms": [8.0, 9.0, 10.0, 9.0],
                    "periods_ms": [11.0, 12.0, 13.0],
                }

        started = []

        def make_worker(mode, _clock, _sleeper):
            started.append(mode)
            return FakeWorker()

        mode = CaptureMode(
            device=2,
            width=1280,
            height=720,
            fps=120,
            pixel_format="MJPEG",
            samples=2,
            warmup=1,
            load="busy",
            load_ms=9.0,
            load_placement="thread",
        )

        result = probe_capture_mode(
            mode,
            capture_factory=FakeCapture,
            clock=StepClock(),
            load_worker_factory=make_worker,
        )

        self.assertEqual(started, [mode])
        self.assertEqual(result.requested["load_placement"], "thread")
        self.assertEqual(result.load_iterations, 4)
        self.assertAlmostEqual(result.avg_load_ms, 9.0)
        self.assertAlmostEqual(result.avg_load_period_ms, 12.0)

    def test_probe_outputs_markdown_and_jsonl(self):
        result = probe_capture_mode(
            CaptureMode(
                device=0,
                width=1280,
                height=720,
                fps=120,
                pixel_format="MJPEG",
                samples=1,
                load="busy",
                load_ms=1.0,
            ),
            capture_factory=FakeCapture,
            clock=StepClock(),
        )

        markdown = format_results_markdown([result])
        self.assertIn("capture mode probe", markdown.lower())
        self.assertIn("MJPEG", markdown)
        self.assertIn("busy 1.000ms inline", markdown)
        self.assertIn("load iters", markdown)
        self.assertIn("load period", markdown)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            write_results_jsonl(path, [result])
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(records[0]["requested"]["pixel_format"], "MJPEG")
        self.assertEqual(records[0]["requested"]["load"], "busy")
        self.assertEqual(records[0]["requested"]["load_placement"], "inline")


if __name__ == "__main__":
    unittest.main()
