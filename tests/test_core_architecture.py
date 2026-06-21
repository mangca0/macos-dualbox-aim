import unittest

import cv2
import numpy as np

from macos_dualbox_aim.core.capture import CaptureConfig, Frame, center_crop, configure_capture, crop_offset, fourcc_code
from macos_dualbox_aim.core.inference import RealtimeInference
from macos_dualbox_aim.v5.inference import RealtimeInferenceV5


class FakeCapture:
    def __init__(self):
        self.set_calls = []

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True


class FakeDetector:
    def __init__(self):
        self.calls = []

    def predict_with_timing(self, image, iou_threshold, confidence_threshold):
        self.calls.append((image.shape, iou_threshold, confidence_threshold))
        return (
            [{"bbox": [0.5, 0.5, 0.1, 0.1], "confidence": 0.9, "class_id": 1}],
            {"coreml_ms": 1.25},
        )

    def visualize_predictions(self, image, detections):
        return image


class CoreArchitectureTests(unittest.TestCase):
    def test_capture_helpers_configure_card_and_center_crop(self):
        config = CaptureConfig(
            device=2,
            target_fps=240,
            crop_size=(320, 320),
            capture_resolution=(1920, 1080),
            pixel_format="MJPEG",
        )
        capture = FakeCapture()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[380, 800] = [255, 255, 255]

        configure_capture(capture, config)
        offset = crop_offset(config.capture_resolution, config.crop_size)
        cropped = center_crop(frame, config.crop_size, offset)

        self.assertEqual(fourcc_code("MJPEG"), "MJPG")
        self.assertIn((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")), capture.set_calls)
        self.assertEqual(offset, (800, 380))
        self.assertEqual(cropped.shape, (320, 320, 3))
        self.assertTrue(np.array_equal(cropped[0, 0], [255, 255, 255]))

    def test_core_realtime_inference_processes_frame_with_injected_detector(self):
        detector = FakeDetector()
        engine = RealtimeInference(
            detector,
            confidence_threshold=0.65,
            iou_threshold=0.3,
            crop_size=(320, 320),
            capture_resolution=(1920, 1080),
        )
        callbacks = []

        def on_detection(result):
            callbacks.append(result)
            engine.running = False

        engine.on_detection = on_detection
        engine.running = True
        engine.frame_queue.put(Frame(
            frame_id=7,
            timestamp=123.0,
            captured_at=0.0,
            capture_ms=2.0,
            crop_ms=0.5,
            image=np.zeros((320, 320, 3), dtype=np.uint8),
        ))

        engine._inference_loop()

        self.assertEqual(detector.calls[0], ((320, 320, 3), 0.3, 0.65))
        self.assertEqual(callbacks[0].detections[0]["class_id"], 1)
        self.assertEqual(callbacks[0].latency_ms["coreml_ms"], 1.25)
        self.assertEqual(engine.get_latency_snapshot()["counters"]["frames_inferred"], 1)

    def test_v5_realtime_inference_uses_core_realtime_base(self):
        self.assertTrue(issubclass(RealtimeInferenceV5, RealtimeInference))


if __name__ == "__main__":
    unittest.main()
