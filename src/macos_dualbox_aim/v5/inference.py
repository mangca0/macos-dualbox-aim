import logging
import queue
import threading
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..v1.inference import DetectionResult, Frame, RealtimeInference
from .model_runtime.detector import CoreMLDetectorV5


class RealtimeInferenceV5(RealtimeInference):
    def __init__(
        self,
        model_path: str,
        class_count: int,
        capture_device: int = 0,
        target_fps: int = 240,
        confidence_threshold: float = 0.65,
        iou_threshold: float = 0.3,
        enable_display: bool = False,
        crop_size: Tuple[int, int] = (320, 320),
        capture_resolution: Tuple[int, int] = (1920, 1080),
        pixel_format: str = "MJPEG",
        frame_queue_size: int = 3,
    ):
        self.capture_device = capture_device
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.enable_display = enable_display
        self.crop_size = crop_size
        self.capture_resolution = capture_resolution
        self.pixel_format = pixel_format
        self.frame_queue_size = max(1, int(frame_queue_size))
        self.crop_offset = (
            (capture_resolution[0] - crop_size[0]) // 2,
            (capture_resolution[1] - crop_size[1]) // 2,
        )

        self.detector = _RealtimeDetectorAdapter(CoreMLDetectorV5(model_path, class_count=class_count))
        self.frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=self.frame_queue_size)
        self.result_queue: queue.Queue[tuple[np.ndarray, DetectionResult]] = queue.Queue(maxsize=3)
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.running = False
        self.frame_id = 0
        self.inference_times = deque(maxlen=100)
        self.latency_samples = deque(maxlen=120)
        self.latency_lock = threading.RLock()
        self.frames_captured = 0
        self.frames_inferred = 0
        self.frames_dropped = 0
        self.frame_queue_replaced = 0
        self.frame_queue_drained = 0
        self.actual_inference_fps = 0.0
        self.on_detection: Optional[Callable[[DetectionResult], None]] = None
        self.logger = logging.getLogger(__name__)


class _RealtimeDetectorAdapter:
    def __init__(self, detector: CoreMLDetectorV5):
        self.detector = detector

    def predict_with_timing(
        self,
        image: np.ndarray,
        iou_threshold: float,
        confidence_threshold: float,
    ) -> tuple[List[Dict], Dict[str, float]]:
        detections, _predictions, timings = self.detector.predict_with_timing(
            image,
            iou_threshold,
            confidence_threshold,
        )
        return detections, timings

    def visualize_predictions(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        output = image.copy()
        height, width = output.shape[:2]
        for detection in detections:
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue
            if all(0.0 <= float(value) <= 1.0 for value in bbox):
                cx, cy, box_w, box_h = [float(value) for value in bbox]
                x1 = int((cx - box_w * 0.5) * width)
                y1 = int((cy - box_h * 0.5) * height)
                x2 = int((cx + box_w * 0.5) * width)
                y2 = int((cy + box_h * 0.5) * height)
            else:
                x1, y1, x2, y2 = [int(float(value)) for value in bbox]
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"{int(detection.get('class_id', 0))} {float(detection.get('confidence', 0.0)):.2f}"
            cv2.putText(output, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return output
