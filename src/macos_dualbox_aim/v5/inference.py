from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..core.inference import DetectionResult, Frame, RealtimeInference
from ..core.model_runtime.detector import CoreMLDetectorV5


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
        super().__init__(
            detector=_RealtimeDetectorAdapter(CoreMLDetectorV5(model_path, class_count=class_count)),
            capture_device=capture_device,
            target_fps=target_fps,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            enable_display=enable_display,
            crop_size=crop_size,
            capture_resolution=capture_resolution,
            pixel_format=pixel_format,
            frame_queue_size=frame_queue_size,
        )


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


__all__ = ["DetectionResult", "Frame", "RealtimeInference", "RealtimeInferenceV5"]
