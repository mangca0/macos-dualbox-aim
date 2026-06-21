import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .capture import CaptureConfig, Frame, configure_capture, crop_offset, open_capture, read_center_crop


@dataclass
class DetectionResult:
    frame_id: int
    timestamp: float
    detections: List[Dict]
    inference_time_ms: float
    fps: float
    latency_ms: Dict[str, float]
    frame: Optional[np.ndarray] = None


class RealtimeDetector(Protocol):
    def predict_with_timing(
        self,
        image: np.ndarray,
        iou_threshold: float,
        confidence_threshold: float,
    ) -> tuple[List[Dict], Dict[str, float]]: ...

    def visualize_predictions(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray: ...


class RealtimeInference:
    def __init__(
        self,
        detector: RealtimeDetector,
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
        self.crop_offset = crop_offset(capture_resolution, crop_size)

        self.detector = detector
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

    def start(self):
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()
        if self.enable_display:
            self._display_loop()
        else:
            self.wait_until_stopped()

    def stop(self):
        self.running = False
        for thread in (self.capture_thread, self.inference_thread):
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        cv2.destroyAllWindows()

    def wait_until_stopped(self):
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def _capture_loop(self):
        config = CaptureConfig(
            device=self.capture_device,
            target_fps=self.target_fps,
            crop_size=self.crop_size,
            capture_resolution=self.capture_resolution,
            pixel_format=self.pixel_format,
        )
        capture = open_capture(config)
        configure_capture(capture, config)

        if not capture.isOpened():
            self.logger.error("Failed to open capture device %s", self.capture_device)
            self.running = False
            return

        while self.running:
            self.frame_id += 1
            frame = read_center_crop(
                capture,
                frame_id=self.frame_id,
                crop_size=self.crop_size,
                offset=self.crop_offset,
            )
            if frame is None:
                continue
            self._increment_counter("frames_captured")
            replaced = self._put_latest(self.frame_queue, frame)
            if replaced:
                self._increment_counter("frame_queue_replaced")
                self._increment_counter("frames_dropped")
        capture.release()

    def _inference_loop(self):
        count = 0
        start = time.perf_counter()
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                while not self.frame_queue.empty():
                    self._increment_counter("frame_queue_drained")
                    self._increment_counter("frames_dropped")
                    frame = self.frame_queue.get_nowait()
            except queue.Empty:
                continue

            inference_start = time.perf_counter()
            queue_ms = max(0.0, (inference_start - frame.captured_at) * 1000.0)
            detections, model_timings = self.detector.predict_with_timing(
                frame.image,
                self.iou_threshold,
                self.confidence_threshold,
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            self.inference_times.append(inference_ms)
            self._increment_counter("frames_inferred")
            count += 1
            elapsed = max(0.001, time.perf_counter() - start)
            self.actual_inference_fps = min(self.target_fps, count / elapsed)

            latency_ms = {
                "capture_read_ms": frame.capture_ms,
                "crop_ms": frame.crop_ms,
                "queue_wait_ms": queue_ms,
                **model_timings,
                "inference_ms": inference_ms,
                "detection_callback_ms": 0.0,
                "target_select_ms": 0.0,
                "pid_ms": 0.0,
                "kmbox_send_ack_ms": 0.0,
                "program_total_ms": 0.0,
                "read_included_total_ms": 0.0,
            }

            result = DetectionResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
                inference_time_ms=inference_ms,
                fps=self.actual_inference_fps,
                latency_ms=latency_ms,
                frame=frame.image,
            )
            if self.on_detection:
                callback_start = time.perf_counter()
                self.on_detection(result)
                latency_ms["detection_callback_ms"] = (time.perf_counter() - callback_start) * 1000.0
            latency_ms["program_total_ms"] = (
                latency_ms["queue_wait_ms"] +
                latency_ms["inference_ms"] +
                latency_ms["detection_callback_ms"]
            )
            latency_ms["read_included_total_ms"] = (
                latency_ms["capture_read_ms"] +
                latency_ms["crop_ms"] +
                latency_ms["program_total_ms"]
            )
            result.latency_ms = dict(latency_ms)
            self._record_latency(result.frame_id, latency_ms)
            if self.enable_display:
                self._put_latest(self.result_queue, (frame.image, result))

    def get_latency_snapshot(self) -> Dict[str, object]:
        with self.latency_lock:
            samples = [dict(sample) for sample in self.latency_samples]
            fps = self.actual_inference_fps
            counters = self._counter_snapshot_locked()

        if not samples:
            return {
                "available": False,
                "fps": fps,
                "window": 0,
                "current": {},
                "avg": {},
                "p95": {},
                "max": {},
                "counters": counters,
            }

        keys = [key for key in samples[-1] if key != "frame_id"]
        return {
            "available": True,
            "fps": fps,
            "window": len(samples),
            "current": {key: samples[-1][key] for key in keys},
            "avg": {key: mean(sample[key] for sample in samples) for key in keys},
            "p95": {key: self._percentile([sample[key] for sample in samples], 95.0) for key in keys},
            "max": {key: max(sample[key] for sample in samples) for key in keys},
            "counters": counters,
        }

    def _record_latency(self, frame_id: int, latency_ms: Dict[str, float]):
        sample = {"frame_id": float(frame_id)}
        sample.update({key: float(value) for key, value in latency_ms.items()})
        with self.latency_lock:
            self.latency_samples.append(sample)

    def _increment_counter(self, name: str, amount: int = 1):
        with self.latency_lock:
            setattr(self, name, int(getattr(self, name)) + int(amount))

    def _counter_snapshot_locked(self) -> Dict[str, int]:
        return {
            "frames_captured": self.frames_captured,
            "frames_inferred": self.frames_inferred,
            "frames_dropped": self.frames_dropped,
            "frame_queue_replaced": self.frame_queue_replaced,
            "frame_queue_drained": self.frame_queue_drained,
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile / 100.0
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = index - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def _display_loop(self):
        while self.running:
            try:
                image, result = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            display = self.detector.visualize_predictions(image, result.detections)
            cv2.imshow("Aimbot Core", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()

    def _put_latest(self, target_queue: queue.Queue, item) -> bool:
        try:
            target_queue.put_nowait(item)
            return False
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            target_queue.put_nowait(item)
            return True
