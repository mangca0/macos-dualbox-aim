import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import coremltools as ct
import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    frame_id: int
    timestamp: float
    detections: List[Dict]
    inference_time_ms: float
    fps: float
    latency_ms: Dict[str, float]
    frame: Optional[np.ndarray] = None


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    captured_at: float
    capture_ms: float
    crop_ms: float
    image: np.ndarray


class CoreMLDetector:
    def __init__(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = ct.models.MLModel(str(model_file))
        self.input_name = ""
        self.input_shape = (0, 0)
        self.iou_threshold_name: Optional[str] = None
        self.confidence_threshold_name: Optional[str] = None
        self.output_names: List[str] = []
        self.has_nms = False
        self.decode_cache = {}
        self._inspect_model()

    def predict(self, image: np.ndarray, iou_threshold: float, confidence_threshold: float) -> List[Dict]:
        detections, _timings = self.predict_with_timing(image, iou_threshold, confidence_threshold)
        return detections

    def predict_with_timing(
        self,
        image: np.ndarray,
        iou_threshold: float,
        confidence_threshold: float,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        preprocess_start = time.perf_counter()
        inputs = {self.input_name: self._preprocess(image)}
        if self.iou_threshold_name:
            inputs[self.iou_threshold_name] = iou_threshold
        if self.confidence_threshold_name:
            inputs[self.confidence_threshold_name] = confidence_threshold
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        model_start = time.perf_counter()
        predictions = self.model.predict(inputs)
        model_ms = (time.perf_counter() - model_start) * 1000.0

        postprocess_start = time.perf_counter()
        if self.has_nms:
            detections = self._parse_nms_predictions(predictions, confidence_threshold)
        else:
            detections = self._parse_raw_predictions(predictions, confidence_threshold, iou_threshold)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        return detections, {
            "preprocess_ms": preprocess_ms,
            "coreml_ms": model_ms,
            "postprocess_ms": postprocess_ms,
        }

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
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"{detection.get('class_id', 0)} {float(detection.get('confidence', 0.0)):.2f}"
            cv2.putText(output, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return output

    def _inspect_model(self):
        spec = self.model.get_spec()
        for input_desc in spec.description.input:
            input_type = input_desc.type.WhichOneof("Type")
            if input_type == "imageType":
                self.input_name = input_desc.name
                image_type = input_desc.type.imageType
                self.input_shape = (image_type.height, image_type.width)
            elif input_type == "doubleType":
                lower_name = input_desc.name.lower()
                if "iou" in lower_name:
                    self.iou_threshold_name = input_desc.name
                elif "conf" in lower_name:
                    self.confidence_threshold_name = input_desc.name

        self.output_names = [output_desc.name for output_desc in spec.description.output]
        self.has_nms = "coordinates" in self.output_names and "confidence" in self.output_names
        if not self.input_name or self.input_shape == (0, 0):
            raise ValueError("CoreML image input was not found")

    def _preprocess(self, image: np.ndarray) -> Image.Image:
        expected_h, expected_w = self.input_shape
        if image.shape[0] != expected_h or image.shape[1] != expected_w:
            image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _parse_nms_predictions(self, predictions: Dict, confidence_threshold: float) -> List[Dict]:
        coordinates = predictions.get("coordinates")
        confidence = predictions.get("confidence")
        if coordinates is None or confidence is None:
            return []

        results = []
        for index in range(coordinates.shape[0]):
            class_scores = confidence[index]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if conf < confidence_threshold:
                continue
            results.append({
                "bbox": coordinates[index].tolist(),
                "confidence": conf,
                "class_id": class_id,
            })
        return results

    def _parse_raw_predictions(self, predictions: Dict, confidence_threshold: float, iou_threshold: float) -> List[Dict]:
        raw_output = list(predictions.values())[0]
        if isinstance(raw_output, dict):
            raw_output = next(iter(raw_output.values()))
        if hasattr(raw_output, "numpy"):
            raw_output = raw_output.numpy()

        predictions_array = raw_output[0]
        boxes_raw = predictions_array[:, :4]
        objectness = predictions_array[:, 4]
        class_probs = predictions_array[:, 5:]
        boxes_sigmoid = 1.0 / (1.0 + np.exp(-boxes_raw))

        input_size = self.input_shape[0]
        grid_x, grid_y, stride = self._decode_grid(input_size, len(predictions_array))
        decoded_x = (boxes_sigmoid[:, 0] * 2.0 - 0.5 + grid_x) * stride
        decoded_y = (boxes_sigmoid[:, 1] * 2.0 - 0.5 + grid_y) * stride
        decoded_w = boxes_sigmoid[:, 2] * 2.0 * stride
        decoded_h = boxes_sigmoid[:, 3] * 2.0 * stride

        x1 = np.clip(decoded_x - decoded_w * 0.5, 0, input_size)
        y1 = np.clip(decoded_y - decoded_h * 0.5, 0, input_size)
        x2 = np.clip(decoded_x + decoded_w * 0.5, 0, input_size)
        y2 = np.clip(decoded_y + decoded_h * 0.5, 0, input_size)

        class_ids = np.argmax(class_probs, axis=1)
        confidences = objectness * np.max(class_probs, axis=1)
        valid = confidences > confidence_threshold
        boxes = np.stack([x1, y1, x2, y2], axis=1)[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]
        if len(boxes) == 0:
            return []

        keep = []
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            class_indices = np.where(mask)[0]
            keep.extend(class_indices[self._nms(boxes[mask], confidences[mask], iou_threshold)])

        sorted_indices = np.array(keep)[np.argsort(confidences[keep])[::-1]]
        results = []
        for index in sorted_indices:
            box = boxes[index]
            cx = (box[0] + box[2]) * 0.5 / input_size
            cy = (box[1] + box[3]) * 0.5 / input_size
            width = (box[2] - box[0]) / input_size
            height = (box[3] - box[1]) / input_size
            results.append({
                "bbox": [float(cx), float(cy), float(width), float(height)],
                "confidence": float(confidences[index]),
                "class_id": int(class_ids[index]),
            })
        return results

    def _decode_grid(self, input_size: int, num_anchors: int):
        key = (input_size, num_anchors)
        if key in self.decode_cache:
            return self.decode_cache[key]

        grid_x_parts = []
        grid_y_parts = []
        stride_parts = []
        for stride_value in np.array([8, 16, 32], dtype=np.float32):
            grid_size = input_size // int(stride_value)
            grid_y, grid_x = np.meshgrid(
                np.arange(grid_size, dtype=np.float32),
                np.arange(grid_size, dtype=np.float32),
                indexing="ij",
            )
            grid_x_parts.append(grid_x.reshape(-1))
            grid_y_parts.append(grid_y.reshape(-1))
            stride_parts.append(np.full(grid_size * grid_size, stride_value, dtype=np.float32))

        grid_x = np.concatenate(grid_x_parts)
        grid_y = np.concatenate(grid_y_parts)
        stride = np.concatenate(stride_parts)
        if len(grid_x) != num_anchors:
            raise ValueError(f"Raw output anchors mismatch: expected {len(grid_x)}, got {num_anchors}")
        self.decode_cache[key] = (grid_x, grid_y, stride)
        return self.decode_cache[key]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        indices = np.argsort(scores)[::-1]
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            ious = self._compute_iou(boxes[current], boxes[indices[1:]])
            indices = indices[1:][ious < iou_threshold]
        return keep

    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return intersection / (box_area + boxes_area - intersection + 1e-6)


class RealtimeInference:
    def __init__(
        self,
        model_path: str,
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

        self.detector = CoreMLDetector(model_path)
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
        capture = cv2.VideoCapture(self.capture_device)
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc_code(self.pixel_format))
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_resolution[1])

        if not capture.isOpened():
            self.logger.error("Failed to open capture device %s", self.capture_device)
            self.running = False
            return

        while self.running:
            capture_start = time.perf_counter()
            ok, frame = capture.read()
            capture_ms = (time.perf_counter() - capture_start) * 1000.0
            if not ok:
                continue
            self.frame_id += 1
            self._increment_counter("frames_captured")
            crop_start = time.perf_counter()
            crop_x, crop_y = self.crop_offset
            width, height = self.crop_size
            frame = frame[crop_y:crop_y + height, crop_x:crop_x + width]
            crop_ms = (time.perf_counter() - crop_start) * 1000.0
            replaced = self._put_latest(
                self.frame_queue,
                Frame(self.frame_id, time.time(), time.perf_counter(), capture_ms, crop_ms, frame),
            )
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
            cv2.imshow("Aimbot V3", display)
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

    def _fourcc_code(self, pixel_format: str) -> str:
        mapping = {
            "MJPEG": "MJPG",
            "MJPG": "MJPG",
            "YUY2": "YUY2",
            "RGB3": "RGB3",
            "BGR3": "BGR3",
            "UYVY": "UYVY",
        }
        return mapping.get(pixel_format, "MJPG")
