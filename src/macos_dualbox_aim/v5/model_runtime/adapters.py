from typing import Dict, List

import numpy as np


class ImageNMSAdapter:
    def parse(self, predictions: Dict, confidence_threshold: float, iou_threshold: float) -> List[Dict]:
        del iou_threshold
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


class YoloV8TensorAdapter:
    def __init__(self, input_size: tuple[int, int], class_count: int):
        self.input_h, self.input_w = input_size
        self.class_count = int(class_count)

    def parse(self, predictions: Dict, confidence_threshold: float, iou_threshold: float) -> List[Dict]:
        raw_output = next(iter(predictions.values()))
        if hasattr(raw_output, "numpy"):
            raw_output = raw_output.numpy()
        output = np.asarray(raw_output, dtype=np.float32)
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"YOLOv8 output must be [1, channels, anchors] or [1, anchors, channels], got {output.shape}")

        channels = 4 + self.class_count
        if output.shape[1] == channels:
            decoded = output[0].T
        elif output.shape[2] == channels:
            decoded = output[0]
        else:
            raise ValueError(f"YOLOv8 output channels mismatch: expected {channels}, got {output.shape}")

        boxes_cxcywh = decoded[:, :4]
        class_probs = decoded[:, 4:]
        class_ids = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)
        valid = confidences >= confidence_threshold
        if not np.any(valid):
            return []

        boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh[valid], self.input_w, self.input_h)
        confidences = confidences[valid]
        class_ids = class_ids[valid]

        keep = []
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            class_indices = np.where(mask)[0]
            keep.extend(class_indices[_nms(boxes_xyxy[mask], confidences[mask], iou_threshold)])

        sorted_indices = np.array(keep, dtype=np.int64)[np.argsort(confidences[keep])[::-1]]
        results = []
        for index in sorted_indices:
            x1, y1, x2, y2 = boxes_xyxy[index]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            results.append({
                "bbox": [
                    float((x1 + x2) * 0.5 / self.input_w),
                    float((y1 + y2) * 0.5 / self.input_h),
                    float(width / self.input_w),
                    float(height / self.input_h),
                ],
                "confidence": float(confidences[index]),
                "class_id": int(class_ids[index]),
            })
        return results


def _cxcywh_to_xyxy(boxes: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    return np.stack(
        [
            np.clip(cx - width * 0.5, 0, input_w),
            np.clip(cy - height * 0.5, 0, input_h),
            np.clip(cx + width * 0.5, 0, input_w),
            np.clip(cy + height * 0.5, 0, input_h),
        ],
        axis=1,
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        ious = _compute_iou(boxes[current], boxes[indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / (box_area + boxes_area - intersection + 1e-6)
