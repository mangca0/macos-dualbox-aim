from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np


def compare_arrays(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    baseline_array = np.asarray(baseline, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    if baseline_array.shape != candidate_array.shape:
        raise ValueError(f"Array shape mismatch: {baseline_array.shape} vs {candidate_array.shape}")

    diff = np.abs(baseline_array - candidate_array)
    return {
        "shape": list(baseline_array.shape),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "baseline_min": float(baseline_array.min()),
        "baseline_max": float(baseline_array.max()),
        "candidate_min": float(candidate_array.min()),
        "candidate_max": float(candidate_array.max()),
    }


def summarize_timings(values_ms: Sequence[float]) -> dict:
    if not values_ms:
        return {
            "runs": 0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    values = [float(value) for value in values_ms]
    return {
        "runs": len(values),
        "median_ms": float(median(values)),
        "p95_ms": _percentile(values, 95),
        "mean_ms": float(mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def summarize_detections(detections: Iterable[dict], limit: int = 5) -> dict:
    sorted_detections = sorted(
        detections,
        key=lambda detection: float(detection.get("confidence", 0.0)),
        reverse=True,
    )
    return {
        "count": len(sorted_detections),
        "top": [
            {
                "bbox": [float(value) for value in detection.get("bbox", [])],
                "confidence": float(detection.get("confidence", 0.0)),
                "class_id": int(detection.get("class_id", 0)),
            }
            for detection in sorted_detections[:limit]
        ],
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
