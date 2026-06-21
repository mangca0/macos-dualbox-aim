#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from macos_dualbox_aim.core.model_runtime.detector import CoreMLDetectorV5
from macos_dualbox_aim.core.model_runtime.probe import (
    compare_arrays,
    summarize_detections,
    summarize_timings,
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Probe V5 Core ML check/fast detector packages.")
    parser.add_argument(
        "--check-model",
        type=Path,
        default=Path("models/converted/cs2_fp16_fp32_check.mlpackage"),
        help="FP32 check Core ML package.",
    )
    parser.add_argument(
        "--fast-model",
        type=Path,
        default=Path("models/converted/cs2_fp16_fp16_fast.mlpackage"),
        help="FP16 fast Core ML package.",
    )
    parser.add_argument("--image", type=Path, help="Optional BGR/RGB image file. Defaults to a black 320x320 frame.")
    parser.add_argument("--input-size", default="320x320", help="Fallback synthetic frame size as HEIGHTxWIDTH.")
    parser.add_argument("--class-count", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args([] if argv is None else argv)


def run_probe(args) -> dict:
    image = load_probe_image(args.image, _parse_size(args.input_size))
    check_detector = CoreMLDetectorV5(str(args.check_model), class_count=args.class_count)
    fast_detector = CoreMLDetectorV5(str(args.fast_model), class_count=args.class_count)

    for _ in range(max(0, args.warmup)):
        check_detector.predict_with_timing(image, args.iou, args.confidence)
        fast_detector.predict_with_timing(image, args.iou, args.confidence)

    check_timings = []
    fast_timings = []
    check_detections = []
    fast_detections = []
    check_predictions = None
    fast_predictions = None
    for _ in range(max(1, args.runs)):
        check_detections, check_predictions, check_timing = check_detector.predict_with_timing(
            image,
            args.iou,
            args.confidence,
        )
        fast_detections, fast_predictions, fast_timing = fast_detector.predict_with_timing(
            image,
            args.iou,
            args.confidence,
        )
        check_timings.append(check_timing)
        fast_timings.append(fast_timing)

    check_output = _first_prediction_array(check_predictions)
    fast_output = _first_prediction_array(fast_predictions)
    return {
        "image": {
            "path": str(args.image) if args.image else None,
            "shape": list(image.shape),
        },
        "check_model": _model_summary(args.check_model, check_detector),
        "fast_model": _model_summary(args.fast_model, fast_detector),
        "raw_output_diff": compare_arrays(check_output, fast_output),
        "detections": {
            "check": summarize_detections(check_detections),
            "fast": summarize_detections(fast_detections),
        },
        "timings": {
            "check": _summarize_timing_records(check_timings),
            "fast": _summarize_timing_records(fast_timings),
        },
    }


def load_probe_image(path: Path | None, fallback_size: tuple[int, int]) -> np.ndarray:
    if path is None:
        height, width = fallback_size
        return np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def _summarize_timing_records(records: list[dict[str, float]]) -> dict:
    keys = sorted(records[0]) if records else []
    return {key: summarize_timings([record[key] for record in records]) for key in keys}


def _first_prediction_array(predictions: dict) -> np.ndarray:
    if predictions is None:
        raise ValueError("No predictions were recorded")
    value = next(iter(predictions.values()))
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _model_summary(path: Path, detector: CoreMLDetectorV5) -> dict:
    contract = detector.contract
    return {
        "path": str(path),
        "input_name": contract.input_name,
        "input_kind": str(contract.input_kind),
        "input_size": list(contract.input_size),
        "output_names": list(contract.output_names),
        "output_kind": str(contract.output_kind),
        "output_layout": str(contract.output_layout),
        "adapter": contract.adapter_name,
    }


def _parse_size(value: str) -> tuple[int, int]:
    try:
        height_text, width_text = value.lower().split("x", 1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Input size must use HEIGHTxWIDTH, for example 320x320") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Input size must be positive")
    return height, width


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = run_probe(args)
    output = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
