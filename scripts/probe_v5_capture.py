#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from macos_dualbox_aim.v5.model_runtime.detector import CoreMLDetectorV5


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Show V5 Core ML detections on the capture-card center crop.")
    parser.add_argument("--model", type=Path, default=Path("models/converted/cs2_fp16_fp16_fast.mlpackage"))
    parser.add_argument("--check-model", type=Path, default=Path("models/converted/cs2_fp16_fp32_check.mlpackage"))
    parser.add_argument("--use-check-model", action="store_true", help="Use the FP32 check model instead of the fast model.")
    parser.add_argument("--class-count", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--capture-device", type=int, default=0)
    parser.add_argument("--target-fps", type=int, default=240)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--fov-width", type=int, default=320)
    parser.add_argument("--fov-height", type=int, default=320)
    parser.add_argument("--pixel-format", default="MJPEG")
    return parser.parse_args([] if argv is None else argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_path = args.check_model if args.use_check_model else args.model
    detector = CoreMLDetectorV5(str(model_path), class_count=args.class_count)

    capture = cv2.VideoCapture(args.capture_device)
    fourcc = cv2.VideoWriter_fourcc(*_fourcc_code(args.pixel_format))
    capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    capture.set(cv2.CAP_PROP_FPS, args.target_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.screen_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.screen_height)
    if not capture.isOpened():
        print(f"Failed to open capture device {args.capture_device}")
        return 1

    window_name = "V5 capture probe - center crop"
    frame_count = 0
    start = time.perf_counter()
    crop_size = (args.fov_width, args.fov_height)
    print(f"Running V5 capture probe with {model_path}. Press q or Esc to exit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            frame_count += 1
            crop, _offset = center_crop(frame, crop_size)
            detections, _predictions, timings = detector.predict_with_timing(
                crop,
                args.iou,
                args.confidence,
            )
            display = draw_detections(crop, detections)
            fps = frame_count / max(0.001, time.perf_counter() - start)
            _draw_overlay(display, detections, timings, fps)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def center_crop(frame: np.ndarray, crop_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    crop_w, crop_h = crop_size
    height, width = frame.shape[:2]
    crop_x = max(0, (width - crop_w) // 2)
    crop_y = max(0, (height - crop_h) // 2)
    return frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy(), (crop_x, crop_y)


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
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


def _draw_overlay(image: np.ndarray, detections: list[dict], timings: dict[str, float], fps: float) -> None:
    text = (
        f"det {len(detections)} | fps {fps:.1f} | "
        f"coreml {timings.get('coreml_ms', 0.0):.2f} ms | "
        f"post {timings.get('postprocess_ms', 0.0):.2f} ms"
    )
    cv2.putText(image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def _fourcc_code(pixel_format: str) -> str:
    code = pixel_format.upper()
    if code == "MJPEG":
        return "MJPG"
    if len(code) != 4:
        return "MJPG"
    return code


if __name__ == "__main__":
    raise SystemExit(main())
