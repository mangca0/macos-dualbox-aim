import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np


PIXEL_FORMAT_FOURCC = {
    "MJPEG": "MJPG",
    "MJPG": "MJPG",
    "YUY2": "YUY2",
    "RGB3": "RGB3",
    "BGR3": "BGR3",
    "UYVY": "UYVY",
}


@dataclass(frozen=True)
class CaptureConfig:
    device: int = 0
    target_fps: int = 240
    crop_size: Tuple[int, int] = (320, 320)
    capture_resolution: Tuple[int, int] = (1920, 1080)
    pixel_format: str = "MJPEG"
    backend: Optional[int] = None


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    captured_at: float
    capture_ms: float
    crop_ms: float
    image: np.ndarray


def open_capture(
    config: CaptureConfig,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
) -> Any:
    if config.backend is None:
        return capture_factory(config.device)
    return capture_factory(config.device, config.backend)


def configure_capture(capture: Any, config: CaptureConfig) -> None:
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_code(config.pixel_format)))
    capture.set(cv2.CAP_PROP_FPS, config.target_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_resolution[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_resolution[1])


def crop_offset(capture_resolution: Tuple[int, int], crop_size: Tuple[int, int]) -> Tuple[int, int]:
    return (
        (int(capture_resolution[0]) - int(crop_size[0])) // 2,
        (int(capture_resolution[1]) - int(crop_size[1])) // 2,
    )


def center_crop(frame: np.ndarray, crop_size: Tuple[int, int], offset: Tuple[int, int]) -> np.ndarray:
    crop_x, crop_y = offset
    width, height = crop_size
    return frame[crop_y:crop_y + height, crop_x:crop_x + width]


def read_center_crop(
    capture: Any,
    *,
    frame_id: int,
    crop_size: Tuple[int, int],
    offset: Tuple[int, int],
    clock: Callable[[], float] = time.perf_counter,
) -> Optional[Frame]:
    capture_start = clock()
    ok, frame = capture.read()
    capture_ms = (clock() - capture_start) * 1000.0
    if not ok:
        return None

    crop_start = clock()
    cropped = center_crop(frame, crop_size, offset)
    crop_ms = (clock() - crop_start) * 1000.0
    return Frame(frame_id, time.time(), clock(), capture_ms, crop_ms, cropped)


def fourcc_code(pixel_format: str) -> str:
    return PIXEL_FORMAT_FOURCC.get(pixel_format.upper(), "MJPG")
