from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import AimbotConfigV7


@dataclass(frozen=True)
class CrosshairResult:
    found: bool = False
    crosshair_x: float = 0.0
    crosshair_y: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class CrosshairDetector:
    def __init__(self, config: AimbotConfigV7):
        self.config = config
        self.last_result = CrosshairResult()

    def detect(self, image: Optional[np.ndarray]) -> CrosshairResult:
        if not self.config.crosshair_enabled or image is None or image.ndim < 3 or image.shape[2] < 3:
            self.last_result = CrosshairResult()
            return self.last_result

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            self.last_result = CrosshairResult()
            return self.last_result

        center_x = width / 2.0
        center_y = height / 2.0
        radius = int(self.config.crosshair_search_radius)
        x_start = max(0, int(center_x) - radius)
        x_end = min(width - 1, int(center_x) + radius)
        y_start = max(0, int(center_y) - radius)
        y_end = min(height - 1, int(center_y) + radius)
        roi = image[y_start:y_end + 1, x_start:x_end + 1]

        mask = self._match_mask(roi, center_x - x_start, center_y - y_start, radius)
        matched_y, matched_x = np.nonzero(mask)
        if matched_x.size < self.config.crosshair_min_pixels:
            self.last_result = CrosshairResult()
            return self.last_result

        crosshair_x = float(np.mean(matched_x) + x_start)
        crosshair_y = float(np.mean(matched_y) + y_start)
        self.last_result = CrosshairResult(
            found=True,
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
            offset_x=crosshair_x - center_x,
            offset_y=crosshair_y - center_y,
        )
        return self.last_result

    def _match_mask(self, image: np.ndarray, center_x: float, center_y: float, radius: int) -> np.ndarray:
        color_mask = self._hsv_mask(image) if self.config.crosshair_use_hsv else self._rgb_mask(image)
        if radius <= 0:
            y = int(center_y)
            x = int(center_x)
            radius_mask = np.zeros(color_mask.shape, dtype=bool)
            if 0 <= y < radius_mask.shape[0] and 0 <= x < radius_mask.shape[1]:
                radius_mask[y, x] = True
            return color_mask & radius_mask

        y_indices, x_indices = np.ogrid[:color_mask.shape[0], :color_mask.shape[1]]
        radius_mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 <= radius * radius
        return color_mask & radius_mask

    def _hsv_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        if self.config.crosshair_h_min <= self.config.crosshair_h_max:
            h_match = (h >= self.config.crosshair_h_min) & (h <= self.config.crosshair_h_max)
        else:
            h_match = (h >= self.config.crosshair_h_min) | (h <= self.config.crosshair_h_max)
        return (
            h_match
            & (s >= self.config.crosshair_s_min)
            & (s <= self.config.crosshair_s_max)
            & (v >= self.config.crosshair_v_min)
            & (v <= self.config.crosshair_v_max)
        )

    def _rgb_mask(self, image: np.ndarray) -> np.ndarray:
        channels = image[:, :, :3].astype(np.int32, copy=False)
        db = channels[:, :, 0] - int(self.config.crosshair_target_b)
        dg = channels[:, :, 1] - int(self.config.crosshair_target_g)
        dr = channels[:, :, 2] - int(self.config.crosshair_target_r)
        tolerance_sq = float(self.config.crosshair_color_tolerance) ** 2
        return (dr * dr + dg * dg + db * db) <= tolerance_sq

    def _matches(self, b: int, g: int, r: int) -> bool:
        if self.config.crosshair_use_hsv:
            return self._matches_hsv(b, g, r)
        return self._matches_rgb(b, g, r)

    def _matches_hsv(self, b: int, g: int, r: int) -> bool:
        h, s, v = self._rgb_to_hsv(r, g, b)
        if self.config.crosshair_h_min <= self.config.crosshair_h_max:
            h_match = self.config.crosshair_h_min <= h <= self.config.crosshair_h_max
        else:
            h_match = h >= self.config.crosshair_h_min or h <= self.config.crosshair_h_max
        return (
            h_match
            and self.config.crosshair_s_min <= s <= self.config.crosshair_s_max
            and self.config.crosshair_v_min <= v <= self.config.crosshair_v_max
        )

    def _matches_rgb(self, b: int, g: int, r: int) -> bool:
        dr = float(r - self.config.crosshair_target_r)
        dg = float(g - self.config.crosshair_target_g)
        db = float(b - self.config.crosshair_target_b)
        return (dr * dr + dg * dg + db * db) ** 0.5 <= self.config.crosshair_color_tolerance

    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[int, int, int]:
        rf = r / 255.0
        gf = g / 255.0
        bf = b / 255.0
        max_c = max(rf, gf, bf)
        min_c = min(rf, gf, bf)
        delta = max_c - min_c
        value = int(max_c * 255.0)
        if max_c < 0.0001:
            return 0, 0, value

        saturation = int((delta / max_c) * 255.0)
        if delta < 0.0001:
            return 0, saturation, value

        if max_c == rf:
            hue = 60.0 * (((gf - bf) / delta) % 6.0)
        elif max_c == gf:
            hue = 60.0 * ((bf - rf) / delta + 2.0)
        else:
            hue = 60.0 * ((rf - gf) / delta + 4.0)
        if hue < 0.0:
            hue += 360.0
        return int(hue / 2.0), saturation, value

    @staticmethod
    def _centroid(pixels: list[tuple[int, int]]) -> Optional[tuple[float, float]]:
        if not pixels:
            return None
        sum_x = sum(x for x, _y in pixels)
        sum_y = sum(y for _x, y in pixels)
        return sum_x / len(pixels), sum_y / len(pixels)
