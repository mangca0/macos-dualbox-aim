import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import AimbotConfigV3
from .kmbox import KmboxConfig, KmboxNet, SUCCESS
from .tracker import BoundingBox, DetectionObject, KalmanP


@dataclass
class Target:
    bbox: List[float]
    confidence: float
    class_id: int
    screen_x: float
    screen_y: float
    track_id: int = -1


class AimbotV3:
    def __init__(self, config: AimbotConfigV3):
        self.config = config
        self.tracker = self._new_tracker()
        self.kmbox: Optional[KmboxNet] = None
        self.active = False
        self.stats = {
            "frames_processed": 0,
            "detections_input": 0,
            "tracks_output": 0,
            "aims_executed": 0,
            "moves_sent": 0,
        }

    def connect(self) -> bool:
        self.kmbox = KmboxNet(KmboxConfig(
            ip=self.config.kmbox_ip,
            port=self.config.kmbox_port,
            mac=self.config.kmbox_mac,
        ))
        result = self.kmbox.init()
        return result == SUCCESS

    def disconnect(self):
        if self.kmbox is not None:
            self.kmbox.close()
            self.kmbox = None

    def activate(self):
        self.active = True
        self.reset_tracking()

    def deactivate(self):
        self.active = False
        self.reset_tracking()

    def on_activate(self):
        self.activate()

    def on_deactivate(self):
        self.deactivate()

    def is_active(self) -> bool:
        return self.active

    def reset_tracking(self):
        self.tracker = self._new_tracker()
        self.tracker.reset(reset_id=True)

    def update(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int],
        crop_offset: Tuple[int, int] = (0, 0),
        timing_ms: Optional[Dict[str, float]] = None,
    ) -> bool:
        if not self.active:
            if timing_ms is not None:
                timing_ms.setdefault("target_select_ms", 0.0)
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        target_start = time.perf_counter()
        target = self.process_detections(detections, frame_shape, crop_offset)
        if timing_ms is not None:
            timing_ms["target_select_ms"] = (time.perf_counter() - target_start) * 1000.0
        self.stats["frames_processed"] += 1

        if target is None:
            if timing_ms is not None:
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False
        return self.aim_at_target(target, timing_ms=timing_ms)

    def process_detections(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int],
        crop_offset: Tuple[int, int] = (0, 0),
    ) -> Optional[Target]:
        frame_h, frame_w = frame_shape[:2]
        objects = self._detections_to_objects(detections, frame_w, frame_h)
        self.stats["detections_input"] += len(objects)

        tracked = self.tracker.predict(objects)
        self.stats["tracks_output"] += len(tracked)
        if not tracked:
            return None

        return self._object_to_target(tracked[0], crop_offset)

    def _detections_to_objects(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
    ) -> List[DetectionObject]:
        objects: List[DetectionObject] = []
        for detection in detections:
            parsed = self._parse_bbox(detection.get("bbox", []), frame_w, frame_h)
            if parsed is None:
                continue
            x1, y1, x2, y2 = parsed
            objects.append(DetectionObject(
                bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                label=int(detection.get("class_id", 0)),
                prob=float(detection.get("confidence", 0.0)),
            ))
        return objects

    def _parse_bbox(
        self,
        bbox: List[float],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        if len(bbox) != 4:
            return None

        values = [float(item) for item in bbox]
        if all(0.0 <= value <= 1.0 for value in values):
            cx, cy, width, height = values
            x1 = (cx - width * 0.5) * frame_w
            y1 = (cy - height * 0.5) * frame_h
            x2 = (cx + width * 0.5) * frame_w
            y2 = (cy + height * 0.5) * frame_h
        else:
            x1, y1, x2, y2 = values
        return x1, y1, x2, y2

    def _object_to_target(self, obj: DetectionObject, crop_offset: Tuple[int, int]) -> Target:
        center_x = obj.bbox.x + obj.bbox.width * 0.5
        center_y = obj.bbox.y + obj.bbox.height * 0.5
        absolute_x = crop_offset[0] + center_x
        absolute_y = crop_offset[1] + center_y
        screen_x = absolute_x - self.config.screen_width * 0.5
        screen_y = absolute_y - self.config.screen_height * 0.5
        return Target(
            bbox=[obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height],
            confidence=obj.prob,
            class_id=obj.label,
            screen_x=screen_x,
            screen_y=screen_y,
            track_id=obj.track_id,
        )

    def aim_at_target(self, target: Target, timing_ms: Optional[Dict[str, float]] = None) -> bool:
        if not self.active or self.kmbox is None:
            if timing_ms is not None:
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        out_x = int(round(target.screen_x))
        out_y = int(round(target.screen_y))
        if out_x == 0 and out_y == 0:
            if timing_ms is not None:
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        send_start = time.perf_counter()
        result = self.kmbox.mouse_move(out_x, out_y)
        if timing_ms is not None:
            timing_ms["kmbox_send_ack_ms"] = (time.perf_counter() - send_start) * 1000.0
        self.stats["aims_executed"] += 1
        if result == SUCCESS:
            self.stats["moves_sent"] += 1
            return True
        return False

    def _new_tracker(self) -> KalmanP:
        return KalmanP(
            generate=self.config.tracker_generate,
            terminate=self.config.tracker_terminate,
            vx_noise=self.config.tracker_vx_noise,
            vy_noise=self.config.tracker_vy_noise,
            w_noise=self.config.tracker_w_noise,
            h_noise=self.config.tracker_h_noise,
            r_std=self.config.tracker_r_std,
        )
