import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import AimbotConfigV6
from .tracker import BoundingBox, DetectionObject, KalmanP
from ..core.kmbox import KmboxConfig, KmboxNet, SUCCESS


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


class IncrementalPid:
    """Incremental PID: out += Kp*(e-e1) + Ki*e + Kd*(e - 2*e1 + e2)."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output = 0.0
        self.previous_output = 0.0
        self.previous_error = 0.0
        self.previous_previous_error = 0.0
        self.output_min = -float("inf")
        self.output_max = float("inf")

    def configure(self, kp, ki, kd):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

    def reset(self, output=0.0):
        self.output = float(output)
        self.previous_output = float(output)
        self.previous_error = 0.0
        self.previous_previous_error = 0.0

    def set_output_limits(self, min_val, max_val):
        self.output_min = float(min_val)
        self.output_max = float(max_val)
        self.output = _clamp(self.output, self.output_min, self.output_max)

    def update(self, error, scale=1.0):
        self.previous_output = self.output
        if abs(error) < 0.3:
            error = 0.0
        delta = (
            self.kp * (error - self.previous_error)
            + self.ki * error
            + self.kd * (error - 2.0 * self.previous_error + self.previous_previous_error)
        )
        delta *= float(scale)
        self.output += delta
        if abs(self.output) < 0.5:
            self.output *= 0.9
        self.output = _clamp(self.output, self.output_min, self.output_max)
        self.previous_previous_error = self.previous_error
        self.previous_error = error
        return self.output


class DerivativePredictor:
    """Motion predictor based on smoothed velocity and acceleration estimates."""

    def __init__(self):
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.smooth_vel_x = 0.0
        self.smooth_vel_y = 0.0
        self.smooth_acc_x = 0.0
        self.smooth_acc_y = 0.0
        self.has_last = False

    def reset(self):
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.smooth_vel_x = 0.0
        self.smooth_vel_y = 0.0
        self.smooth_acc_x = 0.0
        self.smooth_acc_y = 0.0
        self.has_last = False

    def predict(self, error_x, error_y, prev_move_x, prev_move_y, dt):
        if not self.has_last:
            self.last_error_x = error_x
            self.last_error_y = error_y
            self.has_last = True
            return 0.0, 0.0

        dt = _clamp(dt, 0.001, 0.05)
        vel_x = _clamp((error_x - self.last_error_x + prev_move_x) / dt, -3000.0, 3000.0)
        vel_y = _clamp((error_y - self.last_error_y + prev_move_y) / dt, -3000.0, 3000.0)

        if abs(error_x) > 5.0 and vel_x * error_x < 0.0:
            vel_x *= 0.1
        if abs(error_y) > 5.0 and vel_y * error_y < 0.0:
            vel_y *= 0.1

        acc_x = _clamp((vel_x - self.smooth_vel_x) / dt, -5000.0, 5000.0)
        acc_y = _clamp((vel_y - self.smooth_vel_y) / dt, -5000.0, 5000.0)

        alpha_v = _clamp(1.0 - math.pow(0.75, dt / 0.01), 0.05, 0.8)
        alpha_a = _clamp(1.0 - math.pow(0.85, dt / 0.01), 0.05, 0.8)

        self.smooth_vel_x += (vel_x - self.smooth_vel_x) * alpha_v
        self.smooth_vel_y += (vel_y - self.smooth_vel_y) * alpha_v
        self.smooth_acc_x += (acc_x - self.smooth_acc_x) * alpha_a
        self.smooth_acc_y += (acc_y - self.smooth_acc_y) * alpha_a

        self.last_error_x = error_x
        self.last_error_y = error_y

        return (
            self.smooth_vel_x * dt + 0.5 * self.smooth_acc_x * dt * dt,
            self.smooth_vel_y * dt + 0.5 * self.smooth_acc_y * dt * dt,
        )


class PIDController:
    """Learned predictive PID controller: prediction fusion plus smoothstep ramp."""

    _TARGET_JUMP_THRESHOLD = 40.0

    def __init__(
        self,
        kp,
        ki,
        kd,
        slew_limit=40.0,
        max_speed=30.0,
        sensitivity=1.0,
        fov_radius=256,
        init_scale=0.6,
        ramp_time=0.5,
        pred_weight_x=0.5,
        pred_weight_y=0.5,
        **_ignored,
    ):
        self._base_kp = float(kp)
        self._base_ki = float(ki)
        self._base_kd = float(kd)
        self._pid_x = IncrementalPid(kp, ki, kd)
        self._pid_y = IncrementalPid(kp, ki, kd)
        self.slew_limit = float(slew_limit)
        self.max_speed = max(1.0, float(max_speed))
        self.sensitivity = max(0.01, float(sensitivity))
        self.fov_radius = int(fov_radius)
        self.init_scale = max(0.05, min(1.0, float(init_scale)))
        self.ramp_time = max(0.001, float(ramp_time))
        self.pred_weight_x = max(0.0, min(1.0, float(pred_weight_x)))
        self.pred_weight_y = max(0.0, min(1.0, float(pred_weight_y)))

        self._predictor = DerivativePredictor()
        self._last_raw_x = 0.0
        self._last_raw_y = 0.0
        self._last_output_x = 0.0
        self._last_output_y = 0.0
        self._last_time = None
        self._lock_start_time = None

    def update_params(self, **kwargs):
        kp = kwargs.get("kp")
        ki = kwargs.get("ki")
        kd = kwargs.get("kd")
        if kp is not None or ki is not None or kd is not None:
            if kp is not None:
                self._base_kp = float(kp)
            if ki is not None:
                self._base_ki = float(ki)
            if kd is not None:
                self._base_kd = float(kd)
            self._pid_x.configure(
                float(kp if kp is not None else self._base_kp),
                float(ki if ki is not None else self._base_ki),
                float(kd if kd is not None else self._base_kd),
            )
            self._pid_y.configure(
                float(kp if kp is not None else self._base_kp),
                float(ki if ki is not None else self._base_ki),
                float(kd if kd is not None else self._base_kd),
            )
        if kwargs.get("slew_limit") is not None:
            self.slew_limit = float(kwargs["slew_limit"])
        if kwargs.get("max_speed") is not None:
            self.max_speed = max(1.0, float(kwargs["max_speed"]))
        if kwargs.get("sensitivity") is not None:
            self.sensitivity = max(0.01, float(kwargs["sensitivity"]))
        if kwargs.get("fov_radius") is not None:
            self.fov_radius = int(kwargs["fov_radius"])
        if kwargs.get("init_scale") is not None:
            self.init_scale = max(0.05, min(1.0, float(kwargs["init_scale"])))
        if kwargs.get("ramp_time") is not None:
            self.ramp_time = max(0.001, float(kwargs["ramp_time"]))
        if kwargs.get("pred_weight_x") is not None:
            self.pred_weight_x = max(0.0, min(1.0, float(kwargs["pred_weight_x"])))
        if kwargs.get("pred_weight_y") is not None:
            self.pred_weight_y = max(0.0, min(1.0, float(kwargs["pred_weight_y"])))
        pw = kwargs.get("pred_weight")
        if pw is not None:
            pw = max(0.0, min(1.0, float(pw)))
            self.pred_weight_x = pw
            self.pred_weight_y = pw

    def reset(self):
        self._pid_x.reset()
        self._pid_y.reset()
        self._predictor.reset()
        self._last_raw_x = 0.0
        self._last_raw_y = 0.0
        self._last_output_x = 0.0
        self._last_output_y = 0.0
        self._last_time = None
        self._lock_start_time = None

    @staticmethod
    def _trunc_to_int(value):
        return int(value)

    def _smoothstep_ramp_scale(self, now):
        if self._lock_start_time is None:
            return self.init_scale
        elapsed = now - self._lock_start_time
        progress = _clamp(elapsed / self.ramp_time, 0.0, 1.0)
        ramp = progress * progress * (3.0 - 2.0 * progress)
        return self.init_scale + (1.0 - self.init_scale) * ramp

    def update(self, current_x, current_y, target_x, target_y):
        raw_x = float(target_x) - float(current_x)
        raw_y = float(target_y) - float(current_y)
        dist0 = math.hypot(raw_x, raw_y)
        if self.fov_radius > 0 and dist0 > float(self.fov_radius):
            self.reset()
            return 0.0, 0.0

        now = time.monotonic()
        if self._last_time is None:
            dt = 0.001
        else:
            dt = _clamp(now - self._last_time, 0.001, 0.05)
        self._last_time = now

        target_jump = math.hypot(raw_x - self._last_raw_x, raw_y - self._last_raw_y)
        if self._lock_start_time is None or target_jump > self._TARGET_JUMP_THRESHOLD:
            self._lock_start_time = now
            self._predictor.reset()
            self._pid_x.reset()
            self._pid_y.reset()

        pred_x, pred_y = self._predictor.predict(
            raw_x, raw_y, self._last_output_x, self._last_output_y, dt,
        )

        pred_limit_x = min(max(abs(raw_x) * 1.5, 30.0), 60.0)
        pred_limit_y = min(max(abs(raw_y) * 1.5, 30.0), 60.0)
        pred_x = _clamp(pred_x, -pred_limit_x, pred_limit_x)
        pred_y = _clamp(pred_y, -pred_limit_y, pred_limit_y)

        if abs(pred_x) > 100.0 or abs(pred_y) > 100.0:
            self._predictor.reset()
            pred_x = pred_y = 0.0

        fused_x = raw_x + pred_x * self.pred_weight_x
        fused_y = raw_y + pred_y * self.pred_weight_y

        scale = self._smoothstep_ramp_scale(now)
        out_x = self._pid_x.update(fused_x, scale)
        out_y = self._pid_y.update(fused_y, scale)

        cap = self.max_speed
        out_x = _clamp(out_x, -cap, cap)
        out_y = _clamp(out_y, -cap, cap)

        out_x *= self.sensitivity
        out_y *= self.sensitivity

        self._last_raw_x = raw_x
        self._last_raw_y = raw_y
        self._last_output_x = out_x
        self._last_output_y = out_y

        return float(self._trunc_to_int(out_x)), float(self._trunc_to_int(out_y))


@dataclass
class SingleDetectionTarget:
    bbox: List[float]
    confidence: float
    class_id: int
    aim_x: float
    aim_y: float
    track_id: int = -1


TrackedTarget = SingleDetectionTarget


class AimbotV6:
    def __init__(self, config: AimbotConfigV6):
        self.config = config
        self.tracker = self._new_tracker()
        self.controller = PIDController(
            kp=config.pid_kp,
            ki=config.pid_ki,
            kd=config.pid_kd,
            slew_limit=config.slew_limit,
            max_speed=config.max_speed,
            sensitivity=config.sensitivity,
            fov_radius=config.fov_radius,
            init_scale=config.init_scale,
            ramp_time=config.ramp_time,
            pred_weight_x=config.pred_weight_x,
            pred_weight_y=config.pred_weight_y,
        )
        self.kmbox: Optional[KmboxNet] = None
        self.active = False
        self.stats = {
            "frames_processed": 0,
            "detections_received": 0,
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
        self.controller.reset()
        self.reset_tracking()

    def deactivate(self):
        self.active = False
        self.controller.reset()
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
                timing_ms.setdefault("pid_ms", 0.0)
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        target_start = time.perf_counter()
        target = self.process_detection(detections, frame_shape, crop_offset)
        if timing_ms is not None:
            timing_ms["target_select_ms"] = (time.perf_counter() - target_start) * 1000.0
        self.stats["frames_processed"] += 1
        self.stats["detections_received"] += len(detections)
        if target is None:
            self.controller.reset()
            if timing_ms is not None:
                timing_ms.setdefault("pid_ms", 0.0)
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False
        return self.aim_at_target(target, timing_ms=timing_ms)

    def process_detection(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int],
        crop_offset: Tuple[int, int] = (0, 0),
    ) -> Optional[SingleDetectionTarget]:
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
            x1, y1, x2, y2, _center_x, _center_y = parsed
            objects.append(DetectionObject(
                bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                label=int(detection.get("class_id", 0)),
                prob=float(detection.get("confidence", 0.0)),
            ))
        return objects

    def _object_to_target(self, obj: DetectionObject, crop_offset: Tuple[int, int]) -> SingleDetectionTarget:
        center_x = obj.bbox.x + obj.bbox.width * 0.5
        center_y = obj.bbox.y + obj.bbox.height * 0.5
        absolute_x = crop_offset[0] + center_x
        absolute_y = crop_offset[1] + center_y
        aim_x = absolute_x - self.config.screen_width * 0.5 + self.config.aim_offset_x
        if self.config.aim_offset_dynamic:
            aim_y = absolute_y - self.config.screen_height * 0.5 + self.config.aim_offset_y * obj.bbox.height
        else:
            aim_y = absolute_y - self.config.screen_height * 0.5 + self.config.aim_offset_y * 100.0

        return SingleDetectionTarget(
            bbox=[obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height],
            confidence=obj.prob,
            class_id=obj.label,
            aim_x=aim_x,
            aim_y=aim_y,
            track_id=obj.track_id,
        )

    def _parse_bbox(
        self,
        bbox: List[float],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        if len(bbox) != 4:
            return None

        values = [float(item) for item in bbox]
        if all(0.0 <= value <= 1.0 for value in values):
            cx, cy, width, height = values
            x1 = (cx - width * 0.5) * frame_w
            y1 = (cy - height * 0.5) * frame_h
            x2 = (cx + width * 0.5) * frame_w
            y2 = (cy + height * 0.5) * frame_h
            center_x = cx * frame_w
            center_y = cy * frame_h
        else:
            x1, y1, x2, y2 = values
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
        return x1, y1, x2, y2, center_x, center_y

    def aim_at_target(self, target: SingleDetectionTarget, timing_ms: Optional[Dict[str, float]] = None) -> bool:
        if not self.active or self.kmbox is None:
            if timing_ms is not None:
                timing_ms.setdefault("pid_ms", 0.0)
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        pid_start = time.perf_counter()
        move_x, move_y = self.controller.update(0.0, 0.0, target.aim_x, target.aim_y)
        if timing_ms is not None:
            timing_ms["pid_ms"] = (time.perf_counter() - pid_start) * 1000.0
        out_x = int(move_x)
        out_y = int(move_y)
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
