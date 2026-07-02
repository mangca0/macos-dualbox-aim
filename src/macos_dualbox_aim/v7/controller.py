import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import AimbotConfigV7
from .crosshair import CrosshairDetector
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
        self.integral_gate_enabled = False
        self.integral_gate_threshold = 50.0
        self.integral_gate_rate = 0.025
        self.integral_gain = 0.0
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

    def configure_integral_gate(self, enabled=None, threshold=None, rate=None):
        if enabled is not None:
            self.integral_gate_enabled = bool(enabled)
        if threshold is not None:
            self.integral_gate_threshold = max(0.001, float(threshold))
        if rate is not None:
            self.integral_gate_rate = _clamp(float(rate), 0.0, 1.0)

    def reset(self, output=0.0):
        self.output = float(output)
        self.previous_output = float(output)
        self.previous_error = 0.0
        self.previous_previous_error = 0.0
        self.integral_gain = 0.0

    def set_output_limits(self, min_val, max_val):
        self.output_min = float(min_val)
        self.output_max = float(max_val)
        self.output = _clamp(self.output, self.output_min, self.output_max)

    def update(self, error, scale=1.0):
        self.previous_output = self.output
        if abs(error) < 0.3:
            error = 0.0
        integral_gain = self._update_integral_gain(error)
        delta = (
            self.kp * (error - self.previous_error)
            + self.ki * error * integral_gain
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

    def _update_integral_gain(self, error):
        if not self.integral_gate_enabled:
            self.integral_gain = 1.0
            return 1.0

        abs_error = abs(error)
        if abs_error < self.integral_gate_threshold:
            target_gain = 1.0 - (abs_error / self.integral_gate_threshold)
            self.integral_gain += (target_gain - self.integral_gain) * self.integral_gate_rate
        else:
            self.integral_gain += (0.0 - self.integral_gain) * 0.1
        self.integral_gain = _clamp(self.integral_gain, 0.0, 1.0)
        return self.integral_gain


class PerlinNoise1D:
    """One-dimensional smooth value noise used by the learning controller."""

    TABLE_SIZE = 256

    def __init__(self, seed=0):
        rng = random.Random(int(seed))
        self.values = [rng.random() for _ in range(self.TABLE_SIZE)]

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def noise(self, x):
        floor_x = math.floor(float(x))
        xi = int(floor_x) & 255
        xf = float(x) - floor_x
        u = self._fade(xf)
        a = self.values[xi]
        b = self.values[(xi + 1) & 255]
        return self._lerp(a, b, u) * 2.0 - 1.0


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


@dataclass
class AimOutput:
    move_x: float
    move_y: float
    curve_len: float
    predicted_x: float
    predicted_y: float
    fused_x: float
    fused_y: float


class AimController:
    """Strict Python replica of the learning project's aim controller."""

    def __init__(self):
        self.pid_x = IncrementalPid()
        self.pid_y = IncrementalPid()
        self.last_raw_x = 0.0
        self.last_raw_y = 0.0
        self.last_output_x = 0.0
        self.last_output_y = 0.0
        self.last_time = time.monotonic()
        self.lock_start_time = time.monotonic()
        self.noise_x = PerlinNoise1D(12345)
        self.noise_y = PerlinNoise1D(54321)
        self.noise_time_x = 0.0
        self.noise_time_y = 100.0
        self.predictor = DerivativePredictor()
        self.latest_output = AimOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.slew_limit = 40.0
        self.output_max = 30.0
        self.sensitivity = 1.0
        self.fov_radius = 256
        self.init_scale = 0.6
        self.ramp_time = 0.5
        self.pred_weight_x = 0.5
        self.pred_weight_y = 0.5
        self.target_jump_reset = 40.0
        self.noise_amp = 0.0

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.last_raw_x = 0.0
        self.last_raw_y = 0.0
        self.last_output_x = 0.0
        self.last_output_y = 0.0
        self.last_time = time.monotonic()
        self.lock_start_time = time.monotonic()
        self.noise_time_x = 0.0
        self.noise_time_y = 100.0
        self.predictor.reset()
        self.latest_output = AimOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def configure_pid(self, kp, ki, kd):
        self.pid_x.configure(kp, ki, kd)
        self.pid_y.configure(kp, ki, kd)

    def configure_pid_x(self, kp, ki, kd):
        self.pid_x.configure(kp, ki, kd)

    def configure_pid_y(self, kp, ki, kd):
        self.pid_y.configure(kp, ki, kd)

    def set_output_limits(self, min_val, max_val):
        self.pid_x.set_output_limits(min_val, max_val)
        self.pid_y.set_output_limits(min_val, max_val)

    def configure_integral_gate(self, enabled=None, threshold=None, rate=None):
        self.pid_x.configure_integral_gate(enabled=enabled, threshold=threshold, rate=rate)
        self.pid_y.configure_integral_gate(enabled=enabled, threshold=threshold, rate=rate)

    def update_params(self, **kwargs):
        kp = kwargs.get("kp")
        ki = kwargs.get("ki")
        kd = kwargs.get("kd")
        if kp is not None or ki is not None or kd is not None:
            self.pid_x.configure(
                self.pid_x.kp if kp is None else float(kp),
                self.pid_x.ki if ki is None else float(ki),
                self.pid_x.kd if kd is None else float(kd),
            )
            self.pid_y.configure(
                self.pid_y.kp if kp is None else float(kp),
                self.pid_y.ki if ki is None else float(ki),
                self.pid_y.kd if kd is None else float(kd),
            )
        if kwargs.get("slew_limit") is not None:
            self.slew_limit = float(kwargs["slew_limit"])
        if kwargs.get("max_speed") is not None and kwargs.get("output_max") is None:
            self.output_max = max(1.0, float(kwargs["max_speed"]))
        if kwargs.get("output_max") is not None:
            self.output_max = max(1.0, float(kwargs["output_max"]))
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
        if kwargs.get("target_jump_reset") is not None:
            self.target_jump_reset = max(0.0, float(kwargs["target_jump_reset"]))
        if (
            kwargs.get("pid_integral_gate_enabled") is not None
            or kwargs.get("pid_integral_gate_threshold") is not None
            or kwargs.get("pid_integral_gate_rate") is not None
        ):
            self.configure_integral_gate(
                enabled=kwargs.get("pid_integral_gate_enabled"),
                threshold=kwargs.get("pid_integral_gate_threshold"),
                rate=kwargs.get("pid_integral_gate_rate"),
            )
            self.reset()
        pw = kwargs.get("pred_weight")
        if pw is not None:
            pw = max(0.0, min(1.0, float(pw)))
            self.pred_weight_x = pw
            self.pred_weight_y = pw
        if kwargs.get("noise_amp") is not None:
            self.noise_amp = max(0.0, float(kwargs["noise_amp"]))

    def update_runtime(self, current_x, current_y, target_x, target_y):
        raw_x = float(target_x) - float(current_x)
        raw_y = float(target_y) - float(current_y)
        dist0 = math.hypot(raw_x, raw_y)
        if self.fov_radius > 0 and dist0 > float(self.fov_radius):
            self.reset()
            return 0.0, 0.0

        output = self.update(
            raw_x,
            raw_y,
            self.pred_weight_x,
            self.pred_weight_y,
            self.init_scale,
            self.ramp_time,
            self.output_max,
            self.noise_amp,
        )
        out_x = output.move_x * self.sensitivity
        out_y = output.move_y * self.sensitivity
        self.latest_output = AimOutput(
            out_x,
            out_y,
            output.curve_len,
            output.predicted_x,
            output.predicted_y,
            output.fused_x,
            output.fused_y,
        )
        return float(int(out_x)), float(int(out_y))

    def update(
        self,
        raw_x,
        raw_y,
        pred_weight_x,
        pred_weight_y,
        init_scale,
        ramp_time,
        output_max,
        noise_amp,
    ):
        now = time.monotonic()
        dt = _clamp(now - self.last_time, 0.001, 0.05)
        self.last_time = now

        target_jump = math.hypot(raw_x - self.last_raw_x, raw_y - self.last_raw_y)
        if self.target_jump_reset > 0.0 and target_jump > self.target_jump_reset:
            self.lock_start_time = now
            self.predictor.reset()
            self.pid_x.reset()
            self.pid_y.reset()

        pred_x, pred_y = self.predictor.predict(
            raw_x, raw_y, self.last_output_x, self.last_output_y, dt,
        )

        pred_limit_x = min(max(abs(raw_x) * 1.5, 30.0), 60.0)
        pred_limit_y = min(max(abs(raw_y) * 1.5, 30.0), 60.0)
        pred_x = _clamp(pred_x, -pred_limit_x, pred_limit_x)
        pred_y = _clamp(pred_y, -pred_limit_y, pred_limit_y)

        if abs(pred_x) > 100.0 or abs(pred_y) > 100.0:
            self.predictor.reset()
            pred_x = pred_y = 0.0

        fused_x = raw_x + pred_x * float(pred_weight_x)
        fused_y = raw_y + pred_y * float(pred_weight_y)

        elapsed = now - self.lock_start_time
        progress = _clamp(elapsed / max(float(ramp_time), 0.001), 0.0, 1.0)
        ramp = progress * progress * (3.0 - 2.0 * progress)
        scale = float(init_scale) + (1.0 - float(init_scale)) * ramp

        out_x = self.pid_x.update(fused_x, scale)
        out_y = self.pid_y.update(fused_y, scale)

        self.noise_time_x += dt * 5.0
        self.noise_time_y += dt * 5.0
        out_x += self.noise_x.noise(self.noise_time_x) * float(noise_amp)
        out_y += self.noise_y.noise(self.noise_time_y) * float(noise_amp)

        out_x = _clamp(out_x, -float(output_max), float(output_max))
        out_y = _clamp(out_y, -float(output_max), float(output_max))

        self.last_raw_x = raw_x
        self.last_raw_y = raw_y
        self.last_output_x = out_x
        self.last_output_y = out_y

        self.latest_output = AimOutput(
            out_x,
            out_y,
            math.hypot(fused_x, fused_y),
            pred_x,
            pred_y,
            fused_x,
            fused_y,
        )
        return self.latest_output


@dataclass
class SingleDetectionTarget:
    bbox: List[float]
    confidence: float
    class_id: int
    aim_x: float
    aim_y: float
    track_id: int = -1
    crosshair_x: float = 0.0
    crosshair_y: float = 0.0


TrackedTarget = SingleDetectionTarget


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class AimbotV7:
    def __init__(self, config: AimbotConfigV7):
        self.config = config
        self.selected_class_ids = set(config.selected_class_ids or [])
        self.tracker = self._new_tracker()
        self.crosshair_detector = CrosshairDetector(config)
        self.controller = AimController()
        self.controller.update_params(
            kp=config.pid_kp,
            ki=config.pid_ki,
            kd=config.pid_kd,
            slew_limit=config.slew_limit,
            output_max=config.output_max,
            max_speed=config.max_speed,
            sensitivity=config.sensitivity,
            fov_radius=config.fov_radius,
            init_scale=config.init_scale,
            ramp_time=config.ramp_time,
            pred_weight_x=config.pred_weight_x,
            pred_weight_y=config.pred_weight_y,
            target_jump_reset=config.target_jump_reset,
            pid_integral_gate_enabled=config.pid_integral_gate_enabled,
            pid_integral_gate_threshold=config.pid_integral_gate_threshold,
            pid_integral_gate_rate=config.pid_integral_gate_rate,
            noise_amp=config.noise_amp,
        )
        self.latest_aim_output = self.controller.latest_output
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
        self._aim_samples = deque(maxlen=2400)

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

    def reset_aim_metrics(self):
        self._aim_samples.clear()

    def get_aim_metrics_snapshot(self) -> Dict[str, Any]:
        samples = list(self._aim_samples)
        found = [sample for sample in samples if sample["target_found"]]
        lost_count = len(samples) - len(found)
        distances = [sample["distance"] for sample in found]
        signed_x_errors = [sample["aim_x"] for sample in found]
        x_errors = [abs(sample["aim_x"]) for sample in found]
        y_errors = [abs(sample["aim_y"]) for sample in found]
        moves = [sample["move_magnitude"] for sample in found]
        latest = found[-1] if found else {}
        settled_radius = 2.0
        settled_count = sum(1 for value in distances if value <= settled_radius)
        x_dwell_radius_1 = 1.0
        x_dwell_radius_2 = 2.0
        x_dwell_1 = sum(1 for value in x_errors if value <= x_dwell_radius_1)
        x_dwell_2 = sum(1 for value in x_errors if value <= x_dwell_radius_2)

        return {
            "available": bool(samples),
            "samples": len(samples),
            "target_found_samples": len(found),
            "target_lost_samples": lost_count,
            "target_lost_ratio": lost_count / len(samples) if samples else 0.0,
            "settled_radius": settled_radius,
            "settled_ratio": settled_count / len(found) if found else 0.0,
            "mean_abs_error": _mean(distances),
            "p95_abs_error": _percentile(distances, 95.0),
            "p99_abs_error": _percentile(distances, 99.0),
            "max_abs_error": max(distances) if distances else 0.0,
            "mean_signed_x_error": _mean(signed_x_errors),
            "mean_abs_x_error": _mean(x_errors),
            "p95_abs_x_error": _percentile(x_errors, 95.0),
            "p99_abs_x_error": _percentile(x_errors, 99.0),
            "max_abs_x_error": max(x_errors) if x_errors else 0.0,
            "x_center_dwell_radius_1": x_dwell_radius_1,
            "x_center_dwell_ratio_1px": x_dwell_1 / len(found) if found else 0.0,
            "x_center_dwell_radius_2": x_dwell_radius_2,
            "x_center_dwell_ratio_2px": x_dwell_2 / len(found) if found else 0.0,
            "x_crossing_count": self._x_crossing_count(found),
            "time_to_x_settle_ms": self._time_to_x_settle_ms(found, x_dwell_radius_2),
            "mean_abs_y_error": _mean(y_errors),
            "p95_abs_y_error": _percentile(y_errors, 95.0),
            "max_abs_y_error": max(y_errors) if y_errors else 0.0,
            "mean_move": _mean(moves),
            "p95_move": _percentile(moves, 95.0),
            "overshoot_count": self._overshoot_count(found),
            "oscillation_energy": self._oscillation_energy(found),
            "latest": self._public_aim_sample(latest),
        }

    def update_selected_classes(self, class_ids: List[int]):
        self.selected_class_ids = {int(value) for value in class_ids}
        self.reset_tracking()

    def update(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int],
        crop_offset: Tuple[int, int] = (0, 0),
        timing_ms: Optional[Dict[str, float]] = None,
        frame: Optional[Any] = None,
    ) -> bool:
        if not self.active:
            if timing_ms is not None:
                timing_ms.setdefault("target_select_ms", 0.0)
                timing_ms.setdefault("pid_ms", 0.0)
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        target_start = time.perf_counter()
        target = self.process_detection(detections, frame_shape, crop_offset, frame=frame)
        if timing_ms is not None:
            timing_ms["target_select_ms"] = (time.perf_counter() - target_start) * 1000.0
        self.stats["frames_processed"] += 1
        self.stats["detections_received"] += len(detections)
        if target is None:
            self.controller.reset()
            self._record_aim_sample(target_found=False)
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
        frame: Optional[Any] = None,
    ) -> Optional[SingleDetectionTarget]:
        crosshair = self.crosshair_detector.detect(frame)
        if not crosshair.found:
            return None

        frame_h, frame_w = frame_shape[:2]
        objects = self._detections_to_objects(detections, frame_w, frame_h)
        self.stats["detections_input"] += len(objects)

        tracked = self.tracker.predict(objects)
        self.stats["tracks_output"] += len(tracked)
        if not tracked:
            return None

        crosshair_absolute = (
            crop_offset[0] + crosshair.crosshair_x,
            crop_offset[1] + crosshair.crosshair_y,
        )
        return self._object_to_target(tracked[0], crop_offset, crosshair_absolute)

    def _detections_to_objects(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
    ) -> List[DetectionObject]:
        objects: List[DetectionObject] = []
        for detection in detections:
            class_id = int(detection.get("class_id", 0))
            if class_id not in self.selected_class_ids:
                continue
            parsed = self._parse_bbox(detection.get("bbox", []), frame_w, frame_h)
            if parsed is None:
                continue
            x1, y1, x2, y2, _center_x, _center_y = parsed
            objects.append(DetectionObject(
                bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                label=class_id,
                prob=float(detection.get("confidence", 0.0)),
            ))
        return objects

    def _object_to_target(
        self,
        obj: DetectionObject,
        crop_offset: Tuple[int, int],
        crosshair_absolute: Tuple[float, float],
    ) -> SingleDetectionTarget:
        center_x = obj.bbox.x + obj.bbox.width * 0.5
        center_y = obj.bbox.y + obj.bbox.height * 0.5
        absolute_x = crop_offset[0] + center_x
        absolute_y = crop_offset[1] + center_y
        aim_x = absolute_x - crosshair_absolute[0] + self.config.aim_offset_x
        if self.config.aim_offset_dynamic:
            aim_y = absolute_y - crosshair_absolute[1] + self.config.aim_offset_y * obj.bbox.height
        else:
            aim_y = absolute_y - crosshair_absolute[1] + self.config.aim_offset_y * 100.0

        return SingleDetectionTarget(
            bbox=[obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height],
            confidence=obj.prob,
            class_id=obj.label,
            aim_x=aim_x,
            aim_y=aim_y,
            track_id=obj.track_id,
            crosshair_x=crosshair_absolute[0],
            crosshair_y=crosshair_absolute[1],
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
        move_x, move_y = self.controller.update_runtime(0.0, 0.0, target.aim_x, target.aim_y)
        self.latest_aim_output = self.controller.latest_output
        if timing_ms is not None:
            timing_ms["pid_ms"] = (time.perf_counter() - pid_start) * 1000.0
        out_x = int(move_x)
        out_y = int(move_y)
        if out_x == 0 and out_y == 0:
            self._record_aim_sample(target=target, move_x=out_x, move_y=out_y, sent=False)
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
            self._record_aim_sample(target=target, move_x=out_x, move_y=out_y, sent=True)
            return True
        self._record_aim_sample(target=target, move_x=out_x, move_y=out_y, sent=False)
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

    def _record_aim_sample(
        self,
        *,
        target_found: bool = True,
        target: Optional[SingleDetectionTarget] = None,
        move_x: float = 0.0,
        move_y: float = 0.0,
        sent: bool = False,
    ):
        aim_x = float(target.aim_x) if target is not None else 0.0
        aim_y = float(target.aim_y) if target is not None else 0.0
        self._aim_samples.append({
            "timestamp": time.monotonic(),
            "target_found": bool(target_found and target is not None),
            "aim_x": aim_x,
            "aim_y": aim_y,
            "distance": math.hypot(aim_x, aim_y) if target is not None else 0.0,
            "move_x": float(move_x),
            "move_y": float(move_y),
            "move_magnitude": math.hypot(float(move_x), float(move_y)),
            "track_id": int(getattr(target, "track_id", -1)) if target is not None else -1,
            "sent": bool(sent),
        })

    def _overshoot_count(self, samples: List[Dict[str, Any]]) -> int:
        count = 0
        for previous, current in zip(samples, samples[1:]):
            if self._sign_changed(previous["aim_x"], current["aim_x"]):
                count += 1
            if self._sign_changed(previous["aim_y"], current["aim_y"]):
                count += 1
        return count

    def _oscillation_energy(self, samples: List[Dict[str, Any]]) -> float:
        deltas = [
            math.hypot(current["aim_x"] - previous["aim_x"], current["aim_y"] - previous["aim_y"])
            for previous, current in zip(samples, samples[1:])
        ]
        return _mean(deltas)

    def _x_crossing_count(self, samples: List[Dict[str, Any]]) -> int:
        return sum(
            1
            for previous, current in zip(samples, samples[1:])
            if self._sign_changed(previous["aim_x"], current["aim_x"])
        )

    def _time_to_x_settle_ms(self, samples: List[Dict[str, Any]], radius: float) -> float:
        if not samples:
            return 0.0
        start = float(samples[0]["timestamp"])
        for sample in samples:
            if abs(float(sample["aim_x"])) <= radius:
                return max(0.0, (float(sample["timestamp"]) - start) * 1000.0)
        return 0.0

    def _public_aim_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not sample:
            return {}
        return {
            "target_found": bool(sample["target_found"]),
            "aim_x": float(sample["aim_x"]),
            "aim_y": float(sample["aim_y"]),
            "distance": float(sample["distance"]),
            "move_x": float(sample["move_x"]),
            "move_y": float(sample["move_y"]),
            "track_id": int(sample["track_id"]),
            "sent": bool(sample["sent"]),
        }

    @staticmethod
    def _sign_changed(previous: float, current: float) -> bool:
        threshold = 1.0
        return abs(previous) > threshold and abs(current) > threshold and previous * current < 0.0
