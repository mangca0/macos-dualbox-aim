import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import AimbotConfigV2
from .kmbox import KmboxConfig, KmboxNet, SUCCESS


@dataclass
class Target:
    bbox: List[float]
    confidence: float
    class_id: int
    screen_x: float
    screen_y: float
    timestamp: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass
class KalmanState:
    x: float
    y: float
    vx: float
    vy: float


class KalmanFilter2D:
    def __init__(self, process_noise: float, measurement_noise: float, initial_covariance: float):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_covariance = float(initial_covariance)
        self.initialized = False
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_time = 0.0
        self.covariance = [
            [self.initial_covariance, 0.0, 0.0, 0.0],
            [0.0, self.initial_covariance, 0.0, 0.0],
            [0.0, 0.0, self.initial_covariance, 0.0],
            [0.0, 0.0, 0.0, self.initial_covariance],
        ]

    def update(self, measured_x: float, measured_y: float, now: float) -> KalmanState:
        if not self.initialized:
            self.initialized = True
            self.x = measured_x
            self.y = measured_y
            self.last_time = now
            return self.state

        dt = max(0.0, now - self.last_time)
        if dt > 0.0:
            self._predict(dt)
        self._correct(measured_x, measured_y)
        self.last_time = now
        return self.state

    @property
    def state(self) -> KalmanState:
        return KalmanState(self.x, self.y, self.vx, self.vy)

    def _predict(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt

        f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        predicted = self._mat_mul(self._mat_mul(f, self.covariance), self._transpose(f))
        q_pos = self.process_noise * dt * dt
        q_vel = self.process_noise * dt
        for idx, value in enumerate((q_pos, q_pos, q_vel, q_vel)):
            predicted[idx][idx] += value
        self.covariance = predicted

    def _correct(self, measured_x: float, measured_y: float):
        state = [self.x, self.y, self.vx, self.vy]
        for measurement_index, measurement in ((0, measured_x), (1, measured_y)):
            innovation = measurement - state[measurement_index]
            innovation_covariance = self.covariance[measurement_index][measurement_index] + self.measurement_noise
            if innovation_covariance <= 0.0:
                continue
            gain = [self.covariance[row][measurement_index] / innovation_covariance for row in range(4)]
            for row in range(4):
                state[row] += gain[row] * innovation

            old_covariance = [row[:] for row in self.covariance]
            for row in range(4):
                for col in range(4):
                    self.covariance[row][col] = old_covariance[row][col] - gain[row] * old_covariance[measurement_index][col]

        self.x, self.y, self.vx, self.vy = state

    def _mat_mul(self, left: List[List[float]], right: List[List[float]]) -> List[List[float]]:
        rows = len(left)
        cols = len(right[0])
        inner = len(right)
        return [
            [
                sum(left[row][idx] * right[idx][col] for idx in range(inner))
                for col in range(cols)
            ]
            for row in range(rows)
        ]

    def _transpose(self, matrix: List[List[float]]) -> List[List[float]]:
        return [list(row) for row in zip(*matrix)]


class PIDFControllerV2:
    def __init__(self, config: AimbotConfigV2):
        self.config = config
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_time = 0.0

    def reset(self):
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_time = 0.0

    def update(
        self,
        error_x: float,
        error_y: float,
        target_velocity_x: float,
        target_velocity_y: float,
        now: float,
    ) -> Tuple[float, float]:
        if self.last_time > 0.0:
            dt = max(0.0, now - self.last_time)
        else:
            dt = 0.0

        if dt > 0.0:
            derivative_x = (error_x - self.last_error_x) / dt
            derivative_y = (error_y - self.last_error_y) / dt
            self.integral_x += error_x * dt
            self.integral_y += error_y * dt
        else:
            derivative_x = 0.0
            derivative_y = 0.0

        feed_x = target_velocity_x
        feed_y = target_velocity_y

        move_x = (
            self.config.pid_kp * error_x +
            self.config.pid_ki * self.integral_x +
            self.config.pid_kd * derivative_x +
            self.config.pid_kf * feed_x
        )
        move_y = (
            self.config.pid_kp * error_y +
            self.config.pid_ki * self.integral_y +
            self.config.pid_kd * derivative_y +
            self.config.pid_kf * feed_y
        )

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_time = now
        return move_x, move_y


class AimbotV2:
    def __init__(self, config: AimbotConfigV2):
        self.config = config
        self.pidf = PIDFControllerV2(config)
        self.kmbox: Optional[KmboxNet] = None
        self.active = False
        self.previous_targets: List[Target] = []
        self.kalman_tracks: List[KalmanFilter2D] = []
        self.stats = {
            "frames_processed": 0,
            "targets_detected": 0,
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
        self.pidf.reset()
        self.previous_targets.clear()
        self.kalman_tracks.clear()

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
        target = self.process_detections(detections, frame_shape, crop_offset)
        if timing_ms is not None:
            timing_ms["target_select_ms"] = (time.perf_counter() - target_start) * 1000.0
        self.stats["frames_processed"] += 1
        if target is None:
            self.pidf.reset()
            self.kalman_tracks.clear()
            if timing_ms is not None:
                timing_ms.setdefault("pid_ms", 0.0)
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
        targets: List[Target] = []
        now = time.time()

        for detection in detections:
            class_id = int(detection.get("class_id", 0))
            if self.config.target_classes and class_id not in self.config.target_classes:
                continue

            parsed = self._parse_bbox(detection.get("bbox", []), frame_w, frame_h)
            if parsed is None:
                continue
            x1, y1, x2, y2, center_x, center_y = parsed

            absolute_x = crop_offset[0] + center_x
            absolute_y = crop_offset[1] + center_y
            screen_x = absolute_x - self.config.screen_width * 0.5
            screen_y = absolute_y - self.config.screen_height * 0.5

            targets.append(Target(
                bbox=[x1, y1, x2, y2],
                confidence=float(detection.get("confidence", 0.0)),
                class_id=class_id,
                screen_x=screen_x,
                screen_y=screen_y,
                timestamp=now,
            ))

        self._update_target_velocity(targets)
        if self.config.enable_kalman_filter:
            self._update_kalman_tracks(targets)
        self.stats["targets_detected"] += len(targets)
        if not targets:
            return None
        return min(targets, key=self._target_rank)

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

    def _update_target_velocity(self, targets: List[Target]):
        if not self.previous_targets:
            self.previous_targets = [self._copy_target(target) for target in targets]
            return

        used_indices = set()
        for target in targets:
            best_idx = -1
            best_dist_sq = float("inf")
            for idx, previous in enumerate(self.previous_targets):
                if idx in used_indices or previous.class_id != target.class_id:
                    continue
                dt = target.timestamp - previous.timestamp
                if dt <= 0.0:
                    continue
                dist_sq = (
                    (target.screen_x - previous.screen_x) ** 2 +
                    (target.screen_y - previous.screen_y) ** 2
                )
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_idx = idx

            if best_idx < 0:
                continue
            used_indices.add(best_idx)
            previous = self.previous_targets[best_idx]
            dt = target.timestamp - previous.timestamp
            target.velocity_x = (target.screen_x - previous.screen_x) / dt
            target.velocity_y = (target.screen_y - previous.screen_y) / dt

        self.previous_targets = [self._copy_target(target) for target in targets]

    def _update_kalman_tracks(self, targets: List[Target]):
        if not self.kalman_tracks:
            self.kalman_tracks = [self._new_kalman_track() for _target in targets]
            for target, track in zip(targets, self.kalman_tracks):
                state = track.update(target.screen_x, target.screen_y, target.timestamp)
                self._apply_kalman_state(target, state)
            return

        used_indices = set()
        next_tracks: List[KalmanFilter2D] = []
        for target in targets:
            best_idx = -1
            best_dist_sq = float("inf")
            for idx, track in enumerate(self.kalman_tracks):
                if idx in used_indices:
                    continue
                state = track.state
                dist_sq = (
                    (target.screen_x - state.x) ** 2 +
                    (target.screen_y - state.y) ** 2
                )
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_idx = idx

            if best_idx < 0:
                track = self._new_kalman_track()
            else:
                used_indices.add(best_idx)
                track = self.kalman_tracks[best_idx]
            state = track.update(target.screen_x, target.screen_y, target.timestamp)
            self._apply_kalman_state(target, state)
            next_tracks.append(track)

        self.kalman_tracks = next_tracks

    def _new_kalman_track(self) -> KalmanFilter2D:
        return KalmanFilter2D(
            process_noise=self.config.kalman_process_noise,
            measurement_noise=self.config.kalman_measurement_noise,
            initial_covariance=self.config.kalman_initial_covariance,
        )

    def _apply_kalman_state(self, target: Target, state: KalmanState):
        target.screen_x = state.x
        target.screen_y = state.y
        target.velocity_x = state.vx
        target.velocity_y = state.vy

    def _copy_target(self, target: Target) -> Target:
        return Target(
            bbox=list(target.bbox),
            confidence=target.confidence,
            class_id=target.class_id,
            screen_x=target.screen_x,
            screen_y=target.screen_y,
            timestamp=target.timestamp,
            velocity_x=target.velocity_x,
            velocity_y=target.velocity_y,
        )

    def _target_rank(self, target: Target) -> float:
        distance = math.sqrt(target.screen_x * target.screen_x + target.screen_y * target.screen_y)
        return distance / max(0.01, self._class_weight(target.class_id))

    def _class_weight(self, class_id: int) -> float:
        return float(self.config.class_priority_weights.get(class_id, 1.0))

    def aim_at_target(self, target: Target, timing_ms: Optional[Dict[str, float]] = None) -> bool:
        if not self.active or self.kmbox is None:
            if timing_ms is not None:
                timing_ms.setdefault("pid_ms", 0.0)
                timing_ms.setdefault("kmbox_send_ack_ms", 0.0)
            return False

        aim_x = target.screen_x + self.config.aim_offset_x
        if self.config.aim_offset_dynamic:
            bbox_height = target.bbox[3] - target.bbox[1]
            aim_y = target.screen_y + self.config.aim_offset_y * bbox_height
        else:
            aim_y = target.screen_y + self.config.aim_offset_y * 100.0

        pid_start = time.perf_counter()
        move_x, move_y = self.pidf.update(
            aim_x,
            aim_y,
            target.velocity_x,
            target.velocity_y,
            time.time(),
        )
        if timing_ms is not None:
            timing_ms["pid_ms"] = (time.perf_counter() - pid_start) * 1000.0
        out_x = int(round(move_x))
        out_y = int(round(move_y))
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
