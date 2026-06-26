from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Sequence


@dataclass
class BoundingBox:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass
class DetectionObject:
    bbox: BoundingBox
    label: int = 0
    prob: float = 0.0
    track_id: int = -1


def _meas_from_object(obj: DetectionObject) -> List[float]:
    cx = obj.bbox.x + obj.bbox.width * 0.5
    cy = obj.bbox.y + obj.bbox.height * 0.5
    return [float(obj.label), cx, cy, obj.bbox.width, obj.bbox.height]


def _meas_to_bbox(z: Sequence[float]) -> BoundingBox:
    cx, cy, width, height = z[1], z[2], z[3], z[4]
    return BoundingBox(
        x=cx - width * 0.5,
        y=cy - height * 0.5,
        width=width,
        height=height,
    )


def _iou_from_meas(a: Sequence[float], b: Sequence[float]) -> float:
    abox = _meas_to_bbox(a)
    bbox = _meas_to_bbox(b)
    ax0, ay0 = abox.x, abox.y
    ax1, ay1 = abox.x + abox.width, abox.y + abox.height
    bx0, by0 = bbox.x, bbox.y
    bx1, by1 = bbox.x + bbox.width, bbox.y + bbox.height

    x_min = max(ax0, bx0)
    x_max = min(ax1, bx1)
    y_min = max(ay0, by0)
    y_max = min(ay1, by1)

    inter_w = max(x_max - x_min + 1.0, 0.0)
    inter_h = max(y_max - y_min + 1.0, 0.0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _invert_5x5(matrix: Sequence[Sequence[float]]) -> List[List[float]] | None:
    work = [
        [float(matrix[row][col]) for col in range(5)] +
        [1.0 if row == col else 0.0 for col in range(5)]
        for row in range(5)
    ]

    for col in range(5):
        pivot = col
        best = abs(work[pivot][col])
        for row in range(col + 1, 5):
            value = abs(work[row][col])
            if value > best:
                best = value
                pivot = row
        if best < 1e-12:
            return None
        if pivot != col:
            work[pivot], work[col] = work[col], work[pivot]

        divisor = work[col][col]
        for idx in range(10):
            work[col][idx] /= divisor
        for row in range(5):
            if row == col:
                continue
            factor = work[row][col]
            for idx in range(10):
                work[row][idx] -= factor * work[col][idx]

    return [[work[row][col + 5] for col in range(5)] for row in range(5)]


def hungarian_min(cost: Sequence[Sequence[float]]) -> List[int]:
    n = len(cost)
    m = len(cost[0]) if n else 0
    dim = max(n, m)
    max_dim = 32
    inf = 1e18

    if dim == 0:
        return []
    if dim > max_dim:
        return [-1 for _ in range(n)]

    a = [[0.0 for _ in range(max_dim + 1)] for _ in range(max_dim + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a[i][j] = float(cost[i - 1][j - 1])
        for j in range(m + 1, dim + 1):
            a[i][j] = 1.0

    u = [0.0 for _ in range(max_dim + 1)]
    v = [0.0 for _ in range(max_dim + 1)]
    p = [0 for _ in range(max_dim + 1)]
    way = [0 for _ in range(max_dim + 1)]

    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = [inf for _ in range(max_dim + 1)]
        used = [False for _ in range(max_dim + 1)]
        while True:
            used[j0] = True
            i0 = p[j0]
            j1 = 0
            delta = inf
            for j in range(1, dim + 1):
                if used[j]:
                    continue
                cur = a[i0][j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1 for _ in range(n)]
    for j in range(1, m + 1):
        if 1 <= p[j] <= n:
            assignment[p[j] - 1] = j - 1
    return assignment


class KalmanSimple:
    _next_id: ClassVar[int] = 0

    def __init__(
        self,
        terminate_set: int,
        generate_threshold: int,
        vx_noise: float,
        vy_noise: float,
        w_noise: float,
        h_noise: float,
        r_std: float,
    ):
        self.terminate_count = int(terminate_set)
        self.terminate_init = int(terminate_set)
        self.generate_threshold = int(generate_threshold)
        self.hit_streak = 1
        self.id = KalmanSimple._next_id
        KalmanSimple._next_id += 1
        self.has_last_z = False
        self.last_prob = 0.0
        self.vx_noise = float(vx_noise)
        self.vy_noise = float(vy_noise)
        self.w_noise = float(w_noise)
        self.h_noise = float(h_noise)
        self.r_var = float(r_std) * float(r_std)
        self.x_post = [0.0 for _ in range(7)]
        self.x_pri = [0.0 for _ in range(7)]
        self.p_post = [[1.0 if row == col else 0.0 for col in range(7)] for row in range(7)]
        self.p_pri = [[0.0 for _ in range(7)] for _ in range(7)]
        self.last_z6 = [0.0 for _ in range(6)]

    @classmethod
    def reset_next_id(cls, value: int = 0):
        cls._next_id = int(value)

    def set_prob(self, prob: float):
        self.last_prob = float(prob)

    def is_confirmed(self) -> bool:
        return self.hit_streak >= self.generate_threshold

    def init_from_meas(self, z: Sequence[float]):
        for idx in range(5):
            self.x_post[idx] = float(z[idx])
        self.x_post[5] = 0.0
        self.x_post[6] = 0.0

    def predict(self, dt: float = 1.0):
        a = [[0.0 for _ in range(7)] for _ in range(7)]
        for idx in range(7):
            a[idx][idx] = 1.0
        a[1][5] = dt
        a[2][6] = dt

        for row in range(7):
            self.x_pri[row] = sum(a[row][col] * self.x_post[col] for col in range(7))

        ap = [[0.0 for _ in range(7)] for _ in range(7)]
        for row in range(7):
            for col in range(7):
                ap[row][col] = sum(a[row][idx] * self.p_post[idx][col] for idx in range(7))

        q_diag = [0.0, 0.0, 0.0, self.w_noise, self.h_noise, self.vx_noise, self.vy_noise]
        for row in range(7):
            for col in range(7):
                value = sum(ap[row][idx] * a[col][idx] for idx in range(7))
                self.p_pri[row][col] = value + (q_diag[row] if row == col else 0.0)

    def update(self, z: Sequence[float]) -> bool:
        self.hit_streak += 1
        innovation = [float(z[idx]) - self.x_pri[idx] for idx in range(5)]

        s = [[0.0 for _ in range(5)] for _ in range(5)]
        for row in range(5):
            for col in range(5):
                s[row][col] = self.p_pri[row][col] + (self.r_var if row == col else 0.0)

        s_inv = _invert_5x5(s)
        if s_inv is None:
            return self.update_unmatched()

        k_gain = [[0.0 for _ in range(5)] for _ in range(7)]
        for row in range(7):
            for col in range(5):
                k_gain[row][col] = sum(self.p_pri[row][idx] * s_inv[idx][col] for idx in range(5))

        for row in range(7):
            delta = sum(k_gain[row][col] * innovation[col] for col in range(5))
            self.x_post[row] = self.x_pri[row] + delta

        hp = [[self.p_pri[row][col] for col in range(7)] for row in range(5)]
        khp = [[0.0 for _ in range(7)] for _ in range(7)]
        for row in range(7):
            for col in range(7):
                khp[row][col] = sum(k_gain[row][idx] * hp[idx][col] for idx in range(5))

        for row in range(7):
            for col in range(7):
                self.p_post[row][col] = self.p_pri[row][col] - khp[row][col]

        for idx in range(5):
            self.last_z6[idx] = self.x_post[idx]
        self.last_z6[5] = float(self.id)
        self.has_last_z = True
        self.terminate_count = self.terminate_init
        return True

    def update_unmatched(self) -> bool:
        if self.terminate_count == 1:
            return False
        self.terminate_count -= 1
        for idx in range(7):
            self.x_post[idx] = self.x_pri[idx]
        for row in range(7):
            for col in range(7):
                self.p_post[row][col] = self.p_pri[row][col]
        for idx in range(5):
            self.last_z6[idx] = self.x_post[idx]
        self.last_z6[5] = float(self.id)
        self.has_last_z = True
        return True


class KalmanP:
    def __init__(
        self,
        generate: int = 2,
        terminate: int = 5,
        vx_noise: float = 1.0,
        vy_noise: float = 1.0,
        w_noise: float = 0.01,
        h_noise: float = 0.01,
        r_std: float = 5.0,
    ):
        self.init(generate, terminate, vx_noise, vy_noise, w_noise, h_noise, r_std)
        self.tracks: List[KalmanSimple] = []

    def init(
        self,
        generate: int = 2,
        terminate: int = 5,
        vx_noise: float = 1.0,
        vy_noise: float = 1.0,
        w_noise: float = 0.01,
        h_noise: float = 0.01,
        r_std: float = 5.0,
    ):
        self.generate_set = int(generate)
        self.terminate_set = int(terminate)
        self.r_std = float(r_std)
        self.vx_noise = float(vx_noise)
        self.vy_noise = float(vy_noise)
        self.w_noise = float(w_noise)
        self.h_noise = float(h_noise)

    def reset(self, reset_id: bool = True):
        self.tracks.clear()
        if reset_id:
            KalmanSimple.reset_next_id(0)

    def predict(self, detections: Sequence[DetectionObject]) -> List[DetectionObject]:
        for track in self.tracks:
            track.predict()

        meas_list = [_meas_from_object(detection) for detection in detections]
        meas_prob = [float(detection.prob) for detection in detections]
        state_list5 = [track.x_pri[:5] for track in self.tracks]

        cost = [[1.0 for _ in meas_list] for _ in state_list5]
        for i, state in enumerate(state_list5):
            for j, meas in enumerate(meas_list):
                iou = _iou_from_meas(state, meas)
                cost_iou = 1.0 - iou
                dx = state[1] - meas[1]
                dy = state[2] - meas[2]
                w_avg = (state[3] + meas[3]) * 0.5
                h_avg = (state[4] + meas[4]) * 0.5
                dist_norm_x = abs(dx) / (w_avg * 3.0) if w_avg != 0.0 else 1.0
                dist_norm_y = abs(dy) / (h_avg * 3.0) if h_avg != 0.0 else 1.0
                cost_dist = min((dist_norm_x + dist_norm_y) * 0.5, 1.0)
                area_pred = state[3] * state[4]
                area_meas = meas[3] * meas[4]
                area_max = max(area_pred, area_meas)
                cost_shape = abs(area_pred - area_meas) / area_max if area_max > 0.0 else 1.0
                cost[i][j] = 0.1 * cost_iou + 0.7 * cost_dist + 0.2 * cost_shape

        assignment = hungarian_min(cost)
        meas_used = [False for _ in meas_list]
        to_remove = set()
        for i, track in enumerate(self.tracks):
            j = assignment[i] if i < len(assignment) else -1
            matched = 0 <= j < len(meas_list)
            if matched and cost[i][j] < 1.0:
                track.set_prob(meas_prob[j])
                if not track.update(meas_list[j]):
                    to_remove.add(i)
                meas_used[j] = True
            elif not track.update_unmatched():
                to_remove.add(i)

        if to_remove:
            self.tracks = [track for idx, track in enumerate(self.tracks) if idx not in to_remove]

        for j, used in enumerate(meas_used):
            if used:
                continue
            track = KalmanSimple(
                self.terminate_set,
                self.generate_set,
                self.vx_noise,
                self.vy_noise,
                self.w_noise,
                self.h_noise,
                self.r_std,
            )
            track.init_from_meas(meas_list[j])
            track.set_prob(meas_prob[j])
            self.tracks.append(track)

        outputs: List[DetectionObject] = []
        for track in self.tracks:
            if not track.has_last_z or not track.is_confirmed():
                continue
            bbox = _meas_to_bbox(track.last_z6)
            outputs.append(DetectionObject(
                bbox=bbox,
                label=int(track.last_z6[0]),
                prob=track.last_prob,
                track_id=int(track.last_z6[5]),
            ))
        return outputs
