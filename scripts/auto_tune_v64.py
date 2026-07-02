import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_OUT_DIR = "aim_tuning_runs"

AUTOTUNE_FIELDS = (
    "pid_kp",
    "pid_ki",
    "pid_kd",
    "max_speed",
    "sensitivity",
    "init_scale",
    "ramp_time",
    "pred_weight_x",
    "pred_weight_y",
    "target_jump_reset",
    "pid_integral_gate_threshold",
    "pid_integral_gate_rate",
    "stop_brake_radius",
    "stop_brake_output_decay",
    "stop_brake_pred_decay",
    "stop_brake_min_output",
)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    minimum: float
    maximum: float
    relative_steps: tuple[float, ...] = (0.85, 1.15)
    absolute_steps: tuple[float, ...] = ()


FIELD_SPECS = {
    "pid_kp": FieldSpec("pid_kp", 0.0, 2.0, absolute_steps=(0.05, -0.05)),
    "pid_ki": FieldSpec("pid_ki", 0.0, 1.0, absolute_steps=(0.001, 0.005, -0.001)),
    "pid_kd": FieldSpec("pid_kd", 0.0, 2.0, absolute_steps=(0.025, 0.05, -0.025)),
    "max_speed": FieldSpec("max_speed", 1.0, 200.0),
    "sensitivity": FieldSpec("sensitivity", 0.01, 5.0),
    "init_scale": FieldSpec("init_scale", 0.05, 1.0, relative_steps=(0.9, 1.1), absolute_steps=(0.05, -0.05)),
    "ramp_time": FieldSpec("ramp_time", 0.001, 2.0, relative_steps=(0.8, 1.2), absolute_steps=(0.025, -0.025)),
    "pred_weight_x": FieldSpec("pred_weight_x", 0.0, 1.0, relative_steps=(0.8, 1.2), absolute_steps=(0.05, -0.05)),
    "pred_weight_y": FieldSpec("pred_weight_y", 0.0, 1.0, relative_steps=(0.8, 1.2), absolute_steps=(0.05, -0.05)),
    "target_jump_reset": FieldSpec("target_jump_reset", 0.0, 300.0),
    "pid_integral_gate_threshold": FieldSpec("pid_integral_gate_threshold", 1.0, 300.0),
    "pid_integral_gate_rate": FieldSpec("pid_integral_gate_rate", 0.0, 1.0, relative_steps=(0.75, 1.25), absolute_steps=(0.025, -0.025)),
    "stop_brake_radius": FieldSpec("stop_brake_radius", 0.0, 80.0, relative_steps=(0.8, 1.2), absolute_steps=(2.0, -2.0)),
    "stop_brake_output_decay": FieldSpec("stop_brake_output_decay", 0.0, 1.0, relative_steps=(0.75, 1.25), absolute_steps=(0.05, -0.05)),
    "stop_brake_pred_decay": FieldSpec("stop_brake_pred_decay", 0.0, 1.0, relative_steps=(0.75, 1.25), absolute_steps=(0.05, -0.05)),
    "stop_brake_min_output": FieldSpec("stop_brake_min_output", 0.0, 200.0, relative_steps=(0.8, 1.2), absolute_steps=(5.0, -5.0)),
}

COMBO_GROUPS = (
    (
        "pid",
        (
            "pid_kp",
            "pid_ki",
            "pid_kd",
            "pid_integral_gate_threshold",
            "pid_integral_gate_rate",
        ),
    ),
    (
        "response",
        (
            "max_speed",
            "sensitivity",
            "init_scale",
            "ramp_time",
        ),
    ),
    (
        "prediction",
        (
            "pred_weight_x",
            "pred_weight_y",
            "target_jump_reset",
            "stop_brake_radius",
            "stop_brake_output_decay",
            "stop_brake_pred_decay",
            "stop_brake_min_output",
        ),
    ),
)

DEFAULT_COMBO_STRENGTH = 0.12
FAIL_FAST_LOST_RATIO = 0.5
FAIL_FAST_OVERSHOOT_COUNT = 30.0
FAIL_FAST_OSCILLATION_ENERGY = 60.0
AUTO_STAGE_NAMES = ("explore", "exploit", "refine")


def score_metrics(metrics: Dict[str, Any], *, min_samples: int = 20) -> float:
    if not metrics.get("available") or int(metrics.get("samples", 0)) < min_samples:
        return math.inf
    lost_ratio = float(metrics.get("target_lost_ratio", 1.0))
    mean_error = float(metrics.get("mean_abs_error", 0.0))
    p95_error = float(metrics.get("p95_abs_error", mean_error))
    mean_x_error = float(metrics.get("mean_abs_x_error", mean_error))
    p95_x_error = float(metrics.get("p95_abs_x_error", p95_error))
    p99_x_error = float(metrics.get("p99_abs_x_error", p95_x_error))
    signed_x_error = abs(float(metrics.get("mean_signed_x_error", 0.0)))
    mean_y_error = float(metrics.get("mean_abs_y_error", mean_error))
    p95_y_error = float(metrics.get("p95_abs_y_error", p95_error))
    overshoot = float(metrics.get("overshoot_count", 0.0))
    x_crossing = float(metrics.get("x_crossing_count", 0.0))
    oscillation = float(metrics.get("oscillation_energy", 0.0))
    mean_move = float(metrics.get("mean_move", 0.0))
    settled_ratio = float(metrics.get("settled_ratio", 0.0))
    x_dwell_1px = float(metrics.get("x_center_dwell_ratio_1px", 0.0))
    x_dwell_2px = float(metrics.get("x_center_dwell_ratio_2px", 0.0))
    time_to_x_settle_ms = float(metrics.get("time_to_x_settle_ms", 0.0))
    return (
        mean_error * 0.4
        + p95_error * 0.35
        + mean_x_error * 1.8
        + p95_x_error * 1.2
        + p99_x_error * 0.5
        + signed_x_error * 0.7
        + mean_y_error * 0.55
        + p95_y_error * 0.35
        + lost_ratio * 100.0
        + overshoot * 3.0
        + x_crossing * 2.0
        + oscillation * 0.35
        + mean_move * 0.03
        + time_to_x_settle_ms * 0.002
        - settled_ratio * 6.0
        - x_dwell_1px * 12.0
        - x_dwell_2px * 8.0
    )


def candidate_values(config: Dict[str, Any], field: str) -> List[float]:
    spec = FIELD_SPECS[field]
    current = float(config[field])
    values = {current}
    for factor in spec.relative_steps:
        values.add(current * factor)
    for step in spec.absolute_steps:
        values.add(current + step)
    values.update(_boundary_probe_values(current, spec))
    return [
        round(value, 6)
        for value in sorted({_clamp(value, spec.minimum, spec.maximum) for value in values})
        if abs(value - current) > 1e-9
    ]


def generate_candidates(config: Dict[str, Any], fields: Iterable[str]) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    for field in fields:
        for value in candidate_values(config, field):
            candidates.append({field: value})
    return candidates


def generate_combo_candidates(
    config: Dict[str, Any],
    *,
    trials: int,
    rng: random.Random,
    strength: float = DEFAULT_COMBO_STRENGTH,
    shrink: float = 0.75,
) -> List[tuple[str, Dict[str, float]]]:
    candidates: List[tuple[str, Dict[str, float]]] = []
    if trials <= 0:
        return candidates
    for index in range(trials):
        label, fields = COMBO_GROUPS[index % len(COMBO_GROUPS)]
        cycle = index // len(COMBO_GROUPS)
        candidate_strength = strength * (shrink ** cycle)
        candidate = generate_group_combo_candidate(config, fields, rng=rng, strength=candidate_strength)
        if candidate:
            candidates.append((label, candidate))
    return candidates


def generate_group_combo_candidate(
    config: Dict[str, Any],
    fields: Iterable[str],
    *,
    rng: random.Random,
    strength: float,
) -> Dict[str, float]:
    candidate: Dict[str, float] = {}
    for field in fields:
        spec = FIELD_SPECS[field]
        current = float(config[field])
        value = _perturb_value(current, spec, rng=rng, strength=strength)
        if abs(value - current) > 1e-9:
            candidate[field] = value
    return candidate


def generate_mixed_combo_candidate(
    config: Dict[str, Any],
    *,
    rng: random.Random,
    strength: float,
    min_fields: int = 3,
    max_fields: int = 5,
) -> Dict[str, float]:
    field_count = rng.randint(min_fields, max_fields)
    fields = rng.sample(list(AUTOTUNE_FIELDS), min(field_count, len(AUTOTUNE_FIELDS)))
    return generate_group_combo_candidate(config, fields, rng=rng, strength=strength)


def generate_auto_iteration_candidates(
    config: Dict[str, Any],
    *,
    rng: random.Random,
    strength: float,
    trials: int,
) -> List[tuple[str, Dict[str, float]]]:
    candidates: List[tuple[str, Dict[str, float]]] = []
    if trials <= 0:
        return candidates
    for index in range(trials):
        stage = AUTO_STAGE_NAMES[index % len(AUTO_STAGE_NAMES)]
        if stage == "explore":
            candidate = generate_mixed_combo_candidate(
                config,
                rng=rng,
                strength=strength,
                min_fields=4,
                max_fields=6,
            )
        elif stage == "exploit":
            _group_name, fields = rng.choice(COMBO_GROUPS)
            candidate = generate_group_combo_candidate(config, fields, rng=rng, strength=strength * 0.65)
        else:
            field = rng.choice(AUTOTUNE_FIELDS)
            values = candidate_values(config, field)
            if not values:
                continue
            candidate = {field: rng.choice(values)}
        if candidate:
            candidates.append((stage, candidate))
    return candidates


def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    client = TunerClient(args.url, timeout_s=args.timeout)
    snapshot = client.get_config()
    current_config = dict(snapshot["config"])
    original_config = {field: current_config[field] for field in AUTOTUNE_FIELDS}
    best_config = {field: current_config[field] for field in AUTOTUNE_FIELDS}
    best_trial = evaluate_trial(client, args)
    best_score = best_trial["score"]
    best_metrics = best_trial["metrics"]
    output = _new_output_path(args.out_dir)
    write_record(output, "baseline", {}, best_trial, best_config, accepted=True, original_config=original_config, seed=args.seed)

    interrupted = False
    if args.auto_trigger:
        client.set_aim_active(True)
    try:
        print(f"baseline score={_format_score(best_score)} samples={best_metrics.get('samples', 0)}")
        for pass_index in range(args.passes):
            improved = False
            for candidate in generate_candidates(best_config, AUTOTUNE_FIELDS):
                trial = evaluate_candidate(
                    client,
                    args,
                    output,
                    f"pass{pass_index + 1}",
                    candidate,
                    best_config,
                    best_score,
                    original_config=original_config,
                )
                if trial["accepted"]:
                    best_config = trial["config"]
                    best_score = trial["score"]
                    best_metrics = trial["metrics"]
                    improved = True
                else:
                    client.update_config(best_config)
            if not improved:
                break

        rng = random.Random(args.seed)
        for combo_index in range(1, args.combo_trials + 1):
            group_name, fields = COMBO_GROUPS[(combo_index - 1) % len(COMBO_GROUPS)]
            cycle = (combo_index - 1) // len(COMBO_GROUPS)
            candidate_strength = args.combo_strength * (args.combo_shrink ** cycle)
            candidate = generate_group_combo_candidate(best_config, fields, rng=rng, strength=candidate_strength)
            if not candidate:
                continue
            trial = evaluate_candidate(
                client,
                args,
                output,
                f"combo{combo_index}:{group_name}",
                candidate,
                best_config,
                best_score,
                original_config=original_config,
            )
            if trial["accepted"]:
                best_config = trial["config"]
                best_score = trial["score"]
                best_metrics = trial["metrics"]
            else:
                client.update_config(best_config)

        for mixed_index in range(1, args.mixed_trials + 1):
            cycle = (mixed_index - 1) // max(1, args.mixed_shrink_every)
            candidate_strength = args.combo_strength * (args.combo_shrink ** cycle)
            candidate = generate_mixed_combo_candidate(best_config, rng=rng, strength=candidate_strength)
            if not candidate:
                continue
            trial = evaluate_candidate(
                client,
                args,
                output,
                f"mixed{mixed_index}",
                candidate,
                best_config,
                best_score,
                original_config=original_config,
            )
            if trial["accepted"]:
                best_config = trial["config"]
                best_score = trial["score"]
                best_metrics = trial["metrics"]
            else:
                client.update_config(best_config)

        auto_result = run_auto_iterations(
            client,
            args,
            output,
            best_config,
            best_score,
            best_metrics,
            original_config=original_config,
            rng=rng,
        )
        best_config = auto_result["best_config"]
        best_score = auto_result["best_score"]
        best_metrics = auto_result["best_metrics"]
    except KeyboardInterrupt:
        interrupted = True
        restore_config = select_restore_config(args.restore_on_interrupt, original_config, best_config)
        if restore_config is not None:
            client.update_config(restore_config)
        print(f"interrupted; restored={args.restore_on_interrupt}")
    else:
        client.update_config(best_config)
        if args.save_best:
            client.save_config()
    finally:
        if args.auto_trigger:
            client.set_aim_active(False)
    result = {
        "best_score": best_score,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "record_path": str(output),
        "saved": bool(args.save_best and not interrupted),
        "interrupted": interrupted,
        "restore_on_interrupt": args.restore_on_interrupt,
        "auto_trigger": bool(args.auto_trigger),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def run_auto_iterations(
    client: "TunerClient",
    args: argparse.Namespace,
    output: Path,
    best_config: Dict[str, Any],
    best_score: float,
    best_metrics: Dict[str, Any],
    *,
    original_config: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    strength = args.auto_strength
    stale_iterations = 0
    trials_used = 0
    for iteration in range(1, args.auto_iterations + 1):
        if args.auto_max_trials and trials_used >= args.auto_max_trials:
            break
        improved = False
        candidates = generate_auto_iteration_candidates(
            best_config,
            rng=rng,
            strength=strength,
            trials=args.auto_trials_per_iteration,
        )
        for local_index, (stage, candidate) in enumerate(candidates, start=1):
            if args.auto_max_trials and trials_used >= args.auto_max_trials:
                break
            trials_used += 1
            trial = evaluate_candidate(
                client,
                args,
                output,
                f"auto{iteration}.{local_index}:{stage}",
                candidate,
                best_config,
                best_score,
                original_config=original_config,
            )
            if trial["accepted"]:
                best_config = trial["config"]
                best_score = trial["score"]
                best_metrics = trial["metrics"]
                improved = True
            else:
                client.update_config(best_config)
        if improved:
            stale_iterations = 0
            strength = min(args.auto_max_strength, strength * args.auto_expand)
        else:
            stale_iterations += 1
            strength *= args.auto_shrink
        print(
            f"auto iteration={iteration} best={_format_score(best_score)} "
            f"strength={strength:.4f} improved={improved} stale={stale_iterations}"
        )
        if stale_iterations >= args.auto_patience:
            break
        if strength < args.auto_min_strength:
            break
    return {
        "best_config": best_config,
        "best_score": best_score,
        "best_metrics": best_metrics,
    }


def evaluate_candidate(
    client: "TunerClient",
    args: argparse.Namespace,
    output: Path,
    label: str,
    candidate: Dict[str, float],
    best_config: Dict[str, Any],
    best_score: float,
    *,
    original_config: Dict[str, Any],
) -> Dict[str, Any]:
    trial_config = dict(best_config)
    trial_config.update(candidate)
    client.update_config(candidate)
    trial = evaluate_trial(client, args)
    accepted = trial["score"] < best_score * (1.0 - args.min_improvement)
    write_record(
        output,
        label,
        candidate,
        trial,
        trial_config,
        accepted=accepted,
        original_config=original_config,
        seed=args.seed,
    )
    print(f"{label} {candidate} score={_format_score(trial['score'])} accepted={accepted}")
    trial["accepted"] = accepted
    trial["config"] = trial_config
    return trial


def evaluate_trial(client: "TunerClient", args: argparse.Namespace) -> Dict[str, Any]:
    scores: List[float] = []
    metrics_runs: List[Dict[str, Any]] = []
    for _repeat_index in range(args.repeats):
        metrics = run_trial(client, args.warmup, args.duration)
        score = score_metrics(metrics, min_samples=args.min_samples)
        scores.append(score)
        metrics_runs.append(metrics)
        if args.fail_fast and is_hard_failure(metrics):
            break
    score = median(scores) if scores else math.inf
    representative_index = min(range(len(scores)), key=lambda index: abs(scores[index] - score)) if scores else 0
    metrics = metrics_runs[representative_index] if metrics_runs else {}
    return {
        "score": score,
        "metrics": metrics,
        "repeat_scores": scores,
        "repeat_metrics": metrics_runs,
        "hard_failure": any(is_hard_failure(metrics) for metrics in metrics_runs),
    }


def run_trial(client: "TunerClient", warmup_s: float, duration_s: float) -> Dict[str, Any]:
    client.reset_aim()
    if warmup_s > 0.0:
        time.sleep(warmup_s)
        client.reset_aim()
    time.sleep(duration_s)
    return dict(client.get_config().get("aim", {}))


def write_record(
    path: Path,
    label: str,
    candidate: Dict[str, float],
    trial: Dict[str, Any],
    config: Dict[str, Any],
    *,
    accepted: bool,
    original_config: Dict[str, Any],
    seed: int,
):
    record = {
        "schema": "macos-dualbox-aim.v64.autotune_trial.v2",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "stage": _stage_name(label),
        "candidate": candidate,
        "score": trial["score"],
        "repeat_scores": trial.get("repeat_scores", []),
        "hard_failure": bool(trial.get("hard_failure", False)),
        "metrics": trial["metrics"],
        "repeat_metrics": trial.get("repeat_metrics", []),
        "config": config,
        "original_config": original_config,
        "seed": seed,
        "accepted": accepted,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


class TunerClient:
    def __init__(self, base_url: str, *, timeout_s: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def get_config(self) -> Dict[str, Any]:
        return self._request("GET", "/api/config")

    def update_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/config", data)

    def reset_aim(self) -> Dict[str, Any]:
        return self._request("POST", "/api/aim/reset", {})

    def set_aim_active(self, active: bool) -> Dict[str, Any]:
        return self._request("POST", "/api/aim/active", {"active": bool(active)})

    def save_config(self) -> Dict[str, Any]:
        return self._request("POST", "/api/save", {})

    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Failed to reach tuner at {self.base_url}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Tuner response must be a JSON object")
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune V6.4 controller parameters through the live tuner API.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base tuner URL, default: http://127.0.0.1:8765")
    parser.add_argument("--duration", type=float, default=4.0, help="Measured seconds per candidate after warmup.")
    parser.add_argument("--warmup", type=float, default=0.75, help="Warmup seconds after each parameter update.")
    parser.add_argument("--passes", type=int, default=2, help="Maximum coordinate-search passes.")
    parser.add_argument("--timeout", type=float, default=1.0, help="HTTP timeout per tuner request.")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum aim samples required to score a trial.")
    parser.add_argument("--min-improvement", type=float, default=0.01, help="Relative score improvement required to accept.")
    parser.add_argument("--repeats", type=int, default=1, help="Measured repeats per candidate; median score is used.")
    parser.add_argument("--combo-trials", type=int, default=0, help="Grouped random perturbation trials after coordinate search.")
    parser.add_argument("--mixed-trials", type=int, default=0, help="Cross-group random perturbation trials after grouped combos.")
    parser.add_argument("--mixed-shrink-every", type=int, default=4, help="Mixed trials per strength shrink step.")
    parser.add_argument("--combo-strength", type=float, default=DEFAULT_COMBO_STRENGTH, help="Initial relative strength for combo perturbations.")
    parser.add_argument("--combo-shrink", type=float, default=0.75, help="Strength multiplier after each combo group cycle.")
    parser.add_argument("--auto-iterations", type=int, default=0, help="Adaptive search iterations after fixed stages.")
    parser.add_argument("--auto-trials-per-iteration", type=int, default=12, help="Adaptive candidates per iteration.")
    parser.add_argument("--auto-max-trials", type=int, default=0, help="Maximum adaptive candidates; 0 means unlimited by this setting.")
    parser.add_argument("--auto-patience", type=int, default=3, help="Stop after this many adaptive iterations without improvement.")
    parser.add_argument("--auto-strength", type=float, default=DEFAULT_COMBO_STRENGTH, help="Initial adaptive perturbation strength.")
    parser.add_argument("--auto-min-strength", type=float, default=0.01, help="Stop adaptive search below this strength.")
    parser.add_argument("--auto-max-strength", type=float, default=0.30, help="Maximum adaptive perturbation strength after successful iterations.")
    parser.add_argument("--auto-shrink", type=float, default=0.55, help="Adaptive strength multiplier after no-improvement iterations.")
    parser.add_argument("--auto-expand", type=float, default=1.08, help="Adaptive strength multiplier after improved iterations.")
    parser.add_argument("--seed", type=int, default=63, help="Random seed for reproducible combo trials.")
    parser.add_argument("--restore-on-interrupt", choices=("original", "best", "none"), default="original", help="Runtime config to restore after Ctrl-C.")
    parser.add_argument("--auto-trigger", action="store_true", help="Keep V6.4 aim active during tuning and force-disable it on exit.")
    parser.add_argument("--no-fail-fast", action="store_false", dest="fail_fast", help="Disable hard-failure early stop within repeated trials.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for JSONL trial records.")
    parser.add_argument("--save-best", action="store_true", help="Persist the final best hot-updated config to disk.")
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.warmup < 0.0:
        parser.error("--warmup must be zero or positive")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.combo_trials < 0:
        parser.error("--combo-trials must be zero or positive")
    if args.mixed_trials < 0:
        parser.error("--mixed-trials must be zero or positive")
    if args.mixed_shrink_every <= 0:
        parser.error("--mixed-shrink-every must be positive")
    if args.combo_strength <= 0.0:
        parser.error("--combo-strength must be positive")
    if not 0.0 < args.combo_shrink <= 1.0:
        parser.error("--combo-shrink must be in (0, 1]")
    if args.auto_iterations < 0:
        parser.error("--auto-iterations must be zero or positive")
    if args.auto_trials_per_iteration <= 0:
        parser.error("--auto-trials-per-iteration must be positive")
    if args.auto_max_trials < 0:
        parser.error("--auto-max-trials must be zero or positive")
    if args.auto_patience <= 0:
        parser.error("--auto-patience must be positive")
    if args.auto_strength <= 0.0:
        parser.error("--auto-strength must be positive")
    if args.auto_min_strength <= 0.0:
        parser.error("--auto-min-strength must be positive")
    if args.auto_max_strength < args.auto_min_strength:
        parser.error("--auto-max-strength must be >= --auto-min-strength")
    if not 0.0 < args.auto_shrink < 1.0:
        parser.error("--auto-shrink must be in (0, 1)")
    if args.auto_expand < 1.0:
        parser.error("--auto-expand must be >= 1")
    return args


def _new_output_path(out_dir: str | Path) -> Path:
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / f"v64_auto_tune_{started_at}.jsonl"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else (maximum if value > maximum else value)


def _perturb_value(current: float, spec: FieldSpec, *, rng: random.Random, strength: float) -> float:
    if abs(current) > 1e-9:
        value = current * (1.0 + rng.uniform(-strength, strength))
    else:
        value = current + rng.choice((-1.0, 1.0)) * _zero_step(spec, strength) * rng.uniform(0.35, 1.0)
    return round(_clamp(value, spec.minimum, spec.maximum), 6)


def _zero_step(spec: FieldSpec, strength: float) -> float:
    if spec.absolute_steps:
        base_step = max(abs(step) for step in spec.absolute_steps)
    else:
        base_step = (spec.maximum - spec.minimum) * 0.05
    return base_step * (strength / DEFAULT_COMBO_STRENGTH)


def _boundary_probe_values(current: float, spec: FieldSpec) -> List[float]:
    span = spec.maximum - spec.minimum
    if span <= 0.0:
        return []
    if current >= spec.maximum - span * 0.01:
        return [
            spec.maximum - span * 0.05,
            spec.maximum - span * 0.15,
            spec.maximum - span * 0.30,
        ]
    if current <= spec.minimum + span * 0.01:
        return [
            spec.minimum + span * 0.05,
            spec.minimum + span * 0.15,
            spec.minimum + span * 0.30,
        ]
    return []


def is_hard_failure(metrics: Dict[str, Any]) -> bool:
    return (
        float(metrics.get("target_lost_ratio", 0.0)) >= FAIL_FAST_LOST_RATIO
        or float(metrics.get("overshoot_count", 0.0)) >= FAIL_FAST_OVERSHOOT_COUNT
        or float(metrics.get("oscillation_energy", 0.0)) >= FAIL_FAST_OSCILLATION_ENERGY
    )


def select_restore_config(
    mode: str,
    original_config: Dict[str, Any],
    best_config: Dict[str, Any],
) -> Dict[str, Any] | None:
    if mode == "original":
        return original_config
    if mode == "best":
        return best_config
    return None


def _stage_name(label: str) -> str:
    if label == "baseline":
        return "baseline"
    if label.startswith("pass"):
        return "coordinate"
    if label.startswith("combo"):
        return "combo"
    if label.startswith("mixed"):
        return "mixed"
    if label.startswith("auto"):
        return "auto"
    return label.split(":", 1)[0]


def _format_score(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def main() -> int:
    run_search(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
