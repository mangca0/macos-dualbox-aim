import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_tune_v63 as base  # noqa: E402


DEFAULT_BASE_URL = base.DEFAULT_BASE_URL
DEFAULT_OUT_DIR = base.DEFAULT_OUT_DIR
AUTOTUNE_FIELDS = base.AUTOTUNE_FIELDS
FIELD_SPECS = base.FIELD_SPECS
COMBO_GROUPS = base.COMBO_GROUPS
DEFAULT_COMBO_STRENGTH = base.DEFAULT_COMBO_STRENGTH

TunerClient = base.TunerClient
FieldSpec = base.FieldSpec
score_metrics = base.score_metrics
candidate_values = base.candidate_values
generate_candidates = base.generate_candidates
generate_combo_candidates = base.generate_combo_candidates
generate_group_combo_candidate = base.generate_group_combo_candidate
generate_mixed_combo_candidate = base.generate_mixed_combo_candidate
generate_auto_iteration_candidates = base.generate_auto_iteration_candidates
run_auto_iterations = base.run_auto_iterations
evaluate_candidate = base.evaluate_candidate
evaluate_trial = base.evaluate_trial
run_trial = base.run_trial
is_hard_failure = base.is_hard_failure
select_restore_config = base.select_restore_config
random = base.random


def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    originals = _patch_base_module()
    try:
        return base.run_search(args)
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


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
        "schema": "macos-dualbox-aim.v7.autotune_trial.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "stage": base._stage_name(label),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune V7 controller parameters through the live tuner API.")
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
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible combo trials.")
    parser.add_argument("--restore-on-interrupt", choices=("original", "best", "none"), default="original", help="Runtime config to restore after Ctrl-C.")
    parser.add_argument("--auto-trigger", action="store_true", help="Keep V7 aim active during tuning and force-disable it on exit.")
    parser.add_argument("--no-fail-fast", action="store_false", dest="fail_fast", help="Disable hard-failure early stop within repeated trials.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for JSONL trial records.")
    parser.add_argument("--save-best", action="store_true", help="Persist the final best hot-updated config to disk.")
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
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


def _new_output_path(out_dir: str | Path) -> Path:
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / f"v7_auto_tune_{started_at}.jsonl"


def _patch_base_module() -> Dict[str, Any]:
    patches = {
        "TunerClient": TunerClient,
        "write_record": write_record,
        "_new_output_path": _new_output_path,
        "generate_candidates": generate_candidates,
        "generate_group_combo_candidate": generate_group_combo_candidate,
        "generate_mixed_combo_candidate": generate_mixed_combo_candidate,
        "generate_auto_iteration_candidates": generate_auto_iteration_candidates,
        "evaluate_candidate": evaluate_candidate,
        "evaluate_trial": evaluate_trial,
        "run_trial": run_trial,
        "score_metrics": score_metrics,
    }
    originals = {name: getattr(base, name) for name in patches}
    for name, value in patches.items():
        setattr(base, name, value)
    return originals


def _format_score(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.3f}"


def main() -> int:
    run_search(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
