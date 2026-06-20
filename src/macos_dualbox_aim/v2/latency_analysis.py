import argparse
import glob
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_URL = "http://127.0.0.1:8765/api/config"
DEFAULT_OUT_DIR = "latency_runs"
DEFAULT_METRICS = [
    "read_included_total_ms",
    "program_total_ms",
    "capture_read_ms",
    "capture_grab_ms",
    "capture_retrieve_ms",
    "capture_frame_interval_ms",
    "crop_ms",
    "queue_wait_ms",
    "preprocess_ms",
    "coreml_ms",
    "postprocess_ms",
    "inference_ms",
    "detection_callback_ms",
    "target_select_ms",
    "pid_ms",
    "kmbox_send_ack_ms",
]
SAMPLE_SCHEMA = "macos-dualbox-aim.latency_sample.v2"


@dataclass(frozen=True)
class MetricSummary:
    avg_ms: float
    current_avg_ms: float
    current_p95_ms: float
    current_max_ms: float
    samples: int


@dataclass(frozen=True)
class RunSummary:
    label: str
    run: str
    sample_count: int
    metrics: Dict[str, MetricSummary]
    counters: Dict[str, int]
    source_files: List[str]


def fetch_tuner_snapshot(url: str, timeout: float = 1.0) -> Dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Tuner response must be a JSON object")
    return data


def write_sample(
    path: str | Path,
    *,
    label: str,
    run: str,
    sample_index: int,
    url: str,
    snapshot: Dict[str, Any],
):
    latency = snapshot.get("latency", {})
    if not isinstance(latency, dict):
        latency = {}
    record = {
        "schema": SAMPLE_SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "run": run,
        "sample_index": int(sample_index),
        "url": url,
        "runtime": snapshot.get("runtime", {}),
        "latency": latency,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def capture_run(
    *,
    label: str,
    run: str,
    duration_s: float,
    interval_s: float,
    url: str = DEFAULT_URL,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    timeout_s: float = 1.0,
    fetcher: Callable[[str, float], Dict[str, Any]] = fetch_tuner_snapshot,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Path:
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    if interval_s <= 0:
        raise ValueError("interval must be positive")

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_slug(label)}_{_slug(run)}_{started_at}.jsonl"
    output = Path(out_dir) / filename
    deadline = clock() + duration_s
    sample_index = 0
    while clock() < deadline:
        snapshot = fetcher(url, timeout_s)
        _validate_runtime_label(snapshot, label)
        write_sample(
            output,
            label=label,
            run=run,
            sample_index=sample_index,
            url=url,
            snapshot=snapshot,
        )
        sample_index += 1
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleeper(min(interval_s, remaining))
    if sample_index == 0:
        raise RuntimeError("No latency samples were captured")
    return output


def load_records(inputs: Sequence[str]) -> List[Dict[str, Any]]:
    paths = _expand_inputs(inputs)
    records: List[Dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Record must be a JSON object at {path}:{line_number}")
                record.setdefault("source_file", str(path))
                records.append(record)
    return records


def summarize_records(records: Sequence[Dict[str, Any]]) -> List[RunSummary]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        label = str(record.get("label", "unknown"))
        run = str(record.get("run", "run"))
        grouped.setdefault((label, run), []).append(record)
    return [summarize_run(group) for group in grouped.values()]


def summarize_run(records: Sequence[Dict[str, Any]]) -> RunSummary:
    if not records:
        raise ValueError("Cannot summarize an empty run")

    label = str(records[0].get("label", "unknown"))
    run = str(records[0].get("run", "run"))
    source_files = sorted({
        str(record.get("source_file"))
        for record in records
        if record.get("source_file") is not None
    })
    metric_names = sorted({
        key
        for record in records
        for key in _number_map(record, "avg")
    } | {
        key
        for record in records
        for key in _number_map(record, "current")
    })
    metrics: Dict[str, MetricSummary] = {}
    for name in metric_names:
        avg_values = [
            values[name]
            for record in records
            for values in [_number_map(record, "avg")]
            if name in values
        ]
        current_values = [
            values[name]
            for record in records
            for values in [_number_map(record, "current")]
            if name in values
        ]
        primary_values = avg_values or current_values
        if not primary_values:
            continue
        metrics[name] = MetricSummary(
            avg_ms=_mean(primary_values),
            current_avg_ms=_mean(current_values) if current_values else _mean(primary_values),
            current_p95_ms=_percentile(current_values or primary_values, 95.0),
            current_max_ms=max(current_values or primary_values),
            samples=len(primary_values),
        )

    counters = _last_counter_snapshot(records)
    return RunSummary(
        label=label,
        run=run,
        sample_count=len(records),
        metrics=metrics,
        counters=counters,
        source_files=source_files,
    )


def compare_labels(
    summaries: Sequence[RunSummary],
    baseline_label: str,
    candidate_label: str,
    metrics: Sequence[str],
) -> List[Dict[str, float | str | int]]:
    baseline_runs = [summary for summary in summaries if summary.label == baseline_label]
    candidate_runs = [summary for summary in summaries if summary.label == candidate_label]
    if not baseline_runs:
        raise ValueError(f"No runs found for baseline label: {baseline_label}")
    if not candidate_runs:
        raise ValueError(f"No runs found for candidate label: {candidate_label}")

    rows: List[Dict[str, float | str | int]] = []
    for metric in metrics:
        baseline_values = [run.metrics[metric].avg_ms for run in baseline_runs if metric in run.metrics]
        candidate_values = [run.metrics[metric].avg_ms for run in candidate_runs if metric in run.metrics]
        if not baseline_values or not candidate_values:
            continue
        baseline_avg = _mean(baseline_values)
        candidate_avg = _mean(candidate_values)
        delta = candidate_avg - baseline_avg
        change_pct = delta * 100.0 / baseline_avg if baseline_avg else math.nan
        rows.append({
            "metric": metric,
            "baseline_runs": len(baseline_values),
            "candidate_runs": len(candidate_values),
            "baseline_avg_ms": baseline_avg,
            "baseline_std_ms": _stddev(baseline_values),
            "candidate_avg_ms": candidate_avg,
            "candidate_std_ms": _stddev(candidate_values),
            "delta_ms": delta,
            "change_pct": change_pct,
        })
    return rows


def format_run_summary(summary: RunSummary, metrics: Sequence[str] = DEFAULT_METRICS) -> str:
    lines = [
        f"# Latency run: {summary.label} / {summary.run}",
        "",
        f"samples: {summary.sample_count}",
    ]
    if summary.counters:
        captured = summary.counters.get("frames_captured", 0)
        dropped = summary.counters.get("frames_dropped", 0)
        drop_rate = dropped * 100.0 / captured if captured else 0.0
        lines.append(
            f"frames: captured {captured}, inferred {summary.counters.get('frames_inferred', 0)}, "
            f"dropped {dropped} ({drop_rate:.2f}%)"
        )
    lines.extend(["", "| metric | avg ms | sampled current avg | sampled current p95 | sampled current max |", "|---|---:|---:|---:|---:|"])
    for metric in metrics:
        item = summary.metrics.get(metric)
        if item is None:
            continue
        lines.append(
            f"| {metric} | {_fmt(item.avg_ms)} | {_fmt(item.current_avg_ms)} | "
            f"{_fmt(item.current_p95_ms)} | {_fmt(item.current_max_ms)} |"
        )
    return "\n".join(lines)


def format_comparison(
    rows: Sequence[Dict[str, float | str | int]],
    baseline_label: str,
    candidate_label: str,
) -> str:
    lines = [
        f"# Latency comparison: {baseline_label} -> {candidate_label}",
        "",
        "Negative delta/change means the candidate is faster.",
        "",
        "| metric | baseline avg ms | candidate avg ms | delta ms | change | runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {_fmt(float(row['baseline_avg_ms']))} ± {_fmt(float(row['baseline_std_ms']))} | "
            f"{_fmt(float(row['candidate_avg_ms']))} ± {_fmt(float(row['candidate_std_ms']))} | "
            f"{_fmt(float(row['delta_ms']))} | "
            f"{_fmt(float(row['change_pct']))}% | {row['baseline_runs']}->{row['candidate_runs']} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and compare V2 tuner latency snapshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture tuner latency snapshots into JSONL.")
    capture.add_argument("--label", required=True, help="Version label, for example v2.0.0 or v2.2.3.")
    capture.add_argument("--run", default=None, help="Run id. Defaults to run timestamp.")
    capture.add_argument("--duration", type=float, default=60.0, help="Capture duration in seconds.")
    capture.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds.")
    capture.add_argument("--url", default=DEFAULT_URL, help="Tuner /api/config URL.")
    capture.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for JSONL runs.")
    capture.add_argument("--timeout", type=float, default=1.0, help="HTTP timeout per tuner request.")

    compare = subparsers.add_parser("compare", help="Compare captured JSONL files.")
    compare.add_argument("inputs", nargs="+", help="JSONL files or glob patterns.")
    compare.add_argument("--baseline-label", default="v2.0.0")
    compare.add_argument("--candidate-label", default="v2.2.3")
    compare.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated metric list.")
    compare.add_argument("--out", default=None, help="Optional markdown report path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            run = args.run or datetime.now().strftime("run_%Y%m%d_%H%M%S")
            path = capture_run(
                label=args.label,
                run=run,
                duration_s=args.duration,
                interval_s=args.interval,
                url=args.url,
                out_dir=args.out_dir,
                timeout_s=args.timeout,
            )
            summary = summarize_run(load_records([str(path)]))
            print(f"captured: {path}")
            print(format_run_summary(summary))
            return 0

        if args.command == "compare":
            metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
            summaries = summarize_records(load_records(args.inputs))
            rows = compare_labels(summaries, args.baseline_label, args.candidate_label, metrics)
            report = format_comparison(rows, args.baseline_label, args.candidate_label)
            if args.out:
                output = Path(args.out)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(report + "\n", encoding="utf-8")
                print(f"report: {output}")
            print(report)
            return 0
    except (OSError, URLError, ValueError, RuntimeError) as exc:
        print(f"latency_tool: {exc}", file=sys.stderr)
        return 2
    return 2


def _number_map(record: Dict[str, Any], section: str) -> Dict[str, float]:
    latency = record.get("latency", {})
    if not isinstance(latency, dict):
        return {}
    values = latency.get(section, {})
    if not isinstance(values, dict):
        return {}
    result: Dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result[str(key)] = float(value)
    return result


def _last_counter_snapshot(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    for record in reversed(records):
        latency = record.get("latency", {})
        if not isinstance(latency, dict):
            continue
        counters = latency.get("counters", {})
        if not isinstance(counters, dict):
            continue
        return {
            str(key): int(value)
            for key, value in counters.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return {}


def _expand_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Input file not found: {', '.join(missing)}")
    return paths


def _validate_runtime_label(snapshot: Dict[str, Any], label: str):
    runtime = snapshot.get("runtime", {})
    if not isinstance(runtime, dict):
        return
    version = runtime.get("version")
    if version is None:
        return
    normalized = str(label).removeprefix("v")
    if str(version) != normalized:
        raise ValueError(f"Capture label {label!r} does not match tuner runtime version {version!r}")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
