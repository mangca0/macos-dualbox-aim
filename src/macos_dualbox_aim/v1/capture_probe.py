import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


PIXEL_FORMAT_FOURCC = {
    "MJPEG": "MJPG",
    "MJPG": "MJPG",
    "YUY2": "YUY2",
    "RGB3": "RGB3",
    "BGR3": "BGR3",
    "UYVY": "UYVY",
}
DEFAULT_FORMATS = ("MJPEG", "YUY2", "UYVY", "RGB3", "BGR3")
DEFAULT_FPS_VALUES = (60, 120, 240)
DEFAULT_RESOLUTIONS = ((1920, 1080),)


@dataclass(frozen=True)
class CaptureMode:
    device: int
    width: int
    height: int
    fps: int
    pixel_format: str
    samples: int = 120
    warmup: int = 10
    backend: Optional[int] = None
    load: str = "none"
    load_ms: float = 0.0
    load_placement: str = "inline"


@dataclass(frozen=True)
class CaptureProbeResult:
    requested: Dict[str, Any]
    actual: Dict[str, Any]
    opened: bool
    ok_frames: int
    grab_failures: int
    retrieve_failures: int
    open_ms: float
    configure_ms: float
    duration_ms: float
    avg_grab_ms: float
    p95_grab_ms: float
    avg_retrieve_ms: float
    p95_retrieve_ms: float
    avg_read_ms: float
    p95_read_ms: float
    avg_frame_interval_ms: float
    p95_frame_interval_ms: float
    effective_fps: float
    avg_load_ms: float
    p95_load_ms: float
    load_iterations: int
    avg_load_period_ms: float
    p95_load_period_ms: float
    captured_at: str

    def to_record(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "requested": self.requested,
            "actual": self.actual,
            "opened": self.opened,
            "ok_frames": self.ok_frames,
            "grab_failures": self.grab_failures,
            "retrieve_failures": self.retrieve_failures,
            "open_ms": self.open_ms,
            "configure_ms": self.configure_ms,
            "duration_ms": self.duration_ms,
            "avg_grab_ms": self.avg_grab_ms,
            "p95_grab_ms": self.p95_grab_ms,
            "avg_retrieve_ms": self.avg_retrieve_ms,
            "p95_retrieve_ms": self.p95_retrieve_ms,
            "avg_read_ms": self.avg_read_ms,
            "p95_read_ms": self.p95_read_ms,
            "avg_frame_interval_ms": self.avg_frame_interval_ms,
            "p95_frame_interval_ms": self.p95_frame_interval_ms,
            "effective_fps": self.effective_fps,
            "avg_load_ms": self.avg_load_ms,
            "p95_load_ms": self.p95_load_ms,
            "load_iterations": self.load_iterations,
            "avg_load_period_ms": self.avg_load_period_ms,
            "p95_load_period_ms": self.p95_load_period_ms,
        }


def probe_capture_mode(
    mode: CaptureMode,
    *,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    load_worker_factory: Optional[Callable[[CaptureMode, Callable[[], float], Callable[[float], None]], Any]] = None,
) -> CaptureProbeResult:
    if mode.samples <= 0:
        raise ValueError("samples must be positive")
    if mode.warmup < 0:
        raise ValueError("warmup must be non-negative")
    _validate_load(mode.load, mode.load_ms, mode.load_placement)

    requested = _requested_record(mode)
    open_start = clock()
    capture = _open_capture(capture_factory, mode)
    open_ms = (clock() - open_start) * 1000.0

    configure_start = clock()
    _configure_capture(capture, mode)
    configure_ms = (clock() - configure_start) * 1000.0
    opened = bool(capture.isOpened())
    actual = _actual_record(capture)

    grab_values: List[float] = []
    retrieve_values: List[float] = []
    read_values: List[float] = []
    interval_values: List[float] = []
    load_values: List[float] = []
    load_period_values: List[float] = []
    load_iterations = 0
    inline_load_iterations = 0
    grab_failures = 0
    retrieve_failures = 0
    successful_frames = 0
    measured_frames = 0
    first_frame_at: Optional[float] = None
    last_frame_at: Optional[float] = None
    max_attempts = max(10, (mode.samples + mode.warmup) * 4)

    load_worker = None
    if mode.load_placement == "thread" and _load_enabled(mode):
        factory = load_worker_factory or _start_load_worker
        load_worker = factory(mode, clock, sleeper)

    try:
        if opened:
            attempts = 0
            while measured_frames < mode.samples and attempts < max_attempts:
                attempts += 1
                grab_start = clock()
                grabbed = bool(capture.grab())
                grab_end = clock()
                grab_ms = (grab_end - grab_start) * 1000.0
                if not grabbed:
                    grab_failures += 1
                    continue

                retrieve_start = clock()
                retrieved, _frame = capture.retrieve()
                retrieve_end = clock()
                retrieve_ms = (retrieve_end - retrieve_start) * 1000.0
                if not retrieved:
                    retrieve_failures += 1
                    continue

                load_ms = 0.0
                if mode.load_placement == "inline":
                    load_ms = _run_load(mode, clock=clock, sleeper=sleeper)
                    if _load_enabled(mode):
                        inline_load_iterations += 1
                successful_frames += 1
                if successful_frames <= mode.warmup:
                    if mode.load_placement == "inline":
                        load_values.append(load_ms)
                    last_frame_at = retrieve_end
                    continue
                measured_frames += 1
                if first_frame_at is None:
                    first_frame_at = retrieve_end
                if last_frame_at is not None:
                    interval_values.append(max(0.0, (retrieve_end - last_frame_at) * 1000.0))
                last_frame_at = retrieve_end
                grab_values.append(grab_ms)
                retrieve_values.append(retrieve_ms)
                read_values.append(grab_ms + retrieve_ms)
                if mode.load_placement == "inline":
                    load_values.append(load_ms)
    finally:
        if load_worker is not None:
            load_stats = load_worker.stop()
            load_values.extend(load_stats.get("durations_ms", []))
            load_period_values.extend(load_stats.get("periods_ms", []))
            load_iterations = int(load_stats.get("iterations", 0))
        capture.release()

    if first_frame_at is not None and last_frame_at is not None:
        duration_ms = max(0.0, (last_frame_at - first_frame_at) * 1000.0)
    else:
        duration_ms = 0.0
    effective_fps = measured_frames * 1000.0 / duration_ms if duration_ms > 0.0 else 0.0

    return CaptureProbeResult(
        requested=requested,
        actual=actual,
        opened=opened,
        ok_frames=measured_frames,
        grab_failures=grab_failures,
        retrieve_failures=retrieve_failures,
        open_ms=open_ms,
        configure_ms=configure_ms,
        duration_ms=duration_ms,
        avg_grab_ms=_mean(grab_values),
        p95_grab_ms=_percentile(grab_values, 95.0),
        avg_retrieve_ms=_mean(retrieve_values),
        p95_retrieve_ms=_percentile(retrieve_values, 95.0),
        avg_read_ms=_mean(read_values),
        p95_read_ms=_percentile(read_values, 95.0),
        avg_frame_interval_ms=_mean(interval_values),
        p95_frame_interval_ms=_percentile(interval_values, 95.0),
        effective_fps=effective_fps,
        avg_load_ms=_mean(load_values),
        p95_load_ms=_percentile(load_values, 95.0),
        load_iterations=load_iterations or inline_load_iterations,
        avg_load_period_ms=_mean(load_period_values),
        p95_load_period_ms=_percentile(load_period_values, 95.0),
        captured_at=datetime.now(timezone.utc).isoformat(),
    )


def build_modes(
    *,
    device: int,
    pixel_formats: Sequence[str],
    fps_values: Sequence[int],
    resolutions: Sequence[Tuple[int, int]],
    samples: int = 120,
    warmup: int = 10,
    backend: Optional[int] = None,
    load: str = "none",
    load_ms: float = 0.0,
    load_placement: str = "inline",
) -> List[CaptureMode]:
    modes: List[CaptureMode] = []
    for pixel_format in pixel_formats:
        for fps in fps_values:
            for width, height in resolutions:
                modes.append(CaptureMode(
                    device=device,
                    width=width,
                    height=height,
                    fps=int(fps),
                    pixel_format=pixel_format,
                    samples=samples,
                    warmup=warmup,
                    backend=backend,
                    load=load,
                    load_ms=load_ms,
                    load_placement=load_placement,
                ))
    return modes


def parse_resolution(value: str) -> Tuple[int, int]:
    normalized = value.lower().strip()
    if "x" not in normalized:
        raise ValueError("resolution must be WIDTHxHEIGHT")
    width_raw, height_raw = normalized.split("x", 1)
    try:
        width = int(width_raw)
        height = int(height_raw)
    except ValueError as exc:
        raise ValueError("resolution must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    return width, height


def format_results_markdown(results: Sequence[CaptureProbeResult]) -> str:
    lines = [
        "# Capture mode probe",
        "",
        "| requested | actual | backend | ok | failures | avg read | grab | retrieve | interval | effective fps | avg load | load iters | load period |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        requested = result.requested
        actual = result.actual
        requested_label = (
            f"{requested['pixel_format']} {requested['width']}x{requested['height']} "
            f"{requested['fps']}fps"
        )
        load_label = _load_label(
            str(requested.get("load", "none")),
            float(requested.get("load_ms", 0.0)),
            str(requested.get("load_placement", "inline")),
        )
        if load_label:
            requested_label = f"{requested_label} / {load_label}"
        actual_label = (
            f"{actual.get('fourcc') or '--'} {actual.get('width', 0):.0f}x"
            f"{actual.get('height', 0):.0f} {actual.get('fps', 0):.1f}fps"
        )
        failures = result.grab_failures + result.retrieve_failures
        lines.append(
            f"| {requested_label} | {actual_label} | {actual.get('backend') or '--'} | "
            f"{result.ok_frames} | {failures} | {_fmt(result.avg_read_ms)} | "
            f"{_fmt(result.avg_grab_ms)} | {_fmt(result.avg_retrieve_ms)} | "
            f"{_fmt(result.avg_frame_interval_ms)} | {_fmt(result.effective_fps)} | "
            f"{_fmt(result.avg_load_ms)} | {result.load_iterations} | {_fmt(result.avg_load_period_ms)} |"
        )
    return "\n".join(lines)


def write_results_jsonl(path: str | Path, results: Sequence[CaptureProbeResult]):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_record(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe capture device mode latency.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    parser.add_argument("--fps", default=",".join(str(value) for value in DEFAULT_FPS_VALUES))
    parser.add_argument("--resolutions", default=",".join(f"{w}x{h}" for w, h in DEFAULT_RESOLUTIONS))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backend", choices=("auto", "avfoundation"), default="auto")
    parser.add_argument("--load", choices=("none", "sleep", "busy"), default="none")
    parser.add_argument("--load-ms", type=float, default=0.0)
    parser.add_argument("--load-placement", choices=("inline", "thread"), default="inline")
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backend = _parse_backend(args.backend)
        modes = build_modes(
            device=args.device,
            pixel_formats=_parse_csv(args.formats),
            fps_values=[int(value) for value in _parse_csv(args.fps)],
            resolutions=[parse_resolution(value) for value in _parse_csv(args.resolutions)],
            samples=args.samples,
            warmup=args.warmup,
            backend=backend,
            load=args.load,
            load_ms=args.load_ms,
            load_placement=args.load_placement,
        )
        results = [probe_capture_mode(mode) for mode in modes]
        report = format_results_markdown(results)
        print(report)
        if args.out_jsonl:
            write_results_jsonl(args.out_jsonl, results)
        if args.out_md:
            output = Path(args.out_md)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError) as exc:
        print(f"capture_probe: {exc}")
        return 2


def _open_capture(capture_factory: Callable[..., Any], mode: CaptureMode):
    if mode.backend is None:
        return capture_factory(mode.device)
    return capture_factory(mode.device, mode.backend)


def _configure_capture(capture: Any, mode: CaptureMode):
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_fourcc_code(mode.pixel_format)))
    capture.set(cv2.CAP_PROP_FPS, mode.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)


def _requested_record(mode: CaptureMode) -> Dict[str, Any]:
    return {
        "device": mode.device,
        "width": mode.width,
        "height": mode.height,
        "fps": mode.fps,
        "pixel_format": mode.pixel_format,
        "fourcc": _fourcc_code(mode.pixel_format),
        "buffersize": 1,
        "backend": mode.backend,
        "samples": mode.samples,
        "warmup": mode.warmup,
        "load": mode.load,
        "load_ms": mode.load_ms,
        "load_placement": mode.load_placement,
    }


def _actual_record(capture: Any) -> Dict[str, Any]:
    try:
        backend = str(capture.getBackendName())
    except Exception:
        backend = ""
    return {
        "backend": backend,
        "width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "fourcc": _fourcc_to_string(capture.get(cv2.CAP_PROP_FOURCC)),
        "buffersize": float(capture.get(cv2.CAP_PROP_BUFFERSIZE)),
    }


def _parse_backend(value: str) -> Optional[int]:
    if value == "auto":
        return None
    if value == "avfoundation":
        return int(cv2.CAP_AVFOUNDATION)
    raise ValueError(f"Unsupported backend: {value}")


def _validate_load(load: str, load_ms: float, load_placement: str):
    if load not in {"none", "sleep", "busy"}:
        raise ValueError("load must be one of none, sleep, busy")
    if load_placement not in {"inline", "thread"}:
        raise ValueError("load-placement must be one of inline, thread")
    if load_ms < 0.0:
        raise ValueError("load-ms must be non-negative")
    if load == "none" and load_ms > 0.0:
        raise ValueError("load-ms requires --load sleep or --load busy")


def _run_load(
    mode: CaptureMode,
    *,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> float:
    if not _load_enabled(mode):
        return 0.0
    target_s = mode.load_ms / 1000.0
    start = clock()
    if mode.load == "sleep":
        sleeper(target_s)
    elif mode.load == "busy":
        while clock() - start < target_s:
            pass
    else:
        raise ValueError("load must be one of none, sleep, busy")
    return max(0.0, (clock() - start) * 1000.0)


class _LoadWorker:
    def __init__(
        self,
        mode: CaptureMode,
        *,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ):
        self.mode = mode
        self.clock = clock
        self.sleeper = sleeper
        self.stop_event = threading.Event()
        self.durations_ms: List[float] = []
        self.periods_ms: List[float] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        return {
            "iterations": len(self.durations_ms),
            "durations_ms": list(self.durations_ms),
            "periods_ms": list(self.periods_ms),
        }

    def _run(self):
        previous_start: Optional[float] = None
        while not self.stop_event.is_set():
            start = self.clock()
            if previous_start is not None:
                self.periods_ms.append(max(0.0, (start - previous_start) * 1000.0))
            previous_start = start
            self.durations_ms.append(_run_load(self.mode, clock=self.clock, sleeper=self.sleeper))
            if self.mode.load == "sleep":
                continue
            if not self.stop_event.is_set():
                self.sleeper(0.0)


def _start_load_worker(
    mode: CaptureMode,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> _LoadWorker:
    return _LoadWorker(mode, clock=clock, sleeper=sleeper)


def _load_enabled(mode: CaptureMode) -> bool:
    return mode.load != "none" and mode.load_ms > 0.0


def _load_label(load: str, load_ms: float, load_placement: str) -> str:
    if load == "none" or load_ms <= 0.0:
        return ""
    return f"{load} {_fmt(load_ms)}ms {load_placement}"


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _fourcc_code(pixel_format: str) -> str:
    return PIXEL_FORMAT_FOURCC.get(pixel_format.upper(), pixel_format.upper())


def _fourcc_to_string(value: float) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    chars = []
    for shift in (0, 8, 16, 24):
        byte = (code >> shift) & 0xFF
        if byte:
            chars.append(chr(byte))
    return "".join(chars)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
