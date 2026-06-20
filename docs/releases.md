# Releases

Concise release notes. Detailed experiment rationale lives in `docs/latency-optimization-attempts.md`; stable conclusions live in `docs/latency-findings.md`.

## v2.0.0

- Added independent V2 entrypoint, config, package path, and tuner.
- V2 keeps the V1 detection -> PIDF -> KMBox chain but filters target state with a constant-velocity Kalman model.
- Kalman state is `[x, y, vx, vy]`; filtered position and velocity feed the existing PIDF controller.
- V1 runtime files and config schema remain unchanged.

## v1.2.3

- Rolled main runtime capture back to the V1.0 `capture.read()` path.
- Removed main-runtime grab/retrieve/frame-interval diagnostics from tuner snapshots and UI.
- Kept external latency/capture probe scripts for offline exploration.
- No aim control, CoreML model, target selection, PIDF, or KMBox behavior change beyond the capture read-path rollback.

## v1.2.2

- Added `--load-placement inline|thread` to `scripts/capture_probe.py`.
- `inline` preserves V1.2.1 serial capture-then-load behavior.
- `thread` runs load simulation in a background thread while the main thread keeps capturing.
- Markdown and JSONL output now include load iterations and actual load period stats.
- No main runtime behavior change.

## v1.2.1

- Added capture probe load simulation: `--load none|sleep|busy` and `--load-ms`.
- Records requested load mode and measured load timing in JSONL output.
- Markdown output labels load mode in the requested capture mode.
- No main runtime behavior change.

## v1.2.0

- Added standalone capture mode probe: `scripts/capture_probe.py`.
- Supports matrix tests across pixel format, FPS, resolution, and backend.
- Outputs Markdown and JSONL reports with requested mode, actual device properties, capture timing, effective FPS, and failures.
- No main runtime behavior change.

## v1.1.1

- Split capture read timing into `capture_grab_ms`, `capture_retrieve_ms`, and `capture_frame_interval_ms`.
- Added capture backend and actual device property diagnostics to tuner latency snapshots.
- Added capture grab/retrieve failure counters.
- No control behavior change.

## v1.1.0

- Added tuner latency capture and comparison tooling.
- Added runtime version labeling and capture label validation.
- Rolled back attempted runtime micro-optimizations because measured improvements were not practically meaningful.
