# Releases

Concise V1 release notes. Detailed experiment rationale lives in `docs/latency-optimization-attempts.md`; stable conclusions live in `docs/latency-findings.md`.

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
