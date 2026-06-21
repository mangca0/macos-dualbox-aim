# Latency Experiment Index

This file is intentionally short. Use it to recover the reasoning path; use
`docs/latency-findings.md` for stable conclusions and `latency_runs/` for raw
samples/reports.

## 2026-06-20: V1.2.3 Main Runtime Rollback

Status: released rollback.

Decision: restore the V1.0 `capture.read()` path in the main runtime and keep
grab/retrieve/frame-interval diagnostics in standalone tools. Observation-only
instrumentation should not live in the clean runtime unless it is part of a
short-lived diagnostic version.

Reason: V1.1-V1.2 exploration did not produce a practical latency improvement,
and the user wanted the stable behavior restored.

## 2026-06-20: V1.2.2 Threaded Capture Load Probe

Status: experiment-support release.

Decision: add `--load-placement inline|thread` to distinguish serial consumer
cadence from concurrent same-process load.

Measurements:

| Probe | Key result | Interpretation |
|---|---|---|
| `busy9 thread` | `avg_frame_interval_ms` around 21.36 ms, `effective_fps` around 47 | Python CPU/GIL contention can severely disturb capture |
| `sleep9 thread` | `avg_frame_interval_ms` around 8.57 ms, `effective_fps` around 117 | Thread presence and 9 ms cadence alone are not enough to explain degradation |

Decision: do not treat Python busy-loop load as a direct CoreML proxy. Use it
only as a stress signal.

## 2026-06-20: V1.2.1 Capture Load Probe

Status: experiment-support release.

Decision: add `--load none|sleep|busy` and `--load-ms` to probe how artificial
consumer cadence or CPU load changes capture timing outside the main runtime.

## 2026-06-20: V1.2.0 Capture Mode Probe

Status: experiment-support release.

Decision: add `scripts/capture_probe.py` for matrix testing pixel format, FPS,
resolution, backend, actual device properties, frame interval, effective FPS,
and failures. Do not auto-apply probe winners to runtime config; validate any
candidate in the normal main runtime.

## 2026-06-20: V1.1.1 Capture Diagnostics

Status: observation-only release.

Decision: split capture timing into grab/retrieve/frame-interval diagnostics and
expose backend/device properties for measurement. No queueing, model inference,
target selection, or control behavior change.

## 2026-06-20: V1.1.0 Micro-Optimization Attempt

Status: rolled back.

Attempted:

- Default frame queue size `3 -> 1`
- Earlier low-confidence filtering in raw YOLO postprocess
- Squared-distance target ranking
- Less copying in latency snapshot aggregation

Measurement summary from five 60-second runs each:

| Metric | V1.0.0 avg | V1.1.0 attempt avg | Delta |
|---|---:|---:|---:|
| `read_included_total_ms` | 21.808 ms | 21.542 ms | -0.267 ms |
| `program_total_ms` | 11.670 ms | 11.749 ms | +0.079 ms |
| `inference_ms` | 10.104 ms | 9.794 ms | -0.310 ms |
| `queue_wait_ms` | 1.418 ms | 1.812 ms | +0.394 ms |
| `coreml_ms` | 9.191 ms | 8.985 ms | -0.206 ms |

Decision: rollback. The apparent changes were sub-millisecond and not
practically meaningful for aim behavior. Keep the measurement tooling.
