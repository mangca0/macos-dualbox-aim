# Latency Findings

Stable conclusions only. Raw captures and generated reports stay in
`latency_runs/`; version summaries stay in `docs/releases.md`.

## Bottlenecks

- Runtime latency is dominated by capture and CoreML inference.
- In the V1.1.0 comparison, `capture_read_ms` was about 45% of
  `read_included_total_ms`, and `coreml_ms` was about 42%.
- Target selection, PID/controller work, and KMBox send timing have been
  sub-millisecond in collected runs. Do not start latency work there unless new
  measurements contradict this.
- Micro-optimizing postprocess, target ranking, queue size, or latency snapshot
  aggregation did not produce meaningful aim improvement.

## Capture

- At 1920x1080, the tested AVFoundation/OpenCV path behaved like about 120fps
  in practice, even when 240fps was requested.
- Color format switching did not materially improve capture latency on the
  tested device. MJPEG, YUY2, UYVY, RGB3, and BGR3 were within roughly
  0.1-0.2 ms in the V1.2.0 matrix.
- In diagnostic builds, capture read cost was mostly `capture_grab_ms`, which
  means waiting for the backend/device dominated over post-retrieve processing.
- OpenCV-reported properties are not enough. Prefer measured
  `avg_frame_interval_ms` and `effective_fps` from `scripts/capture_probe.py`.

## Probe Interpretation

- Standalone 1080p120 capture probe baseline was roughly 8.3 ms frame interval.
- Inline load probes are intentionally serial and do not model the threaded main
  runtime.
- `sleep9 thread` stayed near the standalone baseline, so a second thread or
  9 ms cadence alone was not the problem.
- `busy9 thread` degraded capture severely, proving Python CPU/GIL contention can
  disturb capture timing. It is a stress signal, not a direct CoreML proxy.
- A lower `avg_grab_ms` under inline load can mean the next frame was already
  buffered, not that capture got faster.

## Direction

- Keep capture diagnostics outside the main runtime first, or isolate them in a
  short-lived diagnostic version.
- Next useful capture experiment: separate-process capture or real-runtime
  overlap diagnostics to test whether isolating capture from Python/CoreML
  scheduling restores the standalone cadence.
- Keep model-runtime optimization ahead of controller micro-optimization while
  capture/CoreML dominate.
