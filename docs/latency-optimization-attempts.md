# Latency Optimization Attempts

## 2026-06-20: V1.2.3 main runtime rollback

### Status

Released as a rollback version.

### Goal

Return the main aimbot runtime to the V1.0 capture behavior after V1.1-V1.2 exploration failed to produce a practical latency improvement.

### Changes

- Restored `RealtimeInference._capture_loop()` to use OpenCV `capture.read()` instead of explicit `grab()`/`retrieve()`.
- Removed main-runtime capture backend/property diagnostics from latency snapshots.
- Removed capture grab/retrieve/frame-interval rows and failure counters from the Web tuner UI.
- Kept standalone `scripts/capture_probe.py` and `scripts/latency_tool.py` for offline diagnosis.

### Decision

Do not carry observation-only capture instrumentation inside the main runtime when the user wants a clean rollback. Future latency experiments should stay outside the runtime first, or be added behind an explicit short-lived diagnostic version.

## 2026-06-20: V1.2.2 threaded capture load probe release

### Status

Released as an experiment-support version.

### Goal

Separate two possible explanations for the main runtime capture slowdown:

- Serial consumer cadence after capture makes the next frame wait.
- Concurrent same-process load affects AVFoundation/OpenCV capture timing through CPU/GIL/runtime scheduling pressure.

### Changes

- Added `--load-placement inline|thread` to `scripts/capture_probe.py`.
- `inline` keeps the V1.2.1 capture-then-load behavior for comparison.
- `thread` starts a background load worker while the main thread keeps capturing.
- Markdown and JSONL output include load iterations plus average load duration and average load period; JSONL also records p95 load duration and p95 load period.

### Decision

Run the V1.2.2 `thread` probe before changing the main runtime. If `busy9 thread` keeps `avg_frame_interval_ms` near the standalone 8.3 ms baseline, deprioritize simple CPU/GIL contention and look at CoreML/AVFoundation/runtime interaction. If it degrades toward the main runtime 9.7-10 ms capture interval, investigate thread scheduling and capture/inference handoff before touching postprocess or controller code.

### Measurement

User ran `MJPEG 1920x1080 120fps --load busy --load-ms 9 --load-placement thread`.

| metric | value |
|---|---:|
| `avg_read_ms` | 21.356 |
| `avg_grab_ms` | 11.994 |
| `avg_retrieve_ms` | 9.363 |
| `avg_frame_interval_ms` | 21.358 |
| `effective_fps` | 47.156 |
| `avg_load_ms` | 9.000 |
| `load_iterations` | 491 |
| `avg_load_period_ms` | 9.004 |

This reproduces a strong same-process contention effect, but it is stronger than the main runtime capture slowdown. Treat it as evidence that Python busy-thread contention can severely disturb capture timing, not as proof that CoreML's real load has the same mechanism.

User then ran the matching `sleep9 thread` control.

| metric | value |
|---|---:|
| `avg_read_ms` | 8.558 |
| `avg_grab_ms` | 7.865 |
| `avg_retrieve_ms` | 0.693 |
| `avg_frame_interval_ms` | 8.565 |
| `effective_fps` | 117.4 |
| `avg_load_ms` | 10.813 |
| `load_iterations` | 157 |
| `avg_load_period_ms` | 10.820 |

The sleep-thread control stays close to the standalone capture baseline. This suggests the severe `busy9 thread` degradation is not caused by simply having a second thread or a 9 ms cadence in the process; it is specifically tied to active CPU/GIL contention from the Python busy loop.

### Next

Do not use Python busy-loop load as a direct proxy for CoreML. The next useful distinction is real-runtime scheduling: either add a capture/inference overlap diagnostic to the main runtime, or add a separate-process capture probe to see whether isolating capture from Python/CoreML scheduling pressure restores the 8.3 ms capture cadence.

## 2026-06-20: V1.2.1 capture load probe release

### Status

Released as an experiment-support version.

### Goal

Explain why standalone capture probe results around 8.3 ms can differ from main runtime `capture_read_ms` around 9.7 ms under CoreML inference load.

### Changes

- Added `--load none|sleep|busy` and `--load-ms` to `scripts/capture_probe.py`.
- The probe records requested load mode and measured `avg_load_ms`/`p95_load_ms` in JSONL.
- Markdown output includes load information in the requested mode label.

### Decision

Use `--load sleep --load-ms 9` first to test consumer cadence, then `--load busy --load-ms 9` if CPU scheduling pressure needs a rough local proxy. Keep this outside the main runtime until measurements show a clear next change.

## 2026-06-20: V1.2.0 capture mode probe release

### Status

Released as an experiment-support version.

### Goal

Support the next capture backend/device-mode optimization pass without changing aim runtime behavior. The probe is meant to identify which requested color format, FPS, resolution, and backend combination actually produces lower capture latency on the current hardware.

### Changes

- Added `scripts/capture_probe.py` for matrix testing capture modes outside the main aimbot process.
- Records requested mode, actual backend/FPS/resolution/FourCC/buffer size, open/configure time, `grab`/`retrieve` timing, frame interval, effective FPS, and failure counters.
- Supports Markdown and JSONL output for later comparison.
- Exports `CaptureMode`, `CaptureProbeResult`, and `probe_capture_mode` from the V1 package.

### Decision

Do not auto-apply probe winners to `configs/config_v1.json`. Use probe output to choose one candidate mode, then validate it in the normal V1 tuner latency capture.

## 2026-06-20: V1.1.1 capture diagnostics release

### Status

Released as an observation-only version.

### Goal

Improve the latency monitor before the next optimization attempt, focused on the capture path because `capture_read_ms` is one of the two dominant contributors to total latency.

### Changes

- Split OpenCV capture timing into `capture_grab_ms` and `capture_retrieve_ms`.
- Added `capture_frame_interval_ms` to show the effective frame cadence delivered by the capture backend.
- Added capture backend and actual device property reporting: resolution, FPS, FourCC, and buffer size.
- Added `capture_grab_failures` and `capture_retrieve_failures` counters.
- Exposed the new capture diagnostics through tuner `/api/config`, the Web tuner latency panel, and `scripts/latency_tool.py` comparisons.

### Decision

Do not change queueing, model inference, target selection, or control behavior in V1.1.1. Use this version to collect cleaner data for capture backend/device-mode experiments.

## 2026-06-20: V1.1.0 micro-optimization attempt

### Status

Rolled back.

### Goal

Reduce average latency shown by the V1 tuner, using the existing tuner latency fields as the measurement source.

### Attempted changes

- Default inference frame queue size changed from `3` to `1`.
- Raw YOLO postprocess filtered low-confidence candidates before bbox decode/NMS.
- Target ranking replaced `sqrt(distance_sq) / weight` with `distance_sq / weight^2`.
- Latency snapshot aggregation reduced copying/repeated iteration.

### Measurement

The user ran five 60-second captures for V1.0.0 and five 60-second captures for the V1.1.0 attempt through `scripts/latency_tool.py`.

Summary from the generated comparison:

| Metric | V1.0.0 avg | V1.1.0 attempt avg | Delta |
|---|---:|---:|---:|
| `read_included_total_ms` | 21.808 ms | 21.542 ms | -0.267 ms |
| `program_total_ms` | 11.670 ms | 11.749 ms | +0.079 ms |
| `inference_ms` | 10.104 ms | 9.794 ms | -0.310 ms |
| `queue_wait_ms` | 1.418 ms | 1.812 ms | +0.394 ms |
| `coreml_ms` | 9.191 ms | 8.985 ms | -0.206 ms |

The apparent changes were sub-millisecond and not practically meaningful for aim behavior.

### Decision

Rollback the runtime micro-optimizations. Keep the latency analysis tooling and runtime version labeling because those improve future measurement quality.

### Lessons

- Micro-optimizing postprocess, target ranking, and latency aggregation is not a useful first-order latency path for this project.
- `capture_read_ms` and `coreml_ms` dominate the pipeline and should be the first targets for any future latency work.
- Queue tuning did not produce the expected average latency reduction because the inference loop already drains to the latest queued frame.
- Future experiments should include runtime version metadata in captured samples and should reject mislabeled captures.

### Next recommended directions

- Investigate capture backend/device settings that reduce `capture_read_ms`.
- Investigate model/runtime options that reduce `coreml_ms`.
- Add deeper capture timing if needed before changing behavior again.
