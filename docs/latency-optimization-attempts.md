# Latency Optimization Attempts

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
