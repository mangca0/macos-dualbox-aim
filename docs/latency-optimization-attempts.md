# Latency Optimization Attempts

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
