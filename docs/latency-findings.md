# Latency Findings

This file records stable conclusions from V1 latency experiments. Raw captures and per-version experiment history stay in `latency_runs/` and `docs/latency-optimization-attempts.md`.

## Current Bottlenecks

- Total measured runtime latency is dominated by capture and CoreML inference.
- In V1.1.0 comparisons, `capture_read_ms` was about 45% of `read_included_total_ms`, and `coreml_ms` was about 42%.
- Control-side work is not a first-order latency target right now: `target_select_ms`, `pid_ms`, and `kmbox_send_ack_ms` are sub-millisecond in the collected runs.

## Capture Path

- At 1920x1080, AVFoundation appears to deliver about 120fps in practice, even when 240fps is requested.
- Color format switching did not produce meaningful capture latency differences in the V1.2.0 probe. MJPEG, YUY2, UYVY, RGB3, and BGR3 were all within about 0.1-0.2 ms in the tested matrix.
- The capture read cost is mostly `capture_grab_ms`, which points to waiting for the backend/device to deliver a frame, not expensive post-retrieve processing.
- `capture_retrieve_ms` is usually small, but can spike in the main runtime p95. That makes tail latency worth watching even when average retrieve time looks harmless.
- OpenCV/AVFoundation reported properties are not always enough. Prefer measured `capture_frame_interval_ms` and `effective_fps` over only trusting reported FPS.

## Probe Interpretation

- V1.2.0 standalone capture probe showed roughly 8.3 ms frame intervals for 1080p120 capture.
- V1.2.1 inline load probe is intentionally serial: it runs capture, then load, then captures again. It proves that serial capture plus 9 ms work drops effective cadence to about 60fps, but it does not model the threaded main runtime.
- V1.2.2 adds a threaded load probe so capture can run continuously while `sleep` or `busy` load is generated in the same process. Use this before making any main runtime scheduling change.
- V1.2.2 `busy9 thread` measured `avg_frame_interval_ms` around 21.36 ms and `effective_fps` around 47fps. This proves Python same-process busy contention can severely degrade capture, but the degradation is stronger than the main runtime and should not be treated as a direct CoreML model.
- V1.2.2 `sleep9 thread` measured `avg_frame_interval_ms` around 8.57 ms and `effective_fps` around 117fps. That makes simple background-thread presence or a 9 ms timing cadence unlikely to be the main cause of capture degradation.
- A lower `avg_grab_ms` under inline load does not mean capture got faster. It can mean the next frame was already waiting in the backend buffer after the artificial load.
- Threaded load probe interpretation should compare no-load, `sleep9 thread`, and `busy9 thread` against the main runtime `capture_frame_interval_ms`/`capture_read_ms`.

## Direction

- Do not spend more time on color format switching unless a new device/backend shows different measured behavior.
- Do not use Python busy-loop load as a direct proxy for CoreML. It is useful only as a stress signal for CPU/GIL contention.
- Next capture-side experiment: add a real-runtime overlap diagnostic or separate-process capture probe to decide whether isolating capture from Python/CoreML scheduling pressure restores the standalone 8.3 ms cadence.
- Keep postprocess and controller micro-optimizations deprioritized until capture/CoreML scheduling and model runtime options are exhausted.
