# Releases

Concise release notes. Architecture and operational docs live in README and the
focused files under `docs/`. Latency experiment rationale lives in
`docs/latency-optimization-attempts.md`; stable conclusions live in
`docs/latency-findings.md`.

## core-runtime-base

- Promoted the validated V5 architecture into `src/macos_dualbox_aim/core/` as the future shared base.
- Added core modules for capture-card setup/cropping, realtime inference orchestration, Core ML model runtime, KMBox transport, and KMBox hotkey monitoring.
- Kept V1-V5 import paths compatible by turning repeated KMBox, hotkey, capture probe, and V5 model-runtime modules into compatibility exports.
- Switched V5 realtime inference away from inheriting V1 inference; it now uses the core realtime base with a V5-compatible wrapper.
- Old inference/capture files remain historical compatibility surfaces and should not receive new feature work.

## v5.0.0-runtime-draft

- Added the V5 model-runtime architecture boundary for ONNX-to-CoreML conversion, Core ML interface contracts, and detector adapters.
- Added `YoloV8TensorAdapter` for raw YOLOv8 outputs shaped `[1, 4 + classes, anchors]` or `[1, anchors, 4 + classes]`.
- Added Core ML contract inspection that classifies existing `ImageType` NMS models and direct ONNX-style tensor models.
- Added `scripts/convert_onnx_to_coreml.py`, which creates FP32 check and FP16 fast Core ML packages from one ONNX source.
- Documented the model-runtime contract in `docs/model-runtime.md`.
- Confirmed the converted `cs2_fp16` FP32 check and FP16 fast packages are usable on live capture-card frames.
- Added `scripts/main_v5.py`, which runs V5 Core ML model inference while reusing the V4 controller, KMBox, hotkey, and tuner stack.
- V1-V4 runtime files and config schemas remain unchanged.

## v3.0.0

- Added independent V3 entrypoint, config, package path, and tuner.
- Ported the C++ multi-object tracker into Python with the same `DetectionObject` shape, 7-state Kalman model, IoU/distance/shape matching cost, Hungarian assignment, generate confirmation threshold, and terminate deletion threshold.
- V3 keeps only the existing capture card, CoreML inference, and KMBox transport. After CoreML boxes are produced, V3 runs the C++-style multi-object tracker and sends the first confirmed track center error directly to KMBox.
- V1 and V2 runtime files and config schemas remain unchanged.

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
