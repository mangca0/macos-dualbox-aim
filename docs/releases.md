# Releases

Concise release notes. Architecture and operational docs live in README and the
focused files under `docs/`. Latency experiment rationale lives in
`docs/latency-optimization-attempts.md`; stable conclusions live in
`docs/latency-findings.md`.

## v6.3.0

- Added isolated V6.3 experiment files: `scripts/main_v63.py`, `configs/config_v63.json`, and `macos_dualbox_aim.v63`.
- V6.3 keeps V6.2 capture, CoreML, crosshair reference, tracker, MPID, hotkey, tuner, and KMBox behavior while adding model-aware target class selection.
- Model load now inspects the CoreML package and same-name sidecar metadata to resolve class count and class names; if names are unavailable, V6.3 falls back to `class_0`, `class_1`, and so on.
- The tuner now shows all model classes and lets the runtime live-select which classes remain eligible for tracking and aim.
- V6.3 is the current recommended runtime and closes the present development stage; older versions remain available as rollback references or isolated experiments.

## v6.2.0

- Added isolated V6.2 experiment files: `scripts/main_v62.py`, `configs/config_v62.json`, and `macos_dualbox_aim.v62`.
- V6.2 keeps V6.1 capture, CoreML, tracker, MPID, hotkey, tuner, and KMBox behavior while changing the aim reference from screen center to a color-detected crosshair.
- Added crosshair config fields for HSV/RGB matching, search radius, and minimum pixel count. If the crosshair is not found, V6.2 stops output for that frame instead of falling back to screen center.
- `scripts/main_v62.py` passes the current cropped frame into target selection so crosshair detection and CoreML detections use the same crop coordinate space.

## v6.1.0

- Added isolated V6.1 experiment files: `scripts/main_v61.py`, `configs/config_v61.json`, and `macos_dualbox_aim.v61`.
- V6.1 keeps V6 capture, CoreML, tracker, prediction, ramp, and KMBox behavior while adding an adaptive integral gate to the MPID controller.
- New V6.1 control fields: `pid_integral_gate_enabled`, `pid_integral_gate_threshold`, `pid_integral_gate_rate`, and `target_jump_reset`.
- `pid_integral_gate_enabled` defaults on so the `Ki` contribution is suppressed on far errors and gradually opens near target; `target_jump_reset` preserves V6's default 40 px reset threshold but makes it tunable.

## v6.0.0

- Added `scripts/main_v6.py`, `configs/config_v6.json`, and `macos_dualbox_aim.v6`.
- V6 uses `core` for shared capture, CoreML, KMBox, and hotkey foundations, but keeps its version behavior in V6-owned modules.
- V6 reads only `configs/config_v6.json`; model path, class count, confidence threshold, IoU threshold, control fields, and tracker fields have one config source.
- V6 combines CoreML detections, learned MPID control, and Kalman+Hungarian multi-object tracking with first confirmed track selection.
- Added V6 tracker config fields consumed by `AimbotV6`: `tracker_generate`, `tracker_terminate`, `tracker_vx_noise`, `tracker_vy_noise`, `tracker_w_noise`, `tracker_h_noise`, and `tracker_r_std`.

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
