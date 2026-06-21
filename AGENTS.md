# macos-dualbox-aim Agent Notes

> Keep this under 8KB. Record only rules and pitfalls that are hard to infer
> from code or have caused real confusion.

## Collaboration Rules

- Before any non-trivial task, ask 2-3 questions to confirm scope and acceptance
  criteria, then wait for the user's answer.
- Behavior-changing work must land as an independent updated version first. Do
  not silently change an existing stable path.
- Use `uv` for Python environments, dependencies, scripts, and CLI tools.
- Before finishing code changes, at minimum run syntax checks for changed Python
  files. If config fields changed, state which version consumes them.

## Project Boundary

The macOS machine captures console video, runs CoreML/YOLO detection, selects a
target, computes control output, and sends relative mouse input through KMBox.

- The console only runs the game and receives mouse input.
- Do not depend on game process modification, memory reads, or client injection.

## Architecture

- `src/macos_dualbox_aim/core/` is the future shared base:
  capture card, realtime inference, CoreML model runtime, KMBox transport, and
  KMBox physical hotkey monitoring.
- Future versions should use `core` and implement only target selection,
  control systems, version config, tuner behavior, and entrypoint assembly.
- Old version-local inference/capture files are historical compatibility
  surfaces. Do not extend them.
- Package root should not grow behavior modules such as `config.py`,
  `controller.py`, or `kmbox.py`.
- `core` must not import from `v1/v2/v3/v4/v5`.
- Entrypoints should import explicit version packages, e.g.
  `macos_dualbox_aim.v5`, and shared base modules from
  `macos_dualbox_aim.core`.

Docs:

- `README.md`: commands and doc index
- `docs/architecture.md`: module/version boundaries
- `docs/runtime-pipeline.md`: coordinate flow, latency fields, KMBox/hotkey
- `docs/model-runtime.md`: CoreML model runtime and conversion workflow
- `docs/latency-findings.md`: stable latency conclusions only
- `docs/latency-optimization-attempts.md`: short experiment index
- `docs/releases.md`: concise version history

## Version Roles

- V1: historical PIDF baseline and rollback reference.
- V2: Kalman target-state experiment.
- V3: multi-object tracker experiment.
- V4: learned MPID controller.
- V5: validated CoreML model-runtime path using V4 control.

Promote to `core` only after manual validation or repeated version reuse proves
the behavior stable.

## Aiming Quality

- Static quality: when target and player are still, the aim should reach target
  quickly without overshoot or oscillation.
- Dynamic quality: when target moves relative to player, aim should stay on
  target with minimal lag, respond quickly to trend changes, and still avoid
  overshoot.

## Coordinates

- Inference input is usually a screen-center crop, not the full screen.
- Detection boxes may be normalized `[cx, cy, w, h]` or pixel
  `[x1, y1, x2, y2]`; confirm format at the adapter/consumer boundary.
- `crop_offset` is the crop top-left relative to full-screen top-left.
- Controllers need target offset relative to the aim reference, not absolute
  screen coordinates.
- Current V1-style reference is screen center; no template/color crosshair path
  is in the main chain.
- Before changing target selection, trace fields like `_current_aim_x/y`,
  `target.screen_x/y`, and `target.aim_x/y`; historical code has mixed absolute
  coordinates and relative offsets.

## KMBox

- KMBox communication is a UDP custom protocol. Packet size, endian order, and
  command constants matter.
- Do not change `mouse_move(dx, dy)` relative movement semantics without an
  explicit behavior decision.
- Do not add implicit destructive monitor-port cleanup such as auto `kill -9`.
- Methods outside the main chain, such as `enc_*` or `mouse_move_auto()` if
  reintroduced, need protocol/payload verification before use.

## Hotkey

- Hotkeys come from KMBox physical mouse monitor events, not macOS global input.
- `trigger_button_secondary` is OR semantics: primary or secondary can trigger.
  Changing to AND is a behavior change and must be explicit.
- Lock key temporarily disables aim; it is not an additional trigger key.

## Config

- When changing tuning fields, verify both the version config dataclass and the
  runtime consumer.
- V5 uses `configs/config_v5.json` for model runtime and `configs/config_v4.json`
  for capture/KMBox/hotkey/control.

## Latency

- Capture and CoreML dominate measured latency; controller/KMBox work has been
  sub-millisecond in collected runs.
- Use `scripts/capture_probe.py` before changing capture scheduling.
- Use `scripts/latency_tool.py` for tuner snapshot capture and comparisons.
- Do not treat Python busy-loop load as a direct CoreML proxy; it is only a
  same-process contention stress signal.
