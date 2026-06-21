# Architecture

## System Boundary

`macos-dualbox-aim` runs on a macOS compute machine. It receives game video from
a capture card, runs CoreML/YOLO detection, selects a target, computes a mouse
movement, and sends that movement to the console through KMBox Net.

The console only runs the game and receives hardware mouse input. The project
must not depend on game memory reads, process injection, or client modification.

## Core Base

`src/macos_dualbox_aim/core/` is the shared base for future versions:

| Module | Responsibility |
|---|---|
| `core.capture` | Open/configure capture card, center crop frames, emit latest frame |
| `core.inference` | Realtime capture/inference threads, latency aggregation, callback |
| `core.model_runtime` | CoreML inspection, preprocessing, detector adapters, model probes |
| `core.kmbox` | KMBox UDP protocol, monitor packets, `mouse_move(dx, dy)` |
| `core.hotkey` | KMBox physical mouse trigger/lock/toggle state |
| `core.capture_probe` | Offline capture-card mode and load probes |

New versions should use `core` for hardware and inference. Version directories
should focus on target selection, control algorithms, config, tuner behavior,
and entrypoint assembly.

Old `v1/v2/v3` KMBox/hotkey/capture probe modules and `v5/model_runtime` modules
are compatibility exports. Old inference/capture implementations are historical
surfaces and should not be extended.

## Version Roles

| Version | Role |
|---|---|
| V1 | Historical PIDF baseline and rollback reference |
| V2 | Kalman target-state experiment |
| V3 | Multi-object tracker experiment |
| V4 | Learned MPID controller |
| V5 | CoreML model-runtime validation on the V4 control chain |

Future versions should not fork capture-card or model-runtime plumbing. If a
shared change is needed, make it in `core` and keep the version layer thin.

## Import Policy

- Entrypoints import explicit version packages, e.g. `macos_dualbox_aim.v5`.
- Shared hardware/inference imports come from `macos_dualbox_aim.core`.
- Package root should not grow behavior modules such as `config.py`,
  `controller.py`, or `kmbox.py`.
- Do not make `core` depend on version packages.

## Promotion Rule

Experimental behavior starts in the active major version. Promote to `core` only
after manual validation or repeated version reuse proves it is stable. Promotion
means the API becomes a base for future versions and should be changed
conservatively.
