# Runtime Pipeline

## Data Flow

```text
capture card
  -> center crop
  -> CoreML detector
  -> detection adapter
  -> target selection
  -> controller
  -> KMBox relative mouse move
```

V5 and later should use `core.capture`, `core.inference`, and
`core.model_runtime` for the first three stages. Version code owns target
selection and control.

## Coordinate Rules

- Inference usually runs on a screen-center crop, not the full screen.
- `crop_offset` is the crop top-left corner relative to the full screen.
- Detections may be normalized `[cx, cy, w, h]` or pixel `[x1, y1, x2, y2]`.
  Confirm format at the adapter/consumer boundary.
- Controllers need target offset relative to the aim reference, not absolute
  screen coordinates.
- V1 through V6.1 use screen center as the aim reference. V6.2 is the first
  experiment that detects a color crosshair in the current crop and uses that
  point as the aim reference; if the crosshair is not found, V6.2 sends no
  movement for that frame.

When changing target selection, trace fields such as `_current_aim_x/y`,
`target.screen_x/y`, and `target.aim_x/y`; historical code has mixed absolute
coordinates and relative offsets.

## Latency Fields

Common realtime fields:

| Field | Meaning |
|---|---|
| `capture_read_ms` | Time spent reading a frame from OpenCV |
| `crop_ms` | Time spent center-cropping |
| `queue_wait_ms` | Time from captured frame to inference start |
| `preprocess_ms` | Detector preprocessing |
| `coreml_ms` | CoreML model execution |
| `postprocess_ms` | Adapter decode/NMS work |
| `inference_ms` | End-to-end detector call inside inference loop |
| `detection_callback_ms` | Target/control callback time |
| `target_select_ms` | Version target selection |
| `pid_ms` | Controller update time |
| `kmbox_send_ack_ms` | KMBox send/ack timing when measured |

Use `scripts/capture_probe.py` for capture-card diagnostics before changing main
runtime scheduling. Use `scripts/latency_tool.py` for tuner snapshot capture and
cross-run comparisons.

## KMBox and Hotkey Rules

- `mouse_move(dx, dy)` is relative movement. Do not convert it to absolute
  coordinates.
- KMBox uses a UDP custom protocol. Packet size, endian order, and command
  constants matter.
- Hotkeys are KMBox physical mouse monitor events, not macOS global input.
- `trigger_button_secondary` is OR semantics: primary or secondary triggers aim.
- The lock key temporarily disables aim; it is not another trigger button.
- Do not add implicit destructive port cleanup such as killing processes on the
  monitor port.
