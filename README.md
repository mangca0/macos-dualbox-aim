# macos-dualbox-aim

macOS dual-box aim runtime: the game console only renders the game and receives
hardware mouse input; the macOS machine captures video, runs CoreML/YOLO
detection, selects targets, computes control output, and sends relative mouse
movement through KMBox Net.

The project does not modify the game process, read game memory, or inject into
the game client.

## Quick Start

```bash
uv sync
uv run python scripts/main_v5.py
```

Current practical entrypoints:

| Command | Purpose |
|---|---|
| `uv run python scripts/main_v5.py` | Current CoreML model-runtime path with V4 control |
| `uv run python scripts/main_v4.py` | Learned MPID controller on the legacy runtime |
| `uv run python scripts/main_v3.py` | Tracker experiment |
| `uv run python scripts/main_v2.py` | Kalman target-state experiment |
| `uv run python scripts/main_v1.py` | Historical PIDF baseline |
| `uv run python -m unittest discover -s tests` | Run tests |

Configs live in `configs/`. V5 uses `configs/config_v5.json` for model runtime
settings and `configs/config_v4.json` for capture, KMBox, hotkey, tuner, and
control settings. The tuner defaults to `http://127.0.0.1:8765`.

## Architecture

Future versions are built on `macos_dualbox_aim.core`:

- `core.capture`: capture-card setup, center crop, frame handoff
- `core.inference`: realtime capture -> detector -> callback pipeline
- `core.model_runtime`: CoreML contract inspection and detection adapters
- `core.kmbox`: KMBox UDP transport and relative mouse movement
- `core.hotkey`: KMBox physical mouse monitor and trigger state

Version directories should contain only version-specific target selection,
control systems, configuration, tuner wiring, and entrypoint assembly. Old
version-local inference/capture/KMBox/hotkey files remain compatibility
surfaces and should not receive new feature work.

Read:

- `docs/architecture.md` for module boundaries and version policy
- `docs/runtime-pipeline.md` for coordinate flow and latency fields
- `docs/model-runtime.md` for CoreML conversion/runtime rules
- `docs/latency-findings.md` for stable latency conclusions
- `docs/releases.md` for concise version history

## Tools

Capture probe, run without the main runtime holding the device:

```bash
uv run python scripts/capture_probe.py \
  --device 0 \
  --formats MJPEG,YUY2,UYVY,RGB3,BGR3 \
  --fps 60,120,240 \
  --resolutions 1920x1080 \
  --samples 180 \
  --warmup 20 \
  --backend auto \
  --out-jsonl latency_runs/capture_probe.jsonl \
  --out-md latency_runs/capture_probe.md
```

Latency capture, run while a tuner-enabled main program is active:

```bash
uv run python scripts/latency_tool.py capture \
  --label v5 \
  --run run1 \
  --duration 60 \
  --interval 0.5
```

Model probe after conversion:

```bash
uv run python scripts/probe_v5_model.py \
  --check-model models/converted/cs2_fp16_fp32_check.mlpackage \
  --fast-model models/converted/cs2_fp16_fp16_fast.mlpackage \
  --runs 20 \
  --warmup 5 \
  --out latency_runs/v5_model_probe.json
```

## Development Rules

- Use `uv` for Python environments, dependencies, scripts, and CLI tools.
- Keep new shared hardware/inference work in `core`; keep version experiments in
  the relevant `vN/` directory until promoted.
- Do not change `mouse_move(dx, dy)` relative movement semantics without an
  explicit behavior decision.
- When changing config fields, verify both the config dataclass and the runtime
  consumer for that version.
