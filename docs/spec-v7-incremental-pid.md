# Spec: V7 Incremental PID Runtime

## Assumptions

1. V7 is an independent runtime version, not an edit to V6.3 in place.
2. V7 must be usable in the real capture -> CoreML -> target selection -> controller -> KMBox chain.
3. The C++ learning project is the behavioral source for the controller and should be replicated closely, including Perlin noise and debug output fields.
4. Existing V6.3 hardware, inference, hotkey, tuner, crosshair, and model-aware class selection behavior stays as the practical runtime base unless V7 explicitly replaces it.

## Objective

Build `macos_dualbox_aim.v7` as a real-use experiment that starts from the current V6.3 runtime chain and replaces the aim control surface with a strict Python replica of the learning project's controller:

- `IncrementalPid`
- `DerivativePredictor`
- `PerlinNoise1D`
- `AimOutput`
- `AimController`

V7 should preserve the project boundary: the macOS machine captures video, runs CoreML detection, selects a target, computes relative mouse movement, and sends it through KMBox. It must not depend on game process modification, memory reads, or client injection.

Success means V7 can run with:

```bash
uv run python scripts/main_v7.py
```

and uses only:

```text
configs/config_v7.json
src/macos_dualbox_aim/v7/
```

for V7-specific behavior.

## Tech Stack

- Python 3.11+
- `uv` for environment, scripts, and tests
- Existing `macos_dualbox_aim.core` modules for capture, CoreML inference, KMBox transport, and KMBox hotkeys
- Existing V6.3-style tracker, crosshair detector, class selection, and web tuner as the runtime base
- Standard library `random` for deterministic Perlin table initialization

## Commands

Development setup:

```bash
uv sync
```

Run V7:

```bash
uv run python scripts/main_v7.py
```

Run V7-focused tests:

```bash
uv run python -m unittest tests.test_v7
```

Run full tests:

```bash
uv run python -m unittest discover -s tests
```

Syntax-check changed Python files:

```bash
uv run python -m py_compile scripts/main_v7.py src/macos_dualbox_aim/v7/*.py tests/test_v7.py
```

## Project Structure

```text
configs/config_v7.json
  V7's single runtime config source.

scripts/main_v7.py
  V7 executable entrypoint.

src/macos_dualbox_aim/v7/__init__.py
  V7 public exports.

src/macos_dualbox_aim/v7/config.py
  V7 config dataclass and validation.

src/macos_dualbox_aim/v7/controller.py
  V7 aimbot, target conversion, and strict C++ controller replica.

src/macos_dualbox_aim/v7/crosshair.py
src/macos_dualbox_aim/v7/inference.py
src/macos_dualbox_aim/v7/tracker.py
src/macos_dualbox_aim/v7/tuner.py
  V7-owned copies of the V6.3 runtime support modules, adjusted only where V7 config/type names require it.

tests/test_v7.py
  Unit and integration tests for V7 config, controller behavior, real-chain wiring, tuner fields, and script wiring.

docs/releases.md
README.md
docs/architecture.md
  Version docs updated after V7 implementation is complete.
```

## Code Style

Prefer small explicit classes and plain dataclasses. Keep controller math close to the C++ source so comparisons are easy.

Example style:

```python
def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


@dataclass
class AimOutput:
    move_x: float
    move_y: float
    curve_len: float
    predicted_x: float
    predicted_y: float
    fused_x: float
    fused_y: float
```

Naming conventions:

- Keep C++ concept names when they carry behavior: `IncrementalPid`, `PerlinNoise1D`, `DerivativePredictor`, `AimController`, `AimOutput`.
- Use V7-specific public names: `AIMBOT_V7_VERSION`, `AimbotConfigV7`, `AimbotV7`, `RealtimeInferenceV7`.
- Avoid shared behavior changes in `core` unless a V7 requirement cannot be met inside the version package.

## Baseline V7 Scope

Always include:

- New isolated version package `macos_dualbox_aim.v7`.
- New `scripts/main_v7.py`.
- New `configs/config_v7.json`.
- Real capture/CoreML/KMBox/hotkey runtime based on V6.3.
- Crosshair-referenced target error from V6.3.
- Model-aware selected class filtering from V6.3.
- Multi-object tracker from V6.3.
- Strict Python replica of the C++ controller math:
  - incremental PID formula
  - hardcoded input deadzone `0.3`
  - hardcoded output damping threshold `0.5`
  - target jump reset default `40.0`
  - derivative predictor velocity and acceleration smoothing
  - prediction limit `min(max(abs(raw) * 1.5, 30.0), 60.0)`
  - smoothstep ramp from `init_scale` to `1.0`
  - Perlin noise added after PID update
  - final output clamp by `output_max`
- Debug output fields equivalent to `AimOutput`.
- Tests proving the controller replica and runtime wiring.

## Candidate Additions Requiring Approval

1. **Live tuner fields for Perlin noise and debug output**
   - Add `noise_amp`, `output_max`, and controller debug snapshot to the web tuner.
   - Reason: strict C++ replica exposes these as update-time inputs, and live tuning is useful in real use.
   - Tradeoff: more UI/config surface and more tests.

2. **Runtime latency snapshot fields for controller internals**
   - Add latest `predicted_x/y`, `fused_x/y`, `curve_len`, and ramp scale to tuner snapshots or latency output.
   - Reason: makes dynamic lag/overshoot tuning observable.
   - Tradeoff: extra instrumentation that is not required to move the mouse.

3. **Keep V6.1 adaptive integral gate as an optional mode**
   - Add a config switch that can choose strict C++ integral behavior or the V6.1 gated integral.
   - Reason: V6.1 gate may be better near static targets.
   - Tradeoff: not a strict replica when enabled, so default must be off if strictness is the priority.

4. **Output slew-rate limiting**
   - Implement the existing `slew_limit` idea as a real per-frame change limiter.
   - Reason: may reduce sudden output changes on noisy detections.
   - Tradeoff: changes controller behavior beyond the C++ source and may add lag.

5. **Offline replay/simulation harness**
   - Add a script that feeds recorded or synthetic target offsets into `AimController` and writes CSV.
   - Reason: safer tuning before live KMBox use.
   - Tradeoff: extra artifact; does not directly improve the live chain.

6. **Configurable deadzones and predictor clamps**
   - Make `0.3`, `0.5`, velocity/acceleration clamps, and prediction clamps configurable.
   - Reason: useful for tuning different frame rates and model noise levels.
   - Tradeoff: less faithful to the learning project and expands tuner complexity.

7. **Controller-only reusable core promotion**
   - Move the replicated controller into `core` after validation.
   - Reason: useful only if later versions reuse it.
   - Tradeoff: premature promotion would make an experimental controller a shared base.

## Testing Strategy

Use `unittest`, matching the existing test suite.

Required V7 tests:

- Config defaults expose V7 version and strict replica fields.
- `PerlinNoise1D` is deterministic for fixed seeds and returns bounded values.
- `IncrementalPid` matches the incremental formula, deadzone, output damping, and output clamp.
- `DerivativePredictor` returns zero on first call, then bounded prediction on later calls.
- `AimController` resets on target jump and returns `AimOutput` with predicted/fused fields.
- With `noise_amp=0`, simple controller scenarios are deterministic.
- `AimbotV7` keeps V6.3-style crosshair and selected-class behavior.
- `scripts/main_v7.py` passes frames, crop offsets, and latency dicts into `AimbotV7.update`.
- Tuner applies any approved new boolean/float fields correctly.

## Boundaries

- Always: keep V7 isolated from stable versions.
- Always: use `core` for capture, inference, KMBox, and hotkey plumbing.
- Always: use `uv` for Python commands.
- Always: run syntax checks for changed Python files before finishing.
- Always: state which config version consumes any new config fields.
- Ask first: enabling V6.1 integral gate in V7 default behavior.
- Ask first: adding output slew limiting.
- Ask first: promoting V7 controller code into `core`.
- Ask first: adding dependencies.
- Never: change `mouse_move(dx, dy)` relative movement semantics.
- Never: depend on game memory, process injection, or client modification.
- Never: edit V6.3 behavior while implementing V7 unless explicitly requested.
- Never: add implicit destructive KMBox monitor-port cleanup.

## Success Criteria

- `uv run python scripts/main_v7.py` starts a V7 real-chain runtime using `configs/config_v7.json`.
- V7 exports `AimbotConfigV7`, `AimbotV7`, `RealtimeInferenceV7`, `WebTuner`, and controller helper classes.
- V7 controller behavior is a close Python replica of the provided C++ files, with deliberate deviations documented.
- V7 uses relative KMBox movement and keeps the console/game boundary intact.
- `uv run python -m unittest tests.test_v7` passes.
- Changed Python files pass `py_compile`.
- README and release docs mention V7 after implementation.

## Decisions

1. Include live tuner fields for `noise_amp`, `output_max`, and controller debug output.
2. Do not add controller internals to latency metrics; expose them separately as controller debug state.
3. Keep the V6.1 adaptive integral gate as an optional config mode, defaulting off for strict replica behavior.
4. Keep `slew_limit` as a legacy/runtime config field only; do not implement extra slew-rate limiting because the learning project does not have that behavior.
5. Do not add an offline replay/simulation harness in this slice; prioritize the real V7 runtime chain.
