# Current Review: V7 and Auto-Tuning

Date: 2026-07-02

This review covers the currently uncommitted workspace state. Raw auto-tuning
JSONL files under `aim_tuning_runs/` are treated as local experiment artifacts,
not source documentation.

## Summary

V7 is the active experiment and is manually usable, but it should still be
treated as an experiment rather than a validated replacement for the V6.3
baseline. It keeps the V6.3 real chain and swaps the controller for a strict
incremental PID replica with predictor and Perlin-noise support.

The auto-tuner is useful for coarse exploration only. Current live metrics can
rank obvious bad candidates, but they do not reliably optimize practical aiming
quality across static settling, target stops, and dynamic tracking.

## Workspace Changes Reviewed

- V6.3 gained aim metric capture, hidden tuner aim activation, and
  auto-tuner support for live aim scoring.
- V6.4 adds an isolated stop-brake experiment on top of V6.3 for sudden
  high-speed target stops.
- V7 adds an isolated strict incremental PID runtime, config, tuner, entrypoint,
  tests, and an auto-tune wrapper that reuses the V6.3 search/scoring machinery.
- `core.hotkey` now supports an explicit override path so auto-tuning can keep
  aim active without relying on a physical KMBox trigger.

## V7 Assessment

The implementation follows the project boundary: capture and CoreML stay in
`core`, the version package owns target/control behavior, and KMBox output
remains relative mouse movement.

Useful properties:

- V7 is isolated in `src/macos_dualbox_aim/v7/` with its own config and
  entrypoint.
- The tuner exposes controller debug fields: move, prediction, fused error, and
  curve length.
- Tests cover controller math, tuner wiring, crosshair/class filtering, hidden
  activation, aim metrics, and script wiring.

Current limitations:

- V7 is based on live short-window scoring, not a repeatable replay or
  synthetic target harness.
- `slew_limit` is exposed and passed through, but the V7 controller currently
  stores it without applying it to output changes.
- `output_max` and `noise_amp` are tuner-visible but excluded from V7 auto-search;
  that is deliberate for safety, but it means auto-tune cannot discover these
  control surfaces.
- The score heavily favors horizontal centering and generic overrun proxies. It
  does not separate static settle, dynamic tracking lag, and sudden target-stop
  behavior.
- Live target availability, crosshair detection, tracker state, and human/device
  motion can dominate a short trial, so accepted candidates may reflect the
  scene more than the controller quality.

## Auto-Tuning Guidance

Use auto-tune output as a hint, not as an authority. Prefer manual validation for
any saved config, especially when a candidate improves score but feels worse in
game.

Recommended next step before more tuning work:

1. Add a controller replay harness for fixed static, moving, and sudden-stop
   target-offset sequences.
2. Score static settling, dynamic lag, overshoot, and output smoothness as
   separate reported metrics before combining them.
3. Decide whether V7 should remain strict replica behavior or intentionally add
   practical extensions such as active slew limiting or stop-brake logic.

## Commit Boundary

Commit source, configs, tests, scripts, and curated docs. Do not commit
`aim_tuning_runs/`; keep those JSONL records local unless a later analysis
extracts stable conclusions into documentation.
