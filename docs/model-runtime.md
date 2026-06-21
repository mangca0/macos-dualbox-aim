# Model Runtime

## Purpose

`core.model_runtime` is the shared CoreML detector base introduced by V5. It
lets future versions test new models without rewriting capture, inference,
target selection, or controller code.

## Standard Contract

Preferred production model:

- Source format: ONNX
- Runtime format: CoreML ML Program
- Input: `ImageType`, RGB, fixed size, usually `320x320`
- Normalization: encoded in CoreML input scale when possible
- Output: CoreML NMS or raw YOLOv8-style tensor decoded by an adapter

`MLMultiArray` input is supported for direct ONNX-style compatibility, but it is
not the preferred long-term path because resize/color/normalization/transposition
stay in Python.

Adapters return:

```python
{
    "bbox": [cx, cy, width, height],
    "confidence": 0.0,
    "class_id": 0,
}
```

`bbox` values are normalized to model input size unless the adapter explicitly
documents otherwise.

## Current Adapters

- `ImageNMSAdapter`: CoreML NMS outputs named `coordinates` and `confidence`.
- `YoloV8TensorAdapter`: raw YOLOv8 tensors shaped `[1, 4 + classes, anchors]`
  or `[1, anchors, 4 + classes]`.

Do not infer model family from filename. Inspect the CoreML spec and output
shape with `inspect_coreml_model()`.

## Conversion

Default conversion creates a precision-check package and a realtime package:

```bash
uv run --with onnx --with onnx2torch --with torch python scripts/convert_onnx_to_coreml.py \
  /path/to/model.onnx \
  --output-dir models/converted \
  --input-name images \
  --input-size 320x320 \
  --image-scale 0.00392156862745098 \
  --minimum-deployment-target macOS13 \
  --compile
```

Outputs:

- `*_fp32_check.mlpackage`: precision/regression debugging
- `*_fp16_fast.mlpackage`: realtime runtime candidate

Use `--tensor-input` only for compatibility tests.

## Acceptance Checklist

Before wiring a model into KMBox control, record:

- ONNX checker result
- FP32 CoreML raw output diff against ONNX/Torch when available
- FP16 fast raw output diff against FP32 check package
- Decoded box stability and target-center pixel error on real frames
- NMS top-k consistency
- Median and p95 CoreML runtime after warmup
- Manual capture-card validation for target jitter and aim feel

Decoded target stability matters more than raw tensor mean absolute error alone.

## Probe

```bash
uv run python scripts/probe_v5_model.py \
  --check-model models/converted/cs2_fp16_fp32_check.mlpackage \
  --fast-model models/converted/cs2_fp16_fp16_fast.mlpackage \
  --runs 20 \
  --warmup 5 \
  --out latency_runs/model_probe.json
```

Pass `--image /path/to/frame.png` for real-frame validation. The default black
frame is only a smoke test and latency baseline.
