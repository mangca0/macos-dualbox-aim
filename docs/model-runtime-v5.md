# V5 Model Runtime Contract

V5 is the first version boundary for the model-runtime architecture. It is not a
control-loop experiment like V1-V4; it isolates ONNX-to-CoreML conversion,
CoreML interface inspection, and detector output adapters so future models can
be tested without changing the existing V1-V4 runtime paths.

## Goals

- Keep ONNX as the source model format.
- Deploy Core ML packages for the lowest macOS inference latency.
- Produce a check build and a fast build for every model.
- Keep model-specific parsing behind adapters.
- Do not promote this runtime into `core/` until it is stable across multiple
  model families and runtime versions.

## Standard Model Contract

Preferred production contract:

- Input kind: `ImageType`
- Input layout: RGB image
- Input size: fixed, usually `320x320` unless measured accuracy requires more
- Image normalization: encoded in the Core ML input scale when possible
- Output kind during exploration: raw tensor
- Output parsing: V5 adapter layer

Transitional support:

- `MLMultiArray` input is supported for models that were converted directly from
  ONNX and still expose `1x3xHxW` tensor input.
- Tensor input should not be the long-term default because it keeps resize,
  color conversion, normalization, and HWC-to-CHW conversion in Python.

## Build Variants

Each ONNX model should generate two Core ML packages:

- `*_fp32_check.mlpackage`
  - Purpose: precision alignment and regression debugging.
  - Uses FP32 compute where Core ML supports it.
  - Used to compare raw output, decoded boxes, target center, and NMS results.
- `*_fp16_fast.mlpackage`
  - Purpose: realtime runtime.
  - Uses ML Program + FP16 compute.
  - Should be precompiled to `.mlmodelc` for deployment when practical.

The fast package is expected to have small numerical differences. It is accepted
only when final decoded detections remain stable enough for aim control.

## Adapter Rules

Adapters receive Core ML predictions and return the existing internal detection
shape:

```python
{
    "bbox": [cx, cy, width, height],  # normalized to model input size
    "confidence": 0.0,
    "class_id": 0,
}
```

Current adapters:

- `ImageNMSAdapter`: existing Core ML NMS output with `coordinates` and
  `confidence`.
- `YoloV8TensorAdapter`: raw YOLOv8-style output shaped either
  `[1, 4 + classes, anchors]` or `[1, anchors, 4 + classes]`.

Do not infer a detector family only from the filename. Inspect the Core ML spec
and output shape, then choose the adapter.

## Conversion Command

The conversion script loads `onnx`, `onnx2torch`, and `torch` only when an
actual conversion is requested. If those packages are not installed in the
project environment, run the script through `uv` with temporary tool deps or add
them to a dedicated converter environment.

Default conversion command:

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

Use `--tensor-input` only for direct compatibility tests. New production
packages should prefer `ImageType` input.

## Acceptance Checks

Before a model becomes a runtime candidate, record:

- ONNX checker result.
- FP32 Core ML raw output diff against ONNX/Torch.
- FP16 Core ML raw output diff against the FP32 check package.
- Decoded box IoU and target-center pixel error.
- NMS top-k consistency.
- Median and p95 Core ML inference time after warmup.
- Whether the selected target jitters in real capture footage.

For aim quality, decoded target stability matters more than raw tensor mean
absolute error alone.
