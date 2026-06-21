#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


COMPAT_CONVERTER_ALIASES = (
    ("Cast", 19, 13),
    ("Reshape", 19, 14),
    ("Resize", 19, 13),
    ("Split", 18, 13),
)


@dataclass(frozen=True)
class ConversionOptions:
    source: Path
    output_dir: Path
    input_name: str
    input_height: int
    input_width: int
    image_input: bool
    image_scale: float
    minimum_deployment_target: str
    compile_model: bool


@dataclass(frozen=True)
class ConversionVariant:
    name: str
    output_path: Path
    compute_precision: str


def build_variants(options: ConversionOptions) -> list[ConversionVariant]:
    stem = options.source.stem
    return [
        ConversionVariant(
            name="fp32_check",
            output_path=options.output_dir / f"{stem}_fp32_check.mlpackage",
            compute_precision="FLOAT32",
        ),
        ConversionVariant(
            name="fp16_fast",
            output_path=options.output_dir / f"{stem}_fp16_fast.mlpackage",
            compute_precision="FLOAT16",
        ),
    ]


def convert(options: ConversionOptions) -> dict[str, Any]:
    onnx, torch, onnx2torch, ct = _load_conversion_modules()
    _install_onnx2torch_compat()
    options.output_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = onnx.load(str(options.source))
    onnx.checker.check_model(onnx_model)
    torch_model = onnx2torch.convert(onnx_model)
    torch_model.eval()

    example_input = torch.zeros(
        (1, 3, options.input_height, options.input_width),
        dtype=torch.float32,
    )
    traced_model = torch.jit.trace(torch_model, example_input)

    summaries = []
    for variant in build_variants(options):
        mlmodel = _convert_variant(ct, traced_model, options, variant)
        mlmodel.save(str(variant.output_path))
        compiled_path = None
        if options.compile_model:
            compiled_path = _compile_model(variant.output_path)
        summaries.append({
            "name": variant.name,
            "output": str(variant.output_path),
            "compiled": compiled_path,
            "compute_precision": variant.compute_precision,
            "interface": _summarize_coreml_interface(mlmodel),
        })

    summary = {
        "source": str(options.source),
        "options": _jsonable_options(options),
        "variants": summaries,
    }
    summary_path = options.output_dir / f"{options.source.stem}_conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _convert_variant(ct, traced_model, options: ConversionOptions, variant: ConversionVariant):
    compute_precision = (
        ct.precision.FLOAT16
        if variant.compute_precision == "FLOAT16"
        else ct.precision.FLOAT32
    )
    model_input = _coreml_input_type(ct, options)
    return ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[model_input],
        compute_precision=compute_precision,
        minimum_deployment_target=_deployment_target(ct, options.minimum_deployment_target),
    )


def _coreml_input_type(ct, options: ConversionOptions):
    shape = (1, 3, options.input_height, options.input_width)
    if options.image_input:
        return ct.ImageType(
            name=options.input_name,
            shape=shape,
            color_layout=ct.colorlayout.RGB,
            scale=options.image_scale,
        )
    return ct.TensorType(name=options.input_name, shape=shape)


def _deployment_target(ct, target_name: str):
    try:
        return getattr(ct.target, target_name)
    except AttributeError as exc:
        available = sorted(name for name in dir(ct.target) if name.startswith("macOS"))
        raise ValueError(f"Unknown deployment target {target_name!r}; available macOS targets: {available}") from exc


def _compile_model(package_path: Path) -> str:
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(package_path), directory],
            check=True,
            text=True,
            capture_output=True,
        )
        compiled = next(Path(directory).glob("*.mlmodelc"))
        final_path = package_path.with_suffix(".mlmodelc")
        subprocess.run(["ditto", str(compiled), str(final_path)], check=True)
    return str(final_path)


def _summarize_coreml_interface(mlmodel) -> dict[str, list[dict[str, Any]]]:
    spec = mlmodel.get_spec()
    return {
        "inputs": [_feature_summary(item) for item in spec.description.input],
        "outputs": [_feature_summary(item) for item in spec.description.output],
    }


def _feature_summary(description) -> dict[str, Any]:
    feature_type = description.type.WhichOneof("Type")
    summary = {"name": description.name, "type": feature_type}
    if feature_type == "imageType":
        image_type = description.type.imageType
        summary.update({"width": image_type.width, "height": image_type.height})
    elif feature_type == "multiArrayType":
        summary["shape"] = list(description.type.multiArrayType.shape)
    return summary


def _jsonable_options(options: ConversionOptions) -> dict[str, Any]:
    data = asdict(options)
    data["source"] = str(options.source)
    data["output_dir"] = str(options.output_dir)
    return data


def _load_conversion_modules():
    try:
        import coremltools as ct
        import onnx
        import onnx2torch
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing conversion dependency. Install the converter toolchain with uv, "
            "for example: uv add --dev onnx onnx2torch torch"
        ) from exc
    return onnx, torch, onnx2torch, ct


def _install_onnx2torch_compat() -> None:
    from onnx2torch.node_converters import registry

    for operation_type, target_version, source_version in COMPAT_CONVERTER_ALIASES:
        source = registry.OperationDescription(
            domain="",
            operation_type=operation_type,
            version=source_version,
        )
        target = registry.OperationDescription(
            domain="",
            operation_type=operation_type,
            version=target_version,
        )
        converter = registry._CONVERTER_REGISTRY.get(source)
        if converter is None:
            continue
        registry._CONVERTER_REGISTRY.setdefault(target, converter)


def parse_args(argv: Sequence[str]) -> ConversionOptions:
    parser = argparse.ArgumentParser(description="Convert an ONNX detector to Core ML check/fast packages.")
    parser.add_argument("source", type=Path, help="Source ONNX model path.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/converted"))
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--input-size", default="320x320", help="Input size as HEIGHTxWIDTH.")
    parser.add_argument("--tensor-input", action="store_true", help="Use MLMultiArray input instead of ImageType.")
    parser.add_argument("--image-scale", type=float, default=1.0 / 255.0)
    parser.add_argument("--minimum-deployment-target", default="macOS13")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    args = parser.parse_args(argv)
    input_height, input_width = _parse_size(args.input_size)
    return ConversionOptions(
        source=args.source,
        output_dir=args.output_dir,
        input_name=args.input_name,
        input_height=input_height,
        input_width=input_width,
        image_input=not args.tensor_input,
        image_scale=args.image_scale,
        minimum_deployment_target=args.minimum_deployment_target,
        compile_model=args.compile_model,
    )


def _parse_size(value: str) -> tuple[int, int]:
    try:
        height_text, width_text = value.lower().split("x", 1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Input size must use HEIGHTxWIDTH, for example 320x320") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Input size must be positive")
    return height, width


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(sys.argv[1:] if argv is None else argv)
    summary = convert(options)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
