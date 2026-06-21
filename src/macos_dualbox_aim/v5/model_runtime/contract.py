from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Optional, Sequence


class ModelInputKind(StrEnum):
    IMAGE = "image"
    MULTI_ARRAY = "multiArray"
    DOUBLE = "double"
    UNKNOWN = "unknown"


class ModelOutputKind(StrEnum):
    COREML_NMS = "coreml_nms"
    YOLO_RAW = "yolo_raw"
    UNKNOWN = "unknown"


class OutputLayout(StrEnum):
    CHANNELS_FIRST = "channels_first"
    ANCHORS_FIRST = "anchors_first"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: ModelInputKind
    shape: tuple[int, ...] = ()
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class ModelContract:
    input_name: str
    input_kind: ModelInputKind
    input_size: tuple[int, int]
    output_names: tuple[str, ...]
    output_kind: ModelOutputKind
    output_layout: OutputLayout
    adapter_name: str
    class_count: int


def classify_model_contract(
    inputs: Sequence[FeatureSpec],
    outputs: Sequence[FeatureSpec],
    class_count: int,
) -> ModelContract:
    model_input = _primary_input(inputs)
    output_names = tuple(output.name for output in outputs)
    output_kind = ModelOutputKind.UNKNOWN
    output_layout = OutputLayout.UNKNOWN
    adapter_name = "unknown"

    if {"coordinates", "confidence"}.issubset(set(output_names)):
        output_kind = ModelOutputKind.COREML_NMS
        adapter_name = "image_nms"
    else:
        raw_output = _first_multi_array(outputs)
        if raw_output is not None:
            output_layout = _classify_yolo_layout(raw_output.shape, class_count)
            if output_layout != OutputLayout.UNKNOWN:
                output_kind = ModelOutputKind.YOLO_RAW
                adapter_name = "yolov8"

    return ModelContract(
        input_name=model_input.name,
        input_kind=model_input.kind,
        input_size=_input_size(model_input),
        output_names=output_names,
        output_kind=output_kind,
        output_layout=output_layout,
        adapter_name=adapter_name,
        class_count=class_count,
    )


def _primary_input(inputs: Sequence[FeatureSpec]) -> FeatureSpec:
    for item in inputs:
        if item.kind in {ModelInputKind.IMAGE, ModelInputKind.MULTI_ARRAY}:
            return item
    raise ValueError("Core ML model has no image or multi-array input")


def _input_size(feature: FeatureSpec) -> tuple[int, int]:
    if feature.kind == ModelInputKind.IMAGE:
        if feature.height is None or feature.width is None:
            raise ValueError(f"Image input {feature.name!r} is missing height/width")
        return (feature.height, feature.width)
    if len(feature.shape) == 4:
        return (int(feature.shape[2]), int(feature.shape[3]))
    raise ValueError(f"Input {feature.name!r} shape is not NCHW: {feature.shape}")


def _first_multi_array(outputs: Iterable[FeatureSpec]) -> Optional[FeatureSpec]:
    for output in outputs:
        if output.kind == ModelInputKind.MULTI_ARRAY:
            return output
    return None


def _classify_yolo_layout(shape: tuple[int, ...], class_count: int) -> OutputLayout:
    if len(shape) != 3:
        return OutputLayout.UNKNOWN
    channels = 4 + int(class_count)
    if shape[1] == channels:
        return OutputLayout.CHANNELS_FIRST
    if shape[2] == channels:
        return OutputLayout.ANCHORS_FIRST
    return OutputLayout.UNKNOWN
