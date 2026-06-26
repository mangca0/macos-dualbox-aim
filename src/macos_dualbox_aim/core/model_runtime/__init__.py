from .adapters import ImageNMSAdapter, YoloV8TensorAdapter
from .contract import (
    FeatureSpec,
    ModelContract,
    ModelInputKind,
    ModelOutputKind,
    OutputLayout,
    classify_model_contract,
)
from .config import AIMBOT_V5_VERSION, MODEL_RUNTIME_VERSION, ModelRuntimeConfig, ModelRuntimeConfigV5
from .detector import (
    CoreMLDetector,
    CoreMLDetectorV5,
    contract_from_spec,
    inspect_coreml_model,
    inspect_coreml_model_classes,
)
from .metadata import ModelClassInfo, infer_class_count_from_spec, inspect_model_class_info, normalize_class_names
from .probe import compare_arrays, summarize_detections, summarize_timings

__all__ = [
    "AIMBOT_V5_VERSION",
    "CoreMLDetector",
    "CoreMLDetectorV5",
    "FeatureSpec",
    "ImageNMSAdapter",
    "ModelClassInfo",
    "ModelContract",
    "ModelInputKind",
    "ModelOutputKind",
    "MODEL_RUNTIME_VERSION",
    "ModelRuntimeConfig",
    "ModelRuntimeConfigV5",
    "OutputLayout",
    "YoloV8TensorAdapter",
    "classify_model_contract",
    "compare_arrays",
    "contract_from_spec",
    "infer_class_count_from_spec",
    "inspect_coreml_model",
    "inspect_coreml_model_classes",
    "inspect_model_class_info",
    "normalize_class_names",
    "summarize_detections",
    "summarize_timings",
]
