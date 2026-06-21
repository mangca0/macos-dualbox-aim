"""Stable hardware, capture, and inference base for future versions."""

from .capture import CaptureConfig, Frame, center_crop, configure_capture, crop_offset, fourcc_code, open_capture
from .capture_probe import CaptureMode, CaptureProbeResult, probe_capture_mode
from .hotkey import HotkeyConfig, HotkeyMonitor
from .inference import DetectionResult, RealtimeInference
from .kmbox import KmboxConfig, KmboxNet, SUCCESS
from .model_runtime import (
    MODEL_RUNTIME_VERSION,
    CoreMLDetector,
    FeatureSpec,
    ImageNMSAdapter,
    ModelContract,
    ModelInputKind,
    ModelOutputKind,
    ModelRuntimeConfig,
    OutputLayout,
    YoloV8TensorAdapter,
    classify_model_contract,
    inspect_coreml_model,
)

__all__ = [
    "MODEL_RUNTIME_VERSION",
    "CaptureConfig",
    "CaptureMode",
    "CaptureProbeResult",
    "CoreMLDetector",
    "DetectionResult",
    "FeatureSpec",
    "Frame",
    "HotkeyConfig",
    "HotkeyMonitor",
    "ImageNMSAdapter",
    "KmboxConfig",
    "KmboxNet",
    "ModelContract",
    "ModelInputKind",
    "ModelOutputKind",
    "ModelRuntimeConfig",
    "OutputLayout",
    "RealtimeInference",
    "SUCCESS",
    "YoloV8TensorAdapter",
    "center_crop",
    "classify_model_contract",
    "configure_capture",
    "crop_offset",
    "fourcc_code",
    "inspect_coreml_model",
    "open_capture",
    "probe_capture_mode",
]
