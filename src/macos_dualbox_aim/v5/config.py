import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


AIMBOT_V5_VERSION = "5.0.0-runtime-draft"


INT_FIELDS = {"class_count"}
FLOAT_FIELDS = {"confidence_threshold", "iou_threshold", "tensor_input_scale"}
BOOL_FIELDS = {"prefer_image_input"}
STR_FIELDS = {"model_path", "adapter"}


@dataclass
class ModelRuntimeConfigV5:
    model_path: str = "models/cs2_fp16.mlpackage"
    class_count: int = 4
    confidence_threshold: float = 0.65
    iou_threshold: float = 0.3
    adapter: str = "auto"
    prefer_image_input: bool = True
    tensor_input_scale: float = 1.0 / 255.0

    @property
    def version(self) -> str:
        return AIMBOT_V5_VERSION

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelRuntimeConfigV5":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Config root must be a JSON object")

        config = cls()
        field_names = {item.name for item in fields(cls)}
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if key not in field_names:
                raise ValueError(f"Unknown config field: {key}")
            setattr(config, key, _coerce_field(key, value))
        return config

    def to_json(self, path: str | Path) -> None:
        data: dict[str, Any] = {
            "_comment": "macos-dualbox-aim V5 model-runtime draft",
            "_version": AIMBOT_V5_VERSION,
            "_description": "ONNX source -> Core ML check/fast packages -> adapter-based detections",
        }
        data.update(asdict(self))
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")


def _coerce_field(key: str, value):
    if key in INT_FIELDS:
        return int(value)
    if key in FLOAT_FIELDS:
        return float(value)
    if key in BOOL_FIELDS:
        return bool(value)
    if key in STR_FIELDS:
        return str(value)
    return value
