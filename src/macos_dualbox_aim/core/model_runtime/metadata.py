from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ModelClassInfo:
    class_count: int
    class_names: tuple[str, ...]
    source: str
    metadata_path: Optional[str] = None


def inspect_model_class_info(
    model_path: str | Path,
    *,
    model: Optional[Any] = None,
    fallback_class_count: Optional[int] = None,
) -> ModelClassInfo:
    model_file = Path(model_path)
    if model is None:
        import coremltools as ct

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = ct.models.MLModel(str(model_file))

    spec = model.get_spec()
    sidecar = _load_sidecar_metadata(model_file)
    embedded = _load_embedded_metadata(model)

    class_names = _extract_class_names(sidecar)
    source = "sidecar"
    metadata_path = str(_sidecar_path(model_file)) if class_names else None
    if not class_names:
        class_names = _extract_class_names(embedded)
        source = "embedded"
        metadata_path = None

    class_count = _extract_class_count(sidecar)
    if class_count is None:
        class_count = _extract_class_count(embedded)
    if class_count is None and class_names:
        class_count = len(class_names)
    if class_count is None:
        class_count = infer_class_count_from_spec(spec)
        source = "spec"
        metadata_path = None
    if class_count is None:
        class_count = fallback_class_count
        source = "fallback"
        metadata_path = None
    if class_count is None or int(class_count) <= 0:
        raise ValueError("Unable to infer model class_count; add model metadata or a valid fallback class_count")

    normalized_names = normalize_class_names(class_names, int(class_count))
    if not class_names and source not in {"spec", "fallback"}:
        source = "generated"
    return ModelClassInfo(
        class_count=int(class_count),
        class_names=tuple(normalized_names),
        source=source,
        metadata_path=metadata_path,
    )


def infer_class_count_from_spec(spec: Any) -> Optional[int]:
    outputs = list(getattr(spec.description, "output", []))
    output_names = {str(getattr(item, "name", "")) for item in outputs}
    if {"coordinates", "confidence"}.issubset(output_names):
        for item in outputs:
            if getattr(item, "name", "") != "confidence":
                continue
            shape = _multiarray_shape(item)
            count = _class_count_from_confidence_shape(shape)
            if count is not None:
                return count

    for item in outputs:
        shape = _multiarray_shape(item)
        count = _class_count_from_raw_output_shape(shape)
        if count is not None:
            return count
    return None


def normalize_class_names(class_names: Sequence[str], class_count: int) -> list[str]:
    normalized = [str(name).strip() for name in class_names if str(name).strip()]
    if len(normalized) >= class_count:
        return normalized[:class_count]
    return normalized + [f"class_{index}" for index in range(len(normalized), class_count)]


def _load_embedded_metadata(model: Any) -> Mapping[str, Any]:
    metadata = getattr(model, "user_defined_metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    spec_metadata = getattr(model.get_spec().description, "metadata", None)
    user_defined = getattr(spec_metadata, "userDefined", None)
    if isinstance(user_defined, Mapping):
        return dict(user_defined)
    return {}


def _load_sidecar_metadata(model_path: Path) -> Mapping[str, Any]:
    path = _sidecar_path(model_path)
    if path is None or not path.exists():
        return {}
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"Sidecar metadata must be a JSON object: {path}")
        return dict(data)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return {"class_names": [line for line in lines if line]}


def _sidecar_path(model_path: Path) -> Optional[Path]:
    for suffix in (".json", ".txt"):
        candidate = model_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return model_path.with_suffix(".json")


def _extract_class_names(metadata: Mapping[str, Any]) -> list[str]:
    for key in ("class_names", "classes", "labels", "names"):
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, Mapping):
            ordered: list[tuple[int, str]] = []
            for raw_key, raw_name in value.items():
                try:
                    index = int(raw_key)
                except (TypeError, ValueError):
                    continue
                ordered.append((index, str(raw_name)))
            ordered.sort(key=lambda item: item[0])
            return [name for _, name in ordered]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
    return []


def _extract_class_count(metadata: Mapping[str, Any]) -> Optional[int]:
    value = metadata.get("class_count")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _multiarray_shape(description: Any) -> tuple[int, ...]:
    feature_type = description.type.WhichOneof("Type")
    if feature_type != "multiArrayType":
        return ()
    return tuple(int(value) for value in description.type.multiArrayType.shape)


def _class_count_from_confidence_shape(shape: tuple[int, ...]) -> Optional[int]:
    positives = [value for value in shape if int(value) > 0]
    if not positives:
        return None
    for value in reversed(positives):
        if value > 1:
            return int(value)
    return None


def _class_count_from_raw_output_shape(shape: tuple[int, ...]) -> Optional[int]:
    if len(shape) < 2:
        return None
    candidates = sorted(value for value in shape if int(value) > 4)
    if not candidates:
        return None
    smallest = int(candidates[0])
    return smallest - 4 if smallest > 4 else None
