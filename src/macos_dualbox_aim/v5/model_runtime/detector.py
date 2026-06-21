from pathlib import Path
from typing import Dict, List

import cv2
import coremltools as ct
import numpy as np
from PIL import Image

from .adapters import ImageNMSAdapter, YoloV8TensorAdapter
from .contract import (
    FeatureSpec,
    ModelContract,
    ModelInputKind,
    ModelOutputKind,
    classify_model_contract,
)


class CoreMLDetectorV5:
    def __init__(
        self,
        model_path: str,
        class_count: int,
        tensor_input_scale: float = 1.0 / 255.0,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = ct.models.MLModel(str(model_file))
        self.contract = contract_from_spec(self.model.get_spec(), class_count=class_count)
        self.tensor_input_scale = float(tensor_input_scale)
        self.adapter = self._build_adapter(self.contract)

    def predict(self, image: np.ndarray, iou_threshold: float, confidence_threshold: float) -> List[Dict]:
        predictions = self.model.predict({self.contract.input_name: self._preprocess(image)})
        return self.adapter.parse(predictions, confidence_threshold, iou_threshold)

    def _preprocess(self, image: np.ndarray):
        expected_h, expected_w = self.contract.input_size
        if image.shape[0] != expected_h or image.shape[1] != expected_w:
            image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.contract.input_kind == ModelInputKind.IMAGE:
            return Image.fromarray(rgb)
        if self.contract.input_kind == ModelInputKind.MULTI_ARRAY:
            tensor = rgb.astype(np.float32) * self.tensor_input_scale
            return np.transpose(tensor, (2, 0, 1))[None, ...]
        raise ValueError(f"Unsupported Core ML input kind: {self.contract.input_kind}")

    def _build_adapter(self, contract: ModelContract):
        if contract.output_kind == ModelOutputKind.COREML_NMS:
            return ImageNMSAdapter()
        if contract.output_kind == ModelOutputKind.YOLO_RAW and contract.adapter_name == "yolov8":
            return YoloV8TensorAdapter(contract.input_size, contract.class_count)
        raise ValueError(f"Unsupported Core ML model contract: {contract}")


def inspect_coreml_model(model_path: str, class_count: int) -> ModelContract:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = ct.models.MLModel(str(model_file))
    return contract_from_spec(model.get_spec(), class_count=class_count)


def contract_from_spec(spec, class_count: int) -> ModelContract:
    inputs = [_feature_from_description(item) for item in spec.description.input]
    outputs = [_feature_from_description(item) for item in spec.description.output]
    return classify_model_contract(inputs, outputs, class_count=class_count)


def _feature_from_description(description) -> FeatureSpec:
    feature_type = description.type.WhichOneof("Type")
    if feature_type == "imageType":
        image_type = description.type.imageType
        return FeatureSpec(
            name=description.name,
            kind=ModelInputKind.IMAGE,
            shape=(1, 3, int(image_type.height), int(image_type.width)),
            width=int(image_type.width),
            height=int(image_type.height),
        )
    if feature_type == "multiArrayType":
        multi_array = description.type.multiArrayType
        return FeatureSpec(
            name=description.name,
            kind=ModelInputKind.MULTI_ARRAY,
            shape=tuple(int(value) for value in multi_array.shape),
        )
    if feature_type == "doubleType":
        return FeatureSpec(name=description.name, kind=ModelInputKind.DOUBLE)
    return FeatureSpec(name=description.name, kind=ModelInputKind.UNKNOWN)
