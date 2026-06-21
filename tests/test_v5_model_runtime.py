import unittest
from pathlib import Path

import numpy as np

from macos_dualbox_aim.v5.config import AIMBOT_V5_VERSION, ModelRuntimeConfigV5
from macos_dualbox_aim.v5.model_runtime.adapters import YoloV8TensorAdapter
from macos_dualbox_aim.v5.model_runtime.contract import (
    FeatureSpec,
    ModelInputKind,
    ModelOutputKind,
    OutputLayout,
    classify_model_contract,
)
from macos_dualbox_aim.v5.model_runtime.detector import inspect_coreml_model


class V5ModelRuntimeTests(unittest.TestCase):
    def test_v5_model_runtime_config_defaults_to_fast_coreml_candidate(self):
        config = ModelRuntimeConfigV5()

        self.assertEqual(config.version, AIMBOT_V5_VERSION)
        self.assertEqual(config.model_path, "models/cs2_fp16.mlpackage")
        self.assertEqual(config.class_count, 4)
        self.assertTrue(config.prefer_image_input)

    def test_contract_classifies_image_input_with_yolov8_raw_output(self):
        contract = classify_model_contract(
            inputs=[
                FeatureSpec(
                    name="image",
                    kind=ModelInputKind.IMAGE,
                    shape=(1, 3, 320, 320),
                    width=320,
                    height=320,
                )
            ],
            outputs=[
                FeatureSpec(
                    name="output0",
                    kind=ModelInputKind.MULTI_ARRAY,
                    shape=(1, 8, 2100),
                )
            ],
            class_count=4,
        )

        self.assertEqual(contract.input_name, "image")
        self.assertEqual(contract.input_kind, ModelInputKind.IMAGE)
        self.assertEqual(contract.input_size, (320, 320))
        self.assertEqual(contract.output_kind, ModelOutputKind.YOLO_RAW)
        self.assertEqual(contract.output_layout, OutputLayout.CHANNELS_FIRST)
        self.assertEqual(contract.adapter_name, "yolov8")

    def test_contract_classifies_existing_image_nms_model(self):
        contract = classify_model_contract(
            inputs=[
                FeatureSpec(
                    name="image",
                    kind=ModelInputKind.IMAGE,
                    shape=(1, 3, 640, 640),
                    width=640,
                    height=640,
                )
            ],
            outputs=[
                FeatureSpec(name="coordinates", kind=ModelInputKind.MULTI_ARRAY, shape=(-1, 4)),
                FeatureSpec(name="confidence", kind=ModelInputKind.MULTI_ARRAY, shape=(-1, 80)),
            ],
            class_count=80,
        )

        self.assertEqual(contract.output_kind, ModelOutputKind.COREML_NMS)
        self.assertEqual(contract.adapter_name, "image_nms")

    def test_yolov8_adapter_decodes_channels_first_output(self):
        raw = np.zeros((1, 8, 3), dtype=np.float32)
        raw[0, :, 0] = [160.0, 160.0, 40.0, 80.0, 0.10, 0.90, 0.20, 0.05]
        raw[0, :, 1] = [162.0, 160.0, 40.0, 80.0, 0.05, 0.80, 0.10, 0.05]
        raw[0, :, 2] = [40.0, 40.0, 20.0, 20.0, 0.70, 0.10, 0.20, 0.05]

        adapter = YoloV8TensorAdapter(input_size=(320, 320), class_count=4)
        detections = adapter.parse({"output0": raw}, confidence_threshold=0.25, iou_threshold=0.5)

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_id"], 1)
        self.assertAlmostEqual(detections[0]["confidence"], 0.90, places=5)
        self.assertEqual(detections[1]["class_id"], 0)
        self.assertAlmostEqual(detections[0]["bbox"][0], 0.5, places=5)
        self.assertAlmostEqual(detections[0]["bbox"][1], 0.5, places=5)

    def test_inspects_current_tensor_coreml_package_as_yolov8(self):
        model_path = "models/cs2_fp16.mlpackage"
        if not Path(model_path).exists():
            self.skipTest(f"Local Core ML fixture is missing: {model_path}")

        contract = inspect_coreml_model(model_path, class_count=4)

        self.assertEqual(contract.input_name, "images")
        self.assertEqual(contract.input_kind, ModelInputKind.MULTI_ARRAY)
        self.assertEqual(contract.input_size, (320, 320))
        self.assertEqual(contract.output_names, ("output0",))
        self.assertEqual(contract.output_layout, OutputLayout.CHANNELS_FIRST)
        self.assertEqual(contract.adapter_name, "yolov8")


if __name__ == "__main__":
    unittest.main()
