import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from macos_dualbox_aim.core.model_runtime.metadata import (
    infer_class_count_from_spec,
    inspect_model_class_info,
    normalize_class_names,
)


class _FakeDescription:
    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.type = SimpleNamespace(
            WhichOneof=lambda _field: "multiArrayType",
            multiArrayType=SimpleNamespace(shape=shape),
        )


class _FakeSpec:
    def __init__(self, outputs):
        self.description = SimpleNamespace(output=outputs)


class _FakeModel:
    def __init__(self, outputs, metadata=None):
        self._spec = _FakeSpec(outputs)
        self.user_defined_metadata = metadata or {}

    def get_spec(self):
        return self._spec


class ModelMetadataTests(unittest.TestCase):
    def test_normalize_class_names_pads_missing_entries(self):
        self.assertEqual(
            normalize_class_names(["head", "body"], 4),
            ["head", "body", "class_2", "class_3"],
        )

    def test_infer_class_count_from_raw_yolo_shape(self):
        spec = _FakeSpec([_FakeDescription("output0", (1, 8, 2100))])
        self.assertEqual(infer_class_count_from_spec(spec), 4)

    def test_inspect_model_class_info_prefers_sidecar_json_names(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "sample.mlpackage"
            model_path.write_text("", encoding="utf-8")
            metadata_path = model_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps({"class_count": 3, "class_names": ["head", "body", "loot"]}),
                encoding="utf-8",
            )
            info = inspect_model_class_info(
                model_path,
                model=_FakeModel([_FakeDescription("output0", (1, 7, 100))]),
            )

        self.assertEqual(info.class_count, 3)
        self.assertEqual(info.class_names, ("head", "body", "loot"))
        self.assertEqual(info.source, "sidecar")
        self.assertTrue(info.metadata_path.endswith(".json"))

    def test_inspect_model_class_info_uses_embedded_metadata_when_no_sidecar(self):
        info = inspect_model_class_info(
            "embedded.mlpackage",
            model=_FakeModel(
                [_FakeDescription("output0", (1, 8, 100))],
                metadata={"labels": ["a", "b", "c", "d"]},
            ),
        )

        self.assertEqual(info.class_count, 4)
        self.assertEqual(info.class_names, ("a", "b", "c", "d"))
        self.assertEqual(info.source, "embedded")

    def test_inspect_model_class_info_falls_back_to_generated_names(self):
        info = inspect_model_class_info(
            "fallback.mlpackage",
            model=_FakeModel([_FakeDescription("output0", (1, 6, 100))]),
        )

        self.assertEqual(info.class_count, 2)
        self.assertEqual(info.class_names, ("class_0", "class_1"))
        self.assertEqual(info.source, "spec")


if __name__ == "__main__":
    unittest.main()
