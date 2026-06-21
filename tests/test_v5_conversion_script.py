import importlib.util
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "convert_onnx_to_coreml.py"
    spec = importlib.util.spec_from_file_location("convert_onnx_to_coreml", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V5ConversionScriptTests(unittest.TestCase):
    def test_default_variants_create_check_and_fast_packages(self):
        module = _load_script_module()
        options = module.ConversionOptions(
            source=Path("models/example.onnx"),
            output_dir=Path("outputs"),
            input_name="images",
            input_height=320,
            input_width=320,
            image_input=True,
            image_scale=1.0 / 255.0,
            minimum_deployment_target="macOS13",
            compile_model=False,
        )

        variants = module.build_variants(options)

        self.assertEqual([variant.name for variant in variants], ["fp32_check", "fp16_fast"])
        self.assertEqual(variants[0].output_path, Path("outputs/example_fp32_check.mlpackage"))
        self.assertEqual(variants[1].output_path, Path("outputs/example_fp16_fast.mlpackage"))

    def test_converter_aliases_cover_newer_onnx_opset_gap(self):
        module = _load_script_module()

        self.assertIn(("Cast", 19, 13), module.COMPAT_CONVERTER_ALIASES)
        self.assertIn(("Reshape", 19, 14), module.COMPAT_CONVERTER_ALIASES)
        self.assertIn(("Resize", 19, 13), module.COMPAT_CONVERTER_ALIASES)
        self.assertIn(("Split", 18, 13), module.COMPAT_CONVERTER_ALIASES)


if __name__ == "__main__":
    unittest.main()
