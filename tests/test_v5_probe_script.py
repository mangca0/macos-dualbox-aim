import importlib.util
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "probe_v5_model.py"
    spec = importlib.util.spec_from_file_location("probe_v5_model", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class V5ProbeScriptTests(unittest.TestCase):
    def test_parse_args_defaults_to_converted_check_and_fast_models(self):
        module = _load_script_module()

        args = module.parse_args([])

        self.assertEqual(args.check_model, Path("models/converted/cs2_fp16_fp32_check.mlpackage"))
        self.assertEqual(args.fast_model, Path("models/converted/cs2_fp16_fp16_fast.mlpackage"))
        self.assertEqual(args.class_count, 4)
        self.assertEqual(args.runs, 20)
        self.assertEqual(args.warmup, 5)


if __name__ == "__main__":
    unittest.main()
