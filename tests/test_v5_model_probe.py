import unittest

import numpy as np

from macos_dualbox_aim.v5.model_runtime.probe import (
    compare_arrays,
    summarize_detections,
    summarize_timings,
)


class V5ModelProbeTests(unittest.TestCase):
    def test_compare_arrays_reports_abs_diff(self):
        baseline = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        candidate = np.array([1.5, 1.0, 4.0], dtype=np.float32)

        summary = compare_arrays(baseline, candidate)

        self.assertEqual(summary["shape"], [3])
        self.assertAlmostEqual(summary["mean_abs_diff"], 0.5)
        self.assertAlmostEqual(summary["max_abs_diff"], 1.0)

    def test_summarize_timings_reports_median_p95_and_mean(self):
        summary = summarize_timings([1.0, 2.0, 10.0, 3.0, 4.0])

        self.assertEqual(summary["runs"], 5)
        self.assertAlmostEqual(summary["median_ms"], 3.0)
        self.assertAlmostEqual(summary["mean_ms"], 4.0)
        self.assertGreater(summary["p95_ms"], summary["median_ms"])

    def test_summarize_detections_keeps_top_k(self):
        detections = [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.4, "class_id": 2},
            {"bbox": [0.5, 0.5, 0.1, 0.1], "confidence": 0.9, "class_id": 1},
        ]

        summary = summarize_detections(detections, limit=1)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["top"][0]["class_id"], 1)
        self.assertEqual(summary["top"][0]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
