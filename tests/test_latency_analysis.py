import json
import tempfile
import unittest
from pathlib import Path

from macos_dualbox_aim.v1.latency_analysis import (
    capture_run,
    compare_labels,
    load_records,
    summarize_run,
    write_sample,
)


class _StepClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 0.25
        return current


class LatencyAnalysisTests(unittest.TestCase):
    def test_summarize_run_uses_tuner_avg_values_for_primary_average(self):
        records = [
            {
                "label": "v1.0.0",
                "run": "run1",
                "latency": {
                    "avg": {"program_total_ms": 10.0, "queue_wait_ms": 2.0},
                    "current": {"program_total_ms": 12.0, "queue_wait_ms": 4.0},
                    "counters": {"frames_captured": 100, "frames_dropped": 2},
                },
            },
            {
                "label": "v1.0.0",
                "run": "run1",
                "latency": {
                    "avg": {"program_total_ms": 14.0, "queue_wait_ms": 6.0},
                    "current": {"program_total_ms": 18.0, "queue_wait_ms": 8.0},
                    "counters": {"frames_captured": 200, "frames_dropped": 6},
                },
            },
        ]

        summary = summarize_run(records)

        self.assertEqual(summary.label, "v1.0.0")
        self.assertEqual(summary.run, "run1")
        self.assertEqual(summary.sample_count, 2)
        self.assertEqual(summary.counters["frames_dropped"], 6)
        self.assertAlmostEqual(summary.metrics["program_total_ms"].avg_ms, 12.0)
        self.assertAlmostEqual(summary.metrics["queue_wait_ms"].avg_ms, 4.0)

    def test_compare_labels_reports_candidate_delta_and_percent_change(self):
        baseline = summarize_run([
            {"label": "v1.0.0", "run": "a", "latency": {"avg": {"program_total_ms": 10.0}, "counters": {}}},
            {"label": "v1.0.0", "run": "a", "latency": {"avg": {"program_total_ms": 14.0}, "counters": {}}},
        ])
        candidate = summarize_run([
            {"label": "v1.1.0", "run": "b", "latency": {"avg": {"program_total_ms": 8.0}, "counters": {}}},
            {"label": "v1.1.0", "run": "b", "latency": {"avg": {"program_total_ms": 10.0}, "counters": {}}},
        ])

        rows = compare_labels([baseline, candidate], "v1.0.0", "v1.1.0", ["program_total_ms"])

        self.assertEqual(rows[0]["metric"], "program_total_ms")
        self.assertAlmostEqual(rows[0]["baseline_avg_ms"], 12.0)
        self.assertAlmostEqual(rows[0]["candidate_avg_ms"], 9.0)
        self.assertAlmostEqual(rows[0]["baseline_std_ms"], 0.0)
        self.assertAlmostEqual(rows[0]["candidate_std_ms"], 0.0)
        self.assertAlmostEqual(rows[0]["delta_ms"], -3.0)
        self.assertAlmostEqual(rows[0]["change_pct"], -25.0)

    def test_write_sample_and_load_records_round_trip_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            write_sample(
                path,
                label="v1.1.0",
                run="run1",
                sample_index=0,
                url="http://127.0.0.1:8765/api/config",
                snapshot={"latency": {"avg": {"program_total_ms": 7.0}}},
            )

            records = load_records([str(path)])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "v1.1.0")
        self.assertEqual(records[0]["run"], "run1")
        self.assertEqual(records[0]["latency"]["avg"]["program_total_ms"], 7.0)
        json.dumps(records[0])

    def test_capture_rejects_label_that_does_not_match_tuner_runtime_version(self):
        def fetcher(_url, _timeout):
            return {
                "runtime": {"version": "1.1.0"},
                "latency": {"avg": {"program_total_ms": 7.0}},
            }

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not match tuner runtime version"):
                capture_run(
                    label="v1.0.0",
                    run="run1",
                    duration_s=1.0,
                    interval_s=0.5,
                    out_dir=directory,
                    fetcher=fetcher,
                    sleeper=lambda _seconds: None,
                    clock=_StepClock(),
                )


if __name__ == "__main__":
    unittest.main()
