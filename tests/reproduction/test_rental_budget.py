import json
import tempfile
import unittest
from pathlib import Path

from reproduction.estimate_rental_budget import estimate_run, event_count, render


class RentalBudgetTests(unittest.TestCase):
    def test_counts_final_non_periodic_event(self) -> None:
        self.assertEqual(event_count(80, 10), 8)
        self.assertEqual(event_count(83, 10), 9)
        self.assertEqual(event_count(80, -1), 0)

    def test_estimate_separates_base_validation_save_and_overhead(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest = {
                "run_status": "completed",
                "exit_code": 0,
                "elapsed_seconds": 150.0,
                "command": [
                    "trainer.total_training_steps=2",
                    "trainer.n_gpus_per_node=1",
                ],
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows = [
                {"step": 1, "metrics": {"timing_s/step": 20.0}},
                {
                    "step": 2,
                    "metrics": {
                        "timing_s/step": 70.0,
                        "timing_s/testing": 30.0,
                        "timing_s/save_checkpoint": 20.0,
                    },
                },
            ]
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            estimate = estimate_run(
                run_dir,
                target_steps=4,
                test_frequency=2,
                save_frequency=4,
            )

        self.assertEqual(estimate.fixed_overhead_seconds, 60.0)
        self.assertEqual(estimate.base_step_seconds, 20.0)
        self.assertEqual(estimate.validation_seconds, 30.0)
        self.assertEqual(estimate.checkpoint_seconds, 20.0)
        self.assertEqual(estimate.seconds_per_run, 220.0)
        self.assertIn("¥", render(estimate, hourly_price=2.5))

    def test_rejects_incomplete_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"run_status": "failed", "exit_code": 1}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "completed"):
                estimate_run(run_dir)


if __name__ == "__main__":
    unittest.main()
