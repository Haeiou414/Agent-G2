import json
import tempfile
import unittest
from pathlib import Path

from reproduction.summarize_results import collect_runs, render_report, summarize


COMMON_COMMAND = [
    "bash",
    "train.sh",
    "actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct",
    "actor_rollout_ref.model.lora_rank=16",
    "actor_rollout_ref.model.lora_alpha=16",
    "data.train_batch_size=4",
    "data.max_prompt_length=7000",
    "data.max_response_length=512",
    "env.max_steps=20",
    "env.rollout.n=4",
    "trainer.n_gpus_per_node=1",
    "trainer.total_training_steps=80",
    "trainer.test_freq=10",
]


def write_run(root: Path, name: str, method: str, seed: int, success: float, *, completed: bool = True) -> None:
    run_dir = root / name
    run_dir.mkdir()
    manifest = {
        "method": method,
        "seed": seed,
        "command": COMMON_COMMAND,
        "run_status": "completed" if completed else "failed",
        "exit_code": 0 if completed else 1,
        "elapsed_seconds": 3600,
        "git": {"commit": "a" * 40, "diff_sha256": "b" * 64},
        "host": {"nvidia_smi": ["0, GPU, 24576, driver"]},
        "data": {"expert_trajectories_sha256": "c" * 64},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {"step": 1, "metrics": {"train/total_prompt_count": 4, "train/expert_match_rate": 0.75}},
        {"step": 80, "metrics": {"val/success_rate": success}},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class ResultSummaryTests(unittest.TestCase):
    def test_excludes_failed_runs_and_renders_completed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, "gmsv-1", "gmsv", 1, 0.75)
            write_run(root, "gmsv-2", "gmsv", 2, 0.85)
            write_run(root, "failed", "grpo", 1, 0.20, completed=False)
            runs, excluded = collect_runs(root)
            report = render_report(runs, excluded)
            self.assertEqual(len(runs), 2)
            self.assertEqual(len(excluded), 1)
            self.assertIn("80.00%", report)
            self.assertIn("25.00%", report)
            self.assertIn("2560", report)
            self.assertIn("run did not finish successfully", report)

    def test_three_seed_interval_uses_student_t(self) -> None:
        mean, std, interval = summarize([0.70, 0.80, 0.90])
        self.assertAlmostEqual(mean, 0.8)
        self.assertAlmostEqual(std, 0.1)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval[0], 0.8 - 4.303 * 0.1 / (3**0.5))

    def test_rejects_mixed_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, "gmsv", "gmsv", 1, 0.8)
            write_run(root, "grpo", "grpo", 1, 0.7)
            manifest_path = root / "grpo" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["command"] = [
                token.replace("trainer.total_training_steps=80", "trainer.total_training_steps=40")
                for token in manifest["command"]
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            rows = [{"step": 40, "metrics": {"val/success_rate": 0.7}}]
            (root / "grpo" / "metrics.jsonl").write_text(
                json.dumps(rows[0]) + "\n", encoding="utf-8"
            )
            runs, excluded = collect_runs(root)
            self.assertEqual(excluded, [])
            with self.assertRaisesRegex(ValueError, "budget mismatch"):
                render_report(runs, excluded)


if __name__ == "__main__":
    unittest.main()
