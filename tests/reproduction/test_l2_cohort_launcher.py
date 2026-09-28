import os
import subprocess
import unittest

from reproduction.audit_config import REPO_ROOT


LAUNCHER = REPO_ROOT / "reproduction" / "run_alfworld_l2_cohort.sh"


class L2CohortLauncherTests(unittest.TestCase):
    def dry_run(self, cohort: str, *seeds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER), cohort, *(str(seed) for seed in seeds)],
            capture_output=True,
            text=True,
            env={**os.environ, "DRY_RUN": "1"},
        )

    def test_primary_cohort_uses_one_tag_and_expected_order(self) -> None:
        completed = self.dry_run("primary", 1)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        labels = [
            "method=grpo seed=1 tag=l2-v1",
            "method=target_acc seed=1 tag=l2-v1",
            "method=gmsv seed=1 tag=l2-v1",
        ]
        positions = [output.index(label) for label in labels]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(output.count("trainer.total_training_steps=80"), 3)
        self.assertEqual(output.count("trainer.max_actor_ckpt_to_keep=1"), 3)

    def test_ablation_cohort_accepts_multiple_seeds(self) -> None:
        completed = self.dry_run("ablations", 1, 2)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("=== L2 cohort=ablations"), 6)
        for method in ("fixed_sigma", "no_aux_sft", "deterministic_mean"):
            self.assertIn(f"method={method} seed=1", completed.stdout)
            self.assertIn(f"method={method} seed=2", completed.stdout)

    def test_rejects_unknown_cohort_and_invalid_seed(self) -> None:
        unknown = self.dry_run("everything", 1)
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown cohort", unknown.stderr)

        invalid = self.dry_run("primary", -1)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("SEED must", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
