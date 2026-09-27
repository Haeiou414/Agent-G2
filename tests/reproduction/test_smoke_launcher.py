import os
import subprocess
import unittest

from reproduction.audit_config import REPO_ROOT


LAUNCHER = REPO_ROOT / "reproduction" / "run_alfworld_smoke.sh"


class SmokeLauncherTests(unittest.TestCase):
    def test_dry_run_uses_low_cost_verified_shape(self) -> None:
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "3"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "DRY_RUN": "1"},
        )
        command = completed.stdout
        self.assertIn("actor_rollout_ref.model.lora_rank=16", command)
        self.assertIn("env.max_steps=5", command)
        self.assertIn("trainer.total_training_steps=8", command)
        self.assertIn("trainer.test_freq=8", command)
        self.assertIn("trainer.val_before_train=false", command)
        self.assertIn("data.val_batch_size=16", command)
        self.assertIn("seed3_smoke", command)


if __name__ == "__main__":
    unittest.main()
