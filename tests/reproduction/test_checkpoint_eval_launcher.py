import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from reproduction.audit_config import REPO_ROOT


LAUNCHER = REPO_ROOT / "reproduction" / "run_alfworld_checkpoint_eval.sh"


class CheckpointEvalLauncherTests(unittest.TestCase):
    def test_dry_run_disables_training_resume_and_validation_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                ["bash", str(LAUNCHER), temp_dir, "3"],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "DRY_RUN": "1"},
            )
        command = completed.stdout
        self.assertIn("trainer.val_only=true", command)
        self.assertIn("trainer.resume_mode=disable", command)
        self.assertIn("trainer.default_local_dir=outputs/", command)
        self.assertIn("gmsv.apply_on_validation=false", command)
        self.assertIn("env.rollout.n=1", command)
        self.assertIn("trainer.logger=\\[console\\,jsonl\\]", command)

    def test_missing_checkpoint_is_rejected(self) -> None:
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "/definitely/not/a/checkpoint", "1"],
            capture_output=True,
            text=True,
            env={**os.environ, "DRY_RUN": "1"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("does not exist", completed.stderr)

    def test_launcher_preserves_eval_exit_code_and_console_log(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('tee "$RUN_DIR/console.log"', script)
        self.assertIn("RUN_EXIT_CODE=${PIPESTATUS[0]}", script)
        self.assertIn("VERL_RESOLVED_CONFIG_PATH", script)


if __name__ == "__main__":
    unittest.main()
