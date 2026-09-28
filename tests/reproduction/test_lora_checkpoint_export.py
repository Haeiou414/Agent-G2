import unittest
from pathlib import Path

from reproduction.audit_config import REPO_ROOT


class LoraCheckpointExportTests(unittest.TestCase):
    def test_single_gpu_no_shard_uses_full_peft_state(self) -> None:
        source = (REPO_ROOT / "verl" / "workers" / "fsdp_workers.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if self.world_size == 1:", source)
        self.assertIn("get_peft_model_state_dict(", source)
        self.assertIn("state_dict=checkpoint_state", source)
        self.assertIn("mmap=True", source)
        self.assertIn("weights_only=True", source)
        self.assertIn('raise RuntimeError("LoRA adapter state is empty")', source)


if __name__ == "__main__":
    unittest.main()
