import json
import tempfile
import unittest
from pathlib import Path

from reproduction.capture_checkpoint_identity import checkpoint_identity, file_sha256


class CheckpointIdentityTests(unittest.TestCase):
    def test_records_small_identity_files_and_weight_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "snapshots" / ("d" * 40)
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"weights")
            identity = checkpoint_identity(checkpoint)
            self.assertEqual(identity["huggingface_snapshot_revision"], "d" * 40)
            self.assertEqual(
                identity["identity_files"]["config.json"]["sha256"],
                file_sha256(checkpoint / "config.json"),
            )
            self.assertEqual(identity["total_weight_bytes"], 7)
            self.assertEqual(identity["weight_shards"][0]["name"], "model-00001-of-00002.safetensors")

    def test_rejects_missing_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            checkpoint_identity(Path("/definitely/not/a/checkpoint"))


if __name__ == "__main__":
    unittest.main()
