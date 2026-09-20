import json
import tempfile
import unittest
from pathlib import Path

from reproduction.capture_run_manifest import write_manifest
from reproduction.finalize_run_manifest import finalize_manifest


class RunManifestTests(unittest.TestCase):
    def test_writes_machine_readable_provenance_and_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "run_manifest.json"
            command = ["bash", "train.sh", "trainer.total_training_steps=8"]
            write_manifest(output, "gmsv", 7, command)

            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["method"], "gmsv")
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["command"], command)
            self.assertEqual(len(manifest["git"]["commit"]), 40)
            self.assertEqual(len(manifest["git"]["diff_sha256"]), 64)
            self.assertEqual(len(manifest["data"]["expert_trajectories_sha256"]), 64)
            self.assertTrue(output.with_name("working_tree.patch").exists())

            finalized = finalize_manifest(output, 0)
            self.assertEqual(finalized["run_status"], "completed")
            self.assertEqual(finalized["exit_code"], 0)
            self.assertGreaterEqual(finalized["elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
