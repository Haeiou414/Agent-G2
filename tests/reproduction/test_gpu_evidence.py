import json
import unittest
from pathlib import Path

from reproduction.audit_config import REPO_ROOT


EVIDENCE = REPO_ROOT / "reproduction" / "evidence"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class GpuEvidenceTests(unittest.TestCase):
    def test_eight_step_smoke_is_complete_and_has_final_validation(self) -> None:
        root = EVIDENCE / "rtx4090_smoke8"
        manifest = read_json(root / "run_manifest.json")
        rows = read_jsonl(root / "metrics.jsonl")
        runtime = read_json(root / "gmsv_runtime_state.json")

        self.assertEqual(manifest["run_status"], "completed")
        self.assertEqual(manifest["exit_code"], 0)
        self.assertEqual([row["step"] for row in rows], list(range(1, 9)))
        self.assertIn("val/success_rate", rows[-1]["metrics"])
        self.assertEqual(runtime["loaded_global_step"], 8)
        self.assertEqual(runtime["mu_global"], 1.0)
        self.assertEqual(set(runtime["scoreboard"]), {"1", "2", "3"})
        self.assertIn("NVIDIA GeForce RTX 4090", manifest["host"]["nvidia_smi"][0])

    def test_resume_advances_one_step_and_preserves_scheduler_state(self) -> None:
        root = EVIDENCE / "rtx4090_resume9"
        manifest = read_json(root / "run_manifest.json")
        rows = read_jsonl(root / "metrics.jsonl")
        runtime = read_json(root / "gmsv_runtime_state.json")

        self.assertEqual(manifest["run_status"], "completed")
        self.assertEqual(manifest["exit_code"], 0)
        self.assertEqual([row["step"] for row in rows], [9])
        self.assertEqual(runtime["loaded_global_step"], 9)
        self.assertEqual(runtime["mu_global"], 1.0)
        command = manifest["command"]
        self.assertIn("trainer.resume_mode=resume_path", command)
        self.assertTrue(any(token.endswith("/global_step_8") for token in command))

    def test_exported_adapter_inventory_is_nonempty_and_reloadable(self) -> None:
        root = EVIDENCE / "rtx4090_resume9"
        config = read_json(root / "adapter_config.json")
        inventory = read_json(root / "adapter_inventory.json")

        self.assertEqual(config["peft_type"], "LORA")
        self.assertEqual(config["r"], 16)
        self.assertEqual(config["lora_alpha"], 16)
        self.assertEqual(inventory["tensor_count"], 392)
        self.assertEqual(inventory["parameter_count"], 18_464_768)
        self.assertEqual(
            inventory["nonzero_parameter_count"], inventory["parameter_count"]
        )
        self.assertGreater(inventory["bytes"], 70_000_000)
        self.assertTrue(inventory["peft_reload_verified"])

    def test_restart_failure_is_kept_as_excluded_evidence(self) -> None:
        manifest = read_json(EVIDENCE / "restart_failure" / "run_manifest.json")
        self.assertEqual(manifest["run_status"], "failed")
        self.assertNotEqual(manifest["exit_code"], 0)
        self.assertGreater(manifest["elapsed_seconds"], 2_000)


if __name__ == "__main__":
    unittest.main()
