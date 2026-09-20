import unittest
from pathlib import Path

from reproduction.audit_config import REPO_ROOT


BOOTSTRAP = REPO_ROOT / "reproduction" / "bootstrap_autodl.sh"
ENV_FILE = REPO_ROOT / "reproduction" / "autodl_env.sh"


class AutoDlBootstrapTests(unittest.TestCase):
    def test_official_cuda_stack_is_pinned(self) -> None:
        script = BOOTSTRAP.read_text(encoding="utf-8")
        for expected in (
            "python=3.12",
            "torch==2.6.0",
            "cu124",
            "flash-attn==2.7.4.post1",
            "vllm==0.8.5",
            "gymnasium==0.29.1",
            "stable-baselines3==2.6.0",
        ):
            self.assertIn(expected, script)

    def test_large_caches_default_to_autodl_data_disk(self) -> None:
        env_script = ENV_FILE.read_text(encoding="utf-8")
        self.assertIn("/root/autodl-tmp/agent-g2-data", env_script)
        self.assertIn("ALFWORLD_DATA", env_script)
        self.assertIn("HF_HOME", env_script)
        self.assertIn("PIP_CACHE_DIR", env_script)

    def test_setup_records_environment_evidence(self) -> None:
        script = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("gpu-preflight.json", script)
        self.assertIn("pip-freeze.txt", script)
        self.assertIn("nvidia-smi.csv", script)
        self.assertIn('"${PIP[@]}" check', script)


if __name__ == "__main__":
    unittest.main()
