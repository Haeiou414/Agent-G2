import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from reproduction.audit_config import REPO_ROOT


BOOTSTRAP = REPO_ROOT / "reproduction" / "bootstrap_autodl.sh"
ENV_FILE = REPO_ROOT / "reproduction" / "autodl_env.sh"
ALFWORLD_INSTALLER = REPO_ROOT / "reproduction" / "install_alfworld_autodl.sh"


class AutoDlBootstrapTests(unittest.TestCase):
    def test_official_cuda_stack_is_pinned(self) -> None:
        script = BOOTSTRAP.read_text(encoding="utf-8")
        for expected in (
            "python=3.12",
            "torch==2.6.0",
            "cu124",
            "flash-attn==2.7.4.post1",
            "vllm==0.8.5",
            "transformers==4.51.1",
            "gymnasium==0.29.1",
            "stable-baselines3==2.6.0",
            "wandb==0.19.11",
            "google-api-core==2.24.2",
            "proto-plus==1.26.1",
            "opentelemetry-exporter-prometheus==0.47b0",
        ):
            self.assertIn(expected, script)

    def test_large_caches_default_to_autodl_data_disk(self) -> None:
        env_script = ENV_FILE.read_text(encoding="utf-8")
        self.assertIn("/root/autodl-tmp/agent-g2-data", env_script)
        self.assertIn("ALFWORLD_DATA", env_script)
        self.assertIn("HF_HOME", env_script)
        self.assertIn("PIP_CACHE_DIR", env_script)
        self.assertIn("TMPDIR", env_script)
        self.assertIn("CONDA_ENVS_PATH", env_script)
        self.assertIn("CONDA_PKGS_DIRS", env_script)

    def test_alfworld_build_is_memory_bounded_and_source_verified(self) -> None:
        installer = ALFWORLD_INSTALLER.read_text(encoding="utf-8")
        self.assertIn("AGENT_G2_BUILD_JOBS", installer)
        self.assertIn("sha256sum --check", installer)
        self.assertIn("fast_downward_limited_jobs.patch", installer)
        self.assertIn("alfred.pddl", installer)
        self.assertIn("alfred.twl2", installer)

    def test_setup_records_environment_evidence(self) -> None:
        script = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("gpu-preflight.json", script)
        self.assertIn("pip-freeze.txt", script)
        self.assertIn("nvidia-smi.csv", script)
        self.assertIn("network-preflight.txt", script)
        self.assertIn('"${PIP[@]}" check', script)

    def test_rejects_blackwell_before_dependency_download(self) -> None:
        script = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertLess(script.index("GPU_COMPUTE_CAP="), script.index('"${PIP[@]}" install'))
        self.assertIn("Blackwell or newer", script)

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_bin = Path(temp_dir)
            commands = {
                "uname": "#!/bin/sh\necho Linux\n",
                "nvidia-smi": "#!/bin/sh\necho 12.0\n",
                "conda": "#!/bin/sh\necho conda-should-not-run >&2\nexit 99\n",
            }
            for name, body in commands.items():
                executable = fake_bin / name
                executable.write_text(body, encoding="utf-8")
                executable.chmod(0o755)
            environment = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
            result = subprocess.run(
                ["bash", str(BOOTSTRAP)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Blackwell or newer", result.stderr)
        self.assertNotIn("conda-should-not-run", result.stderr)


if __name__ == "__main__":
    unittest.main()
