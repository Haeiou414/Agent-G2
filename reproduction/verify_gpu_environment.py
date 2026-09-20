"""Preflight checks for an Agent-G2 CUDA training host."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_DISTRIBUTIONS = ("torch", "vllm", "ray", "flash-attn", "alfworld")


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    devices = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        name, memory_mib, driver = parts
        devices.append(
            {
                "name": name,
                "memory_mib": int(memory_mib),
                "driver": driver,
            }
        )
    return devices


def distribution_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return versions, missing


def check_environment() -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
    }
    failures: list[str] = []

    if platform.system() != "Linux":
        failures.append("training host must be Linux for the official CUDA/vLLM stack")
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        failures.append("Python must be between 3.10 and 3.12")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        report["gpus"] = []
        failures.append("nvidia-smi is unavailable")
    else:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            report["gpus"] = []
            failures.append(f"nvidia-smi failed: {completed.stderr.strip()}")
        else:
            report["gpus"] = parse_nvidia_smi(completed.stdout)
            if not report["gpus"]:
                failures.append("no NVIDIA GPU was detected")
            elif max(device["memory_mib"] for device in report["gpus"]) < 22000:
                failures.append("the single-GPU recipe expects approximately 24 GB VRAM")

    versions, missing = distribution_versions()
    report["packages"] = versions
    if missing:
        failures.append("missing Python distributions: " + ", ".join(missing))

    disk = shutil.disk_usage(Path.cwd())
    report["workspace_disk_free_gib"] = round(disk.free / 1024**3, 2)
    if disk.free < 40 * 1024**3:
        failures.append("less than 40 GiB free disk space remains")

    if "torch" in versions:
        try:
            import torch

            report["torch_cuda_available"] = bool(torch.cuda.is_available())
            report["torch_cuda_version"] = torch.version.cuda
            if not torch.cuda.is_available():
                failures.append("PyTorch cannot access CUDA")
        except Exception as exc:  # pragma: no cover - host-specific import failure
            failures.append(f"PyTorch import/CUDA check failed: {exc}")

    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    report, failures = check_environment()
    payload = {"ok": not failures, "report": report, "failures": failures}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Agent-G2 GPU preflight:", "PASS" if not failures else "FAIL")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        for failure in failures:
            print(f"- {failure}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
