"""Capture provenance before a GPU experiment starts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAMES = ("torch", "vllm", "ray", "flash-attn", "alfworld", "transformers")
EXPERT_DATA = REPO_ROOT / "sft_data" / "alfworld_sft_data.json"


def _run(command: list[str], cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def collect_manifest(method: str, seed: int, command: list[str]) -> tuple[dict[str, Any], str]:
    patch = _run(["git", "diff", "--binary", "HEAD"])
    versions: dict[str, str] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue

    gpu_rows = ""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        gpu_rows = _run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )

    manifest = {
        "schema_version": 2,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "seed": seed,
        "command": command,
        "git": {
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_porcelain": _run(["git", "status", "--porcelain=v1"]),
            "diff_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": versions,
            "nvidia_smi": gpu_rows.splitlines() if gpu_rows else [],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "data": {
            "expert_trajectories_path": str(EXPERT_DATA.relative_to(REPO_ROOT)),
            "expert_trajectories_sha256": hashlib.sha256(EXPERT_DATA.read_bytes()).hexdigest(),
        },
    }
    return manifest, patch


def write_manifest(output: Path, method: str, seed: int, command: list[str]) -> None:
    manifest, patch = collect_manifest(method, seed, command)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_name("working_tree.patch").write_text(patch + ("\n" if patch else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("--command must contain the launched command")
    write_manifest(args.output, args.method, args.seed, command)
    print(f"wrote run provenance to {args.output}")


if __name__ == "__main__":
    main()
