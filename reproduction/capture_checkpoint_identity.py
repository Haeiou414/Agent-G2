"""Record a local Hugging Face checkpoint identity without hashing weight shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IDENTITY_FILES = ("config.json", "tokenizer.json", "model.safetensors.index.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    files = {}
    for name in IDENTITY_FILES:
        path = checkpoint / name
        if path.exists():
            files[name] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    shards = [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in sorted(checkpoint.glob("*.safetensors"))
    ]
    revision = checkpoint.name if checkpoint.parent.name == "snapshots" else None
    return {
        "checkpoint_path": str(checkpoint),
        "huggingface_snapshot_revision": revision,
        "identity_files": files,
        "weight_shards": shards,
        "total_weight_bytes": sum(item["size_bytes"] for item in shards),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = checkpoint_identity(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote checkpoint identity to {args.output}")


if __name__ == "__main__":
    main()
