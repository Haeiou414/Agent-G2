"""Download and verify the pinned Qwen base model used by the experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHTS_NAME = "model.safetensors"
WEIGHTS_SIZE = 3_087_467_144
WEIGHTS_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(model_dir: Path) -> None:
    weights = model_dir / WEIGHTS_NAME
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")
    actual_size = weights.stat().st_size
    if actual_size != WEIGHTS_SIZE:
        raise SystemExit(
            f"unexpected size for {weights}: {actual_size}; expected {WEIGHTS_SIZE}"
        )
    actual_hash = sha256(weights)
    if actual_hash != WEIGHTS_SHA256:
        raise SystemExit(
            f"unexpected SHA-256 for {weights}: {actual_hash}; expected {WEIGHTS_SHA256}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise SystemExit("install huggingface_hub before downloading the base model") from error
    output = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            local_dir=args.local_dir,
            max_workers=args.max_workers,
        )
    )
    verify_weights(output)
    print(f"downloaded and verified {REPO_ID}@{REVISION} to {output}")


if __name__ == "__main__":
    main()
