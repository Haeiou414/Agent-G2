"""Download the released Agent-G2 ALFWorld checkpoint at a pinned revision."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ID = "xiamoent/Agent-G2-alfworld-1.5b"
REVISION = "4556a9bfdf84320267c3a9e9e7b85732ba2835ba"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise SystemExit("install huggingface_hub before downloading the checkpoint") from error
    output = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=args.local_dir,
    )
    print(f"downloaded {REPO_ID}@{REVISION} to {output}")


if __name__ == "__main__":
    main()
