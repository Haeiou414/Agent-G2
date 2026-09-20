"""Mark an experiment manifest as completed or failed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def finalize_manifest(path: Path, exit_code: int) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    finished = datetime.now(timezone.utc)
    started = datetime.fromisoformat(manifest["started_at_utc"])
    elapsed_seconds = max(0.0, (finished - started).total_seconds())
    manifest["finished_at_utc"] = finished.isoformat()
    manifest["elapsed_seconds"] = elapsed_seconds
    manifest["exit_code"] = exit_code
    manifest["run_status"] = "completed" if exit_code == 0 else "failed"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    manifest = finalize_manifest(args.manifest, args.exit_code)
    print(f"run {manifest['run_status']}; updated {args.manifest}")


if __name__ == "__main__":
    main()
