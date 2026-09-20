"""Exact action-vs-token prefix audit using the released checkpoint tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "sft_data" / "alfworld_sft_data.json"
DEFAULT_MODEL = "xiamoent/Agent-G2-alfworld-1.5b"


def format_expert_prefix(actions: list[str]) -> str:
    return "\n".join(f"Step {index}: {action}" for index, action in enumerate(actions, start=1))


def token_step_offsets(actions: list[str], token_length: Callable[[str], int]) -> tuple[list[int], int]:
    offsets = [token_length(format_expert_prefix(actions[:index])) for index in range(1, len(actions) + 1)]
    return offsets, offsets[-1] if offsets else 0


def token_based_prefix_steps(offsets: list[int], total_tokens: int, ratio: float) -> int:
    total_steps = len(offsets)
    if total_steps <= 1 or total_tokens <= 0:
        return 0
    target = max(0, min(math.floor(total_tokens * ratio), total_tokens))
    if target <= 0:
        steps = 0
    elif target >= total_tokens:
        steps = total_steps
    else:
        steps = next(
            (index for index, offset in enumerate(offsets, start=1) if offset >= target),
            total_steps,
        )
    return min(steps, total_steps - 1)


def action_based_prefix_steps(total_steps: int, ratio: float) -> int:
    if total_steps <= 1:
        return 0
    return min(math.ceil(total_steps * ratio), total_steps - 1)


def analyze(
    data_path: Path,
    token_length: Callable[[str], int],
) -> list[dict[str, Any]]:
    trajectories = json.loads(data_path.read_text(encoding="utf-8"))["trajectories"]
    cached = []
    for item in trajectories:
        actions = item["actions"]
        offsets, total_tokens = token_step_offsets(actions, token_length)
        cached.append((len(actions), offsets, total_tokens))

    rows = []
    for ratio in [index / 10 for index in range(1, 10)]:
        differences = []
        for total_steps, offsets, total_tokens in cached:
            token_steps = token_based_prefix_steps(offsets, total_tokens, ratio)
            action_steps = action_based_prefix_steps(total_steps, ratio)
            differences.append(token_steps - action_steps)
        mismatch_count = sum(value != 0 for value in differences)
        rows.append(
            {
                "ratio": ratio,
                "mismatch_count": mismatch_count,
                "mismatch_percent": 100.0 * mismatch_count / len(differences),
                "mean_abs_step_difference": sum(abs(value) for value in differences) / len(differences),
                "token_shallower_count": sum(value < 0 for value in differences),
                "token_deeper_count": sum(value > 0 for value in differences),
            }
        )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact tokenizer prefix-depth audit",
        "",
        f"Model tokenizer: `{payload['model_id']}` at revision `{payload['revision']}`  ",
        f"Tokenizer SHA-256: `{payload['tokenizer_sha256']}`  ",
        f"Dataset SHA-256: `{payload['dataset_sha256']}`",
        "",
        "This audit reproduces the released runtime's former token-based prefix calculation with the exact tokenizer, then compares it with the paper's action-count equation.",
        "",
        "| Ratio | Different trajectories | Mismatch | Token prefix shallower | Token prefix deeper | Mean absolute step difference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['ratio']:.1f} | {row['mismatch_count']:,} | {row['mismatch_percent']:.1f}% | {row['token_shallower_count']:,} | {row['token_deeper_count']:,} | {row['mean_abs_step_difference']:.3f} |"
        )
    half = next(row for row in payload["rows"] if row["ratio"] == 0.5)
    lines.extend(
        [
            "",
            f"At ratio 0.5, the two implementations choose different action prefixes for **{half['mismatch_count']:,}/3,553 trajectories ({half['mismatch_percent']:.1f}%)**.",
            "",
            "Generated with `python -m reproduction.analyze_tokenizer_prefix`. The model weights are not downloaded; only the tokenizer artifact is required.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reproduction" / "reports" / "tokenizer_prefix_audit.md",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: tokenizers and huggingface_hub") from exc

    revision = args.revision or HfApi().model_info(args.model).sha
    tokenizer_path = Path(hf_hub_download(args.model, "tokenizer.json", revision=revision))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_length = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    rows = analyze(args.data, token_length)
    payload = {
        "model_id": args.model,
        "revision": revision,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote exact tokenizer audit to {args.output}")


if __name__ == "__main__":
    main()
