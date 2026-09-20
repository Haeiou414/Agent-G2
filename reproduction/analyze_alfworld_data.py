"""Reproducible audit of the released ALFWorld expert trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "sft_data" / "alfworld_sft_data.json"
TASK_LABELS = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
}


def balanced_thresholds(action_counts: list[int], groups: int) -> list[int]:
    """Pure-Python equivalent of the upstream balanced length clustering."""

    counts = Counter(action_counts)
    unique = sorted(counts)
    if groups < 1 or len(unique) < groups:
        raise ValueError("groups must not exceed the number of unique lengths")
    target = len(action_counts) / groups
    best: tuple[tuple[float, float], tuple[int, ...]] | None = None
    for boundaries in combinations(range(1, len(unique)), groups - 1):
        positions = (0, *boundaries, len(unique))
        sizes = [
            sum(counts[length] for length in unique[positions[i] : positions[i + 1]])
            for i in range(groups)
        ]
        score = (max(sizes) - min(sizes), sum(abs(size - target) for size in sizes))
        if best is None or score < best[0]:
            best = (score, boundaries)
    assert best is not None
    boundaries = best[1]
    return [unique[index - 1] for index in boundaries] + [unique[-1]]


def cluster_for_length(length: int, thresholds: list[int]) -> int:
    return next(index for index, threshold in enumerate(thresholds, start=1) if length <= threshold)


def formatted_word_counts(actions: list[str]) -> list[int]:
    """Approximate token mass while preserving the runtime's Step N formatting."""

    return [len(f"Step {index}: {action}".split()) for index, action in enumerate(actions, start=1)]


def prefix_proxy_difference(actions: list[str], ratio: float) -> int:
    """Return word-proxy prefix steps minus the paper's action-based steps."""

    length = len(actions)
    if length <= 1:
        return 0
    paper_steps = min(math.ceil(ratio * length), length - 1)
    word_counts = formatted_word_counts(actions)
    target = math.floor(ratio * sum(word_counts))
    proxy_steps = 0
    cumulative = 0
    if target > 0:
        for index, words in enumerate(word_counts, start=1):
            cumulative += words
            if cumulative >= target:
                proxy_steps = index
                break
    proxy_steps = min(proxy_steps, length - 1)
    return proxy_steps - paper_steps


def analyze(path: Path = DEFAULT_DATA) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    trajectories = payload["trajectories"]
    lengths = [len(item["actions"]) for item in trajectories]
    thresholds = balanced_thresholds(lengths, groups=3)
    cluster_sizes = Counter(cluster_for_length(length, thresholds) for length in lengths)
    task_counts = Counter(item["task_type"] for item in trajectories)
    cluster_task_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for item, length in zip(trajectories, lengths):
        cluster_task_counts[cluster_for_length(length, thresholds)][item["task_type"]] += 1

    proxy_rows = []
    for ratio in [index / 10 for index in range(1, 10)]:
        differences = [prefix_proxy_difference(item["actions"], ratio) for item in trajectories]
        mismatch_count = sum(difference != 0 for difference in differences)
        proxy_rows.append(
            {
                "ratio": ratio,
                "mismatch_count": mismatch_count,
                "mismatch_percent": 100.0 * mismatch_count / len(differences),
                "mean_abs_step_difference": statistics.fmean(abs(value) for value in differences),
            }
        )

    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "trajectory_count": len(trajectories),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": statistics.fmean(lengths),
        "length_median": statistics.median(lengths),
        "length_counts": dict(sorted(Counter(lengths).items())),
        "task_counts": dict(task_counts),
        "thresholds": thresholds,
        "cluster_sizes": dict(cluster_sizes),
        "cluster_task_counts": {key: dict(value) for key, value in cluster_task_counts.items()},
        "prefix_proxy": proxy_rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld expert-data audit",
        "",
        f"Dataset SHA-256: `{result['sha256']}`",
        "",
        "## Dataset summary",
        "",
        f"The released file contains **{result['trajectory_count']:,} expert trajectories**. Expert action count ranges from **{result['length_min']} to {result['length_max']}**, with mean **{result['length_mean']:.2f}** and median **{result['length_median']:.0f}**.",
        "",
        "| Task type | Trajectories | Share |",
        "|---|---:|---:|",
    ]
    total = int(result["trajectory_count"])
    for task_type, count in sorted(result["task_counts"].items(), key=lambda item: -item[1]):
        lines.append(f"| {TASK_LABELS.get(task_type, task_type)} | {count:,} | {100 * count / total:.1f}% |")

    lines.extend(
        [
            "",
            "## Paper K=3 difficulty clusters",
            "",
            "The upstream balanced-by-length rule yields thresholds **[5, 7, 15]**.",
            "",
            "| Cluster | Expert actions | Trajectories | Share |",
            "|---:|---|---:|---:|",
        ]
    )
    lower = int(result["length_min"]) - 1
    for cluster_id, upper in enumerate(result["thresholds"], start=1):
        count = result["cluster_sizes"][cluster_id]
        lines.append(f"| {cluster_id} | {lower + 1}–{upper} | {count:,} | {100 * count / total:.1f}% |")
        lower = upper

    lines.extend(
        [
            "",
            "## Action-based vs. text-mass prefix depth",
            "",
            "The paper defines prefix depth from expert **action count**. The released runtime previously used tokenizer length. Without downloading the Qwen tokenizer, the following diagnostic uses whitespace word count on the same `Step N: action` formatting as a transparent proxy; it demonstrates sensitivity to action text length but is not an exact tokenizer-level measurement.",
            "",
            "| Guidance ratio | Different prefix steps | Mismatch rate | Mean absolute step difference |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["prefix_proxy"]:
        lines.append(
            f"| {row['ratio']:.1f} | {row['mismatch_count']:,} | {row['mismatch_percent']:.1f}% | {row['mean_abs_step_difference']:.3f} |"
        )
    half = next(row for row in result["prefix_proxy"] if row["ratio"] == 0.5)
    lines.extend(
        [
            "",
            f"At guidance ratio 0.5, the text-mass proxy changes the chosen prefix step for **{half['mismatch_count']:,}/{total:,} trajectories ({half['mismatch_percent']:.1f}%)**. This supports treating the action/token distinction as an experiment-affecting semantic discrepancy rather than a cosmetic implementation detail.",
            "The follow-up [`tokenizer_prefix_audit.md`](tokenizer_prefix_audit.md) replaces this proxy with the exact tokenizer from a pinned public-checkpoint revision.",
            "",
            "![Expert trajectory length distribution](../figures/alfworld_expert_lengths.svg)",
            "",
            "Generated by `python -m reproduction.analyze_alfworld_data`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_svg(result: dict[str, Any]) -> str:
    width, height = 960, 500
    x0, y0, plot_width, plot_height = 82, 78, 820, 330
    counts = {int(key): int(value) for key, value in result["length_counts"].items()}
    lengths = sorted(counts)
    max_count = max(counts.values())
    gap = 6
    bar_width = (plot_width - gap * (len(lengths) - 1)) / len(lengths)
    colors = ("#2563eb", "#f59e0b", "#dc2626")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="480" y="34" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="700">ALFWorld expert trajectory lengths</text>',
        '<text x="480" y="58" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">3,553 released trajectories · paper K=3 thresholds: 5 / 7 / 15 actions</text>',
    ]
    for tick in range(5):
        value = max_count * tick / 4
        y = y0 + plot_height * (1 - tick / 4)
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_width}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x0 - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{value:.0f}</text>')
    for index, length in enumerate(lengths):
        cluster = cluster_for_length(length, list(result["thresholds"]))
        count = counts[length]
        x = x0 + index * (bar_width + gap)
        bar_height = plot_height * count / max_count
        y = y0 + plot_height - bar_height
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="2" fill="{colors[cluster - 1]}"/>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y0 + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#334155">{length}</text>')
    parts.extend(
        [
            f'<text x="{x0 + plot_width / 2}" y="{y0 + plot_height + 52}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Expert action count</text>',
            f'<text x="20" y="{y0 + plot_height / 2}" transform="rotate(-90 20 {y0 + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Trajectories</text>',
        ]
    )
    legend_y = 478
    for index, label in enumerate(("cluster 1: 3–5", "cluster 2: 6–7", "cluster 3: 8–15")):
        x = 280 + index * 190
        parts.append(f'<rect x="{x}" y="{legend_y - 11}" width="20" height="12" rx="2" fill="{colors[index]}"/>')
        parts.append(f'<text x="{x + 28}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#334155">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reproduction" / "reports" / "alfworld_data_audit.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=REPO_ROOT / "reproduction" / "figures" / "alfworld_expert_lengths.svg",
    )
    args = parser.parse_args()
    result = analyze(args.data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result), encoding="utf-8")
    args.figure.write_text(render_svg(result), encoding="utf-8")
    print(f"wrote dataset audit to {args.report}")
    print(f"wrote length figure to {args.figure}")


if __name__ == "__main__":
    main()
