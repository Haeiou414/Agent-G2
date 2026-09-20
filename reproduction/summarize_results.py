"""Validate completed ALFWorld runs and produce a comparison report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BUDGET_KEYS = (
    "actor_rollout_ref.model.path",
    "data.train_batch_size",
    "data.max_prompt_length",
    "env.rollout.n",
    "trainer.n_gpus_per_node",
    "trainer.total_training_steps",
    "trainer.test_freq",
)
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


@dataclass(frozen=True)
class RunResult:
    path: Path
    method: str
    seed: int
    success: float
    mismatch: float | None
    training_rollouts: int
    gpu_hours: float
    budget: tuple[tuple[str, str], ...]
    code_signature: tuple[str, str]
    data_sha256: str


def _overrides(command: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in command:
        normalized = token[1:] if token.startswith("+") else token
        if "=" in normalized:
            key, value = normalized.split("=", 1)
            values[key] = value
    return values


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def load_run(run_dir: Path) -> RunResult:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError("run did not finish successfully")
    if not metrics_path.exists():
        raise ValueError("metrics.jsonl is missing")

    overrides = _overrides(manifest["command"])
    missing = [key for key in BUDGET_KEYS if key not in overrides]
    if missing:
        raise ValueError(f"manifest is missing budget overrides: {', '.join(missing)}")
    expected_step = int(overrides["trainer.total_training_steps"])
    validation_rows = [
        row
        for row in _read_metrics(metrics_path)
        if int(row.get("step", -1)) == expected_step
        and "val/success_rate" in row.get("metrics", {})
    ]
    if not validation_rows:
        raise ValueError(f"no final validation metric at step {expected_step}")
    success = float(validation_rows[-1]["metrics"]["val/success_rate"])
    if not 0.0 <= success <= 1.0:
        raise ValueError(f"invalid success rate: {success}")

    metric_rows = _read_metrics(metrics_path)
    matched = 0.0
    total = 0.0
    for row in metric_rows:
        metrics = row.get("metrics", {})
        if "train/total_prompt_count" in metrics and "train/expert_match_rate" in metrics:
            count = float(metrics["train/total_prompt_count"])
            matched += count * float(metrics["train/expert_match_rate"])
            total += count
    mismatch = 1.0 - matched / total if total else None

    batch_size = int(overrides["data.train_batch_size"])
    rollouts_per_task = int(overrides["env.rollout.n"])
    training_rollouts = expected_step * batch_size * rollouts_per_task
    # nvidia-smi inventories the whole host, which may contain GPUs this run did
    # not reserve. The Hydra resource budget is the authoritative used count.
    gpu_count = int(overrides["trainer.n_gpus_per_node"])
    gpu_hours = float(manifest.get("elapsed_seconds", 0.0)) * gpu_count / 3600.0
    budget = tuple((key, overrides[key]) for key in BUDGET_KEYS)
    git = manifest["git"]
    return RunResult(
        path=run_dir,
        method=str(manifest["method"]),
        seed=int(manifest["seed"]),
        success=success,
        mismatch=mismatch,
        training_rollouts=training_rollouts,
        gpu_hours=gpu_hours,
        budget=budget,
        code_signature=(str(git["commit"]), str(git["diff_sha256"])),
        data_sha256=str(manifest.get("data", {}).get("expert_trajectories_sha256", "")),
    )


def _t_critical(df: int) -> float:
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    larger = sorted(key for key in T_CRITICAL_95 if key >= df)
    return T_CRITICAL_95[larger[0]] if larger else 1.96


def summarize(values: list[float]) -> tuple[float, float | None, tuple[float, float] | None]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    std = statistics.stdev(values)
    margin = _t_critical(len(values) - 1) * std / math.sqrt(len(values))
    return mean, std, (mean - margin, mean + margin)


def render_report(runs: list[RunResult], excluded: list[tuple[Path, str]]) -> str:
    if not runs:
        raise ValueError("no eligible completed runs")
    reference = runs[0]
    for run in runs[1:]:
        if run.budget != reference.budget:
            raise ValueError(f"budget mismatch: {run.path} differs from {reference.path}")
        if run.code_signature != reference.code_signature:
            raise ValueError(f"code mismatch: {run.path} differs from {reference.path}")
        if run.data_sha256 != reference.data_sha256:
            raise ValueError(f"data mismatch: {run.path} differs from {reference.path}")
    identities = [(run.method, run.seed) for run in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate method/seed runs are ambiguous")

    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        grouped[run.method].append(run)
    lines = [
        "# ALFWorld limited-budget reproduction results",
        "",
        "> Auto-generated from completed runs. Failed, interrupted, or partial runs are excluded.",
        "",
        "## Comparable budget",
        "",
    ]
    for key, value in reference.budget:
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            f"- Code signature: `{reference.code_signature[0]}` + diff `{reference.code_signature[1]}`",
            f"- Expert data SHA-256: `{reference.data_sha256}`",
            "",
            "## Aggregate results",
            "",
            "| Method | Seeds | Success mean ± sample std | 95% CI | Training rollouts | GPU-hours |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(grouped):
        method_runs = sorted(grouped[method], key=lambda item: item.seed)
        mean, std, ci = summarize([run.success for run in method_runs])
        spread = "n/a" if std is None else f"{std * 100:.2f}%"
        ci_text = "n/a" if ci is None else f"[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]"
        lines.append(
            f"| {method} | {', '.join(str(run.seed) for run in method_runs)} | "
            f"{mean * 100:.2f}% ± {spread} | {ci_text} | "
            f"{sum(run.training_rollouts for run in method_runs)} | "
            f"{sum(run.gpu_hours for run in method_runs):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Run-level evidence",
            "",
            "| Method | Seed | Final success | Expert mismatch | Training rollouts | GPU-hours | Directory |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in sorted(runs, key=lambda item: (item.method, item.seed)):
        mismatch = "n/a" if run.mismatch is None else f"{run.mismatch * 100:.2f}%"
        lines.append(
            f"| {run.method} | {run.seed} | {run.success * 100:.2f}% | {mismatch} | "
            f"{run.training_rollouts} | {run.gpu_hours:.2f} | `{run.path}` |"
        )
    if excluded:
        lines.extend(["", "## Excluded runs", ""])
        for path, reason in excluded:
            lines.append(f"- `{path}`: {reason}")
    lines.extend(
        [
            "",
            "Training rollouts exclude validation episodes. Confidence intervals use a two-sided Student t interval; one-seed runs deliberately report no variance or interval.",
            "",
        ]
    )
    return "\n".join(lines)


def collect_runs(root: Path) -> tuple[list[RunResult], list[tuple[Path, str]]]:
    runs = []
    excluded = []
    for manifest in sorted(root.glob("*/run_manifest.json")):
        run_dir = manifest.parent
        try:
            runs.append(load_run(run_dir))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            excluded.append((run_dir, str(error)))
    return runs, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("reproduction/reports/results.md"))
    args = parser.parse_args()
    runs, excluded = collect_runs(args.runs_root)
    report = render_report(runs, excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} from {len(runs)} eligible runs; excluded {len(excluded)}")


if __name__ == "__main__":
    main()
