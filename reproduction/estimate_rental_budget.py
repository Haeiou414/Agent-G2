"""Estimate sequential AutoDL time/cost from an actual completed smoke run."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeEstimate:
    source_steps: int
    fixed_overhead_seconds: float
    base_step_seconds: float
    validation_seconds: float
    checkpoint_seconds: float
    target_steps: int
    validation_events: int
    checkpoint_events: int
    seconds_per_run: float
    gpu_count: int


def command_overrides(command: list[str]) -> dict[str, str]:
    values = {}
    for token in command:
        normalized = token[1:] if token.startswith("+") else token
        if "=" in normalized:
            key, value = normalized.split("=", 1)
            values[key] = value
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_count(total_steps: int, frequency: int) -> int:
    if frequency <= 0:
        return 0
    regular = total_steps // frequency
    return regular if total_steps % frequency == 0 else regular + 1


def estimate_run(
    run_dir: Path,
    *,
    target_steps: int = 80,
    test_frequency: int = 10,
    save_frequency: int = 40,
) -> RuntimeEstimate:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError("smoke run must be completed successfully")
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise ValueError("smoke run is missing metrics.jsonl")
    rows = [row for row in read_jsonl(metrics_path) if "timing_s/step" in row.get("metrics", {})]
    if not rows:
        raise ValueError("smoke run has no timing_s/step metrics")

    overrides = command_overrides(manifest["command"])
    source_steps = int(overrides["trainer.total_training_steps"])
    observed_steps = {int(row["step"]) for row in rows}
    if source_steps not in observed_steps:
        raise ValueError(f"smoke run lacks timing for final step {source_steps}")

    step_totals = []
    base_steps = []
    validations = []
    checkpoints = []
    for row in rows:
        metrics = row["metrics"]
        total = float(metrics["timing_s/step"])
        validation = float(metrics.get("timing_s/testing", 0.0))
        checkpoint = float(metrics.get("timing_s/save_checkpoint", 0.0))
        step_totals.append(total)
        base_steps.append(max(0.0, total - validation - checkpoint))
        if validation > 0.0:
            validations.append(validation)
        if checkpoint > 0.0:
            checkpoints.append(checkpoint)

    target_validation_events = event_count(target_steps, test_frequency)
    target_checkpoint_events = event_count(target_steps, save_frequency)
    if target_validation_events and not validations:
        raise ValueError("smoke run contains no validation timing; cannot price target validation")
    if target_checkpoint_events and not checkpoints:
        raise ValueError("smoke run contains no checkpoint timing; cannot price target saves")

    elapsed = float(manifest["elapsed_seconds"])
    fixed_overhead = max(0.0, elapsed - sum(step_totals))
    base_step_seconds = statistics.fmean(base_steps)
    validation_seconds = statistics.fmean(validations) if validations else 0.0
    checkpoint_seconds = statistics.fmean(checkpoints) if checkpoints else 0.0
    seconds_per_run = (
        fixed_overhead
        + target_steps * base_step_seconds
        + target_validation_events * validation_seconds
        + target_checkpoint_events * checkpoint_seconds
    )
    gpu_count = int(overrides.get("trainer.n_gpus_per_node", "1"))
    return RuntimeEstimate(
        source_steps=source_steps,
        fixed_overhead_seconds=fixed_overhead,
        base_step_seconds=base_step_seconds,
        validation_seconds=validation_seconds,
        checkpoint_seconds=checkpoint_seconds,
        target_steps=target_steps,
        validation_events=target_validation_events,
        checkpoint_events=target_checkpoint_events,
        seconds_per_run=seconds_per_run,
        gpu_count=gpu_count,
    )


def render(estimate: RuntimeEstimate, hourly_price: float | None) -> str:
    per_run_hours = estimate.seconds_per_run / 3600.0
    rows = [("主方法闭环", 3, 3), ("完整实验矩阵", 6, 3)]
    lines = [
        "# AutoDL sequential rental estimate",
        "",
        f"Source: completed {estimate.source_steps}-step smoke run.",
        f"Estimated one-time overhead: {estimate.fixed_overhead_seconds / 60:.1f} min/run.",
        f"Estimated base step: {estimate.base_step_seconds:.1f} s.",
        f"Estimated validation event: {estimate.validation_seconds:.1f} s.",
        f"Estimated checkpoint event: {estimate.checkpoint_seconds:.1f} s.",
        "",
        "| Plan | Runs | Sequential wall-hours | GPU-hours | Estimated fee |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, methods, seeds in rows:
        runs = methods * seeds
        wall_hours = per_run_hours * runs
        gpu_hours = wall_hours * estimate.gpu_count
        fee = "n/a" if hourly_price is None else f"¥{wall_hours * hourly_price:.2f}"
        lines.append(f"| {label} | {runs} | {wall_hours:.2f} | {gpu_hours:.2f} | {fee} |")
    lines.extend(
        [
            "",
            f"Per-run target: {estimate.target_steps} steps, {estimate.validation_events} validation events, "
            f"{estimate.checkpoint_events} checkpoint events.",
            "This is a planning estimate from one measured host, not a promised runtime. Keep a 20–30% balance reserve for downloads, compilation, retries, and host variance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=80)
    parser.add_argument("--test-frequency", type=int, default=10)
    parser.add_argument("--save-frequency", type=int, default=40)
    parser.add_argument("--hourly-price", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    estimate = estimate_run(
        args.smoke_run,
        target_steps=args.target_steps,
        test_frequency=args.test_frequency,
        save_frequency=args.save_frequency,
    )
    report = render(estimate, args.hourly_price)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
