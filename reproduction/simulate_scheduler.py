"""Run a deterministic toy simulation of Agent-G2 schedule adaptation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from reproduction.scheduler import AgentG2Scheduler


def success_probability(cluster_id: int, ratio: float, step: int) -> float:
    """A synthetic curriculum: deeper hints help, while the policy improves."""

    difficulty = {1: 0.15, 2: 0.35, 3: 0.55}[cluster_id]
    learning_progress = min(step / 80.0, 0.45)
    return max(0.0, min(1.0, 0.15 + ratio - difficulty + learning_progress))


def run(steps: int, tasks_per_cluster: int, seed: int) -> list[dict[str, float | int]]:
    scheduler = AgentG2Scheduler(cluster_ids=(1, 2, 3), seed=seed)
    rng = random.Random(seed + 1)
    rows: list[dict[str, float | int]] = []

    for step in range(steps):
        outcomes: dict[int, list[float]] = {1: [], 2: [], 3: []}
        sampled_ratios: dict[int, list[float]] = {1: [], 2: [], 3: []}
        for cluster_id in outcomes:
            for _ in range(tasks_per_cluster):
                ratio = scheduler.sample_ratio(cluster_id)
                probability = success_probability(cluster_id, ratio, step)
                outcomes[cluster_id].append(float(rng.random() < probability))
                sampled_ratios[cluster_id].append(ratio)

        scheduler.update(outcomes)
        snapshot = scheduler.snapshot()
        clusters = snapshot["clusters"]
        assert isinstance(clusters, dict)
        for cluster_id in outcomes:
            cluster = clusters[cluster_id]
            rows.append(
                {
                    "step": step,
                    "cluster": cluster_id,
                    "mu_global": float(snapshot["mu_global"]),
                    "mu": float(cluster["mu"]),
                    "sigma": float(cluster["sigma"]),
                    "accuracy_ema": float(cluster["accuracy_ema"]),
                    "variance_ema": float(cluster["variance_ema"]),
                    "mean_sampled_ratio": sum(sampled_ratios[cluster_id]) / tasks_per_cluster,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tasks-per-cluster", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scheduler_simulation.csv"))
    args = parser.parse_args()

    rows = run(args.steps, args.tasks_per_cluster, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
