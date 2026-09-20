"""A dependency-light implementation of the Agent-G2 Gaussian scheduler.

This module isolates the scheduling algorithm from the distributed veRL training
stack. It is intended for unit tests, ablations, and small simulations; the
official training runtime remains in :mod:`gmsv`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, pvariance
from typing import Iterable, Mapping


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SchedulerConfig:
    """Hyperparameters from Algorithm 1 and Appendix B of the paper."""

    delta: float = 0.1
    alpha: float = 0.2
    lambda_: float = 1.0
    gamma: float = 1.0
    sigma_min: float = 0.1
    target_success: float = 0.5
    mu_init: float = 0.8
    mu_min: float = 0.0
    mu_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.delta < 0.0 or self.gamma < 0.0 or self.sigma_min < 0.0:
            raise ValueError("delta, gamma, and sigma_min must be non-negative")
        if self.mu_min > self.mu_max:
            raise ValueError("mu_min must not exceed mu_max")


@dataclass
class ClusterStats:
    accuracy_ema: float = 0.0
    variance_ema: float = 0.0


class AgentG2Scheduler:
    """Online Gaussian guidance scheduler described by Agent-G2.

    Rewards passed to :meth:`update` are prompt-level success estimates. For
    GRPO with ``R`` rollouts, each estimate is the mean of the ``R`` binary
    terminal rewards for that prompt.
    """

    def __init__(
        self,
        cluster_ids: Iterable[int],
        config: SchedulerConfig | None = None,
        seed: int = 1,
    ) -> None:
        self.config = config or SchedulerConfig()
        ids = tuple(dict.fromkeys(int(cluster_id) for cluster_id in cluster_ids))
        if not ids:
            raise ValueError("at least one cluster is required")
        self.stats = {cluster_id: ClusterStats() for cluster_id in ids}
        self.mu_global = _clip(
            self.config.mu_init,
            self.config.mu_min,
            self.config.mu_max,
        )
        self._rng = random.Random(seed)

    def parameters(self, cluster_id: int) -> tuple[float, float]:
        stats = self.stats[int(cluster_id)]
        mu = _clip(
            self.mu_global
            + self.config.lambda_
            * (self.config.target_success - stats.accuracy_ema),
            0.0,
            1.0,
        )
        sigma = max(
            self.config.gamma * stats.variance_ema,
            self.config.sigma_min,
        )
        return mu, sigma

    def sample_ratio(self, cluster_id: int) -> float:
        mu, sigma = self.parameters(cluster_id)
        return _clip(self._rng.gauss(mu, sigma), 0.0, 1.0)

    @staticmethod
    def prefix_length(ratio: float, trajectory_length: int) -> int:
        """Map a sampled ratio to Eq. (5)'s executable expert-prefix length."""

        if trajectory_length <= 1:
            return 0
        clipped_ratio = _clip(float(ratio), 0.0, 1.0)
        return min(math.ceil(clipped_ratio * trajectory_length), trajectory_length - 1)

    def update(self, outcomes: Mapping[int, Iterable[float]]) -> None:
        """Refresh cluster EMAs and the global baseline from one training batch."""

        all_values: list[float] = []
        for raw_cluster_id, raw_values in outcomes.items():
            cluster_id = int(raw_cluster_id)
            if cluster_id not in self.stats:
                raise KeyError(f"unknown cluster: {cluster_id}")
            values = [float(value) for value in raw_values]
            if not values:
                continue
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError("success estimates must be in [0, 1]")

            batch_accuracy = fmean(values)
            batch_variance = pvariance(values)
            stats = self.stats[cluster_id]
            alpha = self.config.alpha
            stats.accuracy_ema = (1.0 - alpha) * stats.accuracy_ema + alpha * batch_accuracy
            stats.variance_ema = (1.0 - alpha) * stats.variance_ema + alpha * batch_variance
            all_values.extend(values)

        if not all_values:
            return

        batch_accuracy = fmean(all_values)
        if batch_accuracy < self.config.target_success:
            self.mu_global += self.config.delta
        elif batch_accuracy > self.config.target_success:
            self.mu_global -= self.config.delta
        self.mu_global = _clip(
            self.mu_global,
            self.config.mu_min,
            self.config.mu_max,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "mu_global": self.mu_global,
            "clusters": {
                cluster_id: {
                    "accuracy_ema": stats.accuracy_ema,
                    "variance_ema": stats.variance_ema,
                    "mu": self.parameters(cluster_id)[0],
                    "sigma": self.parameters(cluster_id)[1],
                }
                for cluster_id, stats in self.stats.items()
            },
        }
