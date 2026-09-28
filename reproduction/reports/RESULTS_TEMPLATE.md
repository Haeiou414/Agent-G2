# ALFWorld limited-budget reproduction results

> Status: awaiting GPU experiments. Do not treat blank cells or paper reference values as reproduced results.

## Environment

| Item | Value |
|---|---|
| Git commit | |
| GPU model / count | |
| CUDA / PyTorch | |
| Base model | |
| Dataset split hash | |
| Tasks per step | |
| Rollouts per task | |
| Training steps | |
| Seeds | |

## Primary results

| Method | Seed | Success % | Expert mismatch % | Training rollouts | GPU-hours | Run directory |
|---|---:|---:|---:|---:|---:|---|
| GRPO | | | | | | |
| Target-accuracy | | | | | | |
| Agent-G² | | | | | | |

Report the mean, sample standard deviation, and 95% confidence interval over seeds. Comparisons are valid only when base model, task split, step count, rollout group size, and evaluation protocol match.

Generate this evidence from completed manifests with `python -m reproduction.summarize_results`. Expert mismatch is the prompt-count-weighted fraction of training tasks that failed to match an expert trajectory. Training rollouts exclude validation episodes.

## Ablations

| Variant | Success mean ± std | Δ vs. Agent-G² | Interpretation |
|---|---:|---:|---|
| Dynamic Gaussian | | | Main method |
| Fixed sigma (`σ=σ_min=0.1`) | | | Tests adaptive spread |
| No auxiliary SFT | | | Tests prefix supervision |
| Deterministic mean (`d=μ`) | | | Tests stochastic sampling |

## Failure analysis

Break down failures by ALFWorld task type, expert trajectory length, sampled prefix ratio, invalid-action count, and termination reason. Include at least three qualitative trajectories where the baseline and Agent-G² diverge.

## Reproduction verdict

- **Exact reproduction:** confidence interval overlaps the paper result under the paper-scale configuration.
- **Trend reproduction:** Agent-G² consistently improves over same-budget baselines, but the setup is smaller.
- **Not reproduced:** the main-method advantage is absent or reverses; report diagnostics without hiding failed seeds.
