# Agent-G² 复现学习手册

这份手册用于在 GPU 实验运行期间系统掌握项目。目标不是背论文摘要，而是能够解释算法、读懂实现、诊断训练并回答面试追问。

## 学习路线

1. GRPO 与 agentic RL 的稀疏奖励问题；
2. Agent-G² 的 Gaussian guidance 调度器；
3. 任务级成功率、EMA 与动态方差；
4. ALFWorld 环境、专家轨迹与 action-level prefix；
5. veRL、Ray、vLLM、FSDP 和 LoRA 的分工；
6. 同预算对照、消融实验与统计汇总；
7. 简历表述、项目讲解和面试追问。

## 第一课：从 GRPO 的无信号批次到 Gaussian guidance

### 1. GRPO 为什么会在长程 agent 任务中失去学习信号

对同一个任务 `i` 采样 `R` 条轨迹，用组内相对奖励构造 advantage：

```text
A[i,j] = (r[i,j] - mean(r[i,:])) / (std(r[i,:]) + epsilon)
```

如果初期的 `R` 条轨迹全部失败，奖励都为零，那么 advantage 也全部接近零。模型花了 rollout 算力，却不知道哪条轨迹更好。

### 2. Agent-G² 的训练支架

训练时先执行一部分正确的专家动作，再让策略从中间状态继续完成任务。验证时关闭 guidance，因此最终指标检验的仍然是模型的自主能力。

对难度组 `k` 采样专家前缀比例：

```text
rho ~ Normal(mu[k], sigma[k]^2)
mu[k] = clip(mu_global + lambda * (0.5 - A[k]), 0, 1)
sigma[k] = max(sigma_min, gamma * V[k])
```

- `A[k]` 是该难度组的成功率 EMA；
- `V[k]` 是该组任务成功率方差的 EMA；
- 成功率低时，`mu[k]` 变大，提供更长的专家前缀；
- 组内表现分化大时，`sigma[k]` 变大，扩大探索范围。

将采样比例转换成可执行的专家动作数：

```text
prefix_steps = min(ceil(rho * trajectory_length), trajectory_length - 1)
```

上限是 `trajectory_length - 1`，因为必须至少留一步给模型自己决策。

### 3. 一个训练批次

1. 根据专家轨迹的动作数将任务分组；
2. 从对应高斯分布采样 guidance 比例；
3. 执行专家前缀，再由模型继续 rollout；
4. 每个任务生成 `R` 条 rollout；
5. 对同一任务的 `R` 个二元结果求均值，得到任务级成功率；
6. 用任务级成功率更新难度组统计和全局中心；
7. 用 GRPO 更新策略，并对专家前缀加入辅助 SFT loss。

### 4. 代码导航

- `reproduction/scheduler.py`：不依赖分布式训练栈的算法实现；
- `gmsv/alfworld.py`：将调度器接入 ALFWorld rollout 和训练指标；
- `tests/reproduction/test_scheduler.py`：检验均值、方差、全局中心和 prefix 边界。

### 5. 练习

已知 `mu_global=0.6`、`lambda=1`、`gamma=1`、`sigma_min=0.1`：

- 简单组：`A=0.7, V=0.02`；
- 困难组：`A=0.1, V=0.20`。

计算两组的 `mu` 和 `sigma`。如果专家轨迹有 10 个动作，采样得到 `rho=0.42`，再计算应该提供几个专家动作。

