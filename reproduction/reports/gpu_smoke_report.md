# RTX 4090 GPU 验证报告

## 结论

单张 RTX 4090 24GB 可以运行本项目的 Qwen2.5-1.5B、LoRA rank 16 受限算力配置。端到端 smoke test、断点恢复、验证、主 checkpoint 和独立 PEFT adapter 重载均已验证。单卡 full-parameter AdamW 不可行，因此后续 L2 对照必须明确标注为 LoRA limited-budget reproduction，不能冒充论文的 8-GPU full-parameter 设置。

## 环境

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| PyTorch / CUDA | 2.6.0+cu124 / CUDA 12.4 |
| vLLM / Ray | 0.8.5 / 2.46.0 |
| Transformers / flash-attn | 4.51.1 / 2.7.4.post1 |
| 模型 | Qwen2.5-1.5B-Instruct，本地固定快照 |
| 训练形式 | LoRA rank 16 / alpha 16 |
| 单步预算 | 4 tasks × 4 rollouts，最多 5 个环境动作 |

## 8-step smoke

- 运行目录：`outputs/limited_gmsv_qwen2.5_1.5b_seed1_smoke5-lora`；
- 代码提交：`82dac90863d8de738febb25f37b64b54c1c0a997`，工作区 clean；
- 状态：`completed`，退出码 0；
- 端到端耗时：1026.947 秒（17.12 分钟）；
- 完成 8 个 optimizer steps，并在 step 8 完成 16-task validation 和 checkpoint；
- step 1 显存峰值：allocated 18.305 GB、reserved 18.674 GB；
- step 8 显存峰值：allocated 18.313 GB、reserved 19.459 GB；
- `mu_global` 从 0.8 更新到 1.0，三个 difficulty group 的 `A_k/V_k/sigma_k` 均写入指标；
- expert match rate 为 100%，辅助 prefix SFT 在 step 1 使用 64 个样本、3,056 个 token；
- 8-step 最终 validation success 为 0%。该运行只用于工程链路和容量验证，训练预算太小，不能作为方法效果结论。

## 断点恢复与 adapter

从 smoke 的 `global_step_8` 恢复后，训练器明确记录 `Setting global step to 8`，随后仅执行 step 9，并完成验证与 `global_step_9` 保存：

- 运行目录：`outputs/limited_gmsv_qwen2.5_1.5b_seed1_resumecheck2`；
- 状态：`completed`，退出码 0；
- 端到端耗时：285.191 秒；
- step 9 耗时：186.669 秒，其中 validation 65.532 秒、checkpoint 10.052 秒；
- 主 checkpoint 中有 392 个 LoRA 张量，共 18,464,768 个参数，全部非零；
- 独立 `adapter_model.safetensors` 为 73,911,112 bytes；
- SHA-256：`d779ef681f15622cd0b05be409805bf1b7def6a44cdcecde45791d035fce1b32`；
- 使用 PEFT `PeftModel.from_pretrained` 重新加载成功。

原单卡 FSDP/NO_SHARD 导出会生成 16-byte 空 safetensors。修复后从刚写入的 durable rank-0 checkpoint 显式读取 state dict，再交给 PEFT 的官方键过滤与 adapter-name 转换逻辑；它避免依赖 NO_SHARD wrapper 的空 `state_dict()`。

## 失败实验与根因

1. vLLM `gpu_memory_utilization=0.35` 无法分配任何 KV cache block；实测可用值为 0.50。
2. `free_cache_engine=true` 与 CUDA graph 不兼容；单卡入口固定 `enforce_eager=true`。
3. full-parameter 训练可完成 rollout 和 backward，但第一次 `AdamW.step()` 创建优化器状态时 OOM；这属于固定参数/优化器内存，不应靠减少 rollout batch 掩盖。
4. AutoDL 重启后，未恢复 `ALFWORLD_DATA` 时 ALFWorld 扫描到 0 个游戏，TextWorld 的空列表 `shuffled_cycle` 形成无限 CPU 忙循环。启动器现会自动加载持久化环境路径，并在训练数据缺失时 fail fast。修复后单环境识别 3,553 个训练游戏，从构建到 reset 仅 10.27 秒。

## 作者 checkpoint 核验

固定公开 checkpoint revision、seed 1、128 个 validation task 的 success 为 59.375%（76/128），运行耗时 515.98 秒。它低于 model card 的 95.3%；由于公开材料未给出该数字的 checkpoint selection step 和完整评测协议，本项目保留该差异，不把作者数字替换成本地结果。

## 尚未完成

尚无足以比较方法效果的同预算 80-step 结果。下一阶段必须先完成 GRPO、target-accuracy 和 Agent-G² 的 seed 1 闭环，再决定是否扩展到 3 seeds 与三项消融。只有自动汇总器通过代码、数据和预算一致性门禁后，提升数字才可写入简历。

容量审计显示数据盘共 150 GB、当前已用 38 GB、可用 113 GB。正式入口保留 step 40 恢复点，但在 step 80 成功保存后只留下最新 checkpoint；这样三方法 × 三种子的最终主 checkpoint 约占 63 GB，可以和现有模型及 smoke 共存。完整 18-run 矩阵需要扩容或经过核验的归档策略。

## 可审计原始证据

仓库保留了不含模型权重的小型证据包：

- `reproduction/evidence/rtx4090_smoke8/`：manifest、8 行 JSONL 指标、Hydra 展开配置和 step 8 调度器状态；
- `reproduction/evidence/rtx4090_resume9/`：manifest、step 9 指标、展开配置、调度器状态、PEFT 配置与 adapter 校验清单；
- `reproduction/evidence/restart_failure/`：零游戏忙循环那次失败运行的 manifest。

`tests/reproduction/test_gpu_evidence.py` 会验证成功/失败状态、step 连续性、最终 validation、断点推进、scheduler 状态和 adapter inventory，防止报告数字与原始证据漂移。7 GB 主 checkpoint 与 73.9 MB adapter 保留在 AutoDL 数据盘，不纳入 Git。
