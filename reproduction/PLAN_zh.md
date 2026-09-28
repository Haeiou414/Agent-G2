# Agent-G² 独立复现计划

## 项目定位

目标不是“把作者仓库跑起来”，而是完成一套可审计、可缩放、可解释的复现：

1. 独立实现并测试 Gaussian Guidance 调度器；
2. 复核论文、官方脚本和实际日志之间的一致性；
3. 在 ALFWorld 上完成低成本对照实验，再按算力扩展到论文设置；
4. 沉淀实验报告、失败分析和可量化的简历描述。

论文主张：对每个任务从在线估计的高斯分布中采样专家前缀深度，在不增加 probe rollout 的情况下缓解长程 Agent RL 的稀疏奖励问题。

## 三层复现标准

| 层级 | 目的 | 计算需求 | 验收证据 |
|---|---|---:|---|
| L0 算法复刻 | 验证公式、边界和调度动态 | CPU | 单元测试、仿真 CSV、参数轨迹 |
| L1 结果核验 | 评估作者公开模型 | 1 张推理 GPU | ALFWorld success、WebShop score/success |
| L2 低成本训练 | 证明方法增益和消融趋势 | 1–2 张训练 GPU | 3 seeds 的均值/方差、曲线、成本统计 |
| L3 论文同规模 | 对齐论文主表 | 8 张训练 GPU | Qwen2.5-1.5B、16 tasks × 8 rollouts、完整对照 |

项目达到 L2 就足以作为高质量日常实习项目；L3 是有算力时的加分项。

## 最小实验矩阵

固定模型、数据划分、rollout 数和训练预算，仅改变 guidance 策略：

| 实验 | 用途 | 必跑 |
|---|---|---|
| GRPO（无 hint） | 下界基线 | 是 |
| Target-accuracy scalar hint | 最强共享深度对照 | 是 |
| Agent-G² dynamic Gaussian | 主方法 | 是 |
| Agent-G² fixed sigma | 验证动态方差 | 是 |
| Agent-G² no auxiliary SFT | 验证辅助损失 | 是 |
| Gaussian → deterministic mean | 验证“采样”本身 | 是 |

主要指标：任务成功率、expert match/mismatch、训练 rollout 数、无效动作率和训练 GPU-hours。所有训练至少记录 3 个随机种子；若算力不足，先报告置信区间并明确 limited-budget setting。

## 配置一致性审计（2026-09-20，官方仓库提交 `e065918`）

官方 README 将 `examples/gmsv_trainer/` 描述为 paper-locked scripts，但当前脚本与论文附录 Table 5 存在以下差异，正式实验前必须锁定采用哪一套配置：

| 参数 | 论文附录 | 官方 ALFWorld 脚本 / 默认配置 | 风险 |
|---|---:|---:|---|
| peak learning rate | `1e-5` | `1e-6` | 直接影响收敛速度 |
| prompt length | `7000` | `4096` | 长轨迹可能被截断 |
| difficulty clusters K | `3` | 默认 `5`（脚本未覆盖） | 改变局部统计粒度 |
| center scale λ | `1.0` | 默认 `0.3`（脚本未覆盖） | 改变难度补偿强度 |
| global baseline 范围 | `[0, 1]` | 默认上限 `0.8` | 困难阶段无法继续增加全局 guidance |
| guidance ratio 范围 | `[0, 1]` | 默认 `max_length=0.8` | 永远不会采样 0.8 以上的前缀比例 |
| ALFWorld training steps | `200` | `total_epochs=300` | 训练预算不可比 |
| 调度统计粒度 | 先对同一任务的 R 次 rollout 求 `p_hat_i` | 原实现优先使用每条 rollout 唯一的 `traj_uid` | 方差被估计为二元回报方差，而非任务成功率方差 |
| 前缀长度基准 | `ceil(r_i × expert action count)` | 原实现按 expert token count 取比例后扩展到动作边界 | 动作文本长度会改变 guidance depth |

处理原则：保留 `official-script` 与 `paper-table5` 两套命名配置，不静默修改；先做小规模 smoke test，再在同一预算下比较。

论文配置入口为 `reproduction/run_alfworld_paper_table5.sh`。它通过末尾 Hydra overrides 调用官方脚本，既不覆盖上游配方，也能保证论文配置可直接执行。

## 近期里程碑

- [x] 拉取官方代码并固定上游提交。
- [x] 独立实现 CPU-only Gaussian scheduler。
- [x] 覆盖初值、难度偏置、动态方差、全局更新、前缀边界的单元测试。
- [x] 修正官方运行时的调度统计粒度，使其对齐论文 Eq. (2)。
- [x] 修正前缀长度计算，使其对齐论文 Eq. (5) 的动作步数定义。
- [x] 用公开 checkpoint 的固定 tokenizer revision 精确量化动作/token 前缀差异。
- [x] 生成调度器仿真曲线并加入报告。
- [x] 核验公开 ALFWorld 1.5B checkpoint（seed 1，128-task success 59.375%；与 model card 数字的协议/选择差异保留为限制）。
- [x] 准备单张 24GB GPU 的同预算三方法启动器与 dry-run 测试。
- [x] 保存逐步 JSONL 指标、运行 provenance、退出状态与 GPU-hours。
- [x] 实现结果汇总门禁，拒绝失败运行或代码、数据、预算不一致的对照。
- [x] 准备中文项目入口、结果模板、简历与面试材料。
- [x] 在 RTX 4090 24GB 上完成 LoRA smoke test（8 steps、validation、checkpoint、断点续训与适配器重载）。
- [ ] 跑完最小实验矩阵和 3 seeds。
- [ ] 用真实 GPU 日志生成结果表、失败分析与英文摘要。

## 简历成果的最终形态

只有实测完成后再填数字，避免把论文数字写成自己的结果：

> 独立复现 Agent-G²，基于 veRL/GRPO 实现按任务难度自适应的高斯专家前缀调度；构建配置一致性审计与 6 组消融实验，在 ALFWorld 的受限算力设置下将成功率从 **X%** 提升至 **Y%**；通过 3-seed 实验、GPU-hours、expert match 与失败类型完成可复现性分析。

## 常用命令

```bash
python -m unittest discover -s tests/reproduction -v
python -m reproduction.audit_config
python -m reproduction.simulate_scheduler --output artifacts/scheduler_simulation.csv
python -m reproduction.render_scheduler_svg
bash reproduction/run_alfworld_paper_table5.sh
python -m reproduction.download_public_checkpoint --local-dir /data/models/agent-g2-alfworld-1.5b
bash reproduction/run_alfworld_checkpoint_eval.sh /data/models/agent-g2-alfworld-1.5b 1
python -m reproduction.summarize_results
```
