# Agent-G² 复现：简历与面试材料

## 当前可用版本（尚未完成 GPU 对照）

**项目名：** Agent-G² Agentic RL 论文复现与实现审计  
**技术栈：** Python、PyTorch/verl、GRPO、Hydra、Ray、vLLM、ALFWorld、unittest、GitHub Actions

可直接使用的两条描述：

- 基于作者开源的 verl/GRPO 训练框架复现 Agent-G² Gaussian Guidance 调度器，将任务成功率 EMA、难度偏置、动态方差与 action-level expert prefix 拆为可独立验证组件，并建立 30 项 CPU 回归测试和 CI。
- 审计 3,553 条 ALFWorld 专家轨迹以及论文/代码的 12 个关键配置字段，定位 7 项标量配置差异和 2 项运行时语义差异；用固定 revision 的官方 tokenizer 发现原 token-level 实现在 guidance ratio=0.5 时有 36.0% 轨迹选择了不同动作深度，并修正任务级成功率聚合与前缀边界。

这版强调的是复现工程、论文—代码审计和可验证修正，不应写“复现 95.3%”或“超过 GRPO”，因为本分支还没有产生 GPU 对照结果。

## GPU 实验完成后的升级模板

只有 `reproduction/reports/results.md` 自动生成并通过同预算校验后，才能把下列占位符替换为真实数字：

- 在 Qwen2.5-1.5B + ALFWorld 的单卡受限预算实验中，对 GRPO、target-accuracy hint 与 Agent-G² 运行 `[N]` 个随机种子；Agent-G² 达到 `[均值 ± 样本标准差]%` 成功率，相比 GRPO 提升 `[百分点]`，共消耗 `[GPU-hours]` GPU 小时。
- 通过 fixed-sigma、no-auxiliary-SFT 与 deterministic-mean 消融，将收益归因于 `[真实结论]`；结合失败类型、专家轨迹长度和 prefix ratio 分析 `[真实现象]`。

不要只选最好 seed，也不要把论文值、作者 checkpoint 值或仿真曲线写成本项目训练结果。

## 60 秒面试讲法

我复现的是 Agent-G²，它解决 agentic RL 冷启动时 rollout 大量失败、GRPO 缺少有效组内信号的问题。核心做法不是给所有任务固定提示，而是先按专家轨迹长度划分难度，再用每个难度簇的成功率 EMA 控制 Gaussian guidance 的均值和方差，采样得到这次 rollout 要展示多少专家动作。

我没有直接照搬仓库结果。先把论文公式写成无训练依赖的 scheduler，用测试验证更新方向和边界；然后逐项审计论文配置、作者脚本与运行时实现。审计发现，公开实现按 token 比例截断，而论文公式按动作数取整。在官方 tokenizer 上，这会使 ratio=0.5 的 36.0% 轨迹选择不同深度。另外，成功率原先按单 rollout 更新，我改成同一任务 R 次 rollout 的均值，和论文估计量一致。最后我加了同预算启动器、运行 provenance、JSONL 指标和结果校验，失败运行或不同代码/数据/预算不会进入汇总。

## 高频追问

### 为什么用 Gaussian，而不是固定 prefix？

均值让困难簇获得更深 guidance；方差让簇内表现分化较大时扩大探索。固定 prefix 无法同时适应训练进度、任务难度和簇内不确定性。

### 为什么任务级成功率不能直接用每条 rollout？

同一任务会采样 R 条 rollout。论文中的估计量是这 R 次二元成功的均值；把每条 rollout 当独立任务会改变 EMA 的统计单位和方差，从而改变后续 guidance 分布。

### 为什么 token-level 与 action-level 的差异重要？

动作文本长度不均匀。按 token 质量截断会偏向长文本动作，不能保证 `ceil(rL)` 个专家动作。这个差异会改变 policy 真正需要完成的剩余决策深度。

### 怎样保证对照公平？

三个主方法共用模型、任务 batch、每任务 rollout 数、训练 step 和验证频率。每次运行记录代码 diff hash、数据 hash、依赖、GPU、命令和耗时；汇总器要求最终验证指标存在，并拒绝预算或 provenance 不一致的运行。

### 如果结果没有提升怎么办？

按“未复现”报告，不删除失败 seed。优先检查 expert match、无效动作率、prefix ratio 分布、各难度簇 EMA、训练/验证划分和 OOM 后是否改过预算，再区分实现问题、缩小预算导致的统计功效不足和方法本身不稳定。
