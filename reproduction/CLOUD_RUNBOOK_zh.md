# 单张 NVIDIA GPU 运行手册

这份手册用于受限预算复现，不宣称等价于论文的单机 8 卡设置。首次租用建议单张 NVIDIA RTX 4090（24GB）做短时试跑；如预算允许，L40/A40 等 48GB 卡更有显存余量。选择 Ubuntu、CUDA 12.4 开发镜像、至少 64GB 主机内存，并为 `/root/autodl-tmp` 准备至少 100GB 数据盘；完整多种子实验预计要扩容，具体以试跑后的 checkpoint 大小为准。24GB 是启动器的设计目标，不是已在 GPU 上验证的显存保证。

不要把 RTX 5090 等 Blackwell 卡与当前 CUDA 12.4 / PyTorch 2.6 / vLLM 0.8.5 固定环境混用；5090 需单独验证 CUDA 12.8+、PyTorch 2.7+ 和相应推理/注意力内核栈。安装脚本会在下载依赖前检查 GPU compute capability 并拒绝此类组合。

## 1. 创建环境

AutoDL 推荐使用自动入口。它把 Conda 环境与包缓存、ALFWorld 数据、Hugging Face 缓存和 pip 缓存放到 `/root/autodl-tmp/agent-g2-data`，避免占满默认 30GB 系统盘；安装结束后会自动执行测试和 GPU 自检：

```bash
cd /root/autodl-tmp/agent-g2-reproduction
bash reproduction/bootstrap_autodl.sh
source reproduction/autodl_env.sh
conda activate agent-g2
```

每个新的 SSH 或 tmux shell 都要重新 `source reproduction/autodl_env.sh`。环境证据保存在 `outputs/setup/`。

如需手动安装，等价步骤如下：

```bash
source reproduction/autodl_env.sh
conda create -n agent-g2 python=3.12 -y
conda activate agent-g2

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
pip install vllm==0.8.5 gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
```

不要在同一个环境里安装 WebShop；官方仓库要求 WebShop 使用 Python 3.10，本项目第一阶段只复现 ALFWorld。

## 2. 环境自检

```bash
source reproduction/autodl_env.sh
python -m reproduction.verify_gpu_environment
conda run -n agent-g2 python -m unittest discover -s tests/reproduction -v
```

自检必须确认：Linux、NVIDIA GPU、约 24GB 显存、CUDA 可被 PyTorch 访问，以及 `torch/vllm/ray/flash-attn/alfworld` 均已安装。

在无卡模式预下载并校验固定 revision 的 Qwen 基础模型，避免 GPU 计费期间等待网络；启动训练前把路径导出给单卡启动器：

```bash
python -m reproduction.download_base_model \
  --local-dir "$AGENT_G2_DATA_ROOT/models/qwen2.5-1.5b-instruct" \
  --max-workers 1
export BASE_MODEL_PATH="$AGENT_G2_DATA_ROOT/models/qwen2.5-1.5b-instruct"
```

下载器同时核对权重的精确字节数与 SHA-256；校验失败时不得开始实验。

## 3. 8-step smoke test

先只运行主方法，验证环境交互、rollout、策略更新、验证和 checkpoint 全链路：

```bash
bash reproduction/run_alfworld_single_gpu.sh gmsv 1 vllm \
  trainer.total_training_steps=8 \
  trainer.test_freq=4 \
  trainer.save_freq=8 \
  data.val_batch_size=16
```

验收条件：

- 训练至少完成 8 个 optimizer steps；
- 日志出现 `mu_global`、各 difficulty group 的 `A_k/V_k/sigma_k`；
- step 4 和 step 8 能完成 validation；
- checkpoint 中存在 `gmsv_runtime_state.json`；
- 不出现 expert trajectory unmatched 或 prefix replay 错误。

smoke test 完成后，用 AutoDL 页面显示的实例每小时价格估算余额。该工具只读取这台机器的实测 manifest 与逐步 timing，不套用论文 8 卡速度：

```bash
python -m reproduction.estimate_rental_budget \
  --smoke-run outputs/limited_gmsv_qwen2.5_1.5b_seed1 \
  --hourly-price 2.50 \
  --output reproduction/reports/autodl_budget.md
```

金额只是线性规划值，建议额外保留 20%–30% 余额用于下载、编译、失败重试和宿主机波动。

## 4. 核验作者公开 checkpoint

公开模型核验与自行训练复现是两条证据链。下载器固定 Hugging Face revision `4556a9bfdf84320267c3a9e9e7b85732ba2835ba`，评测入口强制 `val_only=true`、禁止 validation guidance，并保存 checkpoint identity：

```bash
python -m reproduction.download_public_checkpoint \
  --local-dir /root/autodl-tmp/agent-g2-data/models/agent-g2-alfworld-1.5b

for seed in 1 2 3; do
  bash reproduction/run_alfworld_checkpoint_eval.sh \
    /root/autodl-tmp/agent-g2-data/models/agent-g2-alfworld-1.5b "$seed"
done
```

这组结果只能写成“核验作者公开 checkpoint”，不能写成“自行训练达到”。公开 model card 没有说明其 95.3% 对应的具体 checkpoint selection step，所以报告中必须保留这一限制。

## 5. 同预算主实验

三个方法必须使用同一台机器、同一份代码和同样的额外 overrides：

```bash
for seed in 1 2 3; do
  bash reproduction/run_alfworld_single_gpu.sh grpo "$seed"
  bash reproduction/run_alfworld_single_gpu.sh target_acc "$seed"
  bash reproduction/run_alfworld_single_gpu.sh gmsv "$seed"
done

python -m reproduction.summarize_results
```

推荐先完成 seed 1 的三方法闭环，再开始 seed 2/3。不要在某个方法单独 OOM 时只降低它的 batch 或 rollout 数；参数变更必须同步应用到全部方法。

启动器不会覆盖已有运行目录。失败后重跑同一 seed 时应添加明确标签，例如 `RUN_TAG=retry1 bash reproduction/run_alfworld_single_gpu.sh gmsv 1`；结果汇总会保留失败运行的排除原因，并拒绝同一方法/seed 出现两个含糊的完成结果。

三项关键消融沿用相同入口与预算。`fixed_sigma` 严格使用论文 Table 3 的 `sigma_min=0.1`，而不是仓库默认的 `0.25`：

```bash
for seed in 1 2 3; do
  bash reproduction/run_alfworld_single_gpu.sh fixed_sigma "$seed"
  bash reproduction/run_alfworld_single_gpu.sh no_aux_sft "$seed"
  bash reproduction/run_alfworld_single_gpu.sh deterministic_mean "$seed"
done
```

## 6. 必须保存的证据

- 当前 Git commit 与 `git diff`；
- `python -m reproduction.verify_gpu_environment --json` 输出；
- 完整控制台日志；
- Hydra 展开的最终配置；
- checkpoints 与 `gmsv_runtime_state.json`；
- validation 和 train rollout details；
- 每次运行的开始/结束时间及 GPU 型号。

单卡启动器会把工作目录固定到仓库根，并在 `outputs/<run_name>/run_manifest.json` 自动保存启动命令、Git commit、工作区状态、补丁哈希、数据哈希、依赖版本、GPU 信息、退出状态和耗时，并把实际补丁保存为同目录的 `working_tree.patch`。Hydra 完整展开配置写入 `resolved_config.yaml`，终端输出写入 `console.log`，逐步指标写入 `metrics.jsonl`，训练 checkpoint 写入同一运行目录的 `checkpoints/`；checkpoint 核验还会写入 `checkpoint_identity.json`。每个新运行禁用隐式恢复，避免吸收其他实验的旧状态。

训练完成后运行 `python -m reproduction.summarize_results`。它会剔除失败或缺少最终验证指标的运行，并拒绝不同代码、数据或预算的结果混算。简历只引用该门禁生成的同预算数字。

## 7. OOM 排查顺序

保持实验定义不变，依次尝试：

1. 将 `actor_rollout_ref.rollout.gpu_memory_utilization` 从 `0.35` 降到 `0.30`；
2. 保持 micro-batch 为 1，确认 parameter/optimizer offload 已开启；
3. 开启 activation offload；
4. 三种方法共同降低 `data.max_prompt_length`，并在报告中注明偏离论文配置；
5. 最后才共同降低 tasks/step 或 rollouts/task，因为这会改变训练统计和实验定义。
