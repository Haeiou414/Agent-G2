# 让 Codex 操作 AutoDL GPU 实例

目标是让 Codex 桌面端通过 SSH 打开 AutoDL 上的远程项目。命令、文件读写和训练都发生在租用实例上；本机只负责显示界面和发出审批。

## 安全边界

- 不要把 AutoDL 密码、SSH 私钥、OpenAI API key 或 ChatGPT token 发到聊天中。
- 向 AutoDL 配置的是 `.pub` 公钥；没有 `.pub` 后缀的私钥永远不要上传。
- AutoDL 实例通常以 `root` 登录，因此批准命令前要检查目标路径。
- 释放或转交实例前执行 `codex logout`，并确认需要的日志和 checkpoint 已同步回本地。

## 1. 在本机准备 SSH 密钥

若 `~/.ssh/id_ed25519.pub` 已存在，可直接复用。否则在 Mac 终端执行：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

把下面命令显示的**公钥**完整复制到 AutoDL 控制台的“设置密钥登录”：

```bash
cat ~/.ssh/id_ed25519.pub
```

开机后，AutoDL 会给出类似下面的命令：

```text
ssh -p 10309 root@connect.nmb1.seetacloud.com
```

记录实际的主机名和端口；以下示例中的值必须替换成控制台显示的值。

## 2. 给 Codex 可识别的显式 SSH 别名

编辑本机 `~/.ssh/config`，添加：

```sshconfig
Host autodl-agent-g2
  HostName connect.nmb1.seetacloud.com
  User root
  Port 10309
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 6
```

先在本机终端验证：

```bash
ssh autodl-agent-g2
```

只有这一步能免密码成功，才继续配置 Codex。每次更换 AutoDL 实例或端口后，都要更新这一段。

## 3. 同步当前完整工作区

当前复现代码包含尚未推送到上游仓库的修改，不能只克隆作者仓库。先创建远程目录：

```bash
ssh autodl-agent-g2 'mkdir -p /root/autodl-tmp/agent-g2-reproduction'
```

然后在本机执行。结尾的两个 `/` 都有意义：

```bash
rsync -az --progress \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  /Users/haeiou/.codex/.chatgpt-projects/g-p-6aaf60a42d88819183a7377f489dac83/agent-g2-reproduction/ \
  autodl-agent-g2:/root/autodl-tmp/agent-g2-reproduction/
```

不要加 `--delete`，避免误删远程实验输出。验证代码已经到达：

```bash
ssh autodl-agent-g2 \
  'cd /root/autodl-tmp/agent-g2-reproduction && git status --short && git rev-parse --short HEAD'
```

## 4. 在 AutoDL 安装并登录 Codex

通过 SSH 登录实例后执行官方 Linux 安装器：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex --version
codex login --device-auth
```

设备登录会显示网址和一次性代码。网址在自己的浏览器中打开；不要把代码发给别人。登录后验证：

```bash
codex login status
command -v codex
```

Codex 桌面端使用远程登录 shell 启动服务，所以 `command -v codex` 必须在一次新的 `ssh autodl-agent-g2` 会话中仍能找到命令。如果找不到，把安装器提示的 bin 目录加入远程登录 shell 的 `PATH` 后重新连接。

## 5. 在 Codex 桌面端添加远程项目

1. 打开“设置 → 连接 → SSH”。
2. 添加或启用 `autodl-agent-g2`。
3. 选择远程目录 `/root/autodl-tmp/agent-g2-reproduction`。
4. 在该远程项目中新建任务；此后任务的终端、文件和 GPU 都来自 AutoDL。

若设置中没有 SSH 入口，先更新桌面应用。不要把 Codex App Server 的 WebSocket 端口直接暴露到公网；桌面端会通过 SSH 管理它。

## 6. 新远程任务的首条消息

```text
继续完成 Agent-G² 论文复现。工作区已经包含本地复现改动。先完整阅读 reproduction/README.md、reproduction/PLAN_zh.md、reproduction/CLOUD_RUNBOOK_zh.md 和 reproduction/AUTODL_REMOTE_zh.md；检查 git status，保留现有修改。先执行 GPU 环境自检和全部 CPU 测试，再按运行手册做作者 checkpoint 核验与 8-step smoke test。任何结果只能来自实际 metrics.jsonl 和 run_manifest.json，不得把论文数字写成本项目结果。训练长任务使用 tmux，并持续保存 outputs 证据。
```

## 7. 推荐执行顺序

1. `nvidia-smi` 与 `python -m reproduction.verify_gpu_environment`；
2. 执行 `bash reproduction/bootstrap_autodl.sh`，安装项目依赖并运行全部 CPU 测试；
3. 固定 revision 下载作者 checkpoint，先做只评测核验；
4. 运行 Agent-G² 8-step smoke test；
5. seed 1 的 GRPO、Target-accuracy、Agent-G² 三方法闭环；
6. 再决定是否续租，完成 seed 2/3 和三项关键消融；
7. 运行 `python -m reproduction.summarize_results`；
8. 把 `outputs/`、生成报告和必要 checkpoint 同步回本地。

训练开始后不要依赖 SSH 会话一直在线。使用 `tmux`，并在启动命令前记录 `nvidia-smi`；实例关机前先确认结果已经落在 `/root/autodl-tmp` 数据盘并同步回本地。

## 8. 将结果同步回本地

在本机执行，不使用 `--delete`：

```bash
rsync -az --progress \
  autodl-agent-g2:/root/autodl-tmp/agent-g2-reproduction/outputs/ \
  /Users/haeiou/.codex/.chatgpt-projects/g-p-6aaf60a42d88819183a7377f489dac83/agent-g2-reproduction/outputs/
```

释放实例前：

```bash
ssh autodl-agent-g2 'codex logout'
```
