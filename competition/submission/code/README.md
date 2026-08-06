# 比赛提交复现指南

提交目录只维护入口和容器配置；算法源码始终以 `packages/bo-core` 为唯一来源。

## 仓库内运行

完整复现（2 个数据集 × 20 个种子 × 40 轮）会在全部轨迹完成后调用严格评测器；只有轨迹矩阵、元数据和数据集 oracle 校验全部通过，才原子写入 `competition/submission/results/summary_metrics.csv`：

```bash
uv sync --frozen
uv run --frozen python competition/submission/code/main/run_submission.py
```

快速或自定义子集运行只写轨迹，不生成或宣称完整的 `summary_metrics.csv`：

```bash
uv run --frozen python competition/submission/code/main/run_submission.py \
  --datasets buchwald_sub4 --seeds 100 --n-iters 1 \
  --backend sklearn --output-dir /tmp/submission-smoke/optimization_trajectories
```

## 导出独立快照

```bash
uv run --frozen python competition/submission/export_snapshot.py /tmp/boagent-submission
```

导出目录包含：

- `code/`：提交入口、Dockerfile、环境与评测脚本；
- `packages/bo-core/`：导出时刻的正式算法源码；
- `datasets/`：比赛使用的 Buchwald 与 Suzuki 数据；
- `pyproject.toml` 与 `uv.lock`：与仓库一致、可冻结安装的依赖合同。

独立运行：

```bash
cd /tmp/boagent-submission
uv sync --frozen --no-dev
uv run --frozen python code/main/run_submission.py
```

容器使用同一份 lockfile：

```bash
docker build -f code/Dockerfile -t boagent-submission .
```

轨迹写入 `results/optimization_trajectories/`，完整验证后的汇总写入 `results/summary_metrics.csv`。算法变更必须先在 `packages/bo-core` 验证，再重新导出快照；禁止在提交目录维护第二份算法实现。
