# BOagent

贝叶斯优化（Bayesian Optimization, BO）算法研究仓库。正式算法只存在于 `packages/bo-core`；比赛实验在 `competition` 中消费正式 API，不复制实现。

## 目录边界

| 路径 | 职责 |
|---|---|
| `packages/bo-core/bo_core/` | 已通过验证的正式算法、数据接口与运行器 |
| `packages/bo-core/tests/` | 正式算法契约测试 |
| `competition/auto_research/` | 候选实验与实验评分；不得承载正式算法实现 |
| `competition/submission/` | 比赛提交入口与环境配置；直接依赖安装后的 `bo-core` |
| `datasets/` | 唯一数据源；每个数据集使用固定文件契约 |

## 数据契约

数据集由 `bo_core.benchmark.datasets.DATASETS` 注册，并通过 `load_dataset(dataset_id)` 加载。每个数据集目录必须包含：

- `searchspace.csv`
- `train.csv`
- `test.csv`
- `test_features.csv`
- `options.json`
- `README.md`

算法和实验不得自行拼接数据路径、重复声明特征列或硬编码 `global_best`。

## 固定工作流

1. 在 `competition/auto_research` 编写候选实验；候选代码留在实验目录。
2. 使用固定数据注册表、20 个种子和 40 轮预算与 `gpbo`、正式 `lgbo` 比较。
3. 只有通过预先声明的验证门后，才把候选实现迁移到 `packages/bo-core`。
4. 迁移时补齐正式测试，并删除实验目录中的候选实现，保持单一实现。
5. `competition/submission` 只调用已晋级的 `bo-core` API。

当前正式比赛方法：`LGBOEngine`。已完成但未通过晋级门的 Chem-LGBO、reranking 与旧 hybrid 变体不属于正式核心。

## 常用命令

```bash
uv sync --frozen
uv run --frozen pytest packages/bo-core/tests competition/auto_research/tests competition/submission
uv run --frozen ruff check packages/bo-core competition competition/submission
```
比赛提交（默认完整 2 × 20 × 40 路径；全部轨迹完成并通过严格校验后才生成 `competition/submission/results/summary_metrics.csv`）：

```bash
uv run --frozen python competition/submission/code/main/run_submission.py
```

指定数据集、种子或轮数的子集运行只生成轨迹，不生成完整汇总。

导出独立提交快照（包含 `pyproject.toml` 与 `uv.lock`）：

```bash
uv run --frozen python competition/submission/export_snapshot.py /tmp/boagent-submission
```

快速验证正式 GPBO 路径：

```bash
uv run python -m bo_core.benchmark.lgbo_runner \
  --datasets buchwald_sub4 \
  --methods gpbo \
  --seeds 100 \
  --n_iters 1 \
  --workers 1 \
  --backend sklearn \
  --output_dir /tmp/boagent-smoke
```
