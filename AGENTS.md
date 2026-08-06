# BOagent AI 开发者指南

BOagent 是贝叶斯优化（Bayesian Optimization, BO）算法研究仓库。正式算法实现只存在于 `packages/bo-core`；比赛实验必须消费正式 API，不维护第二套实现。

## 仓库边界

| 路径 | 职责 |
|---|---|
| `packages/bo-core/bo_core/` | 已晋级的算法、数据接口和基准运行器 |
| `packages/bo-core/tests/` | 正式代码的契约与回归测试 |
| `competition/auto_research/` | 候选实验与评估，不承载永久算法实现 |
| `competition/submission/` | 导出可复现快照并运行比赛提交 |
| `datasets/` | 唯一数据注册表内容 |

当前仓库没有受支持的前端或 API 服务。不要恢复或引用已删除的 `apps/api`、`apps/web` 路径。

## 正式执行流

1. `bo_core.benchmark.datasets.DATASETS` 注册数据集 ID、目录、特征列、目标列和优化方向。
2. `bo_core.benchmark.load_dataset(dataset_id)` 从 `datasets/` 加载固定文件契约。
3. `bo_core.optimization.lgbo.LGBOEngine` 承载正式 LGBO 循环；`gpbo` 通过关闭 LLM 引导复用同一引擎。
4. `bo_core.benchmark.lgbo_runner` 执行数据集 × 方法 × 种子矩阵，并输出轨迹和汇总指标。
5. `competition/auto_research` 将候选与正式 `gpbo`、`lgbo` 按同一协议比较。只有通过预先声明的晋级门，候选才迁入 `packages/bo-core`，随后删除实验副本。
6. `competition/submission/code/main/run_submission.py` 调用安装后的 `bo-core`；`competition/submission/export_snapshot.py` 导出算法包、提交代码和比赛数据集。

算法和实验只能通过数据注册表与 `load_dataset` 获取数据。不得自行拼接数据路径、重复声明特征 schema 或硬编码 `global_best`。

## 支持的命令

所有命令从仓库根目录运行：

```bash
uv sync
uv run pytest packages/bo-core/tests competition/auto_research/tests competition/submission/test_export_snapshot.py
uv run ruff check packages/bo-core competition competition/submission
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

比赛提交与快照导出：

```bash
uv run python competition/submission/code/main/run_submission.py
uv run python competition/submission/export_snapshot.py /tmp/boagent-submission
```

## 修改约束

- `packages/bo-core` 保持单一正式实现；实验区只导入正式 API。
- 正式行为变化时，补充或更新可观察契约的测试。
- 优化和基准路径使用局部随机数状态，避免并行运行互相污染。
- API 密钥只从环境变量读取，不得硬编码。
- 未经明确许可，不修改 `.env` 或生成的结果文件。
- 修改数值、硬件或外部库相关代码前，查阅 `docs/` 中对应参考资料。
