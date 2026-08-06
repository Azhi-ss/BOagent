# Competition Auto-Research

候选算法实验区。这里负责提出、运行和评估候选；正式算法实现只存在于 `packages/bo-core`。

## 边界

- 可导入：`bo_core.benchmark.load_dataset`、正式 optimizer、公共评测函数。
- 不可复制：`packages/bo-core` 中的算法实现。
- 不可写入：`packages/bo-core`、`competition/submission`。
- 生成产物统一写入 `artifacts/`、`experiments/`、`summaries/` 或 `reports/`；这些目录不进入版本控制。

## 新实验最小契约

每个候选实验必须显式记录：

- 候选名称和源文件哈希；
- 数据集 ID、固定 train prior、seed、迭代预算；
- 与 `gpbo`、当前正式 `lgbo` 的同协议对照；
- 每个数据集的 `best_found`、`initial_round_found_best`、`t95`、`AUC_best_so_far`；
- 预先声明的晋级条件及通过/失败原因。

数据只能通过 `load_dataset(dataset_id)` 获取。不得自行拼路径、重复定义特征列、候选 options 或 `global_best`。

## 晋级

候选只有在完整矩阵验证通过后才能迁入 `packages/bo-core`。迁移必须同时完成：

1. 正式实现；
2. 正式契约测试；
3. `lgbo_runner` 或新的稳定公共入口；
4. 删除本目录中的重复实现；
5. 提交入口改为消费正式 API。

当前本目录仅保留通用结果聚合器 `analyze.py`。新候选从独立实验文件开始，不恢复已失败的 Chem-LGBO、reranking 或旧 hybrid 实现。
