# 核心算法与物理规范 (Algorithm & Physics Spec)

## 1. 随机性安全机制 (RNG Safety)
**绝对禁止全局随机状态**。在高斯过程 (GP) 训练、搜索空间探索以及 Benchmark 数据集拆分中，严禁使用 `np.random.seed()` 或全局 `np.random` 调用。
* **原因**：全局随机数会被多线程（如配置了 20 线程加速时）并行测试污染，导致实验结果不可复现或发生死锁。
* **规则**：必须强制要求显式传递并使用局部的 `np.random.RandomState(seed)` 实例。

## 2. AI 打分边界 (LLM Bounds)
**大模型绝对不能直接评估整个搜索空间。**
* **原因**：搜索空间极大（可达上万个离散点），直接交由大模型打分会导致严重的 Token 爆炸和 OOM（Out Of Memory），并且彻底破坏 BO 的收敛数学保证。
* **规则**：必须先通过 Gaussian Process (GP Surrogate) 计算所有候选点的 Acquisition Score（如 UCB/EI），筛选出 Top-K（例如默认前 20 个），然后再将这 K 个点交由 LLM 执行物理规则判断（Yes/No 提示词），最终提取 `log_prob` 并通过混合公式打分。

## 3. TDD 强制执行区 (Test-Driven Development)
针对数学推导、矩阵变换以及数据解析代码，强制采用 TDD（测试驱动开发）。
* **范围**：任何涉及核心计算模块（如 `_mean_shift` 均值平移算法中的矩阵乘法）、复杂物理公式计算、以及解析 LLM 返回的字符串文本的功能。
* **规则**：在实现这些逻辑的业务代码前，必须先在 `tests/` 目录下编写明确输入输出断言的 `pytest` 用例，跑通失败后再进行实现。这对于保障科学计算在极端边界条件（如除 0 异常）下的鲁棒性至关重要。

## 4. 矩阵维度与异常兜底 (Matrix Safety)
* 所有的矩阵和张量操作必须在行内注释明确的维度（例如 `shape: (N, D)`）。
* 针对可能出现分母为 0 或 NaN 的数学计算（如 $\lambda = c / \sqrt{a^T \Sigma a}$），必须加入显式的异常捕获兜底逻辑（Fallback），例如 fallback 到 $\lambda=0$，确保算法流程能够不间断运行。

## 5. 数据集特征编码与高维陷阱 (Dataset Encoding & High-Dimensional Trap)

### Common Mistake: 盲目使用 One-Hot 编码导致的高维稀疏性陷阱

**Symptom**: 在某些子集（如 `buchwald_sub4`）上，传统或改进的高斯过程 (GP) 模型的 $t_{95}$ 收敛指标严重恶化，超参数优化导致协方差矩阵坍塌，模型预测质量极差。

**Cause**:
* **数据集跨产物合并**：为利用全局结构信息，`buchwald_sub4_train.csv` 并非仅有当前产物的 7 条数据，而是合并了全部 5 个子集的训练先验（共 35 条数据）。
* **特征维度暴增**：这 35 条数据包含了大量本产物原本不需要的反应物种类。如果使用常规的 One-hot 编码，四个化学变量（Reactant2, Ligand, Additive, Base）的独立类别数量会累加达到 **44 维** (15+4+22+3=44)。
* **结构信息丢失**：在 44 维的稀疏空间中仅有 35 个散落点，如果不引入化学先验，One-hot 会将这些底物视作毫无关联的独立特征，导致 GP 完全无法利用这些跨产物数据中的结构相似性。

**Prevention / Awareness**:
* **意识到高维稀疏性限制**：在处理跨产物合并的先验数据集（如 Buchwald）时，必须充分意识到常规的 44 维 One-hot 编码会导致严重的特征独立化与数据稀疏化问题。
* **特征工程需谨慎**：目前关于如何最优地处理这种 44 维特征孤立问题**尚未最终定性**。在进行后续特征工程时，必须同时兼顾降维和保留结构相似性，避免在极少样本（35条）下盲目采用一刀切的高维 One-hot 编码方案。

## 6. 场景：Acquisition Score 退化检测必须尺度不变

### 1. Scope / Trigger

- 触发条件：新增或修改 acquisition function、候选排序、诊断统计或实验验证门。
- 适用入口：`engine._score_statistics(scores: np.ndarray) -> dict[str, Any]`。
- 目的：区分“所有候选分数确实相同”与“分数绝对值很小、但仍有可用于 `argmax` 的相对差异”。

### 2. Signatures

```python
def _score_statistics(scores: np.ndarray) -> dict[str, Any]: ...
```

返回字段：

```python
{
    "min": float | None,
    "max": float | None,
    "std": float | None,
    "nonfinite_count": int,
    "is_constant": bool,
    "is_degenerate": bool,
}
```

### 3. Contracts

- 输入可为任意可转换为一维 `float64` 数组的 NumPy 数组。
- 统计只基于 finite values；NaN/Inf 必须计入 `nonfinite_count`。
- `is_degenerate = is_constant or nonfinite_count > 0`。
- 非零尺度数组使用相对判据：`std <= 1e-9 * max(abs(scores))`。
- `std == 0` 必须判为 constant。
- 全零数组使用绝对兜底判据 `std <= 1e-12`。
- 不能仅使用 `std <= 1e-12` 判断非零数组；EI（Expected Improvement）在 incumbent 很高时天然可能落在 `1e-11` 量级。

### 4. Validation & Error Matrix

| 条件 | 必须返回 |
|---|---|
| 所有值均非 finite | `min/max/std=None`, `is_degenerate=True` |
| 任一 NaN/Inf | `nonfinite_count>0`, `is_degenerate=True` |
| 全零 | `is_constant=True`, `is_degenerate=True` |
| 非零常数数组 | `std=0`, `is_constant=True` |
| 单一小 spike，其余为零 | 相对差异足够时 `is_constant=False` |
| 正常分散数组 | `is_constant=False`, `is_degenerate=False` |

### 5. Good / Base / Bad Cases

- Good：783 个候选中，索引 554 的 EI 为 `1.821902921587371e-11`，其余为 0；虽然 std 仅约 `6.5068e-13`，`argmax` 仍能唯一选出 554，因此不能判退化。
- Base：`np.full(100, 5.0)` 的 std 为 0，必须判退化。
- Bad：`np.array([0.1, np.nan])` 必须因 non-finite score 判退化。

### 6. Tests Required

- `tests/test_ei_stability.py::test_small_scale_ei_is_not_flagged_degenerate`
  - 断言 finite、`nonfinite_count == 0`、`is_degenerate is False`。
- `tests/test_ei_stability.py::test_truly_constant_scores_still_flagged_degenerate`
  - 断言真正常量数组仍为 degenerate。
- acquisition 单调性测试：均值明显高于 `best_f` 的候选应具有最高 EI。

### 7. Wrong vs Correct

#### Wrong

```python
is_constant = np.std(scores) <= 1e-12
```

绝对阈值把“绝对值很小但相对可排序”的 EI 误判为常量。

#### Correct

```python
score_std = float(np.std(finite))
abs_max = float(np.max(np.abs(finite)))
if abs_max > 0.0:
    is_constant = score_std <= 1e-9 * abs_max or score_std == 0.0
else:
    is_constant = score_std <= 1e-12
```

## 7. 场景：固定训练先验的可恢复 Hybrid 比较矩阵

### 1. Scope / Trigger

- 触发条件：运行、恢复、修改或解释 `HybridComparisonRunner` 的跨 surrogate 比较实验。
- 该场景涉及 CLI、实验持久化、LLM 环境与验证门，属于基础设施/跨层契约。

### 2. Signatures

```bash
uv run python hybrid_runner.py preflight
uv run python hybrid_runner.py run --workers <int>
uv run python hybrid_runner.py status
uv run python hybrid_runner.py report
```

```python
class HybridComparisonRunner:
    COMPOSITIONS = ["lgbo_manifold", "lgbo_dkl"]
    DATASETS = ["buchwald_sub4", "suzuki"]
    SEEDS = [100, 200, ..., 2000]
    N_ITERS = 40
    MATRIX_SIZE = 80
    PRIOR_PROTOCOL = "fixed_train_prior"
```

### 3. Contracts

- 矩阵固定为 `2 compositions × 2 datasets × 20 seeds × 40 iterations`。
- 每个 run 使用完整且固定的 train prior：Buchwald 35 点，Suzuki 29 点；`initial_indices == list(range(n_train_prior))`。
- `best_found` 是 20 seeds 内每个 run 的最终 best-so-far；报告中的值是 run 均值，不等同于单 seed 最大值。
- `global_best`：`buchwald_sub4=86.60`，`suzuki=99.90`。
- 必需环境变量：`DEEPSEEK_API_KEY`；可选：`DEEPSEEK_BASE_URL`、`DEEPSEEK_FLASH_MODEL`、`DEEPSEEK_MODEL`。
- `chat_engine` 只有显式配置时才覆盖环境模型；不得硬编码旧 relay model。
- 输出记录至少包含：`composition`, `dataset`, `seed`, `prior_protocol`, `n_train_prior`, `initial_indices`, `metrics`, `trajectory`, `diagnostics`, `status`。
- 并行运行使用进程级 seed 分组，并通过 `fcntl.flock` 对结果 JSON 做并发 read-modify-write；已通过验证的 matrix key 不得重复运行。

### 4. Validation & Error Matrix

| 条件 | 结果 |
|---|---|
| API key 未配置 | preflight/run 立即失败，不生成伪成功记录 |
| preflight 不完整 | full run 拒绝启动 |
| trajectory 不是恰好 40 步 | run 无效 |
| prior 点数或索引不符 | run 无效 |
| crash / GP fit failure / acquisition fallback | 对应 gate 非零，run 失败 |
| degenerate / non-finite acquisition score | run 失败 |
| mean-shift failure / surrogate fallback | run 失败 |
| LLM failure rate `> 10%` | run 失败；恰好 `10%` 允许 |
| transient HTTP 429/5xx | 最多重试 3 次，总 sleep 不超过 15 秒 |
| HTTP 400 等非 transient 错误 | 不重试，返回 error diagnostic |
| 80 个 run 全部有效 | status=`done`, pending=0，才允许写最终报告 |

### 5. Good / Base / Bad Cases

- Good：80/80 records 为 `ok`，所有强制 gate 为 0，LLM 总失败率不超过 10%，报告可生成。
- Base：已有 79 个有效 run、1 个 failed；resume 只运行该 pending key。
- Bad：把 historical `seeded_subsample`（5 点随机先验）的综合分直接与 `fixed_train_prior`（完整训练先验）排名，属于协议混比，不能据此声明方法本身冠军。

### 6. Tests Required

- matrix key 完整性：恰好 80 个唯一 key。
- resume：有效 key 跳过，failed/missing key 重跑。
- validation gate：逐项注入非零计数并断言失败原因。
- fixed prior：断言 Buchwald 35、Suzuki 29，且索引连续覆盖全部 train prior。
- LLM retry：429/503 后成功、400 不重试、超过上限返回 error。
- 多 worker 持久化：并发写入后不丢 record、不重复 key、JSON 可解析。
- final report：矩阵不完整时拒绝写，完整时包含 per-dataset mean/std/CI 与 composite score。

### 7. Wrong vs Correct

#### Wrong

```python
ctx.extra["chat_engine"] = params.get("chat_engine", "deepseek-v4-pro")
```

这会覆盖 `.env` 中的新 relay model，使真实 LLM 调用返回 `model_not_found`。

#### Correct

```python
ctx.extra["chat_engine"] = params.get("chat_engine")

chat_engine = ctx.extra.get("chat_engine")
if chat_engine:
    client.model = chat_engine
```

环境模型是默认来源；composition 仅在明确声明时覆盖。
