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

## 8. 场景：Chem-LGBO 字段级稀疏子空间引导

### 1. Scope / Trigger

- 触发条件：新增、修改、恢复或解释 `ChemLGBOEngine` 的 LLM guidance、候选池 mask、均值迁移、反事实或 Chem-LGBO 比较矩阵。
- 目标：让 LLM 只表达可验证的字段级稀疏约束；GP/EI 仍负责候选池上的最终选点。
- Legacy LGBO 的 Point/Hamming guidance 保持原协议；Chem-LGBO v1 不把 LLM 变成全池打分器。

### 2. Signatures

```python
def parse_subspace_response(
    text: str,
    feature_cols: Sequence[str],
    options: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]] | None, str]: ...

def build_subspace_mask(
    candidate_features: pd.DataFrame,
    subspace: Mapping[str, Sequence[str]],
) -> np.ndarray: ...

def masked_mean_shift(
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray: ...

def generate_counterfactual_indices(
    *, candidate_features: pd.DataFrame,
    feature_options: Mapping[str, Sequence[str]],
    subspace: Mapping[str, Sequence[str]],
    queried_mask: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    best_f: float,
    expected_improvement: Callable[..., np.ndarray],
    rng: np.random.RandomState,
    count: int,
) -> list[int]: ...


@dataclass(frozen=True)
class LlmCallResult:
    status: str
    provider: str
    model: str
    content: str
    usage: dict[str, Any]
    error: str | None = None
    logprobs: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None

def chat(
    messages: list[dict[str, Any]],
    max_tokens: int = 2048,
    extra_body: dict[str, Any] | None = None,
    *,
    temperature: float = 0.0,
) -> LlmCallResult: ...

class LGBOEngine:
    def __init__(
        self,
        dataset: str,
        seed: int = 100,
        use_llm: bool = False,
        n_iters: int = 40,
        K: int = 50,
        n_restarts: int = 10,
        alpha: float = 1e-2,
        xi: float = 0.01,
        chat_engine: str = "deepseek-v4-flash",
        llm_max_tokens: int = 8192,
        reasoning_effort: str = "low",
        llm_temperature: float = 0.0,
        failure_log: str | Path | None = "lgbo_llm_failures.log",
        backend: BackendName = "botorch",
    ) -> None: ...

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> tuple[LlmCallResult | None, str]: ...

class ChemLGBOEngine(LGBOEngine):
    def __init__(
        self,
        dataset: str,
        seed: int = 100,
        use_llm: bool = True,
        n_iters: int = 40,
        *,
        n_counterfactuals: int = 0,
        outcome_feedback: bool = False,
        llm_temperature: float = 0.2,
        **legacy_kwargs: Any,
    ) -> None: ...
```

### 3. Contracts

- Chem-LGBO 必须通过 `tool_choice` 强制调用 `propose_sparse_subspace`；该 tool 的参数只含必填对象 `subspace`。只接受恰好一个目标 tool call，不得从 prose 回退解析建议；`content == ""` 且 `tool_calls` 非空仍是成功响应。
- Tool arguments 必须规范化为最终 JSON 对象 `{"subspace": {<feature>: [<string>, ...]}}` 后复用 `parse_subspace_response`；顶层键必须恰为 `subspace`，subspace 非空，字段必须属于 `feature_cols`，每个数组非空、元素为字符串、无重复且均来自该字段的 `options`。
- Parser 失败原因固定为：`empty_response`, `invalid_json`, `invalid_schema`, `unknown_field`, `empty_choice`, `duplicate_value`, `unknown_value`；成功返回 `accepted`。
- 每个 Chem step 最多两次 LLM 调用（初次 + 一次纠错）。结构、字典或候选池语义失败可重试；网络、鉴权、超时等 transport 失败不得启动 ReAct 重试。
- ReAct 回执必须追加原 assistant `tool_calls`，再追加 `role="tool"`、匹配被检查调用的 `tool_call_id` 和机器可读错误；缺失 ID、错误/多个 tool call 或无法形成合法 tool 回执时，改用局部 `role="user"` 纠错消息，绝不能发送非法 tool message。重试对话只存在于当前 `_llm_mean_shift()`，不得进入 trajectory 或下一 step prompt。
- `guidance_artifacts` 必须记录 `react_retried`, `react_first_reason`, `llm_attempts`, `tool_call_id`；`raw_response` 保存实际解析的 tool arguments。实验 provenance/model config 必须记录 Chem 实际 `temperature=0.2`, `response_mode="tool_call_react"`, `max_react_retries=1`；普通 LGBO 和其他 client 调用默认温度仍为 `0.0`。
- 继续旧 artifact 时，只有 records 为空才可补齐缺失的 Tool Call 协议字段；已有 records 的 plain-JSON/`temperature=0.0` provenance 必须按不匹配拒绝，避免混合协议 resume。
- `build_subspace_mask` 对每个指定字段做候选行 membership 的逻辑 AND；省略字段不限制候选。联合非法组合、空交集、仅命中已查询点和覆盖全部剩余池都必须走可观测 fallback，不得硬过滤或伪造候选。
- `masked_mean_shift` 是 Chem-LGBO v1 当前实验变体的固定奖励实现，不等同于论文基于 posterior covariance 的 LGBO Proposition 1：
  \[
  \mu'_i = \begin{cases}\mu_i + \sigma_i,& mask_i\\\mu_i,&\text{otherwise}\end{cases}
  \]
  并保持输入 `sigma` 不变；向量必须是一维、finite、等长，mask 必须是一维布尔数组。任何结论必须归因于“categorical subspace + fixed one-sigma bonus”组合，不得外推为整个 LGBO 无效。
- Chem guidance 只改变 mean。最终 EI 在完整候选池上计算，并排除 `queried`；有效 guidance 不能保证最终 query 落在 mask 内。
- `generate_counterfactual_indices` 只接收 feature、posterior、mask 和 EI 所需数值，不得读取 `pool_yield` 或任何 oracle。使用局部 `RandomState(seed * 1000 + iteration)`，最多生成 100 个索引，索引来自未查询池且同一 seed 可重现。
- 每个 Chem iteration 必须保留紧凑 trajectory diagnostics；原始响应、parser reason 和反事实索引只进入 experiment-side `guidance_artifacts`，不得进入 competition `.pt` payload。
- `LGBOEngine.health` 的 `gp_fit_fallbacks`, `gp_predict_fallbacks`, `acquisition_fallbacks`, `nonfinite_acquisition_scores`, `duplicate_queries` 在可纳入实验矩阵的 record 中必须全部为零。非有限 acquisition 不能由 `argmax` 隐式选取已查询点；无剩余候选应显式失败。
- 实验 artifact 的已有 records 必须带非空 provenance；首次原子组落盘时绑定当前 source/config/runtime hashes。缺失或不匹配必须在任何新 LLM 调用前拒绝恢复。
- 比较矩阵固定使用 `fixed_train_prior`（Buchwald 35、Suzuki 29），每个 record 恰好 40 个不重复 query；screening 先比较 seed-level paired AUC 均值，full 只有在四组 paired AUC bootstrap CI 下界均严格大于 0 时才可 `promote_chem_lgbo`，否则 `keep_legacy_lgbo`。
- Prompt-feedback 时序：第 $t$ 轮 prompt 最多接收第 $t-1$ 轮已完成 guidance 的 outcome；当前轮 `observed_yield` 必须在 query 执行后才写入 `previous_outcome`，不得把当前 oracle 结果泄漏进当前 prompt。`outcome_feedback=False` 时不得生成或注入该段。
- `PreviousGuidanceOutcome` 必须记录前轮 guidance status、是否命中所建议子空间、实际 observed yield 和该轮 query 前 incumbent；prompt 只可表述这些已完成事实，不得建议具体下一点。
- 配对 prompt 消融固定从相同 v1 source artifact 重建相同 posterior hash、GP baseline 与 state manifest；Control/Treatment 唯一差异是 Treatment 接收上一轮 completed outcome。每个 state 两个 variant 必须成对，执行顺序按 pair index 交替，防止固定先后偏差。
- 消融 artifact 必须原子持久化 `source_sha256`, `state_manifest`, `model_config`, source hashes, records, analysis, gate 与 `record_count`；source 或 manifest 变化时必须在新 LLM 调用前拒绝 resume。
- 每个消融 record 必须记录 matched-random counterfactual percentile（最多 100 个、局部 RNG）、previous mask repeat、previous outcome stratum 与是否产生新 incumbent；aggregate 必须在 seed 内先配对，再跨 seed 汇总 Treatment-Control、vs-GP、fallback、coverage、counterfactual、repeat 和 incumbent rate。
- CLI 契约：`preflight` 只读 source 并输出 hash/record/state count，不创建 artifact、不调用 LLM；`run` 必须从环境创建并验证 client 后执行/恢复；`report` 只读并返回已有 artifact，不调用 LLM。

### 4. Validation & Error Matrix

| 条件 | 必须结果 |
|---|---|
| 空/非 JSON/错误顶层 schema | parser 返回对应失败 reason，纯 GP fallback |
| 未知字段、未知值、空值或重复值 | parser 返回对应失败 reason，纯 GP fallback |
| 联合候选为空 | `empty_intersection` fallback |
| mask 只包含已查询点 | `already_queried_only` fallback |
| mask 覆盖全部剩余池 | `uninformative_full_pool` fallback |
| guidance transport 失败，或 `content` 与 `tool_calls` 同时为空 | `llm_error` / `empty_response` fallback |
| 成功响应 `content=""` 且含合法目标 tool call | 解析 tool arguments，不得判为空响应 |
| 缺目标调用、错误/多个调用、非法 arguments 或 parser/候选池语义失败 | 最多一次局部纠错；第二次仍失败则记录最终 fallback |
| transport 失败 | 不做 ReAct 重试，直接保留原 `llm_error` fallback |
| tool 回执缺失/不匹配 `tool_call_id` | 不发送非法 `role="tool"` 消息；使用局部 user feedback 重试 |
| GP fit/predict 或 acquisition 非有限 | health counter 非零，record 不得通过矩阵验证 |
| LLM fallback rate 大于 10% | record / matrix 验证失败 |
| provenance 缺失而 records 非空 | 恢复立即拒绝，不得继续调用 LLM |
| 已有 records 的 provenance 缺失 Tool Call 协议字段 | 恢复立即拒绝，不得把旧 plain-JSON 运行静默升级为新协议 |
| screening 任一 paired AUC mean 小于等于 0 | `screening_stopped`, 不运行 full |
| full 任一 paired AUC CI 下界小于等于 0 | `keep_legacy_lgbo`，不得报告 promotion |
| 当前轮 prompt 含当前轮 observed yield | 测试失败；必须改为 query 完成后更新供下一轮使用 |
| Control 收到 previous outcome，或 Treatment 有其他 prompt/state 差异 | 消融无效，拒绝解释结果 |
| source hash / state manifest 与已有消融 artifact 不一致 | resume 立即拒绝，不调用 LLM |
| preflight/report 创建 client 或发起 LLM | 违反 CLI 合同 |

### 5. Good / Base / Bad Cases

- Good：合法 sparse JSON 形成非空且非全池的联合 mask；逐点 shift 后 mask 内恰为 `mu + sigma`，最终 EI 选出未查询点，artifact 与 trajectory 的 selected index 一致。
- Base：LLM 返回合法但 mask 为空、全池或已查询-only；记录 fallback reason，仍由未修改的 GP/EI 选点，且不产生反事实数据。
- Tool-call Good：首轮 `unknown_value` / `already_queried_only` / `uninformative_full_pool` 被拒绝，带匹配 `tool_call_id` 的一次纠错返回合法子空间，最终 artifact 记录 `react_retried=True`, `llm_attempts=2`, `parser_reason="accepted"`。
- Tool-call Base：合法目标 tool call 的 `content` 为空；仍从 arguments 解析并接受。Transport 失败则不重试，保持纯 GP fallback。
- Tool-call Bad：接受多个调用中的任意一个、从 prose 猜 JSON、发送缺失或错误 `tool_call_id` 的 tool message、把局部纠错对话带入下一 step，或让已有 records 的旧协议 artifact 静默 resume。
- Bad：把 `pool_yield` 传入 prompt、mask 或 counterfactual；把 mask 当硬候选过滤器；或以单个 seed/单轮收益替代 seed-level paired AUC gate。以上均违反协议。

### 6. Tests Required

- `packages/bo-core/tests/test_chem_lgbo_parser.py`：逐项断言 schema、字段、值、重复和 JSON failure reason。
- `packages/bo-core/tests/test_chem_lgbo.py`：断言联合 mask、缺省字段、空/full/already-queried fallback、精确 `+1 sigma`、局部 RNG 可重现和无 Yield 读取。
- `packages/bo-core/tests/test_lgbo_engine.py`：断言 Legacy compatibility、健康计数、非有限 acquisition 安全回退和不重复 query。
- `Compitetion/auto_research/tests/test_chem_lgbo_experiment.py`：断言固定 prior、完整矩阵 key、原子恢复、provenance、fallback/health gate、paired AUC mean/CI 和 screening/full 状态机。
- `packages/bo-core/tests/test_lgbo_runner.py`：断言 `gpbo -> LGBOEngine(False)`、`lgbo -> LGBOEngine(True)`、`chem_lgbo -> ChemLGBOEngine(True)`，真实一步落盘 CSV/.pt 且不泄漏 guidance artifacts。
- `Compitetion/auto_research/tests/test_chem_lgbo_prompt_ablation.py`：断言上一轮 outcome contract、相同 posterior/GP/state 重建、交替执行顺序、100 个反事实与 repeat/stratum/incumbent 聚合、source/manifest resume 拒绝以及 preflight/run/report CLI 边界。
- `packages/bo-core/tests/test_llm_retry.py`：断言显式温度、受保护 payload 字段、空 content + tool call 成功及默认纯文本行为不变。
- Tool Call/ReAct 边界测试必须断言：合法首轮接受；字典/语义失败一次自愈；连续失败只落一个最终 fallback；错误/多个/missing-ID 调用使用合法 feedback 角色；重试消息不泄漏；accepted/fallback telemetry 完整；records-bearing 旧 provenance 拒绝恢复；package/submission 镜像等价。

### 7. Wrong vs Correct

#### Wrong

```python
# LLM/heuristic directly ranks or filters the full oracle-backed pool.
scores = llm_score(candidate_features, pool_yield)
query = int(np.argmax(np.where(mask, scores, -np.inf)))
```

这会泄漏 oracle、把 guidance 变成硬过滤，并绕过 GP/EI 的完整池选择与统计对照。

```python
# Wrong: prose fallback and an invalid tool result without a matching id.
subspace = parse_subspace_response(result.content, feature_cols, options)
messages.append({"role": "tool", "content": error})
```

这会绕过强制 Tool Calling，并产生 OpenAI-compatible 网关会拒绝的非法回执。

#### Correct

```python
mask = build_subspace_mask(test_df[feature_cols], subspace) & remaining
shifted = masked_mean_shift(mu, sigma, mask)
ei = expected_improvement(shifted, sigma, best_f)
ei = np.where(remaining & np.isfinite(ei), ei, -np.inf)
query = int(np.argmax(ei))
```

```python
# Correct: parse the forced call's arguments and keep one repair turn local.
result = client.chat(
    messages,
    extra_body={"tools": [PROPOSE_SUBSPACE_TOOL], "tool_choice": PROPOSE_SUBSPACE_TOOL_CHOICE},
    temperature=0.2,
)
raw = result.tool_calls[0]["function"]["arguments"]
subspace = parse_subspace_response(raw, feature_cols, options)
```

若首轮失败，只有存在唯一目标调用及其 ID 时才追加匹配的 assistant/tool 消息；否则追加局部 user feedback。最多纠错一次。

LLM 只产生可验证字段约束；oracle 只在真实 query 后记录 observed yield；最终选点仍由 full-pool GP/EI 完成。

## 9. 场景：Chem-LGBO 离线奖励强度重放

### 1. Scope / Trigger

- 触发条件：已有 Chem-LGBO prompt-ablation artifact，需要在不重新调用 LLM、不改变 posterior 和已保存 subspace 的前提下比较奖励强度。
- 目标：把 guidance 方向与奖励尺度拆开，回答固定 `+1σ` 是否过强；这不是新的在线算法或论文 covariance mean-shift 实现。

### 2. Signatures

```python
def replay_beta_sweep(
    state_source_path: Path,
    guidance_source_path: Path,
    output_path: Path,
    *,
    betas: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
) -> dict[str, Any]: ...
```

### 3. Contracts

- 输入固定为已完成 Chem-LGBO v1 state artifact 与 prompt-ablation guidance artifact：前者提供 `(dataset, seed)` 对应的历史 trajectory，后者提供每个 state 的 `dataset`, `seed`, `step`, `variant`, `subspace`, `posterior_hash`。重放不得创建 client、调用 LLM、修改任一源 artifact 或读取未记录的 response。
- 对每个 state 只重建一次相同固定训练先验、相同历史 trajectory 和相同 posterior；必须验证重建后的 `posterior_hash` 与 record 完全一致。
- `fallback=True` 或 `subspace` 为空的 record 在所有 $\beta$ 下都使用纯 GP；否则以剩余候选上的联合 mask 计算：
  \[
  \mu_\beta(x)=\mu(x)+\beta\sigma(x)\mathbf 1[x\in S],\qquad
  \beta\in\{0,0.1,0.25,0.5,1.0\}.
  \]
- 每个 $\beta$ 的 EI 必须在完整候选池计算、排除已查询点并拒绝非 finite acquisition；`beta=0` 必须逐 state 精确复现保存的 `gp_index` 和 `gp_yield`。
- 输出按原 `state_key` 与 `variant` 保留逐 state 的 `selected_index`, `selected_yield`, `selected_in_subspace`, `improved_incumbent`，并汇总 dataset/variant/beta 的样本数、相对 GP 的 yield delta、胜/平/负计数、命中 subspace 比例和新 incumbent 比例。
- artifact 必须记录两个 source 的 SHA-256、固定 beta 列表、逐 state records、aggregate 与生成时间；同一路径已有 artifact 的任一 source hash 或 beta 列表不匹配时必须拒绝覆盖。
- 解释边界：若所有 $\beta>0$ 均不优于 $\beta=0$，可否定当前已保存 categorical subspace 的离线奖励信号；不得据此否定论文 covariance LGBO 或未测试的 prompt/guidance 表示。

### 4. Validation & Error Matrix
| 条件 | 必须结果 |
|---|---|
| source 缺 records 或必要 state 字段 | 显式失败，不生成部分报告 |
| duplicate `(state_key, variant)` | 显式失败 |
| posterior hash 不一致 | 显式失败，不继续该 artifact |
| beta 非 finite 或小于 0 | 显式失败 |
| `beta=0` 未复现保存 GP 选择 | 显式失败 |
| fallback/空 subspace | 所有 beta 与纯 GP 相同 |
| 非 fallback guidance 的重建 mask 为空或覆盖全部剩余池 | 显式失败，说明 source/state 语义漂移 |
| 任一 source hash / beta 列表与已有输出不匹配 | 拒绝 resume/覆盖 |

### 5. Tests Required

- 小型合成 state 断言 $\beta=0$ 等于 GP、正 $\beta$ 只改变 mask 内 mean，并可改变最终 EI 选择。
- 断言 fallback/空 subspace 对所有 beta 保持 GP 选择。
- 断言 duplicate state、posterior hash 漂移、非法 beta 和输出 provenance 不匹配均失败。
- CLI/入口测试必须在 monkeypatch client 构造为异常时仍成功，证明重放不会调用 LLM。


## 10. 场景：Chem-LGBO 证据门控 shortlist 重排

### 1. Scope / Trigger

- GP/EI 先在完整未查询候选池中确定稳定 top-5 shortlist；LLM 只能重排这 5 个完整候选，不能生成条件、扩大候选池或读取 oracle yield。
- 该路径是可回滚的附加实验；证据 gate 未通过时保持纯 GP 行为。

### 2. Contracts

- `ChemGPShortlistAdapter.shortlist()` 必须按 acquisition 降序、pool index 升序打破并列，排除已查询点，并返回恰好 5 个带公开特征的 `RankedCandidate`。
- `DeepSeekCandidateReranker.rerank()` 必须使用强制 `rank_shortlist` tool call；输入只包含 shortlist ID、GP rank/acquisition 和公开条件字段，不含 observed/selected yield。
- `SelectCandidateUseCase.execute()` 只接受 shortlist ID 的完整排列。gate 禁用/失败、解析失败、LLM 调用失败、重复/缺失/额外 ID 时，必须返回原 shortlist 首项，即完全相同的 GP winner。
- evidence artifact 的 provenance 必须精确匹配 `source_sha256`, `model`, `prompt_version`, `shortlist_size`, `state_manifest`，且 artifact 自身 `passed=true`；否则 gate 关闭。
- 离线 evaluator 必须从保存状态只计算一次 GP shortlist，并让 prompt、GP winner、matched-random 和全部指标共用该不可变 tuple；matched-random 使用完整 `(dataset, seed, step)` state key 的 SHA-256 派生。
- evaluator 只可在 Buchwald/Suzuki 均覆盖完全相同的 `{100,200,300,400,500}` seed/step manifest、所有 gate 指标均 finite、GP/random 配对 bootstrap 95% lower bound 均为正、failure rate 为 0、每个成功 state 都提供 confidence、mean ranking/pairwise accuracy 均至少 `0.6`、mean Brier score 不高于 `0.25` 时设置 `passed=true`；CLI 不接受人工 `--gate-passed` 覆盖。
- `run --model` 必须与实际 client model 完全一致；tool-call payload 只允许 `ordered_ids` 与可选 `confidence`，malformed/额外字段必须作为 `invalid_response`；`preflight` 与 `report` 不得构造 LLM client。

### 3. Validation Matrix

| 条件 | 必须结果 |
|---|---|
| gate 未批准或 provenance 漂移 | 不调用 LLM，返回精确 GP winner |
| tool call 缺失、名称错误或 arguments 非法 | 标记 `invalid_response`，返回精确 GP winner |
| ordered IDs 不是 shortlist 的完整排列 | 标记 `invalid_permutation`，返回精确 GP winner |
| LLM 传输/服务异常 | 标记 `llm_error`，返回精确 GP winner |
| prompt 含 outcome/yield 字段 | 测试失败 |
| duplicate state 或已有 artifact provenance 不匹配 | evaluator 显式失败 |
| 证据覆盖或性能门槛不足 | artifact `passed=false` 并记录 reasons |

### 4. Tests Required

- 用例测试覆盖 gate 关闭、解析失败、LLM 异常及重复/缺失/额外 ID，均断言 exact GP fallback。
- 适配器测试覆盖 top-5、queried exclusion、并列稳定排序、forced tool call 和无 oracle 字段。
- evaluator 测试覆盖 matched-random、逐 seed 指标、自动失败 gate、duplicate/provenance 拒绝，以及 preflight/report 无 LLM 构造。