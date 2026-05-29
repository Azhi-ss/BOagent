# BOagent: GP+LLM Acquisition & Benchmark 对齐 PVK-LLM

**日期:** 2026-05-29
**状态:** 设计中
**范围:** BOagent 后端 acquisition function 重构 + benchmark 基础设施

## 背景

BOagent 基于 PVK-LLM 开发，但存在两个核心偏差：

1. **Acquisition function 缺失 GP 预筛选阶段**：PVK-LLM benchmark 脚本（`band_alignment_opt/`, `defects_doping_opt/`）使用 `CustomLLM_ACQ`，流程为 GP 训练 → UCB 评分 → Top-20 送 LLM 精选。BOagent 当前直接导入核心版 `LLM_ACQ`，纯 LLM 生成候选点，跳过了 GP。
2. **无 benchmark 基础设施**：PVK-LLM benchmark 有 train/test split、multi-seed 运行、CSV+JSON 结果保存、泛化性能评估。BOagent 完全没有。

## 目标

- 实现 GP + LLM 两阶段 acquisition function，对齐 PVK-LLM benchmark 逻辑
- 添加 benchmark 基础设施（CLI + API），支持 multi-seed 运行和结果持久化
- 覆盖 band_alignment 和 defects_doping 两个任务
- 保持 LLM surrogate model（`LLM_DIS_SM` / `LLM_GEN_SM`）不变

## 非目标

- 不修改 PVK-LLM 核心库（`pvk_bo/`）中的任何代码
- 不改变现有 session-based API（`RealPvkBoRuntime`、`OptimizationSessionRuntime`）的接口
- 不修改前端
- 不添加新的 surrogate model 变体

---

## 架构设计

### 文件变更

```
BOagent/backend/
├── gp_llm_acq.py              # [NEW] GP + LLM 两阶段 ACQ 实现
├── benchmark/
│   ├── __init__.py             # [NEW]
│   ├── data_loader.py          # [NEW] Excel 数据加载 + train/test split
│   ├── runner.py               # [NEW] Benchmark 执行引擎
│   └── cli.py                  # [NEW] CLI 入口 (argparse)
├── pvk_llm_bo_runtime.py       # [MODIFY] 可选注入 GP+LLM ACQ
├── pvk_session_runtime.py      # [MODIFY] benchmark 模式路由
└── api.py                      # [MODIFY] benchmark API endpoint
```

### 模块职责

#### `gp_llm_acq.py` — GP+LLM 两阶段 ACQ

```
class GPLLM_ACQ:
    """
    GP 预筛选 + LLM 精选 两阶段 acquisition function。

    对齐 PVK-LLM benchmark 中 CustomLLM_ACQ 的逻辑，
    但泛化为支持多个任务。

    Pipeline:
        1. 加载 Excel 全量数据
        2. 训练 sklearn GaussianProcessRegressor
        3. 对所有未观测点做 UCB 预测
        4. 取 top-k（默认 20）送 LLM 做材料学推理
        5. LLM 返回精选的 n_candidates 个候选
        6. 返回 candidate_points DataFrame
    """

    __init__(self, task_context, n_candidates, n_templates, lower_is_better,
             chat_engine, top_k=20, alpha=0.1):
        ...

    get_candidate_points(self, observed_configs, observed_fvals, alpha):
        # 返回 (candidate_points_df, cost, time_taken)
        ...
```

**关键实现细节：**

- GP 使用 `sklearn.gaussian_process.GaussianProcessRegressor`，kernel = `C(1.0) * RBF(length_scale)`
- 特征标准化：`StandardScaler`
- UCB 公式：`pred_mean + alpha * pred_std`
- LLM prompt 复用 `CustomLLM_ACQ._build_batch_evaluation_prompt` 风格（材料学领域知识 + 观测数据 + 候选列表）
- LLM 调用通过 `llm_client.DeepSeekClient`（兼容 OpenAI 格式）
- 解析失败时 fallback 到 GP UCB 排序直接选取
- 去重：过滤已在 observed_configs 中的点

#### `benchmark/data_loader.py` — 数据加载

```
load_band_alignment_data(file_path, n_train=10, seed=42) -> dict
    # 返回 {train_x, train_y, test_x, test_y, feature_cols, target_col, df}

load_defects_doping_data(file_path, n_train=10, seed=42) -> dict
    # 返回 {train_x, train_y, test_x, test_y, feature_cols, target_col, df}

build_task_context(task_id, data, feature_cols) -> dict
    # 返回 {model, task, metric, tot_feats, ...,
    #        hyperparameter_constraints, df, target_col}
```

对齐 PVK-LLM benchmark 中的 `load_band_alignment_data()` / `load_defects_doping_data()`。

#### `benchmark/runner.py` — Benchmark 执行引擎

```
class BenchmarkRunner:
    """
    管理 benchmark 运行的完整生命周期。

    职责：
    - 数据加载 + train/test split
    - 初始化 PVKBO（使用 GPLLM_ACQ + LLM surrogate）
    - 运行 n_trials 轮 BO
    - 跟踪 train_score + generalization_score
    - 保存结果 CSV + JSON
    """

    __init__(self, task_id, n_initial, n_trials, seed, sm_mode, chat_engine, ...):
        ...

    run() -> BenchmarkResult:
        # 返回 {configs, fvals, best_config, best_score,
        #        cost_breakdown, time_breakdown, search_history}

def run_multi_seed(task_id, seeds, output_dir, **kwargs):
    # 多 seed 顺序执行，每个 seed 输出独立文件
    ...
```

**结果输出格式**（对齐 PVK-LLM benchmark）：

```
{output_dir}/
├── results_{sm_mode}/
│   └── {task_id}/
│       ├── {seed}.csv              # search_history (configs + fvals)
│       ├── {seed}_search_info.json # cost/time metadata
│       └── {seed}_summary.json     # best config, best score, convergence curve
```

#### `benchmark/cli.py` — CLI 入口

```
python -m benchmark.cli \
    --task band_alignment \
    --engine deepseek-v4-flash \
    --sm_mode discriminative \
    --n_trials 20 \
    --n_initial 5 \
    --seed 42 \
    --n_candidates 10 \
    --output_dir results
```

参数对齐 PVK-LLM benchmark 脚本的 argparse 定义。

#### 现有文件修改

**`pvk_llm_bo_runtime.py`：** `RealPvkBoRuntime` 增加 `acq_class` 参数，允许注入 `GPLLM_ACQ` 替代核心版 `LLM_ACQ`。

**`pvk_session_runtime.py`：** `OptimizationSessionRuntime` 增加 benchmark 模式的路由判断。

**`api.py`：** 新增 `POST /api/v1/benchmark` endpoint，接受 benchmark 参数，返回运行 ID，异步执行。

---

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    Benchmark Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Excel 数据 ──→ train/test split (n_train=10)                │
│                                                              │
│  ┌─ 初始化 ─────────────────────────────────────────┐       │
│  │  从 train 随机选 n_initial 个点 → 评估 → observed │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌─ BO Loop (× n_trials) ───────────────────────────┐       │
│  │                                                    │       │
│  │  1. GPLLM_ACQ.get_candidate_points():              │       │
│  │     GP 训练(observed) → UCB 预测(全量未观测)       │       │
│  │     → Top-20 → LLM 材料学推理 → n_candidates 候选  │       │
│  │                                                    │       │
│  │  2. LLM Surrogate.select_query_point():            │       │
│  │     LLM_DIS_SM / LLM_GEN_SM → EI 选最优候选        │       │
│  │                                                    │       │
│  │  3. 黑盒评估:                                      │       │
│  │     Excel 精确/最近邻查找 → score                  │       │
│  │     + test 集 generalization_score                 │       │
│  │                                                    │       │
│  │  4. 更新 observed_configs / observed_fvals         │       │
│  └────────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌─ 输出 ───────────────────────────────────────────┐       │
│  │  {seed}.csv          : 搜索历史 (configs+fvals)    │       │
│  │  {seed}_search_info.json : cost/time 明细          │       │
│  │  {seed}_summary.json : best config, convergence    │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| LLM 调用失败 | GP UCB 排序直接选取 top-n_candidates，不中断 benchmark |
| LLM 返回格式无法解析 | 同上 fallback |
| GP 训练失败（观测点太少） | 随机采样未观测点作为候选 |
| Excel 文件缺失 | 快速失败，给出明确路径提示 |
| DeepSeek API key 未配置 | 快速失败，提示配置 |
| 所有点已观测 | 从已观测中随机选 n_candidates |

---

## 测试策略

### 单元测试

| 测试文件 | 覆盖 |
|----------|------|
| `tests/test_gp_llm_acq.py` | GPLLM_ACQ 初始化、GP 训练、UCB 计算、去重过滤、LLM 调用 mock、fallback 路径 |
| `tests/test_benchmark_data_loader.py` | 数据加载、train/test split、task_context 构建 |
| `tests/test_benchmark_runner.py` | 单 seed 运行、多 seed 运行、结果保存格式 |

### 集成测试

- benchmark CLI 端到端运行（使用 mock LLM 响应）
- API endpoint 端到端（使用 mock LLM 响应）

### 对齐验证

- 用相同 seed/参数运行 band_alignment，对比 BOagent 输出与 PVK-LLM benchmark 输出的一致性

---

## 风险

| 风险 | 缓解 |
|------|------|
| GP+LLM prompt token 消耗大 | top_k 可配置，默认 20 |
| DeepSeek API 限流 | 复用现有 rate limiter，retry 逻辑 |
| PVK-LLM 核心库兼容性 | 不动核心库，GPLLM_ACQ 为独立模块 |
| benchmark 时间长（n_trials=20 × 5 seeds） | CLI 支持单 seed 快速验证模式 |

---

## 实现顺序

1. **`gp_llm_acq.py`** — GP+LLM ACQ 核心实现
2. **`benchmark/data_loader.py`** — 数据加载 + train/test split
3. **`benchmark/runner.py`** — Benchmark 执行引擎
4. **`benchmark/cli.py`** — CLI 入口
5. **修改 `pvk_llm_bo_runtime.py`** — 支持 ACQ 注入
6. **修改 `api.py`** — Benchmark API endpoint
7. **测试** — 单元测试 + 集成测试 + 对齐验证
