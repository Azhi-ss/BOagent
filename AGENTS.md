# BOagent AI 开发者指南

本文档是 BOagent 项目的主要上下文指引。在开始任何实现、重构或测试任务前，请先阅读本文档。

---

## 1. 项目概述与技术栈

BOagent 是一个由 LLM 驱动的贝叶斯优化（Bayesian Optimization, BO）工作流协调器，面向通用科学材料配方优化（如电池正极、化学反应产率等）。系统通过高上下文的领域推理来增强传统的高斯过程（Gaussian Process, GP）统计建模能力。

### 核心技术栈

| 层次 | 组件 |
|---|---|
| **算法核心** | Python 3.11+, BoTorch/GPyTorch（主GP后端）, scikit-learn（兼容后端）, NumPy, Pandas, SciPy, pytest |
| **AI/LLM** | DeepSeek API（`deepseek-v4-flash` / `deepseek-v4-pro`），豆包 Embedding API（火山引擎/Ark），默认模型 `doubao-embedding-vision-250615`（通过 `DOUBAO_EMBEDDING_MODEL` 配置） |
| **数据集** | 本地 `datasets/` 目录下的统一 schema（如 `datasets/battery`），无需 `.env` 额外配置 |

---

## 2. 代码结构与职责

```
BOagent/
├── packages/
│   └── bo-core/                   # 算法核心包（pip install -e 安装）
│       ├── bo_core/
│       │   ├── optimization/      # 贝叶斯优化引擎
│       │   │   ├── optimizer.py   # BayesianOptimizer 主协调器，GP 训练与采集函数评分
│       │   │   ├── knowledge.py   # KnowledgeEngine，领域规则动态提示构建
│       │   │   ├── memory.py      # VectorMemory，豆包 Embedding 与 numpy 向量检索
│       │   │   └── space.py       # SearchSpace 定义（连续型与离散型）
│       │   ├── benchmark/         # 性能评估引擎
│       │   │   ├── data_loader.py # 数据集加载与确定性种子分割
│       │   │   ├── runner.py      # 单种子基准测试协调器
│       │   │   ├── comparison.py  # 多种子并行基准对比
│       │   │   └── bo_step.py     # 分步基准测试，桥接各 BO 组件
│       │   ├── llm_client.py      # 统一的 DeepSeek API 客户端封装
│       │   └── pvk_llm_compat.py  # 历史遗留兼容补丁（勿重构，见第5节约束）
│       ├── tests/                 # 核心算法自动化测试
│       ├── benchmark_agent_team.py # 多智能体基准协调器
│       ├── run_prompt_ablation.py # A/B/C 提示词变体消融实验
│       └── pyproject.toml         # bo-core 包定义与依赖
├── scripts/                       # 编排与评估脚本
└── AGENTS.md                      # 本文档
```

---

## 3. 核心领域知识规则

优化器并非纯粹的统计黑箱，在候选点推荐阶段会注入显式的物理/化学领域启发式规则。

### 领域知识动态集成

系统根据当前任务的数据集（如电池正极、化学反应产率），将任务特征列映射到对应的领域约束规则，并动态注入 LLM 上下文。
- `knowledge.py` 中的 `build_prompt` 负责此映射，**新增参数时修改此处，不要另起炉灶**。

### 混合 LLM-GP 候选筛选流程

1. GP 代理模型从搜索空间中按采集函数分值产出 Top-K 候选池（默认 K=20）。
2. 对每个候选点发出「Yes/No」可行性提示词，提取「Yes」的对数概率。
3. **混合评分**: `GP_Score + (γ × std(GP_Scores)) × log_prob(Yes)`，其中 γ（默认 0.1）控制 LLM 影响权重。
4. LLM 也可生成 Python `score_candidate(c: dict) -> float` 启发式函数，在沙箱 `exec()` 中运行，用于对超大候选池（10k+）进行 GP 评分前的预筛选。

---

## 4. 开发工作流与常用命令

### 算法核心测试（bo-core）

> 我们对算法逻辑强制执行 TDD。新增逻辑**必须**使用 `--cov` 覆盖率检查。
> 覆盖率要求：整体 ≥ 80%，`bo_core/optimization/` 模块 ≥ 90%。

```bash
cd packages/bo-core
uv run pytest --cov=bo_core --cov-report=term-missing
```

### 代码质量检查

```bash
uv run ruff check .                       # Lint
uv run mypy packages/bo-core/bo_core      # 静态类型检查
```

### 其他检验工具

- **数据验证（pandera）**：搜索空间与数据集 DataFrame 的 schema 校验。
- **属性测试（hypothesis）**：GP 操作的边界与矩阵稳定性验证。
- **ML 漂移检测（evidently）**：代理模型数据漂移与质量自动检查。

---

## 5. 架构红线与约束

> [!WARNING]
> **禁止直接用 LLM 评分**：严禁让 LLM 对原始搜索空间中的候选点直接打分或筛选。搜索空间必须先经 GP 代理模型过滤为 Top-K 候选池（通常 top 20），LLM 的职责仅是在此基础上结合领域物理推理做进一步精化。违反此规则将导致 token 预算爆炸，并破坏收敛保证。

> [!IMPORTANT]
> **禁止重构 `pvk_llm_compat.py`**：该文件包含关键的历史兼容补丁，修复了 pandas、langchain、OpenAI schema 的接口变化。未经充分测试，擅自修改会导致核心集成崩溃。

> [!CAUTION]
> **本地 RNG 隔离**：在数据加载器和优化循环中，始终使用本地种子化的 `np.random.RandomState` 实例。禁止使用全局 `np.random`，以防止并行基准测试线程间的随机状态污染。

> [!CAUTION]
> **敏感文件保护**：未经用户明确确认，不得随意修改 `**/.env*` 或 `**/*_results.json` 文件。

---

## 6. 现有能力与复用指引

动手写新工具函数前，先确认这些是否已经存在：

- **领域知识提示构建器**：`packages/bo-core/bo_core/optimization/knowledge.py` — `build_prompt` 已将任务特征列映射到领域公式与提示。新增参数时，在此处添加映射，勿复制构建逻辑。
- **洞察持久化与 RAG**：`packages/bo-core/bo_core/optimization/memory.py` — 负责将实验洞察写入 `insights.jsonl` 并通过豆包 API 计算向量。embedding 客户端不可用时，自动退回到按时间倒序的 `top_k` 条目，不抛出异常。

---

## 7. 完成任务前的验证清单

- [ ] **算法测试**：在 `packages/bo-core` 目录下运行 `uv run pytest`，确认所有核心测试通过。
- [ ] **安全边界**：确认 `output_dir` 防止路径穿越（`..` 和绝对路径校验）。API 密钥不得硬编码，始终从环境变量加载。
- [ ] **记忆回退验证**：确认 VectorMemory 在 `ARK_API_KEY` 缺失或不可达时，能安全退回到基于时间序的检索，不抛出异常。
- [ ] **RNG 隔离**：验证优化与基准测试路径中无全局 `np.random` 调用，所有随机性均通过本地种子化的 `RandomState` 产生。

---

## 8. 参考文档（`docs/`）

`docs/` 目录是外部 API 合约、库使用规范与防御性编程规则的唯一权威来源。

实现功能或修改核心模块前，**必须**查阅 `docs/` 下对应的规范文档。

> [!NOTE]
> **扩展协议**：获取或添加新的 API 规范、领域研究结果时，将标准化的 Markdown 文件保存在 `docs/<category>/` 下，并在类目索引（`docs/<category>/README.md`）中注册。在编写代码前，在任务规划文档中引用这些参考。
