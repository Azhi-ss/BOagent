# references/ — BO 框架仓库索引

分类时间: 2026-08-01

---

## 零、2025~2026 最新 SOTA 路线 B 框架（已新增下载）

| 仓库/目录 | 论文 / 来源 | 核心机制 | 对 BOagent 的参考价值 |
|---|---|---|---|
| **ChemBOMAS** ⭐ | [arXiv:2509.08736](https://arxiv.org/abs/2509.08736) (2025) | **Subspace Decomposition**：RAG + 多 Agent 协作划定化学反应子空间 | **P0**，路线 B 的直接标杆！LLM 负责粗筛子空间，GP 负责细筛选点 |
| **PiEvo** ⭐ | [arXiv:2602.06448](https://arxiv.org/abs/2602.06448) (ICML 2026) | **Principle-Evolvable**：基于不确定性最小化的原则演化与动态搜索空间扩展 | **P0**，指导 LLM 如何根据实验反馈动态修正机制假设 |
| **CALIPER** ⭐ | [arXiv:2606.01730](https://arxiv.org/abs/2606.01730) (2026) | **Evidence-Gated LLM Priors**：利用证据门控与残差先验进行候选池过滤 | **P1**，提供基于 LLM 先验的高鲁棒性多目标 BO 候选池过滤逻辑 |

---

## 一、经典 BO 框架（5 个）

### 定位金字塔

| 层级 | 仓库 | 定位 |
|---|---|---|
| 工业平台 | **Ax** | Meta 全流程 BO 平台，SearchSpace→Experiment→Scheduler 抽象 |
| 核心工厂 | **BoTorch** | PyTorch 上的 GP/采集函数工厂，Ax 和 BOagent 都构建于此 |
| 专业路线 | **Dragonfly** | 多保真 BO（便宜近似→精确优化） |
| 非 GP 路线 | **SMAC3** | SMBO 框架，默认随机森林代理模型 |
| 最小教学 | **BayesianOptimization** | 纯 Python ~200 行核心，GP+EI 最小环 |

| 仓库 | 代理模型 | 采集函数 | 多保真 | 后端 | 对 BOagent 的参考价值 |
|---|---|---|---|---|---|
| **BoTorch** | GP (GPyTorch) | EI/UCB/TS/NEI/qNEI | ✅ | PyTorch 原生 | **正在用**，定制采集函数和 warm-start |
| **Ax** | GP (BoTorch) | 自动选择 | ✅ | BoTorch | 参考 SearchSpace 抽象和 Scheduler 调度 |
| **Dragonfly** | GP + 深度 GP | EI/UCB/TS | ✅ **核心特性** | GPy (TF 1.x) | 40 步迭代里的快筛→精化策略 |
| **SMAC3** | 随机森林 / GP | EI / 自定义 | ✅ Hyperband | sklearn/GPy | 对比 GP vs RF 在化学数据上的表现 |
| **BayesianOptimization** | GP (sklearn) | EI/UCB/PI | ❌ | sklearn | 入门学习 GP+EI 的完整环 |

---

## 二、化学/SDL 专项（6 个）

| 仓库 | 核心特点 | 对 BOagent 的参考价值 |
|---|---|---|
| **currybo** ⭐ | 跨底物通用条件优化，BoTorch 扩展，自带 5 个化学数据集 + Benchmark | **P0**，直接处理 Buchwald/Suzuki/Heck 数据集 |
| **atlas** ⭐ | 自动驾驶实验室(SDL)“大脑”，支持混合参数、多目标、带约束优化 | **P1**，复杂实验流程调度的 SOTA 标杆 |
| **gryffin** ⭐ | 全类别(Categorical)化学变量的连续松弛 KDE 优化算法 | **P1**，纯离散变量 BO 的顶级数学解法 |
| **olympus** | 化学优化策略 Benchmark 竞技场，提供标准化数据集接口 | P2，后续可能用来评测 Chem-LGBO |
| **phoenics** | 连续空间的极速核回归贝叶斯优化器 | P3，连续空间极速优化参考 |
| **cake** | 另一个化学 BO 框架 | 看一眼就行 |

---

## 三、LLM 驱动的 BO（5 个）— 与 KnowledgeEngine 重叠

| 仓库 | 核心思路 | 对 BOagent 的参考价值 |
|---|---|---|
| **LLM4BO** | 用 LLM 替代 GP 做条件建议 | **P1**，KnowledgeEngine 的核心参考 |
| **LLAMBO** | LLM 做 BO 的提议模型 | **P1**，同上 |
| **Text-to-BatteryRecipe** | LLM → 电池配方优化 | P4，领域特定 |
| **PVK-LLM** | LLM 用于钙钛矿发现 | P4，领域特定 |
| **llm-closed-loop-experiments** | LLM 闭环实验设计 | P2，流程参考 |

---

## 四、LLM + BO 混合方案（3 个）

| 仓库 | 核心思路 |
|---|---|
| **LLM-in-the-Loop-BO** | LLM 作为 BO 采集函数的热启动或提议生成 |
| **lmabo** | 语言模型辅助的 BO |
| **Reasoning-BO** | 语言推理增强 BO 决策 |

---

## 五、自动科研 / 泛 AI 科研（4 个）

| 仓库 | 核心思路 |
|---|---|
| **AI-Scientist** | Sakana AI，全自动假设→实验→论文循环 |
| **autoresearch** | karpathy 的自动研究框架 |
| **LABO** | 轻量 BO 变体 |
| **deep_kernel_learning** | 早期实验（你的旧工作副本） |

---

## 六、BOagent 副本 / 旧代号（3 个）

| 仓库 | 说明 |
|---|---|
| **ALAS** | BOagent fork 以 ALAS 名义发布 |
| **multi_agent_acquisition_2603.28959** | 命名留下的代理名 |
| **kernel_manifold** | 早期代号 |