# references/ — BO 相关仓库索引

分类时间：2026-08-03

本索引采用“**物理主分类 + 四组标签**”的二维分类法。仓库已迁移至六个一级分类目录；主目录用于快速定位，标签用于表达跨类别特征。

## 分类规则

### 主类别

| 主类别 | 物理目录 | 收录标准 |
|---|---|---|
| **BO 基础框架** | `01-bo-frameworks/` | 提供通用优化循环、实验管理、代理模型或采集函数基础设施 |
| **BO 算法与代理模型** | `02-bo-algorithms/` | 聚焦核函数、代理模型或特定搜索空间算法，而非完整科研工作流 |
| **LLM + BO** | `03-llm-bo/` | LLM 直接参与候选生成、先验建模、采集策略、核选择或推理增强 |
| **化学/材料与 SDL** | `04-chemistry-materials-sdl/` | 主要服务化学、材料、电池或自驱动实验室（SDL）任务 |
| **自主科研与闭环实验** | `05-autonomous-science/` | 覆盖假设、实验、反馈或论文生成等更完整的科研闭环 |
| **BOagent 历史仓库** | `06-boagent-history/` | 仅收录已确认的 BOagent fork、旧版本或旧代号；当前为空 |

主类别按仓库的**主要参考用途**确定。例如，`ChemBOMAS` 同时涉及多 Agent 和 BO，但主要用于化学实验工作流，因此归入“化学/材料与 SDL”。

### 标签

| 标签组 | 取值示例 | 含义 |
|---|---|---|
| `domain` | `general`、`chemistry`、`materials`、`battery`、`protein` | 主要应用领域 |
| `mechanism` | `GP`、`multi-fidelity`、`LLM-prior`、`multi-agent`、`kernel-learning` | 核心技术机制，可多选 |
| `priority` | `P0`–`P4` | 对 BOagent 当前工作的参考优先级 |
| `relation` | `external`、`fork`、`history` | 与 BOagent 的关系 |

优先级含义：`P0` 直接依赖或当前路线标杆；`P1` 高价值设计参考；`P2` Benchmark 或重要补充；`P3` 辅助/经典参考；`P4` 窄领域案例。

## 一、BO 基础框架（6）

| 仓库路径 | 核心定位 | domain | mechanism | priority | relation |
|---|---|---|---|---|---|
| [`botorch`](01-bo-frameworks/botorch/) | PyTorch 原生 BO 组件库，提供 GP、采集函数和批量优化 | general | GP, acquisition, MOBO | **P0** | external |
| [`Ax`](01-bo-frameworks/Ax/) | Meta 全流程实验与优化平台，封装 SearchSpace、Experiment、Scheduler | general | orchestration, BoTorch, multi-fidelity | **P1** | external |
| [`baybe`](01-bo-frameworks/baybe/) | 默克集团开源的面向离散/混合化学实验与分子特征编码的贝叶斯优化框架 | chemistry | GP, BoTorch, RDKit, constraints | **P1** | external |
| [`BayesianOptimization`](01-bo-frameworks/BayesianOptimization/) | sklearn GP + EI/UCB/PI 的轻量 BO 实现 | general | GP, minimal-loop | P3 | external |
| [`dragonfly`](01-bo-frameworks/dragonfly/) | 连续、离散及多保真黑箱优化框架 | general | GP, multi-fidelity | P2 | external |
| [`SMAC3`](01-bo-frameworks/SMAC3/) | 以随机森林为主的 SMBO/HPO 框架 | general | random-forest, Hyperband, HPO | P2 | external |

## 二、BO 算法与代理模型（5）

| 仓库路径 | 核心定位 | domain | mechanism | priority | relation |
|---|---|---|---|---|---|
| [`ALAS`](02-bo-algorithms/ALAS/) | 可学习 α-stable 谱核，适配平滑与尖锐目标景观 | general | GP, kernel-learning | P2 | external |
| [`deep_kernel_learning`](02-bo-algorithms/deep_kernel_learning/) | 神经网络特征映射与 GP 核结合的经典 DKL | general | deep-kernel, GP | P3 | external |
| [`kernel_manifold`](02-bo-algorithms/kernel_manifold/) | 将组合核映射到连续流形，并在核架构空间中执行 BO | general | kernel-search, manifold, GP | P2 | external |
| [`gryffin`](02-bo-algorithms/gryffin/) | 面向类别变量的连续松弛与核密度优化方法 | chemistry | categorical, KDE | **P1** | external |
| [`phoenics`](02-bo-algorithms/phoenics/) | 面向连续空间的核回归式快速优化器 | chemistry | kernel-regression, continuous | P3 | external |

## 三、LLM + BO（10）

| 仓库路径 | 核心定位 | domain | mechanism | priority | relation |
|---|---|---|---|---|---|
| [`CALIPER`](03-llm-bo/CALIPER/) | 证据门控的目标级 LLM 残差先验层，用于离散多目标 BO | molecules | LLM-prior, evidence-gating, MOBO | **P1** | external |
| [`LABO`](03-llm-bo/LABO/) | 将 LLM 作为低保真 oracle，以残差 GP 决定真实实验投入 | scientific | LLM-oracle, multi-fidelity, residual-GP | **P1** | external |
| [`LLAMBO`](03-llm-bo/LLAMBO/) | LLM 执行 warm start、候选采样和生成式代理建模 | general | LLM-surrogate, warm-start, proposal | **P1** | external |
| [`LLM-in-the-Loop-BO`](03-llm-bo/LLM-in-the-Loop-BO/) | LLM 参与黑箱优化、超参调优和 3D 打印实验建议 | general | LLM-proposal, warm-start | P2 | external |
| [`LLM4BO`](03-llm-bo/LLM4BO/) | 在蛋白质和分子发现任务中系统评测 LLM 优化/选择方法 | protein, molecules | benchmark, LLM-selector, active-learning | P2 | external |
| [`lmabo`](03-llm-bo/lmabo/) | LLM 根据预算、代理模型和历史状态在线选择采集函数 | general | acquisition-selection, LLM-policy | **P1** | external |
| [`Reasoning-BO`](03-llm-bo/Reasoning-BO/) | 使用知识增强多 Agent 推理辅助 Ax 优化循环 | chemistry, general | reasoning, multi-agent, Ax | **P1** | external |
| [`cake`](03-llm-bo/cake/) | 使用 LLM 与进化算法自动演化 GP 核表达式 | general | LLM, kernel-evolution, GP | **P1** | external |
| [`multi_agent_acquisition_2603.28959`](03-llm-bo/multi_agent_acquisition_2603.28959/) | 分离策略 Agent 与候选生成 Agent，显式控制探索/利用 | general | multi-agent, acquisition-policy, candidate-generation | P2 | external |
| [`Unleashing LLMs in Bayesian Optimization`](03-llm-bo/Unleashing%20LLMs%20in%20Bayesian%20Optimization/) | LGBO 通过区域偏好和 GP 均值移位注入 LLM 指导 | scientific | LLM-prior, region-preference, mean-shift | **P0** | external |

## 四、化学/材料与 SDL（6）

| 仓库路径 | 核心定位 | domain | mechanism | priority | relation |
|---|---|---|---|---|---|
| [`ChemBOMAS`](04-chemistry-materials-sdl/ChemBOMAS/) | 基于 Google ADK 与 BayBE 的化学实验多 Agent 闭环系统 | chemistry | multi-agent, BayBE, constraints, closed-loop | **P0** | external |
| [`currybo`](04-chemistry-materials-sdl/currybo/) | 跨底物反应条件优化，包含化学数据集与 Benchmark | chemistry | BoTorch, transfer, benchmark | **P0** | external |
| [`atlas`](04-chemistry-materials-sdl/atlas/) | 面向自驱动实验室的混合变量、多目标、约束优化“大脑” | chemistry, SDL | mixed-space, constraints, MOBO | **P1** | external |
| [`olympus`](04-chemistry-materials-sdl/olympus/) | 化学优化策略与标准化数据集 Benchmark 平台 | chemistry | benchmark, datasets | P2 | external |
| [`PVK-LLM`](04-chemistry-materials-sdl/PVK-LLM/) | 面向钙钛矿能带、缺陷和掺杂设计的 LLM/BO 应用 | perovskite, materials | LLM, BO, domain-model | P4 | external |
| [`Text-to-BatteryRecipe`](04-chemistry-materials-sdl/Text-to-BatteryRecipe/) | 从文本与材料知识生成并优化电池配方 | battery, materials | LLM, recipe-generation, BO | P4 | external |

## 五、自主科研与闭环实验（4）

| 仓库路径 | 核心定位 | domain | mechanism | priority | relation |
|---|---|---|---|---|---|
| [`PiEvo`](05-autonomous-science/PiEvo/) | 多 Agent 在实验反馈下演化科学原则和假设空间 | scientific | multi-agent, principle-evolution, uncertainty | **P0** | external |
| [`llm-closed-loop-experiments`](05-autonomous-science/llm-closed-loop-experiments/) | BORA 的 LLM-only、BO-only 与混合闭环实验评测材料 | chemistry, physics | closed-loop, LLM-BO, benchmark | P2 | external |
| [`AI-Scientist`](05-autonomous-science/AI-Scientist/) | 从研究想法、实验迭代到论文生成的自动科研系统 | AI-research | multi-agent, experiment-loop, paper-generation | P2 | external |
| [`autoresearch`](05-autonomous-science/autoresearch/) | 面向模型训练实验的极简自主研究循环 | AI-research | experiment-loop, code-modification | P2 | external |

## 六、BOagent 历史仓库（0）

`06-boagent-history/` 已创建，当前没有足够证据将任何项目迁入该目录。

特别说明：`ALAS`、`kernel_manifold` 和 `multi_agent_acquisition_2603.28959` 分别是独立的核方法或论文材料，不应归为 BOagent 历史仓库。以后只有确认来源关系后，才使用 `relation: fork` 或 `relation: history`。

## 快速选读路径

| 目标 | 建议顺序 |
|---|---|
| 理解 BOagent 当前数值优化底座 | [`botorch`](01-bo-frameworks/botorch/) → [`Ax`](01-bo-frameworks/Ax/) → [`BayesianOptimization`](01-bo-frameworks/BayesianOptimization/) |
| 研究 LLM 如何安全注入 BO | [`LGBO`](03-llm-bo/Unleashing%20LLMs%20in%20Bayesian%20Optimization/) → [`CALIPER`](03-llm-bo/CALIPER/) → [`LABO`](03-llm-bo/LABO/) |
| 研究 LLM 如何调整 BO 策略 | [`lmabo`](03-llm-bo/lmabo/) → [`Reasoning-BO`](03-llm-bo/Reasoning-BO/) → [`multi-agent acquisition`](03-llm-bo/multi_agent_acquisition_2603.28959/) |
| 研究化学实验优化工作流 | [`currybo`](04-chemistry-materials-sdl/currybo/) → [`atlas`](04-chemistry-materials-sdl/atlas/) → [`ChemBOMAS`](04-chemistry-materials-sdl/ChemBOMAS/) → [`olympus`](04-chemistry-materials-sdl/olympus/) |
| 研究可演化的自主科研闭环 | [`PiEvo`](05-autonomous-science/PiEvo/) → [`closed-loop experiments`](05-autonomous-science/llm-closed-loop-experiments/) → [`AI-Scientist`](05-autonomous-science/AI-Scientist/) |

## 完整性

- 已物理迁移仓库：**30**
- 一级分类目录：**6**（其中 `06-boagent-history/` 当前为空）
- 每个仓库只存在于一个主分类目录，并保留跨领域标签
- 其中 **24** 个仓库保留各自的嵌套 `.git` 元数据，移动目录不会改变其独立仓库身份