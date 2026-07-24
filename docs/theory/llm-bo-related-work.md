# LLM 增强贝叶斯优化：相关工作综述

## 总览

本综述围绕 BOagent 的核心设计——GP + LLM 混合采集函数、logprob 物理可行性打分、LLM 生成启发式代码、物理先验注入、动态科学记忆 (DSM)——调研了六个方向的现有工作。整体发现：LLM 与 BO 的融合是 2024–2025 年快速发展的交叉领域，已有多篇论文探索 LLM 作为采集函数、代理模型、先验注入器和知识管理器等角色；但将 logprob 作为连续采集信号、并与 GP 分数自适应融合的方案，在现有文献中尚无精确匹配，BOagent 的设计具有独特性。

---

## 1. Reasoning-BO（BOagent 代码直接引用）

### 论文信息

- **名称**: Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs
- **来源**: [arxiv 2505.12833](https://arxiv.org/abs/2505.12833)
- **代码**: [github.com/little1d/Reasoning-BO](https://github.com/little1d/Reasoning-BO)
- **作者**: Zhuo Yang, Daolang Wang, Lingli Ge, Beilun Wang, Tianfan Fu, Yuqiang Li

### 核心方法

Reasoning-BO 将 LLM 的推理能力嵌入 BO 循环，核心设计包含三部分：

1. **多智能体知识提取管线**：每轮 BO 迭代后，LLM 生成思维链 (CoT) 推理轨迹，经由三个智能体依次处理——Verifier Agent 提取变量关系和领域知识，Formatter Agent 将推理轨迹解析为 `<entity, relation, entity>` 三元组，Notes Agent 将结构化知识增量写入双通道存储系统（向量数据库 + 知识图谱）。([arxiv 2505.12833](https://arxiv.org/abs/2505.12833))

2. **双通道知识管理**：知识图谱维护概念间的可解释网络关系（三元组），向量数据库（使用 [Milvus](https://milvus.io/)）提供语义相似度检索。检索时先遍历知识图谱获取结构化路径，再查询向量库补充上下文。([arxiv 2505.12833](https://arxiv.org/abs/2505.12833); [github.com/little1d/Reasoning-BO](https://github.com/little1d/Reasoning-BO))

3. **LLM 作为候选过滤器（非生成器）**：BO 采集函数 (qLogEI) 每轮提出 5 个候选点，LLM 分析每个候选、生成假设并赋置信度分数，取 Top-3 执行实验。论文明确指出"避免 LLM 直接生成候选点以防测试数据污染"。([arxiv 2505.12833](https://arxiv.org/abs/2505.12833))

### 算法流程

- **初始化阶段**：用户提供自然语言"实验指南针"(Experiment Compass)，LLM 生成实验概览，提取关键实体存入向量库和知识图谱，选择高置信初始候选。
- **迭代优化**：BO 提议 5 候选 → 检索历史知识 → LLM 分析并打分 → 取 Top-3 → 执行实验 → 提取结构化笔记存入双通道存储。
- **结论阶段**：LLM 汇总所有结果生成最终报告。

### 与 BOagent DSM 的异同

| 维度 | Reasoning-BO | BOagent DSM |
|------|-------------|-------------|
| 知识结构 | 三元组 `<entity, relation, entity>` + 知识图谱 | 结构化 JSON (notes / key_findings / parameter_relationships / optimization_principles) |
| 向量存储 | Milvus（外部向量数据库） | 纯内存 numpy 存储，可选 JSONL 持久化 |
| 嵌入服务 | 论文未明确（代码基于 Camel 框架） | 豆包 embedding API (OpenAI 兼容) |
| LLM 角色 | 候选过滤 + 假设生成 + 置信度打分 | logprob(Yes) 物理可行性判断 + GP 分数融合 |
| 采集函数 | qLogEI 提议候选，LLM 过滤 | Matérn 核 GP 打分 + LLM logprob 混合融合 |
| 复杂度 | 多智能体 + 知识图谱 + 向量库（重） | 轻量级单进程（轻） |

**关键差异**：BOagent 的 `memory.py` 代码注释明确写道 "Inspired by Reasoning-BO's NotesAgent + MilvusAgent pattern, but lightweight"——即 BOagent 借鉴了 NotesAgent（结构化笔记提取）和 MilvusAgent（向量检索）的模式，但用纯内存 numpy + 豆包 embedding 替代了 Milvus 外部依赖，且未使用知识图谱。BOagent 的 Insight 数据结构（notes/key_findings/parameter_relationships/optimization_principles）比 Reasoning-BO 的三元组更面向领域语义。([BOagent `packages/bo-core/bo_core/optimization/memory.py`](../../packages/bo-core/bo_core/optimization/memory.py); [arxiv 2505.12833](https://arxiv.org/abs/2505.12833))

**关键差异（采集侧）**：Reasoning-BO 的 LLM 做"候选过滤"（5 选 3），是离散决策；BOagent 的 LLM 做"logprob 连续打分"，输出 `logprob(Yes)` 作为连续信号与 GP 分数融合（$Score_{hybrid} = GP\_Score + \gamma \sigma_{GP} \cdot logP(Yes)$），粒度更细。

### 基准结果

- Direct Arylation 任务：Reasoning-BO 产率 60.7% vs 传统 BO 25.2%（约 2.4 倍提升）。
- Buchwald-Hartwig 任务：95.06% vs 84.68%。
- 初始性能优势：Direct Arylation 初始性能高 44.6%。([arxiv 2505.12833](https://arxiv.org/abs/2505.12833))

---

## 2. LLM 作为 BO 的采集函数 / 代理模型 / 精炼器 / 初始化器

本方向已有多篇论文，LLM 在 BO 中扮演的角色可分为四类：替代 GP、精炼 GP 结果、提供先验、加速探索。

### 2.1 LLINBO（LLM-in-the-Loop BO）

- **来源**: [arxiv 2505.14756](https://arxiv.org/abs/2505.14756)
- **代码**: [github.com/UMDataScienceLab/LLM-in-the-Loop-BO](https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO)
- **核心方法**: 混合框架，早期探索阶段由 LLM 驱动（利用上下文推理能力应对冷启动），后期开发阶段切换到 GP（提供校准的不确定性估计）。提出三种协作机制并有理论保证。在 3D 打印场景验证。
- **与 BOagent 的关系**: 两者都是 LLM + GP 混合，但 LLINBO 是"阶段切换"（早期 LLM → 后期 GP），BOagent 是"同时融合"（每轮 GP 分数 + LLM logprob 加权）。BOagent 的融合粒度更细。

### 2.2 LGBO（LLM-Guided BO）

- **来源**: [arxiv 2605.17976](https://arxiv.org/abs/2605.17976)
- **核心方法**: 首个将 LLM 偏好持续嵌入每轮 BO 迭代的框架。通过"区域提升偏好机制"(region-lifted preference) 移动代理模型均值，稳定可控。理论证明最坏情况下不劣于标准 BO，偏好与目标对齐时显著加速收敛。在 Fe-Cr 电池电解液湿实验中，6 轮即达最佳值的 90%。
- **与 BOagent 的关系**: LGBO 修改 GP 均值（prior 移动），BOagent 修改采集分数（后验加性融合）。LGBO 是"修改代理模型"，BOagent 是"修改采集函数"。

### 2.3 BORA（Language-Based BO Research Assistant）

- **来源**: [arxiv 2501.16224](https://arxiv.org/abs/2501.16224)（IJCAI-25）
- **核心方法**: LLM 作为"上下文顾问"，周期性提议有前景的搜索区域，BO 引擎负责统计建模和采集函数优化。LLM 不替代代理模型，而是注入领域推理引导搜索方向。同时提供实时优化进度评论。
- **与 BOagent 的关系**: BORA 的 LLM 是"区域级顾问"（提议区域），BOagent 的 LLM 是"候选级裁判"（逐点判断物理可行性）。粒度不同。

### 2.4 LABO（LLM-Accelerated BO）

- **来源**: [arxiv 2605.22054](https://arxiv.org/abs/2605.22054)（ICML 2026）
- **核心方法**: 将 LLM 预测与真实实验观察纳入同一 BO 循环。核心创新是门控机制 (gating criterion)，动态决定何时依赖廉价 LLM 预测（广泛探索）、何时执行昂贵真实实验（高不确定区域）。有累积遗憾界理论分析。
- **与 BOagent 的关系**: LABO 的 LLM 是"廉价代理评估器"（替代部分真实实验），BOagent 的 LLM 是"物理可行性过滤器"（不替代实验，而是引导采集）。LABO 解决"实验成本"问题，BOagent 解决"物理约束"问题。

### 2.5 CAKE（Context-Aware Kernel Evolution）

- **来源**: [arxiv 2509.17998](https://arxiv.org/abs/2509.17998)（NeurIPS 2025 poster）
- **核心方法**: LLM 作为 GP 核函数的进化算子（交叉/变异），根据观测数据动态生成和精炼核函数。配合 BAKER 机制（BIC 拟合 + 期望改善）选择最优核。在超参优化、控制器调参、光子芯片设计上超越基线。
- **与 BOagent 的关系**: CAKE 用 LLM 改进 GP 的核函数（代理模型结构），BOagent 用 LLM 改进 GP 的采集分数（采集函数输出）。两者作用在 BO 流程的不同环节。

### 2.6 Evidence-Gated LLM Priors for MOBO

- **来源**: [arxiv 2606.01730](https://arxiv.org/abs/2606.01730)
- **核心方法**: 将 LLM 生成的先验视为"可证伪的先验来源"，通过目标级别的声誉市场机制 (reputation-market) 在线更新专家权重。引入解耦反事实门 (decoupled counterfactual gate)，可选择信任/不信任/忽略 LLM 先验。在分子优化基准上验证。
- **与 BOagent 的关系**: 该工作强调"不要盲信 LLM 置信度"，发现原始 LLM 置信度在不同任务上表现不一致。BOagent 的 `λ = γ·std(GP_scores)` 自适应机制也有类似的"不盲信"思路——用 GP 分数方差缩放 LLM 信号权重。

### 2.7 A Sober Look at LLMs for BO Over Molecules

- **来源**: [arxiv 2402.05015](https://arxiv.org/abs/2402.05015)（ICML 2024）
- **代码**: [github.com/wiseodd/lapeft-bayesopt](https://github.com/wiseodd/lapeft-bayesopt)
- **核心方法**: 系统评估 LLM 是否真能加速分子空间上的原则性 BO。两种路径：(1) LLM 作为固定特征提取器，嵌入输入标准 BO 代理模型；(2) 参数高效微调 + 贝叶斯神经网络获得后验分布。关键发现：通用 LLM 不足以改进 BO，必须经过领域预训练或微调。
- **与 BOagent 的关系**: 该论文的结论支持 BOagent 的设计选择——BOagent 在 prompt 中注入半导体物理先验（CBO/VBO 能带对齐），而非依赖 LLM 的通用知识。领域先验注入是必要的。

### 2.8 LLMs as Uncertainty-Calibrated Optimizers

- **来源**: [arxiv 2504.06265](https://arxiv.org/abs/2504.06265)
- **核心方法**: 用 BO 的不确定性感知目标（EI、UCB）训练 LLM，将 LLM 的过度自信转化为精确校准机制。在 Buchwald-Hartwig 反应上，从 10 个失败条件出发，50 轮内高产率条件发现率从 24% 提升至 43%。跨 19 个优化问题平均排名第一。
- **与 BOagent 的关系**: 该工作将不确定性校准"训练进"LLM，BOagent 则用 GP 提供不确定性、LLM 提供物理直觉，两者分离但融合。BOagent 不需要训练 LLM，降低了工程成本。

### 2.9 PEBOL

- **来源**: [arxiv 2405.00981](https://arxiv.org/abs/2405.00981)
- **核心方法**: 将自然语言偏好 elicitation 建模为 BO 问题。LLM 用于建模自然语言反馈的似然函数，配合 Thompson Sampling / UCB 采集函数引导查询生成。
- **与 BOagent 的关系**: PEBOL 的 LLM 在"偏好空间"工作，BOagent 的 LLM 在"物理可行性空间"工作。都是 LLM + BO 采集函数结合，但应用域不同。

---

## 3. LLM logprobs 用于打分与重排序

### 3.1 背景与关键工作

BOagent 使用 `logprob("Yes")` 作为连续采集信号，这与 LLM-as-judge 和 LLM reranker 领域的 logprob 打分有共通之处。

#### LLM-as-a-Judge

- **来源**: [arxiv 2306.05685](https://arxiv.org/abs/2306.05685)（NeurIPS 2023）
- **核心方法**: 用强 LLM (GPT-4) 作为评估器对开放式问题打分。分析了位置偏差 (position bias)、冗长偏差 (verbosity bias)、自我增强偏差 (self-enhancement bias)。GPT-4 与人类偏好的一致率超过 80%。
- **与 BOagent 的关系**: LLM-as-judge 通常输出离散评分或偏好排序，BOagent 提取底层 logprob 作为连续信号，绕过了 LLM 的"讨好倾向"和"幻觉评分"问题。([BOagent `docs/theory/logp-metric.md`](./logp-metric.md))

#### Pairwise Ranking Prompting (PRP)

- **来源**: [arxiv 2306.17563](https://arxiv.org/abs/2306.17563)（NAACL 2024）
- **核心方法**: 将文档排名简化为两两比较，降低 LLM 认知负担。Flan-UL2 (20B) 匹配或超越 GPT-4 方案。明确对比了三种范式：
  - **Pointwise**: 逐文档打分，效率高但效果差（缺乏比较信息）。
  - **Pairwise**: 两两比较，效果最好但计算开销大（$O(n^2)$）。
  - **Listwise**: 整列表排序，认知负担最重。

#### Setwise Prompting

- **来源**: [arxiv 2310.09497](https://arxiv.org/abs/2310.09497)（SIGIR 2024）
- **核心方法**: 提出 Setwise 范式作为 Pointwise/Pairwise/Listwise 的补充，在单次 prompt 中处理多个文档（集合），减少 LLM 推理次数和 token 消耗，同时保持高排序质量。

#### RecRanker

- **来源**: [arxiv 2312.16018](https://arxiv.org/abs/2312.16018)
- **核心方法**: 指令微调 LLM 作为推荐系统排序器，结合 pointwise/pairwise/listwise + 混合集成方法。引入自适应采样和位置偏移缓解偏差。

### 3.2 与 BOagent logprob(Yes) 的共通点与差异

| 维度 | LLM Reranker | BOagent logprob(Yes) |
|------|-------------|---------------------|
| **打分信号** | 离散排名 / 偏好对 / logprob | logprob("Yes") 连续值 |
| **任务** | 文档相关性排序 | 候选点物理可行性判断 |
| **比较方式** | pointwise / pairwise / listwise | pointwise（逐候选判断） |
| **与 GP 的关系** | 无 GP，纯 LLM | GP 分数 + LLM logprob 加性融合 |
| **自适应** | 固定 prompt | $\lambda = \gamma \cdot \sigma_{GP}$ 自适应权重 |

**共通点**: 都利用 LLM 的概率分布作为信号源，而非依赖生成的文本内容。都面临 LLM 校准问题。

**关键差异**: BOagent 的 logprob 不是用于排序，而是作为"物理可行性连续信号"与 GP 分数融合。GP 分数方差 $\sigma_{GP}$ 自适应缩放 logprob 权重，这是 reranker 领域没有的设计——reranker 的 LLM 信号是独立的，而 BOagent 的 LLM 信号与 GP 不确定性耦合。([BOagent `docs/theory/logp-metric.md`](./logp-metric.md); [BOagent `packages/bo-core/bo_core/optimization/optimizer.py`](../../packages/bo-core/bo_core/optimization/optimizer.py))

**关于 pointwise vs pairwise 的启示**: BOagent 当前使用 pointwise（逐候选 logprob），根据 PRP 的发现，pairwise 在效果上更优但计算开销更大。如果 Top-K 候选数较小（如 K=5），pairwise 比较的 $O(K^2)$ 开销可控，可能提升物理可行性判断的辨别力。这是潜在的改进方向（推断）。

---

## 4. LLM 生成代码 / 启发式用于优化

BOagent Path A 让 LLM 生成 Python `score_candidate` 函数在沙箱中执行，这与 LLM 生成 reward function / heuristic 的工作直接相关。

### 4.1 Eureka（NVIDIA）

- **来源**: [arxiv 2310.12931](https://arxiv.org/abs/2310.12931)
- **核心方法**: 利用 GPT-4 的零样本代码生成和上下文改进能力，对 reward 函数代码进行进化优化。生成的 reward 用于 RL 训练。在 29 个开源 RL 环境、10 种机器人形态上，83% 的任务超越人类专家设计的 reward，平均标准化提升 52%。支持梯度无关 RLHF 和课程学习。
- **与 BOagent Path A 的关系**: 两者都是"LLM 生成可执行评分代码"。Eureka 生成 RL reward function（连续控制信号），BOagent Path A 生成 `score_candidate` 函数（候选评分）。Eureka 用进化策略迭代改进 reward 代码，BOagent Path A 是单次生成 + 沙箱执行。Eureka 的迭代改进机制是 BOagent Path A 可以借鉴的方向。

### 4.2 OPRO（Optimization by PROmpting）

- **来源**: [arxiv 2309.03409](https://arxiv.org/abs/2309.03409)（ICLR 2024）
- **代码**: [github.com/google-deepmind/opro](https://github.com/google-deepmind/opro)
- **核心方法**: LLM 作为优化器，每步从包含历史解及分数的 prompt 中生成新解，评估后加入下一轮 prompt。在线性回归、TSP 上测试，在 prompt 优化任务上超越人工设计 prompt 达 8%（GSM8K）和 50%（Big-Bench Hard）。
- **与 BOagent 的关系**: OPRO 是"LLM 直接生成候选解"，BOagent Path C 是"LLM 对 GP 候选做 logprob 判断"。OPRO 完全依赖 LLM 生成，BOagent 保持 GP 主导。

### 4.3 LLM 驱动的自动启发式设计

- **MeEvo**: [arxiv 2606.14202](https://arxiv.org/abs/2606.14202) — 自然进化（交叉/变异启发式代码）+ 元认知进化（精炼推理轨迹）的循环框架。
- **MeLA**: [arxiv 2507.20541](https://arxiv.org/abs/2507.20541) — 元认知 LLM 架构，通过"prompt 进化"引导 LLM 生成启发式，包含问题分析器、错误诊断和元认知搜索引擎。
- **LLaMEA-HPO**: [arxiv 2410.16309](https://arxiv.org/abs/2410.16309) — LLM 生成算法结构 + 外部 HPO 调参的混合框架，在 Online Bin Packing、黑盒优化、TSP 上验证。
- **LLMize**: [arxiv 2601.00874](https://arxiv.org/abs/2601.00874) — 开源 Python 框架，将优化建模为黑盒过程，LLM 以自然语言生成候选解，支持 OPRO 和混合策略。

**与 BOagent Path A 的关系**: 这些工作都是"LLM 生成启发式代码/算法"，BOagent Path A 的独特性在于：(1) 生成的函数在 BO 流程内部运行（不是独立优化器）；(2) 与 Path C (logprob) 和 GP 分数三路并行。MeEvo/MeLA 的"元认知进化"和 Eureka 的"进化迭代改进"机制值得借鉴——当前 BOagent Path A 是单次生成，可以引入迭代精炼。

---

## 5. 物理 / 领域先验注入 LLM 做科学优化

### 5.1 PVK-LLM（与 BOagent 最直接相关的钙钛矿工作）

- **来源**: [arxiv 2602.04914](https://arxiv.org/abs/2602.04914)
- **作者**: Penglei Sun, Shuyan Chen, Xiang Liu, Longhan Zhang, Huajie You, Chang Yan, Yongqi Zhang, Xiaowen Chu, Tong-yi Zhang
- **核心方法**: PVK-LLM 是一个领域知识引导的 LLM 框架，将通用语义与钙钛矿领域知识对齐。领域知识嵌入分层贝叶斯优化 (hierarchical BO) 工作流，高效探索高维设计空间。在湿实验中，自主提出未报道的四组分钝化配方 (3MTPAI, PDAI2, EDAI2, PipDI)，实现冠军 PCE 超过 26.0%，接近世界纪录。
- **与 BOagent 的关系**: 两者都是"LLM + BO 优化钙钛矿太阳能电池"。关键异同：
  - PVK-LLM 用"分层 BO"，BOagent 用"GP + LLM 混合采集"。
  - PVK-LLM 的领域知识注入方式是"模型对齐"（微调 LLM），BOagent 是"prompt 注入"（硬编码 CBO/VBO 物理规则到 prompt）。
  - PVK-LLM 在真实湿实验中验证，BOagent 在模拟器上验证。
  - **注**: BOagent 的 `.env` 配置引用了 `PVK-LLM` 数据集路径 (`PVK_LLM_ROOT=../PVK-LLM`)，说明 BOagent 可能使用了 PVK-LLM 的数据集。([BOagent `CLAUDE.md`](../CLAUDE.md))

### 5.2 LEAP（钙钛矿添加剂发现）

- **来源**: [arxiv 2605.20242](https://arxiv.org/abs/2605.20242)
- **核心方法**: 领域专用 LLM 从钙钛矿添加剂文献中提取机制相关知识，用可解释描述符表示候选分子。描述符输入 BO 工作流进行低数据条件下的不确定性感知优先级排序。三轮筛选后，平均 PCE 从 19.25%（对照）提升至 20.13% 和 20.87%，冠军 PCE 21.32%。
- **与 BOagent 的关系**: LEAP 的 LLM 做"特征提取"（从文献到描述符），BOagent 的 LLM 做"可行性判断"（logprob 打分）。LEAP 的 LLM 在 BO 前端（特征工程），BOagent 的 LLM 在 BO 后端（采集精炼）。

### 5.3 PeroMAS（钙钛矿多智能体系统）

- **来源**: [arxiv 2602.13312](https://arxiv.org/abs/2602.13312)
- **核心方法**: 多智能体系统覆盖钙钛矿开发全流程——文献检索、数据提取、属性预测、机制分析。将钙钛矿专用工具封装为 Model Context Protocols (MCPs)，通过规划和工具调用实现多目标约束下的材料设计。真实合成实验验证。
- **与 BOagent 的关系**: PeroMAS 是"全流程多智能体"，BOagent 聚焦"优化环节单智能体"。PeroMAS 的 MCP 工具封装思路可以为 BOagent 扩展更丰富的物理计算工具链提供参考。

### 5.4 LLM as Protein Sequence Optimizer

- **来源**: [arxiv 2501.09274](https://arxiv.org/abs/2501.09274)
- **核心方法**: 发现 LLM 尽管在文本上训练，却可以作为蛋白质序列优化器。结合定向进化 (directed evolution)，LLM 通过 Pareto 优化和实验预算约束优化迭代生成蛋白质变体。在合成和实验适应度景观上验证。
- **与 BOagent 的关系**: 两者都是"LLM 做序列/配方优化"。蛋白质序列是离散氨基酸序列，钙钛矿配方是混合连续-离散参数空间。LLM 作为"序列优化器"的范式在两个领域都有效。

### 5.5 GPT-4 for Scientific Discovery

- **来源**: [arxiv 2311.07361](https://arxiv.org/abs/2311.07361)（Microsoft Research AI4Science）
- **核心方法**: 230 页综合评估 GPT-4 在药物发现、生物学、计算化学 (DFT/MD)、材料设计、PDE 上的能力。初步发现 GPT-4 在科学应用中展现潜力。
- **与 BOagent 的关系**: 该工作奠定了"LLM 有科学知识但需要领域适配"的认知基础。"A Sober Look"（2.7 节）的结论进一步证实：通用 LLM 不足以直接用于 BO，需要领域预训练或先验注入——这正是 BOagent 在 prompt 中硬编码 CBO/VBO 物理规则的原因。

---

## 6. LLM 驱动的实验设计与主动学习

### 6.1 Coscientist（Autonomous Chemical Research）

- **来源**: Boiko et al., "Autonomous chemical research with large language models," *Nature* 624, 570-578 (2023). DOI: [10.1038/s41586-023-06792-0](https://doi.org/10.1038/s41586-023-06792-0)
- **核心方法**: LLM 驱动的自主化学研究系统，能够规划合成路线、控制自动化实验设备、分析结果并迭代优化。实现了从"AI 辅助"到"AI 自主实验"的跨越。
- **注**: Nature 页面需登录访问，上述描述基于二手引用 ([arxiv 2411.07228](https://arxiv.org/abs/2411.07228); [arxiv 2407.16190](https://arxiv.org/abs/2407.16190))。未能获取一手摘要，如需精确引用建议查阅 Nature 原文。
- **与 BOagent 的关系**: Coscientist 是"全自主实验闭环"（包含硬件控制），BOagent 是"优化算法层"（不涉及硬件）。Coscientist 代表了 LLM 驱动实验的终极形态，BOagent 是其中的优化决策组件。

### 6.2 ChemCrow

- **来源**: [arxiv 2304.05376](https://arxiv.org/abs/2304.05376)
- **核心方法**: LLM 化学智能体，集成 18 个专家设计的工具，覆盖有机合成、药物发现和材料设计。自主规划并执行了驱虫剂合成、三种有机催化剂合成，并指导发现了新型发色团。评估发现 GPT-4 作为评估者无法区分 GPT-4 的错误回答和 ChemCrow 的正确结果。
- **与 BOagent 的关系**: ChemCrow 是"LLM + 工具链"模式，BOagent 的三条 LLM 注入路径（A: 生成代码, B: 批量打分, C: logprob）可以看作"BO 专用工具链"。ChemCrow 的多工具集成模式为 BOagent 扩展更多物理计算工具提供参考。

### 6.3 The AI Scientist

- **来源**: [arxiv 2408.06292](https://arxiv.org/abs/2408.06292)
- **核心方法**: 全自动科学发现框架——生成研究想法、编写代码、执行实验、可视化结果、撰写完整论文、运行模拟审稿流程。每个想法到论文成本低于 $15。在扩散模型、Transformer 语言建模、学习动力学三个 ML 子领域验证。
- **与 BOagent 的关系**: AI Scientist 是"科研全流程自动化"，BOagent 是"优化环节自动化"。AI Scientist 的"想法 → 代码 → 实验 → 论文"闭环中，BOagent 类型的系统可以作为其"实验执行"模块。

### 6.4 LLM-AutoSciLab

- **来源**: [arxiv 2605.24043](https://arxiv.org/abs/2605.24043)
- **代码**: [github.com/scientific-discovery/LLM-AutoSciLab](https://github.com/scientific-discovery/LLM-AutoSciLab)
- **核心方法**: 闭环科学发现框架，将假设生成与假设条件实验选择和机制精炼耦合。三步循环：LLM 提出假设 → 选择信息量最大的实验 → 根据证据更新假设空间。比最强基线样本效率高 2-5 倍。
- **与 BOagent 的关系**: LLM-AutoSciLab 的"假设 → 实验 → 精炼"循环与 BOagent 的"GP 候选 → LLM 判断 → 更新"循环在结构上类似。差异在于 LLM-AutoSciLab 的 LLM 做"假设生成"，BOagent 的 LLM 做"可行性判断"。

---

## 对 BOagent 的启示

### 设计验证

1. **GP + LLM 混合采集是主流趋势**: LLINBO、LGBO、LABO、BORA 等多篇论文都采用 LLM + GP 混合策略，验证了 BOagent 设计方向的合理性。但没有任何一篇使用 `logprob(Yes)` 作为连续采集信号——BOagent 的 logprob 融合方案在现有文献中是独特的。

2. **领域先验注入是必要的**: "A Sober Look" (ICML 2024) 明确结论：通用 LLM 不足以改进 BO，需要领域预训练或先验注入。BOagent 在 prompt 中硬编码 CBO/VBO 能带对齐规则，符合这一结论。PVK-LLM 和 LEAP 进一步验证了钙钛矿领域 LLM + BO 的有效性。

3. **DSM 模式有先例但更轻量**: Reasoning-BO 的 NotesAgent + MilvusAgent 双通道存储是 BOagent DSM 的直接灵感来源。BOagent 用纯内存 numpy + 豆包 embedding 替代 Milvus 外部依赖，降低了部署复杂度，但牺牲了知识图谱的结构化推理能力。

### 潜在改进方向

4. **Pairwise logprob 可能优于 Pointwise**: 根据 PRP (NAACL 2024) 的发现，pairwise 比较在效果上显著优于 pointwise。当 Top-K 候选数较小（K=5）时，$O(K^2)$ 的 pairwise logprob 比较开销可控，可能提升物理可行性判断的辨别力。（推断，需实验验证）

5. **LLM 生成代码的迭代精炼**: Eureka 的进化优化和 MeEvo 的元认知进化都表明，迭代改进生成的代码显著优于单次生成。BOagent Path A 当前是单次生成 `score_candidate` 函数，可以引入"生成-评估-精炼"循环。（推断）

6. **LLM 置信度校准**: Evidence-Gated LLM Priors 的工作发现原始 LLM 置信度在不同任务上表现不一致。BOagent 的 $\lambda = \gamma \cdot \sigma_{GP}$ 自适应机制部分缓解了这个问题，但可以考虑引入在线校准（如 Platt scaling）进一步优化。（推断）

7. **与 PVK-LLM 生态对齐**: PVK-LLM (arxiv 2602.04914) 在钙钛矿 BO 领域取得了冠军 PCE > 26% 的湿实验结果，且 BOagent 的数据集路径指向 PVK-LLM。深入对比两者方法、复用 PVK-LLM 的领域知识表征，可能加速 BOagent 的实际应用。（推断）

### 风险提示

8. **LLM BO 领域发展极快**: 本调研中多篇论文的 arxiv ID 在 2605-2606 范围（2026 年 5-6 月），说明该领域处于爆发期。建议定期跟踪 arxiv 更新，特别是 Reasoning-BO 和 PVK-LLM 的后续工作。

---

## 参考文献索引

| 编号 | 名称 | arxiv / 来源 | 会议/期刊 |
|------|------|-------------|----------|
| [1] | Reasoning-BO | [2505.12833](https://arxiv.org/abs/2505.12833) | - |
| [2] | LLINBO | [2505.14756](https://arxiv.org/abs/2505.14756) | - |
| [3] | LGBO | [2605.17976](https://arxiv.org/abs/2605.17976) | - |
| [4] | BORA | [2501.16224](https://arxiv.org/abs/2501.16224) | IJCAI-25 |
| [5] | LABO | [2605.22054](https://arxiv.org/abs/2605.22054) | ICML 2026 |
| [6] | CAKE | [2509.17998](https://arxiv.org/abs/2509.17998) | NeurIPS 2025 |
| [7] | Evidence-Gated LLM Priors | [2606.01730](https://arxiv.org/abs/2606.01730) | - |
| [8] | A Sober Look at LLMs for BO | [2402.05015](https://arxiv.org/abs/2402.05015) | ICML 2024 |
| [9] | LLMs as Uncertainty-Calibrated Optimizers | [2504.06265](https://arxiv.org/abs/2504.06265) | - |
| [10] | PEBOL | [2405.00981](https://arxiv.org/abs/2405.00981) | - |
| [11] | LLM-as-a-Judge | [2306.05685](https://arxiv.org/abs/2306.05685) | NeurIPS 2023 |
| [12] | PRP (Pairwise Ranking Prompting) | [2306.17563](https://arxiv.org/abs/2306.17563) | NAACL 2024 |
| [13] | Setwise Prompting | [2310.09497](https://arxiv.org/abs/2310.09497) | SIGIR 2024 |
| [14] | RecRanker | [2312.16018](https://arxiv.org/abs/2312.16018) | - |
| [15] | Eureka | [2310.12931](https://arxiv.org/abs/2310.12931) | - |
| [16] | OPRO | [2309.03409](https://arxiv.org/abs/2309.03409) | ICLR 2024 |
| [17] | MeEvo | [2606.14202](https://arxiv.org/abs/2606.14202) | - |
| [18] | MeLA | [2507.20541](https://arxiv.org/abs/2507.20541) | - |
| [19] | LLaMEA-HPO | [2410.16309](https://arxiv.org/abs/2410.16309) | - |
| [20] | LLMize | [2601.00874](https://arxiv.org/abs/2601.00874) | - |
| [21] | PVK-LLM | [2602.04914](https://arxiv.org/abs/2602.04914) | - |
| [22] | LEAP | [2605.20242](https://arxiv.org/abs/2605.20242) | - |
| [23] | PeroMAS | [2602.13312](https://arxiv.org/abs/2602.13312) | - |
| [24] | LLM as Protein Sequence Optimizer | [2501.09274](https://arxiv.org/abs/2501.09274) | - |
| [25] | GPT-4 for Scientific Discovery | [2311.07361](https://arxiv.org/abs/2311.07361) | - |
| [26] | Coscientist | [Nature 624, 570-578 (2023)](https://doi.org/10.1038/s41586-023-06792-0) | Nature |
| [27] | ChemCrow | [2304.05376](https://arxiv.org/abs/2304.05376) | - |
| [28] | The AI Scientist | [2408.06292](https://arxiv.org/abs/2408.06292) | - |
| [29] | LLM-AutoSciLab | [2605.24043](https://arxiv.org/abs/2605.24043) | - |
