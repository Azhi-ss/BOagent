# Reasoning BO: 基于长上下文推理模型的贝叶斯优化框架 (ArXiv 2025)

> **论文引用**: [arxiv:2505.12833](https://arxiv.org/abs/2505.12833)  
> **作者**: Zhuo Yang, Daolang Wang, Lingli Ge, Beilun Wang, Tianfan Fu, Yuqiang Li (上海 AI Lab, 西安电子科技大学, 拟复旦, 东南大学, 南京大学)  
> **代码仓库**: `references/Reasoning-BO/`

---

## 1. 算法核心架构设计 (Algorithm Architecture)

Reasoning BO 将长上下文大语言模型（如 QwQ-32B / DeepSeek-R1 / Qwen2.5-Instruct）的思维链（CoT）推理能力引入贝叶斯优化（BO）闭环中，重点解决传统 BO 容易陷入局部最优及缺乏可解释性的痛点。

算法流程包含 **三大部分 + 四步闭环**：

### 1.1 核心三大设计模块
1. **实验指南针 (Experiment Compass)**：
   - 用户用自然语言定义优化任务、目标和参数约束。LLM 依据指南针解析搜索空间并建立初始全局假设。
2. **双通道动态知识库 (Dual-Channel Knowledge Management)**：
   - **知识图谱 (Knowledge Graph / KGAgent)**：提炼 `<实体, 关系, 实体>` 三元组，维护结构化、可解释的概念关联。
   - **向量数据库 (Milvus Database / MilvusAgent)**：基于 OpenAI Embedding（1536 维）将每轮生成的科学笔记（`key_findings`, `parameter_relationships`, `optimization_principles`）向量化存储，支持开箱即用的相似度语义召回。
3. **LLM 候选点过滤机制 (Candidate Filtering & Hypothesis Evaluation)**：
   - 高斯过程（基于 **qLogEI** 采集函数）提出 $N=5$ 个候选点。
   - LLM 结合从双通道知识库召回的动态上下文，评估每个候选点的科学可行性并赋予置信度得分，挑选出 **Top-3 高置信度点** 送入真实/模拟实验。
   - **关键原则**：*避免直接让 LLM 生成候选点以防盲目外推或数据污染*。

### 1.2 迭代优化算法四步循环 (Algorithm 1)
```text
输入: 自然语言实验指南针 C, 预算 N, 采集函数 qLogEI
输出: 最优参数配置 x* 和经验知识库 K

1. 初始化: 
   - 提取实验指南针 C 中的关键概念，生成初始提示与概述
   - 进行首轮采样并用 NotesAgent 解析初始 CoT 推理轨迹写入向量库/知识图谱

2. 循环迭代 (For step = 1 to N):
   a. [知识召回]: 根据当前搜索方向的关键词，从知识图谱与 Milvus 向量库中双路召回 Top-K 相关笔记
   b. [GP 提议]: 拟合 GP Surrogate Model，利用 qLogEI 提议 5 个候选样本
   c. [LLM 评估与假设演化]: 
      - LLM 结合历史数据 + 召回知识，评估候选点
      - 放弃表现差的假设，提出新假设并给候选点赋予置信度
   d. [过滤与执行]: 挑选置信度最高的 Top-3 候选点执行实验，获得新观察值 y
   e. [增量知识写入]: 抽取本轮 CoT 推理轨迹中的科学发现，写入向量库与知识图谱
```

---

## 2. 实验测试具体数据集 (Datasets & Benchmarks)

论文在 **10 个多样化基准任务** 上进行了验证，分为**真实世界科学优化**与**高维合成函数**两大类：

### 2.1 真实世界 BO 基准 (Real-World Benchmarks)
1. **Direct Arylation Reaction (直接芳基化反应优化)**：
   - **领域**: 有机化学合成产率优化。
   - **优化参数**: 配体 (Ligand)、碱 (Base)、溶剂 (Solvent)、浓度、温度。
   - **数据集来源**: McNally et al. 2011 真实数据集。
2. **Suzuki-Miyaura Reaction (铃木-宫浦偶联反应)**：
   - **领域**: 催化交叉偶联反应。
   - **优化参数**: 亲电试剂-亲核试剂配对、催化剂、配体、碱、溶剂。
   - **数据集来源**: Perera et al. 2018。
3. **CPA-Catalyzed Thiol-Imine Reaction (手性磷酸催化反应)**：
   - **领域**: 不对称催化反应。
4. **Lunar Lander (月球着陆器控制策略)**：
   - **领域**: 连续-离散混合控制策略优化，平衡燃料效率与着陆精度。

### 2.2 合成数学函数 (Synthetic Functions)
为了防止 LLM 记住了有名函数的闭式解，论文在 Prompt 中隐去了函数真实名称，替换为通用词 `"mathematical function"`：
1. **Ackley (多峰函数)**：几乎平坦的外围与极陡峭的中央谷底。
2. **Rosenbrock (香蕉函数/非凸函数)**：具有狭窄的抛物线谷底。
3. **Hartmann (高维极值函数)**：多个局部极小值点，存在强烈的参数耦合效应。

---

## 3. 实验关键性能结果 (Performance Highlights)

在最高 30 轮预算限制下（每轮选 3 点），Reasoning BO 在各项指标上均大幅超越 Vanilla BO 和 CMA-ES：

- **Direct Arylation 反应**:
  - **最终产率**: Reasoning BO 达到 **60.76%**（Vanilla BO 仅 43.62%，Random Search 28.74%）。
  - **初始探索性能 (IMP@1)**: 达到 **60.07%**，较 Vanilla BO 初始性能提高 44.6%。
- **Suzuki 偶联反应**:
  - 最终目标值达到 **74.66**（Vanilla BO 为 67.26），且标准差从 12.21 降至 **7.71**（稳定性最高）。

---

## 4. 与 BOagent (`bo-core`) 的继承与异同

| 模块 | Reasoning-BO (ArXiv 2025) | BOagent (`bo-core`) |
| :--- | :--- | :--- |
| **知识库技术栈** | 外置 Milvus 向量数据库 + 知识图谱 (Neo4j/NetworkX) | **纯内存 Numpy 矩阵 + 豆包 Embedding API (`memory.py`)** |
| **部署依赖** | 重型 (依赖 Docker / Milvus 容器) | **极轻量** (开箱即用，零外部数据库依赖) |
| **LLM 筛选机制** | 离散筛选 (从 5 个候选点中取 Top-3) | **连续评分** ($GP\_Score + \gamma \sigma_{GP} \cdot \log P(\text{Yes})$ 混合计算) |
