# Paper Summary: Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization

> **Paper Title**: Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization
> **arXiv ID**: [2603.28959](https://arxiv.org/abs/2603.28959) (Published: March 2026, IISE Annual Conference 2026)
> **Authors**: Andrea Carbonati, Mohammadsina Almasi, Hadis Anahideh (University of Illinois Chicago)
> **Local Resources**:
> - TeX & Prompt Assets: [`references/multi_agent_acquisition_2603.28959/`](file:///home/dministrator/project/BOagent/references/multi_agent_acquisition_2603.28959)
> - PDF Paper: [`references/multi_agent_acquisition_2603.28959/paper.pdf`](file:///home/dministrator/project/BOagent/references/multi_agent_acquisition_2603.28959/paper.pdf)

---

## Executive Summary (核心要点)

本论文针对单 Agent 大语言模型（Single-Agent LLM）在贝叶斯优化（BO）中同时处理“策略选择（Exploration vs Exploitation）”与“候选点生成”时容易遭遇的**认知过载（Cognitive Overload）与不稳定收敛问题**，提出了一种全新的 **Multi-Agent 架构**：

1. **Strategy Agent (策略 Agent)**：专门负责高层决策。根据当前的优化历史轨迹和上下文，输出 Exploration/Exploitation 各维度指标（Informativeness, Diversity, Representativeness, Exploitation）的归一化权重向量 $w_t$。
2. **Tactical / Generation Agent (战术/候选生成 Agent)**：专门负责具体落地。将 Strategy Agent 输出的权重向量 $w_t$ 作为显式约束（Explicit Constraints），在给定参数空间内采样并生成推荐候选点。

---

## 核心算法架构与交互流程

```
┌──────────────────────────────────────────────────────────────────┐
│ Strategy Agent (策略 Agent)                                      │
│ - 输入：迭代历史 D_{t-1} + 4大指标定义 (Exploitation,             │
│        Informativeness, Diversity, Representativeness)           │
│ - 输出：定量的权重向量 w_t (定界符: ** weights **)                │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 v (传递权重向量 w_t 作为约束)
┌──────────────────────────────────────────────────────────────────┐
│ Generation Agent (生成/战术 Agent)                                │
│ - 输入：权重向量 w_t + 参数边界限制                               │
│ - 输出：具体的参数候选点 (定界符: ## parameters ##)                 │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 v
                     [评估函数 f(x) 并更新 D_t]
```

---

## 关键技术细节与 Prompt 规范

1. **Delimiters 定界符设计**：
   - 策略 Agent 的权重向量必须使用 `** weights **` 进行包裹（便于正则直接解析）。
   - 候选生成 Agent 的参数输出必须使用 `## parameters ##` 包裹，防止大模型自由文本输出导致 JSON 解析失败。

2. **四大评估维度 (Metric Definitions)**：
   - **Exploitation (利用)**：倾向于在当前已知函数值最大的区域附近细粒度搜索。
   - **Informativeness (信息度/探索)**：倾向于选在 GP 后验方差（Variance）最大的未知区域。
   - **Diversity (多样性)**：倾向于在搜索空间中距离已知评估点尽可能远的点。
   - **Representativeness (代表性)**：倾向于覆盖全局特征空间的代表性区域。

---

## 对 BOagent 项目的借鉴与改造价值

| 论文维度 | Multi-Agent Framework (2026.03) | BOagent 现有架构 | 改造与引入方案 |
| :--- | :--- | :--- | :--- |
| **Agent 分工** | 双 Agent（Strategy + Generation） | 单一 Hybrid 逻辑（GP Top-K + KnowledgeEngine） | **解耦决策**：在 `bo_core` 中加入 Strategy Sub-agent，专职动态调节 $\gamma$ 与权重向量。 |
| **输出解析** | 严格分隔符 `** weights **` | JSON / Standard Prompt Parsing | **引入正则定界符**：避免深层 API 格式解析报错。 |
| **指标维度** | 多维度定义 (Info/Diversity/Exploit) | 物理约束与 CBO/VBO 启发式 | **融合领域物理**：在 Strategy Agent 中保留钙钛矿 CBO/VBO 半导体约束。 |

---

## 校验与本地归档
- Markdown 报告自动归档至：[`knowledge/summary_multi_agent_acquisition_2026.md`](file:///home/dministrator/project/BOagent/knowledge/summary_multi_agent_acquisition_2026.md)
