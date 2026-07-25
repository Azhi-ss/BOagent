# Paper Summary: LABO (LLM-Accelerated Bayesian Optimization)

> **Paper Title**: LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation
> **arXiv ID**: [2605.22054](https://arxiv.org/abs/2605.22054) (Published: May 2026, ICML 2026)
> **Authors**: Xinzhe Zhang et al.
> **Local Resources**:
> - Code Repo: [`references/LABO/`](file:///home/dministrator/project/BOagent/references/LABO)
> - PDF Paper: [`references/LABO/paper.pdf`](file:///home/dministrator/project/BOagent/references/LABO/paper.pdf)

---

## Executive Summary (核心要点)

LABO 提出了针对昂贵物理实验（如湿实验、高算力模拟）的 **双保真度贝叶斯优化（Multi-Fidelity BO）** 新框架：

1. **LLM 作为低保真度 Oracle (Low-Fidelity Source)**：利用 LLM 廉价、高速度的预测能力对广阔搜索空间进行粗粒度探索（Broad Exploration）。
2. **Kennedy-O'Hagan (KOH) 多保真度融合**：建立 KOH 统计高斯过程模型，将低保真 LLM 预测与高保真真实实验数据有机融合。
3. **Gateway 动态门控**：每一轮根据模型的不确定性与期望提升（EI），自动判断是发起低成本 LLM 推理，还是必须启动高成本的真实物理实验。

---

## 对 BOagent 项目的借鉴与改造价值

| 论文维度 | LABO (ICML 2026) | BOagent 现有架构 | 改造与引入方案 |
| :--- | :--- | :--- | :--- |
| **保真度分级** | LLM 低保真预测 + 真机高保真实验 | SKLearn GP + LLM 校验 | **引入 Gateway 门控**：在可运行模式下，根据 GP 不确定性自适应触发 LLM 推理。 |
| **融合算法** | KOH (Kennedy-O'Hagan) 融合模型 | 线性 Hybrid 评分加权 | **升级混合公式**：使用二元高斯过程或者多保真度 GP。 |

---

## 校验与本地归档
- Markdown 报告自动归档至：[`knowledge/summary_labo_2026.md`](file:///home/dministrator/project/BOagent/knowledge/summary_labo_2026.md)
