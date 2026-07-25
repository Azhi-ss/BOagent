# Deep Research: LLM-Guided Bayesian Optimization 开源仓库全景

> 研究时间: 2026-07-25 | 深度: Thorough | 数据来源: 多轮 Web Search

## Executive Summary

LLM 与 Bayesian Optimization (BO) 的融合在 2024-2025 年爆发式增长，已从学术概念验证走向可落地的开源框架。本报告系统梳理了与 BOagent 算法方向（LLM-guided GP-based BO for scientific formulation optimization）高度相关的**26个**开源仓库，按算法类别分层分析，标注与 BOagent 的对比关系和可借鉴之处。

---

## 已有 References 基线

| 仓库 | 算法类别 | 核心思路 |
|------|---------|---------|
| `PVK-LLM` | Hierarchical LLM-BO | LLM 驱动分层贝叶斯优化（钙钛矿太阳能电池） |
| `Reasoning-BO` | Long-Context Reasoning BO | 利用 LLM 长上下文推理增强 BO 决策 |
| `Text-to-BatteryRecipe` | NLP → Recipe Extraction | 从文献自动提取电池配方 |
| `LGBO (Unleashing LLMs in BO)` | Preference-Guided BO | LLM 偏好引导的科学发现框架 |
| `CAKE` | Context-Aware Kernel Evolution | 上下文感知核函数演化 |

---

## Tier 1: 直接竞品 / 高相关（LLM-in-the-Loop BO）

这些仓库与 BOagent 在算法架构上最为接近，直接构成对标或可借鉴的参考实现。

### 1. LLAMBO — Large Language Models to Enhance Bayesian Optimization

| 属性 | 值 |
|------|-----|
| **GitHub** | [tennisonliu/LLAMBO](https://github.com/tennisonliu/LLAMBO) |
| **论文** | ICLR 2024 |
| **开源时间** | 2023-10 |
| **算法类别** | LLM-Enhanced Surrogate + Warmstart |
| **核心思路** | 将 BO 问题用自然语言描述，LLM 零样本热启动、改进代理模型和采样 |
| **技术栈** | Python, OpenAI API |

**与 BOagent 对比**:
- LLAMBO 让 LLM **替代** GP 做代理建模；BOagent 让 LLM **增强** GP 的 Top-K 筛选
- LLAMBO 更适合超参调优；BOagent 针对物理约束的科学配方优化
- **可借鉴**: 零样本热启动策略、自然语言问题描述模板

---

### 1b. lapeft-bayesopt — Discrete Bayesian Optimization with LLMs

| 属性 | 值 |
|------|-----|
| **GitHub** | [wiseodd/lapeft-bayesopt](https://github.com/wiseodd/lapeft-bayesopt) |
| **论文** | 2024 |
| **开源时间** | 2024-03 |
| **算法类别** | LLM PEFT + Discrete BO |
| **核心思路** | 使用参数高效微调（PEFT）的 LLM 隐空间进行离散贝叶斯优化 |
| **技术栈** | Python, PyTorch |

---

### 2. LLINBO — Trustworthy LLM-in-the-Loop Bayesian Optimization

| 属性 | 值 |
|------|-----|
| **GitHub** | [UMDataScienceLab/LLM-in-the-Loop-BO](https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO) |
| **论文** | arXiv 2024 |
| **开源时间** | 2024-06 |
| **算法类别** | Hybrid LLM + GP Surrogate |
| **核心思路** | LLM 与统计代理（GP）混合，保留不确定性量化的可信度 |
| **技术栈** | Python, GPyTorch |

**与 BOagent 对比**:
- **架构最接近 BOagent** — 都是 GP 为主体，LLM 做辅助判断
- LLINBO 侧重"可信度"理论证明；BOagent 侧重领域物理规则注入
- **可借鉴**: 混合评分公式的理论框架、不确定性量化方法

---

### 3. BORA — Bayesian Optimisation Research Assistant

| 属性 | 值 |
|------|-----|
| **GitHub** | [Ablatif6c/llm-closed-loop-experiments](https://github.com/Ablatif6c/llm-closed-loop-experiments) |
| **论文** | 2024 |
| **开源时间** | 2024-08 |
| **算法类别** | Adaptive LLM-BO Strategy Selection |
| **核心思路** | LLM 根据停滞检测自适应决定：纯 BO / 纯 LLM / 混合策略 |
| **技术栈** | Python |

**与 BOagent 对比**:
- BORA 引入了**策略切换机制**（plateau detection）— BOagent 目前是固定 hybrid 权重 γ
- **可借鉴**: 自适应 γ 权重调节、停滞检测触发 LLM 介入的时机

---

### 4. LLM4BO — Benchmarking LLM approaches for BO

| 属性 | 值 |
|------|-----|
| **GitHub** | [learningmatter-mit/LLM4BO](https://github.com/learningmatter-mit/LLM4BO) |
| **论文** | MIT, 2024 |
| **开源时间** | 2024-09 |
| **算法类别** | Benchmark Suite |
| **核心思路** | 系统评估多种 LLM（DeepSeek-R1、Qwen3 等）在 BO 任务上的表现 |
| **技术栈** | Python, Multiple LLM APIs |

**与 BOagent 对比**:
- 不是优化框架，而是**评测框架** — 可用于评估 BOagent 的算法与其他方法的对比
- **可借鉴**: 标准化的 benchmark 任务设计、蛋白质/分子优化测试集

---

### 5. GOLLuM — Gaussian Process Optimized LLMs

| 属性 | 值 |
|------|-----|
| **GitHub** | [schwallergroup/gollum](https://github.com/schwallergroup/gollum) |
| **论文** | 2024 |
| **开源时间** | 2024-04 |
| **算法类别** | GP-Guided LLM Finetuning |
| **核心思路** | 用 GP 建模 LLM embedding 的隐空间，形成双向优化回路 |
| **技术栈** | Python, GPyTorch, PyTorch |

**与 BOagent 对比**:
- 方向相反：GOLLuM 用 GP 去优化 LLM；BOagent 用 LLM 去增强 GP
- **可借鉴**: Deep Kernel 方法、GP + 神经网络的联合训练技巧

---

## Tier 2: 同域相关（BO 框架 / 科学优化）

### 6. BoTorch — Bayesian Optimization in PyTorch

| 属性 | 值 |
|------|-----|
| **GitHub** | [pytorch/botorch](https://github.com/pytorch/botorch) |
| **开源时间** | 2019-09 |
| **算法类别** | GP-based BO 框架 |
| **核心思路** | 基于 PyTorch + GPyTorch 的模块化 BO 库，支持多目标、约束优化 |
| **Stars** | 3.3k+ |

**与 BOagent 对比**:
- **BOagent 计划中的 GP 后端升级目标**（`07-24-upgrade-botorch` 任务）
- 提供 `qExpectedImprovement`, `qNoisyExpectedImprovement` 等现成 acquisition function

---

### 7. Ax — Adaptive Experimentation Platform

| 属性 | 值 |
|------|-----|
| **GitHub** | [facebook/Ax](https://github.com/facebook/Ax) |
| **开源时间** | 2019-05 |
| **算法类别** | Experiment Orchestration + BO |
| **核心思路** | BoTorch 的上层封装，管理实验全生命周期 |
| **Stars** | 2.5k+ |

**与 BOagent 对比**:
- Ax 是通用实验平台；BOagent 是面向特定物理领域的 LLM-augmented 系统
- **可借鉴**: 实验管理 API 设计、多目标约束处理模式

---

### 8. SMAC3 — Sequential Model-based Algorithm Configuration

| 属性 | 值 |
|------|-----|
| **GitHub** | [automl/SMAC3](https://github.com/automl/SMAC3) |
| **开源时间** | 2016 (v3: 2022) |
| **算法类别** | Random Forest Surrogate BO |
| **核心思路** | 用随机森林替代 GP 做代理，擅长离散/类别型超参 |
| **Stars** | 1.1k+ |

**与 BOagent 对比**:
- BOagent 用 GP；SMAC3 用 Random Forest — 两种代理模型的对比基线
- **可借鉴**: 离散搜索空间的处理方式（BOagent 的 `DiscreteParameter`）

---

### 9. Optuna — Hyperparameter Optimization Framework

| 属性 | 值 |
|------|-----|
| **GitHub** | [optuna/optuna](https://github.com/optuna/optuna) |
| **开源时间** | 2018 |
| **算法类别** | TPE-based HPO |
| **核心思路** | Define-by-run API，TPE 采样器，aggressive pruning |
| **Stars** | 11k+ |

**与 BOagent 对比**:
- 通用 HPO 工具 vs 领域特定 LLM-BO — 定位不同
- **可借鉴**: Pruning 策略、可视化 dashboard 设计

---

### 10. BORE — Bayesian Optimization by Density-Ratio Estimation

| 属性 | 值 |
|------|-----|
| **GitHub** | [ltiao/bore](https://github.com/ltiao/bore) |
| **论文** | ICML 2021 |
| **开源时间** | 2021-06 |
| **算法类别** | Classification-based BO |
| **核心思路** | 将 acquisition function 重铸为二分类问题，避免 GP 解析约束 |
| **技术栈** | TensorFlow 2 |

**与 BOagent 对比**:
- 完全不同的代理模型范式：密度比估计 vs GP
- 后续被集成到 AWS Syne Tune 框架中
- **可借鉴**: 无 GP 的 BO 思路，作为高维场景的替代 backend

---

## Tier 3: 生态扩展（自主科学发现 & 领域 Agent）

### 11. The AI Scientist (v1 & v2)

| 属性 | 值 |
|------|-----|
| **GitHub** | [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) |
| **开源时间** | 2024-08 |
| **算法类别** | End-to-End Autonomous Research Agent |
| **核心思路** | LLM 自主完成"想法→实验→论文"全流程 |
| **Stars** | 10k+ |

**与 BOagent 对比**:
- AI Scientist 是全栈科研 Agent；BOagent 专注优化 loop
- v2 引入 agentic tree search — 类似 BOagent 的多步决策
- **可借鉴**: 实验反馈循环设计、失败诊断机制

---

### 12. ChemCrow — Chemistry LLM Agent

| 属性 | 值 |
|------|-----|
| **GitHub** | [ur-whitelab/chemcrow-public](https://github.com/ur-whitelab/chemcrow-public) |
| **开源时间** | 2023-05 |
| **算法类别** | Tool-Augmented Chemistry LLM |
| **核心思路** | LLM + 18 个化学专用工具（合成规划、性质预测、分子设计） |
| **Stars** | 800+ |

**与 BOagent 对比**:
- ChemCrow 是工具增强型 Agent；BOagent 是优化增强型
- **可借鉴**: 领域工具集成模式（BOagent 的 KnowledgeEngine 类似于 ChemCrow 的 tool registry）

---

### 13. LILO — BO with Natural Language Feedback

| 属性 | 值 |
|------|-----|
| **GitHub** | [facebookresearch/LILO](https://github.com/facebookresearch/LILO) |
| **开源时间** | 2024-02 |
| **算法类别** | NL-Feedback BO |
| **核心思路** | 用自然语言反馈替代标量反馈来近似效用函数 |

**与 BOagent 对比**:
- LILO 用 NL 做反馈；BOagent 用 NL 做知识注入 — 互补方向
- **可借鉴**: 自然语言→效用函数的转换机制

---

## Tier 4: Awesome Lists & 资源聚合

| 仓库 | 链接 | 描述 |
|------|------|------|
| **Awesome-LLM-Scientific-Discovery** | [HKUST-KnowComp/...](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery) | LLM 科学发现全景（Tool/Analyst/Scientist 三级分类） |
| **Awesome Bayesian Optimization** | [wjmaddox/...](https://github.com/wjmaddox/awesome-bayesian-optimization) | BO 领域最全资源列表 |
| **Awesome-AI-Scientists** | [tsinghua-fib-lab/...](https://github.com/tsinghua-fib-lab/Awesome-AI-Scientists) | 清华 AI Scientist Agent 汇总 |
| **Awesome-AI-Science-and-Innovation** | [TJUNLP-xxy/...](https://github.com/TJUNLP-xxy/Awesome-AI-Science-and-Innovation) | 闭环科学实验 Agent 汇总 |

---

## 按算法类别汇总

| 类别 | 仓库 | 与 BOagent 关系 |
|------|------|----------------|
| **Hybrid LLM + GP** | LLINBO, BORA, PVK-LLM, LGBO | ⭐ 直接竞品 |
| **LLM-as-Surrogate** | LLAMBO, GOLLuM | 替代范式 |
| **Long-Context Reasoning** | Reasoning-BO | ⭐ 已在 references |
| **Benchmark Suite** | LLM4BO | 评测工具 |
| **GP-based BO Framework** | BoTorch, Ax | ⭐ 底层引擎 |
| **Alternative Surrogate** | SMAC3, Optuna, BORE | 对比基线 |
| **Kernel Engineering** | CAKE | ⭐ 已在 references |
| **NL-Feedback BO** | LILO | 互补方向 |
| **Autonomous Science Agent** | AI Scientist, ChemCrow | 生态上游 |
| **NLP → Data Extraction** | Text-to-BatteryRecipe | ⭐ 已在 references |

---

## 按开源时间线

```
2018  ──── Optuna
2019  ──── BoTorch, Ax
2021  ──── BORE (ICML)
2022  ──── SMAC3 v3
2023  ──── LLAMBO (→ICLR'24), ChemCrow, PVK-LLM
2024  ──── LLINBO, BORA, GOLLuM, LLM4BO, AI Scientist, LILO, LGBO
2025  ──── Reasoning-BO, CAKE, LLM4BO updates (DeepSeek-R1, Qwen3)
```

---

## 推荐克隆优先级

> [!IMPORTANT]
> 以下排序基于与 BOagent 当前算法架构的相关性和可借鉴价值。

| 优先级 | 仓库 | 理由 |
|--------|------|------|
| 🔴 P0 | **LLINBO** | 架构最接近，Hybrid GP+LLM 理论框架可直接参考 |
| 🔴 P0 | **LLAMBO** | ICLR 2024，LLM warmstart 和 surrogate 增强的标杆实现 |
| 🟠 P1 | **BORA** | 自适应策略切换（γ 动态调节）值得研究 |
| 🟠 P1 | **LLM4BO** | MIT benchmark suite，可用于 BOagent vs 其他方法的公平对比 |
| 🟡 P2 | **AI Scientist** | 全栈科研 Agent 参考，v2 的 tree search 架构有启发 |
| 🟡 P2 | **GOLLuM** | GP + Deep Kernel 技术，BoTorch 升级时可参考 |
| 🟢 P3 | **LILO** | NL-Feedback 方向探索 |
| 🟢 P3 | **ChemCrow** | 工具集成模式参考 |

---

## Rerun Inputs

```
workflow: deep-research (web-search)
topic: LLM-guided Bayesian Optimization open-source repositories
depth: thorough (6 rounds, 8 queries)
output: markdown
baseline: references/ (PVK-LLM, Reasoning-BO, LGBO, CAKE, T2BR)
```
