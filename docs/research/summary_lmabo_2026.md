# Paper Summary: LMABO (Adaptive Acquisition Selection for BO with LLMs)

> **Paper Title**: Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models
> **arXiv ID**: [2602.07904](https://arxiv.org/abs/2602.07904) (Published: Feb 2026, ICLR 2026 Submission)
> **Authors**: Giang Ngo, Dat Phan-Trong, Dang Nguyen, Sunil Gupta, Svetha Venkatesh (Deakin University)
> **Code Repo**: [https://github.com/giang-n-ngo/lmabo](https://github.com/giang-n-ngo/lmabo)

---

## Executive Summary

LMABO 提出了一种利用预训练大语言模型（LLM）作为**零样本在线策略师（Zero-Shot Online Strategist）**的贝叶斯优化（BO）新范式。与以往用 LLM 直接预测候选点或替代 GP 的方法不同，LMABO 保持 GP 代理模型的不确定性量化能力，而让 LLM 在每一轮迭代 $t$ 中，根据当前的优化状态文本摘要 $S_t$，动态从采集函数组合库（Portfolio of Acquisition Functions, $\mathcal{A}$）中挑选最适合当前阶段的采集函数 $\alpha_t$。

在 50 个标准 Benchmark 函数及真实调参问题上的实验表明，LMABO 的收敛速度与最终最优值显著优于静态 AF、传统 Bandit 组合算法（如 Hedge、EXP3）以及现有的 LLM-BO 方法。

---

## 核心算法架构与原理

```
                           +-----------------------------------+
                           |  Initial Prompt P0 (Role & Schema)|
                           +-----------------+-----------------+
                                             |
                                             v
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Loop t = 1, ..., T:                                                                     │
│                                                                                        │
│ 1. Fit Gaussian Process: GP_{t-1}                                                      │
│ 2. Construct State Summary S_t:                                                        │
│    - Process Status: Evaluation count, remaining budget N_rem, dimension D              │
│    - Performance History: Incumbent f_min, y range, min distance to past points       │
│    - GP Characteristics: Kernel outputscale, lengthscales (min/max/mean/std)           │
│                                                                                        │
│ 3. Prompt LLM: P_t = P0 + S_t  --->  LLM Output: "AF Abbr: Justification"               │
│                                                                                        │
│ 4. Select AF: alpha_t (e.g., EI, UCB, LogEI, Thompson Sampling, LCB)                   │
│ 5. Optimize alpha_t: x_t = argmax alpha_t(x)                                          │
│ 6. Evaluate Objective: y_t = f(x_t) + noise                                            │
│ 7. Update Dataset: D_t = D_{t-1} U {(x_t, y_t)}                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 状态文本摘要设计 ($S_t$)

LMABO 将连续/高维的数值状态压缩为简洁的文本表示：
1. **进程状态 (Process Status)**: 当前迭代轮次 $N$、剩余预算 $N_{\text{rem}}$、问题维度 $D$。
2. **历史性能 (Performance History)**: 当前最佳值 $f_{\text{min}}$、观测值范围、上一采样点与已知点集的最短距离（指标化 Exploration 倾向）。
3. **GP 模态特征 (GP Characteristics)**: 代理模型拟合核函数的 `outputscale`，以及 `lengthscales` 的统计量（min, max, mean, std）。

消融实验（Ablation Study）证明：**移除以上任何一部分（如 GP 超参或剩余预算）都会导致算法性能显著下降**，说明 LLM 的决策依赖于全面理解优化的整体状态。

---

## 与 BOagent 的深度关联与可借鉴点

| 维度 | LMABO (Deakin, 2026) | BOagent (现架构) | BOagent 可引入与借鉴之处 |
| :--- | :--- | :--- | :--- |
| **LLM 作用定位** | 在线策略师（动态选择 Acquisition Function） | Top-K 候选池的物理规则与 Log-Prob 筛选 | **可引入组合式策略**：让 LLM 根据剩余预算在 EI / UCB / LogEI 之间动态切换 |
| **混合机制** | LLM 决策 AF $\rightarrow$ 数值优化器求解 $x_t$ | `GP_Score + γ * std * log_prob` | **自适应 γ 调节**：在后期剩余预算不足时增大 Exploitation 权重 |
| **状态输入** | 包含 GP lengthscale 统计量与剩余预算 | 主要是离散/连续参数与目标值历史 | **增强 Prompt State**：将 GP 的超参数（Lengthscales）与剩余 Iteration 加入 KnowledgeEngine |
| **开销与计算** | 每一轮仅 1 次 LLM 推理（选择 AF 缩写） | Top-K (20个) 候选点批量 Evaluation | LMABO Token 消耗极小，可作为轻量级自适应调度器 |

---

## 结论与行动建议

1. **将 LMABO 代码库引入 `references/lmabo`**（或者作为 `references/` 中的核心参考实现）。
2. **在 `bo-core` 中探索可调 Acquisition 机制**：除了目前标准的 Expected Improvement (EI)，可在 `optimizer.py` 中拓展支持 UCB / LogEI 族，并利用 KnowledgeEngine 注入当前轮数与 Lengthscale 状态。
