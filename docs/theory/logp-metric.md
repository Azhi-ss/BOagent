# logP (Log-probability) 物理先验度量机制

## 概述
在 LLMBO (Log-probs Guided Bayesian Optimization) 系统中，我们不直接要求大模型对化学配方的性能进行打分（例如 0-100 分），而是采用了一种基于对数概率 (`logP`) 的度量机制来评估物理可行性。

## 为什么采用 logP？
传统的评分方式容易受到 LLM“讨好倾向”的影响，导致模型倾向于给出虚高的分数，或者在不同实验数据下打分标准不一致（即“幻觉评分”）。

通过 `logP` 机制，我们测量的是模型在生成“Yes”或“No”这个判断时，其**原始的预测概率分布倾向**。这是一种更客观、更具置信度辨析力的量化方式。

## 技术实现逻辑
1.  **约束生成**：在向 DeepSeek API 发起 `evaluate_candidate_viability` 调用时，强制设置 `max_tokens=1`。
2.  **获取概率**：开启 `logprobs=True` 和 `top_logprobs=5` 参数。
3.  **计算倾向**：
    *   模型返回生成的第一个 token（Yes 或 No）及其对应的 `logprob`。
    *   通过 `extract_yes_logprob` 函数，从 `top_logprobs` 中提取出“Yes” token 的对数概率。
    *   如果模型生成“Yes”，其 `logP` 反映了它对此结论的确定程度。
    *   如果模型生成“No”，或者“Yes”的 `logP` 极低，则该配方被判定为不可行。

## 在系统中的应用
获取到的 `logP` 被转化为物理先验分值，直接参与 **Hybrid Fusion (混合融合)** 计算：

$$Score_{hybrid} = GP\_Score + (\gamma \times \sigma_{GP}) \times logP(Yes)$$

其中：
*   $\sigma_{GP}$ 为传统高斯过程（GP）预测的方差（不确定性）。
*   $\gamma$ 为调节物理先验权重的系数。

通过这种方式，只有当 GP 认为候选点有潜力（高 GP Score 或高不确定性），且 LLM 的物理直觉（高 logP）也支持其可行性时，该配方才会获得高 Hybrid Score。
