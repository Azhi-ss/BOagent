# LLAMBO: 基于大语言模型的贝叶斯优化增强框架 (ICLR 2024)

> **论文引用**: [arxiv:2402.03921](https://arxiv.org/abs/2402.03921)  
> **作者**: Tennison Liu*, Nicolás Astorga*, Nabeel Seedat, Mihaela van der Schaar (剑桥大学 DAMTP)  
> **代码仓库**: `references/LLAMBO/`

---

## 1. 核心架构与概要

**LLAMBO** 是一种将大语言模型（LLM）融入传统贝叶斯优化（BO）流程的模块化框架。它无需对大模型进行微调 (No Fine-tuning)，通过将优化历史、搜索空间及领域上下文转化为自然语言提示词（Prompts），充分利用 LLM 的上下文学习（ICL）、零样本迁移能力和领域先验知识。

LLAMBO 包含三个独立且即插即用的核心模块：
1. **零样本热启动 (Zero-Shot Warmstarting)**：根据数据集卡片与模型描述，生成具有物理/领域相关性的优质初始点。
2. **上下文代理模型 (In-Context Surrogate Modeling)**：无需高斯过程（GP），直接通过上下文学习预测目标值及其不确定性。
3. **目标条件候选点采样 (Target-Conditioned Candidate Sampling)**：通过直接对期望目标值（$s'$）进行条件化概率采样，获取高质量候选点。

---

## 2. 三大核心模块详解

### 2.1 零样本热启动 (Warmstarting)
- **痛点**：传统热启动（如元学习 Meta-Learning）依赖于历史相似任务的离线数据库，收集成本高且新领域往往缺失。
- **机制**：基于 Zero-Shot Prompt 提供不同层级的上下文：
  - **无上下文 (No Context)**：仅告知超参数名称与范围。
  - **部分上下文 (Partial Context)**：引入数据卡片（`<DATA CARD>`：样本数、特征数、特征类型、任务类型）。
  - **全上下文 (Full Context)**：进一步提供特征边缘分布及特征-标签相关性信息。
- **结论**：LLM 生成的初始点保留了超参数间的结构相关性，在优化早期（试验数 $n \le 5$）的遗憾度（Regret）显著低于 Sobol 或拉丁超立方（Latin Hypercube）等随机采样方法。

---

### 2.2 上下文代理模型 (Surrogate Model)

#### 判别式代理模型 ($p(s|h; \mathcal{D}_n)$)
- **序列化表示**：将历史观测转化为自然语言文本：  
  *`"max_depth is 15, min_samples_split is 0.5, ..., accuracy is 0.9"`*
- **顺序偏置打乱 (Shuffle Monte Carlo)**：
  - LLM 对 Prompt 中 In-Context 示例的排列顺序非常敏感（存在从左到右的位置偏置）。
  - **解决方案**：在多次采样中对历史示例顺序进行随机打乱 (Shuffling)，计算经验均值与标准差作为预测值和不确定性。
  - **性能表现**：在小样本阶段（$n \le 10$），预测准确度（$NRMSE$, $R^2$）优于传统 GP，但在不确定性校准（LPD/覆盖率）上 GP 仍更具优势。

#### 生成式代理模型 ($p(s \le \tau | h; \mathcal{D}_n)$)
- 将代理模型转化为概率分类任务，估计候选点超参数 $h$ 的目标值优于指定阈值 $\tau$ 的概率。

---

### 2.3 目标条件候选点采样 ($p(h | s'; \mathcal{D}_n)$)

- **与 TPE 的对比**：传统 TPE 将历史样本按阈值划分为 good/bad 两个集合，而 LLAMBO 直接根据指定的期望目标值 $s'$ 进行条件生成：
  $$s' = s_{\text{min}} - \alpha \times (s_{\text{max}} - s_{\text{min}})$$
  其中 $\alpha$ 为**探索超参数 (Exploration Hyperparameter)**。
- **$\alpha$ 的调优效应**：
  - $\alpha > 0$（如 $\alpha = 0.01$）：具备向已知历史最优解以外进行外推 (Extrapolation) 的能力，可发现全新高潜力区域。
  - $-1 \le \alpha < 0$（如 $\alpha = -0.2$）：在已知观察分布范围内进行相对保守的局部采样。
- **候选池筛选**：生成 $M$ 个候选点后，使用采集函数（如 Expected Improvement）计算评分并选取最佳点进行下一次实际试验。

---

## 3. 对 BOagent (`bo-core`) 的借鉴与启发

将 LLAMBO 与 BOagent 的 `LGBOEngine` / `ChemLGBOEngine` 进行对比：

| 特性维度 | LLAMBO (ICLR '24) | BOagent (`LGBOEngine` / `ChemLGBOEngine`) |
| :--- | :--- | :--- |
| **LLM 作用定位** | 直接作为候选点生成器与全量代理模型 | 区域提升后验均值偏移 (Mean Shift) / 子空间约束 |
| **代理模型基座** | LLM ICL 蒙特卡洛采样 / 判别式模型 | BoTorch Matern-5/2 高斯过程 (GP) |
| **候选点产生机制** | 根据期望目标条件采样 $h \sim p(h\|s')$ | GP 对全池打分，叠加 LLM 点/子空间偏移 |
| **Prompt 顺序偏置防御**| 历史示例随机打乱 (In-Context Shuffling) | 按照时间线线性格式化输出 |
| **探索控制机制** | 目标外推参数 $\alpha$ | 子空间偏移量 $\sigma(x)$ / 后验协方差 $\lambda$ |

### 给 BOagent 的可落地优化方案：
1. **历史示例随机打乱 (Prompt Shuffling)**：
   - 改造 `chem_lgbo_prompt.py` 中历史观测的组织方式，引入随机打乱或多视角上下文，减轻 LLM 对首尾历史记录的强位置偏置。
2. **目标导向 Prompt 引导 (Target Conditioning)**：
   - 在向 LLM 询问反应子空间或推荐条件时，显式带上目标期望值 $s'$（例如“请推荐能突破 Yield > 95% 的特征条件”），比纯历史列表更容易触发 LLM 的外推探索能力。
3. **点表达 (Point) 与子空间 (Subspace) 的融合**：
   - LLAMBO 采用点级别条件生成 $h \sim p(h|s')$，天然避免了离散子空间 (Subspace) 掩码计算中硬边界造成的“断崖式”奖励与 `already_queried_only` 异常，印证了回归 `LGBOEngine._mean_shift` 点表达的合理性。
