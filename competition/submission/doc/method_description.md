# 大语言模型增强贝叶斯优化（LGBO）技术方案与算法说明文档

## 1. 方案摘要与执行流程

本提交在有限的离散反应条件池上运行 `LGBOEngine`。算法以固定训练集作为初始观测；每轮重新拟合高斯过程（Gaussian Process, GP），让大语言模型（LLM）提出一个合法反应条件及置信度，用 GP 后验协方差将该建议转换为均值偏移，再以期望改进（Expected Improvement, EI）从所有尚未查询的候选中选择下一次实验。

LLM 不逐候选打分，也不读取候选真实产率。实现中没有 GP Top-20 预筛选、`Yes` 的逐候选对数概率或 $\gamma=0.1$ 混合评分。

---

## 2. 特征构造 (Feature Engineering)

1. **数据集与变量**：
   - `buchwald_sub4`：`Reactant2`、`Ligand`、`Additive`、`Base`；
   - `suzuki`：`Electrophile`、`Nucleophile`、`Ligand`、`Base`、`Solvent`；
   - 两个任务均最大化 `Yield`（%）。
2. **类别独热编码 (One-Hot Encoding)**：
   - 编码器由数据集 `options.json` 中的候选类别构建；测试候选必须严格属于该类别空间。
   - 固定训练先验中仅在训练集出现、但不属于目标候选空间的类别，允许编码为对应特征块的全零向量。
3. **输入变换**：
   - 默认 BoTorch 后端在 `SingleTaskGP` 内使用 `Normalize(d=dimension)`，并用 `Standardize(m=1)` 变换目标值。
   - 可选 `sklearn` 后端使用 `StandardScaler` 标准化输入，并由回归器归一化目标值。

---

## 3. 模型训练与代理模型 (Model Training)

1. **默认代理模型**：
   - 默认 `--backend botorch` 使用 CPU FP64 的 BoTorch `SingleTaskGP`；
   - 协方差核固定为带自动相关维度（ARD）的 `ScaleKernel(MaternKernel(nu=2.5))`，不存在 ALAS 核切换；
   - 每轮使用截至该轮的全部观测重新拟合，并复用兼容的核、似然和均值模块超参数作为暖启动。
2. **拟合参数与数值处理**：
   - 使用 `ExactMarginalLogLikelihood` 和 `fit_gpytorch_mll_scipy`，最大拟合迭代数为 100，最大线搜索步数为 80；
   - `LGBOEngine` 默认 `alpha=1e-2`，依次尝试 $10^{-2}$、$10^{-1}$、$1$ 的 Cholesky jitter；
   - 若拟合或预测失败，该轮降级为观测均值与单位标准差；LLM 调用、解析或均值偏移失败时，则保留纯 GP 均值。
3. **兼容后端**：
   - `--backend sklearn` 使用常数核乘 Matern-5/2 核、`n_restarts_optimizer=10` 和相同的 jitter 尝试序列，仅用于兼容或快速验证。

---

## 4. LLM 引导、均值偏移与候选选择

1. **LLM 点与置信度建议**：
   - 每轮 Prompt 提供反应机理说明、各变量的合法类别，以及最近最多 10 条已观测记录和上一轮 LLM 的思考文本；
   - LLM 返回一个 JSON `point`，或对每个类别维度满足上下界相同约束的 `region`，以及置信度 $c\in[0,1]$；
   - 解析器校验变量顺序和每个类别值，并将置信度截断到 $[0,1]$。合法建议编码为单个独热向量 $x_p$。
2. **后验协方差均值偏移 (Posterior-Covariance Mean Shift)**：
   - 从完整候选池中按类别 Hamming 距离选取距离 $x_p$ 最近的 $K=50$ 个点组成网格 $G$（候选不足 50 时取全部）；
   - 用已拟合 GP 的先验交叉协方差 $k(x_p,x_g)$ 的非负值归一化得到权重 $a$；
   - 以网格上的 GP 后验协方差 $\Sigma_{GG}$ 计算
     $$\lambda=\frac{c}{\sqrt{a^\mathsf{T}\Sigma_{GG}a}},$$
     并对完整候选池的后验均值施加
     $$\mu_{\lambda}(X)=\mu(X)+\lambda\,K_{\mathrm{post}}(X,G)a.$$
   - 此步骤只改变均值，不改变 GP 后验标准差；数值条件无效或计算失败时，$\mu_{\lambda}=\mu$。
3. **采集与选择**：
   - 使用当前最佳已观测值 $f_{\mathrm{best}}$ 和 `xi=0.01`，在偏移后的均值与原 GP 标准差上计算 EI；
   - 已查询候选被屏蔽，算法直接从所有剩余候选中选择有限 EI 最大的点，不进行 Top-K 预筛选；
   - 若全部 EI 非有限，则确定性选择剩余候选中的第一个点。
4. **离线观测更新**：
   - 离线数据加载器保持与候选行对齐的真实 `Yield` 向量，但 GP、LLM Prompt 和 EI 在选择前均不引用该向量；候选索引确定后，评测循环才查询该索引对应的 `Yield`；
   - 选中点的独热向量和产率被追加到观测集，索引标记为已查询，随后进入下一轮拟合。其他候选的真实产率不参与推荐计算。

---

## 5. 实验协议、输出与局限性

1. **默认协议**：入口对 `buchwald_sub4` 和 `suzuki` 分别运行种子 100、200、……、2000，共 20 个种子；每个配置默认运行 40 轮，方法固定为 `lgbo`，默认后端为 `botorch`。
2. **输出**：每个数据集、后端、方法和种子的轨迹写为 CSV，并在安装 PyTorch 时写为 `.pt`；轨迹记录查询索引、条件、单点观测产率、偏移后预测均值和引导状态。指标只基于 40 个预算内查询计算，不把固定训练先验计为新发现。
3. **局限性**：LLM 依赖外部服务与相应环境配置。服务不可用、请求失败、空响应、非法类别或不可解析 JSON 均会使该轮自动退化为纯 GP EI；本文不声明未经当前提交产物支持的性能提升数字。

---

## 6. 防数据泄露与复现入口

1. **信息边界**：LLM 只接收机理背景、合法类别和已观测历史，不接收未查询候选的真实 `Yield`。离线评测数据虽已载入内存，但未查询真值只通过选中索引后的 Oracle 查表进入算法状态。
2. **仓库内复现**：
   ```bash
   uv sync
   uv run python competition/submission/code/main/run_submission.py
   ```
3. **入口参数**：`run_submission.py` 的默认值为 `--datasets buchwald_sub4,suzuki`、`--seeds 100,200,...,2000`、`--n-iters 40`、`--backend botorch`。可通过相应参数覆盖数据集、种子、轮数、后端和输出目录。
4. **独立快照复现**：导出快照后安装其中的 `packages/bo-core`，再运行同一个入口；提交目录不维护第二份算法实现。
