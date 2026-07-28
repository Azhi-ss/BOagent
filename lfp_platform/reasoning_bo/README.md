# Reasoning BO — 离散化学反应基准（Buchwald_sub4 / Suzuki）

基于 Ax/BoTorch 的离散贝叶斯优化（BO）性能测试代码，面向两个经典化学反应条件优化基准：

- **Buchwald_sub4**：Buchwald-Hartwig C-N 胺化偶联，4 个离散决策变量（Reactant2 / Ligand / Additive / Base），测试池 832 行，全局最优 Yield = 86.598%。
- **Suzuki**：Suzuki-Miyaura C-C 偶联，5 个离散决策变量（Electrophile / Nucleophile / Ligand / Base / Solvent），测试池 1728 行，全局最优 Yield = 99.900%。

数据集已内置在本仓库 [`datasets/chemical_reactions/`](../../datasets/chemical_reactions/) 下，脚本会自动解析路径，无需额外下载。

## 方法

| 方法 | 脚本 | 说明 |
|------|------|------|
| **Pool BO** | `scripts/run_chem_bo.py` / `scripts/run_buchwald_bo.py` | 核心算法。对候选池做 one-hot 编码，拟合 GP（SingleTaskGP + 多次随机重启的 MLE，或 SAASBO + NUTS），在**整个离散池上批量评估 qLogEI**，直接取未查询过的最优候选。相比 Ax 默认的连续多起点采集优化 + 候选吸附，更快且无无效/重复候选问题。 |
| **Random 基线** | `scripts/run_buchwald_random.py` | 与 BO 完全相同的评测协议（同种子、同初始训练行、同迭代数、同 oracle），每轮均匀随机取未查询池行，给出无模型下界。 |
| **LLM 重排序** | `scripts/run_chem_bo.py --use_llm` | 每轮由 Pool BO 给出 top-k 候选（含 posterior mean/std、EI、explore/exploit 角色），注入反应机理上下文后让 LLM（DeepSeek / QwQ）重排选点，失败时回退 BO top-1。 |

评测协议：20 个竞赛种子（100..2000），每轮查询 1 个配方共 40 轮；以带标签的 train 行作为 GP 种子试验；oracle 为测试 CSV 查表（离线 dry-run）。指标为 `best_found`（40 轮内最优产率）与 `t95`（best-so-far 首次达到全局最优 95% 的轮数）。

## 安装

> [!NOTE]
> 本模块依赖 `ax-platform==0.5.0` / `botorch`，与仓库主 workspace（`bo-core` pin `botorch>=0.18,<0.19`）可能冲突。建议在**独立虚拟环境**中安装，勿与 `bo-core` / `apps/api` 混装。

```bash
cd lfp_platform/reasoning_bo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
# Buchwald_sub4：Pool BO（20 种子 × 40 轮）
python scripts/run_buchwald_bo.py --num_iterations 40

# Buchwald_sub4：随机基线（同协议公平对比）
python scripts/run_buchwald_random.py --num_iterations 40

# Suzuki：Pool BO（通用 runner，两个数据集共用）
python scripts/run_chem_bo.py --dataset Suzuki --num_iterations 40

# LLM 重排序（需先 cp .env.example .env 并填入 API key）
python scripts/run_chem_bo.py --dataset Buchwald_sub4 --use_llm --llm_backend deepseek --llm_top_k 5

# 其他可选：SAAS 代理模型（horseshoe 先验 + NUTS）、Ax qLogNEI 路径
python scripts/run_buchwald_bo.py --surrogate saas --seeds 100
python scripts/run_chem_bo.py --dataset Suzuki --surrogate saas --seeds 100
python scripts/run_chem_bo.py --dataset Suzuki --acq nei --seeds 100
```

每个种子输出竞赛格式的 `seed_<seed>.pt`（含 trajectory：step / query_index / condition / observed_yield）与 `seed_<seed>_trajectory.csv`（best-so-far 曲线），默认写入 `data/results/<dataset>_bo_<acq>[_llm]/`。

## 性能测试结果

### Buchwald_sub4（全局最优 86.598，95% 阈值 82.268）

| 方法 | 种子数 | best_found 均值 ± 标准差 | best_found 范围 | t95 均值 | t95 达标 | 单轮耗时 |
|------|--------|--------------------------|------------------|----------|----------|----------|
| **Pool BO** | 20 | **85.456 ± 1.083** | [83.149, 86.598] | **15.3** | 20/20 | 1.38s |
| Pool BO + LLM (DeepSeek, top-5) | 10 | 85.232 ± 1.347 | [83.055, 86.598] | 15.4 | 10/10 | 27.41s |
| Random | 20 | 81.481 ± 2.485 | [74.235, 86.598] | 21.7 | 9/20 | — |

### Suzuki（全局最优 99.900，95% 阈值 94.905）

| 方法 | 种子数 | best_found 均值 ± 标准差 | best_found 范围 | t95 均值 | t95 达标 | 单轮耗时 |
|------|--------|--------------------------|------------------|----------|----------|----------|
| **Pool BO** | 20 | **98.079 ± 1.110** | [96.040, 99.900] | 15.7 | 20/20 | 2.17s |
| Pool BO + LLM (DeepSeek, top-5) | 8 | 97.794 ± 1.157 | [96.300, 98.690] | **12.0** | 8/8 | 20.42s |

Pool BO 在两个基准的全部种子上均达到全局最优的 95%，且均值显著高于随机基线（Buchwald +3.98，标准差减半）；LLM 重排序在 Suzuki 上将收敛速度（t95）从 15.7 轮提前到 12.0 轮。

## 目录结构

```
reasoning_bo/
├── scripts/
│   ├── run_buchwald_bo.py        # Buchwald_sub4 Pool BO / qLogNEI 主脚本
│   ├── run_buchwald_random.py    # Buchwald_sub4 随机搜索基线（同协议）
│   └── run_chem_bo.py            # 通用离散化学 BO runner（Buchwald_sub4 / Suzuki，支持 --use_llm）
├── src/
│   ├── bo/
│   │   ├── models.py             # BOModel：Ax BOTORCH_MODULAR 封装（qLogNEI、后验预测）
│   │   └── pool_bo.py            # PoolBO 核心：one-hot 编码 + GP（SingleTask/SAAS）+ 池上批量 qLogEI
│   ├── tasks/buchwald/buchwald.py# DiscreteChemMetric：离散化学查表 oracle（Ax Metric）
│   ├── prompts/                  # PromptManager 与 chem_optimization_loop 提示词（中/英）
│   ├── llms/                     # DeepSeek / QwQ 客户端（OpenAI 兼容接口）
│   ├── config/                   # 环境变量配置 + 两个基准的参数定义 JSON
│   └── utils/jsonl.py            # JSONL 记录工具
├── requirements.txt
└── .env.example                  # LLM API 配置（仅 --use_llm 需要）
```
