# PVK-LLM / PVK-BO: 基于 LLM 的钙钛矿太阳能电池贝叶斯优化框架

> **项目路径**: `references/PVK-LLM/`  
> **官方主页**: [sites.google.com/view/pvk-llm](https://sites.google.com/view/pvk-llm/main-page)  
> **继承基座**: 基于 **LLAMBO** (ICLR '24) 架构针对半导体器件物理领域进行的重构与扩展

---

## 1. 核心定位与项目概要

**PVK-BO** 是首个专门面向**钙钛矿太阳能电池（Perovskite Solar Cells, PSC）**材料配方与能带结构参数优化的大语言模型驱动贝叶斯优化框架。

它将半导体物理学约束（如导带/价带偏移 CBO/VBO、界面缺陷陷阱密度 $N_t$、受主/施主掺杂浓度 $N_a, N_d$）转化为 Prompt 语言描述，结合 LLM 的先验推论能力与少样本上下文学习（ICL），大幅降低寻找高效率光伏器件所需的试验轮次。

---

## 2. 优化任务与物理参数空间

框架重点针对钙钛矿光伏电池的两大关键物理瓶颈建模：

### 2.1 能带匹配优化 (Band Alignment Optimization)
- **目标**: 极大化光电转换效率 ($\eta$ / `eta`)
- **关键物理参数 (5维)**:
  - `CHI_PVK`: 钙钛矿吸光层电子亲和能
  - `Eg_HTL`, `CHI_HTL`: 空穴传输层 (HTL) 禁带宽度与电子亲和能
  - `Eg_ETL`, `CHI_ETL`: 电子传输层 (ETL) 禁带宽度与电子亲和能
- **物理约束规则**: 
  - $CBO = \chi_{PVK} - \chi_{ETL}$ (理想范围 $[-0.1, 0.3]\text{ eV}$)，避免悬崖 (Cliff) 导致 $V_{oc}$ 损失或突起 (Spike) 阻挡电子萃取。
  - $VBO = (\chi_{HTL} + E_{g,HTL}) - \chi_{PVK}$ (理想范围 $[1.7, 2.0]\text{ eV}$)。

### 2.2 缺陷与掺杂优化 (Defects & Doping Optimization)
- **目标**: 极大化电池光电转换效率
- **关键物理参数 (8维)**:
  - 缺陷密度: `Nt_PVK`, `Nt_ETL`, `Nt_HTL`, `Nt_HTL/PVK`, `Nt_PVK/ETL` (界面陷阱)
  - 掺杂浓度: `Na_PVK`, `Nd_PVK`, `Na_HTL`
- **物理约束规则**: 
  - 陷阱密度 $N_t$ 对数级降低可线性提升开路电压 $V_{oc}$。
  - 掺杂浓度必须严格限制在 $10^{19}\text{ cm}^{-3}$ 以下，防止隧穿复合与界面漏电。

---

## 3. 关键技术特性

1. **输入空间扭曲 (Input Warping / Log Transformation)**
   - 缺陷密度与掺杂浓度跨越 5~6 个数量级（例如 $10^{14} \sim 10^{19}$）。PVK-BO 在 `pvk_bo/warping.py` 中实现了 `NumericalTransformer`，将宽量级参数变换至对数空间后序列化输入给 LLM，解决了文本大模型对跨数量级浮点数感知迟钝的问题。
2. **双模代理模型 (Generative & Discriminative SM)**
   - **Discriminative SM**: 使用 LLM 对候选点进行 Pairwise 成对对比，预测优劣顺序。
   - **Generative SM**: 使用 LLM 预测目标值的经验概率分布。
3. **频率控制与容错 (RateLimiter & Legacy Compat)**
   - 内置 `RateLimiter`（限制 100k tokens/min, 720 requests/min）防御 API 限制。
   - 包含了对 Legacy Pandas (`Series.__getitem__` 降级 `iloc`) 和 LangChain / OpenAI API 的 Schema 拦截补丁。

---

## 4. 与 BOagent (`bo-core`) 的深度联动

1. **兼容补丁集成 (`pvk_llm_compat.py`)**:
   - BOagent 直接复用了 PVK-LLM 的猴子补丁逻辑，确保旧版 Pandas 索引与 DeepSeek/OpenAI 异步调用的向后兼容性。
2. **物理规则 Prompt 构建 (`knowledge.py`)**:
   - BOagent 中的半导体物理启发式规则（CBO/VBO 计算、电子阻挡层 $0.5\text{ eV}$ 偏移限额）均直接继承自 PVK-LLM 的领域建模。
3. **数据集集成 (`datasets/perovskite`)**:
   - 本地 `datasets/perovskite/` 中的能带匹配与缺陷掺杂数据集直接对接了 PVK-LLM 的实验 Benchmark。
