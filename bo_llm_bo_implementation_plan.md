# BO/LLM-BO Agent Demo Implementation Plan

## 1. 计划目标

本计划从 `bo_llm_bo_product_agent_rd_v_0.md` 生成，目标是把当前 RD 收敛成一个可执行的 leader demo 实施路径。

当前版本不做真实科研可用产品，也不承诺真实 BO 性能。P0 只交付一个本地 Streamlit 演示原型：读取现有 `demo_optimization_table.csv`，展示数据证据，生成 3-5 组下一轮实验建议，并用清晰标注说明哪些内容来自文献抽取、人工整理或 demo 模拟。

## 2. 审查后关键决策

### 2.1 范围决策

- P0 是 leader demo，不是 MVP 平台。
- P0 只支持本地 CSV，不做 PDF 抽取、ELN/LIMS、数据库/API。
- P0 使用规则推荐或 mock acquisition score，不把 RandomForest/BO 包装成已验证能力。
- P0 的多智能体是产品化 pipeline 展示，不实现真实 agent runtime。
- P0 的趋势曲线是 workflow simulation，不展示真实算法优于 baseline 的结论。

### 2.2 可信边界

- `literature_extracted` 只能表示文献/Excel 抽取记录，不等于同一体系下可直接比较的 BO 训练数据。
- `manual_labeled` 表示人工补齐字段，必须在 UI 中标注。
- `simulated_for_demo` 只用于演示流程，不进入真实最佳值或真实性能提升声明。
- 所有 `predicted_PCE`、`acquisition_score`、趋势曲线和 PipDI 推荐都必须显示 demo/simulated 标记。
- 每条推荐理由必须拆成数据依据、规则依据、领域假设和待验证风险，不能把 LLM 假设写成确定科学结论。

## 3. P0 交付范围

### 3.1 用户路径

P0 demo 压缩成 4 个线性场景：

1. **Setup**：展示预填任务和优化目标。
2. **Data + Analysis**：读取 CSV，展示数据摘要、当前最佳、passivator 分布和数据健康状态。
3. **Recommendation + Plan**：生成 3-5 组下一轮推荐，并展开实验方案、理由、风险、假设。
4. **Simulated Feedback Loop**：点击模拟回填，展示 workflow trajectory 和 best-so-far 状态变化。

### 3.2 P0 必须实现

- `app.py` 可通过 `streamlit run app.py` 启动。
- `requirements.txt` 固定最小依赖。
- 能读取 `demo_optimization_table.csv`。
- 能处理当前 CSV 字段：`experiment_id`、`passivator_combo`、`has_3MTPAI`、`has_PDAI2`、`has_EDAI2`、`has_PipDI`、`PCE_percent`、`Voc_V`、`Jsc_mA_cm2`、`FF_percent`、`data_type`、`evidence_text`。
- 能展示 Total Records、Best PCE、Mean PCE、PipDI missing、mock/simulated 状态。
- 能生成 3-5 张推荐卡片，覆盖 `Exploitation`、`Exploration`、`Control`。
- 每张推荐卡片包含配方、操作步骤、推荐理由、风险、假设、证据来源、demo 标记。
- 页面顶部或侧边持续展示 `Demo only / Not experimentally validated`。
- 推荐生成失败或数据不足时，用规则 fallback 展示可讲的降级结果，而不是报错栈。

### 3.3 P0 明确不做

- 不做真实 PDF/论文自动抽取。
- 不做真实 RandomForest BO 或多目标 BO。
- 不做真实 LLM 调用或多 agent 编排。
- 不做企业权限、审计日志、租户隔离。
- 不做导出功能，除非时间充足且导出文件包含不可删除 disclaimer。
- 不做跨材料体系泛化。

## 4. 推荐文件结构

```text
BOagent/
  app.py
  requirements.txt
  demo_optimization_table.csv
  bo_llm_bo_product_agent_rd_v_0.md
  bo_llm_bo_implementation_plan.md
```

P0 保持单文件 Streamlit 应用即可。只有当 `app.py` 明显过长时，再拆成 `data_loader.py`、`recommendation.py`、`ui_components.py`。

## 5. 实施单元

### Unit 1: Streamlit Skeleton

**目标**：先保证本地应用能启动，并形成 4 个线性 demo 场景。

**文件**：
- `app.py`
- `requirements.txt`

**实现内容**：
- 页面标题：`PVK-BO Agent Demo`
- 全局 disclaimer：`Demo workflow only. Not experimentally validated.`
- 侧边栏显示 demo progress：Setup、Data Analysis、Recommendation、Feedback Loop。
- 使用 tabs 或 sections，不做复杂路由。

**验收**：
- `streamlit run app.py` 能启动。
- 页面无数据也能显示基本结构。

### Unit 2: Data Loader And Data Health

**目标**：读取当前 CSV，并诚实展示数据可用性。

**文件**：
- `app.py`

**实现内容**：
- 读取 `demo_optimization_table.csv`。
- 将 `PCE_percent` 映射为展示用 `PCE`。
- 统计记录数、best PCE、mean PCE、各 passivator 出现次数。
- 检查 PipDI 是否存在真实样本。
- 展示 Data Health 面板：
  - loaded records
  - missing PipDI evidence
  - non-numeric or mixed concentration text
  - literature vs manual vs simulated data counts

**验收**：
- 当前 CSV 24 条记录可加载。
- 页面能识别 best PCE 为 26.3 附近。
- PipDI 显示为缺失真实样本或 demo-only exploration candidate。

### Unit 3: Analysis View

**目标**：让 leader 看到推荐不是凭空生成，而是先理解数据。

**文件**：
- `app.py`

**实现内容**：
- 展示当前最佳实验卡片。
- 展示 passivator 分布。
- 展示高性能 seed 列表。
- 展示 evidence text 的可展开片段。
- 明确说明跨体系 PCE 不可直接当作严格因果比较。

**验收**：
- 任意高性能实验能追溯到 `experiment_id` 和 `evidence_text`。
- 页面有文字解释：当前数据用于 demo grounding，不用于真实科研结论。

### Unit 4: Rule-Based Recommendation Engine

**目标**：用可解释规则生成 3-5 个推荐，而不是假装真实 BO 已成立。

**文件**：
- `app.py`

**推荐策略**：
- `Exploitation`：靠近当前 high-performance seed 的组合，例如 3MTPAI + PDAI2 或 EDAI2-based 微调。
- `Exploration`：加入 PipDI，但标注 `demo-only / no real evidence in current dataset`。
- `Control`：去掉或单独测试某个 passivator，用于判断贡献。
- `Balanced`：保留高潜力分子，同时只改变一个变量。

**每条推荐包含**：
- `experiment_id`
- `recommendation_type`
- passivator combination
- fixed process assumptions
- `demo_score` 或 `mock_acquisition_score`
- supporting records
- domain hypothesis
- risks
- validation_required

**验收**：
- 点击按钮后稳定生成 3-5 条推荐。
- 推荐中至少包含 1 个 exploration 和 1 个 control。
- PipDI 推荐必须显示无真实样本证据。

### Unit 5: Experiment Plan Cards

**目标**：把推荐转换成实验员能看懂的操作方案，同时保留证据边界。

**文件**：
- `app.py`

**实现内容**：
- 推荐卡片支持展开。
- 展开内容按顺序展示：
  1. 配方与固定工艺条件
  2. 操作步骤
  3. 推荐理由
  4. 风险
  5. 待验证假设
  6. 证据来源和数据类型
- 风险等级使用文字 + 颜色，不只靠颜色表达。

**验收**：
- 每条推荐至少有一条 supporting record 或明确标注 `demo-only hypothesis`。
- 所有未验证科学表述都写成 hypothesis，不写成 confirmed conclusion。

### Unit 6: Simulated Feedback Loop

**目标**：展示闭环形态，而不是证明算法收益。

**文件**：
- `app.py`

**实现内容**：
- 按钮：`Simulate Experiment Feedback`
- 展示一个合成 workflow trajectory：
  - initial best
  - recommended batch
  - simulated feedback
  - updated best-so-far
- 图表标题和 caption 明确写：`Synthetic walkthrough, not experimental validation.`
- 不展示 PVK-BO 击败 General LLM/Random/BO 的性能结论。

**验收**：
- 点击后图表和状态能更新。
- 图表不暗示真实算法优越性。

## 6. P1 / Future Backlog

### P1: 更像产品的 Demo

- 合并 UI 成 3 个主场景，减少页面切换。
- 增加 Agent Pipeline 组件：Data Check、Domain Hypothesis、Candidate Scoring、Critic Review、Experiment Plan。
- 增加导出 Markdown/CSV，但导出物必须内置 disclaimer、data_type 和 evidence_level。
- 增加更完整的 loading、empty、warning、error 状态。

### P1: 数据加工

- 生成一个 `demo_feature_table.csv`，把当前布尔特征和文本浓度转成 demo 可用的数值字段。
- 明确同一 perovskite system / device configuration 的子集，避免跨体系直接比较。
- 手工补齐 PipDI demo 样本，标注为 `simulated_for_demo`。

### Future: 真正的产品能力

- PDF/论文自动抽取。
- ELN/LIMS/数据库/API 接入。
- 真实 RandomForest/GP BO、多目标 BO、约束 BO。
- 真实 LLM/Agent 编排。
- 企业权限、审计、数据隔离。
- 跨材料体系 schema 配置和插件式优化器。

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| 当前数据不支持真实 BO | P0 明确使用 rules/mock acquisition，不宣称真实 BO 性能 |
| PipDI 无真实样本 | PipDI 只作为 demo-only exploration candidate |
| leader 追问数值来源 | 每个数值分为 literature extracted、manual labeled、simulated |
| 推荐理由像 LLM 幻觉 | 推荐卡片必须显示 supporting records、规则依据和待验证假设 |
| 3-5 分钟讲不完 | 主路径压缩为 4 个线性场景，高级内容折叠 |
| Streamlit 现场报错 | 所有数据加载、推荐生成、空筛选都必须有 fallback |

## 8. 验收清单

P0 完成后必须逐项检查：

- [ ] `streamlit run app.py` 可启动。
- [ ] 当前 CSV 可加载并显示 24 条记录。
- [ ] 页面显示 demo/not experimentally validated 标记。
- [ ] Data Health 面板显示 PipDI 缺失、浓度字段混合文本等限制。
- [ ] 能展示 best PCE、mean PCE、passivator 分布。
- [ ] 能生成 3-5 条推荐。
- [ ] 推荐覆盖 exploitation、exploration、control。
- [ ] 每条推荐都包含实验方案、理由、风险、假设和证据边界。
- [ ] PipDI 推荐明确显示 demo-only/no real sample。
- [ ] 模拟趋势图明确标注 synthetic workflow，不作为算法收益证明。
- [ ] 3-5 分钟演示脚本可以按 Setup → Data → Recommendation → Feedback Loop 讲完。

## 9. 建议开发顺序

1. 建 `requirements.txt` 和 `app.py` 骨架。
2. 接入 CSV 读取和数据健康面板。
3. 做 Data Analysis 卡片与图表。
4. 做规则推荐生成器。
5. 做推荐详情和实验方案卡片。
6. 做模拟回填与趋势图。
7. 补齐 loading/error/fallback 状态。
8. 走一遍 3-5 分钟 demo 脚本，删掉不服务主线的组件。

