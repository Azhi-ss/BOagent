# BO/LLM-BO 配方优化产品 Agent RD（需求文档）v0

## 0. 文档状态
- 当前状态：根据 leader 反馈，RD 从“可用产品 MVP”调整为“给 leader 看的一版可演示 Demo 原型”
- Demo 目标：按现有图中钙钛矿配方优化案例，跑出一个能展示端到端流程的 Agent demo
- 目标观众：leader，重点是快速判断方向是否值得继续投入，而不是验证真实科研可用性
- 首个 Demo 场景：钙钛矿太阳能电池钝化配方优化，尽量复刻图中 3MTPAI、PDAI₂、EDAI₂、PipDI 等候选钝化剂/配方示例
- 首要展示价值：让 leader 看到 Agent 如何从论文/实验描述出发，生成下一轮完整实验方案，并解释推荐理由、风险和假设
- 第一用户视角：实验员
- 当前数据状态：暂无内部实验数据，允许使用论文数据、手工整理数据或半合成 mock 数据支撑 demo
- Demo 成功标准：能完整演示“初始化 → 筛选 → 优化推荐 → 实验方案输出 → 结果回填/趋势展示”的闭环，不要求真实可投入实验使用
- 重要边界：所有科学结论和性能预测在 Demo 中都应标注为“模拟/示例”，避免被理解为真实验证结果

---

## 1. 产品一句话定位
面向钙钛矿太阳能电池研发中的钝化配方与关键工艺参数优化，构建一个帮助实验员基于论文/历史实验数据快速生成下一轮可执行实验方案的 BO/LLM-BO Agent。该 Agent 不只给出“下一个点”，还要解释推荐理由、潜在风险和背后的科学假设，从而缩短实验探索周期。

待进一步确认：
- 首个 demo 是否聚焦 FAPbI₃ / FA-Cs / MAPbI₃ 中某一个体系？
- 优化对象是钝化剂选择、浓度、混合比例，还是钝化处理工艺？
- demo 的成功标准是“看起来合理”，还是要能与论文 reported optimum 对齐？

---

## 2. 业务背景与问题定义

### 2.1 当前业务痛点
首要痛点：钙钛矿器件优化周期长，实验员需要在钝化剂、浓度、溶剂、处理时间、退火温度、旋涂参数等多维空间中反复试错。

具体问题：
1. 论文中有大量配方与工艺经验，但信息分散在文本、图表和 supporting information 中，难以直接转化为下一轮实验方案。
2. 实验员往往需要依赖人工经验判断“下一步做什么”，缺少系统化探索策略。
3. 纯 BO 在冷启动时数据不足；纯 LLM 能读论文但缺乏严谨的数值优化与不确定性建模。
4. 实验员需要的是可执行方案，而不是抽象建议：包括试剂、浓度、步骤、推荐理由、风险和验证假设。

### 2.2 MVP 要解决的真实问题
在没有内部实验数据的情况下，MVP 先解决：

> 给定一批论文中抽取的钙钛矿钝化实验数据，Agent 能否帮助实验员快速生成下一轮合理、可执行、可解释的钝化实验方案，从而模拟缩短工艺探索周期。

待量化：
- demo 中要模拟多少轮实验？
- 每轮推荐几组实验？
- 以什么 baseline 对比：人工规则、随机搜索、网格搜索、普通 BO，还是论文中的实验路径？
- “缩短周期”在 demo 中如何体现：更少实验次数达到论文最优附近？更快找到高 PCE 区域？还是减少人工整理时间？

---

## 3. 目标用户与使用角色

### 3.1 第一用户：实验员
实验员的核心诉求不是看复杂模型，而是快速得到下一轮能做的实验方案，并理解为什么值得做。

实验员关心：
- 今天/下一轮应该做哪几组实验？
- 每组实验具体怎么配、怎么处理、怎么测？
- 为什么推荐这些组合，而不是别的组合？
- 哪些风险需要注意？例如溶解性、浓度过高、工艺窗口窄、与已有结果冲突。
- 实验失败后应该如何调整下一轮？

### 3.2 非第一优先用户
以下角色暂不作为 MVP 的主用户，但后续可能需要支持：
- PI / 课题负责人：看优化进展、机制假设和阶段报告。
- 数据科学家：调试 BO/LLM-BO 策略。
- 企业研发经理：关注周期缩短和项目 ROI。

### 3.3 MVP 用户旅程
1. 实验员创建“钙钛矿钝化优化任务”。
2. 上传论文数据表或由系统从论文/方法部分抽取实验记录。
3. 定义目标：例如提高 PCE，或兼顾 PCE 与稳定性。
4. Agent 识别可优化变量：钝化剂、浓度、溶剂、处理顺序、退火温度等。
5. Agent 整理已有实验规律与数据空白。
6. Agent 推荐下一轮若干组实验。
7. Agent 输出完整实验步骤、理由、风险和待验证假设。
8. 用户回填实验结果或用论文中的 held-out 数据模拟回填。
9. Agent 更新推荐并生成阶段总结。

---

## 4. 目标场景与范围

### 4.1 Demo 场景
根据 leader 反馈，首个版本直接按现有图中的钙钛矿配方优化案例构建 demo：

> 面向钙钛矿太阳能电池钝化处理，Agent 根据已有论文/实验记录，围绕 3MTPAI、PDAI₂、EDAI₂、PipDI 等候选钝化剂，推荐下一轮配方组合与实验方案。

### 4.2 Demo 叙事路径
Demo 不追求真实工业可用，而是追求“看得懂、能跑通、能体现产品价值”。建议按图中三步组织：

1. **Step I. Initialization**  
   用户输入一个自然语言任务：例如“请设计单结正常带隙钙钛矿器件的钝化配方实验”。Agent 输出一个初始实验方案。

2. **Step II. Screening**  
   用户上传或输入一批实验结果。Agent 分析已有结果，识别可能有效的钝化剂与组合，例如 3MTPAI、PDAI₂、EDAI₂、PipDI。

3. **Step III. Optimization**  
   Agent 结合 LLM 先验和 BO 优化策略，推荐下一轮配方比例/浓度/工艺条件，并输出完整实验步骤、推荐理由、风险和假设。

### 4.3 Demo 变量范围建议
为避免 demo 过重，建议变量控制在 4–6 个以内：
- 钝化剂候选：3MTPAI、PDAI₂、EDAI₂、PipDI
- 每种钝化剂是否加入：0/1
- 浓度或相对比例：低/中/高，或 0–12 mM
- 溶剂：先固定，避免变量过多
- 退火温度/时间：可固定或作为 1–2 个离散参数
- 优化目标：PCE 为主，Voc/FF/稳定性作为解释或辅助展示项

### 4.4 Demo 成功标准
Demo 的验收不以真实实验有效为标准，而以端到端可展示为标准：
1. 能导入或展示一份结构化实验数据表。
2. 能从实验结果中识别“当前最优”和“有潜力区域”。
3. 能推荐下一轮 3–5 组实验。
4. 每组实验都包含可执行步骤、推荐理由、风险和假设。
5. 能展示优化趋势图，例如 PCE 随迭代轮次提升。
6. 能体现多智能体分工，而不是单轮聊天生成。

### 4.5 暂不做范围
Demo 暂不覆盖：
- 真实实验自动闭环。
- 真实论文级复现准确性保证。
- 真实 PCE 预测能力保证。
- 大规模材料体系泛化。
- 企业级 LIMS/ELN 集成。
- 严格的科学发现验证。

---

## 5. 核心功能需求

### 5.1 任务创建与实验空间定义
用户需要定义：
- 优化目标：单目标或多目标
- 可控变量：类别变量、连续变量、离散变量、条件变量
- 变量范围：上下限、候选列表、步长
- 硬约束：不可行组合、工艺限制、安全限制、成本限制
- 软偏好：优先考虑稳定、常见、低成本、易采购方案

待确认：
- 用户是否必须自己定义变量空间，还是 Agent 可以从历史数据/文本中自动建议变量空间？

### 5.2 历史实验数据导入与结构化
输入来源候选：
- Excel/CSV
- ELN/LIMS
- PDF/论文/专利
- 自然语言实验记录
- 数据库/API

Agent 能力：
- 字段识别
- 单位归一化
- 缺失值/异常值提示
- 配方变量抽取
- 工艺步骤抽取
- 结果指标抽取
- 数据质量评分

### 5.3 先验知识注入
知识类型：
- 参数组合经验
- 常见化学体系
- 禁忌组合
- 文献规律
- 物理/化学机制假设
- 企业内部历史项目经验

Agent 应输出：
- 哪些变量更值得优先探索
- 哪些区域不建议探索
- 为什么某个候选点有希望
- 推荐背后的假设

### 5.4 BO/LLM-BO 优化推荐
能力候选：
- 冷启动推荐
- 批量推荐下一轮实验
- 约束 BO
- 多目标 BO
- 类别变量/混合变量优化
- 不确定性估计
- Exploration / Exploitation 权衡解释
- 与普通 BO、人工经验推荐对比

### 5.5 多智能体协作机制
候选 Agent：
- Data Agent：清洗、结构化、质量检查
- Domain Agent：提取领域先验与机制假设
- Optimizer Agent：执行 BO/LLM-BO 推荐
- Critic Agent：检查推荐是否违反约束或缺乏科学合理性
- Experiment Planner Agent：生成可执行实验方案
- Report Agent：输出阶段报告与最终总结

### 5.6 人机协同与审批
待确认：
- 推荐方案是否需要人工审批后才能进入执行？
- 是否需要解释每个推荐点的依据、风险、预期收益？
- 是否需要用户可以手动锁定/排除某些候选方案？

### 5.7 实验结果回填与闭环迭代
能力：
- 录入实验结果
- 更新优化模型
- 记录每轮推荐理由与实验结果
- 自动判断是否继续优化
- 输出优化轨迹与收敛分析

---

## 6. 输出物设计

### 6.1 每轮推荐输出
每个推荐实验点应包含：
- 实验编号
- 推荐配方/工艺参数
- 具体实验步骤
- 预测目标表现，例如 PCE 提升方向或高潜力区间
- 不确定性或置信说明
- 推荐理由
- 风险提示
- 需要验证的科学假设
- 与已有最佳方案的差异
- 是否适合作为探索点或利用点

### 6.2 推荐输出示例结构
| 字段 | 内容 |
|---|---|
| 推荐实验 | Exp-Next-01 |
| 钝化剂 | 待定 |
| 浓度 | 待定 |
| 溶剂 | 待定 |
| 工艺步骤 | 待定 |
| 推荐理由 | 基于历史高 PCE 区域、相似分子先验或 BO 不确定性 |
| 风险 | 溶解性、重复性、浓度过高、与基底/薄膜体系不兼容 |
| 假设 | 该处理可能降低界面缺陷或改善能级匹配 |
| 验证方式 | PCE、Voc、PL/TRPL、稳定性或缺陷相关表征 |

### 6.3 阶段报告
包括：
- 当前最优方案
- 实验轮次与样本数
- 指标提升幅度
- 参数重要性分析
- 有效/无效区域总结
- 下一阶段建议
- 哪些推荐来自数据证据，哪些来自 LLM 领域假设

---

## 7. 成功指标与验收标准

### 7.1 业务指标
待量化：
- 实验次数减少比例
- 优化周期缩短比例
- 最优指标提升幅度
- 人工分析时间减少比例
- 历史数据复用率
- 用户采纳率

### 7.2 模型/算法指标
候选：
- Best-so-far improvement
- Regret
- Hit rate of promising candidates
- Constraint violation rate
- Recommendation acceptance rate
- Compared with random search / DOE / human baseline / vanilla BO

### 7.3 产品体验指标
候选：
- 创建任务耗时
- 数据导入成功率
- 推荐解释满意度
- 回填数据耗时
- 报告生成可用率

---

## 8. 非功能需求

### 8.1 安全与合规
- 企业数据隔离
- 权限管理
- 审计日志
- 数据脱敏
- 模型输出可追溯

### 8.2 可解释性
- 每个推荐点必须给出依据
- 区分数据驱动证据、文献先验、LLM 假设与算法探索因素
- 不允许把未经验证的推断包装成确定结论

### 8.3 可扩展性
- 支持不同材料/化学/配方体系的变量 schema 配置
- 支持插件式优化器
- 支持知识库扩展

---

## 9. Demo / MVP 边界

### 9.1 Demo 必须有
1. 一个清晰的钙钛矿钝化配方优化任务入口。
2. 一份可展示的实验数据表，来源可以是论文手工整理或半合成 mock 数据。
3. 图中类似的候选分子/配方体系：3MTPAI、PDAI₂、EDAI₂、PipDI。
4. Agent 对已有结果的分析：当前最佳方案、可能有效变量、可能无效区域。
5. 下一轮 3–5 组推荐实验。
6. 每组推荐包含完整实验方案、推荐理由、风险、科学假设。
7. 一个简单优化趋势展示，例如 PCE 或综合评分随迭代轮次上升。
8. 多智能体流程展示：Data Agent、Domain Agent、Optimizer Agent、Critic Agent、Experiment Planner Agent。

### 9.2 面向 leader 的 Demo 重点
Leader 观看 demo 时，重点不是判断推荐方案是否已经能真实指导实验，而是判断：
1. 这个产品形态是否能讲清楚业务价值。
2. 多智能体 + LLM + BO 的组合是否有差异化。
3. 当前 demo 是否能复刻图上的产品故事。
4. 后续是否值得投入更多真实数据和工程资源。

因此 demo 应优先保证：
- 流程顺畅，不要在现场暴露太多配置细节。
- 输出结果直观，最好一眼能看到“推荐了什么、为什么推荐、风险是什么”。
- 页面像产品，不像纯算法 notebook。
- BO/优化曲线可以简化，但必须有“越迭代越好”的视觉反馈。
- 所有模拟结果要低调标注为 demo/sample，避免被追问真实性时失控。

### 9.3 Demo 可以 mock / 简化
1. PCE 预测值可以由 mock 函数或半合成数据生成，但界面上需标注“模拟示例”。
2. BO 优化可以先用简化实现，不必追求最优算法性能。
3. 论文抽取可以先手工整理成 CSV，不必第一版做完整 PDF 自动解析。
4. 分子结构图可以先使用静态图片或占位图。
5. 实验结果回填可以用预置数据模拟。
6. 对比曲线可以用预置的 PVK-BO、General LLM、BO 等曲线模拟展示。

### 9.4 Demo 不应假装已经做到
1. 不应声称推荐方案已经被真实实验验证。
2. 不应声称模型能真实预测 PCE。
3. 不应声称已经支持所有钙钛矿体系。
4. 不应声称可直接替代实验员判断。

### 9.5 Demo 数据策略
当前没有内部实验数据，因此 demo 建议采用：
1. 按图中配方体系构造一份 20–50 条实验记录。
2. 字段包括：钙钛矿体系、钝化剂组合、浓度/比例、溶剂、旋涂参数、退火条件、PCE、Voc、Jsc、FF、稳定性标签。
3. 数据分成初始数据和模拟回填数据。
4. Agent 第一轮只看到部分数据，推荐下一轮实验。
5. 用户点击“回填结果”后，系统展示下一轮 PCE 提升和推荐更新。

---

## 10. 当前上传数据可用性评估

### 10.1 数据概况
用户上传的 `副本副本数据标注new.xlsx` 已完成第一版 demo 数据抽取。

抽取结果：
- 原始 Sheet1 共 1158 条记录。
- 第一版筛选出 75 条含 passivation 相关关键词的记录（为避免重复，优先使用前半部分非重复块）。
- 抽取出 24 条可进入 Demo Optimization Table 的核心记录。
- 其中包含：3MTPAI 2 条、PDAI2 5 条、EDAI2 19 条。
- 未在上传数据中检索到 PipDI，因此 PipDI 若要完全复刻图中案例，需要后续单独 mock 或补充论文数据。
- 当前抽取表最高 PCE 为 26.3%，可作为 demo 中的 high-performance seed。

已生成两个文件：
- `perovskite_agent_demo_extracted_data.xlsx`：包含 Summary、Demo_Optimization_Table、Raw_Filtered_Records、Streamlit_Field_Map。
- `demo_optimization_table.csv`：可直接给 Streamlit 读取的结构化表。

### 10.2 对当前 Demo 的价值
该数据对 demo 有用，但更适合作为“论文/历史实验库”和“文本抽取来源”，不适合直接作为 BO 配方优化表。

适合用于：
1. 展示 Agent 读取历史钙钛矿实验记录。
2. 展示从长文本实验方法中抽取钝化剂、溶剂、旋涂、退火等工艺信息。
3. 展示按 PCE/Voc/FF 筛选高性能样本。
4. 构建 demo 的初始实验数据池。
5. 补充图中 3MTPAI、PDAI2、EDAI2 的真实文本依据。

暂不适合直接用于：
1. 严格 BO 优化，因为缺少结构化的配方变量列，例如 3MTPAI_mM、PDAI2_mM、EDAI2_mM、PipDI_mM。
2. 严格复刻图中混合配方比例，因为当前数据更多是非结构化工艺文本和 HTL/器件性能记录。
3. 真实实验决策，因为数据来源、重复实验、体系一致性和变量控制仍需清洗。

### 10.3 推荐处理方式
针对 leader demo，建议将该文件加工成两层数据：

**第一层：Raw Literature Records**
- 保留原始字段，用于展示 Data Agent 正在读取论文/历史实验记录。

**第二层：Demo Optimization Table**
从原始文本中抽取或手工标注出 demo 所需结构化字段：
- experiment_id
- perovskite_system
- device_configuration
- htl_name
- passivator_3MTPAI_mM
- passivator_PDAI2_mM
- passivator_EDAI2_mM
- passivator_PipDI_mM
- solvent
- spin_speed_rpm
- spin_time_s
- anneal_temp_C
- anneal_time_min
- PCE
- Voc
- Jsc
- FF
- source
- evidence_text
- is_mock

### 10.4 数据使用策略
1. 先从上传数据中筛选出包含 3MTPAI / PDAI2 / EDAI2 / passivation 的记录。
2. 对这些记录进行 LLM/规则抽取，形成 10–20 条半结构化真实样本。
3. 对缺失的 PipDI 和混合比例部分，用 mock 数据补齐。
4. 在界面中标注数据类型：`literature extracted`、`manual labeled`、`simulated for demo`。
5. BO/LLM-BO 推荐基于结构化 demo table 运行，而不是直接基于原始长文本运行。

---

## 11. Demo 实施方案：步骤与模块设计

### 11.1 项目本质
当前项目不是先做一个真实可投产的材料优化平台，而是做一个面向 leader 演示的 Streamlit 本地小应用。

核心目标：
> 让用户看到一个钙钛矿配方优化 Agent 如何读取已有实验数据，分析当前结果，推荐下一轮实验，并给出推荐理由、风险和科学假设。

### 11.2 Demo 主流程
建议按 5 个页面/步骤实现：

1. **Task Setup：任务初始化**
   - 输入：自然语言任务，例如“优化钙钛矿太阳能电池钝化配方，提高 PCE”。
   - 输出：优化目标、候选钝化剂、变量空间、固定实验条件。

2. **Data Ingestion：数据读取与展示**
   - 输入：`demo_optimization_table.csv`。
   - 输出：实验数据表、字段说明、数据来源标签。

3. **Data Analysis：已有结果分析**
   - 分析当前最佳 PCE、平均 PCE、不同钝化剂表现、数据缺口。
   - 输出：当前最佳方案、潜力分子、待探索区域。

4. **Optimization Recommendation：下一轮推荐**
   - 输入：已有结构化实验数据。
   - 输出：3–5 组下一轮推荐实验。
   - 推荐逻辑可使用简化 BO / 启发式规则 / mock acquisition score。

5. **Experiment Plan：实验方案生成**
   - 对每组推荐实验生成：配方、步骤、推荐理由、风险、待验证假设、预期观察指标。

6. **Iteration View：模拟回填与趋势展示**
   - 用户点击“模拟回填结果”。
   - 系统展示 PCE 趋势曲线和下一轮优化方向。

### 11.3 模块设计

#### Module 1: Data Loader
职责：读取 CSV/XLSX，统一字段，处理缺失值。
输入：`demo_optimization_table.csv`。
输出：标准 DataFrame。

关键字段：
- experiment_id
- passivator_system
- passivator_3MTPAI_mM
- passivator_PDAI2_mM
- passivator_EDAI2_mM
- passivator_PipDI_mM
- PCE
- Voc
- Jsc
- FF
- evidence_text
- data_type

#### Module 2: Data Agent
职责：总结已有数据。
能力：
- 找当前最佳实验。
- 统计不同钝化剂出现次数和性能分布。
- 识别缺失字段和数据质量问题。
- 给出数据是否足够支持优化的判断。

#### Module 3: Domain Agent
职责：把材料/钙钛矿领域语言补上。
能力：
- 解释 3MTPAI、PDAI2、EDAI2 等钝化剂可能作用。
- 输出机制假设。
- 输出实验风险，例如溶解性、过量添加、薄膜形貌扰动、重复性风险。

#### Module 4: Optimizer Agent
职责：推荐下一轮实验点。
Demo 阶段可简化为：
- 从当前高 PCE 区域附近做 exploitation。
- 对未充分探索的组合做 exploration。
- 给每个候选点一个 acquisition score。
- 输出 3–5 个候选方案。

#### Module 5: Critic Agent
职责：检查推荐方案是否过于激进或不合理。
检查项：
- 浓度是否过高。
- 是否同时改变太多变量。
- 是否缺少对照组。
- 是否与已有高风险结果冲突。

#### Module 6: Planner Agent
职责：把推荐点转成实验员可执行方案。
输出：
- 实验名称。
- 配方和浓度。
- 操作步骤。
- 测试指标。
- 推荐理由。
- 风险。
- 假设。

#### Module 7: UI Layer / Streamlit
职责：把流程包装成一个顺滑 demo。
页面建议：
- 首页：任务说明和图示流程。
- 数据页：实验数据表与筛选。
- 分析页：当前最佳、分子表现、数据质量。
- 推荐页：下一轮实验方案卡片。
- 趋势页：模拟优化曲线。

### 11.4 推荐开发顺序
1. 先做 Streamlit 页面骨架。
2. 接入 `demo_optimization_table.csv`。
3. 做数据分析卡片和当前最佳实验展示。
4. 做规则版推荐器，不急着接复杂 BO。
5. 做实验方案卡片。
6. 做模拟趋势图。
7. 最后再包装多智能体叙事。

### 11.5 Leader 演示话术
演示时不要说“我们已经能真实预测 PCE”，而要说：
> 这是一个本地 demo，目的是验证产品流程。现在数据来自论文整理和示例化加工，优化结果用于演示工作流。后续接入真实实验数据后，可以替换推荐器和评估模块。

---

## 12. 页面交互设计

### 12.1 Demo 形态
- 技术形态：本地 Streamlit 小应用
- 启动方式：`streamlit run app.py`
- 目标观众：leader
- 演示时长：3–5 分钟
- 核心体验：像产品，而不是像 notebook

### 12.2 页面结构

#### Page 1: Overview / Task Setup
目标：让 leader 立刻明白这个 Agent 要解决什么问题。

页面内容：
- 产品标题：PVK-BO Agent / Perovskite Formulation Optimization Agent
- 一句话说明：基于实验数据和 LLM-BO 推荐下一轮钙钛矿钝化配方实验
- 自然语言任务输入框
- 初始化按钮：`Initialize Optimization Task`

默认输入示例：
> 请设计一个单结正常带隙钙钛矿太阳能电池的钝化配方优化实验，目标是提高 PCE，并兼顾 Voc、FF 和稳定性。

点击后输出：
- 优化目标：PCE 最大化，Voc/FF/稳定性作为辅助指标
- 候选钝化剂：3MTPAI、PDAI2、EDAI2、PipDI
- 可优化变量：钝化剂组合、浓度、退火温度、退火时间
- 固定条件：器件结构、溶剂、旋涂流程可在 demo 中固定
- 数据来源：论文/历史实验整理 + demo mock 补齐

#### Page 2: Data Library
目标：展示 Agent 不是凭空生成，而是基于实验数据。

页面组件：
- 数据表预览
- passivator 筛选器
- PCE 范围筛选器
- 数据类型筛选器：`literature_extracted` / `manual_labeled` / `simulated_for_demo`
- 原始 evidence text 展示区

输出：
- 当前加载记录数
- 当前筛选结果
- 每条实验记录的 PCE、Voc、Jsc、FF、passivator system

#### Page 3: Data Analysis
目标：展示 Data Agent 对已有实验结果的理解。

页面组件：
- KPI 卡片：Total Records、Best PCE、Mean PCE、Main Passivator、Missing Candidate
- 当前最佳实验卡片
- 不同钝化剂样本数柱状图
- 不同钝化剂 PCE 分布图
- 数据质量提醒框

输出示例：
- 当前最佳 PCE = 26.3%
- EDAI2 样本最多，PDAI2 次之，3MTPAI 较少
- PipDI 当前没有真实样本，只作为 exploration candidate
- 当前数据缺少完整结构化浓度字段，需要 demo mock 补齐

#### Page 4: Recommendation
目标：展示 Optimizer Agent 生成下一轮实验推荐。

页面组件：
- 推荐数量选择器：3 / 5
- Exploration 权重滑块：`beta`
- 按钮：`Generate Next Experiments`
- 推荐实验卡片列表

每张卡片包含：
- Experiment ID
- Recommendation Type：Exploitation / Exploration / Balanced / Control
- 配方和浓度
- 工艺条件
- Predicted PCE
- Uncertainty
- Acquisition Score
- 推荐理由
- 风险等级
- 科学假设
- 重点观测指标

#### Page 5: Experiment Plan
目标：把推荐结果转成实验员可执行方案。

页面组件：
- 推荐实验详情卡片
- 每组实验的具体步骤
- 对照组建议
- 表征建议：PCE、Voc、Jsc、FF、PL/TRPL、稳定性
- 一键导出按钮：导出推荐实验方案 CSV / Markdown

#### Page 6: Iteration Trend
目标：展示“越迭代越好”的产品价值感。

页面组件：
- 模拟优化曲线
- 对比曲线：PVK-BO Agent、General LLM、Random Search、Vanilla BO
- 当前 best-so-far PCE
- 回填按钮：`Simulate Experiment Feedback`

页面标注：
> Simulated demo trajectory for workflow illustration only. Not experimentally validated.

---

## 13. 数据 Schema

### 13.1 主数据表：Demo Optimization Table
Streamlit 第一版读取 `demo_optimization_table.csv`。

| 字段名 | 类型 | 是否必须 | 示例 | 用途 |
|---|---|---:|---|---|
| experiment_id | string | 是 | Exp-001 | 实验编号 |
| perovskite_system | string | 否 | FAPbI3-based | 钙钛矿体系说明 |
| device_configuration | string | 否 | n-i-p | 器件结构 |
| htl_name | string | 否 | Spiro-OMeTAD | 空穴传输层信息 |
| passivator_system | string | 是 | 3MTPAI + PDAI2 | 钝化剂组合描述 |
| passivator_3MTPAI_mM | float | 是 | 8.0 | BO 输入特征 |
| passivator_PDAI2_mM | float | 是 | 3.0 | BO 输入特征 |
| passivator_EDAI2_mM | float | 是 | 2.0 | BO 输入特征 |
| passivator_PipDI_mM | float | 是 | 0.0 | BO 输入特征 / demo exploration |
| solvent | string | 否 | IPA | 工艺展示字段 |
| spin_speed_rpm | float | 否 | 4000 | 工艺展示字段 |
| spin_time_s | float | 否 | 30 | 工艺展示字段 |
| anneal_temp_C | float | 是 | 100 | BO 输入特征 |
| anneal_time_min | float | 是 | 5 | BO 输入特征 |
| PCE | float | 是 | 24.2 | 主优化目标 |
| Voc | float | 否 | 1.15 | 辅助指标 |
| Jsc | float | 否 | 25.1 | 辅助指标 |
| FF | float | 否 | 0.82 | 辅助指标 |
| stability_label | string | 否 | medium | 辅助展示 |
| evidence_text | string | 否 | 原始实验方法片段 | 可追溯依据 |
| source | string | 否 | paper DOI / row id | 数据来源 |
| data_type | enum | 是 | literature_extracted | 数据可信度标记 |
| is_mock | bool | 是 | false | 是否为 demo mock 数据 |

### 13.2 data_type 枚举
- `literature_extracted`：从上传 Excel / 论文数据中抽取
- `manual_labeled`：人工补充结构化字段
- `simulated_for_demo`：为 demo 补齐的模拟记录

### 13.3 必须保证的最小字段
如果字段过多来不及清洗，第一版至少保证：
- experiment_id
- passivator_3MTPAI_mM
- passivator_PDAI2_mM
- passivator_EDAI2_mM
- passivator_PipDI_mM
- anneal_temp_C
- anneal_time_min
- PCE
- passivator_system
- evidence_text
- data_type
- is_mock

---

## 14. BO 推荐算法说明

### 14.1 第一版算法选择
Demo 第一版采用简化 BO：
- Surrogate model：`RandomForestRegressor`
- Uncertainty：使用多棵树预测结果的标准差近似
- Acquisition function：`predicted_PCE + beta × uncertainty`
- 推荐数量：默认 5 组

选择原因：
- 对小数据和 mock 数据更稳定
- 易实现，适合 Streamlit demo
- 能展示 BO 的核心思想：高预测值 + 高不确定性

### 14.2 输入特征
第一版 BO 使用以下特征：
- passivator_3MTPAI_mM
- passivator_PDAI2_mM
- passivator_EDAI2_mM
- passivator_PipDI_mM
- anneal_temp_C
- anneal_time_min

### 14.3 优化目标
主目标：
- PCE 最大化

辅助展示指标：
- Voc
- Jsc
- FF
- stability_label

第一版不做严格多目标优化，但在推荐理由和风险分析中体现辅助指标。

### 14.4 候选空间生成
候选浓度网格：
- 3MTPAI：0, 2, 4, 6, 8, 10, 12 mM
- PDAI2：0, 1, 2, 3, 4, 6 mM
- EDAI2：0, 1, 2, 3, 4, 6 mM
- PipDI：0, 0.5, 1, 2 mM

工艺网格：
- anneal_temp_C：80, 100
- anneal_time_min：5, 10

### 14.5 候选约束
为保证推荐方案像真实实验员能接受的方案，候选点需满足：
1. 总钝化剂浓度 > 0。
2. 总钝化剂浓度 ≤ 12 mM。
3. 同时加入的钝化剂数量 ≤ 3。
4. PipDI 只能作为低剂量探索，≤ 2 mM。
5. 不推荐与已有实验完全重复的点。
6. 至少保留 1 个 exploitation 点、1 个 exploration 点、1 个 control/ablation 点。

### 14.6 Acquisition Score
公式：

`acquisition_score = predicted_PCE + beta × uncertainty`

其中：
- `predicted_PCE`：代理模型预测值
- `uncertainty`：随机森林中不同树预测结果的标准差
- `beta`：探索权重，默认 0.5

解释：
- predicted_PCE 高：说明模型认为该配方有潜力
- uncertainty 高：说明该区域探索不足，可能有发现空间
- beta 越高：越偏 exploration

### 14.7 推荐类型划分
- `Exploitation`：靠近当前高 PCE 区域，主要做局部微调
- `Exploration`：包含 PipDI 或数据稀缺组合，探索未知区域
- `Balanced`：兼顾高潜力组分和少量新变量
- `Control`：用于判断某个钝化剂或组合是否真正有贡献

---

## 15. Agent 输入输出协议

### 15.1 Data Loader
输入：
- `demo_optimization_table.csv`

输出：
```json
{
  "dataframe": "standardized_experiment_dataframe",
  "load_status": "success",
  "record_count": 24,
  "required_fields_missing": []
}
```

### 15.2 Data Agent
输入：
- 标准化实验数据表

输出：
```json
{
  "best_experiment": {
    "experiment_id": "Exp-014",
    "PCE": 26.3,
    "passivator_system": "EDAI2-based"
  },
  "summary_metrics": {
    "total_records": 24,
    "mean_PCE": 23.1,
    "max_PCE": 26.3
  },
  "passivator_summary": [
    {
      "name": "EDAI2",
      "count": 19,
      "mean_PCE": 23.4,
      "max_PCE": 26.3
    }
  ],
  "data_quality_notes": [
    "PipDI is not found in the uploaded dataset.",
    "Some concentration fields are manually completed for demo."
  ]
}
```

### 15.3 Domain Agent
输入：
- passivator_summary
- candidate passivators
- data_quality_notes

输出：
```json
{
  "molecule_insights": {
    "3MTPAI": {
      "possible_role": "surface defect passivation and interface stability improvement",
      "risk": "high concentration may form insulating organic layer",
      "evidence_level": "limited in current dataset"
    },
    "PDAI2": {
      "possible_role": "diammonium-based interface passivation",
      "risk": "excessive amount may disturb film morphology",
      "evidence_level": "moderate"
    },
    "EDAI2": {
      "possible_role": "defect passivation and possible low-dimensional interface formation",
      "risk": "over-passivation may reduce charge transport",
      "evidence_level": "relatively stronger in current dataset"
    },
    "PipDI": {
      "possible_role": "exploration candidate",
      "risk": "not observed in current dataset; solubility and compatibility unknown",
      "evidence_level": "mock/demo only"
    }
  }
}
```

### 15.4 Optimizer Agent
输入：
- 标准化实验数据表
- 候选空间
- 约束条件
- beta
- n_recommendations

输出：
```json
{
  "recommendations": [
    {
      "experiment_id": "Next-01",
      "type": "Exploitation",
      "formula": {
        "3MTPAI_mM": 6,
        "PDAI2_mM": 3,
        "EDAI2_mM": 2,
        "PipDI_mM": 0
      },
      "process": {
        "solvent": "IPA",
        "spin_speed_rpm": 4000,
        "spin_time_s": 30,
        "anneal_temp_C": 100,
        "anneal_time_min": 5
      },
      "predicted_PCE": 24.7,
      "uncertainty": 0.4,
      "acquisition_score": 24.9
    }
  ]
}
```

### 15.5 Critic Agent
输入：
- Optimizer Agent recommendations
- Domain Agent molecule insights
- 约束条件

输出：
```json
{
  "reviewed_recommendations": [
    {
      "experiment_id": "Next-01",
      "risk_level": "Low",
      "risk_reasons": [
        "Close to current high-performance region.",
        "Only minor concentration adjustment."
      ],
      "suggested_control": "Keep a 3MTPAI-only control group."
    }
  ]
}
```

### 15.6 Planner Agent
输入：
- Optimizer recommendations
- Critic review
- Domain molecule insights

输出：
```json
{
  "experiment_plans": [
    {
      "experiment_id": "Next-01",
      "objective": "Validate whether moderate EDAI2 reduction improves Voc and FF.",
      "formula": "3MTPAI 6 mM + PDAI2 3 mM + EDAI2 2 mM",
      "procedure": [
        "Prepare passivation solution in IPA.",
        "Spin-coat passivation solution on perovskite film.",
        "Anneal at 100 °C for 5 min.",
        "Complete charge transport layer and electrode deposition.",
        "Measure PCE, Voc, Jsc and FF."
      ],
      "recommendation_reason": "This point is close to the current high-performance region while testing a lower EDAI2 level.",
      "risk": "If EDAI2 is too low, defect passivation may be insufficient.",
      "hypothesis": "Moderate EDAI2 may reduce interfacial recombination without harming charge transport.",
      "expected_observation": "Voc and FF should improve or remain stable while Jsc should not drop significantly."
    }
  ]
}
```

---

## 16. Demo 验收标准

### 16.1 功能验收
| 验收项 | 标准 |
|---|---|
| 本地启动 | 执行 `streamlit run app.py` 可正常打开页面 |
| 数据加载 | 能读取并展示 `demo_optimization_table.csv` |
| 数据分析 | 能展示 Total Records、Best PCE、Mean PCE、passivator 分布 |
| 最佳实验识别 | 能正确显示当前 PCE 最高实验 |
| 推荐生成 | 点击按钮后能输出 3–5 组下一轮实验 |
| 推荐解释 | 每组实验包含推荐理由、风险、科学假设 |
| 实验方案 | 每组实验包含可执行步骤和测试指标 |
| 趋势展示 | 能展示模拟 PCE 迭代曲线 |
| 数据标注 | 页面明确标注 demo / simulated / not experimentally validated |

### 16.2 展示验收
| 验收项 | 标准 |
|---|---|
| 3–5 分钟可讲完 | 演示不依赖复杂配置 |
| 产品感 | 页面像一个实验员工具，而不是纯代码输出 |
| 逻辑自洽 | 从数据分析到推荐实验之间有清晰因果链 |
| 风险可控 | 不声称真实预测 PCE，不声称实验已验证 |
| 差异化明确 | 能讲清楚 BO 负责选点，Agent 负责理解、审查和解释 |

### 16.3 不作为验收项
- 不要求真实实验验证推荐方案。
- 不要求严格论文级复现。
- 不要求企业级权限、数据库和部署。
- 不要求真正支持所有钙钛矿体系。
- 不要求第一版实现完整 PDF 自动抽取。

---

## 17. 3–5 分钟 Leader 演示脚本

### 17.1 开场 20 秒
话术：
> 这个 demo 展示的是一个面向钙钛矿配方实验员的优化 Agent。它现在不是为了证明已经能真实预测 PCE，而是为了验证一个产品闭环：读取已有实验数据，分析当前结果，推荐下一轮实验，并解释推荐理由、风险和待验证假设。

### 17.2 Step 1：任务初始化 40 秒
操作：
- 打开 Overview 页面。
- 展示自然语言任务。
- 点击 `Initialize Optimization Task`。

话术：
> 实验员只需要输入一个相对自然的目标，比如优化钙钛矿钝化配方、提升 PCE。Agent 会把它转成优化问题，包括目标、候选钝化剂、变量空间和固定条件。

### 17.3 Step 2：数据读取 40 秒
操作：
- 切到 Data Library 页面。
- 展示实验数据表。
- 点开 evidence text。

话术：
> 这里的数据来自我们已有的钙钛矿实验/论文数据整理。Demo 里先用结构化表承载，后续可以替换成真实 ELN、Excel 或论文抽取结果。

### 17.4 Step 3：已有结果分析 50 秒
操作：
- 切到 Data Analysis 页面。
- 展示 Best PCE、passivator 分布、数据质量提醒。

话术：
> Data Agent 会先分析已有结果，而不是直接推荐。它会识别当前最优实验、不同钝化剂的表现，以及哪些区域数据不足。例如 PipDI 在当前数据中没有真实记录，所以后面只能作为探索型候选。

### 17.5 Step 4：生成下一轮实验 80 秒
操作：
- 切到 Recommendation 页面。
- 调整 beta 或保持默认。
- 点击 `Generate Next Experiments`。
- 展示 3–5 张推荐实验卡片。

话术：
> Optimizer Agent 使用一个简化 BO 流程，在候选配方空间中平衡 predicted PCE 和 uncertainty。推荐结果不是只有最高预测点，而是包含 exploitation、exploration 和 control。随后 Critic Agent 会检查风险，Planner Agent 会把推荐转成实验员可执行的方案。

### 17.6 Step 5：趋势展示 40 秒
操作：
- 切到 Iteration Trend 页面。
- 展示模拟曲线。
- 点击模拟回填。

话术：
> 这里的趋势曲线是 demo 模拟，用来展示未来接入真实实验闭环后，系统可以如何持续更新模型并缩短实验探索周期。

### 17.7 结尾 20 秒
话术：
> 所以这一版不是最终科研工具，而是验证产品形态：BO 负责选下一轮实验点，LLM/Agent 负责读数据、解释机制、审查风险和生成实验方案。后续如果方向确认，我们可以把 mock 数据替换为真实实验数据，把简化 BO 替换为更严格的 LLM-BO 或多目标 BO。

---

## 18. 风险与开放问题

候选风险：
1. 客户历史数据质量差，导致模型无法稳定推荐。
2. LLM 生成的化学假设可能不可靠，需要强约束与可追溯机制。
3. BO 在高维类别变量空间中效率不稳定。
4. 用户可能不信任黑盒推荐。
5. 工业客户场景差异大，平台化和定制化边界不清。

开放问题：
- 首个付费客户/样板客户是谁？
- 是否已有真实历史实验数据？
- 是否需要与现有实验系统集成？
- 最终产品形态是 SaaS、私有化部署、还是项目制交付工具？

---

## 11. 下一步需求澄清清单
优先澄清：
1. 首个行业和具体业务场景。
2. 目标用户和购买决策人。
3. 当前人工优化流程和痛点强度。
4. 可用数据类型、规模和质量。
5. 推荐结果最终如何被使用。
6. MVP 成功标准。
7. 产品边界与不做事项。

