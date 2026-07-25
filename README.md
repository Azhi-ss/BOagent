# BOagent: 钙钛矿太阳能电池优化代理系统 (PVK-BO Agent Workflow)

这是一个面向钙钛矿太阳能电池优化的、由大语言模型（LLM）驱动的贝叶斯优化（Bayesian Optimization, BO）智能决策系统。系统集成了高斯过程（Gaussian Process, GP）代理模型与材料物理启发的大模型获取函数，能够高效推进能带对齐（Band Alignment）与缺陷掺杂（Defects & Doping）等核心材料参数的优化。

---

## 🚀 系统核心特性

1. **物理启发式大模型提示（Physics-informed Prompts）**：
   - 摆脱单纯的数值统计黑盒优化。系统在每次选点决策前，将半导体物理先验知识（如：导带偏移 $CBO = \chi_{PVK} - \chi_{ETL}$、价带偏移 $VBO = (\chi_{HTL} + E_{g,HTL}) - \chi_{PVK}$ 及电子阻挡 LUMO 能量级差）动态注入 LLM 提示词中，指导材料配方的物理性筛选。
2. **结构化推理决策链（Structured Reasoning Loop）**：
   - 强制 LLM 遵循 `Thinking Process (思考链)` -> `Analysis (物理依据)` -> `Selected Formulations (推荐配方)` 格式进行多维度设备物理推理，使每次决策均有物理逻辑支撑且完全可被人类追溯。
3. **动态科学记忆（Dynamic Scientific Memory, DSM）**：
   - 引入会话级的 RAG（检索增强生成）经验记忆环路。一旦系统观测到更优的器件效率（PCE）得分，即刻调用 LLM 抽取多字段 JSON 结构化知识（包含核心发现、参数因果关系及优化原则），并使用豆包（Ark）嵌入向量模型将其持久化存储。后续决策中会自动时序检索前 3 条经验，防止算法重复探索低效材料区间。
4. **多模型与多种采集函数支持（Modular & Configurable）**：
   - 后端支持统一的 `BayesianOptimizer`，完美解耦并支持 **UCB**、**EI**、**PI** 等经典采集函数；前端支持一键切换 `DeepSeek-v4-Flash` 与更高推理精度的 `DeepSeek-v4-Pro` 引擎。
5. **双重运行模式**：
   - **性能评测 (Benchmark Mode)**：自动化测试代理团队（Agent Team）可以对算法展开多随机种子（Multi-seed）、多迭代次数（Trials）的并行跑数评测，自动保存历史轨迹并生成 Markdown 分析报告。
   - **实验实操 (Operational Mode)**：面向湿实验设计的人机协同（Human-in-the-loop）模式，科研人员可录入实测点，咨询 Agent 获取下一步推荐配方。

---

## 📊 钙钛矿能带对齐任务最新评测表现 (Task: band_alignment)

在 5 个评估随机种子 (`[42, 100, 123, 456, 789]`)、每个种子 20 轮迭代的严格对比中，最新优化的 LLMBO 取得了显著的性能拉升，且展现出极高的收敛稳定性：

| 评估指标 | 传统贝叶斯优化 (Traditional BO) | 基础版 LLMBO (Baseline LLMBO) | 优化版 LLMBO (Ours, 引入物理先验与 DSM) | 相对基线 LLMBO 净提升 |
| :--- | :---: | :---: | :---: | :---: |
| **最大 PCE 平均值 (Mean)** | 27.0889 | 27.5219 | **27.8138** | **+1.06%** |
| **最大 PCE 标准差 (Std)** | 1.4394 | 0.6981 | **0.4163** (方差减小) | **-40.37%** (收敛更稳定) |
| **相比传统 BO 提升 (Lift)** | - | 1.60% | **2.68%** | - |

> [!TIP]
> 评测数据表明，传统高斯过程贝叶斯优化在纯数值表格上容易陷入局部最优或面临较高的不确定性（标准差达 1.4394）。而引入物理公式计算与记忆反馈的优化版 LLMBO 能够稳定且迅速地逼近全局效率极值点（PCE 近 27.81%）。

---

## 📂 项目模块结构

项目去除了历史遗留的冗余代码，重构为高度模块化的工程体系：

```
BOagent/
├── apps/
│   ├── api/                        # FastAPI 后端服务
│   │   ├── api.py                  # 接口层：实时流日志与实操推荐 API
│   │   ├── conftest.py             # pytest 路径配置
│   │   ├── tests/                  # API 单元测试
│   │   └── requirements.txt        # 后端依赖（含 bo-core editable 安装）
│   └── web/                        # React 前端工程
│       ├── src/
│       │   ├── App.tsx             // 性能评测面板、实验控制及 Recharts 可视化折线图
│       │   ├── OperationalMode.tsx // 人机实操面板，支持自定义输入空间及 Agent 推理详情卡
│       │   ├── components/         // 包含三维曲面图、采集设置与日志流组件
│       │   └── lib/api.ts          // 请求封装与 SSE (Server-Sent Events) 事件流接收
│       └── tests/e2e/              // Playwright E2E 测试
├── packages/
│   └── bo-core/                    # 算法核心包（可独立 pip install -e）
│       ├── bo_core/
│       │   ├── optimization/       # 贝叶斯优化与智能决策核心
│       │   │   ├── optimizer.py    # 高斯过程代理模型与 UCB/EI/PI 采集计算
│       │   │   ├── knowledge.py    # 物理公式组装、结构化 Prompt 生成与 LLM 交互
│       │   │   └── memory.py       # DSM 向量记忆存储，内置豆包文本嵌入与 recency 兜底
│       │   ├── benchmark/          # 评测环境与数据装载
│       │   │   ├── comparison.py   # 多种子对比曲线生成与数据持久化
│       │   │   ├── bo_step.py      # 分步步进式传统与 LLM 优化对比引擎
│       │   │   └── data_loader.py  # 钙钛矿能带/掺杂 Excel 实验数据的安全划分与加载
│       │   ├── llm_client.py       # 统一 DeepSeek API 大模型调用客户端
│       │   └── pvk_llm_compat.py   # PVK-LLM 兼容层
│       ├── tests/                  # 算法单元测试集
│       ├── benchmark_agent_team.py # 自动化测试与指标分析多代理协调器
│       ├── run_prompt_ablation.py  # Prompt 消融实验脚本
│       └── pyproject.toml          # bo-core 包定义与依赖
└── README.md                       # 系统使用说明书
```

---

## 🛠️ 快速开始

### 1. 配置环境变量

在 `apps/api/` 下创建 `.env` 文件（或修改 `.env.example`）：

```bash
# 大模型服务配置
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_REASONING_EFFORT=high

# 豆包文本嵌入服务（用于 DSM 经验记忆向量化检索）
ARK_API_KEY=ark-your-key
ARK_API_BASE=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_EMBEDDING_MODEL=doubao-embedding-large-text-250515

# 数据集相对路径 (同级目录下克隆的 PVK-LLM 工程)
PVK_LLM_ROOT=../PVK-LLM
PVK_DATA_ROOT=../PVK-LLM/custom_perovskite_dataset
```

### 2. 启动后端

```bash
cd apps/api
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 会自动 editable 安装 ../packages/bo-core

# 启动 FastAPI (端口 8000)
python -m uvicorn api:app --reload --port 8000
```

### 3. 启动前端

```bash
cd apps/web
npm install

# 启动 Vite 开发服务器 (端口 5173)
npm run dev
```
打开浏览器访问 [http://localhost:5173](http://localhost:5173) 即可使用。

---

## 🧪 单元测试验证

要验证后端所有改动的正确性，可在虚拟环境中运行：

```bash
cd apps/api
python -m pytest
```
系统包含完善的测试，覆盖了 API 路由响应、高斯过程拟合、数据边界解析及能带物理逻辑校验。
