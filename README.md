# BOagent

钙钛矿材料优化的贝叶斯优化 Agent 系统 - PVK-BO Agent Workflow

## 功能特性

- **三步 Agent 流程**: Initialization → Screening → Optimization
- **LLM 驱动**: 集成 DeepSeek LLM，支持自然语言交互
- **实时 BO 曲线**: 展示贝叶斯优化轨迹和 best-so-far 进度
- **安全边界**: 明确区分算法推荐与湿实验验证
- **React 前端**: 现代化对话式交互界面
- **FastAPI 后端**: 高性能 RESTful API

## 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  React Chat UI  │────▶│  FastAPI Backend│────▶│   PVKBO Runtime │
│  (Vite)         │     │  (api.py)       │     │  (pvk_*.py)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              ▼                        ▼
                     ┌─────────────────┐     ┌─────────────────┐
                     │  ChatAgent      │     │  LLM_ACQ /      │
                     │  (LLM Planner)  │     │  Surrogate Model│
                     └─────────────────┘     └─────────────────┘
```

数据流向:
1. 用户通过 React 界面发送自然语言请求
2. `ChatAgent` 解析任务，调用 PVKBO 运行时
3. PVKBO 执行贝叶斯优化计算（LLM_ACQ 获取函数 + Surrogate）
4. 结果经 LLM 解释器转化为自然语言回复
5. 前端展示 BO 曲线、推荐候选和证据面板

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- (可选) DeepSeek API Key 用于 LLM 增强功能

### 一键启动

```bash
# 1. 安装 Python 依赖
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 启动后端 (端口 8000)
python -m uvicorn api:app --reload --port 8000

# 3. 新终端: 启动前端 (端口 5173)
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

### 访问地址

- 前端界面: http://localhost:5173
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health

## 完整安装指南

### 环境变量配置

复制 `backend/.env.example` 到 `backend/.env` 并配置：

```bash
cd backend
cp .env.example .env
# 编辑 .env 文件配置 API Key
```

```bash
# LLM 配置 (二选一)
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

# 或使用 OpenAI 兼容格式
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash

# PVK-LLM 数据路径 (可选，默认 ../../PVK-LLM)
PVK_LLM_ROOT=/path/to/PVK-LLM
PVK_DATA_ROOT=/path/to/PVK-LLM/custom_perovskite_dataset
```

### PVK-LLM 数据集

真实任务需要 PVK-LLM 数据集（克隆到项目同级目录）：

```bash
# 克隆 PVK-LLM 到项目同级目录（与 BOagent 同一父目录）
git clone https://github.com/your-org/PVK-LLM ../../PVK-LLM

# 确保数据集文件存在
ls ../PVK-LLM/custom_perovskite_dataset/
# - bandAlignment.xlsx  (band_alignment 任务)
# - defectsAndDoping.xlsx (defects_doping 任务)
```

## 测试

运行完整测试套件：

```bash
cd backend

# 快速冒烟测试
python -m pytest -q tests/test_api.py tests/test_pvk_demo.py

# 完整测试
python -m pytest -q

# 查看覆盖率
python -m pytest --cov=. --cov-report=term-missing
```

## API 使用示例

### 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

响应:
```json
{"data":{"status":"ok","service":"boagent-api","version":"0.1.0"}}
```

### 创建 Agent 运行

```bash
curl -X POST http://localhost:8000/api/v1/agent-runs \
  -H "Content-Type: application/json" \
  -d '{
    "task_text": "优化钙钛矿钝化配方，提高 PCE",
    "recommendation_count": 3,
    "language": "zh",
    "use_llm": false
  }'
```

### 获取运行结果

```bash
curl http://localhost:8000/api/v1/agent-runs/{RUN_ID}
```

### 对话式交互

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我优化 band alignment，推荐 5 个候选配方",
    "session_id": "optional-session-id"
  }'
```

## 任务类型

| 任务类型 | 数据源 | 描述 |
|---------|--------|------|
| `passivation_demo` | demo_optimization_table.csv | 演示模式，使用静态数据集 |
| `band_alignment` | bandAlignment.xlsx | 能带对齐优化，真实 Excel 黑盒查询 |
| `defects_doping` | defectsAndDoping.xlsx | 缺陷掺杂优化，真实 Excel 黑盒查询 |

## 科学边界声明

> ⚠️ **重要**: 本系统输出的是算法推荐，不是湿实验验证结果

1. **真实任务**: `band_alignment` / `defects_doping` 使用 PVKBO 的 LLM_ACQ 获取函数、LLM surrogate 模型和 Excel 黑盒查询
2. **Demo 模式**: `passivation ratio` 是策略/组合标签，不是真实摩尔浓度
3. **高风险候选**: PipDI 等应标注为高风险，不应呈现为已验证推荐
4. **BO 曲线**: 只展示当前 session best-so-far，不是正式 benchmark 结果
5. **算法局限性**: 所有推荐均需湿实验验证，算法结果仅作参考

## 开发指南

### 项目结构

```
BOagent/
├── backend/                 # Python 后端
│   ├── api.py              # FastAPI 主入口
│   ├── chat_agent.py       # 对话 Agent 核心逻辑
│   ├── pvk_session_runtime.py  # PVKBO Session 运行时
│   ├── pvk_llm_bo_runtime.py  # LLM-BO 运行时
│   ├── pvk_demo.py         # Demo 数据加载
│   ├── pvk_mvp.py          # MVP 工具函数
│   ├── pvk_tools.py        # 通用工具函数
│   ├── llm_client.py       # LLM API 客户端
│   ├── agent_runtime.py    # Agent 运行框架
│   ├── data/               # 数据文件
│   │   └── demo_optimization_table.csv
│   ├── tests/              # 测试套件
│   │   ├── test_api.py
│   │   └── test_pvk_*.py
│   ├── requirements.txt    # Python 依赖
│   └── .env.example        # 环境变量模板
├── frontend/               # React + Vite 前端
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

### 添加新任务类型

1. 在 `backend/pvk_session_runtime.py` 中注册新的任务处理器
2. 在 `backend/api.py` 中添加对应的 API 端点
3. 更新前端任务选择组件
4. 添加测试用例到 `backend/tests/`

### 前端开发

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build
```

### E2E 端到端测试

使用 Playwright 进行 E2E 测试：

```bash
cd frontend

# 运行所有 E2E 测试 (Chromium)
npm run test:e2e:chromium

# 运行测试并打开 UI 界面
npm run test:e2e:ui

# 调试模式运行测试
npm run test:e2e:debug

# 查看测试报告
npm run test:e2e:report
```

**测试文件结构**：
- `tests/e2e/app.spec.ts` - 应用基础功能和 API 测试
- `tests/e2e/chat-flow.spec.ts` - 聊天流程测试
- `tests/pages/AppPage.ts` - Page Object Model

**测试覆盖范围**：
- 应用加载和 UI 元素显示
- 聊天消息发送和响应
- 快捷操作按钮
- 后端 API 健康检查
- 任务列表 API
- 视频录制和截图（失败时自动保存）
npm install

# 开发模式 (热重载)
VITE_API_BASE_URL=http://localhost:8000 npm run dev

# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

## 验收清单

发布前请验证：

- [ ] 所有测试通过: `cd backend && python -m pytest -q`
- [ ] 前端构建成功: `cd frontend && npm run build`
- [ ] 三阶段流程在浏览器中可完整走通
- [ ] BO 曲线正常显示，科学边界文案正确
- [ ] LLM 功能正常（配置 API Key 后）
- [ ] 真实任务可用（数据集路径正确）
- [ ] 端口 8000 被占用时可切换到 8010

## 常见问题

### 端口被占用

```bash
# 后端改用 8010
cd backend
python -m uvicorn api:app --reload --port 8010

# 前端同步修改
cd frontend
VITE_API_BASE_URL=http://localhost:8010 npm run dev
```

### 真实任务失败

检查：
1. PVK-LLM 数据集路径是否正确
2. Excel 文件是否存在
3. Python 依赖是否完整（openpyxl, pandas 等）

### LLM 调用失败

检查：
1. API Key 是否正确配置
2. 网络连接是否正常
3. 模型名称是否有效

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

基于 PVK-LLM 研究项目构建，感谢原作者提供的贝叶斯优化算法实现。
