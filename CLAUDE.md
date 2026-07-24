# BOagent 项目配置

这是一个钙钛矿太阳能电池优化的 LLM 驱动贝叶斯优化系统。

## 项目概况

- **后端**: Python + FastAPI + scikit-learn
- **前端**: React 19 + Vite 6 + TypeScript 6 + Recharts
- **测试**: pytest (后端) + Playwright (前端 E2E)
- **核心领域**: 贝叶斯优化、高斯过程、材料科学、能带对齐

---

## 开发工作流

### 1. 启动服务

```bash
# 后端 (端口 8000)
cd apps/api
source .venv/bin/activate
pip install -r requirements.txt   # 首次运行需安装（含 bo-core editable 安装）
python -m uvicorn api:app --reload --port 8000

# 前端 (端口 5173)
cd apps/web
npm run dev
```

### 2. 运行测试

```bash
# API 单元测试
cd apps/api
python -m pytest tests/ -v

# 算法核心测试
cd packages/bo-core
python -m pytest tests/ -v

# 前端 E2E 测试
cd apps/web
npm run test:e2e

# E2E 调试模式
npm run test:e2e:ui
```

---

## 代码审查要求

### 科学计算代码

修改以下文件时，**必须**使用 `scientific-reviewer` agent 审查：
- `packages/bo-core/bo_core/optimization/optimizer.py` - 贝叶斯优化核心
- `packages/bo-core/bo_core/optimization/knowledge.py` - 物理公式计算
- `packages/bo-core/bo_core/optimization/memory.py` - 向量记忆存储
- `packages/bo-core/bo_core/benchmark/bo_step.py` - 优化步进逻辑

**审查重点**:
- 数值稳定性（除零、矩阵条件数、对数空间）
- 数学正确性（高斯过程、采集函数、能带公式）
- 物理一致性（单位、参数边界、公式符号）

### 前端功能

添加新的前端功能时，**必须**使用 `e2e-test-writer` agent 生成 E2E 测试。

---

## 物理公式参考

### 钙钛矿能带对齐

```python
# 导带偏移 (Conduction Band Offset)
CBO = χ_PVK - χ_ETL

# 价带偏移 (Valence Band Offset)
VBO = (χ_HTL + E_g_HTL) - χ_PVK

# LUMO 电子阻挡
LUMO_barrier = LUMO_HTL - LUMO_PVK
```

**单位约定**:
- 能量: eV
- 效率 (PCE): % [0, 100]

**参数范围**:
- χ (电子亲和能): [3.0, 6.0] eV
- E_g (带隙): [1.0, 4.0] eV

---

## 测试覆盖率要求

- **最低覆盖率**: 80%
- **核心模块**: `packages/bo-core/bo_core/optimization/` 要求 ≥90%

检查覆盖率:
```bash
cd apps/api
python -m pytest --cov=. --cov-report=term-missing
```

---

## 敏感文件保护

以下文件受 hooks 保护，修改时需要确认：
- `**/.env*` - 环境变量文件
- `**/*_results.json` - 评测结果数据

---

## 依赖管理

### 后端依赖 (apps/api)

```bash
# 安装依赖（requirements.txt 含 -e ../packages/bo-core，会自动安装算法包）
cd apps/api
pip install -r requirements.txt
```

### 算法核心依赖 (packages/bo-core)

```bash
# bo-core 是独立可安装包，依赖在 pyproject.toml 声明
cd packages/bo-core
pip install -e .
```

### 前端依赖

```bash
# 安装依赖
cd apps/web
npm install

# 添加新依赖
npm install <package-name>
```

---

## 数据集配置

项目依赖外部数据集 `PVK-LLM`，配置在 `apps/api/.env`:

```bash
PVK_LLM_ROOT=../PVK-LLM
PVK_DATA_ROOT=../PVK-LLM/custom_perovskite_dataset
```

确保数据集路径正确，否则评测无法运行。

---

## 常用命令

```bash
# 后端测试（快速模式，遇到失败立即停止）
cd apps/api && python -m pytest -x -v

# 后端测试（覆盖率报告）
cd apps/api && python -m pytest --cov=. --cov-report=html

# 前端 E2E 测试（仅 Chromium）
cd apps/web && npm run test:e2e:chromium

# 前端 E2E 测试（显示浏览器）
cd apps/web && npm run test:e2e:headed

# 查看 E2E 测试报告
cd apps/web && npm run test:e2e:report
```

---

## 技能和 Agent

### 自动触发的技能

- **scientific-computing**: 修改 `packages/bo-core/bo_core/optimization/` 时自动审查科学计算正确性
- **tdd-workflow**: 添加新功能或修复 bug 时强制 TDD 循环

### 可调用的 Agent

- **scientific-reviewer**: 深度审查科学计算代码
- **e2e-test-writer**: 为前端功能生成 Playwright E2E 测试

---

## 架构决策记录 (ADR)

重要的架构决策记录在 `docs/adr/` 目录：
- 贝叶斯优化算法选择
- LLM 提示词设计
- 动态科学记忆 (DSM) 实现

---

## 贡献指南

1. **创建分支**: `git checkout -b feature/your-feature`
2. **TDD 开发**: 先写测试，再实现功能
3. **运行测试**: 确保所有测试通过
4. **代码审查**: 使用 `scientific-reviewer` 审查科学计算代码
5. **提交代码**: 遵循 conventional commits 格式
6. **创建 PR**: 使用 `gh pr create`

---

## 相关文档

- [README.md](README.md) - 项目概述和快速开始
- [ARCHITECTURAL_ANALYSIS.md](ARCHITECTURAL_ANALYSIS.md) - 架构分析
- [AGENTS.md](AGENTS.md) - Agent 配置说明
