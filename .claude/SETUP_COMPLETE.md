# Claude Code 自动化配置完成

## ✅ 已完成的配置

### 1. 项目配置文件

#### `.claude/settings.json`
- ✅ 权限配置：允许测试、开发服务器、git 命令
- ✅ PostToolUse Hook：修改后端代码后自动运行 pytest
- ✅ PreToolUse Hook：保护 `.env` 文件和评测结果文件

---

### 2. 技能 (Skills)

#### `scientific-computing` (.claude/skills/scientific-computing/SKILL.md)
**用途**: 审查科学计算代码的数值稳定性、物理一致性和数学正确性

**触发时机**: 修改 `backend/optimization/` 或 `backend/benchmark/` 时自动应用

**审查内容**:
- 数值稳定性（除零、矩阵条件数、对数空间）
- 物理一致性（能带公式、单位、参数边界）
- 贝叶斯优化实现（高斯过程、采集函数）
- 数据验证（NaN、Inf、形状匹配）
- 性能优化（向量化、内存管理）

---

#### `tdd-workflow` (.claude/skills/tdd-workflow/SKILL.md)
**用途**: 强制测试驱动开发循环

**工作流程**:
1. **RED**: 先写失败的测试
2. **GREEN**: 实现最小可用代码
3. **REFACTOR**: 重构并保持测试通过

**覆盖率要求**: ≥80%（核心模块 ≥90%）

---

### 3. 子 Agent (Subagents)

#### `scientific-reviewer` (.claude/agents/scientific-reviewer.md)
**用途**: 深度审查科学计算代码

**审查范围**:
- `backend/optimization/optimizer.py` - 贝叶斯优化核心
- `backend/optimization/knowledge.py` - 物理公式计算
- `backend/optimization/memory.py` - 向量记忆存储
- `backend/benchmark/bo_step.py` - 优化步进逻辑

**输出格式**: 按严重程度分类（CRITICAL / HIGH / MEDIUM / LOW）

---

#### `e2e-test-writer` (.claude/agents/e2e-test-writer.md)
**用途**: 为 React 组件自动生成 Playwright E2E 测试

**测试类型**:
- 性能评测模式测试（模型切换、日志流、图表渲染）
- 实验实操模式测试（参数输入、推荐获取、推理详情）
- 架构可视化测试（3D 能带图、交互旋转）

**使用 Page Object 模式**: 提高测试可维护性

---

### 4. 项目文档

#### `CLAUDE.md`
项目级配置文档，包含：
- 开发工作流（启动服务、运行测试）
- 代码审查要求
- 物理公式参考
- 测试覆盖率要求
- 敏感文件保护
- 常用命令

---

## 🔌 现有 MCP 服务器

你已经配置了以下 MCP 服务器：
- ✅ **codegraph**: 代码知识图谱（已在使用）
- ✅ **grok-search**: 网络搜索
- ✅ **claude-mem**: 跨会话记忆
- ✅ **sensenova-image**: 图像生成

**建议添加**:
- **context7**: 实时文档查询（React 19、Vite 6、Playwright）
- **playwright**: 浏览器自动化（如果需要交互式 E2E 调试）

---

## 📋 配置文件清单

```
BOagent/
├── .claude/
│   ├── settings.json              ✅ 权限和 Hooks 配置
│   ├── skills/
│   │   ├── scientific-computing/
│   │   │   └── SKILL.md          ✅ 科学计算审查技能
│   │   └── tdd-workflow/
│   │       └── SKILL.md          ✅ TDD 工作流技能
│   └── agents/
│       ├── scientific-reviewer.md ✅ 科学计算审查 Agent
│       └── e2e-test-writer.md    ✅ E2E 测试生成 Agent
└── CLAUDE.md                      ✅ 项目配置文档
```

---

## 🚀 如何使用

### 自动触发

1. **修改后端代码** → 自动运行 pytest
   ```bash
   # 你修改了 backend/api.py
   # Hook 自动执行: cd backend && python -m pytest tests/ -v --tb=short -x
   ```

2. **修改科学计算代码** → 自动应用 `scientific-computing` 技能
   ```bash
   # 你修改了 backend/optimization/optimizer.py
   # Claude 自动审查数值稳定性、物理一致性
   ```

---

### 手动调用

1. **深度科学计算审查**
   ```
   请使用 scientific-reviewer agent 审查 optimizer.py
   ```

2. **生成 E2E 测试**
   ```
   请使用 e2e-test-writer agent 为新的参数输入表单生成测试
   ```

3. **TDD 开发**
   ```
   我要添加 EI 采集函数，请使用 tdd-workflow
   ```

---

## 🔒 安全保护

### 受保护的文件

修改以下文件时会弹出确认提示：
- `**/.env*` - 环境变量文件
- `backend/*_results.json` - 评测结果数据

### 示例

```bash
# 你尝试修改 backend/.env
⚠️  警告：正在修改环境变量文件，请确认操作安全性
[确认] [取消]
```

---

## 📊 测试覆盖率

### 检查覆盖率

```bash
# 后端
cd backend
python -m pytest --cov=. --cov-report=term-missing

# 前端（如果配置了）
cd frontend
npm run test:coverage
```

### 覆盖率要求

- **最低**: 80%
- **核心模块** (`backend/optimization/`): 90%

---

## 🎯 下一步建议

### 可选：添加更多 MCP 服务器

如果需要实时文档查询，可以手动添加 context7：

1. 访问 [context7 官网](https://context7.com) 获取安装说明
2. 或使用其他文档查询工具

### 可选：添加更多 Hooks

你可以在 `.claude/settings.json` 中添加更多 hooks：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "name": "auto-format",
        "match": {
          "tools": ["Edit", "Write"],
          "pathPattern": "**/*.py"
        },
        "command": "black .",
        "continueOnError": true
      }
    ]
  }
}
```

---

## ✨ 配置总结

你的 BOagent 项目现在已经配置了：

✅ **2 个技能** - 科学计算审查 + TDD 工作流  
✅ **2 个 Agent** - 科学计算审查员 + E2E 测试生成器  
✅ **3 个 Hooks** - 自动测试 + 环境变量保护 + 评测结果保护  
✅ **1 个项目文档** - CLAUDE.md 项目配置指南  
✅ **4 个 MCP 服务器** - codegraph + grok-search + claude-mem + sensenova-image

---

## 🎉 开始使用

现在你可以：

1. **修改代码** - 自动运行测试和审查
2. **添加功能** - 使用 TDD 工作流
3. **审查代码** - 调用 scientific-reviewer
4. **生成测试** - 调用 e2e-test-writer

试试修改一个文件，看看自动化是否生效！
