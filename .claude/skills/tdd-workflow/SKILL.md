---
name: tdd-workflow
description: 强制测试驱动开发循环 - 先写测试（RED）→ 实现功能（GREEN）→ 重构（REFACTOR）
---

# Test-Driven Development Workflow

当添加新功能、修复 bug 或重构代码时，强制执行 TDD 循环。

## TDD 三步循环

```
┌─────────────────────────────────────┐
│  1. RED: 写失败的测试                │
│     ↓                                │
│  2. GREEN: 实现最小可用代码          │
│     ↓                                │
│  3. REFACTOR: 重构并保持测试通过     │
└─────────────────────────────────────┘
```

---

## Phase 1: RED - 先写失败的测试

### Backend (pytest)

**测试文件位置**: `backend/tests/test_*.py`

**测试模板**:
```python
import pytest
from backend.optimization.optimizer import BayesianOptimizer

def test_ucb_acquisition_function():
    """测试 UCB 采集函数计算正确性"""
    optimizer = BayesianOptimizer(
        bounds=[(0, 1), (0, 1)],
        acquisition="ucb",
        kappa=2.0
    )
    
    # 添加初始观测点
    X_init = [[0.2, 0.3], [0.7, 0.8]]
    y_init = [25.0, 27.0]
    optimizer.tell(X_init, y_init)
    
    # 测试 UCB 计算
    X_test = [[0.5, 0.5]]
    ucb_values = optimizer._calculate_ucb(X_test)
    
    # 断言：UCB = mean + kappa * std
    assert ucb_values.shape == (1,)
    assert ucb_values[0] > 0  # UCB 应为正值
    assert not np.isnan(ucb_values[0])  # 不应为 NaN
```

**运行测试**:
```bash
cd backend
python -m pytest tests/test_optimizer.py::test_ucb_acquisition_function -v
```

**预期结果**: ❌ 测试失败（因为功能尚未实现）

---

### Frontend (Playwright E2E)

**测试文件位置**: `frontend/tests/e2e/*.spec.ts`

**测试模板**:
```typescript
import { test, expect } from '@playwright/test';

test('应该能够切换 LLM 模型并启动评测', async ({ page }) => {
  await page.goto('http://localhost:5173');
  
  // 切换到 DeepSeek Pro 模型
  await page.selectOption('select[name="model"]', 'deepseek-v4-pro');
  
  // 点击启动评测按钮
  await page.click('button:has-text("启动评测")');
  
  // 验证日志流开始输出
  await expect(page.locator('.log-stream')).toBeVisible();
  await expect(page.locator('.log-stream')).toContainText('开始评测');
  
  // 验证图表渲染
  await expect(page.locator('.recharts-wrapper')).toBeVisible();
});
```

**运行测试**:
```bash
cd frontend
npm run test:e2e -- --headed
```

**预期结果**: ❌ 测试失败（因为功能尚未实现）

---

## Phase 2: GREEN - 实现最小可用代码

### 实现原则

1. **最小实现**: 只写让测试通过的代码，不添加额外功能
2. **快速迭代**: 优先让测试变绿，不追求完美
3. **避免过度设计**: 不添加"可能需要"的功能

### Backend 实现示例

```python
# backend/optimization/optimizer.py

def _calculate_ucb(self, X):
    """计算 UCB 采集函数值"""
    X = np.atleast_2d(X)
    mean, std = self.gp.predict(X, return_std=True)
    
    # 添加 epsilon 防止 std=0
    std = np.maximum(std, 1e-10)
    
    # UCB = mean + kappa * std
    return mean + self.kappa * std
```

**运行测试**:
```bash
python -m pytest tests/test_optimizer.py::test_ucb_acquisition_function -v
```

**预期结果**: ✅ 测试通过

---

### Frontend 实现示例

```typescript
// frontend/src/App.tsx

const [selectedModel, setSelectedModel] = useState('deepseek-v4-flash');
const [isRunning, setIsRunning] = useState(false);

const handleStartBenchmark = async () => {
  setIsRunning(true);
  
  const response = await fetch('/api/benchmark/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: selectedModel })
  });
  
  // 处理 SSE 日志流
  const reader = response.body.getReader();
  // ... 流处理逻辑
};
```

**运行测试**:
```bash
npm run test:e2e
```

**预期结果**: ✅ 测试通过

---

## Phase 3: REFACTOR - 重构代码

### 重构检查清单

- [ ] **消除重复代码**: 提取公共逻辑到函数
- [ ] **改善命名**: 变量和函数名清晰表达意图
- [ ] **简化逻辑**: 减少嵌套层级，使用早返回
- [ ] **添加注释**: 解释"为什么"而非"是什么"
- [ ] **性能优化**: 向量化操作，避免循环

### 重构示例

**重构前**:
```python
def calculate_acquisition(self, X, method):
    if method == "ucb":
        mean, std = self.gp.predict(X, return_std=True)
        return mean + self.kappa * std
    elif method == "ei":
        mean, std = self.gp.predict(X, return_std=True)
        # ... EI 计算
    elif method == "pi":
        mean, std = self.gp.predict(X, return_std=True)
        # ... PI 计算
```

**重构后**:
```python
def calculate_acquisition(self, X, method):
    """计算采集函数值
    
    Args:
        X: 候选点 (n_samples, n_features)
        method: 采集函数类型 ("ucb", "ei", "pi")
    
    Returns:
        采集函数值 (n_samples,)
    """
    mean, std = self._predict_with_uncertainty(X)
    
    acquisition_funcs = {
        "ucb": self._calculate_ucb,
        "ei": self._calculate_ei,
        "pi": self._calculate_pi
    }
    
    return acquisition_funcs[method](mean, std)

def _predict_with_uncertainty(self, X):
    """预测均值和标准差（添加数值稳定性保护）"""
    mean, std = self.gp.predict(X, return_std=True)
    std = np.maximum(std, 1e-10)  # 防止 std=0
    return mean, std
```

**验证重构**:
```bash
# 确保所有测试仍然通过
python -m pytest tests/ -v
```

**预期结果**: ✅ 所有测试通过，代码更清晰

---

## 测试覆盖率要求

### 最低覆盖率: 80%

**检查覆盖率**:
```bash
# Backend
cd backend
python -m pytest --cov=. --cov-report=term-missing

# Frontend (如果配置了 coverage)
cd frontend
npm run test:coverage
```

**覆盖率报告示例**:
```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
backend/optimization/optimizer.py    120      8    93%   45-47, 89
backend/optimization/knowledge.py    200     15    92%   
backend/optimization/memory.py        85      5    94%   
---------------------------------------------------------------
TOTAL                                405     28    93%
```

---

## 测试类型

### 1. 单元测试 (Unit Tests)
- **范围**: 单个函数或类方法
- **速度**: 快速（< 1s）
- **示例**: 测试 UCB 计算、能带公式

### 2. 集成测试 (Integration Tests)
- **范围**: 多个模块交互
- **速度**: 中等（1-5s）
- **示例**: 测试完整的 BO 迭代流程

### 3. E2E 测试 (End-to-End Tests)
- **范围**: 完整用户流程
- **速度**: 慢（5-30s）
- **示例**: 测试从启动评测到查看结果的完整流程

---

## 常见陷阱

### ❌ 反模式

1. **先写实现再补测试** - 违背 TDD 原则
2. **测试覆盖率作弊** - 写无意义的测试只为提高覆盖率
3. **测试依赖外部服务** - 应使用 mock 隔离外部依赖
4. **测试过于脆弱** - 实现细节变化导致测试失败

### ✅ 最佳实践

1. **测试行为而非实现** - 关注输入输出，不关注内部实现
2. **使用 fixtures 复用测试数据** - `conftest.py` 定义共享 fixtures
3. **测试边界条件** - 空输入、极值、异常情况
4. **保持测试独立** - 每个测试可独立运行

---

## 示例工作流

### 场景：添加 EI（期望改进）采集函数

**Step 1: RED - 写失败的测试**
```python
# backend/tests/test_optimizer.py
def test_ei_acquisition_function():
    optimizer = BayesianOptimizer(acquisition="ei")
    optimizer.tell([[0.5, 0.5]], [26.0])
    
    ei_values = optimizer._calculate_ei([[0.3, 0.7]])
    
    assert ei_values[0] >= 0  # EI 应非负
    assert not np.isnan(ei_values[0])
```

运行: `pytest tests/test_optimizer.py::test_ei_acquisition_function`  
结果: ❌ `AttributeError: 'BayesianOptimizer' object has no attribute '_calculate_ei'`

---

**Step 2: GREEN - 最小实现**
```python
# backend/optimization/optimizer.py
def _calculate_ei(self, mean, std):
    """计算期望改进"""
    from scipy.stats import norm
    
    # 当前最优值
    y_best = np.max(self.y_observed)
    
    # 改进量
    improvement = mean - y_best
    
    # 标准化改进
    Z = improvement / std
    
    # EI = improvement * CDF(Z) + std * PDF(Z)
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    
    return ei
```

运行: `pytest tests/test_optimizer.py::test_ei_acquisition_function`  
结果: ✅ 测试通过

---

**Step 3: REFACTOR - 重构优化**
```python
def _calculate_ei(self, mean, std):
    """计算期望改进（Expected Improvement）
    
    EI 衡量候选点相对当前最优值的期望改进量。
    数值稳定的实现避免了 std=0 时的除零错误。
    
    Args:
        mean: 预测均值 (n_samples,)
        std: 预测标准差 (n_samples,)
    
    Returns:
        期望改进值 (n_samples,)
    """
    from scipy.stats import norm
    
    y_best = np.max(self.y_observed)
    improvement = mean - y_best
    
    # 数值稳定性：当 std 很小时，EI ≈ 0
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0] = 0  # std=0 时 EI=0
    
    return ei
```

运行: `pytest tests/ -v`  
结果: ✅ 所有测试通过

---

## 相关配置

- **pytest 配置**: `backend/pytest.ini` 或 `backend/conftest.py`
- **Playwright 配置**: `frontend/playwright.config.ts`
- **覆盖率配置**: `backend/.coveragerc`

---

## 快捷命令

```bash
# Backend 快速测试
alias pytest-fast="python -m pytest -x -v"

# Backend 覆盖率报告
alias pytest-cov="python -m pytest --cov=. --cov-report=html"

# Frontend E2E 测试
alias e2e="npm run test:e2e"

# Frontend E2E 调试
alias e2e-debug="npm run test:e2e:debug"
```
