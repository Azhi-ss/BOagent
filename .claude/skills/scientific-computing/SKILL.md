---
name: scientific-computing
description: 审查科学计算代码的数值稳定性、物理单位一致性和数学正确性
user-invocable: false
---

# Scientific Computing Review

当修改 `backend/optimization/` 或 `backend/benchmark/` 中的代码时，自动应用此技能审查科学计算的正确性。

## 审查清单

### 1. 数值稳定性

**必查项**:
- [ ] **除零保护**: 所有除法操作添加 epsilon（如 `1e-10`）
- [ ] **矩阵条件数**: 高斯过程协方差矩阵求逆前检查条件数
- [ ] **对数空间计算**: 使用 `np.log1p(x)` 而非 `np.log(1 + x)`
- [ ] **指数溢出**: 使用 `scipy.special.logsumexp` 处理 log-sum-exp
- [ ] **浮点比较**: 使用 `np.isclose()` 而非 `==`

**示例问题**:
```python
# ❌ 危险：可能除零
variance = sigma ** 2
ucb = mean + kappa * variance

# ✅ 安全：添加 epsilon
variance = sigma ** 2 + 1e-10
ucb = mean + kappa * np.sqrt(variance)
```

---

### 2. 物理一致性

**钙钛矿能带对齐公式**:
- `CBO = χ_PVK - χ_ETL` (导带偏移)
- `VBO = (χ_HTL + E_g_HTL) - χ_PVK` (价带偏移)
- `LUMO_barrier = LUMO_HTL - LUMO_PVK` (电子阻挡)

**必查项**:
- [ ] **单位一致性**: 能量单位统一为 eV
- [ ] **参数边界**: PCE ∈ [0, 100]，能带 ∈ 合理物理范围
- [ ] **公式正确性**: 对照论文验证物理公式
- [ ] **符号约定**: 确保正负号符合物理意义

---

### 3. 贝叶斯优化实现

**高斯过程**:
- [ ] **核函数选择**: RBF/Matern 核的长度尺度合理
- [ ] **噪声项**: `alpha` 参数防止矩阵奇异
- [ ] **归一化**: 输入特征标准化到 [0, 1]

**采集函数**:
- [ ] **UCB**: `mean + kappa * std`，kappa 通常 ∈ [1, 3]
- [ ] **EI**: 期望改进计算使用 CDF 和 PDF
- [ ] **PI**: 改进概率阈值设置合理

**示例审查**:
```python
# 检查 UCB 实现
def ucb(mean, std, kappa=2.0):
    # ✅ 正确：std 已经是标准差
    return mean + kappa * std
    
    # ❌ 错误：重复开方
    # return mean + kappa * np.sqrt(std)
```

---

### 4. 数据验证

**输入验证**:
- [ ] **NaN 检查**: `np.isnan()` 检测缺失值
- [ ] **Inf 检查**: `np.isinf()` 检测无穷大
- [ ] **形状验证**: 矩阵维度匹配
- [ ] **范围验证**: 参数在定义域内

**示例**:
```python
def validate_data(X, y):
    assert not np.any(np.isnan(X)), "输入包含 NaN"
    assert not np.any(np.isinf(y)), "目标包含 Inf"
    assert X.shape[0] == y.shape[0], "样本数不匹配"
    assert np.all((X >= 0) & (X <= 1)), "特征未归一化"
```

---

### 5. 性能优化

**向量化操作**:
- [ ] 避免 Python 循环，使用 NumPy 广播
- [ ] 批量计算而非逐点计算
- [ ] 缓存重复计算结果（如协方差矩阵）

**内存管理**:
- [ ] 大矩阵使用 `dtype=np.float32` 节省内存
- [ ] 及时释放不再使用的数组
- [ ] 避免不必要的数组拷贝

---

## 输出格式

对每个问题标注严重程度：

| 级别 | 含义 | 示例 |
|------|------|------|
| **CRITICAL** | 数学错误或数值不稳定，会导致错误结果 | 除零、矩阵奇异、公式错误 |
| **HIGH** | 物理不一致或严重性能问题 | 单位错误、参数越界、O(n³) 循环 |
| **MEDIUM** | 代码质量问题，可能影响可维护性 | 缺少验证、硬编码常数 |
| **LOW** | 风格建议 | 变量命名、注释完整性 |

---

## 使用示例

当我修改 `backend/optimization/optimizer.py` 时，自动触发此技能：

```python
# 修改前
def calculate_ucb(self, X):
    mean, std = self.gp.predict(X, return_std=True)
    return mean + self.kappa * std  # ⚠️ 缺少数值稳定性检查

# 审查后建议
def calculate_ucb(self, X):
    mean, std = self.gp.predict(X, return_std=True)
    # ✅ 添加 epsilon 防止 std=0
    std = np.maximum(std, 1e-10)
    return mean + self.kappa * std
```

---

## 相关文件

- `backend/optimization/optimizer.py` - 贝叶斯优化核心
- `backend/optimization/knowledge.py` - 物理公式计算
- `backend/benchmark/bo_step.py` - 优化步进逻辑
- `backend/benchmark/data_loader.py` - 数据加载与验证
