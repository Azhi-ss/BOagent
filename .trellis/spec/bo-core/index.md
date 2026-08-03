# BO-Core (算法与核心逻辑) 规范入口

本目录包含了 `bo-core` 包（算法核心）的开发规范与强制约束。
在修改或新增算法逻辑（特别是高斯过程、知识引擎、嵌入检索等）之前，必须严格遵守以下规范。

## 目录
- [算法与物理约束规范](./algorithm-spec.md)
  - Acquisition score 尺度不变退化检测
  - 固定训练先验的可恢复 Hybrid 比较矩阵
  - Chem-LGBO 强制 Tool Calling、单次 ReAct、telemetry 与 provenance 契约

## 核心自检清单 (Pre-Development Checklist)
- [ ] 我是否确保了任何新增的数学/Parser模块都使用了 TDD（测试驱动开发）？
- [ ] 我是否绝对避免了修改 `pvk_llm_compat.py`？
- [ ] 我是否避免了使用全局的 `np.random`？

## Quality Check
执行代码检查前，请确认所有 `bo-core` 下的 `pytest` 用例全部通过，且绝对没有使用 `np.random.seed` 导致多线程冲突。
