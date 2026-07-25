# 电池正极材料合成优化评测集 — battery_cathode

## 数据集信息

| 属性 | 值 |
| --- | --- |
| 领域 | 锂离子电池正极材料 (LiFePO4 Cathode Synthesis) |
| 数据来源 | Text-to-BatteryRecipe (KIST-CSRC), 论文原文 NLP 抽取 |
| 优化目标 | 最大化 `Discharge_Capacity_mAh_g` (mAh/g) |
| 优化变量数量 | 4 |
| 搜索空间样本数 | 549 |
| 默认训练集样本数 | 10 |
| 测试样本数 | 539 |

## 特征与目标说明

| 列名 | 说明 |
| --- | --- |
| `Precursor` | 正极合成前驱体 (如 Li2CO3, FePO4, H3PO4) |
| `Sintering_Time` | 烧结保温时间 (如 10 h, 2 h) |
| `Atmosphere` | 烧结气氛 (如 Ar, N2, air) |
| `Solvent` | 合成溶剂 (如 deionized water, ethanol) |
| `Discharge_Capacity_mAh_g` | 目标: 放电比容量 (mAh/g), 理论上限 170 mAh/g |

## 数据清洗说明

原始 `cathode_recipes.xlsx` 共 2,840 条配方，经以下清洗步骤：
1. 剔除 `Target_Material` (59.5% 缺失) 和 `Sintering_Temp` (63.8% 缺失) 两列
2. 通过 DOI 关联 `raw_paragraphs.xlsx` 提取放电比容量作为目标变量 y
3. 过滤所有仍含缺失值的行，确保每条记录完整可用
4. 按工艺特征去重，同组取最高容量

## 文件结构

- `searchspace.csv`: 完整 549 条候选搜索空间（包含目标变量）。
- `train.csv`: 默认 10 条带标签初始化训练集。
- `test.csv`: 测试集带标签数据（作为 Oracle 反馈）。
- `test_features.csv`: 测试集不含目标变量的特征池（提供给 Agent 探索）。
- `options.json`: 各离散特征列的候选值列表。

## 目标变量提取方法

目标变量 `Discharge_Capacity_mAh_g` 通过正则表达式从论文原文段落中提取：
- 数据源文件: `raw_paragraphs.xlsx` (包含 46,602 条论文段落)
- 正则模式: `(\d+\.?\d*)\s*mAh\s*(?:g[−\-–]?1|/g)`
- 过滤范围: 10.0 ~ 200.0 mAh/g (LiFePO4 物理合理区间)
- 选取策略: 同一 DOI 内取最大值 (代表该配方的最佳性能)

## 数据来源文献

Text-to-Battery Recipe: A language modeling-based protocol for automatic
battery recipe extraction and retrieval (KIST-CSRC)
https://github.com/KIST-CSRC/Text-to-BatteryRecipe
