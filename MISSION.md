# Mission: 为 BOagent 选择有化学意义的表示与核

## Why
理解表示空间与高斯过程核各自承担的职责，从而为 Buchwald 和 Suzuki 比赛任务设计有依据、可验证且不依赖测试标签的代理模型，而不是盲目更换核函数名称。

## Success looks like
- 能解释 One-Hot 为什么把不同试剂放成几乎等距的类别点
- 能判断分子指纹、连续描述符和学习型 embedding 分别适合什么相似度
- 能判断 ALAS 在什么输入表示上有意义，并设计公平的候选实验

## Constraints
- 两个比赛任务都是小样本、有限候选池、纯类别反应条件
- 不使用未查询测试产率进行表示、超参数或模型选择
- 候选算法先在 `competition/auto_research` 验证，不直接进入正式 `bo-core`

## Out of scope
- 当前不追求完整推导 GP 或 alpha-stable 分布理论
- 当前不直接实现或晋级新核
