# 化学表示与高斯过程核 Resources

## Knowledge

- [Book chapter: *Gaussian Processes for Machine Learning*, Chapter 4 — Rasmussen & Williams](https://gaussianprocess.org/gpml/chapters/RW4.pdf)
  GP 核的权威基础材料。用于理解核如何定义输入之间的“接近”和协方差。
- [Paper: *ALAS: Additive Learnable Alpha-Stable Kernels for Flexible Bayesian Optimization*](https://arxiv.org/abs/2607.18282)
  ALAS 的一手来源。用于核对 alpha、频率调制、ALAS 与 ALAS-Sep 的真实定义。
- [RDKit Book: Morgan and Feature Morgan fingerprints](https://rdkit.org/docs/RDKit_Book.html#morgan-and-feature-morgan-fingerprints)
  RDKit 官方指纹说明。用于理解结构片段如何映射为 bit/count vector。
- [RDKit Python guide: fingerprints and Tanimoto similarity](https://rdkit.org/docs/GettingStartedInPython.html)
  用于实现和检查 Morgan fingerprint 及分子相似度，而不是把指纹位直接当连续物性。

## Wisdom (Communities)

- [RDKit Discussions](https://github.com/rdkit/rdkit/discussions)
  适合核查分子解析、盐、配体和指纹参数等工程问题。

## Gaps

- 尚未为比赛所有 IUPAC 名称建立经过人工校验的结构映射。
- 尚未验证通用分子 embedding 是否能表达具体偶联反应中的配体、碱和溶剂效应。
