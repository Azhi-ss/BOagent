# Hardware Adaptation & Acceleration Specs

This directory contains technical documentation, API guides, and official code patterns for deploying BOagent on specialized hardware accelerators (CPUs, GPUs, and NPUs) and deep learning frameworks (PyTorch, BoTorch, GPyTorch).

## Documents

1. [AMD Hardware Adaptation Guide](file:///home/dministrator/project/BOagent/docs/hardware/amd_adaptation.md)
   - Hardware mapping for Ryzen AI 9 / Zen 5, Radeon 880M, and XDNA 2 NPU.
   - ROCm 7.2+ WSL2 configuration (`ROCDXG`) and DirectML options.

2. [Official BoTorch & GPyTorch Integration Guide](file:///home/dministrator/project/BOagent/docs/hardware/botorch_official_guide.md)
   - Code patterns for `SingleTaskGP`, `fit_gpytorch_mll`, and data transforms.
   - Warm-start (`state_dict`) hyperparameter refitting strategy.
   - Discrete pool acquisition scoring (UCB, EI, PI) vs continuous `optimize_acqf`.

3. [Official PyTorch CPU & Zen 5 Optimization Guide](file:///home/dministrator/project/BOagent/docs/hardware/pytorch_cpu_optimization.md)
   - Thread tuning (OpenMP / MKL / `OMP_NUM_THREADS`).
   - AVX-512 vectorization and AMD ZenDNN (`zentorch`) integration.
   - `torch.compile` and FP64 precision rules for Gaussian Processes.
