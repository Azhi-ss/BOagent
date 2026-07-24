# Hardware Adaptation & Acceleration Specs

This directory contains technical documentation, Context7 API references, official code patterns, and defensive programming specs for deploying BOagent on specialized hardware accelerators (CPUs, GPUs, and NPUs) and deep learning frameworks (PyTorch, BoTorch, GPyTorch).

## Documents

1. [AMD Hardware Adaptation Guide](file:///home/dministrator/project/BOagent/docs/hardware/amd_adaptation.md)
   - Hardware mapping for Ryzen AI 9 / Zen 5, Radeon 880M, and XDNA 2 NPU.
   - ROCm 7.2+ WSL2 configuration (`ROCDXG`) and DirectML options.

2. [Context7 Official BoTorch 0.18.x API Documentation](file:///home/dministrator/project/BOagent/docs/hardware/context7_botorch_docs.md)
   - Context7 real-time API specifications for `SingleTaskGP`, `optimize_acqf`, and `fit_gpytorch_mll`.
   - Parameter tables, constructor signatures, and constraint options.

3. [Official BoTorch & GPyTorch Integration Guide](file:///home/dministrator/project/BOagent/docs/hardware/botorch_official_guide.md)
   - Code patterns for `SingleTaskGP`, `fit_gpytorch_mll`, and data transforms.
   - Warm-start (`state_dict`) hyperparameter refitting strategy.
   - Discrete pool acquisition scoring (UCB, EI, PI) vs continuous `optimize_acqf`.

4. [Official PyTorch CPU & Zen 5 Optimization Guide](file:///home/dministrator/project/BOagent/docs/hardware/pytorch_cpu_optimization.md)
   - Thread tuning (OpenMP / MKL / `OMP_NUM_THREADS`).
   - AVX-512 vectorization and AMD ZenDNN (`zentorch`) integration.
   - `torch.compile` and FP64 precision rules for Gaussian Processes.

5. [Critical Pitfalls & Defensive Rules](file:///home/dministrator/project/BOagent/docs/hardware/gotchas_and_pitfalls.md)
   - Handling `NotPSDError` with `cholesky_jitter` context managers.
   - Avoiding CPU denormal float slowdown with `torch.set_flush_denormal(True)`.
   - Outcome unscaling parity for hybrid LLM+GP scoring (`lambda_t`).
   - Multi-feature ARD lengthscales (`ard_num_dims`).
   - PyTorch memory leak prevention in FastAPI/SSE threads.
