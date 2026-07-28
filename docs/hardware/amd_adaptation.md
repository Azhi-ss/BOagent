# AMD Hardware Adaptation Guide

This document defines the supported BOagent execution path for the AMD Ryzen AI 9 H 365 system.

## Workload Mapping

| Hardware | BOagent use | Status |
|---|---|---|
| Zen 5 CPU, 10C/20T, AVX-512 | sklearn and BoTorch FP64 GP fitting, pool scoring, LGBO covariance | Supported and verified |
| Radeon 880M iGPU | No current GP workload | Deferred pending a measured FP64 benefit |
| XDNA 2 NPU | No GP workload; no FP64 support | Out of scope |

The installed PyTorch CPU build reports MKL, oneDNN, OpenMP, and AVX-512 capability. BOagent therefore keeps GP computation on the CPU and uses explicit `torch.float64` tensors.

## Thread Policy

- Single-worker benchmark: 10 numerical threads.
- Multi-worker chemical benchmark: one process per configuration and 1 numerical thread per process.
- Thread limits are configured at runner process startup, not inside surrogate instances.

This avoids BLAS/PyTorch oversubscription while keeping a single fit able to use the 10 physical cores.

## BoTorch Contract

The shared surrogate layer uses:

- `SingleTaskGP`
- `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))`
- `Normalize(d)` and `Standardize(m=1)`
- SciPy L-BFGS-B fitting with bounded iterations and jitter fallback
- public `model.posterior()` outputs in original target units
- filtered warm starts for `covar_module.*`, `likelihood.*`, and `mean_module.*` only

Transform state is recomputed from each growing training set. The implementation does not set global Torch dtype or RNG state.

## Discrete Search Spaces

BOagent scores UCB, EI, or PI directly over the existing candidate pool. It does not call `optimize_acqf` for the categorical chemical datasets because continuous optimization could generate invalid one-hot points.

LGBO batches pool-to-grid covariance so each public posterior call covers at most 512 pool rows plus the K=50 grid, then retains only the requested cross block. The Suzuki full-pool covariance must never be constructed.

## Optional Acceleration

`zentorch`, `torch.compile`, ROCm/DirectML, and NPU paths are not part of the current architecture. They require an isolated benchmark showing a repeatable improvement before adoption. No performance claim should be made from hardware capability alone.
