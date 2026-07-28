# PyTorch CPU Optimization on AMD Zen 5

This guide describes the measured BOagent runtime on the AMD Ryzen AI 9 H 365 (10C/20T).

## Verified Runtime

The current CPU-only PyTorch build reports MKL, oneDNN, OpenMP, and `CPU capability usage: AVX512`. CUDA and ROCm are disabled. This is the supported path for the small FP64 Gaussian processes used by BOagent.

## Thread Budgets

Thread allocation depends on process-level parallelism:

| Mode | Worker processes | Numerical threads per worker |
|---|---:|---:|
| Single GP benchmark | 1 | 10 |
| Multi-seed chemical benchmark | 2 or more | 1 |

A multi-worker run must not give every worker 10 BLAS/PyTorch threads. That creates oversubscription and makes timing unstable. `lgbo_runner` applies the budget at process startup through BLAS environment variables, `threadpoolctl`, and `torch.set_num_threads`.

For an isolated single-process run, the equivalent shell configuration is:

```bash
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=CORES
```

## FP64 Without Global State

Gaussian-process inputs and targets are explicitly converted to FP64 at the NumPy/Torch boundary:

```python
array = np.array(X, dtype=float, copy=True, order="C")
tensor = torch.from_numpy(array).to(dtype=torch.float64)
```

Do not call `torch.set_default_dtype(torch.float64)`. A process-wide dtype change can affect unrelated application code and tests.

## Features Not Enabled

`zentorch`, `torch.compile`, iGPU execution, and NPU execution are not enabled for the current GP path. The observed training sets are generally below 100 rows, where compilation and device-transfer overhead can dominate. Add any of these only after a reproducible benchmark demonstrates a benefit for fit or pool scoring.

## Performance Reporting

BoTorch is not assumed to be faster than sklearn. Benchmark cold and warm fits separately and report:

- GP fit time
- full-pool prediction time
- LGBO K=50 covariance time
- total wall time
- peak RSS
- worker and thread configuration
