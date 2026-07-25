# Official PyTorch CPU & Zen 5 Optimization Guide

This guide documents official performance tuning strategies for PyTorch on modern x86_64 CPUs, specifically AMD Zen 5 processors with AVX-512.

---

## 1. Threading & OpenMP Configuration

PyTorch relies on OpenMP and MKL/oneDNN backends for multithreaded matrix operations. Incorrect thread settings can cause thread contention and performance degradation.

### Environment Setup for AMD Ryzen AI 9 (10 Physical Cores)

```bash
# Set OpenMP thread count to physical core count (not logical hyperthreads)
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

# Thread affinity binding for Linux / WSL2
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=CORES

# Suppress oneDNN verbose runtime output
export DNNL_VERBOSE=0
```

---

## 2. oneDNN & ZenDNN (`zentorch`) Acceleration

### 2.1 Native oneDNN AVX-512
PyTorch automatically dispatches 512-bit vector kernels via `oneDNN` when running on CPUs supporting AVX-512 (like AMD Zen 5).

To verify AVX-512 kernel execution at runtime:
```bash
DNNL_VERBOSE=1 python -c "import torch; x=torch.randn(1000,1000); y=x@x"
```
Look for `avx512` in the log output.

### 2.2 AMD ZenDNN (`zentorch`)
AMD provides `zentorch` plugin specifically optimized for Zen architecture:
```bash
pip install zentorch
```

In Python code:
```python
import torch
import zentorch

# ZenDNN optimized GEMM and attention ops are automatically registered
```

---

## 3. `torch.compile` Graph Optimizations

For repeated batch calculations, wrap models in `torch.compile`:

```python
import torch

# Compile model using Inductor backend
compiled_gp = torch.compile(gp_model, mode="reduce-overhead")
```

---

## 4. Double Precision (FP64) Considerations for GP

While neural networks use `bfloat16` or `float32`, Gaussian Processes **must use `float64`**:

```python
# Set default float dtype for GPyTorch / BoTorch operations
torch.set_default_dtype(torch.float64)
```
Zen 5 handles FP64 AVX-512 vector operations efficiently without frequency throttling.
