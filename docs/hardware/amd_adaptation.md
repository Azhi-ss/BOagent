# AMD Hardware Adaptation & Acceleration Guide

This document provides official configuration guidelines and setup instructions for running BOagent and PyTorch / BoTorch / GPyTorch on AMD Ryzen AI 300 series processors (Zen 5 + Radeon 880M + XDNA 2 NPU).

## 1. Hardware Overview & Workload Mapping

| Hardware Module | Architecture / Specs | Optimal BOagent Workload | Acceleration Library & Tech |
| :--- | :--- | :--- | :--- |
| **CPU** | AMD Ryzen AI 9 H 365 (Zen 5, 10C/20T, AVX-512) | **GPyTorch / BoTorch / GP Surrogate Fitting** (FP64 precision) | PyTorch `oneDNN` + AMD `ZenDNN` (`zentorch`), OpenMP (`OMP_NUM_THREADS=10`) |
| **iGPU** | AMD Radeon 880M (RDNA 3.5) | **LLM Embedding & Batch Candidate Feature Scoring** | AMD ROCm 7.2+ (via `ROCDXG` on WSL2) / `torch-directml` |
| **NPU** | AMD XDNA 2 NPU (50 TOPS) | **Local LLM Inference** (ONNX / INT8 Quantized Models) | AMD Ryzen AI SDK / Vitis AI Execution Provider |

---

## 2. CPU Acceleration Guide (Zen 5 + AVX-512)

### 2.1 Why CPU is Preferred for Gaussian Processes
Gaussian Process surrogate modeling requires Cholesky decomposition of covariance matrices $K \in \mathbb{R}^{N \times N}$. For typical BO sample sizes ($N < 1000$):
- Matrix inversions strictly require **FP64 (Double Precision)** to prevent non-positive definite matrix errors.
- Zen 5 features a full 512-bit wide AVX-512 execution pipeline with zero clock throttling penalties.
- Host-to-Device memory transfers over PCIe to GPU introduce latency that exceeds CPU computation time for small $N$.

### 2.2 Recommended Environment Variables
Set the following environment variables before starting BOagent backend processes:

```bash
# Set OpenMP thread allocation to physical cores (10 cores for Ryzen AI 9 H 365)
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

# Disable oneDNN verbose logging
export DNNL_VERBOSE=0

# Enable PyTorch JIT and compile optimization
export PYTORCH_TENSOREXPR_FALLBACK=0
```

### 2.3 AMD ZenDNN (`zentorch`) Integration
AMD provides `zentorch`, an official PyTorch plugin containing hand-optimized kernels for Zen 5 microarchitecture.
```bash
# Install AMD ZenDNN PyTorch plugin
pip install zentorch
```

---

## 3. iGPU Acceleration Guide (Radeon 880M + ROCm 7.x / WSL2)

### 3.1 Official ROCm 7.x WSL2 Support (`ROCDXG`)
AMD introduced production support for Strix / Strix Halo SKUs (Radeon 880M) on WSL2 using the `ROCDXG` (`librocdxg`) translation library.

#### Installation Steps (WSL2 / Ubuntu 22.04 / 24.04):
1. **Windows Host Driver**: Install AMD Software: Adrenalin Edition **26.2.2** or newer.
2. **WSL2 ROCm Setup**: Follow `amdgpu-install` script for ROCm 7.2.1+.
3. **PyTorch ROCm Installation**: Install PyTorch from AMD's official Radeon repository wheel:
   ```bash
   pip install torch torchvision --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
   ```
4. **Verification**:
   ```bash
   python -c "import torch; print('ROCm Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
   ```

### 3.2 Windows Native PyTorch DirectML
If running PyTorch directly on Windows Host (outside WSL2):
```bash
pip install torch-directml
```
```python
import torch
import torch_directml
dml = torch_directml.device()
# Tensor execution on Radeon 880M
x = torch.randn(100, 100).to(dml)
```

---

## 4. NPU Adaptations (XDNA 2 - 50 TOPS)

- **Target Workloads**: Low-precision (INT8 / FP16) neural network inference (e.g. `KnowledgeEngine` local LLM candidate scoring).
- **Constraints**: **No FP64 support** (cannot run GPyTorch / BoTorch matrix inversions).
- **Execution Provider**: Use ONNX Runtime with Vitis AI Execution Provider (`VitisAIExecutionProvider`) when loading quantized ONNX models.

---

## 5. BoTorch + GPyTorch Integration Parity

When upgrading `BayesianOptimizer` to BoTorch:
1. Always enforce `torch.float64` for `train_X`, `train_Y`, and `test_X`.
2. Apply Warm Start via `load_state_dict(old_state, strict=False)` to reuse hyperparameter fits across BO iterations.
3. Compute acquisition scores (UCB/EI/PI) on `posterior.mean` and `posterior.variance` to maintain 1:1 mathematical parity with the scikit-learn backend.
