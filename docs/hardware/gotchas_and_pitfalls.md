# Critical Edge Cases, Pitfalls & Defensive Rules for BoTorch / PyTorch

This document records **critical pitfalls, numerical stability traps, and thread-safety bugs** that will cause runtime crashes or performance degradation if overlooked during PyTorch / BoTorch integration.

---

## 1. Covariance Matrix Decomposition Failures (`NotPSDError`)

### The Pitfall
In iterative Bayesian Optimization, covariance matrices $K(X, X)$ can become ill-conditioned when candidate points are close to each other. In GPyTorch, if the Cholesky decomposition fails, it raises `gpytorch.utils.errors.NotPSDError`, crashing the entire BO step.

### Defensive Fix
Wrap model fitting and posterior predictions in `gpytorch.settings.cholesky_jitter`:

```python
import gpytorch

# Use fallback jitter if numerical instability occurs
try:
    with gpytorch.settings.cholesky_jitter(1e-4):
        fit_gpytorch_mll(mll)
except gpytorch.utils.errors.NotPSDError:
    # Aggressive jitter fallback to prevent server crash
    with gpytorch.settings.cholesky_jitter(1e-3):
        fit_gpytorch_mll(mll)
```

---

## 2. CPU Denormal Floating-Point Slowdown (`flush_denormal`)

### The Pitfall
On x86 CPUs with AVX-512 (like AMD Zen 5), when floating-point numbers become extremely close to zero ($< 10^{-38}$), the hardware drops into CPU microcode exception handlers to process "denormal" numbers. This can cause matrix operations to run **2x to 10x slower** out of nowhere.

### Defensive Fix
Enable flush-to-zero at application startup:

```python
import torch

# Flush subnormal/denormal floating-point numbers to zero on CPU
if torch.get_num_threads() > 1:
    torch.set_flush_denormal(True)
```

---

## 3. Outcome Scale Parity with LLM Hybrid Scoring

### The Pitfall
BOagent computes hybrid LLM+GP scores using `lambda_t = gamma * std(GP_scores)`.
If `SingleTaskGP` standardizes the target `y_train` to $\mathcal{N}(0, 1)$, predicting on `X_pool` yields standardized $\mu_{\text{scaled}}$ and $\sigma_{\text{scaled}}$.

If these are not explicitly unscaled back to original target units (e.g. yield percentage $0 \sim 100\%$), then:
- $\sigma_{\text{scaled}}$ will be $\approx 1.0$ instead of actual standard deviation (e.g. $15.2\%$).
- The LLM log-prob weight `lambda_t` will be off by orders of magnitude, breaking the physics-informed LLM refinement.

### Defensive Fix
Always unscale posterior predictions before returning to `optimizer.py`:

```python
y_mean = float(np.mean(y_train))
y_std = float(np.std(y_train))
if y_std < 1e-9:
    y_std = 1.0

# Predict in scaled space
mu_scaled = posterior.mean.squeeze(-1).numpy()
sigma_scaled = np.sqrt(np.maximum(posterior.variance.squeeze(-1).numpy(), 1e-9))

# Strictly unscale back to original units!
mu = mu_scaled * y_std + y_mean
sigma = sigma_scaled * y_std
```

---

## 4. Multi-Feature ARD (Automatic Relevance Determination)

### The Pitfall
If `SingleTaskGP` is instantiated without ARD (Automatic Relevance Determination), it fits a single global lengthscale parameter $\ell$ shared across all input features. In chemical formulations where features have different physical dimensions (e.g. Temperature vs Concentration vs CBO), this severely degrades model fit.

### Defensive Fix
Ensure input dimensions use individual lengthscales:

```python
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, MaternKernel

d = train_X.shape[-1]
# Pass ard_num_dims=d so each feature gets its own learned lengthscale
covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
model = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
```

---

## 5. PyTorch Memory Accumulation in FastAPI / SSE Threads

### The Pitfall
In FastAPI backend streams (`apps/api/api.py`), multiple requests or multi-seed benchmarks run in worker threads. PyTorch builds autograd computational graphs by default. If `torch.no_grad()` is omitted during pool scoring, memory accumulates endlessly across 40 BO steps, causing RAM exhaustion.

### Defensive Fix
Always detach tensors and disable grad in candidate scoring methods:

```python
model.eval()
with torch.no_grad():
    posterior = model.posterior(test_X)
    mu = posterior.mean.squeeze(-1).detach().cpu().numpy()
```
