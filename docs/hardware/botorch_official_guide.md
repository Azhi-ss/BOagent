# Official BoTorch & GPyTorch Integration Guide

This guide records the version-locked BoTorch 0.18.x patterns used by BOagent.

## 1. Surrogate Model

BOagent uses a CPU FP64 `SingleTaskGP` with a Matern-5/2 ARD kernel and explicit transforms:

```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel

train_X = torch.as_tensor(X_numpy, dtype=torch.float64)
train_Y = torch.as_tensor(y_numpy, dtype=torch.float64).unsqueeze(-1)

model = SingleTaskGP(
    train_X,
    train_Y,
    covar_module=ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])
    ),
    input_transform=Normalize(d=train_X.shape[-1]),
    outcome_transform=Standardize(m=1),
)
```

Fit with `ExactMarginalLogLikelihood` and `fit_gpytorch_mll_scipy`. The BOagent implementation bounds the SciPy iterations, allows up to 80 L-BFGS-B line-search steps per iteration, and retries only `NotPSDError` with configured Cholesky jitter levels. The expanded line-search budget avoids the reproducible SciPy `ABNORMAL` termination caused by its default budget of 20. Its BoTorch path performs one deterministic SciPy fit; the shared `n_restarts` setting applies only to sklearn's kernel optimizer and is not a cross-backend equivalent. The successful fit jitter must also wrap posterior prediction and covariance extraction because GPyTorch otherwise restores its lower default inference jitter and can fail Cholesky decomposition on the same fitted model.

Do not equate sklearn `GaussianProcessRegressor(alpha=...)` with BoTorch Cholesky jitter. sklearn `alpha` is fixed model noise that changes the posterior; GPyTorch jitter only stabilizes matrix factorization, while `SingleTaskGP` learns likelihood noise. Cross-backend predictions are therefore not expected to be numerically identical.

## 2. Posterior Scale Contract

`model.posterior(X)` applies `Standardize.untransform_posterior` in BoTorch 0.18.1. Its mean, variance, covariance, and lazy covariance are already in the original target units. Do not multiply them by the training target standard deviation again.

`model(X)` is a lower-level latent distribution and does not provide this public transformed-output contract. BOagent surrogate callers must use `model.posterior(X)`.

```python
with torch.no_grad():
    posterior = model.posterior(test_X)
    mu = posterior.mean.squeeze(-1).cpu().numpy()
    variance = posterior.variance.squeeze(-1).cpu().numpy()
```

## 3. Filtered Warm Start

Each BO iteration constructs a new model so input and outcome transform statistics are computed from the growing training set. Warm starts copy only learned model hyperparameters:

```python
WARM_START_PREFIXES = (
    "covar_module.",
    "likelihood.",
    "mean_module.",
)
```

Do not load `input_transform.*` or `outcome_transform.*`. `strict=False` does not protect against stale transform buffers when matching keys exist.

The current SciPy fit path is deterministic and does not require `torch.manual_seed`. Do not mutate global Torch RNG state inside concurrent benchmark workers.

## 4. Discrete Candidate Scoring

Both current BOagent optimizers score an existing discrete pool. UCB, EI, and PI are computed from posterior mean and standard deviation over that pool. `optimize_acqf` is intentionally out of scope: continuous optimization can generate invalid categorical one-hot combinations.

Large BoTorch pool predictions are evaluated in batches of at most 512 rows. This preserves each point's public posterior mean and marginal variance while avoiding the large joint covariance allocation that `posterior(pool).variance` can trigger for Suzuki's 5,731 candidates.

For LGBO posterior cross-covariance, process the pool side in batches of at most 512 rows. Each batch concatenates only that pool block with the K=50 grid, materializes the public joint posterior covariance for at most 562 points, and retains only the requested at-most `512 x 50` cross block. Never request the full Suzuki pool covariance matrix.

## 5. Runtime Checklist

1. Construct every GP tensor with `torch.float64`; do not change the process-wide default dtype.
2. Use `torch.no_grad()` for posterior prediction and covariance extraction.
3. Use 10 numerical threads for a single H365 worker and 1 thread per process for multi-worker runs.
4. BoTorch/GPyTorch is the default backend; sklearn is available as an opt-in compatibility backend. Always record the selected backend in benchmark outputs.
5. Measure fit, prediction, covariance, wall time, and memory before claiming a speedup.
