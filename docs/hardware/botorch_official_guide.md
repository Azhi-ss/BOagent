# Official BoTorch & GPyTorch Integration Guide

This guide documents the official API patterns, code examples, and best practices for integrating **BoTorch** and **GPyTorch** into BOagent.

---

## 1. SingleTaskGP Model Fitting

BoTorch provides `SingleTaskGP` for standard Gaussian Process surrogate modeling.

### Basic Code Pattern

```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# 1. Prepare training tensors (FP64 precision recommended)
# train_X shape: (N, d), train_Y shape: (N, 1)
train_X = torch.tensor(X_numpy, dtype=torch.float64)
train_Y = torch.tensor(y_numpy, dtype=torch.float64).unsqueeze(-1)

# 2. Instantiate SingleTaskGP with data transforms
gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=train_X.shape[-1]),
    outcome_transform=Standardize(m=1)
)

# 3. Define Marginal Log Likelihood (MLL)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# 4. Optimize hyperparameters
fit_gpytorch_mll(mll)
```

---

## 2. Hyperparameter Warm-Start (`state_dict`)

When adding new observations in sequential Bayesian Optimization loops, re-fitting hyperparameters from scratch with L-BFGS-B is computationally expensive.

### Recommended Warm-Start Pattern

To avoid issues with input/output transforms updating internal statistics, BoTorch recommends re-instantiating `SingleTaskGP` and loading the previous `state_dict`:

```python
# 1. Save state_dict from previous BO step
old_state = gp_model.state_dict()

# 2. Instantiate new model with updated observation dataset
new_gp_model = SingleTaskGP(new_train_X, new_train_Y)

# 3. Load previous hyperparameters (warm start)
new_gp_model.load_state_dict(old_state, strict=False)

# 4. Refit hyperparameters (converges in significantly fewer steps)
mll = ExactMarginalLogLikelihood(new_gp_model.likelihood, new_gp_model)
fit_gpytorch_mll(mll)
```

---

## 3. Acquisition Function Scoring

BoTorch supports both analytic and Monte Carlo (MC) acquisition functions.

### 3.1 Discrete Candidate Pool Scoring (BOagent Scenario)
In chemistry formulation optimization, candidates are drawn from a discrete search space (`pool_df`). Evaluating posterior mean and variance directly over the candidate tensor `X_pool` is faster than continuous multi-start optimization:

```python
gp_model.eval()
test_X = torch.tensor(X_pool_numpy, dtype=torch.float64)

with torch.no_grad():
    posterior = gp_model.posterior(test_X)
    mu = posterior.mean.squeeze(-1).numpy()
    variance = posterior.variance.squeeze(-1).numpy()
    sigma = np.sqrt(np.maximum(variance, 1e-9))

# Upper Confidence Bound (UCB)
scores_ucb = mu + kappa * sigma

# Expected Improvement (EI)
best_f = y_numpy.max()
imp = mu - best_f - xi
z = imp / sigma
scores_ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
```

### 3.2 Continuous Acquisition Optimization (`optimize_acqf`)
For continuous parameter spaces, use `botorch.optim.optimize_acqf`:

```python
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

# Define bounds: 2 x d tensor [[lower_bounds], [upper_bounds]]
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

# Instantiate acquisition function
EI = ExpectedImprovement(model=gp_model, best_f=best_f)

# Multi-start acquisition optimization
candidate, acq_value = optimize_acqf(
    acq_function=EI,
    bounds=bounds,
    q=1,               # Batch size
    num_restarts=20,   # Multistart restarts
    raw_samples=512    # Initial samples
)
```

---

## 4. Key Performance Checklist for BOagent

1. **Precision**: Always enforce `dtype=torch.float64` for GPyTorch matrices to prevent non-positive-definite errors.
2. **CPU Threading**: Set `torch.set_num_threads(10)` matching physical Zen 5 cores.
3. **No-Grad Inference**: Wrap all candidate pool predictions in `with torch.no_grad():` to avoid memory growth.
