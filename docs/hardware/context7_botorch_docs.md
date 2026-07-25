# Context7 Official Documentation Reference: BoTorch 0.18.x

This document stores the official API specifications, parameter tables, and usage contracts for **BoTorch 0.18.x** and **GPyTorch**, fetched via real-time documentation retrieval.

---

## 1. `botorch.models.SingleTaskGP` API Reference

`SingleTaskGP` is an exact Gaussian Process model designed for single-task regression, inheriting from `GPyTorchModel`.

### Constructor Signature
```python
SingleTaskGP(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    likelihood: Likelihood | None = None,
    covar_module: Module | None = None,
    input_transform: InputTransform | None = None,
    outcome_transform: OutcomeTransform | None = None
)
```

### Parameter Table

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`train_X`** | `Tensor` | Training features tensor of shape `(batch_shape, n, d)`. | *Required* |
| **`train_Y`** | `Tensor` | Target observations tensor of shape `(batch_shape, n, m)`. | *Required* |
| **`train_Yvar`** | `Tensor` | Optional observation noise variance tensor `(batch_shape, n, m)`. | `None` |
| **`likelihood`** | `Likelihood` | Custom GPyTorch Likelihood module. Defaults to `GaussianLikelihood`. | `None` |
| **`covar_module`** | `Module` | Custom Covariance Kernel. Defaults to `ScaleKernel(MaternKernel(nu=2.5))`. | `None` |
| **`input_transform`** | `InputTransform` | Transform for `train_X`, e.g. `Normalize(d=d)`. | `None` |
| **`outcome_transform`**| `OutcomeTransform`| Transform for `train_Y`, e.g. `Standardize(m=m)`. | `None` |

---

## 2. `botorch.optim.optimize_acqf` API Reference

Multi-start optimization utility for continuous acquisition functions.

### Constructor Signature
```python
optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple] | None = None,
    equality_constraints: list[tuple] | None = None,
    fixed_features: dict[int, float] | None = None,
    post_processing_func: Callable | None = None,
    batch_initial_conditions: Tensor | None = None,
    return_best_only: bool = True,
    sequential: bool = False
) -> tuple[Tensor, Tensor]
```

### Parameter Table

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`acq_function`** | `AcquisitionFunction` | Instance of `UpperConfidenceBound`, `ExpectedImprovement`, etc. |
| **`bounds`** | `Tensor` | A `2 x d` tensor specifying `[lower_bounds, upper_bounds]`. |
| **`q`** | `int` | Number of candidate points to generate in the batch. |
| **`num_restarts`** | `int` | Number of starting points for multi-start L-BFGS-B optimization. |
| **`raw_samples`** | `int` | Number of random samples evaluated to select initial conditions. |
| **`options`** | `dict` | Optimizer options (e.g. `{"batch_limit": 5, "maxiter": 200}`). |
| **`fixed_features`** | `dict` | Map of feature indices to fixed constant values during optimization. |
| **`return_best_only`**| `bool` | Returns shape `(q, d)` if `True`, or `(num_restarts, q, d)` if `False`. |

---

## 3. `botorch.fit.fit_gpytorch_mll` API Reference

Hyperparameter optimization wrapper around L-BFGS-B / Scipy.

```python
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

- **Warm Start Support**: To load previous state dict, execute `model.load_state_dict(old_state_dict, strict=False)` before calling `fit_gpytorch_mll(mll)`.
- **Numerical Guard**: Always wrap inside `with gpytorch.settings.cholesky_jitter(1e-4):` to handle ill-conditioned matrices.
