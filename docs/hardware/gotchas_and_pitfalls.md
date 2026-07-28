# BoTorch and PyTorch Pitfalls

These rules apply to BOagent's sklearn/BoTorch surrogate layer.

## 1. Posterior Output Scale

With `outcome_transform=Standardize(m=1)`, BoTorch 0.18.1 automatically calls `untransform_posterior` from the public `model.posterior(X)` API. Mean, variance, and covariance are already in original target units.

Do not manually multiply public posterior outputs by `y_std`; doing so scales the result twice and corrupts EI, UCB, LLM hybrid weights, and LGBO confidence calibration.

For sklearn `GaussianProcessRegressor(normalize_y=True)`, manually reconstructed kernel covariance is different: multiply by `_y_train_std**2` to restore original target variance units.

## 2. Warm-Start Transform State

Loading an entire prior `state_dict` with `strict=False` can still overwrite matching `input_transform.*` and `outcome_transform.*` buffers. This leaves new observations paired with stale normalization statistics.

Create a new `SingleTaskGP` each iteration and copy only:

```python
WARM_START_PREFIXES = (
    "covar_module.",
    "likelihood.",
    "mean_module.",
)
```

## 3. Noise and Cholesky Jitter Are Different

In sklearn, `GaussianProcessRegressor(alpha=...)` adds fixed model noise to the training covariance and therefore changes the fitted posterior. In BoTorch, `gpytorch.settings.cholesky_jitter(...)` is a numerical stabilization setting for matrix factorization, while `SingleTaskGP` learns likelihood noise from data. The shared `alpha` input selects each backend's existing stability policy; it does not make their posterior noise models equivalent.

The BoTorch factory raises numerical jitter below `1e-4` to that floor. This is intentional and must not be interpreted as matching sklearn `alpha=1e-4`.

## 4. Cholesky Failures

Catch `linear_operator.utils.errors.NotPSDError` during fitting and retry with explicit `gpytorch.settings.cholesky_jitter(double_value=...)` levels. Configure L-BFGS-B with up to 80 line-search steps per iteration; its default of 20 produced reproducible SciPy `ABNORMAL` terminations even when one additional line-search step reached the same optimum normally. If all jitter levels fail, clear the fit model and let the caller use its documented fallback. Do not return predictions from a stale model.

Retain the jitter level that completed fitting and apply it to `model.posterior(...)` prediction and covariance calls. The setting is context-local; leaving the fit context restores GPyTorch's default inference jitter, which can make posterior Cholesky decomposition fail even though fitting completed.

## 5. Categorical Acquisition

Do not use continuous `optimize_acqf` for the chemical one-hot candidate pools. Score the finite pool directly so every selected point maps to a valid dataset row.

## 6. Pool Prediction and Covariance Memory

A full Suzuki `M x M` posterior covariance is unnecessarily large. Even requesting `posterior(pool).variance` can cause the lazy operator to allocate a large joint covariance internally. Evaluate public posterior mean and marginal variance in pool batches of at most 512 rows.

LGBO needs only `K x K` grid covariance and `M x K` pool-to-grid cross-covariance as outputs. Process the pool side in batches: each public posterior call materializes a joint covariance for at most `512 + K` points, then retains only the at-most `512 x K` cross block before stacking the results. On the H365 smoke workload, batching pool prediction reduced Suzuki BoTorch peak RSS from approximately 1.71 GiB to 454 MiB; this is a local measurement, not a general memory guarantee.

## 7. Thread Oversubscription

A multi-seed run must not combine several workers with 10 or 20 numerical threads each. Use 10 threads for one worker and 1 thread per process for multiple workers. Configure this at process startup; never race on global thread settings from individual surrogate objects.

## 8. Global Process State

Do not call `torch.set_default_dtype` or `torch.manual_seed` inside surrogate fitting. Construct FP64 tensors explicitly and preserve the application's global RNG behavior.

## 9. NumPy Memory Ownership

Pandas can expose read-only NumPy views. Before `torch.from_numpy`, create a writable contiguous copy:

```python
array = np.array(X, dtype=float, copy=True, order="C")
```

This prevents undefined writes and suppresses the PyTorch read-only-array warning.
