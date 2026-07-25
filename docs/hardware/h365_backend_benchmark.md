# H365 sklearn vs BoTorch Backend Benchmark

This report measures surrogate work only; it excludes LLM calls and data loading.
Times are means across seeds. Peak RSS is the per-case process high-water mark.

| Dataset | Mode | Workers x threads | Fit (s) | Predict (s) | Covariance (s) | Total (s) | Peak RSS (MiB) | Opt. warnings/case |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| buchwald_sub4 | botorch_cold | 1 x 10 | 27.390 | 1.056 | 1.180 | 29.644 | 456.0 | 0.0 |
| buchwald_sub4 | botorch_cold | 4 x 1 | 24.249 | 1.506 | 1.666 | 27.437 | 453.0 | 0.0 |
| buchwald_sub4 | botorch_warm | 1 x 10 | 8.749 | 0.977 | 1.132 | 10.875 | 456.0 | 0.0 |
| buchwald_sub4 | botorch_warm | 4 x 1 | 6.383 | 1.361 | 1.613 | 9.372 | 454.4 | 0.0 |
| buchwald_sub4 | sklearn | 1 x 10 | 28.662 | 0.134 | 0.361 | 29.172 | 383.5 | 0.0 |
| buchwald_sub4 | sklearn | 4 x 1 | 25.970 | 0.128 | 0.208 | 26.319 | 381.1 | 0.0 |
| suzuki | botorch_cold | 1 x 10 | 26.698 | 5.759 | 7.476 | 39.985 | 481.0 | 0.0 |
| suzuki | botorch_cold | 4 x 1 | 25.795 | 8.925 | 12.505 | 47.276 | 476.8 | 0.0 |
| suzuki | botorch_warm | 1 x 10 | 9.415 | 6.069 | 7.956 | 23.487 | 480.8 | 0.0 |
| suzuki | botorch_warm | 4 x 1 | 9.606 | 9.376 | 12.465 | 31.503 | 477.6 | 0.0 |
| suzuki | sklearn | 1 x 10 | 13.837 | 0.925 | 1.983 | 16.799 | 410.4 | 0.0 |
| suzuki | sklearn | 4 x 1 | 7.900 | 0.880 | 1.563 | 10.389 | 406.9 | 0.0 |

## Interpretation

- buchwald_sub4 / 1 x 10: warm-start BoTorch was 2.68x faster than sklearn.
- buchwald_sub4 / 4 x 1: warm-start BoTorch was 2.81x faster than sklearn.
- suzuki / 1 x 10: warm-start BoTorch was 1.40x slower than sklearn.
- suzuki / 4 x 1: warm-start BoTorch was 3.03x slower than sklearn.
- BoTorch emitted 0 `OptimizationWarning` instances across 960 fits (0.0%). The cases completed without this convergence warning.
- BoTorch/GPyTorch is the default backend. sklearn is available as a compatibility backend via explicit selection.

## Configuration

- Steps per case: 40
- Seeds: [100, 200, 300]
- sklearn optimizer restarts: 0
- BoTorch maximum SciPy iterations: 100
- BoTorch L-BFGS-B line-search steps per iteration: 80
- `n_restarts` configures only sklearn's kernel optimizer; the BoTorch path performs one bounded deterministic SciPy fit.
- Covariance workload: one 50 x 50 posterior block and one pool x 50 cross block per step.
- Peak RSS is the high-water mark of one case process, not aggregate memory across concurrent workers.
- `botorch_warm` reuses fitted hyperparameters; `botorch_cold` creates a fresh model each step.
