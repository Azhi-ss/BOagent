"""Kernel Manifold: MDS-embedded kernel library + LML-based selection.

Reference: "The Kernel Manifold: A Geometric Approach to Gaussian Process Model
Selection" (arXiv:2601.05371).

This module implements a pragmatic variant of the paper's method:
  1. Define a compositional kernel library (base kernels + simple compositions)
  2. Pre-compute the Hellinger distance matrix between kernels (QMC over
     hyperparameters), then MDS-embed them into 2D for diagnostics.
  3. Every `evolve_interval` fits, select the kernel with the highest log
     marginal likelihood (LML) on the current observed data.

The full paper performs BO over the MDS embedding to pick the next kernel to
evaluate. With our small library (<= 11 kernels) we use exhaustive (greedy)
evaluation, which is the natural baseline that BO-on-manifold should beat.
The MDS embedding is computed and exposed for H7 (interpretability) but does
not drive selection in this implementation.

Fallback: on any failure (parse, fit, LML), the surrogate falls back to the
Matern-5/2 kernel — identical to the submission's fixed kernel — so
`use_kernel_opt=False` and "evolution failed" produce identical behavior.
"""
from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Make submission/code importable (read-only use)
_SUBMISSION_CODE = Path(__file__).resolve().parents[3] / "submission" / "code"
if str(_SUBMISSION_CODE) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_CODE))

from bo_core.optimization.surrogate import (
    LBFGSB_MAX_LINE_SEARCH_STEPS,
    BoTorchSurrogate,
)

# ---------------------------------------------------------------------------
# Kernel library and expression parsing (adapted from references/cake/gp.py)
# ---------------------------------------------------------------------------

def _base_kernel_factory(name: str, d: int):
    """Return a fresh gpytorch base kernel instance for `name` in dim `d`."""
    from gpytorch.kernels import (
        LinearKernel,
        MaternKernel,
        PeriodicKernel,
        RBFKernel,
        RQKernel,
    )
    table = {
        "SE": lambda: RBFKernel(ard_num_dims=d),
        "PER": lambda: PeriodicKernel(ard_num_dims=d),
        "LIN": lambda: LinearKernel(ard_num_dims=d),
        "RQ": lambda: RQKernel(ard_num_dims=d),
        "M1": lambda: MaternKernel(nu=0.5, ard_num_dims=d),
        "M3": lambda: MaternKernel(nu=1.5, ard_num_dims=d),
        "M5": lambda: MaternKernel(nu=2.5, ard_num_dims=d),
    }
    if name not in table:
        raise ValueError(f"Unknown base kernel: {name!r}")
    return table[name]()


def parse_kernel(expression: str, d: int):
    """Parse a kernel expression like 'SE+PER' into a gpytorch ScaleKernel.

    Supports '+' and '*' over base kernels; parentheses allowed for grouping.
    Mirrors references/cake/gp.py:parse_kernel but with fresh kernel factories
    so each call returns independent module instances.
    """
    from gpytorch.kernels import ScaleKernel

    base_kernels: dict[str, Any] = {name: _base_kernel_factory(name, d) for name in
                                    ("SE", "PER", "LIN", "RQ", "M1", "M3", "M5")}

    def apply_op(left, op, right):
        if op == "+":
            return left + right
        if op == "*":
            return left * right
        raise ValueError(f"Unknown operator: {op!r}")

    def parse_subexpr(subexpr: str):
        names = re.findall(r"\w+", subexpr)
        ops = re.findall(r"[+*]", subexpr)
        if not names:
            raise ValueError(f"Empty subexpression: {subexpr!r}")
        if names[0] not in base_kernels:
            raise ValueError(f"Unknown kernel name: {names[0]!r}")
        result = base_kernels[names[0]]
        for i, op in enumerate(ops):
            nxt = names[i + 1]
            if nxt not in base_kernels:
                raise ValueError(f"Unknown kernel name: {nxt!r}")
            result = apply_op(result, op, base_kernels[nxt])
        return ScaleKernel(result)

    pattern = r"\(([^()]+)\)"
    cache: dict[str, Any] = {}
    while "(" in expression:
        for subexpr in re.findall(pattern, expression):
            if subexpr not in cache:
                sub_kernel = parse_subexpr(subexpr)
                cache[subexpr] = sub_kernel
                base_kernels[f"SubKernel{len(base_kernels)}"] = sub_kernel
            expression = expression.replace(f"({subexpr})", f"SubKernel{len(base_kernels) - 1}", 1)
    return parse_subexpr(expression)


# Default kernel library: 7 base + 4 simple compositions
DEFAULT_KERNEL_LIBRARY: list[str] = [
    "SE", "PER", "LIN", "RQ", "M1", "M3", "M5",
    "SE+PER", "SE*RQ", "M5+LIN", "SE+M5",
]


# ---------------------------------------------------------------------------
# Hellinger distance + MDS embedding (precompute once per dimension)
# ---------------------------------------------------------------------------

def hellinger_distance_matrix(
    kernel_library: list[str],
    d: int,
    n_ref: int = 30,
    n_qmc: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Expected squared Hellinger distance between GP priors induced by each kernel.

    For each pair (i, j), sample n_qmc lengthscale settings per kernel from a
    uniform [0.1, 2.0] prior, compute the squared Hellinger distance between the
    resulting N(0, K_i) and N(0, K_j) priors on `n_ref` reference points, and
    average over the QMC grid.

    Returns an (n_lib, n_lib) symmetric matrix of distances in [0, 1].
    """
    import torch

    rng = np.random.RandomState(seed)
    # Reference input points in [0, 1]^d (first axis varied, others held at 0.5)
    x_ref = np.full((n_ref, d), 0.5, dtype=float)
    x_ref[:, 0] = np.linspace(0.0, 1.0, n_ref)
    X_ref = torch.from_numpy(x_ref).to(dtype=torch.float64)

    # For each kernel, compute a list of prior covariance matrices under
    # sampled hyperparameters.
    covs_per_kernel: list[list[np.ndarray]] = []
    for kexpr in kernel_library:
        covs: list[np.ndarray] = []
        for _ in range(n_qmc):
            try:
                kernel = parse_kernel(kexpr, d)
                # Sample lengthscale from uniform [0.1, 2.0] per dim where applicable
                lengthscale = rng.uniform(0.1, 2.0, size=d)
                _assign_lengthscale(kernel, lengthscale)
                K = kernel(X_ref).to_dense().detach().cpu().numpy()
                # Add tiny jitter for numerical PD
                K = K + 1e-6 * np.eye(n_ref)
                # Sanity-check PD
                sign, _ = np.linalg.slogdet(K)
                if sign <= 0:
                    continue
                covs.append(K)
            except Exception:  # noqa: BLE001, S112 - kernel may be unfit; skip
                continue
        covs_per_kernel.append(covs)

    n = len(kernel_library)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            total = 0.0
            count = 0
            for Ki in covs_per_kernel[i]:
                for Kj in covs_per_kernel[j]:
                    h_sq = _hellinger_sq_zero_mean(Ki, Kj)
                    if not np.isfinite(h_sq):
                        continue
                    total += h_sq
                    count += 1
            d_ij = total / count if count > 0 else 1.0
            D[i, j] = D[j, i] = float(np.clip(d_ij, 0.0, 1.0))
    return D


def _assign_lengthscale(kernel, lengthscale: np.ndarray) -> None:
    """Best-effort lengthscale assignment to a (possibly composite) kernel."""
    import torch
    ls_tensor = torch.tensor(lengthscale, dtype=torch.float64)
    # ScaleKernel wraps a base_kernel
    if hasattr(kernel, "base_kernel") and kernel.base_kernel is not None:
        bk = kernel.base_kernel
        if hasattr(bk, "lengthscale"):
            try:
                bk.lengthscale = ls_tensor
            except Exception:  # noqa: BLE001, S110 - lengthscale shape mismatch; skip
                pass
    elif hasattr(kernel, "lengthscale"):
        try:
            kernel.lengthscale = ls_tensor
        except Exception:  # noqa: BLE001, S110 - lengthscale shape mismatch; skip
            pass


def _hellinger_sq_zero_mean(Ki: np.ndarray, Kj: np.ndarray) -> float:
    """Squared Hellinger distance between N(0, Ki) and N(0, Kj).

    H^2(N0,N1) = 1 - [det(Ki)^(1/4) * det(Kj)^(1/4)] / det((Ki+Kj)/2)^(1/2).
    Computed in log-space for stability.
    """
    sign_i, logdet_i = np.linalg.slogdet(Ki)
    sign_j, logdet_j = np.linalg.slogdet(Kj)
    Kavg = 0.5 * (Ki + Kj)
    sign_a, logdet_a = np.linalg.slogdet(Kavg)
    if sign_i <= 0 or sign_j <= 0 or sign_a <= 0:
        return float("nan")
    log_num = 0.25 * (logdet_i + logdet_j)
    log_den = 0.5 * logdet_a
    log_ratio = log_num - log_den
    h_sq = 1.0 - math.exp(log_ratio)
    return float(max(0.0, min(1.0, h_sq)))


def mds_embed(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS embedding of a distance matrix into R^{n_components}."""
    from sklearn.manifold import MDS
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress="auto",
    )
    return mds.fit_transform(D)


# ---------------------------------------------------------------------------
# ManifoldSurrogate: BoTorchSurrogate with kernel evolution
# ---------------------------------------------------------------------------

class ManifoldSurrogate(BoTorchSurrogate):
    """BoTorchSurrogate subclass that evolves the kernel via LML selection.

    Every `evolve_interval` calls to `fit()`, evaluates every kernel in
    `kernel_library` on the current (X, y), picks the kernel with the highest
    log marginal likelihood, and uses it for subsequent fits. On any failure
    the surrogate falls back to the Matern-5/2 kernel (matching the
    submission's fixed-kernel behavior exactly).
    """

    def __init__(
        self,
        *,
        seed: int,
        alpha: float = 1e-4,
        jitter_levels: Sequence[float] | None = None,
        max_fit_iterations: int = 100,
        kernel_library: list[str] | None = None,
        evolve_interval: int = 5,
        selection_max_iter: int = 50,
        compute_manifold: bool = True,
        n_ref: int = 30,
        n_qmc: int = 8,
    ) -> None:
        super().__init__(
            seed=seed,
            alpha=alpha,
            jitter_levels=jitter_levels,
            max_fit_iterations=max_fit_iterations,
        )
        self.kernel_library = list(kernel_library or DEFAULT_KERNEL_LIBRARY)
        self.evolve_interval = max(1, int(evolve_interval))
        self.selection_max_iter = int(selection_max_iter)
        self.compute_manifold = bool(compute_manifold)
        self.n_ref = int(n_ref)
        self.n_qmc = int(n_qmc)

        self._fit_count = 0
        self._current_kernel = "M5"
        self._kernel_history: list[tuple[int, str, float]] = []  # (fit_idx, kernel, lml)
        # Cached manifold artifacts (depend only on dimension d, not data)
        self._distance_matrix: np.ndarray | None = None
        self._embedding: np.ndarray | None = None
        self._manifold_dim: int | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray):
        import gpytorch
        from botorch.fit import fit_gpytorch_mll_scipy
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from linear_operator.utils.errors import NotPSDError

        self._fit_count += 1
        train_X = self._tensor(X)
        train_Y = self._tensor(np.asarray(y, dtype=float).reshape(-1, 1))
        dimension = train_X.shape[-1]
        self.model = None

        # Trigger kernel evolution on the first fit and every `evolve_interval`
        if self._fit_count == 1 or self._fit_count % self.evolve_interval == 1:
            try:
                self._evolve_kernel(train_X, train_Y, dimension)
            except Exception as exc:  # noqa: BLE001 - kernel evolution is best-effort
                print(f"[ManifoldSurrogate] kernel evolution failed: {exc}; using M5")
                self._current_kernel = "M5"

        # Build the GP with the current (evolved) kernel
        try:
            covar_module = parse_kernel(self._current_kernel, dimension)
        except Exception as exc:  # noqa: BLE001 - parse fallback is best-effort
            print(f"[ManifoldSurrogate] kernel parse failed for {self._current_kernel!r}: {exc}; fallback M5")
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimension))
            self._current_kernel = "M5"

        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=dimension),
            outcome_transform=Standardize(m=1),
        )
        if self._warm_state:
            current = model.state_dict()
            current.update(
                {
                    key: value
                    for key, value in self._warm_state.items()
                    if key in current
                    and key.startswith(self._WARM_START_PREFIXES)
                    and getattr(current[key], "shape", None) == getattr(value, "shape", None)
                }
            )
            model.load_state_dict(current)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        last_exc: Exception | None = None
        for jitter in self.jitter_levels:
            try:
                with gpytorch.settings.cholesky_jitter(double_value=float(jitter)):
                    fit_gpytorch_mll_scipy(
                        mll,
                        options={
                            "maxiter": self.max_fit_iterations,
                            "maxls": LBFGSB_MAX_LINE_SEARCH_STEPS,
                        },
                    )
                self._inference_jitter = float(jitter)
                last_exc = None
                break
            except NotPSDError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc

        model.eval()
        self.model = model
        self._warm_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key.startswith(self._WARM_START_PREFIXES)
        }
        return self

    # -------------------------------------------------------------- evolution

    def _evolve_kernel(self, train_X, train_Y, dimension: int) -> None:
        """Evaluate every kernel in the library and pick the one with max LML.

        Also computes the Hellinger distance matrix + MDS embedding on the
        first evolution step (cached by dimension) for diagnostics.
        """
        import gpytorch
        import torch
        from botorch.fit import fit_gpytorch_mll_scipy
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from linear_operator.utils.errors import NotPSDError

        # First-time: compute manifold (cached by dimension)
        if self.compute_manifold and (
            self._embedding is None or self._manifold_dim != dimension
        ):
            try:
                D = hellinger_distance_matrix(
                    self.kernel_library, dimension,
                    n_ref=self.n_ref, n_qmc=self.n_qmc, seed=self.seed,
                )
                self._distance_matrix = D
                self._embedding = mds_embed(D, n_components=2)
                self._manifold_dim = dimension
            except Exception as exc:  # noqa: BLE001 - manifold computation is best-effort
                print(f"[ManifoldSurrogate] manifold computation failed: {exc}")
                self._embedding = None

        # Greedy: evaluate every kernel in library, pick max LML
        lmls: dict[str, float] = {}
        for kexpr in self.kernel_library:
            try:
                covar_module = parse_kernel(kexpr, dimension)
                model = SingleTaskGP(
                    train_X, train_Y,
                    covar_module=covar_module,
                    input_transform=Normalize(d=dimension),
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                with gpytorch.settings.cholesky_jitter(double_value=float(self.alpha)):
                    fit_gpytorch_mll_scipy(
                        mll,
                        options={
                            "maxiter": self.selection_max_iter,
                            "maxls": LBFGSB_MAX_LINE_SEARCH_STEPS,
                        },
                    )
                with torch.no_grad():
                    output = model(train_X)
                    lml = mll(output, train_Y.squeeze(-1)).item()
                if np.isfinite(lml):
                    lmls[kexpr] = lml
            except (NotPSDError, Exception):  # noqa: BLE001, S112 - kernel unfit for this data; skip
                # Kernel unfit for this data; skip
                continue

        if not lmls:
            self._current_kernel = "M5"
            return

        best_kernel = max(lmls, key=lmls.get)
        prev = self._current_kernel
        self._current_kernel = best_kernel
        self._kernel_history.append((self._fit_count, best_kernel, lmls[best_kernel]))
        print(
            f"[ManifoldSurrogate] evolve@fit#{self._fit_count}: {prev} -> {best_kernel} "
            f"(LML={lmls[best_kernel]:.4f}; evaluated {len(lmls)}/{len(self.kernel_library)})"
        )

    # -------------------------------------------------------------- accessors

    @property
    def current_kernel(self) -> str:
        return self._current_kernel

    @property
    def kernel_history(self) -> list[tuple[int, str, float]]:
        return list(self._kernel_history)

    @property
    def distance_matrix(self) -> np.ndarray | None:
        return None if self._distance_matrix is None else self._distance_matrix.copy()

    @property
    def embedding(self) -> np.ndarray | None:
        return None if self._embedding is None else self._embedding.copy()


# ---------------------------------------------------------------------------
# Factory for the component registry
# ---------------------------------------------------------------------------

def create_manifold_surrogate(
    backend: str,
    seed: int,
    **kwargs: Any,
) -> ManifoldSurrogate:
    """Factory matching the SURROGATES registry signature."""
    if backend != "botorch":
        raise ValueError(f"ManifoldSurrogate only supports 'botorch' backend, got {backend!r}")
    return ManifoldSurrogate(
        seed=seed,
        alpha=kwargs.get("alpha", 1e-2),
        kernel_library=kwargs.get("kernel_library"),
        evolve_interval=kwargs.get("evolve_interval", 5),
        selection_max_iter=kwargs.get("selection_max_iter", 50),
        compute_manifold=kwargs.get("compute_manifold", True),
        n_ref=kwargs.get("n_ref", 30),
        n_qmc=kwargs.get("n_qmc", 8),
    )
