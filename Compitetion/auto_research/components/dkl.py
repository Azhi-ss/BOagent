"""Deep Kernel Learning (DKL).

Reference: "Deep Kernel Learning" (AISTATS 2016, arXiv:1511.02222).

DKL combines a non-linear deep neural network feature representation
:math:`\\phi(x)` with a Gaussian Process kernel :math:`k(\\phi(x), \\phi(x'))`,
allowing end-to-end learning of complex structures that a fixed kernel
cannot capture.

Concretely, the kernel is

    k_dkl(x, x') = k_base(\\phi(x), \\phi(x'))

where :math:`\\phi : \\mathbb{R}^d \\to \\mathbb{R}^h` is a small MLP and
:math:`k_base` is a Matern-5/2 (or RBF) kernel in feature space. All
parameters (NN weights + kernel hyperparameters) are jointly learned by
maximizing the GP marginal likelihood, so the NN learns a representation
that is useful for GP regression.

For categorical one-hot inputs (chemical reactions), the MLP maps the
one-hot vector to a dense learned embedding before applying the GP kernel.
This is analogous to learning a continuous embedding of discrete
reactions, which a fixed Matern on the raw one-hot cannot do.

Fallback: on any failure (NN init, fit, PSD), the surrogate falls back to
the submission's fixed Matern-5/2 kernel on the raw one-hot input, so
``use_kernel_opt=False`` and "DKL fit failed" produce identical behavior.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_SUBMISSION_CODE = Path(__file__).resolve().parents[3] / "submission" / "code"
if str(_SUBMISSION_CODE) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_CODE))

from bo_core.optimization.surrogate import (
    LBFGSB_MAX_LINE_SEARCH_STEPS,
    BoTorchSurrogate,
)


def _build_dkl_kernel(dimension: int, hidden_dim: int = 16, n_layers: int = 2):
    """Build a deep kernel: MLP feature extractor + Matern-5/2 in feature space.

    Architecture (small for d<=~100 one-hot):
      - Linear(d, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU
      - Then Matern-5/2 on the learned features

    Uses gpytorch's ScaleKernel wrapping a MaternKernel composed with
    a learned Linear+ReLU projection. The projection is implemented as
    a torch.nn.Module registered on the GP model (via the covar_module's
    feature_extractor hook).
    """
    import torch
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.module import Module

    class _MLPFeatureExtractor(Module):
        """Small MLP that maps input to a dense learned representation.

        Used as the feature_extractor of the deep kernel: the kernel
        operates on the MLP output, not the raw input.
        """
        def __init__(self, in_dim: int, hidden_dim: int, n_layers: int) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            last = in_dim
            for _ in range(max(1, n_layers)):
                layers.append(torch.nn.Linear(last, hidden_dim))
                layers.append(torch.nn.ReLU())
                last = hidden_dim
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Build the composite kernel: Matern-5/2 (feature space) wrapped in ScaleKernel.
    # The MLP is applied as a pre-kernel feature transform on the SingleTaskGP
    # via the model's input_transform; here we just return the base kernel and
    # the feature extractor, and the surrogate wires them together.
    feature_dim = hidden_dim
    base_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=feature_dim))
    feature_extractor = _MLPFeatureExtractor(dimension, hidden_dim, n_layers)
    return base_kernel, feature_extractor


# ---------------------------------------------------------------------------
# DKLSurrogate
# ---------------------------------------------------------------------------

class DKLSurrogate(BoTorchSurrogate):
    """BoTorchSurrogate subclass using a deep kernel (MLP + Matern-5/2).

    The MLP feature extractor maps the one-hot input to a dense learned
    embedding; the Matern-5/2 kernel operates on that embedding. All
    parameters (MLP weights + kernel hyperparameters) are jointly optimized
    by maximizing the GP marginal likelihood.

    On any failure (NN construction, fit, PSD), falls back to the
    submission's fixed Matern-5/2 kernel on the raw one-hot input.
    """

    def __init__(
        self,
        *,
        seed: int,
        alpha: float = 1e-4,
        jitter_levels: Sequence[float] | None = None,
        max_fit_iterations: int = 100,
        hidden_dim: int = 16,
        n_layers: int = 2,
    ) -> None:
        super().__init__(
            seed=seed,
            alpha=alpha,
            jitter_levels=jitter_levels,
            max_fit_iterations=max_fit_iterations,
        )
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self._fit_count = 0

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray):

        self._fit_count += 1
        train_X = self._tensor(X)
        train_Y = self._tensor(np.asarray(y, dtype=float).reshape(-1, 1))
        dimension = train_X.shape[-1]
        self.model = None

        # Try to build the deep kernel GP; fall back to fixed Matern-5/2 on failure
        try:
            self._fit_dkl(train_X, train_Y, dimension)
        except Exception as exc:  # noqa: BLE001 - DKL is best-effort
            print(f"[DKLSurrogate] DKL fit failed ({exc}); falling back to Matern-5/2")
            self._fit_matern_fallback(train_X, train_Y, dimension)

        return self

    def _fit_dkl(self, train_X, train_Y, dimension: int) -> None:
        """Fit the deep kernel GP (MLP + Matern-5/2 in feature space)."""
        import gpytorch
        from botorch.fit import fit_gpytorch_mll_scipy
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from linear_operator.utils.errors import NotPSDError

        # Build feature extractor + base kernel
        base_kernel, feature_extractor = _build_dkl_kernel(
            dimension, hidden_dim=self.hidden_dim, n_layers=self.n_layers
        )

        # Build a deep-kernel SingleTaskGP variant: wrap the input transform
        # so the kernel sees the MLP features.
        model = _DeepKernelGP(
            train_X,
            train_Y,
            feature_extractor=feature_extractor,
            covar_module=base_kernel,
            input_dim=dimension,
        )

        # Warm-start (skip for the first fit because NN init is random)
        if self._warm_state:
            current = model.state_dict()
            current.update(
                {
                    key: value
                    for key, value in self._warm_state.items()
                    if key in current
                    and getattr(current[key], "shape", None) == getattr(value, "shape", None)
                }
            )
            try:
                model.load_state_dict(current, strict=False)
            except Exception:  # noqa: BLE001, S110 - warm-start is best-effort
                pass

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
        }

    def _fit_matern_fallback(self, train_X, train_Y, dimension: int) -> None:
        """Fallback: plain Matern-5/2 on raw one-hot (identical to submission)."""
        import gpytorch
        from botorch.fit import fit_gpytorch_mll_scipy
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from linear_operator.utils.errors import NotPSDError

        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimension)),
            input_transform=Normalize(d=dimension),
            outcome_transform=Standardize(m=1),
        )
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
        }


# ---------------------------------------------------------------------------
# Deep kernel GP wrapper: SingleTaskGP with an MLP input transform
# ---------------------------------------------------------------------------

class _DeepKernelGP:
    """Minimal wrapper around SingleTaskGP that applies an MLP feature extractor.

    Because BoTorch's SingleTaskGP accepts an ``input_transform`` that must
    implement ``transform(X)`` and be a ``Module``, we wrap the MLP as an
    input transform: ``transform(x) = feature_extractor(x)``. The GP kernel
    then operates on the learned features.

    We return a SingleTaskGP instance with this input transform set, so all
    downstream predict/posterior/covariance calls flow through the MLP
    automatically.
    """

    def __new__(
        cls,
        train_X,
        train_Y,
        feature_extractor,
        covar_module,
        input_dim: int,
    ):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import InputTransform, Standardize
        from gpytorch.module import Module

        class _MLPInputTransform(Module, InputTransform):
            """InputTransform that applies the MLP feature extractor."""
            def __init__(self, mlp, output_dim: int) -> None:
                super().__init__()
                self.mlp = mlp
                self._output_dim = output_dim

            def transform(self, X):
                return self.mlp(X)

            def forward(self, X):
                return self.transform(X)

        mlp_transform = _MLPInputTransform(feature_extractor, feature_extractor.net[-2].out_features)
        # After the MLP, we want the Normalize to operate on the learned feature dim,
        # not the raw input dim. Compose: raw -> MLP -> Normalize.
        # SingleTaskGP applies input_transform before the kernel.
        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=covar_module,
            input_transform=mlp_transform,
            outcome_transform=Standardize(m=1),
        )
        return model


# ---------------------------------------------------------------------------
# Factory for the component registry
# ---------------------------------------------------------------------------

def create_dkl_surrogate(
    backend: str,
    seed: int,
    **kwargs: Any,
) -> DKLSurrogate:
    """Factory matching the SURROGATES registry signature."""
    if backend != "botorch":
        raise ValueError(f"DKLSurrogate only supports 'botorch' backend, got {backend!r}")
    return DKLSurrogate(
        seed=seed,
        alpha=kwargs.get("alpha", 1e-2),
        hidden_dim=kwargs.get("hidden_dim", 16),
        n_layers=kwargs.get("n_layers", 2),
    )
