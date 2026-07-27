"""Shared Gaussian-process surrogate backends for BOagent.

Both implementations consume NumPy arrays in the caller's feature space and
return predictions and covariances in the original target units.  Keeping that
contract here prevents the generic optimizer and LGBO's posterior mean shift
from developing different scaling behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any, Literal, Protocol, Self

import numpy as np
from scipy.linalg import cho_solve

BackendName = Literal["botorch", "sklearn"]
_MIN_STD = 1e-9
# SciPy's default maxls=20 terminates reproducibly with ABNORMAL on the chemical
# priors; 80 also covers the later Suzuki warm-start failure without random retries.
LBFGSB_MAX_LINE_SEARCH_STEPS = 80


def _validate_prediction(
    mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
        raise RuntimeError("Surrogate prediction contains non-finite values")
    return mean_array, np.maximum(std_array, _MIN_STD)


def _validate_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance_array = np.asarray(covariance, dtype=float)
    if not np.all(np.isfinite(covariance_array)):
        raise RuntimeError("Surrogate covariance contains non-finite values")
    return covariance_array


class SurrogateModel(Protocol):
    """Minimal posterior contract shared by the two optimization engines."""

    @property
    def is_fit(self) -> bool: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self: ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def prior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray: ...

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray: ...

    def posterior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray: ...


class SklearnSurrogate:
    """Existing sklearn Matern-5/2 GP behind the shared posterior contract."""

    def __init__(
        self,
        *,
        seed: int,
        n_restarts: int = 10,
        alpha: float = 1e-6,
        jitter_levels: Sequence[float] | None = None,
    ) -> None:
        if find_spec("sklearn") is None:
            raise ImportError(
                "The sklearn compatibility backend requires the optional "
                "'scikit-learn' dependency (install with 'pip install bo-core[sklearn]')."
            )
        from sklearn.preprocessing import StandardScaler

        self.seed = seed
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.jitter_levels = tuple(jitter_levels or (alpha,))
        self.scaler = StandardScaler()
        self.model: Any | None = None

    @property
    def is_fit(self) -> bool:
        return (
            self.model is not None
            and getattr(self.model, "X_train_", None) is not None
            and getattr(self.model, "L_", None) is not None
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C
        from sklearn.gaussian_process.kernels import Matern

        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        X_scaled = self.scaler.fit_transform(X_array)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            [1.0] * X_array.shape[1], (1e-2, 1e2), nu=2.5
        )
        last_exc: Exception | None = None
        self.model = None
        for jitter in self.jitter_levels:
            candidate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                alpha=jitter,
                normalize_y=True,
                random_state=self.seed,
            )
            try:
                candidate.fit(X_scaled, y_array)
                self.model = candidate
                return self
            except Exception as exc:  # noqa: BLE001 - caller owns fallback policy
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No sklearn jitter levels configured")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = self._require_model()
        mu, sigma = model.predict(
            self.scaler.transform(np.asarray(X, dtype=float)), return_std=True
        )
        return _validate_prediction(mu, sigma)

    def prior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray:
        model = self._require_model()
        XA_scaled = self.scaler.transform(np.asarray(XA, dtype=float))
        XB_scaled = self.scaler.transform(np.asarray(XB, dtype=float))
        return _validate_covariance(model.kernel_(XA_scaled, XB_scaled))

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        model = self._require_model()
        X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
        target_variance = float(model._y_train_std) ** 2
        K_XX = model.kernel_(X_scaled)
        K_X_train = model.kernel_(X_scaled, model.X_train_)
        solved = cho_solve((model.L_, True), K_X_train.T)
        return _validate_covariance(
            (K_XX - K_X_train @ solved) * target_variance
        )

    def posterior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray:
        model = self._require_model()
        XA_scaled = self.scaler.transform(np.asarray(XA, dtype=float))
        XB_scaled = self.scaler.transform(np.asarray(XB, dtype=float))
        target_variance = float(model._y_train_std) ** 2
        K_AB = model.kernel_(XA_scaled, XB_scaled)
        K_A_train = model.kernel_(XA_scaled, model.X_train_)
        K_train_B = model.kernel_(model.X_train_, XB_scaled)
        solved = cho_solve((model.L_, True), K_train_B)
        return _validate_covariance(
            (K_AB - K_A_train @ solved) * target_variance
        )

    def _require_model(self) -> Any:
        if not self.is_fit or self.model is None:
            raise RuntimeError("Surrogate must be fit before prediction")
        return self.model


class BoTorchSurrogate:
    """CPU FP64 SingleTaskGP with filtered hyperparameter warm starts."""

    _WARM_START_PREFIXES = ("covar_module.", "likelihood.", "mean_module.")
    _PREDICTION_BATCH_SIZE = 512
    _CROSS_COVARIANCE_BATCH_SIZE = 512

    def __init__(
        self,
        *,
        seed: int,
        alpha: float = 1e-4,
        jitter_levels: Sequence[float] | None = None,
        max_fit_iterations: int = 100,
    ) -> None:
        self.seed = seed
        self.alpha = alpha
        self.jitter_levels = tuple(jitter_levels or (alpha, alpha * 10.0))
        self.max_fit_iterations = max_fit_iterations
        self.model: Any | None = None
        self._inference_jitter = float(self.jitter_levels[0])
        self._warm_state: dict[str, object] | None = None

    @property
    def is_fit(self) -> bool:
        return self.model is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        import gpytorch
        from botorch.fit import fit_gpytorch_mll_scipy
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from linear_operator.utils.errors import NotPSDError

        train_X = self._tensor(X)
        train_Y = self._tensor(np.asarray(y, dtype=float).reshape(-1, 1))
        self.model = None
        dimension = train_X.shape[-1]
        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=dimension)
            ),
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
                    and getattr(current[key], "shape", None)
                    == getattr(value, "shape", None)
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

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import gpytorch
        import torch

        model = self._require_model()
        X_array = np.asarray(X, dtype=float)
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        with (
            torch.no_grad(),
            gpytorch.settings.cholesky_jitter(
                double_value=self._inference_jitter
            ),
        ):
            for start in range(0, len(X_array), self._PREDICTION_BATCH_SIZE):
                X_block = X_array[start : start + self._PREDICTION_BATCH_SIZE]
                posterior = model.posterior(self._tensor(X_block))
                means.append(
                    posterior.mean.squeeze(-1).detach().cpu().numpy()
                )
                variances.append(
                    posterior.variance.squeeze(-1).detach().cpu().numpy()
                )
        if not means:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        mean_array = np.concatenate(means)
        variance_array = np.asarray(np.concatenate(variances), dtype=float)
        if not np.all(np.isfinite(variance_array)):
            raise RuntimeError("Surrogate posterior variance contains non-finite values")
        sigma = np.sqrt(np.maximum(variance_array, _MIN_STD**2))
        return _validate_prediction(mean_array, sigma)

    def prior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray:
        import torch

        model = self._require_model()
        with torch.no_grad():
            XA_transformed = model.transform_inputs(self._tensor(XA))
            XB_transformed = model.transform_inputs(self._tensor(XB))
            covariance = model.covar_module(
                XA_transformed, XB_transformed
            ).to_dense()
        return _validate_covariance(covariance.detach().cpu().numpy())

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        import gpytorch
        import torch

        model = self._require_model()
        with (
            torch.no_grad(),
            gpytorch.settings.cholesky_jitter(
                double_value=self._inference_jitter
            ),
        ):
            covariance = model.posterior(self._tensor(X)).covariance_matrix
        return _validate_covariance(covariance.detach().cpu().numpy())

    def posterior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray:
        import gpytorch
        import torch

        model = self._require_model()
        XA_array = np.asarray(XA, dtype=float)
        XB_array = np.asarray(XB, dtype=float)
        blocks: list[np.ndarray] = []
        with (
            torch.no_grad(),
            gpytorch.settings.cholesky_jitter(
                double_value=self._inference_jitter
            ),
        ):
            for start in range(0, len(XA_array), self._CROSS_COVARIANCE_BATCH_SIZE):
                XA_block = XA_array[
                    start : start + self._CROSS_COVARIANCE_BATCH_SIZE
                ]
                n_a = len(XA_block)
                joint_X = self._tensor(np.vstack([XA_block, XB_array]))
                covariance = model.posterior(joint_X).covariance_matrix
                covariance = covariance[..., :n_a, n_a:]
                blocks.append(
                    _validate_covariance(covariance.detach().cpu().numpy())
                )
        if not blocks:
            return np.empty((0, len(XB_array)), dtype=float)
        return np.vstack(blocks)

    @staticmethod
    def _tensor(X: np.ndarray):
        import torch

        array = np.array(X, dtype=float, copy=True, order="C")
        return torch.from_numpy(array).to(dtype=torch.float64)

    def _require_model(self):
        if self.model is None:
            raise RuntimeError("Surrogate must be fit before prediction")
        return self.model


def create_surrogate(
    backend: BackendName | str = "botorch",
    *,
    seed: int,
    n_restarts: int = 10,
    alpha: float = 1e-6,
    jitter_levels: Sequence[float] | None = None,
    max_fit_iterations: int = 100,
) -> SurrogateModel:
    """Create one of the two supported surrogate backends.

    BoTorch/GPyTorch is the primary backend; sklearn is an opt-in compatibility
    backend that requires the optional ``scikit-learn`` dependency.
    """
    if backend == "botorch":
        return BoTorchSurrogate(
            seed=seed,
            alpha=max(alpha, 1e-4),
            jitter_levels=jitter_levels,
            max_fit_iterations=max_fit_iterations,
        )
    if backend == "sklearn":
        return SklearnSurrogate(
            seed=seed,
            n_restarts=n_restarts,
            alpha=alpha,
            jitter_levels=jitter_levels,
        )
    raise ValueError(
        f"Unknown surrogate backend: {backend!r}. Use 'botorch' (default) or 'sklearn'."
    )
