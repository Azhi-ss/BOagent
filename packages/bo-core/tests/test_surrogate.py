from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bo_core.optimization.surrogate import (
    BoTorchSurrogate,
    SklearnSurrogate,
    create_surrogate,
)


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    X = rng.uniform(-1.0, 1.0, size=(10, 3))
    y = 25.0 + 7.0 * X[:, 0] - 4.0 * X[:, 1] + 2.0 * X[:, 2]
    X_test = rng.uniform(-1.0, 1.0, size=(7, 3))
    return X, y, X_test


@pytest.fixture(autouse=True)
def _suppress_gp_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_default_backend_creates_botorch_surrogate():
    surrogate = create_surrogate(seed=42)

    assert isinstance(surrogate, BoTorchSurrogate)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown surrogate backend"):
        create_surrogate("unknown", seed=42)


def test_sklearn_backend_fails_at_construction_when_dependency_is_missing():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "bo_core.optimization.surrogate.find_spec",
            lambda package: None if package == "sklearn" else object(),
        )
        with pytest.raises(ImportError, match="scikit-learn"):
            create_surrogate("sklearn", seed=42)


def test_sklearn_predict_rejects_non_finite_posterior(
    regression_data, monkeypatch: pytest.MonkeyPatch
):
    X, y, X_test = regression_data
    surrogate = SklearnSurrogate(seed=42, n_restarts=0)
    surrogate.fit(X, y)
    monkeypatch.setattr(
        surrogate.model,
        "predict",
        lambda *_args, **_kwargs: (
            np.full(len(X_test), np.nan),
            np.ones(len(X_test)),
        ),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        surrogate.predict(X_test)


def test_botorch_predict_rejects_non_finite_posterior():
    surrogate = BoTorchSurrogate(seed=42)
    surrogate.model = SimpleNamespace(
        posterior=lambda X: SimpleNamespace(
            mean=torch.full((len(X), 1), torch.nan, dtype=torch.float64),
            variance=torch.ones((len(X), 1), dtype=torch.float64),
        )
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        surrogate.predict(np.zeros((2, 1)))


def test_sklearn_covariance_rejects_non_finite_values(
    regression_data, monkeypatch: pytest.MonkeyPatch
):
    X, y, X_test = regression_data
    surrogate = SklearnSurrogate(seed=42, n_restarts=0)
    surrogate.fit(X, y)
    monkeypatch.setattr(
        surrogate.model,
        "kernel_",
        lambda XA, XB=None: np.full(
            (len(XA), len(XA) if XB is None else len(XB)),
            np.nan if XB is None else 0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="covariance contains non-finite"):
        surrogate.posterior_covariance(X_test)


def test_botorch_covariance_rejects_non_finite_values():
    surrogate = BoTorchSurrogate(seed=42)
    surrogate.model = SimpleNamespace(
        posterior=lambda X: SimpleNamespace(
            covariance_matrix=torch.full(
                (len(X), len(X)), torch.nan, dtype=torch.float64
            )
        )
    )

    with pytest.raises(RuntimeError, match="covariance contains non-finite"):
        surrogate.posterior_covariance(np.zeros((2, 1)))


def test_sklearn_surrogate_matches_existing_gp_contract(regression_data):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    X, y, X_test = regression_data
    surrogate = SklearnSurrogate(seed=42, n_restarts=0, alpha=1e-6)
    surrogate.fit(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        [1.0] * X.shape[1], (1e-2, 1e2), nu=2.5
    )
    reference = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=0,
        alpha=1e-6,
        normalize_y=True,
        random_state=42,
    )
    reference.fit(X_scaled, y)
    expected_mu, expected_std = reference.predict(
        scaler.transform(X_test), return_std=True
    )

    mu, std = surrogate.predict(X_test)
    assert np.allclose(mu, expected_mu, rtol=1e-10, atol=1e-12)
    assert np.allclose(std, expected_std, rtol=1e-10, atol=1e-12)


def test_sklearn_posterior_covariance_is_in_original_target_scale(regression_data):
    X, y, X_test = regression_data
    surrogate = SklearnSurrogate(seed=42, n_restarts=0, alpha=1e-6)
    surrogate.fit(X, y)

    _, expected_cov = surrogate.model.predict(
        surrogate.scaler.transform(X_test), return_cov=True
    )
    actual_cov = surrogate.posterior_covariance(X_test)

    assert np.allclose(actual_cov, expected_cov, rtol=1e-9, atol=1e-10)
    assert np.allclose(actual_cov, actual_cov.T, atol=1e-10)
    assert np.linalg.eigvalsh(actual_cov).min() >= -1e-8


def test_sklearn_cross_covariance_matches_joint_posterior_block(regression_data):
    X, y, X_test = regression_data
    surrogate = SklearnSurrogate(seed=42, n_restarts=0, alpha=1e-6)
    surrogate.fit(X, y)
    XA, XB = X_test[:4], X_test[4:]

    expected = surrogate.posterior_covariance(np.vstack([XA, XB]))[:4, 4:]
    actual = surrogate.posterior_cross_covariance(XA, XB)

    assert actual.shape == (4, 3)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_botorch_surrogate_returns_original_scale_fp64_posterior(regression_data):
    X, y, X_test = regression_data
    surrogate = BoTorchSurrogate(seed=42, max_fit_iterations=50)
    surrogate.fit(X, y)

    mu, std = surrogate.predict(X_test)

    assert surrogate.model.train_inputs[0].dtype == torch.float64
    assert surrogate.model.train_targets.dtype == torch.float64
    assert mu.shape == std.shape == (len(X_test),)
    assert np.all(np.isfinite(mu))
    assert np.all(std >= 1e-9)
    assert float(np.mean(mu)) > 10.0


def test_botorch_cross_covariance_matches_joint_public_posterior(regression_data):
    X, y, X_test = regression_data
    surrogate = BoTorchSurrogate(seed=42, max_fit_iterations=50)
    surrogate.fit(X, y)
    XA, XB = X_test[:4], X_test[4:]

    with torch.no_grad():
        joint = surrogate.model.posterior(
            torch.as_tensor(np.vstack([XA, XB]), dtype=torch.float64)
        ).covariance_matrix.detach().cpu().numpy()
    actual = surrogate.posterior_cross_covariance(XA, XB)

    assert actual.shape == (4, 3)
    assert np.allclose(actual, joint[:4, 4:], rtol=1e-7, atol=1e-9)


def test_botorch_predict_batches_large_candidate_pool(
    regression_data, monkeypatch: pytest.MonkeyPatch
):
    X, y, X_test = regression_data
    surrogate = BoTorchSurrogate(seed=42, max_fit_iterations=20)
    surrogate.fit(X, y)
    surrogate._PREDICTION_BATCH_SIZE = 3
    posterior_sizes: list[int] = []
    original_posterior = surrogate.model.posterior

    def recording_posterior(X):
        posterior_sizes.append(len(X))
        return original_posterior(X)

    monkeypatch.setattr(surrogate.model, "posterior", recording_posterior)
    mean, std = surrogate.predict(X_test)

    assert mean.shape == std.shape == (len(X_test),)
    assert posterior_sizes == [3, 3, 1]
    assert np.all(np.isfinite(mean))
    assert np.all(std >= 1e-9)


def test_botorch_cross_covariance_batches_large_left_side(
    regression_data, monkeypatch: pytest.MonkeyPatch
):
    X, y, X_test = regression_data
    surrogate = BoTorchSurrogate(seed=42, max_fit_iterations=20)
    surrogate.fit(X, y)
    surrogate._CROSS_COVARIANCE_BATCH_SIZE = 3
    XA = np.vstack([X_test, X_test[:2]])
    XB = X_test[:2]
    posterior_sizes: list[int] = []
    original_posterior = surrogate.model.posterior

    def recording_posterior(X):
        posterior_sizes.append(len(X))
        return original_posterior(X)

    monkeypatch.setattr(surrogate.model, "posterior", recording_posterior)
    actual = surrogate.posterior_cross_covariance(XA, XB)

    assert actual.shape == (len(XA), len(XB))
    assert posterior_sizes == [5, 5, 5]


def test_botorch_fit_expands_lbfgsb_line_search_budget(
    regression_data, monkeypatch: pytest.MonkeyPatch
):
    import botorch.fit

    X, y, _ = regression_data
    observed_options: list[dict[str, int]] = []

    def recording_fit(_mll, *, options):
        observed_options.append(options)
        return SimpleNamespace()

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll_scipy", recording_fit)

    BoTorchSurrogate(seed=42, max_fit_iterations=17).fit(X, y)

    assert observed_options == [{"maxiter": 17, "maxls": 80}]


def test_botorch_posterior_operations_reuse_successful_fit_jitter(
    regression_data, monkeypatch: pytest.MonkeyPatch
):
    import gpytorch

    X, y, X_test = regression_data
    fit_jitter = 0.125
    surrogate = BoTorchSurrogate(
        seed=42,
        jitter_levels=(fit_jitter,),
        max_fit_iterations=1,
    )
    surrogate.fit(X, y)
    observed_jitters: list[float] = []

    def recording_posterior(X):
        observed_jitters.append(
            float(gpytorch.settings.cholesky_jitter.value(torch.float64))
        )
        size = len(X)
        covariance = torch.eye(size, dtype=torch.float64)
        return SimpleNamespace(
            mean=torch.zeros((size, 1), dtype=torch.float64),
            variance=torch.ones((size, 1), dtype=torch.float64),
            covariance_matrix=covariance,
            distribution=SimpleNamespace(lazy_covariance_matrix=covariance),
        )

    monkeypatch.setattr(surrogate.model, "posterior", recording_posterior)

    surrogate.predict(X_test[:2])
    surrogate.posterior_covariance(X_test[:2])
    surrogate.posterior_cross_covariance(X_test[:2], X_test[2:4])

    assert observed_jitters == [fit_jitter, fit_jitter, fit_jitter]


def test_botorch_cross_covariance_uses_public_posterior_covariance():
    surrogate = BoTorchSurrogate(seed=42)

    def posterior(X):
        size = len(X)
        covariance = torch.arange(
            size * size, dtype=torch.float64
        ).reshape(size, size)
        return SimpleNamespace(covariance_matrix=covariance)

    surrogate.model = SimpleNamespace(posterior=posterior)
    XA = np.zeros((2, 1))
    XB = np.ones((2, 1))

    actual = surrogate.posterior_cross_covariance(XA, XB)

    assert np.array_equal(actual, np.array([[2.0, 3.0], [6.0, 7.0]]))


def test_botorch_warm_start_refreshes_transform_statistics():
    X1 = np.array([[0.0], [0.5], [1.0]], dtype=float)
    y1 = np.array([10.0, 11.0, 12.0])
    X2 = np.array([[0.0], [0.5], [1.0], [3.0]], dtype=float)
    y2 = np.array([10.0, 11.0, 12.0, 30.0])
    surrogate = BoTorchSurrogate(seed=42, max_fit_iterations=20)

    surrogate.fit(X1, y1)
    first_mean = surrogate.model.outcome_transform.means.item()
    surrogate.fit(X2, y2)

    refreshed_mean = surrogate.model.outcome_transform.means.item()
    assert first_mean == pytest.approx(float(np.mean(y1)))
    assert refreshed_mean == pytest.approx(float(np.mean(y2)))
    assert refreshed_mean != pytest.approx(first_mean)


def test_botorch_fit_does_not_advance_global_torch_rng(regression_data):
    X, y, _ = regression_data
    before = torch.random.get_rng_state().clone()

    BoTorchSurrogate(seed=42, max_fit_iterations=20).fit(X, y)

    assert torch.equal(before, torch.random.get_rng_state())
