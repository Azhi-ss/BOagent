"""Regression tests for the Deep Kernel Learning surrogate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

from components.dkl import DKLSurrogate, _build_dkl_kernel


def _flatten_parameters(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def test_dkl_kernel_initialization_is_seeded_without_global_rng_leak() -> None:
    torch.manual_seed(999)
    expected_next = torch.rand(4)

    torch.manual_seed(999)
    _, first = _build_dkl_kernel(5, hidden_dim=4, n_layers=2, seed=100)
    actual_next = torch.rand(4)
    _, repeated = _build_dkl_kernel(5, hidden_dim=4, n_layers=2, seed=100)
    _, different = _build_dkl_kernel(5, hidden_dim=4, n_layers=2, seed=200)

    assert torch.equal(actual_next, expected_next)
    assert torch.equal(_flatten_parameters(first), _flatten_parameters(repeated))
    assert not torch.equal(_flatten_parameters(first), _flatten_parameters(different))


def test_dkl_model_builds_with_installed_botorch() -> None:
    surrogate = DKLSurrogate(seed=100, hidden_dim=4, n_layers=1, max_fit_iterations=1)
    train_x = surrogate._tensor(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    train_y = surrogate._tensor(np.array([[0.0], [1.0], [0.5]]))
    kernel, extractor = _build_dkl_kernel(2, hidden_dim=4, n_layers=1, seed=100)

    model = surrogate._build_model(train_x, train_y, extractor, kernel)

    assert model.input_transform(train_x).shape == (3, 4)
    assert any("input_transform.mlp" in key for key in model.state_dict())


def test_dkl_fit_uses_deep_kernel_with_installed_botorch() -> None:
    surrogate = DKLSurrogate(seed=100, hidden_dim=4, n_layers=1, max_fit_iterations=1)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.5])

    surrogate.fit(X, y)
    mean, std = surrogate.predict(X)

    assert surrogate.diagnostics["fits"][-1]["status"] == "dkl"
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))


def test_dkl_warm_state_excludes_data_dependent_transforms() -> None:
    surrogate = DKLSurrogate(seed=100, hidden_dim=4, n_layers=1, max_fit_iterations=1)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.5])

    surrogate.fit(X, y)

    assert surrogate.diagnostics["fits"][-1]["status"] == "dkl"
    assert surrogate._dkl_warm_state
    assert all(
        key.startswith(surrogate._WARM_START_PREFIXES)
        for key in surrogate._dkl_warm_state
    )


def test_dkl_fallback_is_reported(monkeypatch) -> None:
    surrogate = DKLSurrogate(seed=100, max_fit_iterations=1)
    monkeypatch.setattr(
        surrogate,
        "_fit_dkl",
        lambda *args: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    monkeypatch.setattr(surrogate, "_fit_matern_fallback", lambda *args: None)

    surrogate.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    assert surrogate.diagnostics["summary"]["fallback_fits"] == 1
    assert surrogate.diagnostics["fits"][0]["status"] == "fallback"


def test_dkl_and_fallback_warm_states_are_isolated(monkeypatch) -> None:
    surrogate = DKLSurrogate(seed=100, max_fit_iterations=1)

    def fail_dkl(*args) -> None:
        surrogate._dkl_warm_state = {"dkl": object()}
        raise RuntimeError("forced")

    def fit_fallback(*args) -> None:
        assert surrogate._fallback_warm_state is None
        surrogate._fallback_warm_state = {"fallback": object()}

    monkeypatch.setattr(surrogate, "_fit_dkl", fail_dkl)
    monkeypatch.setattr(surrogate, "_fit_matern_fallback", fit_fallback)

    surrogate.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    assert set(surrogate._dkl_warm_state) == {"dkl"}
    assert set(surrogate._fallback_warm_state) == {"fallback"}
