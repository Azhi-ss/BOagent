from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from botorch.exceptions.warnings import OptimizationWarning


class FakeSurrogate:
    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> FakeSurrogate:
        self.fit_calls += 1
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(X)), np.ones(len(X))

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        return np.eye(len(X))

    def posterior_cross_covariance(
        self, XA: np.ndarray, XB: np.ndarray
    ) -> np.ndarray:
        return np.zeros((len(XA), len(XB)))


class WarningSurrogate(FakeSurrogate):
    def fit(self, X: np.ndarray, y: np.ndarray) -> WarningSurrogate:
        warnings.warn("fit did not converge", OptimizationWarning, stacklevel=2)
        return super().fit(X, y)


class FakeEngine:
    def __init__(self, *args, **kwargs) -> None:
        self.X_obs = np.array([[0.0], [1.0]])
        self.y_obs = np.array([10.0, 20.0])
        self.pool_X = np.arange(6.0).reshape(-1, 1)
        self.pool_yield = np.arange(30.0, 36.0)
        self.M = len(self.pool_X)


def test_run_case_reuses_only_warm_botorch_surrogate(monkeypatch):
    from bo_core.benchmark import backend_benchmark

    created: list[FakeSurrogate] = []

    def fake_create(*args, **kwargs):
        surrogate = FakeSurrogate()
        created.append(surrogate)
        return surrogate

    monkeypatch.setattr(backend_benchmark, "LGBOEngine", FakeEngine)
    monkeypatch.setattr(backend_benchmark, "create_surrogate", fake_create)
    monkeypatch.setattr(backend_benchmark, "_configure_numerical_threads", lambda _: None)
    monkeypatch.setattr(
        backend_benchmark.resource,
        "getrusage",
        lambda _: SimpleNamespace(ru_maxrss=2048),
    )

    warm = backend_benchmark.run_case(
        "buchwald_sub4", "botorch_warm", seed=100, n_steps=3, n_threads=1
    )
    assert len(created) == 1
    assert created[0].fit_calls == 3
    assert warm["query_indices"] == [1, 2, 4]
    assert warm["peak_rss_mb"] == pytest.approx(2.0)
    assert warm["fit_s"] >= 0.0
    assert warm["predict_s"] >= 0.0
    assert warm["covariance_s"] >= 0.0

    created.clear()
    cold = backend_benchmark.run_case(
        "buchwald_sub4", "botorch_cold", seed=100, n_steps=3, n_threads=1
    )
    assert len(created) == 3
    assert [surrogate.fit_calls for surrogate in created] == [1, 1, 1]
    assert cold["query_indices"] == warm["query_indices"]


def test_run_case_records_botorch_optimization_warnings(monkeypatch):
    from bo_core.benchmark import backend_benchmark

    monkeypatch.setattr(backend_benchmark, "LGBOEngine", FakeEngine)
    monkeypatch.setattr(
        backend_benchmark,
        "create_surrogate",
        lambda *args, **kwargs: WarningSurrogate(),
    )
    monkeypatch.setattr(backend_benchmark, "_configure_numerical_threads", lambda _: None)

    result = backend_benchmark.run_case(
        "buchwald_sub4",
        "botorch_warm",
        seed=100,
        n_steps=3,
        n_threads=1,
    )

    assert result["optimization_warning_count"] == 3


def test_worker_mode_always_sets_max_tasks_per_child(monkeypatch):
    from bo_core.benchmark import backend_benchmark

    executor = MagicMock()
    executor.__enter__.return_value = executor

    with patch(
        "bo_core.benchmark.backend_benchmark.ProcessPoolExecutor",
        return_value=executor,
    ) as pool:
        backend_benchmark._run_worker_mode(
            [],
            workers=4,
            n_steps=1,
            n_restarts=0,
            max_fit_iterations=1,
        )

    assert pool.call_args.kwargs["max_tasks_per_child"] == 1


def test_markdown_interpretation_uses_report_values():
    from bo_core.benchmark.backend_benchmark import _markdown

    report = {
        "configuration": {
            "n_steps": 2,
            "seeds": [1],
            "n_restarts": 0,
            "max_fit_iterations": 5,
        },
        "summary": [
            {
                "dataset": "custom",
                "mode": "sklearn",
                "workers": 1,
                "n_threads": 10,
                "fit_s_mean": 4.0,
                "predict_s_mean": 1.0,
                "covariance_s_mean": 1.0,
                "total_s_mean": 6.0,
                "peak_rss_mb_mean": 100.0,
                "optimization_warning_count_mean": 0.0,
            },
            {
                "dataset": "custom",
                "mode": "botorch_warm",
                "workers": 1,
                "n_threads": 10,
                "fit_s_mean": 1.0,
                "predict_s_mean": 1.0,
                "covariance_s_mean": 1.0,
                "total_s_mean": 3.0,
                "peak_rss_mb_mean": 120.0,
                "optimization_warning_count_mean": 1.0,
            },
        ],
        "cases": [
            {"backend": "sklearn", "n_steps": 2, "optimization_warning_count": 0},
            {"backend": "botorch", "n_steps": 2, "optimization_warning_count": 1},
        ],
    }

    markdown = _markdown(report)

    assert "custom / 1 x 10: warm-start BoTorch was 2.00x faster" in markdown
    assert "1 `OptimizationWarning` instances across 2 fits (50.0%)" in markdown
    assert "Buchwald" not in markdown


def test_markdown_interpretation_reports_clean_botorch_convergence():
    from bo_core.benchmark.backend_benchmark import _markdown

    report = {
        "configuration": {
            "n_steps": 2,
            "seeds": [1],
            "n_restarts": 0,
            "max_fit_iterations": 5,
        },
        "summary": [],
        "cases": [
            {"backend": "botorch", "n_steps": 2, "optimization_warning_count": 0}
        ],
    }

    markdown = _markdown(report)

    assert "0 `OptimizationWarning` instances across 2 fits (0.0%)" in markdown
    assert "convergence signal must be resolved" not in markdown
    assert "L-BFGS-B line-search steps per iteration: 80" in markdown


def test_markdown_interpretation_says_botorch_is_default():
    from bo_core.benchmark.backend_benchmark import _markdown

    report = {
        "configuration": {
            "n_steps": 2,
            "seeds": [1],
            "n_restarts": 0,
            "max_fit_iterations": 5,
        },
        "summary": [],
        "cases": [
            {"backend": "botorch", "n_steps": 2, "optimization_warning_count": 0}
        ],
    }

    markdown = _markdown(report)

    assert "BoTorch/GPyTorch is the default backend" in markdown
    assert "sklearn remains the default" not in markdown


def test_summarize_groups_worker_modes():
    from bo_core.benchmark.backend_benchmark import summarize

    base = {
        "dataset": "suzuki",
        "mode": "sklearn",
        "workers": 1,
        "n_threads": 10,
        "fit_s": 1.0,
        "predict_s": 2.0,
        "covariance_s": 3.0,
        "total_s": 6.0,
        "peak_rss_mb": 100.0,
        "optimization_warning_count": 2,
    }
    summary = summarize([base, {**base, "fit_s": 3.0, "total_s": 8.0}])

    assert len(summary) == 1
    assert summary[0]["n"] == 2
    assert summary[0]["fit_s_mean"] == pytest.approx(2.0)
    assert summary[0]["total_s_mean"] == pytest.approx(7.0)
    assert summary[0]["workers"] == 1
    assert summary[0]["n_threads"] == 10
    assert summary[0]["optimization_warning_count_mean"] == pytest.approx(2.0)
