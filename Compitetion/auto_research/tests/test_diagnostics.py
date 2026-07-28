"""Tests for auditable BO runtime diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import components.library  # noqa: F401
from components.protocol import Composition, StepContext
from engine import HybridEngine


class _Encoder:
    def __init__(self, *, fail_mean_shift: bool = False) -> None:
        self.fail_mean_shift = fail_mean_shift

    def encode_df(
        self,
        frame: pd.DataFrame,
        *,
        allow_unknown: bool = False,
    ) -> np.ndarray:
        return np.zeros((len(frame), 2), dtype=float)

    def encode_rows(self, rows: list[dict[str, str]]) -> np.ndarray:
        if self.fail_mean_shift:
            raise ValueError("invalid LLM point")
        return np.zeros((len(rows), 2), dtype=float)


def _first_unqueried(scores, ctx) -> int:
    return next(i for i in range(len(scores)) if i not in ctx.queried)


def _engine(
    *,
    surrogate,
    acquisition,
    llm_strategy=lambda ctx: None,
    fail_mean_shift: bool = False,
    n_iters: int = 2,
) -> HybridEngine:
    engine = HybridEngine.__new__(HybridEngine)
    engine.composition = Composition(
        name="diagnostic_test",
        surrogate="test",
        acquisition="test_acquisition",
        selector="argmax",
        llm_strategy="test_llm",
        params={"use_llm": llm_strategy is not None},
    )
    engine.dataset = "buchwald_sub4"
    engine.seed = 100
    engine.n_iters = n_iters
    engine.backend = "botorch"
    engine.feature_cols = ["a", "b"]
    engine.target_col = "yield"
    engine.train_df = pd.DataFrame(
        {"a": ["a0", "a1"], "b": ["b0", "b1"], "yield": [10.0, 20.0]}
    )
    engine.options = {"a": ["a0", "a1"], "b": ["b0", "b1"]}
    engine.encoder = _Encoder(fail_mean_shift=fail_mean_shift)
    engine.pool_X = np.zeros((3, 2), dtype=float)
    engine.pool_yield = np.array([30.0, 40.0, 50.0])
    engine.M = 3
    engine.pool_conditions = [
        {"a": f"a{i}", "b": f"b{i}"} for i in range(engine.M)
    ]
    engine.surrogate = surrogate
    engine.acq_fn = acquisition
    engine.sel_fn = _first_unqueried
    engine.llm_fn = llm_strategy or (lambda ctx: None)
    engine.initial_indices = (0, 1)
    return engine


def test_engine_records_fit_and_acquisition_fallbacks() -> None:
    surrogate = MagicMock()
    surrogate.fit.side_effect = RuntimeError("not positive definite")
    acquisition = MagicMock(side_effect=ValueError("invalid posterior"))
    engine = _engine(surrogate=surrogate, acquisition=acquisition)

    engine.run()

    diagnostics = engine.diagnostics
    assert diagnostics["summary"]["gp_fit_failures"] == 2
    assert diagnostics["summary"]["acquisition_fallbacks"] == 2
    assert diagnostics["summary"]["fallback_row_order_selections"] == 2
    first = diagnostics["iterations"][0]
    assert first["gp_fit"]["status"] == "failed"
    assert "RuntimeError" in first["gp_fit"]["error"]
    assert first["acquisition"]["status"] == "fallback"
    assert first["acquisition"]["scores"]["is_constant"] is True
    assert first["selection"]["after_acquisition_fallback"] is True
    assert first["selection"]["is_first_unqueried"] is True


def test_engine_records_mean_shift_failure() -> None:
    decision = {
        "action": "mean_shift",
        "point": {"a": "a0", "b": "b0"},
        "confidence": 0.8,
    }
    engine = _engine(
        surrogate=MagicMock(),
        acquisition=MagicMock(return_value=np.array([0.1, 0.2, 0.3])),
        llm_strategy=lambda ctx: decision,
        fail_mean_shift=True,
        n_iters=1,
    )

    engine.run()

    iteration = engine.diagnostics["iterations"][0]
    assert iteration["llm"]["action"] == "mean_shift"
    assert iteration["mean_shift"]["status"] == "failed"
    assert "invalid LLM point" in iteration["mean_shift"]["error"]
    assert engine.diagnostics["summary"]["mean_shift_failures"] == 1


def test_cake_records_kernel_and_llm_diagnostics(monkeypatch) -> None:
    from components.cake import CAKESurrogate

    surrogate = CAKESurrogate(seed=100, population_size=3, evolve_interval=99)
    surrogate._population = {"M5": 1.0, "SE": 2.0, "RQ": 3.0}
    monkeypatch.setattr(surrogate, "_tensor", lambda value: np.asarray(value))
    monkeypatch.setattr(surrogate, "_evolve_kernel", lambda *args: None)

    def fit_kernel(train_x, train_y, expression, dimension):
        if expression == "SE":
            raise RuntimeError("kernel fit failed")
        return object(), float({"M5": 1.0, "RQ": 2.0}[expression])

    monkeypatch.setattr(surrogate, "_fit_one_kernel", fit_kernel)
    surrogate.fit(np.zeros((4, 2)), np.arange(4, dtype=float))

    client = MagicMock()
    success = MagicMock(status="success", content="Kernel: M5 + RQ\nReasoning: test")
    failure = MagicMock(status="error", content="")
    client.chat.side_effect = [success, failure]
    surrogate._llm_crossover(
        client,
        ["M5", "RQ"],
        np.array([0.5, 0.5]),
        {"M5": 1.0, "RQ": 2.0},
    )
    surrogate._llm_mutation(client, "M5", 1.0, [1.0, 2.0])

    diagnostics = surrogate.diagnostics
    assert diagnostics["summary"]["kernel_fit_failures"] == 1
    assert diagnostics["summary"]["llm_attempts"] == 2
    assert diagnostics["summary"]["llm_successes"] == 1
    assert diagnostics["summary"]["llm_failures"] == 1
    assert diagnostics["fits"][0]["active_kernels"] == ["M5", "RQ"]
    assert diagnostics["fits"][0]["failed_kernels"][0]["kernel"] == "SE"
    assert set(diagnostics["fits"][0]["population_weights"]) == {"M5", "RQ"}


def test_lgbo_strategy_reports_parse_failure(monkeypatch) -> None:
    from components import library

    client = MagicMock()
    client.is_configured.return_value = True
    client.chat.return_value = MagicMock(status="success", content="not parseable")
    monkeypatch.setattr(
        "bo_core.llm_client.DeepSeekClient.from_env", lambda: client
    )
    monkeypatch.setattr(
        "bo_core.optimization.lgbo_parser.parse_llm_response", lambda *args: None
    )
    ctx = StepContext(
        iteration=0,
        n_iters=1,
        feature_cols=["a"],
        options={"a": ["a0"]},
        history=[({"a": "a0"}, 10.0)],
        queried=set(),
        best_f=10.0,
        remaining=1,
        extra={
            "use_llm": True,
            "dataset": "buchwald_sub4",
            "target_col": "yield",
        },
    )

    decision = library._llm_lgbo(ctx)

    assert decision is None
    assert ctx.extra["_llm_diagnostic"] == {
        "attempted": True,
        "status": "parse_failed",
        "error": "LLM response could not be parsed",
    }


def test_lgbo_strategy_reports_api_failure(monkeypatch) -> None:
    from components import library

    client = MagicMock()
    client.is_configured.return_value = True
    client.chat.side_effect = RuntimeError("network unavailable")
    monkeypatch.setattr(
        "bo_core.llm_client.DeepSeekClient.from_env", lambda: client
    )
    ctx = StepContext(
        iteration=0,
        n_iters=1,
        feature_cols=["a"],
        options={"a": ["a0"]},
        history=[({"a": "a0"}, 10.0)],
        queried=set(),
        best_f=10.0,
        remaining=1,
        extra={
            "use_llm": True,
            "dataset": "buchwald_sub4",
            "target_col": "yield",
        },
    )

    decision = library._llm_lgbo(ctx)

    assert decision is None
    assert ctx.extra["_llm_diagnostic"]["attempted"] is True
    assert ctx.extra["_llm_diagnostic"]["status"] == "failed"
    assert "network unavailable" in ctx.extra["_llm_diagnostic"]["error"]


def test_lmabo_strategy_reports_api_failure(monkeypatch) -> None:
    from components import library

    client = MagicMock()
    client.is_configured.return_value = True
    client.chat.side_effect = RuntimeError("network unavailable")
    monkeypatch.setattr(
        "bo_core.llm_client.DeepSeekClient.from_env", lambda: client
    )
    ctx = StepContext(
        iteration=0,
        n_iters=1,
        feature_cols=["a"],
        options={"a": ["a0"]},
        history=[({"a": "a0"}, 10.0)],
        queried=set(),
        best_f=10.0,
        remaining=1,
        extra={"use_llm": True, "dataset": "buchwald_sub4"},
    )

    decision = library._llm_lmabo(ctx)

    assert decision is None
    assert ctx.extra["_llm_diagnostic"]["attempted"] is True
    assert ctx.extra["_llm_diagnostic"]["status"] == "failed"
    assert "network unavailable" in ctx.extra["_llm_diagnostic"]["error"]


def test_bora_strategy_reports_policy_without_llm_attempt() -> None:
    from components import library

    ctx = StepContext(
        iteration=0,
        n_iters=1,
        feature_cols=["a"],
        options={"a": ["a0"]},
        history=[({"a": "a0"}, 10.0)],
        queried=set(),
        best_f=10.0,
        remaining=1,
        extra={"use_llm": True},
    )

    decision = library._llm_bora(ctx)

    assert decision == {"action": "llm_pick"}
    assert ctx.extra["_llm_diagnostic"] == {
        "attempted": False,
        "status": "policy_decision",
        "error": None,
    }


def test_engine_records_nonfinite_acquisition_as_degenerate() -> None:
    engine = _engine(
        surrogate=MagicMock(),
        acquisition=MagicMock(return_value=np.full(3, np.nan)),
        n_iters=1,
    )

    engine.run()

    scores = engine.diagnostics["iterations"][0]["acquisition"]["scores"]
    summary = engine.diagnostics["summary"]
    assert scores["is_degenerate"] is True
    assert scores["nonfinite_count"] == 3
    assert summary["degenerate_score_iterations"] == 1
    assert summary["row_order_selections_after_degenerate_scores"] == 1


def test_engine_does_not_count_policy_decision_as_llm_success() -> None:
    engine = _engine(
        surrogate=MagicMock(),
        acquisition=MagicMock(return_value=np.array([0.1, 0.2, 0.3])),
        llm_strategy=lambda ctx: {"action": "bo_pick"},
        n_iters=1,
    )

    engine.run()

    summary = engine.diagnostics["summary"]
    assert summary["llm_attempts"] == 0
    assert summary["llm_successes"] == 0
    assert summary["llm_diagnostics_missing"] == 1


def test_engine_does_not_misclassify_llm_pick_as_row_order_fallback() -> None:
    engine = _engine(
        surrogate=MagicMock(),
        acquisition=MagicMock(side_effect=RuntimeError("acquisition failed")),
        llm_strategy=lambda ctx: {"action": "pool_pick", "pool_index": 0},
        n_iters=1,
    )

    engine.run()

    assert engine.diagnostics["summary"]["fallback_row_order_selections"] == 0


def test_engine_surfaces_cake_kernel_degradation() -> None:
    surrogate = MagicMock()
    surrogate.diagnostics = {
        "summary": {
            "kernel_fit_failures": 3,
            "llm_attempts": 4,
            "llm_successes": 3,
            "llm_failures": 1,
        },
        "fits": [
            {"active_kernels": ["M5", "RQ"], "failed_kernels": []},
            {"active_kernels": ["M5"], "failed_kernels": [{"kernel": "RQ"}]},
        ],
    }
    engine = _engine(
        surrogate=surrogate,
        acquisition=MagicMock(return_value=np.array([0.1, 0.2, 0.3])),
        n_iters=1,
    )

    engine.run()

    summary = engine.diagnostics["summary"]
    assert summary["surrogate_kernel_fit_failures"] == 3
    assert summary["surrogate_degraded_fits"] == 1
    assert summary["surrogate_min_active_kernels"] == 1
    assert summary["surrogate_llm_attempts"] == 4
    assert summary["surrogate_llm_successes"] == 3
    assert summary["surrogate_llm_failures"] == 1


def test_engine_preserves_completed_diagnostics_after_crash() -> None:
    engine = _engine(
        surrogate=MagicMock(),
        acquisition=MagicMock(return_value=np.array([0.1, 0.2, 0.3])),
        n_iters=2,
    )
    engine.sel_fn = MagicMock(side_effect=[0, RuntimeError("selector crashed")])

    with pytest.raises(RuntimeError, match="selector crashed"):
        engine.run()

    assert engine.diagnostics["summary"]["iterations_completed"] == 1
    assert engine.diagnostics["summary"]["iterations_recorded"] == 2
    assert engine.diagnostics["summary"]["crashed_iterations"] == 1
    crashed = engine.diagnostics["iterations"][1]
    assert crashed["status"] == "crashed"
    assert crashed["crash_stage"] == "selection"
    assert "selector crashed" in crashed["error"]


def test_runner_returns_partial_diagnostics_on_failure() -> None:
    import agent_step
    import loop

    composition = Composition(
        name="gpbo_ei",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False},
    )
    fake_engine = MagicMock()
    fake_engine.run.side_effect = RuntimeError("selector crashed")
    fake_engine.initial_indices = tuple(range(35))
    fake_engine.diagnostics = {
        "summary": {"iterations_completed": 1},
        "iterations": [{"step": 1}],
    }

    with patch("loop.HybridEngine", return_value=fake_engine):
        loop_result = loop.run_one(composition, "buchwald_sub4", seed=100, n_iters=2)
    with patch("agent_step.HybridEngine", return_value=fake_engine):
        agent_result = agent_step.run_one(
            {
                "name": composition.name,
                "surrogate": composition.surrogate,
                "acquisition": composition.acquisition,
                "selector": composition.selector,
                "llm_strategy": composition.llm_strategy,
                "params": composition.params,
            },
            "buchwald_sub4",
            seed=100,
            n_iters=2,
        )

    for result in (loop_result, agent_result):
        assert result["status"] == "failed"
        assert result["prior_protocol"] == "fixed_train_prior"
        assert result["n_train_prior"] == 35
        assert result["initial_indices"] == list(range(35))
        assert result["diagnostics"] == fake_engine.diagnostics
        assert "selector crashed" in result["error"]


def test_loop_result_includes_engine_diagnostics() -> None:
    from loop import run_one

    composition = Composition(
        name="gpbo_ei",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False},
    )
    fake_engine = MagicMock()
    fake_engine.run.return_value = [
        {
            "step": 1,
            "query_index": 0,
            "condition": {},
            "observed_yield": 50.0,
            "acquisition": "ei",
            "llm_action": None,
        }
    ]
    fake_engine.initial_indices = tuple(range(35))
    fake_engine.diagnostics = {"summary": {"gp_fit_failures": 0}, "iterations": []}

    with patch("loop.HybridEngine", return_value=fake_engine):
        result = run_one(composition, "buchwald_sub4", seed=100, n_iters=1)

    assert result["status"] == "ok"
    assert result["prior_protocol"] == "fixed_train_prior"
    assert result["n_train_prior"] == 35
    assert result["initial_indices"] == list(range(35))
    assert result["diagnostics"] == fake_engine.diagnostics


def test_runners_report_engine_construction_failure() -> None:
    import agent_step
    import loop

    composition = Composition(
        name="broken",
        surrogate="missing",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={},
    )
    comp_dict = {
        "name": composition.name,
        "surrogate": composition.surrogate,
        "acquisition": composition.acquisition,
        "selector": composition.selector,
        "llm_strategy": composition.llm_strategy,
        "params": composition.params,
    }

    with patch("loop.HybridEngine", side_effect=KeyError("missing dataset")):
        loop_result = loop.run_one(composition, "missing", seed=100, n_iters=1)
    with patch("agent_step.HybridEngine", side_effect=KeyError("missing dataset")):
        agent_result = agent_step.run_one(comp_dict, "missing", seed=100, n_iters=1)

    for result in (loop_result, agent_result):
        assert result["status"] == "failed"
        assert result["initial_indices"] == []
        assert result["diagnostics"] is None
        assert "missing dataset" in result["error"]
