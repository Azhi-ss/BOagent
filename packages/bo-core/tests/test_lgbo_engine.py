from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from bo_core.optimization.lgbo import LGBOEngine
from bo_core.optimization.surrogate import BoTorchSurrogate
from botorch.exceptions.warnings import OptimizationWarning


def _conditions_match(engine: LGBOEngine) -> bool:
    """Every trajectory entry's condition must equal the test_df row at query_index."""
    for t in engine.trajectory:
        row = engine.test_df.iloc[t["query_index"]]
        for col in engine.feature_cols:
            if str(row[col]) != t["condition"][col]:
                return False
    return True


@pytest.fixture(autouse=True)
def _suppress_gp_warnings():
    # The underdetermined one-hot GP routinely hits kernel length-scale bounds;
    # these ConvergenceWarnings are expected and not test failures.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_gpbo_smoke_buchwald():
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=3, n_restarts=2)
    e.run()
    assert e.iteration == 3
    assert len(e.trajectory) == 3
    # Unique, in-range query indices.
    idxs = [t["query_index"] for t in e.trajectory]
    assert len(set(idxs)) == 3
    assert all(0 <= i < e.M for i in idxs)
    # Conditions match the pool rows.
    assert _conditions_match(e)
    # best_found is the max observed yield and is finite.
    assert e.best_found() == pytest.approx(max(t["observed_yield"] for t in e.trajectory))
    assert e.best_found() > 0


def test_gpbo_smoke_suzuki():
    e = LGBOEngine("suzuki", seed=100, use_llm=False, n_iters=2, n_restarts=1)
    e.run()
    assert e.iteration == 2
    idxs = [t["query_index"] for t in e.trajectory]
    assert len(set(idxs)) == 2
    assert all(0 <= i < e.M for i in idxs)
    assert _conditions_match(e)


def test_gpbo_no_repeated_query_index_over_longer_run():
    e = LGBOEngine("buchwald_sub4", seed=200, use_llm=False, n_iters=6, n_restarts=1)
    e.run()
    idxs = [t["query_index"] for t in e.trajectory]
    assert len(set(idxs)) == len(idxs)  # no repeats
    # Trajectory steps are 1..n.
    assert [t["step"] for t in e.trajectory] == list(range(1, 7))


def test_gpbo_observed_yields_match_oracle():
    """observed_yield in trajectory must equal test.csv Yield at query_index."""
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=2, n_restarts=1)
    e.run()
    for t in e.trajectory:
        oracle = float(e.test_df[e.target_col].iloc[t["query_index"]])
        assert t["observed_yield"] == pytest.approx(oracle)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        LGBOEngine("not_a_dataset", seed=100, use_llm=False, n_iters=1)


def test_buchwald_uses_fixed_35_row_prior_in_32_dimensions():
    engine = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=0,
    )

    assert len(engine.train_df) == 35
    assert engine.X_obs.shape == (35, 32)
    assert engine.pool_X.shape == (783, 32)
    assert np.all(engine.pool_X.sum(axis=0) > 0)

    reactant_idx = engine.feature_cols.index("Reactant2")
    offset = engine.encoder._offsets[reactant_idx]
    size = engine.encoder._sizes[reactant_idx]
    reactant_block = engine.X_obs[:, offset:offset + size]
    target_rows = engine.train_df["Reactant2"].isin(
        engine.encoder.options["Reactant2"]
    ).to_numpy()
    assert int(target_rows.sum()) == 7
    assert np.allclose(reactant_block[target_rows].sum(axis=1), 1.0)
    assert np.allclose(reactant_block[~target_rows], 0.0)


def test_suzuki_keeps_35_dimensional_fixed_prior():
    engine = LGBOEngine("suzuki", seed=100, use_llm=False, n_iters=0)

    assert len(engine.train_df) == 29
    assert engine.X_obs.shape == (29, 35)
    assert engine.pool_X.shape[1] == 35
    assert np.all(engine.pool_X.sum(axis=0) > 0)


# ---- Slice 5: mean-shift (Proposition 1) math ----

def _fit_and_get_mu(engine):
    surrogate = engine._fit_gp()
    mu, sigma = engine._predict_pool(surrogate)
    return surrogate, mu, sigma


@pytest.mark.parametrize("backend", ["sklearn", "botorch"])
def test_fixed_prior_produces_nondegenerate_acquisition(backend):
    engine = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=0,
        n_restarts=2,
        backend=backend,
    )

    _, mean, std = _fit_and_get_mu(engine)
    acquisition = engine._expected_improvement(
        mean,
        std,
        float(np.max(engine.y_obs)),
    )

    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(np.isfinite(acquisition))
    assert np.ptp(acquisition) > 0.0
    assert 0 <= int(np.argmax(acquisition)) < engine.M


def test_gpbo_botorch_backend_smoke():
    e = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=1,
        backend="botorch",
    )
    e.run()
    assert e.iteration == 1
    assert np.isfinite(e.trajectory[0]["predicted_yield"])
    assert _conditions_match(e)


def test_botorch_fit_converges_on_recorded_buchwald_prior():
    engine = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=0,
        backend="botorch",
    )
    surrogate = BoTorchSurrogate(seed=100, max_fit_iterations=100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizationWarning)
        surrogate.fit(engine.X_obs, engine.y_obs)

    optimization_warnings = [
        item for item in caught if issubclass(item.category, OptimizationWarning)
    ]
    mean, std = surrogate.predict(engine.pool_X[:16])
    assert not optimization_warnings
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))


def test_mean_shift_zero_confidence_is_noop():
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    surrogate, mu, _ = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    assert np.allclose(e._mean_shift(surrogate, mu, x_p, 0.0), mu)


def test_mean_shift_matches_independent_reconstruction():
    """Verify mu_lambda = mu + lambda * K_post(pool,grid) @ a, lambda = c/sqrt(a'Sigma a)."""
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    surrogate, mu, _ = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    c = 0.5
    ml = e._mean_shift(surrogate, mu, x_p, c)

    # Independently reconstruct the shift from the shared posterior contract.
    d = len(e.feature_cols)
    hamming = d - e.pool_X @ x_p
    K = min(e.K, e.M)
    grid_idx = np.argpartition(hamming, K - 1)[:K]
    X_grid = e.pool_X[grid_idx]

    a = np.maximum(
        surrogate.prior_cross_covariance(x_p.reshape(1, -1), X_grid).ravel(),
        0.0,
    )
    a = a / a.sum()
    Sigma_GG = surrogate.posterior_covariance(X_grid)
    lam = c / float(np.sqrt(a @ Sigma_GG @ a))

    K_post = surrogate.posterior_cross_covariance(e.pool_X, X_grid)
    expected = mu + lam * (K_post @ a)

    assert np.allclose(ml, expected, rtol=1e-6, atol=1e-8)
    assert lam > 0


def test_mean_shift_lifts_proposed_point_and_is_mean_only():
    """The shift raises the mean at the proposed point; sigma is untouched."""
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    surrogate, mu, sigma = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    ml = e._mean_shift(surrogate, mu, x_p, 0.5)
    # x_p is its own nearest grid point; the lift increases its mean.
    assert ml[0] > mu[0]
    # Mean-only shift: sigma array is not modified by _mean_shift (passed separately).
    sigma_after = sigma.copy()
    _ = e._mean_shift(surrogate, mu, x_p, 0.5)
    assert np.array_equal(sigma, sigma_after)


def test_mean_shift_scales_linearly_with_confidence():
    """lambda ~ c, so doubling c doubles the shift (Proposition 1 calibration)."""
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    surrogate, mu, _ = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    shift_half = e._mean_shift(surrogate, mu, x_p, 0.5) - mu
    shift_full = e._mean_shift(surrogate, mu, x_p, 1.0) - mu
    assert np.allclose(shift_full, 2.0 * shift_half, rtol=1e-6, atol=1e-8)


class _DeterministicSurrogate:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, _x: np.ndarray, _y: np.ndarray) -> _DeterministicSurrogate:
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(x) == len(self.mean)
        return self.mean.copy(), self.std.copy()

    def prior_cross_covariance(
        self, xa: np.ndarray, xb: np.ndarray
    ) -> np.ndarray:
        return np.ones((len(xa), len(xb)))

    def posterior_covariance(self, x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    def posterior_cross_covariance(
        self, xa: np.ndarray, xb: np.ndarray
    ) -> np.ndarray:
        return 1.0 + xa @ xb.T


class _FixedClient:
    def __init__(self, content: str, tool_calls: list[dict[str, object]] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.calls = 0
        self.call_kwargs: dict[str, object] = {}

    def is_configured(self) -> bool:
        return True

    def chat(
        self,
        messages: list[dict[str, object]],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
        *,
        temperature: float = 0.0,
    ) -> SimpleNamespace:
        self.calls += 1
        self.call_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
            "temperature": temperature,
        }
        return SimpleNamespace(
            status="success",
            content=self.content,
            tool_calls=self.tool_calls,
            error=None,
            usage={},
        )


def _deterministic_posterior(
    engine: LGBOEngine,
) -> tuple[_DeterministicSurrogate, np.ndarray, np.ndarray]:
    best_f = float(np.max(engine.y_obs))
    mean = np.linspace(best_f - 2.0, best_f + 2.0, engine.M)
    std = np.linspace(0.4, 0.8, engine.M)
    return _DeterministicSurrogate(mean, std), mean, std

def test_call_llm_forwards_tools_and_configured_temperature() -> None:
    engine = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=0,
        llm_temperature=0.3,
    )
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "propose_sparse_subspace", "arguments": "{}"},
        }
    ]
    client = _FixedClient("", tool_calls)
    engine._client = client
    tools = [{"type": "function", "function": {"name": "propose_sparse_subspace"}}]
    tool_choice = {
        "type": "function",
        "function": {"name": "propose_sparse_subspace"},
    }

    result, reason = engine._call_llm(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=tools,
        tool_choice=tool_choice,
    )

    assert reason == "accepted"
    assert result is not None
    assert client.call_kwargs["temperature"] == 0.3
    assert client.call_kwargs["extra_body"] == {
        "reasoning_effort": "low",
        "tools": tools,
        "tool_choice": tool_choice,
    }


def test_call_llm_defaults_temperature_to_zero() -> None:
    engine = LGBOEngine("buchwald_sub4", use_llm=False, n_iters=0)
    client = _FixedClient("text")
    engine._client = client

    result, reason = engine._call_llm([{"role": "user", "content": "hi"}])

    assert reason == "accepted"
    assert result is not None
    assert engine.llm_temperature == 0.0
    assert client.call_kwargs["temperature"] == 0.0


class _FailingSurrogate:
    @property
    def is_fit(self) -> bool:
        return False

    def fit(self, _x: np.ndarray, _y: np.ndarray) -> _FailingSurrogate:
        raise RuntimeError("fit failed")

    def predict(self, _x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("predict failed")


def test_legacy_step_preserves_point_hamming_shift_and_single_llm_call() -> None:
    engine = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=0,
        K=3,
    )
    surrogate, mean, std = _deterministic_posterior(engine)
    engine._surrogate = surrogate
    values = [str(engine.test_df[field].iloc[0]) for field in engine.feature_cols]
    confidence = 0.4
    client = _FixedClient(
        "Thinking: fixed legacy proposal\n"
        + '{"mode":"point","values":'
        + repr(values).replace("'", '"')
        + f',"confidence":{confidence}}}'
    )
    engine.use_llm = True
    engine._client = client

    proposed = engine.encoder.encode_rows(
        [dict(zip(engine.feature_cols, values))]
    )[0]
    hamming = len(engine.feature_cols) - engine.pool_X @ proposed
    grid_indices = np.argpartition(hamming, engine.K - 1)[: engine.K]
    grid = engine.pool_X[grid_indices]
    weights = np.full(engine.K, 1.0 / engine.K)
    denominator = float(weights @ np.eye(engine.K) @ weights)
    strength = confidence / np.sqrt(denominator)
    shifted = mean + strength * ((1.0 + engine.pool_X @ grid.T) @ weights)
    expected_index = int(
        np.argmax(engine._expected_improvement(shifted, std, float(np.max(engine.y_obs))))
    )

    row = engine.step()

    assert client.calls == 1
    assert row["query_index"] == expected_index
    assert row["predicted_yield"] == pytest.approx(shifted[expected_index])
    assert {
        "step",
        "query_index",
        "condition",
        "observed_yield",
        "predicted_yield",
    } <= row.keys()


def test_gpbo_step_matches_direct_ei_and_records_disabled_guidance() -> None:
    engine = LGBOEngine(
        "buchwald_sub4",
        seed=100,
        use_llm=False,
        n_iters=0,
    )
    surrogate, mean, std = _deterministic_posterior(engine)
    engine._surrogate = surrogate
    expected_index = int(
        np.argmax(engine._expected_improvement(mean, std, float(np.max(engine.y_obs))))
    )

    row = engine.step()

    assert row["query_index"] == expected_index
    assert row["predicted_yield"] == pytest.approx(mean[expected_index])
    assert row["guidance_status"] == "disabled"
    assert row["guidance_reason"] == "use_llm_false"
    assert row["selected_in_subspace"] is None


def test_engine_health_tracks_success_and_surrogate_fallbacks() -> None:
    successful = LGBOEngine(
        "buchwald_sub4", seed=100, use_llm=False, n_iters=0
    )
    surrogate, _, _ = _deterministic_posterior(successful)
    successful._surrogate = surrogate

    successful.step()

    assert successful.health == {
        "gp_fit_fallbacks": 0,
        "gp_predict_fallbacks": 0,
        "acquisition_fallbacks": 0,
        "nonfinite_acquisition_scores": 0,
        "duplicate_queries": 0,
    }

    failing = LGBOEngine(
        "buchwald_sub4", seed=100, use_llm=False, n_iters=0
    )
    failing._surrogate = _FailingSurrogate()

    row = failing.step()

    assert np.isfinite(row["predicted_yield"])
    assert failing.health["gp_fit_fallbacks"] == 1
    assert failing.health["gp_predict_fallbacks"] == 1


def test_engine_health_tracks_nonfinite_acquisition_fallback() -> None:
    engine = LGBOEngine(
        "buchwald_sub4", seed=100, use_llm=False, n_iters=0
    )
    surrogate, _, _ = _deterministic_posterior(engine)
    engine._surrogate = surrogate
    engine._expected_improvement = lambda *_: np.full(engine.M, np.nan)

    row = engine.step()

    assert row["query_index"] == 0
    assert engine.health["nonfinite_acquisition_scores"] == engine.M
    assert engine.health["acquisition_fallbacks"] == 1
    assert engine.health["duplicate_queries"] == 0

