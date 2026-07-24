from __future__ import annotations

import warnings

import numpy as np
import pytest

from bo_core.optimization.lgbo import LGBOEngine


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


# ---- Slice 5: mean-shift (Proposition 1) math ----

def _fit_and_get_mu(engine):
    scaler, gp = engine._fit_gp()
    mu, sigma = engine._predict_pool(scaler, gp)
    return scaler, gp, mu, sigma


def test_mean_shift_zero_confidence_is_noop():
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    scaler, gp, mu, _ = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    assert np.allclose(e._mean_shift(scaler, gp, mu, x_p, 0.0), mu)


def test_mean_shift_matches_independent_reconstruction():
    """Verify mu_lambda = mu + lambda * K_post(pool,grid) @ a, lambda = c/sqrt(a'Sigma a)."""
    from scipy.linalg import cho_solve

    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    scaler, gp, mu, _ = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    c = 0.5
    ml = e._mean_shift(scaler, gp, mu, x_p, c)

    # Independently reconstruct the shift.
    x_p_s = scaler.transform(x_p.reshape(1, -1))
    d = len(e.feature_cols)
    hamming = d - e.pool_X @ x_p
    K = min(e.K, e.M)
    grid_idx = np.argpartition(hamming, K - 1)[:K]
    X_grid_s = scaler.transform(e.pool_X[grid_idx])

    a = np.maximum(gp.kernel_(x_p_s, X_grid_s).ravel(), 0.0)
    a = a / a.sum()
    Sigma_GG = e._posterior_cov(gp, X_grid_s)
    lam = c / float(np.sqrt(a @ Sigma_GG @ a))

    X_pool_s = scaler.transform(e.pool_X)
    K_post = e._posterior_cross_cov(gp, X_pool_s, X_grid_s)
    expected = mu + lam * (K_post @ a)

    assert np.allclose(ml, expected, rtol=1e-6, atol=1e-8)
    assert lam > 0


def test_mean_shift_lifts_proposed_point_and_is_mean_only():
    """The shift raises the mean at the proposed point; sigma is untouched."""
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    scaler, gp, mu, sigma = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    ml = e._mean_shift(scaler, gp, mu, x_p, 0.5)
    # x_p is its own nearest grid point; the lift increases its mean.
    assert ml[0] > mu[0]
    # Mean-only shift: sigma array is not modified by _mean_shift (passed separately).
    sigma_after = sigma.copy()
    _ = e._mean_shift(scaler, gp, mu, x_p, 0.5)
    assert np.array_equal(sigma, sigma_after)


def test_mean_shift_scales_linearly_with_confidence():
    """lambda ~ c, so doubling c doubles the shift (Proposition 1 calibration)."""
    e = LGBOEngine("buchwald_sub4", seed=100, use_llm=False, n_iters=0, n_restarts=2)
    scaler, gp, mu, _ = _fit_and_get_mu(e)
    x_p = e.pool_X[0]
    shift_half = e._mean_shift(scaler, gp, mu, x_p, 0.5) - mu
    shift_full = e._mean_shift(scaler, gp, mu, x_p, 1.0) - mu
    assert np.allclose(shift_full, 2.0 * shift_half, rtol=1e-6, atol=1e-8)
