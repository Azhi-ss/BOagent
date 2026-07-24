from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, HealthCheck
from bo_core.optimization.space import ContinuousSearchSpace, DiscreteSearchSpace
from bo_core.optimization.optimizer import BayesianOptimizer

@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
@given(
    v1_min=st.floats(min_value=0.1, max_value=2.0),
    v1_max=st.floats(min_value=2.1, max_value=5.0),
    v2_min=st.floats(min_value=10.0, max_value=50.0),
    v2_max=st.floats(min_value=50.1, max_value=100.0),
)
def test_continuous_space_hypothesis_bounds(v1_min: float, v1_max: float, v2_min: float, v2_max: float):
    """Property test: LatinHypercube sampling strictly respects variable min/max bounds."""
    variables = [
        {"name": "feat_a", "min": v1_min, "max": v1_max},
        {"name": "feat_b", "min": v2_min, "max": v2_max},
    ]
    space = ContinuousSearchSpace(variables, n_samples=50, seed=123)
    pool = space.get_unobserved(pd.DataFrame())
    
    assert len(pool) == 50
    assert pool["feat_a"].min() >= v1_min - 1e-6
    assert pool["feat_a"].max() <= v1_max + 1e-6
    assert pool["feat_b"].min() >= v2_min - 1e-6
    assert pool["feat_b"].max() <= v2_max + 1e-6
    assert not pool.isnull().any().any()


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=10)
@given(
    obs_count=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=1, max_value=100)
)
def test_optimizer_gp_scoring_stability_hypothesis(obs_count: int, seed: int):
    """Property test: GP scoring returns finite float values without NaNs for arbitrary valid observations."""
    feature_cols = ["x1", "x2"]
    rng = np.random.RandomState(seed)
    
    raw_x = rng.uniform(low=0.1, high=10.0, size=(obs_count, 2))
    raw_y = rng.uniform(low=0.0, high=100.0, size=obs_count)
    
    df_pool = pd.DataFrame(rng.uniform(low=0.1, high=10.0, size=(30, 2)), columns=feature_cols)
    space = DiscreteSearchSpace(df_pool, feature_cols)
    
    optimizer = BayesianOptimizer(space, seed=seed)
    
    for i in range(obs_count):
        optimizer.observe(dict(zip(feature_cols, raw_x[i])), float(raw_y[i]))
        
    scored = optimizer._score_candidates(df_pool, acquisition="ucb", kappa=2.576, xi=0.01)
    
    assert not scored["score"].isnull().any()
    assert not scored["mean"].isnull().any()
    assert not scored["std"].isnull().any()
    assert np.all(np.isfinite(scored["score"].values))
    assert np.all(scored["std"].values >= 0)
