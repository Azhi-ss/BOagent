from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from optimization.optimizer import BayesianOptimizer
from optimization.space import DiscreteSearchSpace

def mock_pvk_data():
    """Synthetic band alignment data."""
    feature_cols = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
    data = np.random.rand(100, 5)
    df = pd.DataFrame(data, columns=feature_cols)
    df["eta"] = np.random.rand(100)
    return df, feature_cols

class TestAcquisitionUnification:
    """Tests ensuring BayesianOptimizer handles PVK-LLM requirements."""

    def test_high_fidelity_gp_config(self):
        """
        Verify that BayesianOptimizer can be configured with PVK-LLM parameters.
        """
        df, features = mock_pvk_data()
        space = DiscreteSearchSpace(df, features)
        
        optimizer = BayesianOptimizer(
            space=space,
            target_name="eta",
            n_restarts_optimizer=10 # PVK-LLM requirement
        )
        
        # Add some history
        obs_configs = df.head(5)[features]
        obs_scores = df.head(5)["eta"].values
        for i in range(5):
            optimizer.observe(obs_configs.iloc[i].to_dict(), obs_scores[i])

        # Suggest
        result = optimizer.suggest(
            top_k=20,
            n_candidates=5,
            kappa=0.1,
            use_llm=False
        )
        
        assert len(result.suggestions) == 5
        # Suggestions are enriched with physics parameters (CBO, VBO)
        for sug in result.suggestions:
            for feat in features:
                assert feat in sug
            assert "CBO" in sug
            assert "CBO_Status" in sug
            assert "VBO" in sug
            assert "VBO_Status" in sug

    def test_vectorized_performance_preserved(self):
        """
        Verify that DiscreteSearchSpace still handles large pools efficiently.
        """
        data = np.random.rand(5000, 5)
        feature_cols = ["A", "B", "C", "D", "E"]
        df = pd.DataFrame(data, columns=feature_cols)
        space = DiscreteSearchSpace(df, feature_cols)
        
        obs_df = df.head(100)
        
        import time
        start = time.time()
        unobserved = space.get_unobserved(obs_df)
        duration = time.time() - start
        
        assert len(unobserved) == 4900
        assert duration < 0.1 # Should be very fast with numpy
