from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gp_llm_acq import GPLLM_ACQ


def _band_alignment_task_context() -> dict:
    """Minimal task_context for band_alignment testing."""
    df = pd.DataFrame(
        [
            {"CHI_PVK": 3.8, "Eg_HTL": 2.1, "CHI_HTL": 2.4, "Eg_ETL": 3.0, "CHI_ETL": 4.1, "eta": 22.4},
            {"CHI_PVK": 3.9, "Eg_HTL": 2.0, "CHI_HTL": 2.5, "Eg_ETL": 3.1, "CHI_ETL": 4.0, "eta": 23.2},
            {"CHI_PVK": 4.0, "Eg_HTL": 2.2, "CHI_HTL": 2.6, "Eg_ETL": 3.2, "CHI_ETL": 4.2, "eta": 21.9},
            {"CHI_PVK": 4.1, "Eg_HTL": 2.3, "CHI_HTL": 2.3, "Eg_ETL": 3.3, "CHI_ETL": 4.3, "eta": 24.1},
            {"CHI_PVK": 3.7, "Eg_HTL": 1.9, "CHI_HTL": 2.7, "Eg_ETL": 2.9, "CHI_ETL": 3.9, "eta": 20.8},
            {"CHI_PVK": 4.2, "Eg_HTL": 2.4, "CHI_HTL": 2.8, "Eg_ETL": 3.4, "CHI_ETL": 4.4, "eta": 25.0},
            {"CHI_PVK": 3.6, "Eg_HTL": 1.8, "CHI_HTL": 2.2, "Eg_ETL": 2.8, "CHI_ETL": 3.8, "eta": 19.5},
        ]
    )
    feature_cols = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
    constraints = {}
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        constraints[col] = ["float", "linear", [float(values.min()), float(values.max())]]

    return {
        "model": "band_alignment",
        "task": "regression",
        "metric": "neg_mean_squared_error",
        "num_classes": 1,
        "n_classes": 1,
        "lower_is_better": False,
        "num_samples": len(df),
        "tot_feats": len(feature_cols),
        "cat_feats": 0,
        "num_feats": len(feature_cols),
        "feature_cols": feature_cols,
        "target_col": "eta",
        "hyperparameter_constraints": constraints,
        "df": df,
    }


def _observed_data(task_context: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First 2 rows as observed."""
    df = task_context["df"]
    configs = df.iloc[:2][task_context["feature_cols"]].reset_index(drop=True)
    fvals = pd.DataFrame({"score": df.iloc[:2]["eta"].values})
    return configs, fvals


class TestGPLLM_ACQ:
    def test_init_stores_attributes(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=10,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        assert acq.n_candidates == 10
        assert acq.feature_cols == ctx["feature_cols"]
        assert acq.target_col == "eta"
        assert acq.top_k == 20
        assert acq.alpha == 0.1

    def test_find_unobserved_filters_observed_points(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=10,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        configs, fvals = _observed_data(ctx)

        unobserved = acq._find_unobserved(configs)
        # 7 total rows, 2 observed → 5 unobserved
        assert len(unobserved) == 5

    def test_get_candidate_points_returns_dataframe(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=3,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
            top_k=5,
        )
        configs, fvals = _observed_data(ctx)

        # Mock LLM to return a valid selection
        with patch.object(acq, "_call_llm", return_value="Analysis: test\nSelected Formulations: 1, 2, 3"):
            candidates, cost, elapsed = acq.get_candidate_points(configs, fvals)

        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) <= 3
        for col in ctx["feature_cols"]:
            assert col in candidates.columns
        assert isinstance(cost, float)
        assert isinstance(elapsed, float)

    def test_fallback_when_llm_fails(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=3,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
            top_k=5,
        )
        configs, fvals = _observed_data(ctx)

        # Mock LLM to raise
        with patch.object(acq, "_call_llm", side_effect=RuntimeError("API error")):
            candidates, cost, elapsed = acq.get_candidate_points(configs, fvals)

        # Should fall back to GP UCB
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 3

    def test_all_points_observed_returns_random_candidates(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=2,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        # All rows as observed
        df = ctx["df"]
        configs = df[ctx["feature_cols"]].copy()
        fvals = pd.DataFrame({"score": df["eta"].values})

        candidates, cost, elapsed = acq.get_candidate_points(configs, fvals)
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 2

    def test_build_batch_evaluation_prompt_contains_features(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=5,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        configs, fvals = _observed_data(ctx)
        unobserved = acq._find_unobserved(configs)
        X_train, y_train = acq._build_training_data(configs, fvals)
        predictions = acq._gp_ucb_predict(X_train, y_train, unobserved, 0.1)
        top = predictions[:3]
        observed_data = acq._build_observed_data(configs, fvals)

        prompt = acq._build_batch_evaluation_prompt(top, observed_data)

        assert "CHI_PVK" in prompt
        assert "Eg_HTL" in prompt
        assert "eta" in prompt
        assert "Selected Formulations" in prompt
        assert "GP Prediction" in prompt

    def test_parse_llm_response_standard_format(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=5,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        response = "Analysis: These look promising.\nSelected Formulations: 1, 3, 5, 7, 9"
        indices = acq._parse_llm_response(response, 20)
        assert indices == [0, 2, 4, 6, 8]

    def test_parse_llm_response_fallback_to_numbers(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=5,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        response = "I recommend formulations 2, 4, and 6."
        indices = acq._parse_llm_response(response, 20)
        assert 1 in indices  # formulation 2 → index 1
        assert 3 in indices  # formulation 4 → index 3
        assert 5 in indices  # formulation 6 → index 5
