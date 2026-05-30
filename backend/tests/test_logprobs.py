from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from llm_client import LlmCallResult, DeepSeekClient
from optimization.knowledge import extract_yes_logprob, KnowledgeEngine
from optimization.optimizer import BayesianOptimizer
from optimization.space import DiscreteSearchSpace


# ---------------------------------------------------------------------------
# 1. Test extract_yes_logprob
# ---------------------------------------------------------------------------

def test_extract_yes_logprob_handles_direct_yes():
    # Case 1: Generated token is "Yes" (with logprob 0.0)
    payload = {
        "content": [
            {
                "token": "Yes",
                "logprob": -0.05,
                "top_logprobs": [
                    {"token": "Yes", "logprob": -0.05},
                    {"token": "No", "logprob": -3.0}
                ]
            }
        ]
    }
    assert extract_yes_logprob(payload) == -0.05

    # Case 2: Generated token is " yes" (space prefix and lowercase)
    payload["content"][0]["token"] = " yes"
    assert extract_yes_logprob(payload) == -0.05


def test_extract_yes_logprob_handles_yes_in_top_logprobs():
    # Generated token is "No", but "Yes" is in top_logprobs
    payload = {
        "content": [
            {
                "token": "No",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "No", "logprob": -0.1},
                    {"token": "Yes", "logprob": -2.5}
                ]
            }
        ]
    }
    assert extract_yes_logprob(payload) == -2.5


def test_extract_yes_logprob_handles_yes_not_in_top_logprobs():
    # Generated token is "No", and "Yes" is NOT in top_logprobs
    payload = {
        "content": [
            {
                "token": "No",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "No", "logprob": -0.1},
                    {"token": "Maybe", "logprob": -3.0}
                ]
            }
        ]
    }
    # Should fallback to min_top_logprob - 2.0 (i.e. -3.0 - 2.0 = -5.0)
    assert extract_yes_logprob(payload) == -5.0

    # No top_logprobs at all
    payload["content"][0]["top_logprobs"] = []
    assert extract_yes_logprob(payload) == -10.0


def test_extract_yes_logprob_handles_empty_or_none():
    assert extract_yes_logprob(None) == -20.0
    assert extract_yes_logprob({}) == -20.0
    assert extract_yes_logprob({"content": []}) == -20.0


# ---------------------------------------------------------------------------
# 2. Test system prompt construction
# ---------------------------------------------------------------------------

def test_build_system_prompt_for_viability():
    engine = KnowledgeEngine(chat_engine=None)
    
    # 1. Band alignment context
    feature_cols = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
    observed_data = [
        ({"CHI_PVK": 3.9, "Eg_HTL": 1.9, "CHI_HTL": 4.0, "Eg_ETL": 2.0, "CHI_ETL": 4.1}, 20.5),
        ({"CHI_PVK": 3.8, "Eg_HTL": 1.8, "CHI_HTL": 3.9, "Eg_ETL": 1.9, "CHI_ETL": 4.0}, 18.0)
    ]
    
    prompt = engine.build_system_prompt_for_viability("PCE", feature_cols, observed_data)
    
    assert "Semiconductor Physics Rules for Energy Alignment" in prompt
    assert "CHI_PVK" in prompt
    assert "PCE=20.5000" in prompt

    # 2. Defects/Doping context
    feature_cols = ["Nt_PVK/ETL", "Na_PVK"]
    prompt2 = engine.build_system_prompt_for_viability("PCE", feature_cols, observed_data)
    assert "Physical Guidelines for Defect/Doping Optimization" in prompt2


# ---------------------------------------------------------------------------
# 3. Test pointwise candidate evaluation
# ---------------------------------------------------------------------------

@patch("llm_client.requests.post")
def test_evaluate_candidate_viability(mock_post):
    # Mock client config
    client = DeepSeekClient(api_key="sk-test")
    engine = KnowledgeEngine(chat_engine="deepseek-v4-flash")
    engine._client = client
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Yes"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Yes",
                            "logprob": -0.01,
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.01}
                            ]
                        }
                    ]
                }
            }
        ],
        "usage": {}
    }
    mock_post.return_value = mock_response
    
    candidate = {"CHI_PVK": 3.9, "Eg_HTL": 1.9}
    feature_cols = ["CHI_PVK", "Eg_HTL"]
    
    logprob = engine.evaluate_candidate_viability(candidate, "System Prompt", feature_cols)
    assert logprob == -0.01
    
    # Verify post payload parameters
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["max_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5


# ---------------------------------------------------------------------------
# 4. Test suggest hybrid scoring
# ---------------------------------------------------------------------------

def test_suggest_hybrid_scoring_calculates_correct_ranking():
    # Setup mock knowledge engine
    knowledge = MagicMock()
    # Assume client is configured
    knowledge._client.is_configured.return_value = True
    knowledge.build_system_prompt_for_viability.return_value = "Mock System Prompt"
    
    # Mock evaluate_candidate_viability:
    # Candidate 1: viability -0.05 (highly viable)
    # Candidate 2: viability -5.0 (not viable)
    # Candidate 3: viability -0.1 (viable)
    def mock_eval(candidate, system_prompt, feature_cols):
        if candidate["A"] == 1.0:
            return -0.05
        elif candidate["A"] == 2.0:
            return -5.0
        else:
            return -0.1
            
    knowledge.evaluate_candidate_viability.side_effect = mock_eval

    # Create dummy space and optimizer
    df = pd.DataFrame([
        {"A": 1.0, "score": 21.0, "mean": 21.0, "std": 1.0}, # UCB=21.0, Logprob=-0.05
        {"A": 2.0, "score": 22.0, "mean": 22.0, "std": 1.0}, # UCB=22.0, Logprob=-5.0 (High UCB but physically bad)
        {"A": 3.0, "score": 20.5, "mean": 20.5, "std": 1.0}, # UCB=20.5, Logprob=-0.1
    ])
    space = DiscreteSearchSpace(df, ["A"])
    
    optimizer = BayesianOptimizer(
        space=space,
        target_name="score",
        knowledge_engine=knowledge
    )
    
    # Inject observed data so we don't return random initial points
    optimizer.observe({"A": 4.0}, 15.0)

    # Let's mock _score_candidates to return our custom df
    optimizer._score_candidates = MagicMock(return_value=df)
    
    # Call suggest with logprobs enabled
    result = optimizer.suggest(
        top_k=3,
        n_candidates=3,
        use_llm=True,
        gamma=0.5,
        use_logprobs=True
    )
    
    # UCB scores: [21.0, 22.0, 20.5]. std(UCB) = std([21.0, 22.0, 20.5]) = 0.6236.
    # For gamma = 0.5, lambda_t = 0.5 * 0.6236 = 0.3118.
    # Hybrid scores:
    # Cand 1 (A=1.0): 21.0 + 0.3118 * (-0.05) = 21.0 - 0.0156 = 20.9844
    # Cand 2 (A=2.0): 22.0 + 0.3118 * (-5.0) = 22.0 - 1.559 = 20.441
    # Cand 3 (A=3.0): 20.5 + 0.3118 * (-0.1) = 20.5 - 0.0312 = 20.4688
    #
    # Ranking order should be: Cand 1 (A=1.0) > Cand 3 (A=3.0) > Cand 2 (A=2.0)
    # Note that Candidate 2 had the highest UCB (22.0) but got penalized heavily by the LLM logprob!
    
    assert len(result.suggestions) == 3
    assert result.suggestions[0]["A"] == 1.0
    assert result.suggestions[1]["A"] == 3.0
    assert result.suggestions[2]["A"] == 2.0
    assert "Selected Candidate 1: GP Score=21.0" in result.analysis
    assert "LLM Log-prob=-0.05" in result.analysis
