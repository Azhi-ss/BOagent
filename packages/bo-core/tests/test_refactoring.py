import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from bo_core.optimization.memory import VectorMemory, Insight, EmbeddingClient
from bo_core.optimization.optimizer import BayesianOptimizer
from bo_core.optimization.space import DiscreteSearchSpace

def test_vector_memory_background_embedding():
    # Mock embedding client to return a dummy vector after a short sleep
    mock_client = MagicMock(spec=EmbeddingClient)
    mock_client.is_available.return_value = True
    
    dummy_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    # We want a small delay to verify the background thread nature
    def slow_embed(texts):
        time.sleep(0.1)
        return np.array([dummy_vector] * len(texts))
        
    mock_client.embed.side_effect = slow_embed

    with patch("bo_core.optimization.memory.EmbeddingClient", return_value=mock_client):
        mem = VectorMemory()
        insight = Insight(
            iteration=1,
            best_score=20.0,
            notes=["A note about perovskite"],
            key_findings=["Finding A"],
            parameter_relationships=["Rel B"],
            optimization_principles=["Rule C"]
        )
        
        # Add insight
        mem.add(insight)
        
        # Verify it is appended immediately, but embedding is initially None (or we wait a tiny bit)
        assert len(mem._insights) == 1
        assert mem._insights[0] == insight
        
        # Wait a bit for the background thread to finish
        time.sleep(0.3)
        
        # Verify embedding is updated
        assert mem._embeddings[0] is not None
        assert np.array_equal(mem._embeddings[0], dummy_vector)

def test_vector_memory_query_with_partial_embeddings():
    # Create memory with 2 insights: one has embedding, one doesn't
    mem = VectorMemory()
    insight1 = Insight(iteration=1, best_score=10.0, notes=["one"])
    insight2 = Insight(iteration=2, best_score=12.0, notes=["two"])
    
    mem._insights = [insight1, insight2]
    mem._embeddings = [np.array([1.0, 0.0], dtype=np.float32), None]
    
    # Mock embed client query to return [1.0, 0.0]
    mem._embed_client = MagicMock(spec=EmbeddingClient)
    mem._embed_client.is_available.return_value = True
    mem._embed_client.embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    
    # Query should match insight1 (the only one with a valid embedding)
    res = mem.query("test query", top_k=1)
    assert len(res) == 1
    assert res[0] == insight1

def test_optimizer_viability_timeout():
    # Setup mock knowledge engine where evaluate_candidate_viability hangs
    knowledge = MagicMock()
    knowledge._client.is_configured.return_value = True
    knowledge.build_system_prompt_for_viability.return_value = "System prompt"
    knowledge.enrich_suggestions.side_effect = lambda x: x
    
    # Let evaluate_candidate_viability sleep for 0.5s to trigger timeout
    # We will pass a low timeout in the test to verify timeout handling
    def slow_eval(cand, prompt, cols):
        time.sleep(0.2)
        return -0.01

    knowledge.evaluate_candidate_viability.side_effect = slow_eval

    # Create dummy space and optimizer
    df = pd.DataFrame([
        {"A": 1.0, "score": 21.0, "mean": 21.0, "std": 1.0},
    ])
    space = DiscreteSearchSpace(df, ["A"])
    
    optimizer = BayesianOptimizer(
        space=space,
        target_name="score",
        knowledge_engine=knowledge
    )
    optimizer.observe({"A": 4.0}, 15.0)
    optimizer._score_candidates = MagicMock(return_value=df)
    
    # Patch f.result(timeout=30.0) inside suggest to use timeout=0.01 to force timeout
    with patch("concurrent.futures.Future.result", side_effect=TimeoutError("Forced timeout")):
        result = optimizer.suggest(
            top_k=1,
            n_candidates=1,
            use_llm=True,
            gamma=0.5,
            use_logprobs=True
        )
        
        # Verify it fallback to using default log_prob -2.0
        assert len(result.suggestions) == 1
        assert "LLM Log-prob=-2.0000" in result.analysis
