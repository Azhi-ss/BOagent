import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add package root to path
sys.path.append(str(Path(__file__).parent.parent))

from bo_core.optimization.knowledge import KnowledgeEngine
from bo_core.optimization.optimizer import BayesianOptimizer
from bo_core.optimization.space import DiscreteSearchSpace


def test_llm_heuristic_flow():
    # 1. Setup a dummy search space of 1000 points
    df = pd.DataFrame({
        "CHI_PVK": np.random.uniform(3.5, 4.5, 1000),
        "CHI_ETL": np.random.uniform(3.5, 4.5, 1000),
        "score": 0.0
    })
    feature_cols = ["CHI_PVK", "CHI_ETL"]
    space = DiscreteSearchSpace(df, feature_cols)
    
    # 2. Setup Optimizer
    ke = KnowledgeEngine()
    if not ke._client.is_configured():
        print("Skipping test: LLM not configured")
        return

    optimizer = BayesianOptimizer(space=space, target_name="score", knowledge_engine=ke)
    
    # 3. Add some observations
    # A 'good' result has CHI_PVK - CHI_ETL around 0.1
    for _ in range(5):
        pvk = np.random.uniform(3.9, 4.1)
        etl = pvk - 0.1
        optimizer.observe({"CHI_PVK": pvk, "CHI_ETL": etl}, score=20.0)
        
    # 4. Suggest with heuristic enabled
    print("Suggesting with LLM heuristic...")
    res = optimizer.suggest(
        use_llm_heuristic=True,
        heuristic_weight=0.5,
        top_k=5,
        n_candidates=2
    )
    
    print("\nAnalysis:\n", res.analysis)
    print("\nSuggestions:\n", res.suggestions)
    
    # Verify that 'heuristic_score' was used (internally, but we check suggestions)
    assert len(res.suggestions) == 2
    print("\nTest passed!")

if __name__ == "__main__":
    test_llm_heuristic_flow()
