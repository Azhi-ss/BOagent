from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from optimization.space import SearchSpace
from optimization.knowledge import KnowledgeEngine, SuggestionResult

class BayesianOptimizer:
    """A deep module for Bayesian Optimization cycles.
    
    Provides leverage by hiding the complexity of surrogate modeling,
    acquisition scoring, and domain-informed refinement.
    """

    def __init__(
        self,
        space: SearchSpace,
        target_name: str = "score",
        knowledge_engine: Optional[KnowledgeEngine] = None
    ):
        self.space = space
        self.target_name = target_name
        self.knowledge_engine = knowledge_engine or KnowledgeEngine()
        self.observed_configs = pd.DataFrame(columns=space.feature_cols)
        self.observed_scores: List[float] = []

    def observe(self, config: Dict[str, float], score: float):
        """Record a new observation."""
        new_row = pd.DataFrame([config], columns=self.space.feature_cols)
        self.observed_configs = pd.concat([self.observed_configs, new_row], ignore_index=True)
        self.observed_scores.append(score)

    def suggest(
        self,
        top_k: int = 20,
        kappa: float = 2.576,
        use_llm: bool = True
    ) -> SuggestionResult:
        """Generate the next best experiments to try."""
        # 1. Get candidates from the search space
        pool_df = self.space.get_unobserved(self.observed_configs)
        
        # 2. Fit GP and predict UCB
        scored_df = self._score_candidates(pool_df, kappa)
        
        # 3. Take Top-K for refinement
        top_candidates = scored_df.sort_values("ucb", ascending=False).head(top_k)
        
        # 4. Domain Refinement (LLM)
        observed_data = []
        for i, (_, row) in enumerate(self.observed_configs.iterrows()):
            observed_data.append((row.to_dict(), self.observed_scores[i]))

        prompt = self.knowledge_engine.build_prompt(
            self.target_name,
            self.space.feature_cols,
            top_candidates,
            observed_data
        )
        
        if use_llm:
            suggestions, analysis = self.knowledge_engine.refine_suggestions(
                prompt, top_candidates, self.space.feature_cols
            )
        else:
            suggestions = [row[self.space.feature_cols].to_dict() for _, row in top_candidates.head(5).iterrows()]
            analysis = "LLM refinement skipped. Using GP top candidates."

        # Guarantee at least one suggestion: fall back to GP-ranked top candidates.
        if not suggestions and not top_candidates.empty:
            suggestions = [
                row[self.space.feature_cols].to_dict()
                for _, row in top_candidates.head(5).iterrows()
            ]

        return SuggestionResult(
            suggestions=suggestions,
            analysis=analysis,
            prompt=prompt
        )

    def _score_candidates(self, pool_df: pd.DataFrame, kappa: float) -> pd.DataFrame:
        """Train GP and compute UCB scores for the pool."""
        if self.observed_configs.empty:
            result = pool_df.copy()
            result["ucb"] = 0.0
            result["mean"] = 0.0
            result["std"] = 1.0
            return result

        X_train = self.observed_configs[self.space.feature_cols].values
        y_train = np.array(self.observed_scores)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * len(self.space.feature_cols), (1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, normalize_y=True
        )
        gp.fit(X_train_scaled, y_train)
        
        X_pool = pool_df[self.space.feature_cols].values
        X_pool_scaled = scaler.transform(X_pool)
        
        mu, sigma = gp.predict(X_pool_scaled, return_std=True)
        ucb = mu + kappa * sigma
        
        result = pool_df.copy()
        result["ucb"] = ucb
        result["mean"] = mu
        result["std"] = sigma
        return result
