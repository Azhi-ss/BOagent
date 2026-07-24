from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from bo_core.optimization.space import SearchSpace
from bo_core.optimization.knowledge import KnowledgeEngine, SuggestionResult

class BayesianOptimizer:
    """A deep module for Bayesian Optimization cycles.
    
    Provides leverage by hiding the complexity of surrogate modeling,
    acquisition scoring, and domain-informed refinement.
    """

    def __init__(
        self,
        space: SearchSpace,
        target_name: str = "score",
        knowledge_engine: Optional[KnowledgeEngine] = None,
        n_restarts_optimizer: int = 10,
        seed: int = 42
    ):
        self.space = space
        self.target_name = target_name
        self.knowledge_engine = knowledge_engine or KnowledgeEngine()
        self.observed_configs = pd.DataFrame(columns=space.feature_cols)
        self.observed_scores: List[float] = []
        self.best_score_so_far = -float("inf")
        self.scientific_notes = ""
        self.n_restarts_optimizer = n_restarts_optimizer
        self.seed = seed

    def observe(self, config: Dict[str, float], score: float):
        """Record a new observation."""
        new_row = pd.DataFrame([config], columns=self.space.feature_cols)
        self.observed_configs = pd.concat([self.observed_configs, new_row], ignore_index=True)
        self.observed_scores.append(score)
        
        # Dynamic Scientific Memory (DSM): Summarize lessons learned if a new best is found
        # Delay until at least 10 points to ensure meaningful physical trends
        if score > self.best_score_so_far and len(self.observed_scores) >= 10:
            self.best_score_so_far = score
            try:
                self.scientific_notes = self.knowledge_engine.summarize_lessons(
                    self.observed_configs, self.observed_scores, self.space.feature_cols, self.target_name,
                    best_score=self.best_score_so_far
                )
            except Exception:
                self.scientific_notes = ""
        elif score > self.best_score_so_far:
            self.best_score_so_far = score

    def suggest(
        self,
        top_k: int = 20,
        n_candidates: int = 5,
        acquisition: str = "ucb",
        kappa: float = 2.576,
        xi: float = 0.01,
        use_llm: bool = True,
        gamma: float = 0.1,
        use_logprobs: bool = True,
        use_llm_heuristic: bool = False,
        heuristic_weight: float = 0.3,
        use_direct_full_pool: bool = False
    ) -> SuggestionResult:
        """Generate the next best experiments to try.
        
        Args:
            top_k: Number of candidates to pass to LLM for final refinement.
            n_candidates: Final number of suggestions to return.
            acquisition: GP acquisition function ('ucb', 'ei', 'pi').
            use_llm: Whether to use LLM refinement.
            use_logprobs: Use pointwise log-probs for top-K candidates.
            use_llm_heuristic: Use LLM-generated Python code to score the ENTIRE pool (10k+).
            heuristic_weight: Weight of LLM heuristic score in hybrid fusion [0, 1].
            use_direct_full_pool: Use batch-based direct LLM scoring for the pool (sampled).
        """
        # 1. Get candidates from the search space
        pool_df = self.space.get_unobserved(self.observed_configs)
        
        # 2. Fit GP and predict Acquisition scores
        scored_df = self._score_candidates(pool_df, acquisition, kappa, xi)
        
        # 3. Apply Full-Pool LLM Scoring (Directly scoring the raw search space)
        observed_data = []
        for i, (_, row) in enumerate(self.observed_configs.iterrows()):
            observed_data.append((row.to_dict(), self.observed_scores[i]))

        # Path A: LLM-Generated Heuristic (Scalable to 1M+ points)
        if use_llm_heuristic and self.knowledge_engine._client.is_configured() and not pool_df.empty:
            h_code = self.knowledge_engine.generate_physical_heuristic(
                self.target_name, self.space.feature_cols, observed_data
            )
            h_scores = self.knowledge_engine.apply_heuristic_to_pool(scored_df, h_code)
            
            if h_scores.std() > 0:
                h_min, h_max = h_scores.min(), h_scores.max()
                h_norm = (h_scores - h_min) / (h_max - h_min)
            else:
                h_norm = h_scores
                
            scored_df["llm_score"] = h_norm
            scored_df["score"] = (1 - heuristic_weight) * scored_df["score"] + heuristic_weight * h_norm

        # Path B: Direct Batch Scoring (Higher fidelity, sampled for 10k+)
        elif use_direct_full_pool and self.knowledge_engine._client.is_configured() and not pool_df.empty:
            # For 10,000 formulations, we sample a representative set to keep costs down
            # or we could batch everything. Here we take top 100 by GP as representative samples.
            sample_size = min(100, len(scored_df))
            sample_df = scored_df.sort_values("score", ascending=False).head(sample_size)
            
            batch_scores = self.knowledge_engine.score_candidates_direct_batch(
                self.target_name, self.space.feature_cols, 
                sample_df[self.space.feature_cols].to_dict("records"),
                observed_data
            )
            
            # Map back to full scored_df (others get 0 or mean)
            scored_df["llm_score"] = 0.0
            scored_df.loc[sample_df.index, "llm_score"] = batch_scores
            scored_df["score"] = (1 - heuristic_weight) * scored_df["score"] + heuristic_weight * scored_df["llm_score"]
        
        # 4. Take Top-K for refinement
        top_candidates = scored_df.sort_values("score", ascending=False).head(top_k)
        
        # 5. Domain Refinement (LLM Pointwise Log-probs)

        if use_llm and use_logprobs and self.knowledge_engine._client.is_configured() and not top_candidates.empty:
            # Pointwise Log-probs based evaluation
            system_prompt = self.knowledge_engine.build_system_prompt_for_viability(
                self.target_name,
                self.space.feature_cols,
                observed_data
            )
            prompt = system_prompt # Store system prompt for audit / logs
            
            candidates_list = [row[self.space.feature_cols].to_dict() for _, row in top_candidates.iterrows()]
            gp_means = top_candidates["mean"].values if "mean" in top_candidates.columns else [None] * len(candidates_list)
            gp_stds = top_candidates["std"].values if "std" in top_candidates.columns else [None] * len(candidates_list)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates_list)) as executor:
                futures = [
                    executor.submit(
                        self.knowledge_engine.evaluate_candidate_viability,
                        cand,
                        system_prompt,
                        self.space.feature_cols,
                        gp_mean=float(gp_mean) if gp_mean is not None else None,
                        gp_std=float(gp_std) if gp_std is not None else None,
                    )
                    for cand, gp_mean, gp_std in zip(candidates_list, gp_means, gp_stds)
                ]
                log_probs = []
                for f in futures:
                    try:
                        log_probs.append(f.result(timeout=30.0))
                    except concurrent.futures.TimeoutError:
                        print("[BO] Candidate viability evaluation timed out. Using default log_prob.")
                        log_probs.append(-2.0)
                    except Exception as e:
                        print(f"[BO] Candidate viability evaluation failed: {e}. Using default log_prob.")
                        log_probs.append(-2.0)
            
            # Adaptive scaling weight: lambda_t = gamma * std(GP_scores)
            gp_scores = top_candidates["score"].values
            gp_std = float(np.std(gp_scores)) if len(gp_scores) > 1 else 1.0
            if gp_std == 0:
                gp_std = 1.0
            lambda_t = gamma * gp_std
            
            # Compute hybrid score
            hybrid_scores = gp_scores + lambda_t * np.array(log_probs)
            
            top_candidates_copy = top_candidates.copy()
            top_candidates_copy["hybrid_score"] = hybrid_scores
            top_candidates_copy["log_prob"] = log_probs
            sorted_candidates = top_candidates_copy.sort_values("hybrid_score", ascending=False)
            
            suggestions = [
                row[self.space.feature_cols].to_dict()
                for _, row in sorted_candidates.head(n_candidates).iterrows()
            ]
            suggestions = self.knowledge_engine.enrich_suggestions(suggestions)
            
            analysis_lines = ["Log-probs Hybrid Selection Analysis:"]
            for idx, (_, row) in enumerate(sorted_candidates.head(n_candidates).iterrows()):
                cbo_str = ""
                if "CHI_PVK" in row and "CHI_ETL" in row:
                    cbo = row["CHI_PVK"] - row["CHI_ETL"]
                    cbo_str = f", CBO={cbo:.3f}eV"
                analysis_lines.append(
                    f"Selected Candidate {idx+1}: GP Score={row['score']:.4f}, LLM Log-prob={row['log_prob']:.4f}, Hybrid Score={row['hybrid_score']:.4f}{cbo_str}"
                )
            analysis = "\n".join(analysis_lines)
            
        else:
            # Fallback to legacy/baseline prompt list selection or pure GP
            prompt = self.knowledge_engine.build_prompt(
                self.target_name,
                self.space.feature_cols,
                top_candidates,
                observed_data,
                n_candidates=n_candidates,
                scientific_notes=self.scientific_notes
            )
            
            if use_llm:
                suggestions, analysis = self.knowledge_engine.refine_suggestions(
                    prompt, top_candidates, self.space.feature_cols, n_candidates=n_candidates
                )
            else:
                suggestions = [row[self.space.feature_cols].to_dict() for _, row in top_candidates.head(n_candidates).iterrows()]
                suggestions = self.knowledge_engine.enrich_suggestions(suggestions)
                analysis = "LLM refinement skipped. Using GP top candidates."

        # Guarantee at least one suggestion: fall back to GP-ranked top candidates.
        if not suggestions and not top_candidates.empty:
            suggestions = [
                row[self.space.feature_cols].to_dict()
                for _, row in top_candidates.head(n_candidates).iterrows()
            ]
            suggestions = self.knowledge_engine.enrich_suggestions(suggestions)

        return SuggestionResult(
            suggestions=suggestions,
            analysis=analysis,
            prompt=prompt
        )

    def get_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[pd.DataFrame, float, float]:
        """Legacy compatibility method for PVKBO.

        Maps alpha to kappa and calls suggest().
        """
        import time
        start_time = time.time()
        
        # Sync internal state if needed (though typically observe() is used)
        if not observed_configs.empty:
            # We assume current BayesianOptimizer state might need sync or just use provided data
            # For pure compatibility with PVKBO which passes state every call:
            self.observed_configs = observed_configs[self.space.feature_cols].copy()
            self.observed_scores = observed_fvals["score"].tolist()
            
            # Update best score and notes if needed
            max_score = max(self.observed_scores)
            if max_score > self.best_score_so_far:
                self.best_score_so_far = max_score
                try:
                    self.scientific_notes = self.knowledge_engine.summarize_lessons(
                        self.observed_configs, self.observed_scores, self.space.feature_cols, self.target_name,
                        best_score=max_score
                    )
                except Exception:
                    pass

        # Use kappa = alpha if provided, matching legacy behavior
        kappa = alpha if alpha is not None else getattr(self, "kappa", getattr(self, "alpha", 0.1))
        n_candidates = getattr(self, "n_candidates", 5)
        acquisition = getattr(self, "acquisition", "ucb")
        xi = getattr(self, "xi", 0.01)
        
        res = self.suggest(
            kappa=kappa,
            use_llm=True,
            n_candidates=n_candidates,
            acquisition=acquisition,
            xi=xi
        )
        
        candidate_df = pd.DataFrame(res.suggestions)
        return candidate_df, 0.0, time.time() - start_time

    def _score_candidates(self, pool_df: pd.DataFrame, acquisition: str, kappa: float, xi: float) -> pd.DataFrame:
        """Train GP and compute acquisition scores for the pool."""
        rng = np.random.RandomState(self.seed) # Local seeded RNG to avoid global lock/state issues in threads
        if self.observed_configs.empty or len(self.observed_configs) < 1:
            result = pool_df.copy()
            result["score"] = rng.uniform(0, 1, size=len(pool_df))
            result["mean"] = 0.0
            result["std"] = 1.0
            return result

        try:
            from scipy.stats import norm
            X_train = self.observed_configs[self.space.feature_cols].values
            y_train = np.array(self.observed_scores)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            kernel = C(1.0, (1e-3, 1e3)) * Matern([1.0] * len(self.space.feature_cols), (1e-2, 1e2), nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel, 
                n_restarts_optimizer=self.n_restarts_optimizer, 
                alpha=1e-6, 
                normalize_y=True,
                random_state=self.seed
            )
            gp.fit(X_train_scaled, y_train)
            
            X_pool = pool_df[self.space.feature_cols].values
            X_pool_scaled = scaler.transform(X_pool)
            
            mu, sigma = gp.predict(X_pool_scaled, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            
            best_f = np.max(y_train)
            
            if acquisition == "ucb":
                scores = mu + kappa * sigma
            elif acquisition == "ei":
                imp = mu - best_f - xi
                z = imp / sigma
                scores = imp * norm.cdf(z) + sigma * norm.pdf(z)
            elif acquisition == "pi":
                imp = mu - best_f - xi
                z = imp / sigma
                scores = norm.cdf(z)
            else:
                scores = mu # Default to mean
            
            result = pool_df.copy()
            result["score"] = scores
            result["mean"] = mu
            result["std"] = sigma
            return result
        except Exception as e:
            # Fallback for GP training failures
            print(f"[BO] GP scoring failed: {e}. Falling back to random scores.")
            result = pool_df.copy()
            result["score"] = rng.uniform(0, 1, size=len(pool_df))
            result["mean"] = 0.0
            result["std"] = 1.0
            return result
