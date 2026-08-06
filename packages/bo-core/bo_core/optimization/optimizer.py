from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from bo_core.optimization.space import SearchSpace
from bo_core.optimization.surrogate import BackendName, create_surrogate


@dataclass
class SuggestionResult:
    suggestions: list[dict[str, float]]
    analysis: str
    prompt: str


class KnowledgeEngine(Protocol):
    """Optional domain guidance consumed by BayesianOptimizer."""

    def is_configured(self) -> bool: ...

    def summarize_lessons(
        self,
        observed_configs: pd.DataFrame,
        observed_scores: list[float],
        feature_cols: list[str],
        target_name: str,
        best_score: float = 0.0,
    ) -> str: ...

    def build_system_prompt_for_viability(
        self,
        target_name: str,
        feature_cols: list[str],
        observed_data: list[tuple[dict[str, float], float]],
    ) -> str: ...

    def evaluate_candidate_viability(
        self,
        candidate: dict[str, float],
        system_prompt: str,
        feature_cols: list[str],
        gp_mean: float | None = None,
        gp_std: float | None = None,
    ) -> float: ...

    def enrich_suggestions(
        self, suggestions: list[dict[str, float]]
    ) -> list[dict[str, float]]: ...


class BayesianOptimizer:
    """Domain-neutral Bayesian optimization with optional explicit guidance."""

    def __init__(
        self,
        space: SearchSpace,
        target_name: str = "score",
        knowledge_engine: KnowledgeEngine | None = None,
        n_restarts_optimizer: int = 10,
        seed: int = 42,
        backend: BackendName = "botorch",
    ) -> None:
        self.space = space
        self.target_name = target_name
        self.knowledge_engine = knowledge_engine
        self.observed_configs = pd.DataFrame(columns=space.feature_cols)
        self.observed_scores: list[float] = []
        self.best_score_so_far = -float("inf")
        self.scientific_notes = ""
        self.n_restarts_optimizer = n_restarts_optimizer
        self.seed = seed
        self.backend = backend
        self._surrogate = create_surrogate(
            backend,
            seed=seed,
            n_restarts=n_restarts_optimizer,
            alpha=1e-6,
        )

    def observe(self, config: dict[str, float], score: float) -> None:
        """Record a new observation."""
        new_row = pd.DataFrame([config], columns=self.space.feature_cols)
        self.observed_configs = pd.concat(
            [self.observed_configs, new_row], ignore_index=True
        )
        self.observed_scores.append(score)

        is_new_best = score > self.best_score_so_far
        if is_new_best:
            self.best_score_so_far = score
        if (
            is_new_best
            and self.knowledge_engine is not None
            and len(self.observed_scores) >= 10
        ):
            try:
                self.scientific_notes = self.knowledge_engine.summarize_lessons(
                    self.observed_configs,
                    self.observed_scores,
                    self.space.feature_cols,
                    self.target_name,
                    best_score=score,
                )
            except Exception:
                self.scientific_notes = ""

    def suggest(
        self,
        top_k: int = 20,
        n_candidates: int = 5,
        acquisition: str = "ucb",
        kappa: float = 2.576,
        xi: float = 0.01,
        use_llm: bool = False,
        gamma: float = 0.1,
    ) -> SuggestionResult:
        """Generate the next experiments, optionally refined by domain guidance."""
        if use_llm and self.knowledge_engine is None:
            raise ValueError("use_llm=True requires an explicit knowledge_engine")

        pool_df = self.space.get_unobserved(self.observed_configs)
        scored_df = self._score_candidates(pool_df, acquisition, kappa, xi)
        top_candidates = scored_df.sort_values("score", ascending=False).head(top_k)
        observed_data = [
            (row.to_dict(), self.observed_scores[i])
            for i, (_, row) in enumerate(self.observed_configs.iterrows())
        ]

        if not use_llm:
            suggestions = [
                row[self.space.feature_cols].to_dict()
                for _, row in top_candidates.head(n_candidates).iterrows()
            ]
            return SuggestionResult(
                suggestions=suggestions,
                analysis="LLM refinement skipped. Using GP top candidates.",
                prompt="",
            )

        knowledge = self.knowledge_engine
        assert knowledge is not None
        if not knowledge.is_configured():
            raise RuntimeError("LLM refinement requested but LLM client is not configured")

        if top_candidates.empty:
            raise RuntimeError("No candidates available for LLM refinement")

        prompt = knowledge.build_system_prompt_for_viability(
            self.target_name,
            self.space.feature_cols,
            observed_data,
        )
        candidates = [
            row[self.space.feature_cols].to_dict()
            for _, row in top_candidates.iterrows()
        ]
        means = top_candidates.get("mean", pd.Series([None] * len(candidates)))
        stds = top_candidates.get("std", pd.Series([None] * len(candidates)))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(candidates)
        ) as executor:
            futures = [
                executor.submit(
                    knowledge.evaluate_candidate_viability,
                    candidate,
                    prompt,
                    self.space.feature_cols,
                    gp_mean=float(mean) if mean is not None else None,
                    gp_std=float(std) if std is not None else None,
                )
                for candidate, mean, std in zip(candidates, means, stds)
            ]
            log_probs = []
            for future in futures:
                try:
                    log_probs.append(future.result(timeout=30.0))
                except concurrent.futures.TimeoutError as exc:
                    raise RuntimeError(
                        f"[BO] Candidate viability evaluation timed out: {exc}"
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(
                        f"[BO] Candidate viability evaluation failed: {exc}"
                    ) from exc

        gp_scores = top_candidates["score"].to_numpy()
        scale = float(np.std(gp_scores)) if len(gp_scores) > 1 else 1.0
        hybrid_scores = gp_scores + gamma * (scale or 1.0) * np.array(log_probs)
        ranked = top_candidates.copy()
        ranked["hybrid_score"] = hybrid_scores
        ranked["log_prob"] = log_probs
        ranked = ranked.sort_values("hybrid_score", ascending=False)
        suggestions = [
            row[self.space.feature_cols].to_dict()
            for _, row in ranked.head(n_candidates).iterrows()
        ]
        suggestions = knowledge.enrich_suggestions(suggestions)
        analysis = "\n".join(
            ["Log-probs Hybrid Selection Analysis:"]
            + [
                f"Selected Candidate {i}: GP Score={row['score']:.4f}, "
                f"LLM Log-prob={row['log_prob']:.4f}, "
                f"Hybrid Score={row['hybrid_score']:.4f}"
                for i, (_, row) in enumerate(
                    ranked.head(n_candidates).iterrows(), start=1
                )
            ]
        )
        return SuggestionResult(suggestions=suggestions, analysis=analysis, prompt=prompt)

    def _score_candidates(
        self, pool_df: pd.DataFrame, acquisition: str, kappa: float, xi: float
    ) -> pd.DataFrame:
        """Train GP and compute acquisition scores for the pool."""
        rng = np.random.RandomState(self.seed)
        if self.observed_configs.empty:
            result = pool_df.copy()
            result["score"] = rng.uniform(0, 1, size=len(pool_df))
            result["mean"] = 0.0
            result["std"] = 1.0
            return result

        try:
            from scipy.stats import norm

            x_train = self.observed_configs[self.space.feature_cols].values
            y_train = np.array(self.observed_scores)
            self._surrogate.fit(x_train, y_train)
            mu, sigma = self._surrogate.predict(
                pool_df[self.space.feature_cols].values
            )
            best_f = np.max(y_train)
            if acquisition == "ucb":
                scores = mu + kappa * sigma
            elif acquisition == "ei":
                improvement = mu - best_f - xi
                z = improvement / sigma
                scores = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            elif acquisition == "pi":
                improvement = mu - best_f - xi
                scores = norm.cdf(improvement / sigma)
            else:
                scores = mu

            result = pool_df.copy()
            result["score"] = scores
            result["mean"] = mu
            result["std"] = sigma
            return result
        except Exception as exc:
            print(f"[BO] GP scoring failed: {exc}. Falling back to random scores.")
            result = pool_df.copy()
            result["score"] = rng.uniform(0, 1, size=len(pool_df))
            result["mean"] = 0.0
            result["std"] = 1.0
            return result
