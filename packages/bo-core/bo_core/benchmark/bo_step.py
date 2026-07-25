from __future__ import annotations

from typing import Any, Literal

import numpy as np

from bo_core.optimization.knowledge import KnowledgeEngine
from bo_core.optimization.optimizer import BayesianOptimizer
from bo_core.optimization.space import DiscreteSearchSpace
from bo_core.optimization.surrogate import BackendName

AcquisitionType = Literal["ei", "ucb", "pi"]

class BOStepEngine:
    """Step-based Bayesian optimization over a fixed Excel dataset pool.
    
    Refactored to use the deep BayesianOptimizer module for locality and depth.
    """

    def __init__(
        self,
        method: Literal["traditional", "llmbo"],
        data: dict[str, Any],
        n_initial: int,
        n_trials: int,
        seed: int,
        acquisition: AcquisitionType = "ei",
        xi: float = 0.01,
        kappa: float = 2.576,
        chat_engine: str | None = None,
        n_candidates: int = 1,
        backend: BackendName = "botorch",
    ) -> None:
        self.method = method
        self.data = data
        self.n_initial = n_initial
        self.n_trials = n_trials
        self.seed = seed
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        self.n_candidates_per_step = n_candidates
        self.backend = backend
        
        self.feature_cols = list(data["feature_cols"])
        self.target_col = str(data["target_col"])
        self.df = data["df"]
        self.test_x = data["test_x"]
        self.test_y = data["test_y"]
        self.all_x = self.df[self.feature_cols].values.astype(float)
        self.all_y = self.df[self.target_col].values.astype(float)

        # Initialize deep modules
        space = DiscreteSearchSpace(self.df, self.feature_cols)
        
        knowledge = None
        if self.method == "llmbo" and chat_engine:
            knowledge = KnowledgeEngine(chat_engine=chat_engine)

        # LLM warm-start initializer (LLMBO only)
        self._llm_initializer = knowledge

        self.optimizer = BayesianOptimizer(
            space=space,
            target_name=self.target_col,
            knowledge_engine=knowledge,
            seed=self.seed,
            backend=self.backend,
        )
        
        self.iteration = 0
        self.completed = False
        self._initialize()

    def _initialize(self) -> None:
        """Seed with shared initial points.

        Traditional BO uses random initial points. LLMBO mirrors PVK-LLM's
        warm-start: sample a candidate pool, let the LLM pick the most
        promising ones on physical reasoning (eta hidden). Falls back to the
        same random draw if the LLM is unavailable.
        """
        if self.method == "llmbo" and self._llm_initializer is not None:
            configs = self._llm_select_initial_configs()
        else:
            configs = self._random_initial_configs()

        for config in configs:
            score = self._evaluate(config)[0]
            self.optimizer.observe(config, score)

    def _random_initial_configs(self) -> list[dict[str, float]]:
        rng = np.random.RandomState(self.seed)
        train_x = self.data["train_x"]
        n = min(self.n_initial, len(train_x))
        indices = rng.choice(len(train_x), n, replace=False)
        return [
            {col: float(train_x[idx, j]) for j, col in enumerate(self.feature_cols)}
            for idx in indices
        ]

    def _llm_select_initial_configs(self) -> list[dict[str, float]]:
        """Sample a pool from train_x (same split as Traditional BO), let the LLM choose.

        Using train_x (not the full df) ensures the pool contains feasible, non-zero-eta
        candidates, preventing the GP from fitting on constant y=0 which breaks EI/UCB.
        """
        rng = np.random.RandomState(self.seed)
        train_x = self.data["train_x"]
        n_pool = min(50, len(train_x))
        pool_idx = rng.choice(len(train_x), n_pool, replace=False)
        candidate_points = [
            {col: float(train_x[idx, j]) for j, col in enumerate(self.feature_cols)}
            for idx in pool_idx
        ]
        n = min(self.n_initial, len(candidate_points))
        try:
            selected = self._llm_initializer.select_initial_points(
                candidate_points, self.feature_cols, self.target_col, n
            )
        except Exception as e:
            print(f"[LLMBO] LLM initialization failed: {e}, using random init")
            return self._random_initial_configs()
        return [candidate_points[i] for i in selected[:n]]

    def _evaluate(self, config: dict[str, float]) -> tuple[float, float]:
        vec = np.array([[config[col] for col in self.feature_cols]])
        dists = np.sqrt(np.sum((self.all_x - vec) ** 2, axis=1))
        score = float(self.all_y[int(np.argmin(dists))])
        test_dists = np.sqrt(np.sum((self.test_x - vec) ** 2, axis=1))
        gen_score = float(self.test_y[int(np.argmin(test_dists))])
        return score, gen_score

    def step(self) -> dict[str, Any]:
        if self.completed:
            return self.snapshot(candidate_score=None)

        # Leverage the deep optimizer
        use_llm = (self.method == "llmbo")
        # For benchmark consistency with old code, top_k might be different
        res = self.optimizer.suggest(
            top_k=20,
            n_candidates=self.n_candidates_per_step,
            acquisition=self.acquisition,
            kappa=self.kappa,
            xi=self.xi,
            use_llm=use_llm,
        )

        if not res.suggestions:
            # Pool exhausted — nothing left to evaluate.
            self.completed = True
            return self.snapshot(candidate_score=None)

        chosen_config = res.suggestions[0]
        train_score, _gen = self._evaluate(chosen_config)
        
        self.optimizer.observe(chosen_config, train_score)
        self.iteration += 1
        if self.iteration >= self.n_trials:
            self.completed = True
        return self.snapshot(candidate_score=train_score)

    def snapshot(self, candidate_score: float | None) -> dict[str, Any]:
        obs_configs = self.optimizer.observed_configs
        obs_scores = self.optimizer.observed_scores
        
        best_idx = int(np.argmax(obs_scores))
        best_config = obs_configs.iloc[best_idx].to_dict()
        best_score = float(obs_scores[best_idx])
        _, gen_score = self._evaluate(best_config)
        
        return {
            "method": self.method,
            "iteration": self.iteration,
            "best_score": best_score,
            "generalization_score": gen_score,
            "candidate_score": candidate_score,
            "best_config": best_config,
            "completed": self.completed,
        }
