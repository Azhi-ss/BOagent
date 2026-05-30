from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

AcquisitionType = Literal["ei", "ucb", "pi"]


class BOStepEngine:
    """Step-based Bayesian optimization over a fixed Excel dataset pool.

    Both traditional BO and LLMBO share this engine. Each call to ``step()``
    advances exactly one iteration and returns a snapshot of progress, so a
    caller can stream per-iteration events.

    Traditional mode: GP surrogate + analytic acquisition (EI/UCB/PI) over the
    unobserved dataset pool; picks the single argmax point.

    LLMBO mode: GP pre-filters the top-k unobserved points by acquisition, an
    injected LLM acquisition selects among them, and the best is evaluated.
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
        llm_acq: Any | None = None,
    ) -> None:
        self.method = method
        self.data = data
        self.n_initial = n_initial
        self.n_trials = n_trials
        self.seed = seed
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        self.llm_acq = llm_acq

        self.feature_cols: list[str] = list(data["feature_cols"])
        self.target_col: str = str(data["target_col"])
        self.df: pd.DataFrame = data["df"]
        self.test_x: np.ndarray = data["test_x"]
        self.test_y: np.ndarray = data["test_y"]

        self.all_x = self.df[self.feature_cols].values.astype(float)
        self.all_y = self.df[self.target_col].values.astype(float)

        self.observed_configs = pd.DataFrame(columns=self.feature_cols)
        self.observed_scores: list[float] = []
        self.iteration = 0
        self.completed = False

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Seed the engine with shared initial points from the train pool."""
        rng = np.random.RandomState(self.seed)
        train_x = self.data["train_x"]
        n = min(self.n_initial, len(train_x))
        indices = rng.choice(len(train_x), n, replace=False)
        rows = []
        for idx in indices:
            config = {col: float(train_x[idx, j]) for j, col in enumerate(self.feature_cols)}
            rows.append(config)
            self.observed_scores.append(self._evaluate(config)[0])
        self.observed_configs = pd.DataFrame(rows, columns=self.feature_cols)

    # ------------------------------------------------------------------
    # Evaluation (Excel black-box: nearest-neighbour lookup)
    # ------------------------------------------------------------------

    def _evaluate(self, config: dict[str, float]) -> tuple[float, float]:
        """Return (train_score, generalization_score) via nearest-neighbour lookup."""
        vec = np.array([[config[col] for col in self.feature_cols]])
        dists = np.sqrt(np.sum((self.all_x - vec) ** 2, axis=1))
        score = float(self.all_y[int(np.argmin(dists))])

        test_dists = np.sqrt(np.sum((self.test_x - vec) ** 2, axis=1))
        gen_score = float(self.test_y[int(np.argmin(test_dists))])
        return score, gen_score

    # ------------------------------------------------------------------
    # Unobserved pool
    # ------------------------------------------------------------------

    def _unobserved(self) -> list[tuple[dict[str, float], int]]:
        observed = self.observed_configs[self.feature_cols].values.astype(float)
        pool: list[tuple[dict[str, float], int]] = []
        for i, row in enumerate(self.all_x):
            is_obs = any(np.allclose(row, o, rtol=1e-5) for o in observed)
            if not is_obs:
                config = {col: float(row[j]) for j, col in enumerate(self.feature_cols)}
                pool.append((config, i))
        return pool

    # ------------------------------------------------------------------
    # GP surrogate + acquisition
    # ------------------------------------------------------------------

    def _fit_gp(self) -> tuple[GaussianProcessRegressor, StandardScaler]:
        x = self.observed_configs[self.feature_cols].values.astype(float)
        y = np.array(self.observed_scores)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * len(self.feature_cols), (1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, normalize_y=True
        )
        gp.fit(x_scaled, y)
        return gp, scaler

    def _acquisition_scores(
        self, gp: GaussianProcessRegressor, scaler: StandardScaler,
        pool: list[tuple[dict[str, float], int]],
    ) -> np.ndarray:
        from scipy.stats import norm

        pool_x = np.array([[c[col] for col in self.feature_cols] for c, _ in pool])
        pool_scaled = scaler.transform(pool_x)
        mu, sigma = gp.predict(pool_scaled, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        best = max(self.observed_scores)

        if self.acquisition == "ucb":
            return mu + self.kappa * sigma
        if self.acquisition == "pi":
            z = (mu - best - self.xi) / sigma
            return norm.cdf(z)
        # default: expected improvement
        imp = mu - best - self.xi
        z = imp / sigma
        return imp * norm.cdf(z) + sigma * norm.pdf(z)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> dict[str, Any]:
        """Advance one BO iteration. Returns a progress snapshot."""
        if self.completed:
            return self.snapshot(candidate_score=None)

        pool = self._unobserved()
        if not pool:
            self.completed = True
            return self.snapshot(candidate_score=None)

        gp, scaler = self._fit_gp()
        acq = self._acquisition_scores(gp, scaler, pool)

        if self.method == "traditional" or self.llm_acq is None:
            chosen_config = pool[int(np.argmax(acq))][0]
        else:
            chosen_config = self._llmbo_select(pool, acq)

        train_score, _gen = self._evaluate(chosen_config)
        new_row = pd.DataFrame([chosen_config], columns=self.feature_cols)
        self.observed_configs = pd.concat(
            [self.observed_configs, new_row], ignore_index=True
        )
        self.observed_scores.append(train_score)
        self.iteration += 1
        if self.iteration >= self.n_trials:
            self.completed = True
        return self.snapshot(candidate_score=train_score)

    def _llmbo_select(
        self, pool: list[tuple[dict[str, float], int]], acq: np.ndarray
    ) -> dict[str, float]:
        """Top-k by acquisition, then let the injected LLM acquisition choose."""
        order = np.argsort(acq)[::-1]
        top_k = getattr(self.llm_acq, "top_k", 20)
        top = [pool[i][0] for i in order[:top_k]]
        try:
            obs_scores = pd.DataFrame({"score": self.observed_scores})
            candidates, _cost, _t = self.llm_acq.get_candidate_points(
                self.observed_configs, obs_scores
            )
            if candidates is not None and len(candidates) > 0:
                return {col: float(candidates.iloc[0][col]) for col in self.feature_cols}
        except Exception:
            pass
        return top[0]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, candidate_score: float | None) -> dict[str, Any]:
        best_idx = int(np.argmax(self.observed_scores))
        best_config = {
            col: float(self.observed_configs.iloc[best_idx][col])
            for col in self.feature_cols
        }
        best_score = float(self.observed_scores[best_idx])
        _train, gen_score = self._evaluate(best_config)
        return {
            "method": self.method,
            "iteration": self.iteration,
            "best_score": best_score,
            "generalization_score": gen_score,
            "candidate_score": candidate_score,
            "best_config": best_config,
            "completed": self.completed,
        }
