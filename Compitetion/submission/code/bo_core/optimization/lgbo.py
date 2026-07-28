"""LGBO engine: LLM-Guided Bayesian Optimization for fully-categorical datasets.

Implements the Region-Lifted Preference mechanism from arxiv 2605.17976v1
(Proposition 1: LLM point+confidence -> GP mean shift -> acquisition on the
shifted surrogate), adapted to the fully-categorical Buchwald/Suzuki datasets
(one-hot encoding, point-mode only). The existing ``llmbo`` log-prob path in
``BayesianOptimizer`` is left untouched; this is a parallel engine.

Per-iteration (R2):
  1. Fit GP (Matern-5/2 on one-hot observed data).
  2. Predict mean/std over the candidate pool (``test_features`` == ``test.csv``
     minus Yield; row-aligned, so ``query_index`` indexes both).
  3. (LGBO only) query LLM, parse point+confidence, mean-shift the GP mean.
  4. EI on the (shifted) mean; argmax over unobserved pool.
  5. Evaluate via ``test.csv`` oracle; observe; record trajectory.

GPBO = ``use_llm=False`` (lambda=0, no shift) - the pure-GP baseline that shares
identical prior/encoding/GP/acquisition with LGBO, isolating the LLM's effect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from bo_core.benchmark.data_loader import DATA_LOADERS, UNIFIED_DATASET_ROOT
from bo_core.optimization.categorical import OneHotEncoder
from bo_core.optimization.lgbo_parser import parse_llm_response
from bo_core.optimization.lgbo_prompt import (
    DatasetMeta,
    build_system_prompt,
    build_user_prompt,
)
from bo_core.optimization.surrogate import (
    BackendName,
    SurrogateModel,
    create_surrogate,
)

# Seeds fixed by the competition README.
COMPETITION_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                     1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]


class LGBOEngine:
    """Faithful LGBO (point-mode mean-shift) over a categorical candidate pool."""

    def __init__(
        self,
        dataset: str,
        seed: int = 100,
        use_llm: bool = False,
        n_iters: int = 40,
        K: int = 50,
        n_restarts: int = 10,
        alpha: float = 1e-2,
        xi: float = 0.01,
        chat_engine: str = "deepseek-v4-flash",
        llm_max_tokens: int = 8192,
        reasoning_effort: str = "low",
        failure_log: str | Path | None = "lgbo_llm_failures.log",
        backend: BackendName = "botorch",
    ) -> None:
        if dataset not in DATA_LOADERS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATA_LOADERS)}")
        self.dataset = dataset
        self.seed = seed
        self.use_llm = use_llm
        self.n_iters = n_iters
        self.K = K
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.xi = xi
        self.chat_engine = chat_engine
        self.llm_max_tokens = llm_max_tokens
        self.reasoning_effort = reasoning_effort
        self.failure_log = Path(failure_log) if failure_log else None
        self.backend = backend
        self._surrogate = create_surrogate(
            backend,
            seed=seed,
            n_restarts=n_restarts,
            alpha=alpha,
            jitter_levels=(alpha, alpha * 10.0, 1.0),
        )

        self._load_data()
        self._init_state()

    # ------------------------------------------------------------------ setup

    def _load_data(self) -> None:
        data = DATA_LOADERS[self.dataset]()
        self.feature_cols: list[str] = list(data["feature_cols"])
        self.target_col: str = str(data["target_col"])
        self.train_df = data["train_df"]            # prior (full train.csv)
        self.test_df = data["test_df"]              # pool + oracle (row-aligned)

        # options.json = the dataset's own valid option space (for the LLM prompt).
        opts_path = UNIFIED_DATASET_ROOT / "chemical_reactions" / self.dataset / "options.json"
        self.options_json: dict[str, list[str]] = json.loads(opts_path.read_text())

        # The BO input schema is the task's four decision variables only. The
        # merged prior may contain cross-product reactants, but those must not
        # create dimensions that are permanently zero in the candidate pool.
        self.encoder = OneHotEncoder(
            self.feature_cols,
            {col: self.options_json[col] for col in self.feature_cols},
        )

        # Candidate pool (== test_features) and its yield oracle (test.csv).
        self.pool_X = self.encoder.encode_df(self.test_df)             # (M, D)
        self.pool_yield = self.test_df[self.target_col].to_numpy(dtype=float)
        self.M = len(self.test_df)

        self.meta = DatasetMeta(
            dataset=self.dataset,
            feature_cols=self.feature_cols,
            options=self.options_json,
            target_name=self.target_col,
        )

    def _init_state(self) -> None:
        # Training-only categories are represented by an all-zero feature block;
        # candidate-pool and LLM encodes remain strict.
        self.X_obs = self.encoder.encode_df(
            self.train_df,
            allow_unknown=True,
        )  # (N, D)
        self.y_obs = self.train_df[self.target_col].to_numpy(dtype=float)
        self.queried: set[int] = set()                                  # pool indices evaluated
        self.trajectory: list[dict[str, Any]] = []
        self.prev_thinking: str | None = None
        self.iteration = 0
        # LLM client (lazy, only for LGBO). Reuses DeepSeekClient.from_env().
        self._client = None
        if self.use_llm:
            from bo_core.llm_client import DeepSeekClient
            self._client = DeepSeekClient.from_env()
            self._client.model = self.chat_engine
            # Reasoning model responses can take >30s; widen the HTTP timeout.
            self._client.timeout_s = 120

    # ----------------------------------------------------------------- GP + EI

    def _fit_gp(self) -> SurrogateModel:
        """Fit the configured Matern-5/2 surrogate on observed one-hot data."""
        try:
            self._surrogate.fit(self.X_obs, self.y_obs)
        except Exception as exc:  # noqa: BLE001 - step has a mean fallback
            print(f"[LGBO] GP fit failed ({exc}); using mean predictor.")
        return self._surrogate

    def _predict_pool(
        self, surrogate: SurrogateModel
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            return surrogate.predict(self.pool_X)
        except Exception:
            # Non-fitted GP fallback: constant mean = observed mean, unit std.
            return (
                np.full(self.M, float(np.mean(self.y_obs))),
                np.ones(self.M),
            )

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, best_f: float) -> np.ndarray:
        imp = mu - best_f - self.xi
        z = imp / sigma
        return imp * norm.cdf(z) + sigma * norm.pdf(z)

    # ------------------------------------------------------------- mean shift

    def _mean_shift(
        self,
        surrogate: SurrogateModel,
        mu: np.ndarray,
        x_proposed: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        """Region-lifted mean shift (Proposition 1).

        mu_lambda(x) = mu(x) + lambda * sum_g a_g k_post(x, x_g), with
        lambda = c / sqrt(a' Sigma_GG a). Grid G = K nearest pool candidates to
        the proposed point (Hamming); weights a_g = k(x_p, x_g) normalized.
        ``k_post`` is the GP posterior covariance (not the prior kernel), per the
        paper's Proposition 1 lift on the posterior GP. Covariance is unchanged
        (mean-only shift). Returns mu unchanged if anything fails (lambda=0).

        Complexity is O(M * n_train * K): the cross-covariance K_post(pool, grid)
        is computed directly via the fitted kernel + Cholesky, avoiding the
        O(M^2) full-pool posterior covariance that would be prohibitive for suzuki.
        """

        try:
            if not surrogate.is_fit:
                return mu
            x_p = np.asarray(x_proposed, dtype=float).reshape(1, -1)

            # Grid G = K nearest pool points to x_p by Hamming distance
            # (number of mismatched categories = d - pool . x_p on one-hot).
            d = len(self.feature_cols)
            hamming = d - self.pool_X @ x_proposed              # (M,)
            K = min(self.K, self.M)
            grid_idx = np.argpartition(hamming, K - 1)[:K]
            X_grid = self.pool_X[grid_idx]                       # (K, D)

            # Weights a_g = k(x_p, x_g), clipped non-negative, normalized.
            a = np.maximum(
                surrogate.prior_cross_covariance(x_p, X_grid).ravel(),
                0.0,
            )
            if a.sum() <= 0:
                return mu
            a = a / a.sum()

            # Sigma_GG = posterior covariance on the grid (K x K).
            Sigma_GG = surrogate.posterior_covariance(X_grid)
            denom = float(a @ Sigma_GG @ a)
            if not np.isfinite(denom) or denom <= 0:
                return mu
            lam = float(confidence) / float(np.sqrt(denom))

            # Shift = lam * K_post(pool, grid) @ a  (M,).
            K_post_pool_grid = surrogate.posterior_cross_covariance(
                self.pool_X, X_grid
            )
            return mu + lam * (K_post_pool_grid @ a)
        except Exception as exc:  # noqa: BLE001 - mean-shift is best-effort
            print(f"[LGBO] mean-shift failed ({exc}); using pure GP this iter.")
            return mu

    # -------------------------------------------------------------------- loop

    def step(self) -> dict[str, Any]:
        surrogate = self._fit_gp()
        mu, sigma = self._predict_pool(surrogate)
        best_f = float(np.max(self.y_obs))

        thinking: str | None = None
        if self.use_llm:
            mu, thinking = self._llm_mean_shift(surrogate, mu)

        ei = self._expected_improvement(mu, sigma, best_f)
        # Mask already-queried pool points.
        mask = np.ones(self.M, dtype=bool)
        for q in self.queried:
            if 0 <= q < self.M:
                mask[q] = False
        ei = np.where(mask, ei, -np.inf)

        idx = int(np.argmax(ei))
        observed_yield = float(self.pool_yield[idx])
        condition = {col: str(self.test_df[col].iloc[idx]) for col in self.feature_cols}

        self.trajectory.append({
            "step": self.iteration + 1,
            "query_index": idx,
            "condition": condition,
            "observed_yield": observed_yield,
            "predicted_yield": float(mu[idx]),
        })
        if thinking:
            self.prev_thinking = thinking

        # Observe the selected pool point.
        self.X_obs = np.vstack([self.X_obs, self.pool_X[idx:idx + 1]])
        self.y_obs = np.append(self.y_obs, observed_yield)
        self.queried.add(idx)
        self.iteration += 1
        return self.trajectory[-1]

    def _llm_mean_shift(
        self, surrogate: SurrogateModel, mu: np.ndarray
    ) -> tuple[np.ndarray, str | None]:
        """Query the LLM, parse point+confidence, apply mean shift.

        Returns (shifted_mu, thinking_text). On any LLM/parse failure, returns
        (mu, None) so the iteration falls back to a pure-GP step (lambda=0).
        """
        if self._client is None or not self._client.is_configured():
            return mu, None

        # History = prior (train.csv) + queried trajectory, newest last.
        prior_hist = [
            ({col: str(self.train_df[col].iloc[i]) for col in self.feature_cols},
             float(self.train_df[self.target_col].iloc[i]))
            for i in range(len(self.train_df))
        ]
        traj_hist = [(t["condition"], t["observed_yield"]) for t in self.trajectory]
        history = prior_hist + traj_hist

        system_prompt = build_system_prompt(self.meta)
        user_prompt = build_user_prompt(self.meta, history, self.prev_thinking)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = self._client.chat(
                messages,
                max_tokens=self.llm_max_tokens,
                extra_body={"reasoning_effort": self.reasoning_effort},
            )
        except Exception as exc:  # noqa: BLE001 - LLM is best-effort
            print(f"[LGBO] LLM call raised ({exc}); pure GP this iter.")
            return mu, None

        if getattr(result, "status", None) != "success" or not result.content:
            reason = "EMPTY_CONTENT" if getattr(result, "status", None) == "success" else "STATUS_ERROR"
            print(f"[LGBO] LLM {reason}: status={getattr(result,'status',None)} "
                  f"error={getattr(result,'error',None)}; pure GP this iter.")
            self._log_failure(reason, result)
            return mu, None

        parsed = parse_llm_response(result.content, self.feature_cols, self.options_json)
        if parsed is None:
            print("[LGBO] LLM output unparseable or invalid; pure GP this iter.")
            self._log_failure("PARSE_FAIL", result)
            return mu, None

        _mode, values, confidence = parsed
        proposed_cond = dict(zip(self.feature_cols, values))
        try:
            x_proposed = self.encoder.encode_rows([proposed_cond])[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[LGBO] proposed point encode failed ({exc}); pure GP this iter.")
            return mu, None

        thinking = self._extract_thinking(result.content)
        mu_shifted = self._mean_shift(
            surrogate, mu, x_proposed, confidence
        )
        return mu_shifted, thinking

    def _log_failure(self, reason: str, result: Any) -> None:
        """Append failed LLM responses to failure_log for post-hoc diagnosis."""
        if not self.failure_log:
            return
        try:
            with open(self.failure_log, "a", encoding="utf-8") as f:
                f.write(f"=== dataset={self.dataset} seed={self.seed} iter={self.iteration} reason={reason} ===\n")
                f.write(f"status={getattr(result, 'status', None)} error={getattr(result, 'error', None)}\n")
                usage = getattr(result, "usage", None) or {}
                f.write(f"usage={usage}\n")
                f.write(f"--- content (len={len(getattr(result, 'content', '') or '')}) ---\n")
                f.write((getattr(result, "content", None) or "<EMPTY>") + "\n\n")
        except OSError as exc:
            print(f"[LGBO] failure_log write failed: {exc}")

    @staticmethod
    def _extract_thinking(text: str) -> str | None:
        """Capture the Thinking block (text before the Final Answer JSON line)."""
        lines = text.splitlines()
        fa_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].lstrip().startswith("{"):
                fa_idx = i
                break
        if fa_idx is None:
            return text.strip()[:800] or None
        thinking = "\n".join(lines[:fa_idx]).strip()
        return thinking[:800] or None

    def run(self) -> list[dict[str, Any]]:
        for _ in range(self.n_iters):
            self.step()
        return self.trajectory

    # --------------------------------------------------------------- metrics

    def best_found(self) -> float:
        """Max observed yield discovered within the optimization budget."""
        if not self.trajectory:
            return float("-inf")
        return float(max(t["observed_yield"] for t in self.trajectory))
