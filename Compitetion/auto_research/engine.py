"""HybridEngine: composes components into a runnable BO loop."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure competition code is importable
CODE_ROOT = Path(__file__).resolve().parents[2] / "submission" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from bo_core.benchmark.data_loader import DATA_LOADERS, UNIFIED_DATASET_ROOT  # noqa: E402
from bo_core.optimization.categorical import OneHotEncoder, union_options  # noqa: E402

from components.protocol import (  # noqa: E402
    ACQUISITIONS,
    LLM_STRATEGIES,
    SELECTORS,
    SURROGATES,
    Composition,
    StepContext,
)
import components.library  # noqa: E402,F401  # force component registration


DEFAULT_N_INITIAL = 5


class HybridEngine:
    """Runs one (composition, dataset, seed) BO loop over the categorical pool."""

    def __init__(
        self,
        composition: Composition,
        dataset: str,
        seed: int = 100,
        n_iters: int = 40,
        backend: str = "botorch",
        n_initial: int | None = DEFAULT_N_INITIAL,
    ) -> None:
        self.composition = composition
        self.dataset = dataset
        self.seed = seed
        self.n_iters = n_iters
        self.backend = backend
        self.n_initial = n_initial
        self._load_data()
        self._build_components()

    def _load_data(self) -> None:
        data = DATA_LOADERS[self.dataset]()
        self.feature_cols = list(data["feature_cols"])
        self.target_col = str(data["target_col"])
        full_train_df = data["train_df"]
        self.test_df = data["test_df"]

        if self.n_initial is None:
            self.initial_indices = tuple(range(len(full_train_df)))
        else:
            if self.n_initial <= 0:
                raise ValueError("n_initial must be positive or None")
            n = min(self.n_initial, len(full_train_df))
            rng = np.random.RandomState(self.seed)
            self.initial_indices = tuple(int(i) for i in rng.choice(len(full_train_df), n, replace=False))
        self.train_df = full_train_df.iloc[list(self.initial_indices)].reset_index(drop=True)

        opts_path = UNIFIED_DATASET_ROOT / "chemical_reactions" / self.dataset / "options.json"
        self.options = json.loads(opts_path.read_text())

        self.encoder = OneHotEncoder(
            self.feature_cols,
            union_options(self.feature_cols, full_train_df, self.test_df),
        )
        self.pool_X = self.encoder.encode_df(self.test_df)
        self.pool_yield = self.test_df[self.target_col].to_numpy(dtype=float)
        self.M = len(self.test_df)
        self.pool_conditions = [
            {col: str(self.test_df[col].iloc[i]) for col in self.feature_cols}
            for i in range(self.M)
        ]

    def _build_components(self) -> None:
        params = self.composition.params
        # Surrogate
        sur_factory = SURROGATES[self.composition.surrogate]
        self.surrogate = sur_factory(
            self.backend,
            self.seed,
            n_restarts=params.get("n_restarts", 10),
            alpha=params.get("alpha", 1e-2),
        )
        # Acquisition / Selector / LLM
        self.acq_fn = ACQUISITIONS[self.composition.acquisition]
        self.sel_fn = SELECTORS[self.composition.selector]
        self.llm_fn = LLM_STRATEGIES[self.composition.llm_strategy]

    def run(self) -> list[dict[str, Any]]:
        # Prior = full train.csv
        X_obs = self.encoder.encode_df(self.train_df)
        y_obs = self.train_df[self.target_col].to_numpy(dtype=float)
        queried: set[int] = set()
        trajectory: list[dict[str, Any]] = []
        history: list[tuple[dict[str, str], float]] = [
            ({col: str(self.train_df[col].iloc[i]) for col in self.feature_cols},
             float(self.train_df[self.target_col].iloc[i]))
            for i in range(len(self.train_df))
        ]

        for it in range(self.n_iters):
            # Fit surrogate
            try:
                self.surrogate.fit(X_obs, y_obs)
            except Exception as exc:
                print(f"[HybridEngine] GP fit failed ({exc}); mean fallback.")

            best_f = float(np.max(y_obs))
            ctx = StepContext(
                iteration=it,
                n_iters=self.n_iters,
                feature_cols=self.feature_cols,
                options=self.options,
                history=history,
                queried=queried,
                best_f=best_f,
                remaining=self.n_iters - it,
                extra={
                    "dataset": self.dataset,
                    "target_col": self.target_col,
                    "seed": self.seed,
                    "use_llm": self.composition.params.get("use_llm", False),
                    "chat_engine": self.composition.params.get("chat_engine", "deepseek-v4-flash"),
                    "reasoning_effort": self.composition.params.get("reasoning_effort", "low"),
                    "xi": self.composition.params.get("xi", 0.01),
                    "kappa": self.composition.params.get("kappa", 2.576),
                    "plateau_window": self.composition.params.get("plateau_window", 5),
                    "uncertainty_threshold": self.composition.params.get("uncertainty_threshold", 0.5),
                    "pool_conditions": self.pool_conditions,
                    "prev_thinking": self.composition.params.get("prev_thinking"),
                },
            )

            # LLM decision (may override acquisition or pick directly)
            llm_decision = self.llm_fn(ctx)
            acq_name = self.composition.acquisition
            if llm_decision and llm_decision.get("action") == "switch_acq":
                acq_name = llm_decision["acq_type"].lower().replace(" ", "_")
                if acq_name not in ACQUISITIONS:
                    acq_name = self.composition.acquisition

            # Compute scores
            acq_fn = ACQUISITIONS.get(acq_name, self.acq_fn)
            try:
                scores = acq_fn(self.surrogate, self.pool_X, best_f, ctx)
            except Exception:
                mu = np.full(self.M, float(np.mean(y_obs)))
                scores = mu

            # Apply mean-shift if LGBO decision
            if llm_decision and llm_decision.get("action") == "mean_shift":
                scores = self._apply_mean_shift(scores, llm_decision, ctx)

            # Select
            if llm_decision and llm_decision.get("action") == "pool_pick":
                idx = int(llm_decision["pool_index"])
            else:
                idx = self.sel_fn(scores, ctx)

            # Evaluate oracle
            observed = float(self.pool_yield[idx])
            cond = self.pool_conditions[idx]

            # Update state
            X_obs = np.vstack([X_obs, self.pool_X[idx : idx + 1]])
            y_obs = np.append(y_obs, observed)
            queried.add(idx)
            history.append((cond, observed))
            trajectory.append({
                "step": it + 1,
                "query_index": idx,
                "condition": cond,
                "observed_yield": observed,
                "acquisition": acq_name,
                "llm_action": llm_decision.get("action") if llm_decision else None,
            })

        return trajectory

    def _apply_mean_shift(
        self, scores: np.ndarray, decision: dict[str, Any], ctx: StepContext
    ) -> np.ndarray:
        """Simplified LGBO mean-shift on scores."""
        try:
            x_prop = self.encoder.encode_rows([decision["point"]])[0]
            d = len(self.feature_cols)
            hamming = d - self.pool_X @ x_prop
            K = min(20, self.M)
            grid_idx = np.argpartition(hamming, K - 1)[:K]
            confidence = float(decision.get("confidence", 0.5))
            # Boost scores near the LLM proposal
            boost = np.zeros(self.M)
            boost[grid_idx] = confidence * 0.1 * np.std(scores)
            return scores + boost
        except Exception:
            return scores


def compute_metrics(trajectory: list[dict[str, Any]], global_best: float) -> dict[str, float]:
    yields = [t["observed_yield"] for t in trajectory]
    if not yields:
        return {"best_found": float("-inf"), "initial_round_found_best": float("-inf"),
                "t95": float("inf"), "AUC_best_so_far": 0.0}
    best_so_far = []
    cur = float("-inf")
    for y in yields:
        cur = max(cur, y)
        best_so_far.append(cur)
    target = 0.95 * global_best
    t95 = next((i for i, b in enumerate(best_so_far, 1) if b >= target), len(yields) + 1)
    return {
        "best_found": float(best_so_far[-1]),
        "initial_round_found_best": float(best_so_far[0]),
        "t95": int(t95),
        "AUC_best_so_far": float(np.mean(best_so_far)),
    }
