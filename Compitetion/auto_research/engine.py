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

import components.library  # noqa: E402,F401  # force component registration
from bo_core.benchmark.data_loader import (  # noqa: E402
    DATA_LOADERS,
    UNIFIED_DATASET_ROOT,
)
from bo_core.optimization.categorical import OneHotEncoder  # noqa: E402
from components.protocol import (  # noqa: E402
    ACQUISITIONS,
    LLM_STRATEGIES,
    SELECTORS,
    SURROGATES,
    Composition,
    StepContext,
)


class HybridEngine:
    """Runs one (composition, dataset, seed) BO loop over the categorical pool."""

    def __init__(
        self,
        composition: Composition,
        dataset: str,
        seed: int = 100,
        n_iters: int = 40,
        backend: str = "botorch",
    ) -> None:
        self.composition = composition
        self.dataset = dataset
        self.seed = seed
        self.n_iters = n_iters
        self.backend = backend
        self._load_data()
        self._build_components()

    def _load_data(self) -> None:
        data = DATA_LOADERS[self.dataset]()
        self.feature_cols = list(data["feature_cols"])
        self.target_col = str(data["target_col"])
        self.train_df = data["train_df"].reset_index(drop=True)
        self.test_df = data["test_df"]
        self.initial_indices = tuple(range(len(self.train_df)))

        opts_path = UNIFIED_DATASET_ROOT / "chemical_reactions" / self.dataset / "options.json"
        self.options = json.loads(opts_path.read_text())

        self.encoder = OneHotEncoder(
            self.feature_cols,
            {col: self.options[col] for col in self.feature_cols},
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
        # Prior = the complete fixed competition train.csv.
        X_obs = self.encoder.encode_df(self.train_df, allow_unknown=True)
        y_obs = self.train_df[self.target_col].to_numpy(dtype=float)
        queried: set[int] = set()
        trajectory: list[dict[str, Any]] = []
        self._iteration_diagnostics: list[dict[str, Any]] = []
        history: list[tuple[dict[str, str], float]] = [
            ({col: str(self.train_df[col].iloc[i]) for col in self.feature_cols},
             float(self.train_df[self.target_col].iloc[i]))
            for i in range(len(self.train_df))
        ]

        for it in range(self.n_iters):
            gp_fit = self._fit_surrogate(X_obs, y_obs)
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
                    "chat_engine": self.composition.params.get("chat_engine", "deepseek-v4-pro"),
                    "reasoning_effort": self.composition.params.get("reasoning_effort", "low"),
                    "xi": self.composition.params.get("xi", 0.01),
                    "kappa": self.composition.params.get("kappa", 2.576),
                    "plateau_window": self.composition.params.get("plateau_window", 5),
                    "uncertainty_threshold": self.composition.params.get("uncertainty_threshold", 0.5),
                    "pool_conditions": self.pool_conditions,
                    "prev_thinking": self.composition.params.get("prev_thinking"),
                },
            )

            llm_decision = self.llm_fn(ctx)
            llm_diagnostic = self._llm_diagnostic(ctx, llm_decision)
            acq_name = self.composition.acquisition
            if llm_decision and llm_decision.get("action") == "switch_acq":
                acq_name = llm_decision["acq_type"].lower().replace(" ", "_")
                if acq_name not in ACQUISITIONS:
                    acq_name = self.composition.acquisition

            acq_fn = ACQUISITIONS.get(acq_name, self.acq_fn)
            scores, acquisition = self._compute_acquisition(
                acq_fn, acq_name, best_f, ctx, y_obs
            )
            mean_shift = {"status": "not_requested", "error": None}
            if llm_decision and llm_decision.get("action") == "mean_shift":
                scores, mean_shift = self._apply_mean_shift(scores, llm_decision)

            first_unqueried = next((i for i in range(self.M) if i not in queried), None)
            crash_stage = "selection"
            try:
                if llm_decision and llm_decision.get("action") == "pool_pick":
                    idx = int(llm_decision["pool_index"])
                    if not 0 <= idx < self.M or idx in queried:
                        raise ValueError(f"LLM selected unavailable pool index {idx}")
                    selection_source = "llm_pool_pick"
                else:
                    idx = self.sel_fn(scores, ctx)
                    selection_source = self.composition.selector

                crash_stage = "evaluation"
                observed = float(self.pool_yield[idx])
                cond = self.pool_conditions[idx]
                crash_stage = "state_update"
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
            except Exception as exc:
                self._iteration_diagnostics.append({
                    "step": it + 1,
                    "status": "crashed",
                    "crash_stage": crash_stage,
                    "error": _format_error(exc),
                    "gp_fit": gp_fit,
                    "acquisition": acquisition,
                    "llm": llm_diagnostic,
                    "mean_shift": mean_shift,
                    "selection": {
                        "index": None,
                        "source": (
                            "llm_pool_pick"
                            if llm_decision and llm_decision.get("action") == "pool_pick"
                            else self.composition.selector
                        ),
                        "after_acquisition_fallback": acquisition["status"] == "fallback",
                        "is_first_unqueried": False,
                    },
                })
                raise
            self._iteration_diagnostics.append({
                "step": it + 1,
                "status": "completed",
                "crash_stage": None,
                "error": None,
                "gp_fit": gp_fit,
                "acquisition": acquisition,
                "llm": llm_diagnostic,
                "mean_shift": mean_shift,
                "selection": {
                    "index": int(idx),
                    "source": selection_source,
                    "after_acquisition_fallback": acquisition["status"] == "fallback",
                    "is_first_unqueried": idx == first_unqueried,
                },
            })

        return trajectory

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Build a snapshot, including completed iterations from failed runs."""
        return self._build_diagnostics(getattr(self, "_iteration_diagnostics", []))

    def _fit_surrogate(self, X_obs: np.ndarray, y_obs: np.ndarray) -> dict[str, Any]:
        diagnostic = {
            "status": "ok",
            "error": None,
            "n_observations": int(len(y_obs)),
        }
        try:
            self.surrogate.fit(X_obs, y_obs)
        except Exception as exc:
            diagnostic = {**diagnostic, "status": "failed", "error": _format_error(exc)}
            print(f"[HybridEngine] GP fit failed ({exc}); mean fallback.")
        else:
            jitter = getattr(self.surrogate, "_inference_jitter", None)
            if isinstance(jitter, (int, float, np.integer, np.floating)):
                diagnostic = {**diagnostic, "inference_jitter": float(jitter)}
        return diagnostic

    def _compute_acquisition(
        self,
        acq_fn: Any,
        acq_name: str,
        best_f: float,
        ctx: StepContext,
        y_obs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        error = None
        status = "ok"
        try:
            scores = np.asarray(acq_fn(self.surrogate, self.pool_X, best_f, ctx), dtype=float)
        except Exception as exc:
            error = _format_error(exc)
            status = "fallback"
            scores = np.full(self.M, float(np.mean(y_obs)))
        return scores, {
            "name": acq_name,
            "status": status,
            "error": error,
            "scores": _score_statistics(scores),
        }

    def _llm_diagnostic(
        self, ctx: StepContext, decision: dict[str, Any] | None
    ) -> dict[str, Any]:
        reported = ctx.extra.get("_llm_diagnostic")
        if isinstance(reported, dict):
            return {
                "strategy": self.composition.llm_strategy,
                **reported,
                "action": decision.get("action") if decision else None,
            }
        is_disabled = (
            self.composition.llm_strategy == "none"
            or not self.composition.params.get("use_llm", False)
        )
        return {
            "strategy": self.composition.llm_strategy,
            "attempted": False if is_disabled else None,
            "status": "disabled" if is_disabled else "diagnostic_missing",
            "error": None if is_disabled else "LLM strategy did not report call status",
            "action": decision.get("action") if decision else None,
        }

    def _apply_mean_shift(
        self, scores: np.ndarray, decision: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply simplified LGBO score shift and report whether it succeeded."""
        try:
            x_prop = self.encoder.encode_rows([decision["point"]])[0]
            d = len(self.feature_cols)
            hamming = d - self.pool_X @ x_prop
            k = min(20, self.M)
            grid_idx = np.argpartition(hamming, k - 1)[:k]
            confidence = float(decision.get("confidence", 0.5))
            boost = np.zeros(self.M)
            score_std = float(np.std(scores))
            if not np.isfinite(score_std):
                raise ValueError("acquisition score standard deviation is non-finite")
            boost[grid_idx] = confidence * 0.1 * score_std
            shifted = scores + boost
            return shifted, {
                "status": "applied",
                "error": None,
                "confidence": confidence,
                "boosted_candidates": int(k),
                "score_std_before": score_std,
                "score_std_after": float(np.std(shifted)),
            }
        except Exception as exc:
            return scores, {
                "status": "failed",
                "error": _format_error(exc),
                "confidence": None,
                "boosted_candidates": 0,
            }

    def _build_diagnostics(
        self, iterations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            surrogate_diagnostics = getattr(self.surrogate, "diagnostics", None)
        except Exception as exc:
            surrogate_diagnostics = {
                "status": "diagnostics_failed",
                "error": _format_error(exc),
            }
        surrogate_summary, surrogate_fits = _surrogate_health(surrogate_diagnostics)
        summary = {
            "iterations_completed": sum(
                d.get("status", "completed") == "completed" for d in iterations
            ),
            "iterations_recorded": len(iterations),
            "crashed_iterations": sum(d.get("status") == "crashed" for d in iterations),
            "gp_fit_failures": sum(d["gp_fit"]["status"] == "failed" for d in iterations),
            "acquisition_fallbacks": sum(
                d["acquisition"]["status"] == "fallback" for d in iterations
            ),
            "constant_score_iterations": sum(
                d["acquisition"]["scores"]["is_constant"] for d in iterations
            ),
            "degenerate_score_iterations": sum(
                d["acquisition"]["scores"]["is_degenerate"] for d in iterations
            ),
            "nonfinite_acquisition_scores": sum(
                d["acquisition"]["scores"]["nonfinite_count"] for d in iterations
            ),
            "llm_attempts": sum(d["llm"].get("attempted") is True for d in iterations),
            "llm_successes": sum(
                d["llm"].get("status") == "success" for d in iterations
            ),
            "llm_failures": sum(
                d["llm"].get("status") in {"failed", "error", "parse_failed"}
                for d in iterations
            ),
            "llm_not_configured": sum(
                d["llm"].get("status") == "not_configured" for d in iterations
            ),
            "llm_diagnostics_missing": sum(
                d["llm"].get("status") == "diagnostic_missing" for d in iterations
            ),
            "mean_shift_actions": sum(
                d["mean_shift"]["status"] != "not_requested" for d in iterations
            ),
            "mean_shift_failures": sum(
                d["mean_shift"]["status"] == "failed" for d in iterations
            ),
            "fallback_row_order_selections": sum(
                d["selection"]["after_acquisition_fallback"]
                and d["selection"]["is_first_unqueried"]
                and d["selection"]["source"] == "argmax"
                for d in iterations
            ),
            "row_order_selections_after_degenerate_scores": sum(
                d["acquisition"]["scores"]["is_degenerate"]
                and d["selection"]["is_first_unqueried"]
                and d["selection"]["source"] == "argmax"
                for d in iterations
            ),
            "surrogate_kernel_fit_failures": surrogate_summary.get(
                "kernel_fit_failures", 0
            ),
            "surrogate_degraded_fits": sum(
                bool(fit.get("failed_kernels")) for fit in surrogate_fits
            ),
            "surrogate_min_active_kernels": min(
                (len(fit.get("active_kernels", [])) for fit in surrogate_fits),
                default=None,
            ),
            "surrogate_llm_attempts": surrogate_summary.get("llm_attempts", 0),
            "surrogate_llm_successes": surrogate_summary.get("llm_successes", 0),
            "surrogate_llm_failures": surrogate_summary.get("llm_failures", 0),
        }
        return {
            "schema_version": 1,
            "composition": self.composition.name,
            "dataset": self.dataset,
            "seed": int(self.seed),
            "summary": summary,
            "iterations": [dict(d) for d in iterations],
            "surrogate": surrogate_diagnostics if isinstance(surrogate_diagnostics, dict) else None,
        }


def _surrogate_health(
    diagnostics: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(diagnostics, dict):
        return {}, []
    summary = diagnostics.get("summary")
    fits = diagnostics.get("fits")
    safe_summary = summary if isinstance(summary, dict) else {}
    safe_fits = [fit for fit in fits if isinstance(fit, dict)] if isinstance(fits, list) else []
    return safe_summary, safe_fits


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _score_statistics(scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    nonfinite_count = int(values.size - finite.size)
    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "std": None,
            "nonfinite_count": nonfinite_count,
            "is_constant": False,
            "is_degenerate": True,
        }
    score_std = float(np.std(finite))
    is_constant = bool(score_std <= 1e-12)
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": score_std,
        "nonfinite_count": nonfinite_count,
        "is_constant": is_constant,
        "is_degenerate": bool(is_constant or nonfinite_count > 0),
    }


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
