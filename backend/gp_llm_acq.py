from __future__ import annotations

import os
import time
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from llm_client import DeepSeekClient, LlmCallResult


class GPLLM_ACQ:
    """GP pre-filtering + LLM selection two-stage acquisition function.

    Aligns with PVK-LLM benchmark CustomLLM_ACQ logic:
        1. Train sklearn GP on observed (config, score) pairs
        2. UCB-score all unobserved points in the full dataset
        3. Send top-k candidates to LLM for materials-science reasoning
        4. LLM returns refined n_candidates selections
        5. Fall back to GP UCB ranking if LLM fails
    """

    def __init__(
        self,
        task_context: dict[str, Any],
        n_candidates: int,
        n_templates: int,
        lower_is_better: bool,
        chat_engine: str,
        top_k: int = 20,
        alpha: float = 0.1,
    ) -> None:
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.lower_is_better = lower_is_better
        self.chat_engine = chat_engine
        self.top_k = top_k
        self.alpha = alpha
        self.feature_cols: list[str] = list(task_context["feature_cols"])
        self.target_col: str = str(task_context["target_col"])
        self.df: pd.DataFrame = task_context["df"]
        self.hyperparameter_constraints: dict = task_context.get(
            "hyperparameter_constraints", {}
        )
        self._llm_client: DeepSeekClient | None = None

    def _get_llm_client(self) -> DeepSeekClient:
        if self._llm_client is None:
            self._llm_client = DeepSeekClient.from_env()
            if self.chat_engine:
                self._llm_client.model = self.chat_engine
        return self._llm_client

    # ------------------------------------------------------------------
    # Public interface (matches PVK-LLM LLM_ACQ signature)
    # ------------------------------------------------------------------

    def get_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[pd.DataFrame, float, float]:
        """Generate candidate points via GP pre-filter + LLM refinement.

        Returns:
            (candidate_points_df, cost, time_taken)
        """
        start_time = time.time()
        alpha = alpha if alpha is not None else self.alpha

        # 1. Build GP training data from observed points
        X_train, y_train = self._build_training_data(observed_configs, observed_fvals)

        # 2. Find unobserved points in full dataset
        unobserved = self._find_unobserved(observed_configs)

        # 3. Handle edge case: no unobserved points left
        if not unobserved:
            return self._fallback_random_candidates(observed_configs), 0.0, time.time() - start_time

        # 4. Train GP and compute UCB scores
        try:
            gp_predictions = self._gp_ucb_predict(X_train, y_train, unobserved, alpha)
        except Exception:
            # GP training failed (e.g. too few points) — random sample
            return self._fallback_random_from(unobserved), 0.0, time.time() - start_time

        # 5. Take top-k by UCB
        gp_predictions.sort(key=lambda x: x[3], reverse=True)
        top_formulas = gp_predictions[: self.top_k]

        # 6. LLM batch evaluation
        observed_data = self._build_observed_data(observed_configs, observed_fvals)

        try:
            prompt = self._build_batch_evaluation_prompt(top_formulas, observed_data)
            response_text = self._call_llm(prompt)
            selected_indices = self._parse_llm_response(response_text, len(top_formulas))
            selected = [top_formulas[i] for i in selected_indices[: self.n_candidates]]
        except Exception:
            # LLM failed — fall back to GP UCB top-n_candidates
            top_formulas.sort(key=lambda x: x[3], reverse=True)
            selected = top_formulas[: self.n_candidates]

        # 7. Build result DataFrame
        candidate_points = pd.DataFrame()
        for formula_dict, _idx, _actual, _ucb, _mean, _std in selected:
            candidate_points = pd.concat(
                [candidate_points, pd.DataFrame([formula_dict])],
                ignore_index=True,
            )

        end_time = time.time()
        return candidate_points, 0.0, end_time - start_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_train = []
        y_train = []
        for i, (_, obs_config) in enumerate(observed_configs.iterrows()):
            features = [obs_config[col] for col in self.feature_cols]
            X_train.append(features)
            y_train.append(observed_fvals.iloc[i]["score"])
        return np.array(X_train), np.array(y_train)

    def _find_unobserved(
        self, observed_configs: pd.DataFrame
    ) -> list[tuple[dict[str, float], int, float]]:
        unobserved: list[tuple[dict[str, float], int, float]] = []
        for i, row in self.df.iterrows():
            formula = row[self.feature_cols].values.astype(float)
            is_observed = False
            for _, obs in observed_configs.iterrows():
                obs_values = np.array(
                    [float(obs[col]) for col in self.feature_cols]
                )
                if np.allclose(formula, obs_values, rtol=1e-5):
                    is_observed = True
                    break
            if not is_observed:
                formula_dict = {
                    col: float(formula[j]) for j, col in enumerate(self.feature_cols)
                }
                unobserved.append(
                    (formula_dict, i, float(row[self.target_col]))
                )
        return unobserved

    def _gp_ucb_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        unobserved: list[tuple[dict[str, float], int, float]],
        alpha: float,
    ) -> list[tuple[dict[str, float], int, float, float, float, float]]:
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)

        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            [1.0] * len(self.feature_cols), (1e-2, 1e2)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
        )
        gp.fit(X_train_scaled, y_train)

        predictions: list[
            tuple[dict[str, float], int, float, float, float, float]
        ] = []
        for formula_dict, idx, actual_val in unobserved:
            features = np.array(
                [[formula_dict[col] for col in self.feature_cols]]
            )
            features_scaled = scaler_X.transform(features)
            pred_mean, pred_std = gp.predict(features_scaled, return_std=True)
            ucb_score = float(pred_mean[0] + alpha * pred_std[0])
            predictions.append(
                (
                    formula_dict,
                    idx,
                    actual_val,
                    ucb_score,
                    float(pred_mean[0]),
                    float(pred_std[0]),
                )
            )
        return predictions

    def _build_observed_data(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
    ) -> list[tuple[dict[str, float], float]]:
        observed_data: list[tuple[dict[str, float], float]] = []
        for i, (_, obs_config) in enumerate(observed_configs.iterrows()):
            obs_values = {
                col: float(obs_config[col]) for col in self.feature_cols
            }
            obs_val = float(observed_fvals.iloc[i]["score"])
            observed_data.append((obs_values, obs_val))
        return observed_data

    def _build_batch_evaluation_prompt(
        self,
        top_formulas: list[
            tuple[dict[str, float], int, float, float, float, float]
        ],
        observed_data: list[tuple[dict[str, float], float]],
    ) -> str:
        """Build materials-science domain prompt for LLM batch evaluation."""
        model_name = self.task_context.get("model", "perovskite")
        target_name = self.target_col

        prompt = f"""
You are a professional materials scientist specializing in perovskite solar cell optimization. I will provide you with multiple candidate formulations, some observed formulation data, and the prediction results from a Gaussian process model. Please select the formulation from these that is most likely to produce high power conversion efficiency ({target_name}).

Known Information:

The efficiency ({target_name}) of perovskite solar cells is closely related to the material characteristics.

Key parameters include:
"""
        for col in self.feature_cols:
            prompt += f"\n{col}: Feature parameter for {model_name} optimization"

        prompt += "\n\nObserved formulations and their efficiencies:"

        for i, (obs_values, obs_eta) in enumerate(observed_data):
            prompt += f"\n formulation{i + 1}: "
            for col in self.feature_cols:
                prompt += f"{col}={obs_values[col]:.4f}, "
            prompt += f"{target_name}={obs_eta:.4f}"

        prompt += "\n\nCandidate formulations:"

        for i, (
            formula_dict,
            _idx,
            _actual,
            _ucb,
            pred_mean,
            pred_std,
        ) in enumerate(top_formulas):
            prompt += f"\n\nCandidate formulation {i + 1}:\n"
            for col in self.feature_cols:
                prompt += f"{col}={formula_dict[col]:.4f}, "
            prompt += f"\nGP Prediction: mean={pred_mean:.4f}, std={pred_std:.4f}"

        prompt += f"""

Based on materials science principles, the observed data, and the GP model predictions, please select the {self.n_candidates} formulations from the candidates above that are most likely to produce high {target_name}. Considerations should include:
1. Similarity to known high-efficiency formulations
2. The likely efficiency of charge separation and transport
3. The potential for recombination losses
4. Novelty or innovation compared to observed formulations

Please respond in the following format:
Analysis: [Your detailed analysis]
Selected Formulations: [List of formulation numbers, e.g., 1, 3, 5, 7, 9]
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via DeepSeekClient for batch evaluation."""
        client = self._get_llm_client()

        if not client.is_configured():
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not configured; cannot call LLM for batch evaluation"
            )

        result: LlmCallResult = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional materials scientist specializing in "
                        "perovskite solar cell optimization. Your task is to select "
                        "formulations from the candidates that are most likely to "
                        "produce high power conversion efficiency."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )

        if result.status != "success":
            raise RuntimeError(
                f"LLM call failed: {result.error or result.status}"
            )

        return result.content

    def _parse_llm_response(
        self, response: str, num_formulas: int
    ) -> list[int]:
        """Parse LLM response to extract selected formulation indices (0-indexed)."""
        # Try "Selected Formulations:" pattern first
        patterns = [
            r"Selected\s+Formulations?\s*:\s*(.+)",
            r"Selected\s+formulations?\s*:\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                numbers_text = match.group(1).strip()
                indices = [
                    int(x.strip()) - 1
                    for x in re.split(r"[,\s]+", numbers_text)
                    if x.strip().isdigit()
                ]
                if indices:
                    return [i for i in indices if 0 <= i < num_formulas]

        # Fallback: find all numbers in response
        all_numbers = re.findall(r"\b(\d+)\b", response)
        indices = [int(n) - 1 for n in all_numbers if 1 <= int(n) <= num_formulas]
        if indices:
            seen: set[int] = set()
            unique = [i for i in indices if not (i in seen or seen.add(i))]
            return unique

        # Last resort: first n_candidates
        return list(range(min(self.n_candidates, num_formulas)))

    def _fallback_random_candidates(
        self, observed_configs: pd.DataFrame
    ) -> pd.DataFrame:
        """Return random candidates from observed when no unobserved points remain."""
        n = min(self.n_candidates, len(observed_configs))
        sampled = observed_configs.sample(n=n, replace=False)
        result = pd.DataFrame()
        for _, row in sampled.iterrows():
            config = {col: float(row[col]) for col in self.feature_cols}
            result = pd.concat(
                [result, pd.DataFrame([config])], ignore_index=True
            )
        return result

    def _fallback_random_from(
        self,
        unobserved: list[tuple[dict[str, float], int, float]],
    ) -> tuple[pd.DataFrame, float, float]:
        """Return random unobserved candidates when GP training fails."""
        n = min(self.n_candidates, len(unobserved))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(unobserved), size=n, replace=False)
        result = pd.DataFrame()
        for idx in indices:
            result = pd.concat(
                [result, pd.DataFrame([unobserved[idx][0]])],
                ignore_index=True,
            )
        return result, 0.0, 0.0
