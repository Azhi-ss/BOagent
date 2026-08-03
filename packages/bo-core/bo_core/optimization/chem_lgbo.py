"""Pure candidate-pool helpers for Chem-LGBO."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from bo_core.optimization.chem_lgbo_parser import parse_subspace_response
from bo_core.optimization.chem_lgbo_prompt import (
    PreviousGuidanceOutcome,
    build_compatibility_pairs,
    build_system_prompt,
    build_user_prompt,
)
from bo_core.optimization.lgbo import LGBOEngine
from bo_core.optimization.surrogate import SurrogateModel

PROPOSE_SUBSPACE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_sparse_subspace",
        "description": "Propose a sparse categorical subspace for the remaining experiment pool.",
        "parameters": {
            "type": "object",
            "properties": {
                "subspace": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "minProperties": 1,
                }
            },
            "required": ["subspace"],
            "additionalProperties": False,
        },
    },
}

PROPOSE_SUBSPACE_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "propose_sparse_subspace"},
}

_RETRYABLE_REASONS = {
    "missing_tool_call",
    "wrong_tool_name",
    "multiple_tool_calls",
    "invalid_tool_arguments",
    "empty_response",
    "invalid_json",
    "invalid_schema",
    "unknown_field",
    "empty_choice",
    "duplicate_value",
    "unknown_value",
    "empty_intersection",
    "already_queried_only",
    "uninformative_full_pool",
}


def build_subspace_mask(
    candidate_features: pd.DataFrame,
    subspace: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Build an exact joint membership mask over candidate rows."""
    mask = np.ones(len(candidate_features), dtype=bool)
    for field, values in subspace.items():
        if field not in candidate_features:
            return np.zeros(len(candidate_features), dtype=bool)
        mask &= candidate_features[field].isin(values).to_numpy()
    return mask


def _validate_float_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def masked_mean_shift(
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return a copy of ``mu`` with one posterior sigma added under ``mask``."""
    mu_array = _validate_float_vector("mu", mu)
    sigma_array = _validate_float_vector("sigma", sigma)
    mask_array = np.asarray(mask)
    if mask_array.ndim != 1 or mask_array.dtype != np.bool_:
        raise TypeError("mask must be a one-dimensional boolean array")
    if len(mu_array) != len(sigma_array) or len(mu_array) != len(mask_array):
        raise ValueError("mu, sigma, and mask must have equal lengths")

    shifted = mu_array.copy()
    shifted[mask_array] += sigma_array[mask_array]
    return shifted


def generate_counterfactual_indices(
    *,
    candidate_features: pd.DataFrame,
    feature_options: Mapping[str, Sequence[str]],
    subspace: Mapping[str, Sequence[str]],
    queried_mask: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    best_f: float,
    expected_improvement: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    rng: np.random.RandomState,
    count: int,
) -> list[int]:
    """Generate bounded matched random-subspace EI choices without oracle data."""
    if count <= 0:
        return []

    mu_array = _validate_float_vector("mu", mu)
    sigma_array = _validate_float_vector("sigma", sigma)
    queried = np.asarray(queried_mask)
    if queried.ndim != 1 or queried.dtype != np.bool_:
        raise TypeError("queried_mask must be a one-dimensional boolean array")
    if len(candidate_features) != len(mu_array):
        raise ValueError("candidate_features and posterior arrays must have equal lengths")
    if len(queried) != len(mu_array) or len(sigma_array) != len(mu_array):
        raise ValueError("queried_mask, mu, and sigma must have equal lengths")
    if not subspace:
        return []

    fields = tuple(subspace)
    lengths = {field: len(subspace[field]) for field in fields}
    if any(length == 0 for length in lengths.values()):
        return []
    for field in fields:
        if field not in feature_options:
            return []
        if lengths[field] > len(feature_options[field]):
            return []

    remaining = ~queried
    remaining_size = int(np.count_nonzero(remaining))
    if remaining_size == 0:
        return []

    accepted: list[int] = []
    seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
    max_attempts = max(100, count * 100)

    for _ in range(max_attempts):
        proposal: dict[str, list[str]] = {}
        for field in fields:
            values = tuple(str(value) for value in rng.choice(
                list(feature_options[field]),
                size=lengths[field],
                replace=False,
            ))
            proposal[field] = list(values)

        key = tuple((field, tuple(proposal[field])) for field in fields)
        if key in seen:
            continue
        seen.add(key)

        raw_mask = build_subspace_mask(candidate_features, proposal)
        mask = raw_mask & remaining
        mask_size = int(np.count_nonzero(mask))
        if mask_size == 0 or mask_size == remaining_size:
            continue

        shifted = masked_mean_shift(mu_array, sigma_array, mask)
        acquisition = np.asarray(expected_improvement(shifted, sigma_array, best_f))
        if acquisition.ndim != 1 or len(acquisition) != len(mu_array):
            raise ValueError("expected_improvement must return a vector matching mu")
        if not np.issubdtype(acquisition.dtype, np.number):
            raise TypeError("expected_improvement must return numeric values")
        acquisition = np.where(np.isfinite(acquisition), acquisition, -np.inf)
        acquisition = np.where(remaining, acquisition, -np.inf)
        if not np.any(np.isfinite(acquisition)):
            continue
        accepted.append(int(np.argmax(acquisition)))
        if len(accepted) == count:
            break

    return accepted


class ChemLGBOEngine(LGBOEngine):
    """LGBO with sparse candidate-pool subspaces and a fixed one-sigma shift."""

    def __init__(
        self,
        dataset: str,
        seed: int = 100,
        use_llm: bool = True,
        n_iters: int = 40,
        *,
        n_counterfactuals: int = 0,
        outcome_feedback: bool = False,
        llm_temperature: float = 0.2,
        **legacy_kwargs: Any,
    ) -> None:
        self.n_counterfactuals = min(100, max(0, int(n_counterfactuals)))
        self.outcome_feedback = outcome_feedback
        self.previous_outcome: PreviousGuidanceOutcome | None = None
        self.guidance_artifacts: list[dict[str, Any]] = []
        self._pending_guidance_artifact: dict[str, Any] | None = None
        super().__init__(
            dataset,
            seed=seed,
            use_llm=use_llm,
            n_iters=n_iters,
            llm_temperature=llm_temperature,
            **legacy_kwargs,
        )
        pair = ("Electrophile", "Nucleophile")
        pairs = build_compatibility_pairs(
            self.test_df.loc[:, self.feature_cols], pair
        )
        self.compatibility_pairs = {pair: pairs} if pairs else {}

    def _record_guidance_selection(self, selected_index: int) -> None:
        artifact = self._pending_guidance_artifact
        if artifact is None:
            return
        artifact["selected_index"] = selected_index
        self._pending_guidance_artifact = None
        if artifact["parser_reason"] != "accepted":
            self.previous_outcome = None
            return
        row = self.trajectory[-1]
        self.previous_outcome = PreviousGuidanceOutcome(
            proposed_subspace=artifact["subspace"],
            selected_condition=row["condition"],
            selected_in_subspace=bool(row["selected_in_subspace"]),
            observed_yield=float(row["observed_yield"]),
            incumbent_before=float(np.max(self.y_obs)),
        )

    def _store_guidance_artifact(
        self,
        *,
        raw_response: str | None,
        reason: str,
        subspace: dict[str, list[str]] | None,
        mask_size: int | None,
        remaining_pool_size: int,
        counterfactual_seed: int | None,
        counterfactual_indices: list[int],
        react_retried: bool = False,
        react_first_reason: str | None = None,
        llm_attempts: int = 1,
        tool_call_id: str | None = None,
    ) -> None:
        artifact = {
            "step": self.iteration + 1,
            "raw_response": raw_response,
            "parser_reason": reason,
            "subspace": subspace,
            "mask_size": mask_size,
            "remaining_pool_size": remaining_pool_size,
            "counterfactual_seed": counterfactual_seed,
            "counterfactual_indices": counterfactual_indices,
            "react_retried": react_retried,
            "react_first_reason": react_first_reason,
            "llm_attempts": llm_attempts,
            "tool_call_id": tool_call_id,
            "selected_index": None,
        }
        self.guidance_artifacts.append(artifact)
        self._pending_guidance_artifact = artifact

    def _fallback(
        self,
        mu: np.ndarray,
        *,
        reason: str,
        raw_response: str | None,
        subspace: dict[str, list[str]] | None = None,
        mask_size: int | None = None,
        remaining_pool_size: int,
        coverage: float | None = None,
        react_retried: bool = False,
        react_first_reason: str | None = None,
        llm_attempts: int = 1,
        tool_call_id: str | None = None,
    ) -> tuple[np.ndarray, None, dict[str, Any], None]:
        self._store_guidance_artifact(
            raw_response=raw_response,
            reason=reason,
            subspace=subspace,
            mask_size=mask_size,
            remaining_pool_size=remaining_pool_size,
            counterfactual_seed=None,
            counterfactual_indices=[],
            react_retried=react_retried,
            react_first_reason=react_first_reason,
            llm_attempts=llm_attempts,
            tool_call_id=tool_call_id,
        )
        return (
            mu,
            None,
            self._guidance_diagnostics(
                "fallback",
                reason,
                subspace=subspace,
                mask_size=mask_size,
                coverage=coverage,
            ),
            None,
        )

    def _llm_mean_shift(
        self,
        surrogate: SurrogateModel,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, str | None, dict[str, Any], np.ndarray | None]:
        del surrogate
        remaining = np.ones(self.M, dtype=bool)
        for index in self.queried:
            if 0 <= index < self.M:
                remaining[index] = False
        remaining_pool_size = int(np.count_nonzero(remaining))
        candidate_features = self.test_df.loc[:, self.feature_cols]

        prior_history = [
            (
                {field: str(self.train_df[field].iloc[i]) for field in self.feature_cols},
                float(self.train_df[self.target_col].iloc[i]),
            )
            for i in range(len(self.train_df))
        ]
        trajectory_history = [
            (row["condition"], row["observed_yield"]) for row in self.trajectory
        ]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.meta, outcome_feedback=self.outcome_feedback
                ),
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    self.meta,
                    prior_history + trajectory_history,
                    self.compatibility_pairs,
                    self.prev_thinking,
                    previous_outcome=self.previous_outcome,
                    outcome_feedback=self.outcome_feedback,
                ),
            },
        ]
        first_reason: str | None = None

        for attempt in range(2):
            result, call_reason = self._call_llm(
                messages,
                tools=[PROPOSE_SUBSPACE_TOOL],
                tool_choice=PROPOSE_SUBSPACE_TOOL_CHOICE,
            )
            if result is None:
                if attempt == 0 and call_reason == "empty_response":
                    first_reason = call_reason
                    messages = messages + [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "status": "rejected",
                                    "reason": call_reason,
                                    "constraint": "Call propose_sparse_subspace with a subspace containing at least one unqueried candidate and not the entire remaining pool.",
                                },
                                separators=(",", ":"),
                            ),
                        }
                    ]
                    continue
                return self._fallback(
                    mu,
                    reason=call_reason,
                    raw_response=None,
                    remaining_pool_size=remaining_pool_size,
                    react_retried=attempt == 1,
                    react_first_reason=first_reason,
                    llm_attempts=attempt + 1,
                )

            tool_calls = getattr(result, "tool_calls", None) or []
            target_calls = [
                call
                for call in tool_calls
                if call.get("function", {}).get("name") == "propose_sparse_subspace"
            ]
            inspected = target_calls[0] if target_calls else (tool_calls[0] if tool_calls else None)
            tool_call_id = str(inspected.get("id")) if inspected and inspected.get("id") else None
            raw_response: str | None = None
            subspace: dict[str, list[str]] | None = None
            mask_size: int | None = None
            coverage: float | None = None

            if len(tool_calls) > 1:
                reason = "multiple_tool_calls"
            elif not tool_calls:
                reason = "missing_tool_call"
            elif not target_calls:
                reason = "wrong_tool_name"
            else:
                arguments = target_calls[0].get("function", {}).get("arguments")
                if not isinstance(arguments, str):
                    reason = "invalid_tool_arguments"
                else:
                    raw_response = arguments
                    subspace, reason = parse_subspace_response(
                        raw_response, self.feature_cols, self.options_json
                    )
                    if subspace is not None:
                        raw_mask = build_subspace_mask(candidate_features, subspace)
                        raw_size = int(np.count_nonzero(raw_mask))
                        mask = raw_mask & remaining
                        mask_size = int(np.count_nonzero(mask))
                        coverage = (
                            mask_size / remaining_pool_size
                            if remaining_pool_size
                            else None
                        )
                        if raw_size == 0:
                            reason = "empty_intersection"
                        elif mask_size == 0:
                            reason = "already_queried_only"
                        elif mask_size == remaining_pool_size:
                            reason = "uninformative_full_pool"
                        else:
                            shifted = masked_mean_shift(mu, sigma, mask)
                            counterfactual_seed = None
                            counterfactual_indices: list[int] = []
                            if self.n_counterfactuals:
                                counterfactual_seed = self.seed * 1000 + self.iteration
                                counterfactual_indices = generate_counterfactual_indices(
                                    candidate_features=candidate_features,
                                    feature_options=self.options_json,
                                    subspace=subspace,
                                    queried_mask=~remaining,
                                    mu=mu,
                                    sigma=sigma,
                                    best_f=float(np.max(self.y_obs)),
                                    expected_improvement=self._expected_improvement,
                                    rng=np.random.RandomState(counterfactual_seed),
                                    count=self.n_counterfactuals,
                                )
                            self._store_guidance_artifact(
                                raw_response=raw_response,
                                reason="accepted",
                                subspace=subspace,
                                mask_size=mask_size,
                                remaining_pool_size=remaining_pool_size,
                                counterfactual_seed=counterfactual_seed,
                                counterfactual_indices=counterfactual_indices,
                                react_retried=attempt == 1,
                                react_first_reason=first_reason,
                                llm_attempts=attempt + 1,
                                tool_call_id=tool_call_id,
                            )
                            return (
                                shifted,
                                self._extract_thinking(raw_response),
                                self._guidance_diagnostics(
                                    "applied",
                                    "accepted",
                                    subspace=subspace,
                                    mask_size=mask_size,
                                    coverage=coverage,
                                    counterfactual_seed=counterfactual_seed,
                                ),
                                mask,
                            )

            if attempt == 1 or reason not in _RETRYABLE_REASONS:
                return self._fallback(
                    mu,
                    reason=reason,
                    raw_response=raw_response,
                    subspace=subspace,
                    mask_size=mask_size,
                    remaining_pool_size=remaining_pool_size,
                    coverage=coverage,
                    react_retried=attempt == 1,
                    react_first_reason=first_reason,
                    llm_attempts=attempt + 1,
                    tool_call_id=tool_call_id,
                )

            first_reason = reason
            error_content = {
                "status": "rejected",
                "reason": reason,
                "constraint": "Choose a subspace containing at least one unqueried candidate and not the entire remaining pool.",
            }
            if inspected is None or tool_call_id is None or len(tool_calls) != 1:
                messages = messages + [
                    {
                        "role": "user",
                        "content": json.dumps(error_content, separators=(",", ":")),
                    }
                ]
            else:
                messages = messages + [
                    {"role": "assistant", "content": "", "tool_calls": tool_calls},
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(error_content, separators=(",", ":")),
                    },
                ]

        raise AssertionError("unreachable")
