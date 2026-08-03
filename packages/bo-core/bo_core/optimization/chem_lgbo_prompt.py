"""Prompt construction for Chem-LGBO sparse-subspace guidance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .lgbo_prompt import DatasetMeta

CompatibilityPairs = Mapping[tuple[str, str], Sequence[tuple[str, str]]]


@dataclass(frozen=True)
class PreviousGuidanceOutcome:
    """Observed result of the previous accepted guidance proposal."""

    proposed_subspace: dict[str, list[str]]
    selected_condition: dict[str, str]
    selected_in_subspace: bool
    observed_yield: float
    incumbent_before: float

    @property
    def improvement(self) -> float:
        return self.observed_yield - self.incumbent_before


def build_compatibility_pairs(
    candidate_features: pd.DataFrame,
    feature_pair: tuple[str, str],
) -> list[tuple[str, str]]:
    """Return candidate-pool pairs in first-occurrence order."""
    left, right = feature_pair
    if left not in candidate_features or right not in candidate_features:
        return []
    rows = candidate_features.loc[:, [left, right]].drop_duplicates()
    return list(rows.itertuples(index=False, name=None))


def build_system_prompt(meta: DatasetMeta, *, outcome_feedback: bool = False) -> str:
    """Build the fixed sparse-guidance protocol prompt."""
    prompt = f"""You are a senior chemist optimizing {meta.reaction_name} for {meta.target_name}.
Use mechanism and the observed history to propose a promising sparse categorical subspace.
Choose one or more declared feature fields; omitted fields are unconstrained.
Each chosen field maps to a non-empty array of exact literal option strings.
Final Answer must contain exactly this JSON shape and nothing after it:
{{"subspace":{{"Ligand":["L1"],"Base":["B1","B2"]}}}}
Do not emit Point, Region, bounds, or confidence. Do not emit empty arrays.
The suggestion guides the posterior mean; it does not hard-filter the experiment pool.
Compatibility pairs are feature-only hints from the unlabeled candidate pool, not validation rules."""
    if not outcome_feedback:
        return prompt
    return prompt + """

The previous guidance outcome is evidence about one selected point, not proof that the entire proposed subspace is good or bad.
Silently reassess the previous proposal before producing the next subspace:
- If the selected point was outside it, treat the proposal as untested.
- If it was inside, compare its observed yield with the incumbent before the trial and inspect the visible history for independent support or contradiction.
- Retain, narrow, broaden, or revise the previous subspace only as supported by observed evidence.
- Do not repeat it merely because it was proposed before, and do not discard it merely because one point failed to improve.
Return only the required JSON. Do not output this reassessment."""


def build_user_prompt(
    meta: DatasetMeta,
    history: Sequence[tuple[dict[str, str], float]],
    compatibility_pairs: CompatibilityPairs,
    prev_thinking: str | None = None,
    max_history: int = 10,
    *,
    previous_outcome: PreviousGuidanceOutcome | None = None,
    outcome_feedback: bool = False,
) -> str:
    """Build feature-only background plus observed optimization history."""
    lines = [
        "[Background]",
        f"- Reaction: {meta.reaction_name}",
        f"- Mechanism: {meta.mechanism}",
        "- Literal options:",
    ]
    for field in meta.feature_cols:
        lines.append(f"  - {field}: {meta.options.get(field, [])}")

    for (left, right), pairs in compatibility_pairs.items():
        if not pairs:
            continue
        lines.append(f"- Observed candidate compatibility pairs for {left} + {right}:")
        lines.extend(f"  - ({a}, {b})" for a, b in pairs)

    lines.extend(("", "[Review]"))
    recent = list(history[-max_history:])[::-1]
    if recent:
        lines.append("- Historical observations (newest first):")
        for index, (condition, observed_yield) in enumerate(recent, start=1):
            values = ", ".join(
                f"{field}={condition.get(field, '?')}" for field in meta.feature_cols
            )
            lines.append(
                f"  [{index}] {values} -> {meta.target_name}={observed_yield:.4f}"
            )
    else:
        lines.append("- Historical observations: (none yet)")
    lines.append(
        "- Previous thinking: "
        + (prev_thinking.strip() if prev_thinking else "(none)")
    )
    if outcome_feedback:
        lines.extend(("", "[Previous guidance outcome]"))
        if previous_outcome is None:
            lines.append("- (none)")
        else:
            lines.extend(
                (
                    "- Proposed subspace: "
                    + json.dumps(previous_outcome.proposed_subspace, sort_keys=True),
                    "- Selected condition: "
                    + json.dumps(previous_outcome.selected_condition, sort_keys=True),
                    "- Selected point was inside the proposed subspace: "
                    + str(previous_outcome.selected_in_subspace).lower(),
                    f"- Observed {meta.target_name}: {previous_outcome.observed_yield:.4f}",
                    f"- Incumbent before this trial: {previous_outcome.incumbent_before:.4f}",
                    f"- Improvement over incumbent: {previous_outcome.improvement:.4f}",
                )
            )
    return "\n".join(lines)
