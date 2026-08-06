"""Chemistry-adapted LGBO prompt builder (paper Appendix B).

Faithfully reproduces the modular system + user prompt structure from the LGBO
paper (arxiv 2605.17976v1, Appendix B), adapted to the Buchwald-Hartwig C-N
and Suzuki-Miyaura C-C coupling datasets. The system prompt enforces the
evidence hierarchy, strict bracketed output protocol, categorical-literal
output, and anti-collapse rules. The user prompt carries the chemistry
``[Background]`` and the per-round ``[Review]`` (history + previous thinking).

Objective direction is flipped from the paper's "Minimize f(x)" to
"Maximize f(x) = Yield (%)" because the chemical-reaction target is yield.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from bo_core.benchmark.datasets import get_dataset


@dataclass(frozen=True)
class _Background:
    """Reaction name and mechanism parsed from a dataset README."""

    name: str
    mechanism: str


def _extract_reaction_name(text: str) -> str:
    """Pull the reaction name from the dataset README's ``反应类型`` table row.

    The READMEs name the reaction in Chinese (``偶联反应``); the surrounding
    prompt template is English, so the coupling term is translated to keep the
    filled prompt coherent and aligned with the README's English heading.
    """
    match = re.search(r"\|\s*反应类型\s*\|\s*([^|]+?)\s*\|", text)
    if match is None:
        raise KeyError("reaction type")
    return (
        match.group(1)
        .replace("偶联反应", "coupling")
        .replace("偶联", "coupling")
        .strip()
    )


def _extract_mechanism(text: str) -> str:
    """Pull the first prose paragraph under the README's ``反应任务描述``."""
    section = re.search(
        r"^##\s*反应任务描述\s*$\n+(.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if section is None:
        raise KeyError("mechanism")
    for block in section.group(1).split("\n\n"):
        paragraph = block.strip()
        if paragraph and not paragraph.startswith(("```", "|")):
            return paragraph
    raise KeyError("mechanism")


@cache
def _load_background(dataset: str) -> _Background:
    """Read and parse a dataset's README; cached to avoid repeated disk reads."""
    readme = get_dataset(dataset).directory / "README.md"
    text = readme.read_text(encoding="utf-8")
    return _Background(
        name=_extract_reaction_name(text),
        mechanism=_extract_mechanism(text),
    )


@dataclass(frozen=True)
class DatasetMeta:
    """Algorithm-facing prompt input assembled from the dataset registry."""

    dataset: str
    feature_cols: list[str]
    options: dict[str, list[str]]
    target_name: str
    objective: str = "Maximize"

    @property
    def reaction_name(self) -> str:
        return _load_background(self.dataset).name

    @property
    def mechanism(self) -> str:
        return _load_background(self.dataset).mechanism


_SYSTEM_PROMPT_TEMPLATE = """You are a senior chemist specializing in experimental optimization of {reaction}.
Your goal is to propose the next reaction condition most likely to achieve a high {target}.

# Evidence hierarchy (critical)
- PRIMARY: Background knowledge, chemical/physical mechanisms, constraints, and units.
- SECONDARY (auxiliary only): Historical trial points/observations and thinking in the review.
- If background implications conflict with historical points, SIDE WITH BACKGROUND.

# Background (fixed across rounds)
- We run iterative chemistry experiments ({reaction}).
- Parameters are categorical reagent/condition choices. Always use the declared parameter order and the literal IUPAC option names; do NOT normalize or encode them.

# Modes (pick exactly ONE) - emit as a JSON object in Final Answer
1) Point:  {{"mode": "point",  "values": ["x1", "x2", ..., "xd"], "confidence": ccc}}
2) Region: {{"mode": "region", "lb": ["lb1", ..., "lbd"], "ub": ["ub1", ..., "ubd"], "confidence": ccc}}
- ccc in [0,1] is your confidence that this guidance improves the objective.
- Every value must be the literal IUPAC option string (e.g. "1-bromo-4-ethylbenzene"), double-quoted.
- For categorical variables a "range" is not meaningful: in region mode set lb[i]=ub[i] to the same literal value for every dimension. (Region is kept for protocol parity; for this fully-categorical task a point is expected.)

# How to reason (prioritize background over past points)
- Start from first principles: mechanism-driven trends (leaving-group reactivity, ligand sterics/electronics, base strength, additive effects), known interactions, and feasible ranges.
- Use historical data ONLY as weak corroboration or disproof of a background-based hypothesis.
- Do NOT anchor on previous best/nearest points; avoid proposing a point merely because it appeared before.
- If historical points cluster narrowly, consider a background-justified exploratory move (e.g. switch ligand class or base strength).
- Prefer REGION when background suggests multiple nearby settings could satisfy the mechanistic target; choose POINT only when background+data imply a sharp optimum.

# Output protocol (two blocks)
1) Thinking:
   - Be concise but informative, in this order:
     (a) Background-based rationale (mechanism/constraints) that leads to your proposal.
     (b) How (if at all) historical data supports/contradicts this mechanism (<=2 sentences).
     (c) Why point vs region given the mechanism and uncertainty.
2) Final Answer:
   - A single JSON object on its own line, no extra text before or after:
     {{"mode": "point", "values": ["v1", "v2", ..., "vd"], "confidence": ccc}}
     OR
     {{"mode": "region", "lb": ["lb1", ..., "lbd"], "ub": ["ub1", ..., "ubd"], "confidence": ccc}}
   - Double-quote every value string; the JSON must be valid (commas/brackets inside IUPAC names are fine inside quotes).

# Hard constraints
- Do NOT normalize, encode, or re-order parameters.
- Every value must be one of the declared options for its variable (see [Background] in the user prompt).
- Keep parameter order consistent with the declared order.
- No extra commentary in Final Answer beyond the bracketed structure.

# Anti-collapse checks
- Never center a region or point on a past observation unless mechanistically justified.
- If you reuse a past setting, explicitly state the mechanism that makes it optimal (in Thinking)."""


def build_system_prompt(meta: DatasetMeta) -> str:
    """Build the fixed system prompt (Appendix B structure, chemistry-adapted)."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        reaction=meta.reaction_name,
        target=meta.target_name,
    )


def build_user_prompt(
    meta: DatasetMeta,
    history: Sequence[tuple[dict[str, str], float]],
    prev_thinking: str | None = None,
    max_history: int = 10,
) -> str:
    """Build the per-round user prompt: [Background] + [Review].

    ``history`` is a list of (condition_dict, observed_yield) tuples; the most
    recent ``max_history`` are shown newest first. ``prev_thinking`` is the LLM's
    own Thinking block from the previous round (for iterative refinement).
    """
    d = len(meta.feature_cols)
    order_str = ", ".join(meta.feature_cols)

    lines: list[str] = ["[Background]"]
    lines.append(f"- Experiment type & purpose: {meta.reaction_name} reaction-condition "
                 f"optimization to maximize {meta.target_name} (%).")
    lines.append(f"- Mechanism: {meta.mechanism}")
    lines.append(f"- Parameter order (d={d}): {order_str}")
    lines.append(f"- Objective: {meta.objective} f(x) = {meta.target_name} (%) (single objective).")
    lines.append("- Constraints: all parameters are categorical; output the literal IUPAC "
                 "option name for each; respect the declared order; do NOT normalize or encode.")
    lines.append("- Bounds (categorical options per variable):")
    for col in meta.feature_cols:
        opts = meta.options.get(col, [])
        lines.append(f"  - {col}: {opts}")

    lines.append("")
    lines.append("[Review]")
    recent = list(history[-max_history:])[::-1]  # newest first
    if recent:
        lines.append("- Historical data (newest first):")
        for i, (cond, yld) in enumerate(recent, start=1):
            feats = ", ".join(f"{c}={cond.get(c, '?')}" for c in meta.feature_cols)
            lines.append(f"  [{i}] {feats} -> {meta.target_name}={yld:.4f}")
    else:
        lines.append("- Historical data: (none yet)")
    lines.append(f"- Thinking from the previous round: {prev_thinking.strip() if prev_thinking else '(none)'}")
    lines.append("- Adoption note: Suggestions were used as guidance; actual tested points may differ.")
    return "\n".join(lines)
