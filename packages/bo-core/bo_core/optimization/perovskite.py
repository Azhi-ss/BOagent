from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from bo_core.llm_client import DeepSeekClient, LlmCallResult
from bo_core.optimization.memory import Insight, VectorMemory


def extract_yes_logprob(logprobs_payload: dict[str, Any] | None) -> float:
    """Helper function to parse token log-probabilities for 'Yes' from OpenAI/DeepSeek schema."""
    raw_val = _extract_yes_logprob_raw(logprobs_payload)
    if raw_val <= -20.0:
        return raw_val
    return max(-10.0, raw_val)


def _extract_yes_logprob_raw(logprobs_payload: dict[str, Any] | None) -> float:
    if not logprobs_payload:
        return -20.0

    content = logprobs_payload.get("content", [])
    if not content:
        return -20.0

    first_token_info = content[0]
    # Check if the generated token itself is Yes/yes
    generated_token = str(first_token_info.get("token", "")).strip().lower()
    if generated_token == "yes":
        return float(first_token_info.get("logprob", 0.0))

    # Check top_logprobs
    top_list = first_token_info.get("top_logprobs", [])
    for top_token_info in top_list:
        token = str(top_token_info.get("token", "")).strip().lower()
        if token == "yes":
            return float(top_token_info.get("logprob", -20.0))

    # If not found but generated is "no", return a default low logprob based on the lowest top_logprob
    if generated_token == "no":
        if top_list:
            min_top_logprob = min(float(t.get("logprob", -20.0)) for t in top_list)
            return min_top_logprob - 2.0
        return -10.0

    return -20.0


class PerovskiteKnowledgeEngine:
    """Perovskite-specific LLM reasoning and scientific memory."""

    def __init__(self, chat_engine: str = "deepseek-v4-flash", memory_path: str | None = None):
        self.chat_engine = chat_engine
        self._client = DeepSeekClient.from_env()
        if chat_engine:
            self._client.model = chat_engine
        # Persistent cumulative insight memory with Doubao embedding retrieval
        self.memory = VectorMemory(persist_path=memory_path)
        self._iteration = 0

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def _get_physical_context_and_hints(
        self, feature_cols: list[str], include_task_prefix: bool = True
    ) -> tuple[str, dict[str, str]]:
        physical_context = ""
        feature_hints = {}

        if "CHI_PVK" in feature_cols and "Eg_HTL" in feature_cols:
            prefix = "This task involves optimizing energy band alignment in perovskite solar cells to maximize PCE.\n" if include_task_prefix else ""
            physical_context = (
                prefix +
                "Semiconductor Physics Rules for Energy Alignment:\n"
                "1. Conduction Band Offset (CBO): (CHI_PVK - CHI_ETL). Ideal range: [-0.1, 0.3] eV. A negative CBO (cliff) causes huge V_oc loss due to interface recombination. A large positive CBO (spike) blocks electron extraction, reducing J_sc.\n"
                "2. Valence Band Offset (VBO): Approximated as (CHI_HTL + Eg_HTL) - CHI_PVK. Ideal range: [1.7, 2.0] eV. Values below 1.7 eV cause V_oc loss; above 2.0 eV blocks hole extraction.\n"
                "3. Electron Blocking: The HTL LUMO (CHI_HTL) should be much higher than PVK LUMO (CHI_PVK) to block electrons. Difference > 0.5 eV preferred."
            )
            feature_hints = {
                "CHI_PVK": "Electron affinity of Perovskite (LUMO position)",
                "Eg_HTL": "Band gap of HTL",
                "CHI_HTL": "Electron affinity of HTL (LUMO position)",
                "Eg_ETL": "Band gap of ETL",
                "CHI_ETL": "Electron affinity of ETL (LUMO position)"
            }
        elif "Nt_PVK/ETL" in feature_cols or "Na_PVK" in feature_cols:
            prefix = "This task involves optimizing defects and doping to minimize non-radiative recombination and maximize carrier extraction.\n" if include_task_prefix else ""
            physical_context = (
                prefix +
                "Physical Guidelines for Defect/Doping Optimization:\n"
                "1. Recombination Centers: Interface trap densities (Nt) are the primary source of V_oc loss. Logarithmic reduction in Nt usually leads to linear gain in V_oc.\n"
                "2. Built-in Potential (V_bi): Higher doping in HTL (Na_HTL) and ETL (Nd_ETL) increases V_bi, improving charge separation. However, excessive doping (> 10^19 cm^-3) can cause tunneling-assisted recombination or shunting.\n"
                "3. Shunt Resistance: Counter-doping (e.g. Nd_HTL) must be minimized as it introduces parasitic leakage paths, killing the Fill Factor (FF)."
            )
            feature_hints = {
                "Nt_PVK/ETL": "Trap density at PVK/ETL interface (minimize to reduce J_0)",
                "Nt_HTL/PVK": "Trap density at HTL/PVK interface (minimize to reduce J_0)",
                "Na_PVK": "P-type doping in Perovskite absorber",
                "Nd_PVK": "N-type doping in Perovskite absorber",
                "Na_HTL": "Acceptor doping in HTL (improves hole extraction)",
                "Nd_ETL": "Donor doping in ETL (improves electron extraction)",
                "Nd_HTL": "Parasitic donor doping in HTL (causes shunting)",
                "Na_ETL": "Parasitic acceptor doping in ETL (causes shunting)"
            }
        return physical_context, feature_hints

    def _get_cbo_metrics(self, data: dict[str, float], for_ui: bool = False) -> tuple[float, str] | None:
        """Calculate CBO and map to status string."""
        if "CHI_PVK" not in data or "CHI_ETL" not in data:
            return None
        cbo = data["CHI_PVK"] - data["CHI_ETL"]
        if for_ui:
            if cbo < -0.1: status = "Cliff (Recombination Loss)"
            elif cbo > 0.3: status = "Spike (Extraction Barrier)"
            else: status = "Ideal"
        else:
            if cbo < -0.1: status = "Cliff (Violated)"
            elif cbo > 0.3: status = "Spike (Violated)"
            else: status = "Ideal"
        return cbo, status

    def _get_vbo_metrics(self, data: dict[str, float]) -> tuple[float, str] | None:
        """Calculate VBO and map to status string."""
        if not all(k in data for k in ["CHI_HTL", "Eg_HTL", "CHI_PVK"]):
            return None
        vbo = (data["CHI_HTL"] + data["Eg_HTL"]) - data["CHI_PVK"]
        status = "Ideal" if 1.7 <= vbo <= 2.0 else "Sub-optimal"
        return vbo, status


    def summarize_lessons(
        self,
        observed_configs: pd.DataFrame,
        observed_scores: list[float],
        feature_cols: list[str],
        target_name: str,
        best_score: float = 0.0,
    ) -> str:
        """Extract structured Insight from current history and store in VectorMemory.

        Returns a plain-text summary string (for backward compatibility with optimizer.py),
        but also writes a structured Insight into self.memory for future semantic retrieval.
        """
        if observed_configs.empty or not self._client.is_configured():
            return ""

        self._iteration += 1
        history_str = ""
        for i, (_, row) in enumerate(observed_configs.iterrows()):
            feats = ", ".join(f"{col}={row[col]:.4f}" for col in feature_cols)
            cbo_str = ""
            cbo_metrics = self._get_cbo_metrics(row)
            if cbo_metrics:
                cbo_str = f", CBO={cbo_metrics[0]:.3f}eV"
            history_str += f"\n  [{i+1}] {feats}{cbo_str} → {target_name}={observed_scores[i]:.4f}"

        # Ask LLM to produce structured JSON insight (Reasoning-BO ReasoningNotesResponse pattern)
        prompt = f"""You are a materials scientist analyzing perovskite solar cell optimization experiments.
Extract structured insights from the experiment history below.

Experiment History:{history_str}

Respond ONLY with a valid JSON object (no markdown, no extra text) with these keys:
{{
  "notes": ["<general observation>"],
  "key_findings": ["<verifiable result or discovery>"],
  "parameter_relationships": ["<cause-effect relationship between parameters>"],
  "optimization_principles": ["<actionable rule for future trials>"]
}}

Each list should have 1-3 concise bullet strings. Focus on what drives high {target_name}."""

        result: LlmCallResult = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a professional materials scientist. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            extra_body={"thinking": {"type": "disabled"}}
        )

        if result.status != "success":
            return ""

        # Parse structured JSON and store in VectorMemory
        raw = result.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
            insight = Insight(
                iteration=self._iteration,
                best_score=best_score,
                notes=data.get("notes", []),
                key_findings=data.get("key_findings", []),
                parameter_relationships=data.get("parameter_relationships", []),
                optimization_principles=data.get("optimization_principles", []),
            )
            self.memory.add(insight)
            # Return flat text for backward-compatible single-string usage
            return insight.to_text()
        except (json.JSONDecodeError, Exception):
            # Fallback: store raw as a plain note
            insight = Insight(
                iteration=self._iteration,
                best_score=best_score,
                notes=[raw[:200]],
            )
            self.memory.add(insight)
            return raw[:200]


    def build_system_prompt_for_viability(
        self,
        target_name: str,
        feature_cols: list[str],
        observed_data: list[tuple[dict[str, float], float]],
    ) -> str:
        """Construct a system prompt detailing physical rules and observations for pointwise viability queries."""
        physical_context, feature_hints = self._get_physical_context_and_hints(feature_cols, include_task_prefix=False)

        prompt = f"""You are a senior materials scientist specializing in device physics for Perovskite Solar Cells.
Your goal is to evaluate if a candidate formulation is physically viable and likely to achieve high power conversion efficiency ({target_name}).

### Device Physics Rules
{physical_context}

### Parameter Definitions"""
        for col in feature_cols:
            hint = feature_hints.get(col, "Feature parameter")
            prompt += f"\n- {col}: {hint}"

        # Sort history to show all observed formulations
        sorted_obs = sorted(observed_data, key=lambda x: x[1], reverse=True)
        if sorted_obs:
            prompt += "\n\n### All Observed Formulations (sorted by performance)"
            for i, (obs_values, obs_score) in enumerate(sorted_obs):
                feats_str = ", ".join(f"{k}={obs_values.get(k, 0.0):.4f}" for k in feature_cols)
                prompt += f"\n [{i+1}] {feats_str} -> {target_name}={obs_score:.4f}"

        # Inject memory insights if available
        if len(self.memory) > 0:
            memory_block = self.memory.format_all_for_prompt(max_items=3)
            if memory_block:
                prompt += f"\n\n{memory_block}"

        prompt += "\n\nBased on these rules and guidelines, evaluate the candidate formulation provided in the user message. Answer strictly with either 'Yes' or 'No'."
        return prompt


    def evaluate_candidate_viability(
        self,
        candidate: dict[str, float],
        system_prompt: str,
        feature_cols: list[str],
        gp_mean: float | None = None,
        gp_std: float | None = None,
    ) -> float:
        """Query LLM for a single candidate's physical viability and return log_prob(Yes)."""
        if not self._client.is_configured():
            return 0.0

        user_prompt = "Candidate formulation:\n"
        for col in feature_cols:
            user_prompt += f"- {col}: {candidate.get(col, 0.0):.4f}\n"

        cbo_metrics = self._get_cbo_metrics(candidate)
        if cbo_metrics:
            user_prompt += f"- Calculated CBO: {cbo_metrics[0]:.4f} eV ({cbo_metrics[1]})\n"

        vbo_metrics = self._get_vbo_metrics(candidate)
        if vbo_metrics:
            user_prompt += f"- Calculated VBO: {vbo_metrics[0]:.4f} eV ({vbo_metrics[1]})\n"

        if gp_mean is not None and gp_std is not None:
            user_prompt += "\nGP Surrogate Predictions:\n"
            user_prompt += f"- Predicted Score (mean): {gp_mean:.4f}\n"
            user_prompt += f"- Prediction Uncertainty (std): {gp_std:.4f}\n"

        user_prompt += "\nIs this candidate formulation physically viable and likely to achieve high PCE? Answer 'Yes' or 'No'."

        result: LlmCallResult = self._client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1,
            extra_body={
                "thinking": {"type": "disabled"},
                "logprobs": True,
                "top_logprobs": 5
            }
        )

        if result.status != "success" or not result.logprobs:
            return -2.0
        return extract_yes_logprob(result.logprobs)

    def enrich_suggestions(
        self, suggestions: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """Attach perovskite metrics used by downstream displays."""
        enriched = []
        for suggestion in suggestions:
            item = dict(suggestion)
            cbo_metrics = self._get_cbo_metrics(item, for_ui=True)
            if cbo_metrics:
                item["CBO"] = cbo_metrics[0]
                item["CBO_Status"] = cbo_metrics[1]
            vbo_metrics = self._get_vbo_metrics(item)
            if vbo_metrics:
                item["VBO"] = vbo_metrics[0]
                item["VBO_Status"] = vbo_metrics[1]
            enriched.append(item)
        return enriched
