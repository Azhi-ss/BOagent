from __future__ import annotations
import json
import os
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from llm_client import DeepSeekClient, LlmCallResult
from optimization.memory import VectorMemory, Insight

@dataclass
class SuggestionResult:
    suggestions: List[Dict[str, float]]
    analysis: str
    prompt: str


def extract_yes_logprob(logprobs_payload: dict[str, Any] | None) -> float:
    """Helper function to parse token log-probabilities for 'Yes' from OpenAI/DeepSeek schema."""
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


class KnowledgeEngine:
    """Module responsible for materials science domain reasoning via LLM.
    
    Integrates VectorMemory for persistent, embedding-based Insight retrieval
    across all BO iterations (inspired by Reasoning-BO's NotesAgent pattern).
    """

    def __init__(self, chat_engine: str = "deepseek-v4-flash", memory_path: str | None = None):
        self.chat_engine = chat_engine
        self._client = DeepSeekClient.from_env()
        if chat_engine:
            self._client.model = chat_engine
        # Persistent cumulative insight memory with Doubao embedding retrieval
        self.memory = VectorMemory(persist_path=memory_path)
        self._iteration = 0

    def build_prompt(
        self,
        target_name: str,
        feature_cols: List[str],
        top_candidates: pd.DataFrame,
        observed_data: List[Tuple[Dict[str, float], float]],
        n_candidates: int = 5,
        scientific_notes: str = ""
    ) -> str:
        """Construct the materials-science domain prompt with task-specific hints."""
        
        # 1. Determine task-specific physical context and semiconductor rules
        physical_context = ""
        feature_hints = {}
        
        if "CHI_PVK" in feature_cols and "Eg_HTL" in feature_cols:
            physical_context = (
                "This task involves optimizing energy band alignment in perovskite solar cells to maximize PCE.\n"
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
            physical_context = (
                "This task involves optimizing defects and doping to minimize non-radiative recombination and maximize carrier extraction.\n"
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

        prompt = f"""You are a senior materials scientist specializing in device physics for Perovskite Solar Cells. 
Your goal is to evaluate candidate formulations and select those that best balance charge extraction, recombination mitigation, and physical stability to achieve the highest PCE ({target_name}).

### Task-Specific Physical Context
{physical_context}

### Parameter Definitions"""
        for col in feature_cols:
            hint = feature_hints.get(col, "Feature parameter for material optimization")
            prompt += f"\n- {col}: {hint}"

        # Inject recent insights from cumulative memory (no embedding call on hot path)
        if len(self.memory) > 0:
            memory_block = self.memory.format_all_for_prompt(max_items=3)
            if memory_block:
                prompt += f"\n\n{memory_block}"
        elif scientific_notes:
            # Legacy fallback for single-string notes
            prompt += f"\n\n### Dynamic Scientific Memory (Lessons from prior trials)\n{scientific_notes}"

        prompt += "\n\n### Observed Experimental History"
        for i, (obs_values, obs_eta) in enumerate(observed_data):
            prompt += f"\n formulation{i + 1}: "
            for col in feature_cols:
                prompt += f"{col}={obs_values.get(col, 0):.4f}, "
            prompt += f"{target_name}={obs_eta:.4f}"

        prompt += "\n\n### Candidate Formulations (GP UCB High-potential candidates)"
        for i, (_, row) in enumerate(top_candidates.iterrows()):
            prompt += f"\n\nCandidate formulation {i + 1}:\n"
            for col in feature_cols:
                prompt += f"{col}={row[col]:.4f}, "
            
            # Explicitly calculate critical offsets if relevant features are present
            if "CHI_PVK" in row and "CHI_ETL" in row:
                cbo = row["CHI_PVK"] - row["CHI_ETL"]
                prompt += f"\nCalculated CBO: {cbo:.4f} eV"
                if not (-0.1 <= cbo <= 0.3):
                    prompt += " (!! OUTSIDE IDEAL RANGE [-0.1, 0.3] !!)"
            
            if all(k in row for k in ["CHI_HTL", "Eg_HTL", "CHI_PVK"]):
                vbo = (row["CHI_HTL"] + row["Eg_HTL"]) - row["CHI_PVK"]
                prompt += f", Calculated VBO: {vbo:.4f} eV"
            
            prompt += f"\nGP Prediction: Mean PCE={row.get('mean', 0):.4f}, Uncertainty(std)={row.get('std', 0):.4f}"

        prompt += f"""

### Decision Instructions
Based on the physical rules above, the historical trends, and GP predictions, select the {n_candidates} formulations most likely to yield the highest {target_name}.

Analyze the candidates using the following methodology:
1. **Band/Interface Calculation**: For each candidate, mentally calculate the critical offsets (CBO, VBO) or doping ratios.
2. **Recombination vs. Extraction**: Identify which candidates might suffer from "cliff-like" recombination or "spike-like" blocking.
3. **Exploration Trade-off**: Prefer candidates that are physically robust but also explore regions with high GP uncertainty (std) if the mean is promising.

Please respond strictly in the following format. Ensure the Analysis section is concise (max 3-4 sentences) to prevent response truncation.
Thinking Process:
[1. Calculate critical physical metrics for top candidates. 2. Compare candidates based on device physics. 3. Finalize selection.]

Analysis:
[A concise technical justification for your selection, highlighting the physical advantages of chosen recipes.]

Selected Formulations:
[List of formulation numbers, e.g., 1, 3, 5]
"""
        return prompt

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
            history_str += f"\n  [{i+1}] {feats} → {target_name}={observed_scores[i]:.4f}"

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

    def refine_suggestions(
        self,
        prompt: str,
        top_candidates: pd.DataFrame,
        feature_cols: List[str],
        n_candidates: int = 5
    ) -> Tuple[List[Dict[str, float]], str]:
        """Call LLM and parse the refined suggestions."""
        if not self._client.is_configured():
            # Fallback if no LLM
            return [row[feature_cols].to_dict() for _, row in top_candidates.head(n_candidates).iterrows()], "LLM not configured. Using GP top candidates."

        result: LlmCallResult = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a professional materials scientist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            extra_body={"thinking": {"type": "disabled"}}
        )

        if result.status != "success":
            return [row[feature_cols].to_dict() for _, row in top_candidates.head(n_candidates).iterrows()], f"LLM Error: {result.error}"

        analysis = result.content
        selected_indices = self._parse_response(analysis, len(top_candidates), n_candidates)
        selected_indices = sorted(selected_indices)
        
        suggestions = []
        for idx in selected_indices:
            row = top_candidates.iloc[idx]
            sug = row[feature_cols].to_dict()
            
            # Add calculated physical parameters for UI display
            if "CHI_PVK" in row and "CHI_ETL" in row:
                sug["CBO"] = row["CHI_PVK"] - row["CHI_ETL"]
            if all(k in row for k in ["CHI_HTL", "Eg_HTL", "CHI_PVK"]):
                sug["VBO"] = (row["CHI_HTL"] + row["Eg_HTL"]) - row["CHI_PVK"]
                
            suggestions.append(sug)
            
        return suggestions, analysis

    def _parse_response(self, response: str, num_candidates: int, n_expected: int = 5) -> List[int]:
        # 1. Look for Selected Formulations:
        match = re.search(r"Selected\s+Formulations?\s*:\s*(.+)", response, re.IGNORECASE)
        if match:
            numbers_text = match.group(1).strip()
            indices = [int(x.strip()) - 1 for x in re.split(r"[,\s]+", numbers_text) if x.strip().isdigit()]
            valid_indices = [i for i in indices if 0 <= i < num_candidates]
            if valid_indices:
                return valid_indices
        
        # 2. Try to search for the final list of numbers in the last paragraph/lines
        # to avoid matching candidate descriptions at the beginning of the text.
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines:
            # Check the last 3 lines
            for line in lines[-3:]:
                nums = re.findall(r"\b(\d+)\b", line)
                if nums and len(nums) <= n_expected + 2:
                    indices = [int(n) - 1 for n in nums if 1 <= int(n) <= num_candidates]
                    valid_indices = []
                    seen = set()
                    for idx in indices:
                        if idx not in seen:
                            seen.add(idx)
                            valid_indices.append(idx)
                    if len(valid_indices) >= 2:
                         return valid_indices
        
        # 3. Fallback: GP top candidates (safe and high-performing)
        return list(range(min(n_expected, num_candidates)))

    def select_initial_points(
        self,
        candidate_points: List[Dict[str, float]],
        feature_cols: List[str],
        target_name: str,
        n_select: int,
    ) -> List[int]:
        """Pick the most promising initial points from a candidate pool via LLM.

        Mirrors PVK-LLM's warm-start strategy: the LLM sees only feature values
        (NOT the target/eta) and chooses on physical reasoning, balancing
        diversity (exploration) and likely performance (exploitation).

        Returns indices into ``candidate_points``. Falls back to a spread of
        indices if the LLM is unavailable or its response can't be parsed.
        """
        n_cand = len(candidate_points)
        if n_select >= n_cand:
            return list(range(n_cand))

        def _fallback() -> List[int]:
            # Evenly spread across the pool for diversity, deterministic.
            return [round(i * (n_cand - 1) / (n_select - 1)) for i in range(n_select)] if n_select > 1 else [0]

        if not self._client.is_configured():
            return _fallback()

        prompt = self._build_initialization_prompt(
            candidate_points, feature_cols, target_name, n_select
        )
        result: LlmCallResult = self._client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in materials science and perovskite solar cells. Provide concise, technical responses.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if result.status != "success":
            return _fallback()

        indices = self._parse_initialization_response(result.content, n_cand)
        if not indices:
            return _fallback()

        # Pad with spread points if the LLM returned too few.
        if len(indices) < n_select:
            for i in _fallback():
                if i not in indices:
                    indices.append(i)
                if len(indices) >= n_select:
                    break
        return indices[:n_select]

    def _build_initialization_prompt(
        self,
        candidate_points: List[Dict[str, float]],
        feature_cols: List[str],
        target_name: str,
        n_select: int,
    ) -> str:
        # Determine task-specific physical context and semiconductor rules
        physical_context = ""
        feature_hints = {}
        
        if "CHI_PVK" in feature_cols and "Eg_HTL" in feature_cols:
            physical_context = (
                "This task involves optimizing energy band alignment in perovskite solar cells to maximize PCE.\n"
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
            physical_context = (
                "This task involves optimizing defects and doping to minimize non-radiative recombination and maximize carrier extraction.\n"
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

        n_cand = len(candidate_points)
        prompt = f"""You are a senior materials scientist specializing in device physics for Perovskite Solar Cells.
Your goal is to select {n_select} most promising initial points from {n_cand} candidate perovskite compositions for Bayesian optimization to maximize power conversion efficiency ({target_name}).

### Task-Specific Physical Context
{physical_context}

### Parameter Definitions"""
        for col in feature_cols:
            hint = feature_hints.get(col, "Feature parameter for material optimization")
            prompt += f"\n- {col}: {hint}"

        prompt += "\n\n### Candidate Points:"
        for i, point in enumerate(candidate_points):
            feats = ", ".join(f"{col}={point.get(col, 0):.3f}" for col in feature_cols)
            prompt += f"\nPoint {i + 1}: {feats}"
            
            # Explicitly calculate critical offsets
            if "CHI_PVK" in point and "CHI_ETL" in point:
                cbo = point["CHI_PVK"] - point["CHI_ETL"]
                prompt += f" | CBO: {cbo:.3f} eV"
                if not (-0.1 <= cbo <= 0.3):
                    prompt += " (OUTSIDE RANGE)"
            if all(k in point for k in ["CHI_HTL", "Eg_HTL", "CHI_PVK"]):
                vbo = (point["CHI_HTL"] + point["Eg_HTL"]) - point["CHI_PVK"]
                prompt += f" | VBO: {vbo:.3f} eV"

        prompt += f"""

### Selection Instructions
Based on the physical rules above, select exactly {n_select} points that would be most valuable as initial points. Consider both exploration (diversity) and exploitation (likely high performance based on physical constraints).

Respond with only the point numbers (1-{n_cand}) separated by commas, like: 1,5,12
"""
        return prompt

    def _parse_initialization_response(self, response: str, n_cand: int) -> List[int]:
        clean_response = response.strip()
        lines = [l.strip() for l in clean_response.splitlines() if l.strip()]
        if lines:
            last_line = lines[-1]
            nums = re.findall(r"\b(\d+)\b", last_line)
            if nums and len(nums) >= 2:
                indices = []
                for n in nums:
                    idx = int(n) - 1
                    if 0 <= idx < n_cand and idx not in indices:
                        indices.append(idx)
                if indices:
                    return indices

        nums = re.findall(r"\b(\d+)\b", clean_response)
        indices = []
        for n in nums:
            idx = int(n) - 1
            if 0 <= idx < n_cand and idx not in indices:
                indices.append(idx)
        return indices

    def build_system_prompt_for_viability(
        self,
        target_name: str,
        feature_cols: list[str],
        observed_data: list[tuple[dict[str, float], float]],
    ) -> str:
        """Construct a system prompt detailing physical rules and observations for pointwise viability queries."""
        physical_context = ""
        feature_hints = {}
        
        if "CHI_PVK" in feature_cols and "Eg_HTL" in feature_cols:
            physical_context = (
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
            physical_context = (
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

        prompt = f"""You are a senior materials scientist specializing in device physics for Perovskite Solar Cells.
Your goal is to evaluate if a candidate formulation is physically viable and likely to achieve high power conversion efficiency ({target_name}).

### Device Physics Rules
{physical_context}

### Parameter Definitions"""
        for col in feature_cols:
            hint = feature_hints.get(col, "Feature parameter")
            prompt += f"\n- {col}: {hint}"

        # Sort history to show top performing ones (up to 5)
        sorted_obs = sorted(observed_data, key=lambda x: x[1], reverse=True)[:5]
        if sorted_obs:
            prompt += "\n\n### Top Performing Observed Formulations (For reference)"
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
    ) -> float:
        """Query LLM for a single candidate's physical viability and return log_prob(Yes)."""
        if not self._client.is_configured():
            return 0.0

        user_prompt = "Candidate formulation:\n"
        for col in feature_cols:
            user_prompt += f"- {col}: {candidate.get(col, 0.0):.4f}\n"

        if "CHI_PVK" in candidate and "CHI_ETL" in candidate:
            cbo = candidate["CHI_PVK"] - candidate["CHI_ETL"]
            user_prompt += f"- Calculated CBO: {cbo:.4f} eV\n"
        if all(k in candidate for k in ["CHI_HTL", "Eg_HTL", "CHI_PVK"]):
            vbo = (candidate["CHI_HTL"] + candidate["Eg_HTL"]) - candidate["CHI_PVK"]
            user_prompt += f"- Calculated VBO: {vbo:.4f} eV\n"

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
