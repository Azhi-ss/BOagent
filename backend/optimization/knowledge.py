from __future__ import annotations
import os
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from llm_client import DeepSeekClient, LlmCallResult

@dataclass
class SuggestionResult:
    suggestions: List[Dict[str, float]]
    analysis: str
    prompt: str

class KnowledgeEngine:
    """Module responsible for materials science domain reasoning via LLM."""

    def __init__(self, chat_engine: str = "deepseek-v4-flash"):
        self.chat_engine = chat_engine
        self._client = DeepSeekClient.from_env()
        if chat_engine:
            self._client.model = chat_engine

    def build_prompt(
        self,
        target_name: str,
        feature_cols: List[str],
        top_candidates: pd.DataFrame,
        observed_data: List[Tuple[Dict[str, float], float]]
    ) -> str:
        """Construct the materials-science domain prompt."""
        prompt = f"""You are a professional materials scientist specializing in perovskite solar cell optimization. I will provide you with multiple candidate formulations, some observed formulation data, and the prediction results from a Gaussian process model. Please select the formulation from these that is most likely to produce high power conversion efficiency ({target_name}).

Known Information:
The efficiency ({target_name}) of perovskite solar cells is closely related to the material characteristics.

Key parameters include:"""
        for col in feature_cols:
            prompt += f"\n{col}: Feature parameter for material optimization"

        prompt += "\n\nObserved formulations and their efficiencies:"
        for i, (obs_values, obs_eta) in enumerate(observed_data):
            prompt += f"\n formulation{i + 1}: "
            for col in feature_cols:
                prompt += f"{col}={obs_values.get(col, 0):.4f}, "
            prompt += f"{target_name}={obs_eta:.4f}"

        prompt += "\n\nCandidate formulations (top candidates from GP model):"
        for i, (_, row) in enumerate(top_candidates.iterrows()):
            prompt += f"\n\nCandidate formulation {i + 1}:\n"
            for col in feature_cols:
                prompt += f"{col}={row[col]:.4f}, "
            prompt += f"\nGP Prediction: mean={row.get('mean', 0):.4f}, std={row.get('std', 0):.4f}"

        prompt += f"""

Based on materials science principles, the observed data, and the GP model predictions, please select the 5 formulations from the candidates above that are most likely to produce high {target_name}. Considerations should include:
1. Similarity to known high-efficiency formulations
2. The likely efficiency of charge separation and transport
3. The potential for recombination losses
4. Novelty or innovation compared to observed formulations

Please respond in the following format:
Analysis: [Your detailed analysis]
Selected Formulations: [List of formulation numbers, e.g., 1, 3, 5, 7, 9]
"""
        return prompt

    def refine_suggestions(
        self,
        prompt: str,
        top_candidates: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[List[Dict[str, float]], str]:
        """Call LLM and parse the refined suggestions."""
        if not self._client.is_configured():
            # Fallback if no LLM
            return [row[feature_cols].to_dict() for _, row in top_candidates.head(5).iterrows()], "LLM not configured. Using GP top candidates."

        result: LlmCallResult = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a professional materials scientist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            extra_body={"thinking": {"type": "disabled"}}
        )

        if result.status != "success":
            return [row[feature_cols].to_dict() for _, row in top_candidates.head(5).iterrows()], f"LLM Error: {result.error}"

        analysis = result.content
        selected_indices = self._parse_response(analysis, len(top_candidates))
        
        suggestions = []
        for idx in selected_indices:
            suggestions.append(top_candidates.iloc[idx][feature_cols].to_dict())
            
        return suggestions, analysis

    def _parse_response(self, response: str, num_candidates: int) -> List[int]:
        match = re.search(r"Selected\s+Formulations?\s*:\s*(.+)", response, re.IGNORECASE)
        if match:
            numbers_text = match.group(1).strip()
            indices = [int(x.strip()) - 1 for x in re.split(r"[,\s]+", numbers_text) if x.strip().isdigit()]
            return [i for i in indices if 0 <= i < num_candidates]
        return list(range(min(5, num_candidates)))

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
        n_cand = len(candidate_points)
        prompt = f"""You are an expert in perovskite solar cell optimization. I need you to select {n_select} most promising initial points from {n_cand} candidate perovskite compositions for Bayesian optimization.

Optimization goal: Maximize power conversion efficiency ({target_name})

Key principles for selection:
1. Diversity: Choose points that span different regions of the parameter space
2. Performance: Prefer compositions likely to yield high efficiency on physical grounds
3. Balance: Consider trade-offs between band gaps and electronegativity values
4. Physical insight: Select compositions that represent different design strategies

Candidate points:"""
        for i, point in enumerate(candidate_points):
            feats = ", ".join(f"{col}={point.get(col, 0):.3f}" for col in feature_cols)
            prompt += f"\nPoint {i + 1}: {feats}"

        prompt += f"""

Please select exactly {n_select} points that would be most valuable as initial points for Bayesian optimization. Consider both exploration (diversity) and exploitation (likely high performance).

Respond with only the point numbers (1-{n_cand}) separated by commas, like: 1,5,12,23,45
"""
        return prompt

    def _parse_initialization_response(self, response: str, n_cand: int) -> List[int]:
        nums = re.findall(r"\d+", response)
        indices = []
        for n in nums:
            idx = int(n) - 1
            if 0 <= idx < n_cand and idx not in indices:
                indices.append(idx)
        return indices
