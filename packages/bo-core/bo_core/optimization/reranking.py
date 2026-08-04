from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class RankedCandidate:
    pool_index: int
    gp_rank: int
    acquisition_score: float
    features: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


@dataclass(frozen=True)
class RerankingProposal:
    ordered_ids: tuple[int, ...]
    model: str
    prompt_version: str
    confidence: tuple[float, ...] | None = None


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    selected: RankedCandidate
    source: str
    gp_winner: RankedCandidate
    shortlist: tuple[RankedCandidate, ...]
    gate_reason: str
    fallback_reason: str | None = None
    proposal: RerankingProposal | None = None


class GPShortlistPort(Protocol):
    def shortlist(self) -> tuple[RankedCandidate, ...]: ...


class CandidateRerankerPort(Protocol):
    def rerank(
        self, shortlist: tuple[RankedCandidate, ...]
    ) -> RerankingProposal: ...


class RerankingGatePort(Protocol):
    def decide(self) -> GateDecision: ...


class SelectCandidateUseCase:
    def __init__(
        self,
        gp: GPShortlistPort,
        reranker: CandidateRerankerPort,
        gate: RerankingGatePort,
    ) -> None:
        self._gp = gp
        self._reranker = reranker
        self._gate = gate

    def execute(self) -> SelectionResult:
        shortlist = self._gp.shortlist()
        if not shortlist:
            raise ValueError("GP shortlist is empty")
        gp_winner = shortlist[0]
        gate = self._gate.decide()
        if not gate.enabled:
            return SelectionResult(
                gp_winner, "gp", gp_winner, shortlist, gate.reason
            )

        try:
            proposal = self._reranker.rerank(shortlist)
        except (TypeError, ValueError, json.JSONDecodeError):
            return SelectionResult(
                gp_winner, "gp", gp_winner, shortlist, gate.reason, "invalid_response"
            )
        except Exception:  # noqa: BLE001 - strict GP fallback at adapter boundary
            return SelectionResult(
                gp_winner, "gp", gp_winner, shortlist, gate.reason, "llm_error"
            )

        ids = tuple(candidate.pool_index for candidate in shortlist)
        if len(proposal.ordered_ids) != len(ids) or set(proposal.ordered_ids) != set(ids):
            return SelectionResult(
                gp_winner,
                "gp",
                gp_winner,
                shortlist,
                gate.reason,
                "invalid_permutation",
            )
        selected = next(
            candidate
            for candidate in shortlist
            if candidate.pool_index == proposal.ordered_ids[0]
        )
        return SelectionResult(
            selected,
            "llm_reranked",
            gp_winner,
            shortlist,
            gate.reason,
            proposal=proposal,
        )


class ChemGPShortlistAdapter:
    def __init__(self, engine: Any, *, top_k: int = 5) -> None:
        if top_k != 5:
            raise ValueError("reranking shortlist size is fixed at 5")
        self._engine = engine

    def shortlist(self) -> tuple[RankedCandidate, ...]:
        engine = self._engine
        surrogate = engine._fit_gp()
        mean, std = engine._predict_pool(surrogate)
        acquisition = np.asarray(
            engine._expected_improvement(mean, std, float(np.max(engine.y_obs))),
            dtype=float,
        )
        remaining = np.ones(engine.M, dtype=bool)
        if engine.queried:
            remaining[list(engine.queried)] = False
        eligible = np.flatnonzero(remaining & np.isfinite(acquisition))
        if eligible.size < 5:
            raise ValueError("fewer than five finite remaining GP acquisitions")
        ordered = eligible[np.lexsort((eligible, -acquisition[eligible]))][:5]
        return tuple(
            RankedCandidate(
                pool_index=int(pool_index),
                gp_rank=rank,
                acquisition_score=float(acquisition[pool_index]),
                features={
                    field: str(engine.test_df[field].iloc[pool_index])
                    for field in engine.feature_cols
                },
            )
            for rank, pool_index in enumerate(ordered, start=1)
        )


class DeepSeekCandidateReranker:
    def __init__(self, client: Any, prompt_version: str) -> None:
        self._client = client
        self._prompt_version = prompt_version

    def rerank(
        self, shortlist: tuple[RankedCandidate, ...]
    ) -> RerankingProposal:
        if not shortlist:
            raise ValueError("shortlist is empty")
        tool_name = "rank_shortlist"
        candidates = [
            {
                "id": candidate.pool_index,
                "gp_rank": candidate.gp_rank,
                "acquisition_score": candidate.acquisition_score,
                "features": dict(candidate.features),
            }
            for candidate in shortlist
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Return every supplied candidate ID exactly once, best first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ordered_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": len(shortlist),
                                "maxItems": len(shortlist),
                            },
                            "confidence": {
                                "type": "array",
                                "items": {"type": "number", "minimum": 0, "maximum": 1},
                                "minItems": len(shortlist),
                                "maxItems": len(shortlist),
                            },
                        },
                        "required": ["ordered_ids"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        result = self._client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rank only the supplied complete candidates. "
                        "Do not add, omit, or alter candidates."
                    ),
                },
                {"role": "user", "content": json.dumps({"candidates": candidates})},
            ],
            max_tokens=256,
            extra_body={
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": tool_name}},
            },
            temperature=0.0,
        )
        if getattr(result, "status", None) != "success":
            raise RuntimeError("LLM ranking call failed")
        tool_calls = getattr(result, "tool_calls", None)
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise ValueError("expected one ranking tool call")
        tool_call = tool_calls[0]
        if not isinstance(tool_call, Mapping):
            raise TypeError("invalid ranking tool call")
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            raise TypeError("invalid ranking tool function")
        if function.get("name") != tool_name:
            raise ValueError("unexpected ranking tool")
        arguments = function.get("arguments")
        try:
            payload = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            raise ValueError("invalid ranking arguments") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("ordered_ids"), list):
            raise TypeError("invalid ranking arguments")
        if set(payload) not in ({"ordered_ids"}, {"ordered_ids", "confidence"}):
            raise TypeError("unexpected ranking arguments")
        ordered_ids = payload["ordered_ids"]
        if any(type(value) is not int for value in ordered_ids):
            raise TypeError("invalid ranking IDs")
        confidence = payload.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, list)
            or len(confidence) != len(shortlist)
            or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in confidence)
        ):
            raise TypeError("invalid ranking confidence")
        return RerankingProposal(
            ordered_ids=tuple(ordered_ids),
            model=str(getattr(self._client, "model", "unknown")),
            prompt_version=self._prompt_version,
            confidence=tuple(float(value) for value in confidence) if confidence is not None else None,
        )


class ArtifactEvidenceGate:
    def __init__(
        self,
        artifact: Mapping[str, Any] | None,
        expected: Mapping[str, Any],
    ) -> None:
        self._artifact = artifact
        self._expected = dict(expected)

    def decide(self) -> GateDecision:
        if self._artifact is None:
            return GateDecision(False, "evidence_missing")
        if any(self._artifact.get(key) != value for key, value in self._expected.items()):
            return GateDecision(False, "provenance_mismatch")
        if self._artifact.get("passed") is not True:
            return GateDecision(False, "evidence_failed")
        return GateDecision(True, "approved")


def load_evidence_gate(path: Any, expected: Mapping[str, Any]) -> ArtifactEvidenceGate:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        artifact = None
    return ArtifactEvidenceGate(artifact, expected)
