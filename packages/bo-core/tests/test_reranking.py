from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from bo_core.optimization.reranking import (
    ArtifactEvidenceGate,
    ChemGPShortlistAdapter,
    DeepSeekCandidateReranker,
    GateDecision,
    RankedCandidate,
    RerankingProposal,
    SelectCandidateUseCase,
    load_evidence_gate,
)


class _GP:
    def __init__(self) -> None:
        self.candidates = tuple(
            RankedCandidate(i, rank, 10.0 - rank, {"Ligand": f"L{i}"})
            for rank, i in enumerate((5, 4, 3, 2, 1), start=1)
        )

    def shortlist(self) -> tuple[RankedCandidate, ...]:
        return self.candidates


class _Gate:
    def __init__(self, enabled: bool, reason: str = "approved") -> None:
        self.decision = GateDecision(enabled, reason)

    def decide(self) -> GateDecision:
        return self.decision


class _Reranker:
    def __init__(self, order: tuple[int, ...] | Exception) -> None:
        self.order = order
        self.calls = 0

    def rerank(self, shortlist: tuple[RankedCandidate, ...]) -> RerankingProposal:
        self.calls += 1
        assert shortlist
        if isinstance(self.order, Exception):
            raise self.order
        return RerankingProposal(self.order, model="fake", prompt_version="v1")


def test_use_case_gate_and_reranker_failures_return_exact_gp_winner() -> None:
    gp = _GP()
    disabled = _Reranker((1, 2, 3, 4, 5))

    gated = SelectCandidateUseCase(gp, disabled, _Gate(False, "gate_failed")).execute()

    assert gated.selected == gp.candidates[0]
    assert gated.source == "gp"
    assert gated.gate_reason == "gate_failed"
    assert gated.fallback_reason is None
    assert disabled.calls == 0

    for order, reason in (
        ((5, 5, 3, 2, 1), "invalid_permutation"),
        ((5, 4, 3, 2), "invalid_permutation"),
        ((5, 4, 3, 2, 999), "invalid_permutation"),
        (ValueError("bad json"), "invalid_response"),
        (RuntimeError("down"), "llm_error"),
    ):
        reranker = _Reranker(order)
        result = SelectCandidateUseCase(gp, reranker, _Gate(True)).execute()
        assert result.selected == gp.candidates[0]
        assert result.source == "gp"
        assert result.fallback_reason == reason


def test_use_case_accepts_only_a_total_shortlist_permutation() -> None:
    gp = _GP()
    result = SelectCandidateUseCase(
        gp,
        _Reranker((1, 2, 3, 4, 5)),
        _Gate(True),
    ).execute()

    assert result.selected == gp.candidates[-1]
    assert result.source == "llm_reranked"
    assert result.gp_winner == gp.candidates[0]
    assert result.shortlist == gp.candidates
    assert result.fallback_reason is None


def test_use_case_rejects_an_empty_gp_shortlist() -> None:
    class EmptyGP:
        def shortlist(self) -> tuple[RankedCandidate, ...]:
            return ()

    with pytest.raises(ValueError, match="GP shortlist is empty"):
        SelectCandidateUseCase(
            EmptyGP(),
            _Reranker(()),
            _Gate(True),
        ).execute()


def test_deepseek_reranker_forces_one_tool_call_without_oracle_fields() -> None:
    class Client:
        model = "test-model"

        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def chat(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 2048,
            extra_body: dict[str, Any] | None = None,
            *,
            temperature: float = 0.0,
        ) -> SimpleNamespace:
            self.kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
                "temperature": temperature,
            }
            return SimpleNamespace(
                status="success",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "rank_shortlist",
                            "arguments": (
                                '{"ordered_ids":[4,3,2,1,0],'
                                '"confidence":[0.8,0.7,0.6,0.5,0.4]}'
                            ),
                        },
                    }
                ],
                content="",
                error=None,
            )

    client = Client()
    candidates = tuple(
        RankedCandidate(i, i + 1, float(5 - i), {"Ligand": f"L{i}"})
        for i in range(5)
    )

    proposal = DeepSeekCandidateReranker(client, "prompt-v1").rerank(candidates)

    assert proposal.ordered_ids == (4, 3, 2, 1, 0)
    assert proposal.model == "test-model"
    assert proposal.confidence == (0.8, 0.7, 0.6, 0.5, 0.4)
    payload = str(client.kwargs)
    assert "observed_yield" not in payload
    assert "pool_yield" not in payload
    assert "acquisition_score" in payload
    assert client.kwargs["temperature"] == 0.0
    extra_body = client.kwargs["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["tool_choice"]["function"]["name"] == "rank_shortlist"
    schema = extra_body["tools"][0]["function"]["parameters"]
    assert "confidence" in schema["properties"]


def test_deepseek_reranker_rejects_schema_invalid_ids() -> None:
    class Client:
        model = "test-model"

        def chat(self, *args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return SimpleNamespace(
                status="success",
                tool_calls=[
                    {
                        "function": {
                            "name": "rank_shortlist",
                            "arguments": '{"ordered_ids":[4.9,3.9,2.9,1.9,0.9]}',
                        }
                    }
                ],
            )

    candidates = tuple(
        RankedCandidate(i, i + 1, float(5 - i), {"Ligand": f"L{i}"})
        for i in range(5)
    )
    with pytest.raises(TypeError, match="ranking IDs"):
        DeepSeekCandidateReranker(Client(), "v1").rerank(candidates)


@pytest.mark.parametrize(
    "tool_call",
    [
        None,
        {"function": None},
        {
            "function": {
                "name": "rank_shortlist",
                "arguments": '{"ordered_ids":[4,3,2,1,0],"outcome":99}',
            }
        },
    ],
)
def test_deepseek_reranker_rejects_malformed_or_extra_tool_fields(
    tool_call: object,
) -> None:
    class Client:
        model = "test-model"

        def chat(self, *args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return SimpleNamespace(status="success", tool_calls=[tool_call])

    candidates = tuple(
        RankedCandidate(i, i + 1, float(5 - i), {"Ligand": f"L{i}"})
        for i in range(5)
    )
    with pytest.raises((TypeError, ValueError)):
        DeepSeekCandidateReranker(Client(), "v1").rerank(candidates)


def test_artifact_gate_requires_exact_passed_provenance() -> None:
    expected = {
        "source_sha256": "abc",
        "dataset": "suzuki",
        "model": "m",
        "prompt_version": "v1",
        "shortlist_size": 5,
    }
    artifact = {**expected, "passed": True}

    assert ArtifactEvidenceGate(artifact, expected).decide() == GateDecision(True, "approved")
    assert ArtifactEvidenceGate(
        {**artifact, "model": "other"}, expected
    ).decide() == GateDecision(False, "provenance_mismatch")
    assert ArtifactEvidenceGate(
        {**expected, "passed": False}, expected
    ).decide() == GateDecision(False, "evidence_failed")
    assert ArtifactEvidenceGate(None, expected).decide() == GateDecision(
        False, "evidence_missing"
    )


def test_load_evidence_gate_treats_missing_or_invalid_files_as_missing(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")

    assert load_evidence_gate(missing, {}).decide() == GateDecision(
        False, "evidence_missing"
    )
    assert load_evidence_gate(invalid, {}).decide() == GateDecision(
        False, "evidence_missing"
    )


def test_chem_gp_adapter_ranks_all_remaining_with_deterministic_ties() -> None:
    class Engine:
        M = 7
        queried: ClassVar[set[int]] = {6}
        feature_cols: ClassVar[list[str]] = ["Ligand"]
        y_obs: ClassVar[np.ndarray] = np.array([5.0])

        def __init__(self) -> None:
            import pandas as pd

            self.test_df = pd.DataFrame({"Ligand": [f"L{i}" for i in range(7)]})

        def _fit_gp(self):
            return object()

        def _predict_pool(self, _surrogate):
            return np.arange(7, dtype=float), np.ones(7)

        def _expected_improvement(self, _mean, _std, _best):
            return np.array([1.0, 3.0, 3.0, 2.0, 0.0, 4.0, 99.0])

    result = ChemGPShortlistAdapter(Engine()).shortlist()

    assert [candidate.pool_index for candidate in result] == [5, 1, 2, 3, 0]
    assert [candidate.gp_rank for candidate in result] == [1, 2, 3, 4, 5]
    assert all(len(candidate.features) == 1 for candidate in result)


def test_chem_gp_adapter_requires_exactly_five_finite_candidates() -> None:
    class Engine:
        M = 4
        queried: ClassVar[set[int]] = set()
        feature_cols: ClassVar[list[str]] = ["Ligand"]
        y_obs: ClassVar[np.ndarray] = np.array([5.0])

        def _fit_gp(self):
            return object()

        def _predict_pool(self, _surrogate):
            return np.arange(4, dtype=float), np.ones(4)

        def _expected_improvement(self, _mean, _std, _best):
            return np.arange(4, dtype=float)

    with pytest.raises(ValueError, match="fewer than five"):
        ChemGPShortlistAdapter(Engine()).shortlist()

    with pytest.raises(ValueError, match="shortlist size is fixed at 5"):
        ChemGPShortlistAdapter(Engine(), top_k=4)
