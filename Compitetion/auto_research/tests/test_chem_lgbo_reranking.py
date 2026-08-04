"""Contract tests for evidence-gated Chem-LGBO shortlist reranking."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import chem_lgbo_reranking as experiment
from chem_lgbo_reranking import (
    _gate_verdict,
    _matched_random_seed,
    evaluate_records,
    main,
)
from test_chem_lgbo_prompt_ablation import _real_record


class _ScriptedClient:
    model = "scripted-model"

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        extra_body: dict[str, object] | None = None,
        *,
        temperature: float = 0.0,
    ) -> SimpleNamespace:
        del max_tokens, extra_body, temperature
        self.payloads.append(messages[1]["content"])
        candidates = json.loads(messages[1]["content"])["candidates"]
        order = [candidate["id"] for candidate in reversed(candidates)]
        return SimpleNamespace(
            status="success",
            content="",
            error=None,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "rank_shortlist",
                        "arguments": json.dumps(
                            {"ordered_ids": order, "confidence": [0.8] * len(order)}
                        ),
                    },
                }
            ],
        )


def _states(tmp_path: Path) -> Path:
    path = tmp_path / "states.json"
    path.write_text(json.dumps({"records": [_real_record()]}), encoding="utf-8")
    return path


def test_offline_evaluator_reports_dataset_seed_metrics_and_matched_random(
    tmp_path: Path,
) -> None:
    client = _ScriptedClient()
    artifact = evaluate_records(
        _states(tmp_path),
        tmp_path / "run.json",
        state_keys=[("suzuki", 100, 2)],
        client_factory=lambda: client,
        model="scripted-model",
        prompt_version="v1",
    )

    record = artifact["records"][0]
    assert len(record["shortlist_indices"]) == 5
    assert record["selected_index"] in record["shortlist_indices"]
    assert record["selection_source"] == "llm_reranked"
    assert all("yield" not in payload.lower() for payload in client.payloads)
    metrics = artifact["analysis"]["suzuki"]["seeds"]["100"]
    assert {
        "ranking_accuracy", "pairwise_accuracy", "delta_vs_gp",
        "delta_vs_random", "brier_score", "failure_rate",
    } <= metrics.keys()
    assert metrics["brier_score"] is not None
    assert artifact["analysis"]["suzuki"]["seed_count"] == 1
    assert artifact["passed"] is False
    assert "missing datasets: buchwald_sub4" in artifact["gate"]["reasons"]


def test_gate_requires_full_paired_evidence_and_quality() -> None:
    strong_seed = {
        "state_count": 1,
        "steps": [2],
        "confidence_coverage": 1.0,
        "ranking_accuracy": 1.0,
        "pairwise_accuracy": 1.0,
        "delta_vs_gp": 1.0,
        "delta_vs_random": 1.0,
        "brier_score": 0.04,
        "failure_rate": 0.0,
    }
    analysis = {
        dataset: {
            "seeds": {seed: dict(strong_seed) for seed in ("100", "200")},
            "seed_count": 2,
            "state_manifest": [("100", 2), ("200", 2)],
            "delta_vs_gp_lcb": 1.0,
            "delta_vs_random_lcb": 1.0,
            "mean_ranking_accuracy": 1.0,
            "mean_pairwise_accuracy": 1.0,
            "mean_brier_score": 0.04,
            "confidence_coverage": 1.0,
            "failure_rate": 0.0,
        }
        for dataset in ("buchwald_sub4", "suzuki")
    }

    verdict = _gate_verdict(analysis)

    assert verdict["passed"] is False
    assert any("must cover seeds" in reason for reason in verdict["reasons"])


def test_matched_random_seed_uses_complete_state_key() -> None:
    assert _matched_random_seed("suzuki", 100, 2) != _matched_random_seed(
        "buchwald_sub4", 100, 2
    )
    assert _matched_random_seed("suzuki", 100, 2) != _matched_random_seed(
        "suzuki", 99, 3
    )

def test_gate_rejects_step_mismatch_and_incomplete_confidence() -> None:
    seeds = {
        seed: {
            "steps": [2],
            "confidence_coverage": 1.0,
            "ranking_accuracy": 1.0,
            "pairwise_accuracy": 1.0,
            "delta_vs_gp": 1.0,
            "delta_vs_random": 1.0,
            "brier_score": 0.04,
            "failure_rate": 0.0,
        }
        for seed in ("100", "200", "300", "400", "500")
    }
    base = {
        "seeds": seeds,
        "seed_count": 5,
        "delta_vs_gp_lcb": 1.0,
        "delta_vs_random_lcb": 1.0,
        "mean_ranking_accuracy": 1.0,
        "mean_pairwise_accuracy": 1.0,
        "mean_brier_score": 0.04,
        "confidence_coverage": 1.0,
        "failure_rate": 0.0,
    }
    analysis = {
        "suzuki": {**base, "state_manifest": [(seed, 2) for seed in seeds]},
        "buchwald_sub4": {
            **base,
            "state_manifest": [(seed, 3) for seed in seeds],
            "confidence_coverage": 0.8,
        },
    }

    verdict = _gate_verdict(analysis)

    assert verdict["passed"] is False
    assert "target datasets must use the same paired seed/step manifest" in verdict["reasons"]
    assert "buchwald_sub4 confidence coverage is incomplete" in verdict["reasons"]


def test_gate_rejects_nonfinite_metrics() -> None:
    seeds = {
        seed: {
            "steps": [2],
            "confidence_coverage": 1.0,
            "ranking_accuracy": 1.0,
            "pairwise_accuracy": 1.0,
            "delta_vs_gp": 1.0,
            "delta_vs_random": 1.0,
            "brier_score": 0.04,
            "failure_rate": 0.0,
        }
        for seed in ("100", "200", "300", "400", "500")
    }
    base = {
        "seeds": seeds,
        "seed_count": 5,
        "state_manifest": [(seed, 2) for seed in seeds],
        "delta_vs_gp_lcb": float("nan"),
        "delta_vs_random_lcb": 1.0,
        "mean_ranking_accuracy": 1.0,
        "mean_pairwise_accuracy": 1.0,
        "mean_brier_score": float("nan"),
        "confidence_coverage": 1.0,
        "failure_rate": 0.0,
    }

    verdict = _gate_verdict({dataset: dict(base) for dataset in ("suzuki", "buchwald_sub4")})

    assert verdict["passed"] is False
    assert all("non-finite metrics" in reason for reason in verdict["reasons"])


def test_report_and_preflight_never_construct_llm_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = _states(tmp_path)
    output = tmp_path / "run.json"
    report = tmp_path / "report.json"
    expected = {
        "source_sha256": experiment._sha256(states),
        "model": "scripted-model",
        "prompt_version": "v1",
        "shortlist_size": 5,
        "state_manifest": [["suzuki", 100, 2]],
    }
    output.write_text(
        json.dumps({**expected, "passed": False, "records": [], "analysis": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiment.DeepSeekClient,
        "from_env",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be constructed")),
    )

    assert main(["preflight", "--states", str(states), "--model", "scripted-model", "--prompt-version", "v1"]) == 0
    assert main(["report", "--artifact", str(output), "--output", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False


def test_evaluator_rejects_duplicate_states_and_provenance_drift(tmp_path: Path) -> None:
    states = _states(tmp_path)
    output = tmp_path / "run.json"

    with pytest.raises(ValueError, match="duplicate state"):
        evaluate_records(
            states,
            output,
            state_keys=[("suzuki", 100, 2), ("suzuki", 100, 2)],
            client_factory=lambda: _ScriptedClient(),
            model="scripted-model",
            prompt_version="v1",
        )

    output.write_text(json.dumps({"source_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        evaluate_records(
            states,
            output,
            state_keys=[("suzuki", 100, 2)],
            client_factory=lambda: _ScriptedClient(),
            model="scripted-model",
            prompt_version="v1",
        )


def test_evaluator_rejects_model_provenance_mismatch(tmp_path: Path) -> None:
    client = _ScriptedClient()
    client.model = "actual-model"

    with pytest.raises(ValueError, match="client model"):
        evaluate_records(
            _states(tmp_path),
            tmp_path / "run.json",
            state_keys=[("suzuki", 100, 2)],
            client_factory=lambda: client,
            model="claimed-model",
            prompt_version="v1",
        )
