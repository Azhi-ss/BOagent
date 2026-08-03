"""Contract tests for the Chem-LGBO prompt-feedback paired ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import chem_lgbo_prompt_ablation as ablation_module
from chem_lgbo_prompt_ablation import (
    PromptAblationExperiment,
    aggregate_paired_results,
    classify_previous_outcome,
    evaluate_state,
    main,
    phase_for_step,
    pre_screen_passes,
    replay_engine_to_step,
    same_mask,
    select_pre_screen_states,
)


def _row(
    step: int,
    *,
    status: str = "applied",
    selected_in: bool | None = True,
    observed_yield: float = 10.0,
) -> dict[str, object]:
    return {
        "step": step,
        "guidance_status": status,
        "selected_in_subspace": selected_in,
        "observed_yield": observed_yield,
    }


@pytest.mark.parametrize(
    ("step", "phase"),
    [(1, "1-10"), (10, "1-10"), (11, "11-20"), (20, "11-20"),
     (21, "21-30"), (30, "21-30"), (31, "31-40"), (40, "31-40")],
)
def test_phase_for_step_uses_fixed_ten_round_bins(step: int, phase: str) -> None:
    assert phase_for_step(step) == phase


@pytest.mark.parametrize("step", [0, 41])
def test_phase_for_step_rejects_out_of_protocol_steps(step: int) -> None:
    with pytest.raises(ValueError, match="step"):
        phase_for_step(step)


def test_previous_outcome_strata_preserve_untested_and_improved_semantics() -> None:
    assert classify_previous_outcome(_row(1, status="fallback"), 20.0) == "fallback"
    assert classify_previous_outcome(_row(1, selected_in=False), 20.0) == "selected_outside"
    assert classify_previous_outcome(_row(1, observed_yield=21.0), 20.0) == "tested_improved"
    assert classify_previous_outcome(_row(1, observed_yield=20.0), 20.0) == "tested_nonimproving"


def test_same_mask_uses_boolean_candidate_membership_not_json_order() -> None:
    left = np.array([True, False, True])
    right = np.array([True, False, True])

    assert same_mask(left, right) is True
    assert same_mask(left, np.array([False, True, True])) is False
    with pytest.raises(ValueError, match="shape"):
        same_mask(left, np.array([True]))


def test_pre_screen_state_selection_is_deterministic_and_stratified() -> None:
    records = []
    for dataset in ("buchwald_sub4", "suzuki"):
        for seed in (100, 200):
            trajectory = []
            incumbent = 20.0
            for step in range(1, 41):
                if step in {5, 15, 25, 35}:
                    row = _row(step, observed_yield=incumbent + 1.0)
                    incumbent += 1.0
                elif step in {6, 16, 26, 36}:
                    row = _row(step, selected_in=False, observed_yield=5.0)
                else:
                    row = _row(step, observed_yield=5.0)
                trajectory.append(row)
            records.append({"dataset": dataset, "seed": seed, "trajectory": trajectory})

    first = select_pre_screen_states(records, per_phase=2, per_extra_stratum=2)
    second = select_pre_screen_states(records, per_phase=2, per_extra_stratum=2)

    assert first == second
    assert len(first) == len(set(first))
    assert all(len(state) == 3 for state in first)
    selected = set(first)
    for dataset in ("buchwald_sub4", "suzuki"):
        for seed in (100, 200):
            for phase_start in (2, 12, 22, 32):
                assert sum(
                    dataset_name == dataset
                    and state_seed == seed
                    and phase_for_step(step) == phase_for_step(phase_start)
                    for dataset_name, state_seed, step in selected
                ) >= 2


def test_pre_screen_selection_classifies_against_fixed_prior_incumbent() -> None:
    record = {
        "dataset": "buchwald_sub4",
        "seed": 100,
        "prior_best": 50.0,
        "trajectory": [
            _row(1, observed_yield=40.0),
            _row(2, observed_yield=60.0),
            _row(3, observed_yield=30.0),
        ],
    }

    selected = select_pre_screen_states(
        [record], per_phase=1, per_extra_stratum=1
    )

    assert selected == [
        ("buchwald_sub4", 100, 2),
        ("buchwald_sub4", 100, 3),
    ]


def test_aggregate_paired_results_averages_states_within_seed_first() -> None:
    records = [
        {"dataset": "buchwald_sub4", "seed": 100, "variant": "control", "selected_yield": 10.0, "gp_yield": 9.0, "fallback": False, "coverage": 0.1},
        {"dataset": "buchwald_sub4", "seed": 100, "variant": "treatment", "selected_yield": 14.0, "gp_yield": 9.0, "fallback": False, "coverage": 0.1},
        {"dataset": "buchwald_sub4", "seed": 100, "variant": "control", "selected_yield": 30.0, "gp_yield": 25.0, "fallback": False, "coverage": 0.2},
        {"dataset": "buchwald_sub4", "seed": 100, "variant": "treatment", "selected_yield": 28.0, "gp_yield": 25.0, "fallback": False, "coverage": 0.2},
        {"dataset": "buchwald_sub4", "seed": 200, "variant": "control", "selected_yield": 20.0, "gp_yield": 19.0, "fallback": False, "coverage": 0.1},
        {"dataset": "buchwald_sub4", "seed": 200, "variant": "treatment", "selected_yield": 24.0, "gp_yield": 19.0, "fallback": False, "coverage": 0.1},
    ]

    summary = aggregate_paired_results(records)
    buchwald = summary["buchwald_sub4"]

    assert buchwald["seed_deltas"] == {"100": 1.0, "200": 4.0}
    assert buchwald["mean_delta"] == pytest.approx(2.5)
    assert buchwald["treatment_vs_gp"] == pytest.approx(4.5)
    assert buchwald["control_vs_gp"] == pytest.approx(2.0)


def test_pre_screen_gate_requires_buchwald_gain_and_no_suzuki_harm() -> None:
    passing = {
        "buchwald_sub4": {
            "mean_delta": 1.0,
            "treatment_vs_gp": -1.0,
            "control_vs_gp": -5.0,
            "control_fallback_rate": 0.02,
            "treatment_fallback_rate": 0.02,
            "treatment_mean_coverage": 0.2,
        },
        "suzuki": {
            "mean_delta": 0.1,
            "treatment_vs_gp": 2.0,
            "control_vs_gp": 2.0,
            "control_fallback_rate": 0.02,
            "treatment_fallback_rate": 0.02,
            "treatment_mean_coverage": 0.1,
        },
    }

    assert pre_screen_passes(passing) == {"passed": True, "reasons": []}

    failing = {dataset: values.copy() for dataset, values in passing.items()}
    failing["suzuki"]["mean_delta"] = -1.0
    result = pre_screen_passes(failing)
    assert result["passed"] is False
    assert "suzuki treatment harms control" in result["reasons"]

    failing = {dataset: values.copy() for dataset, values in passing.items()}
    failing["buchwald_sub4"]["treatment_vs_gp"] = -5.0
    result = pre_screen_passes(failing)
    assert "buchwald treatment does not reduce GP loss" in result["reasons"]


class _Client:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[list[dict[str, Any]]] = []

    def is_configured(self) -> bool:
        return True

    def chat(
        self, messages: list[dict[str, Any]], **_kwargs: object
    ) -> SimpleNamespace:
        self.messages.append(messages)
        return SimpleNamespace(
            status="success",
            content="",
            error=None,
            usage={},
            tool_calls=[
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "propose_sparse_subspace",
                        "arguments": self.content,
                    },
                }
            ],
        )


def _real_record() -> dict[str, Any]:
    from bo_core.optimization.chem_lgbo import ChemLGBOEngine

    engine = ChemLGBOEngine(
        "suzuki",
        seed=100,
        use_llm=False,
        n_iters=0,
        n_restarts=0,
        backend="sklearn",
    )
    engine.use_llm = True
    ligand = str(engine.test_df["Ligand"].iloc[0])
    client = _Client(json.dumps({"subspace": {"Ligand": [ligand]}}))
    engine._client = client
    engine.step()
    engine.step()
    return {
        "composition": "chem_lgbo",
        "dataset": "suzuki",
        "seed": 100,
        "backend": "sklearn",
        "trajectory": list(engine.trajectory),
        "guidance_artifacts": list(engine.guidance_artifacts),
    }


def test_replay_engine_reconstructs_state_before_target_step() -> None:
    record = _real_record()

    engine = replay_engine_to_step(record, 2)

    first = record["trajectory"][0]
    assert engine.iteration == 1
    assert engine.queried == {first["query_index"]}
    assert engine.trajectory == [first]
    assert engine.y_obs[-1] == pytest.approx(first["observed_yield"])
    assert engine.previous_outcome is not None
    assert engine.previous_outcome.proposed_subspace == first["subspace"]
    assert engine.previous_outcome.incumbent_before == pytest.approx(
        max(engine.y_obs[:-1])
    )


def test_evaluate_state_uses_identical_gp_state_for_both_prompt_variants() -> None:
    record = _real_record()
    ligand = record["trajectory"][0]["subspace"]["Ligand"][0]
    clients: dict[str, _Client] = {}

    def client_factory(variant: str) -> _Client:
        client = _Client(json.dumps({"subspace": {"Ligand": [ligand]}}))
        clients[variant] = client
        return client

    control = evaluate_state(record, 2, "control", client_factory("control"))
    treatment = evaluate_state(
        record, 2, "treatment", client_factory("treatment")
    )

    assert control["state_key"] == treatment["state_key"]
    assert control["gp_index"] == treatment["gp_index"]
    assert control["gp_yield"] == treatment["gp_yield"]
    assert control["posterior_hash"] == treatment["posterior_hash"]
    assert "[Previous guidance outcome]" not in clients["control"].messages[0][1]["content"]
    assert "[Previous guidance outcome]" in clients["treatment"].messages[0][1]["content"]
    assert treatment["variant"] == "treatment"
    assert isinstance(treatment["selected_yield"], float)


def test_prompt_ablation_experiment_persists_and_resumes_variant_records(
    tmp_path: Path,
) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    output = tmp_path / "ablation.json"
    calls: list[str] = []

    def client_factory(variant: str, _record: dict[str, Any], _step: int) -> _Client:
        calls.append(variant)
        ligand = record["trajectory"][0]["subspace"]["Ligand"][0]
        return _Client(json.dumps({"subspace": {"Ligand": [ligand]}}))

    experiment = PromptAblationExperiment(
        source,
        output,
        client_factory=client_factory,
        state_keys=[("suzuki", 100, 2)],
    )

    first = experiment.run()
    second = experiment.run()

    assert calls == ["control", "treatment"] or calls == ["treatment", "control"]
    assert first == second
    assert len(first["records"]) == 2
    assert first["analysis"]["suzuki"]["seed_deltas"] == {"100": 0.0}
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_replay_uses_fixed_ablation_counterfactual_count() -> None:
    engine = replay_engine_to_step(_real_record(), 2)

    assert engine.n_counterfactuals == 100


def test_experiment_rejects_duplicate_source_dataset_seed(tmp_path: Path) -> None:
    record = _real_record()
    duplicate = json.loads(json.dumps(record))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"records": [record, duplicate]}), encoding="utf-8"
    )

    experiment = PromptAblationExperiment(
        source,
        tmp_path / "ablation.json",
        client_factory=lambda _variant, _record, _step: None,
        state_keys=[("suzuki", 100, 2)],
    )

    with pytest.raises(ValueError, match="duplicate source record"):
        experiment.run()


def test_evaluate_state_records_counterfactual_and_repeat_diagnostics() -> None:
    record = _real_record()
    ligand = record["trajectory"][0]["subspace"]["Ligand"][0]
    client = _Client(json.dumps({"subspace": {"Ligand": [ligand]}}))

    result = evaluate_state(record, 2, "treatment", client)

    assert result["previous_outcome_stratum"] in {
        "tested_improved",
        "tested_nonimproving",
        "selected_outside",
    }
    assert result["same_previous_mask"] is True
    assert result["counterfactual_count"] > 0
    assert 0.0 <= result["counterfactual_percentile"] <= 1.0
    assert isinstance(result["improved_incumbent"], bool)


def test_experiment_persists_manifest_provenance_model_and_gate(
    tmp_path: Path,
) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "config": {
                    "chat_engine": "test-model",
                    "llm_max_tokens": 123,
                    "reasoning_effort": "low",
                    "temperature": 0.2,
                    "response_mode": "tool_call_react",
                    "max_react_retries": 1,
                },
                "records": [record],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "ablation.json"

    def client_factory(
        _variant: str, _record: dict[str, Any], _step: int
    ) -> _Client:
        ligand = record["trajectory"][0]["subspace"]["Ligand"][0]
        return _Client(json.dumps({"subspace": {"Ligand": [ligand]}}))

    artifact = PromptAblationExperiment(
        source,
        output,
        client_factory=client_factory,
        state_keys=[("suzuki", 100, 2)],
    ).run()

    assert artifact["state_manifest"] == [["suzuki", 100, 2]]
    assert artifact["model_config"] == {
        "chat_engine": "test-model",
        "llm_max_tokens": 123,
        "reasoning_effort": "low",
        "temperature": 0.2,
        "response_mode": "tool_call_react",
        "max_react_retries": 1,
    }
    assert artifact["provenance"]["source_artifact_sha256"] == artifact[
        "source_sha256"
    ]
    assert artifact["provenance"]["sources"]
    assert artifact["gate"] == {
        "passed": False,
        "reasons": ["both datasets are required"],
    }


def test_legacy_ablation_artifact_without_tool_protocol_remains_readable(
    tmp_path: Path,
) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    output = tmp_path / "ablation.json"
    experiment = PromptAblationExperiment(
        source,
        output,
        client_factory=lambda _variant, _record, _step: None,
        state_keys=[("suzuki", 100, 2)],
    )
    legacy = experiment._load()
    legacy["model_config"] = {"temperature": 0.0}
    output.write_text(json.dumps(legacy), encoding="utf-8")

    assert experiment._load()["model_config"] == {"temperature": 0.0}


def test_experiment_rejects_changed_state_manifest_on_resume(tmp_path: Path) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    output = tmp_path / "ablation.json"

    def client_factory(
        _variant: str, _record: dict[str, Any], _step: int
    ) -> _Client:
        ligand = record["trajectory"][0]["subspace"]["Ligand"][0]
        return _Client(json.dumps({"subspace": {"Ligand": [ligand]}}))

    PromptAblationExperiment(
        source,
        output,
        client_factory=client_factory,
        state_keys=[("suzuki", 100, 2)],
    ).run()

    with pytest.raises(ValueError, match="state manifest"):
        PromptAblationExperiment(
            source,
            output,
            client_factory=client_factory,
            state_keys=[("suzuki", 100, 1)],
        ).run()


def test_cli_preflight_reads_source_without_llm_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    output = tmp_path / "ablation.json"

    assert main(["preflight", "--source", str(source), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state_count"] > 0
    assert payload["source_sha256"]
    assert not output.exists()


def test_aggregate_reports_counterfactual_repeat_and_incumbent_metrics() -> None:
    records = []
    for variant, selected_yield, repeat, percentile, improved in (
        ("control", 10.0, True, 0.4, False),
        ("treatment", 12.0, False, 0.7, True),
    ):
        records.append(
            {
                "state_key": "suzuki:100:2",
                "dataset": "suzuki",
                "seed": 100,
                "variant": variant,
                "selected_yield": selected_yield,
                "gp_yield": 9.0,
                "fallback": False,
                "coverage": 0.1,
                "previous_outcome_stratum": "tested_nonimproving",
                "same_previous_mask": repeat,
                "counterfactual_percentile": percentile,
                "improved_incumbent": improved,
            }
        )

    summary = aggregate_paired_results(records)["suzuki"]

    assert summary["control_counterfactual_percentile"] == pytest.approx(0.4)
    assert summary["treatment_counterfactual_percentile"] == pytest.approx(0.7)
    assert summary["control_repeat_rate"] == pytest.approx(1.0)
    assert summary["treatment_repeat_rate"] == pytest.approx(0.0)
    assert summary["control_incumbent_rate"] == pytest.approx(0.0)
    assert summary["treatment_incumbent_rate"] == pytest.approx(1.0)
    assert summary["repeat_by_stratum"]["tested_nonimproving"] == {
        "control": 1.0,
        "treatment": 0.0,
    }


def test_cli_run_and_report_use_environment_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    output = tmp_path / "ablation.json"
    ligand = record["trajectory"][0]["subspace"]["Ligand"][0]

    class EnvironmentClient(_Client):
        model = "environment-model"

        @classmethod
        def from_env(cls) -> EnvironmentClient:
            return cls(json.dumps({"subspace": {"Ligand": [ligand]}}))

    monkeypatch.setattr(ablation_module, "DeepSeekClient", EnvironmentClient)
    monkeypatch.setattr(
        ablation_module,
        "select_pre_screen_states",
        lambda _records: [("suzuki", 100, 2)],
    )

    assert main(["run", "--source", str(source), "--output", str(output)]) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["record_count"] == 2
    assert run_payload["model_config"]["chat_engine"] == "environment-model"

    assert main(["report", "--source", str(source), "--output", str(output)]) == 0
    report_payload = json.loads(capsys.readouterr().out)
    assert report_payload == run_payload




def test_cli_report_recomputes_partial_artifact_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _real_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    output = tmp_path / "ablation.json"
    ligand = record["trajectory"][0]["subspace"]["Ligand"][0]
    monkeypatch.setattr(
        ablation_module,
        "select_pre_screen_states",
        lambda _records: [("suzuki", 100, 2)],
    )
    experiment = PromptAblationExperiment(
        source,
        output,
        client_factory=lambda variant, _record, _step: _Client(
            json.dumps({"subspace": {"Ligand": [ligand]}})
        ),
        state_keys=[("suzuki", 100, 2)],
    )
    artifact = experiment.run()
    artifact["analysis"] = {}
    artifact["gate"] = {"passed": False, "reasons": ["stale"]}
    output.write_text(json.dumps(artifact), encoding="utf-8")

    class ForbiddenClient:
        @classmethod
        def from_env(cls) -> ForbiddenClient:
            raise AssertionError("report must not create an LLM client")

    monkeypatch.setattr(ablation_module, "DeepSeekClient", ForbiddenClient)

    assert main(["report", "--source", str(source), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["analysis"]["suzuki"]["seed_deltas"] == {"100": 0.0}
    assert payload["gate"]["reasons"] == ["both datasets are required"]
def test_cli_run_rejects_unconfigured_environment_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": [_real_record()]}), encoding="utf-8")

    class UnconfiguredClient:
        model = "missing"

        @classmethod
        def from_env(cls) -> UnconfiguredClient:
            return cls()

        def is_configured(self) -> bool:
            return False

    monkeypatch.setattr(ablation_module, "DeepSeekClient", UnconfiguredClient)

    with pytest.raises(RuntimeError, match="not configured"):
        main(
            [
                "run",
                "--source",
                str(source),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
