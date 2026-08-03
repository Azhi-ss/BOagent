"""Contract tests for the resumable Chem-LGBO comparison."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))
import chem_lgbo_experiment as experiment_module
from chem_lgbo_experiment import (
    ChemLGBOExperiment,
    counterfactual_percentile,
    paired_bootstrap_ci,
)


class FakeEngine:
    def __init__(
        self,
        composition: str,
        call_order: list[str],
        *,
        seed: int,
        n_iters: int,
        fail_at: int | None = None,
    ) -> None:
        self.composition = composition
        self.call_order = call_order
        self.seed = seed
        self.n_iters = n_iters
        self.fail_at = fail_at
        self.train_df = list(range(35))
        self.M = 100
        self.trajectory: list[dict[str, Any]] = []
        self.guidance_artifacts: list[dict[str, Any]] = []
        self.health = {
            "gp_fit_fallbacks": 0,
            "gp_predict_fallbacks": 0,
            "acquisition_fallbacks": 0,
            "nonfinite_acquisition_scores": 0,
            "duplicate_queries": 0,
        }

    def step(self) -> dict[str, Any]:
        iteration = len(self.trajectory)
        self.call_order.append(self.composition)
        if self.fail_at == iteration:
            raise RuntimeError(f"{self.composition} failed at {iteration}")

        remaining_pool_size = self.M - iteration
        row: dict[str, Any] = {
            "step": iteration + 1,
            "query_index": iteration,
            "condition": {"Ligand": "L1"},
            "observed_yield": float(iteration + 1),
            "predicted_yield": float(iteration) + 0.5,
            "remaining_pool_size": remaining_pool_size,
        }
        if self.composition == "gpbo":
            row.update(
                guidance_status="disabled",
                guidance_reason="use_llm_false",
                subspace=None,
                mask_size=None,
                coverage=None,
                counterfactual_seed=None,
                selected_in_subspace=None,
            )
        elif self.composition == "legacy_lgbo":
            row.update(
                guidance_status="applied",
                guidance_reason="Point",
                subspace=None,
                mask_size=None,
                coverage=None,
                counterfactual_seed=None,
                selected_in_subspace=None,
            )
        else:
            counterfactual_seed = self.seed * 1000 + iteration
            subspace = {"Ligand": ["L1"]}
            row.update(
                guidance_status="applied",
                guidance_reason="accepted",
                subspace=subspace,
                mask_size=1,
                coverage=1 / remaining_pool_size,
                counterfactual_seed=counterfactual_seed,
                selected_in_subspace=True,
            )
            self.guidance_artifacts.append(
                {
                    "step": iteration + 1,
                    "raw_response": '{"subspace":{"Ligand":["L1"]}}',
                    "parser_reason": "accepted",
                    "subspace": subspace,
                    "mask_size": 1,
                    "remaining_pool_size": remaining_pool_size,
                    "counterfactual_seed": counterfactual_seed,
                    "counterfactual_indices": [50 + iteration],
                    "selected_index": iteration,
                }
            )
        self.trajectory.append(row)
        return row


def make_factory(
    call_order: list[str],
    constructions: list[dict[str, Any]],
    *,
    fail: tuple[str, int] | None = None,
):
    def factory(**kwargs: Any) -> FakeEngine:
        constructions.append(kwargs)
        composition = str(kwargs["composition"])
        fail_at = fail[1] if fail and fail[0] == composition else None
        return FakeEngine(
            composition,
            call_order,
            seed=int(kwargs["seed"]),
            n_iters=int(kwargs["n_iters"]),
            fail_at=fail_at,
        )

    return factory


def test_run_group_interleaves_arms_and_persists_complete_group(
    tmp_path: Path,
):
    order: list[str] = []
    constructions: list[dict[str, Any]] = []
    output = tmp_path / "chem_lgbo.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory(order, constructions),
    )

    records = experiment.run_group("buchwald_sub4", 100)

    assert order == [
        "gpbo",
        "legacy_lgbo",
        "chem_lgbo",
        "gpbo",
        "chem_lgbo",
        "legacy_lgbo",
        "gpbo",
        "legacy_lgbo",
        "chem_lgbo",
        "gpbo",
        "chem_lgbo",
        "legacy_lgbo",
    ]
    assert [record["composition"] for record in records] == [
        "gpbo",
        "legacy_lgbo",
        "chem_lgbo",
    ]
    by_arm = {item["composition"]: item for item in constructions}
    assert by_arm["gpbo"]["use_llm"] is False
    assert by_arm["legacy_lgbo"]["use_llm"] is True
    assert by_arm["chem_lgbo"]["use_llm"] is True
    for field in ("chat_engine", "llm_max_tokens", "reasoning_effort"):
        assert by_arm["legacy_lgbo"][field] == by_arm["chem_lgbo"][field]
    assert by_arm["chem_lgbo"]["n_counterfactuals"] == 100

    assert by_arm["legacy_lgbo"]["llm_temperature"] == 0.0
    assert by_arm["chem_lgbo"]["llm_temperature"] == 0.2


    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["experiment"] == "chem_lgbo_v1"
    assert len(artifact["records"]) == 3
    assert artifact["provenance"] == experiment.provenance()


def test_complete_group_is_skipped_and_partial_group_is_fully_replaced(
    tmp_path: Path,
):
    output = tmp_path / "chem_lgbo.json"
    first_order: list[str] = []
    first_constructions: list[dict[str, Any]] = []
    first = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory(first_order, first_constructions),
    )
    complete = first.run_group("buchwald_sub4", 100)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    artifact["records"].append({**complete[0], "seed": 200})
    output.write_text(json.dumps(artifact), encoding="utf-8")

    order: list[str] = []
    constructions: list[dict[str, Any]] = []
    resumed = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory(order, constructions),
    )

    assert resumed.pending_groups() == [("buchwald_sub4", 200)]
    assert resumed.run_group("buchwald_sub4", 100) == complete
    assert constructions == []

    resumed.run_group("buchwald_sub4", 200)
    latest = json.loads(output.read_text(encoding="utf-8"))["records"]
    keys = [(record["composition"], record["dataset"], record["seed"]) for record in latest]
    assert len(keys) == len(set(keys)) == 6
    assert {key[0] for key in keys if key[2] == 200} == {
        "gpbo",
        "legacy_lgbo",
        "chem_lgbo",
    }


def test_step_exception_does_not_replace_last_atomic_artifact(tmp_path: Path):
    output = tmp_path / "chem_lgbo.json"
    successful = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    successful.run_group("buchwald_sub4", 100)
    before = output.read_bytes()

    failing = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], [], fail=("chem_lgbo", 1)),
    )
    with pytest.raises(RuntimeError, match="chem_lgbo failed"):
        failing.run_group("buchwald_sub4", 200)

    assert output.read_bytes() == before
    assert failing.pending_groups() == [("buchwald_sub4", 200)]


def valid_record(
    experiment: ChemLGBOExperiment,
    composition: str = "chem_lgbo",
    *,
    dataset: str = "buchwald_sub4",
    seed: int = 100,
) -> dict[str, Any]:
    prior_size = 35 if dataset == "buchwald_sub4" else 29
    pool_size = 100
    trajectory: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for iteration in range(experiment.n_iters):
        remaining_pool_size = pool_size - iteration
        row: dict[str, Any] = {
            "step": iteration + 1,
            "query_index": iteration,
            "condition": {"Ligand": "L1"},
            "observed_yield": float(iteration + 1),
            "predicted_yield": float(iteration) + 0.5,
            "remaining_pool_size": remaining_pool_size,
            "subspace": None,
            "mask_size": None,
            "coverage": None,
            "counterfactual_seed": None,
            "selected_in_subspace": None,
        }
        if composition == "gpbo":
            row.update(
                guidance_status="disabled",
                guidance_reason="use_llm_false",
            )
        elif composition == "legacy_lgbo":
            row.update(guidance_status="applied", guidance_reason="Point")
        else:
            counterfactual_seed = seed * 1000 + iteration
            subspace = {"Ligand": ["L1"]}
            row.update(
                guidance_status="applied",
                guidance_reason="accepted",
                subspace=subspace,
                mask_size=1,
                coverage=1 / remaining_pool_size,
                counterfactual_seed=counterfactual_seed,
                selected_in_subspace=True,
            )
            artifacts.append(
                {
                    "step": iteration + 1,
                    "raw_response": '{"subspace":{"Ligand":["L1"]}}',
                    "parser_reason": "accepted",
                    "subspace": subspace,
                    "mask_size": 1,
                    "remaining_pool_size": remaining_pool_size,
                    "counterfactual_seed": counterfactual_seed,
                    "counterfactual_indices": [50 + iteration],
                    "selected_index": iteration,
                }
            )
        trajectory.append(row)

    attempts = 0 if composition == "gpbo" else experiment.n_iters
    return {
        "composition": composition,
        "dataset": dataset,
        "seed": seed,
        "backend": experiment.backend,
        "config": experiment._config(),
        "prior_protocol": "fixed_train_prior",
        "n_train_prior": prior_size,
        "initial_indices": list(range(prior_size)),
        "pool_size": pool_size,
        "trajectory": trajectory,
        "metrics": {
            "best_found": 40.0,
            "initial_round_found_best": 1.0,
            "t95": 41,
            "AUC_best_so_far": 20.5,
        },
        "diagnostics": {
            "llm_attempts": attempts,
            "llm_successes": attempts,
            "llm_fallbacks": 0,
            "fallback_reasons": {},
            "gp_fit_fallbacks": 0,
            "gp_predict_fallbacks": 0,
            "acquisition_fallbacks": 0,
            "nonfinite_acquisition_scores": 0,
            "duplicate_queries": 0,
        },
        "guidance_artifacts": artifacts,
        "status": "ok",
    }


def invalidate(record: dict[str, Any], case: str) -> None:
    trajectory = record["trajectory"]
    diagnostics = record["diagnostics"]
    artifacts = record["guidance_artifacts"]
    if case == "status":
        record["status"] = "failed"
    elif case == "composition":
        record["composition"] = "unknown"
    elif case == "dataset":
        record["dataset"] = "unknown"
    elif case == "seed":
        record["seed"] = 999
    elif case == "backend":
        record["backend"] = "sklearn"
    elif case == "config":
        record["config"]["chat_engine"] = "different"
    elif case == "prior_protocol":
        record["prior_protocol"] = "random_prior"
    elif case == "prior_count":
        record["n_train_prior"] -= 1
    elif case == "prior_indices":
        record["initial_indices"][-1] = 99
    elif case == "trajectory_length":
        trajectory.pop()
    elif case == "step_number":
        trajectory[1]["step"] = 1
    elif case == "duplicate_query":
        trajectory[1]["query_index"] = trajectory[0]["query_index"]
    elif case == "out_of_range_query":
        trajectory[0]["query_index"] = record["pool_size"]
    elif case == "observed_nonfinite":
        trajectory[0]["observed_yield"] = float("nan")
    elif case == "predicted_nonfinite":
        trajectory[0]["predicted_yield"] = float("inf")
    elif case == "metric_nonfinite":
        record["metrics"]["AUC_best_so_far"] = float("nan")
    elif case == "t95":
        record["metrics"]["t95"] = 42
    elif case == "compact_diagnostics":
        trajectory[0].pop("guidance_status")
    elif case == "llm_attempts":
        diagnostics["llm_attempts"] -= 1
    elif case == "fallback_rate":
        for row in trajectory[:5]:
            row["guidance_status"] = "fallback"
            row["guidance_reason"] = "llm_error"
        diagnostics.update(
            llm_successes=35,
            llm_fallbacks=5,
            fallback_reasons={"llm_error": 5},
        )
    elif case == "health":
        diagnostics["gp_predict_fallbacks"] = 1
    elif case == "mask_size":
        trajectory[0]["mask_size"] = trajectory[0]["remaining_pool_size"]
    elif case == "coverage":
        trajectory[0]["coverage"] = 0.5
    elif case == "selected_in_subspace":
        trajectory[0]["selected_in_subspace"] = False
    elif case == "counterfactual_seed":
        trajectory[0]["counterfactual_seed"] += 1
    elif case == "counterfactual_index":
        artifacts[0]["counterfactual_indices"] = [record["pool_size"]]
    elif case == "artifact_selected_index":
        artifacts[0]["selected_index"] = 99
    else:  # pragma: no cover - test helper guard
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "status",
        "composition",
        "dataset",
        "seed",
        "backend",
        "config",
        "prior_protocol",
        "prior_count",
        "prior_indices",
        "trajectory_length",
        "step_number",
        "duplicate_query",
        "out_of_range_query",
        "observed_nonfinite",
        "predicted_nonfinite",
        "metric_nonfinite",
        "t95",
        "compact_diagnostics",
        "llm_attempts",
        "fallback_rate",
        "health",
        "mask_size",
        "coverage",
        "selected_in_subspace",
        "counterfactual_seed",
        "counterfactual_index",
        "artifact_selected_index",
    ],
)
def test_validate_record_rejects_each_invalid_contract(
    tmp_path: Path,
    case: str,
):
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )
    record = valid_record(experiment)
    invalidate(record, case)

    with pytest.raises(ValueError):
        experiment.validate_record(record)


@pytest.mark.parametrize("composition", ["gpbo", "legacy_lgbo", "chem_lgbo"])
def test_validate_record_accepts_each_arm(tmp_path: Path, composition: str):
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )

    experiment.validate_record(valid_record(experiment, composition))


def test_validate_matrix_requires_exact_record_keys(tmp_path: Path):
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
    )
    records = [
        valid_record(experiment, arm, seed=seed)
        for seed in experiment.seeds
        for arm in experiment.COMPOSITIONS
    ]

    experiment.validate_matrix(records)
    with pytest.raises(ValueError, match="matrix"):
        experiment.validate_matrix(records[:-1])

    duplicate = deepcopy(records)
    duplicate[-1] = deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="matrix"):
        experiment.validate_matrix(duplicate)


def matrix_records(
    experiment: ChemLGBOExperiment,
    auc: dict[tuple[str, str, int], float],
) -> list[dict[str, Any]]:
    records = [
        valid_record(experiment, arm, dataset=dataset, seed=seed)
        for dataset in experiment.datasets
        for seed in experiment.seeds
        for arm in experiment.COMPOSITIONS
    ]
    for record in records:
        key = (record["composition"], record["dataset"], record["seed"])
        record["metrics"]["AUC_best_so_far"] = auc[key]
    return records


def auc_values(
    experiment: ChemLGBOExperiment,
    *,
    chem: tuple[float, ...],
    gpbo: tuple[float, ...],
    legacy: tuple[float, ...],
) -> dict[tuple[str, str, int], float]:
    values: dict[tuple[str, str, int], float] = {}
    per_arm = {"gpbo": gpbo, "legacy_lgbo": legacy, "chem_lgbo": chem}
    for dataset in experiment.datasets:
        for arm, arm_values in per_arm.items():
            for seed, value in zip(experiment.seeds, arm_values):
                values[(arm, dataset, seed)] = value
    return values


def test_counterfactual_percentile_uses_midranks() -> None:
    assert counterfactual_percentile(5.0, [4.0, 5.0, 6.0, 5.0]) == pytest.approx(
        0.5
    )


def test_paired_bootstrap_uses_default_10000_and_supplied_rng() -> None:
    class RecordingRng:
        def __init__(self) -> None:
            self.size: tuple[int, int] | None = None

        def randint(
            self,
            low: int,
            high: int,
            *,
            size: tuple[int, int],
        ) -> np.ndarray:
            assert low == 0
            assert high == 2
            self.size = size
            return np.zeros(size, dtype=int)

    rng = RecordingRng()
    interval = paired_bootstrap_ci(np.array([2.0, 4.0]), rng=rng)

    assert rng.size == (10_000, 2)
    assert interval == pytest.approx((2.0, 2.0))


def test_analysis_pairs_auc_by_seed_not_record_position(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4", "suzuki"],
        seeds=[100, 200],
    )
    records = matrix_records(
        experiment,
        auc_values(
            experiment,
            chem=(11.0, 120.0),
            gpbo=(1.0, 100.0),
            legacy=(6.0, 110.0),
        ),
    )
    records = records[::2] + records[1::2]

    analysis = experiment.analyze(records=records, screening=True)

    for dataset in experiment.datasets:
        assert analysis["paired_auc"][dataset]["gpbo"]["differences"] == [
            10.0,
            20.0,
        ]
        assert analysis["paired_auc"][dataset]["legacy_lgbo"][
            "differences"
        ] == [5.0, 10.0]
    assert analysis["screening_passed"] is True
    assert analysis["verdict"] == "keep_legacy_lgbo"


def test_screening_requires_all_four_positive_means(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4", "suzuki"],
        seeds=[100, 200],
    )
    values = auc_values(
        experiment,
        chem=(10.0, 12.0),
        gpbo=(9.0, 11.0),
        legacy=(8.0, 10.0),
    )
    values[("legacy_lgbo", "suzuki", 200)] = 14.0

    analysis = experiment.analyze(
        records=matrix_records(experiment, values),
        screening=True,
    )

    assert analysis["paired_auc"]["suzuki"]["legacy_lgbo"]["mean"] == 0.0
    assert analysis["screening_passed"] is False
    assert analysis["verdict"] == "keep_legacy_lgbo"


def test_ci_lower_bound_equal_to_zero_blocks_promotion(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4", "suzuki"],
        seeds=[100, 200],
    )
    records = matrix_records(
        experiment,
        auc_values(
            experiment,
            chem=(10.0, 12.0),
            gpbo=(10.0, 10.0),
            legacy=(10.0, 10.0),
        ),
    )

    analysis = experiment.analyze(records=records)

    assert analysis["paired_auc"]["buchwald_sub4"]["gpbo"]["ci95"][0] == 0
    assert analysis["verdict"] == "keep_legacy_lgbo"


def test_auc_gate_overrides_better_secondary_and_composite_metrics(
    tmp_path: Path,
) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4", "suzuki"],
        seeds=[100, 200],
    )
    records = matrix_records(
        experiment,
        auc_values(
            experiment,
            chem=(10.0, 100.0),
            gpbo=(9.0, 9.0),
            legacy=(11.0, 0.0),
        ),
    )
    for record in records:
        if record["composition"] == "chem_lgbo":
            record["metrics"].update(
                best_found=90.0,
                initial_round_found_best=80.0,
                t95=1,
            )
        else:
            record["metrics"].update(
                best_found=1.0,
                initial_round_found_best=1.0,
                t95=41,
            )

    analysis = experiment.analyze(records=records)

    assert analysis["composite_scores"]["chem_lgbo"] > max(
        analysis["composite_scores"]["gpbo"],
        analysis["composite_scores"]["legacy_lgbo"],
    )
    assert analysis["paired_auc"]["buchwald_sub4"]["legacy_lgbo"][
        "ci95"
    ][0] < 0
    assert analysis["verdict"] == "keep_legacy_lgbo"


def test_counterfactual_rounds_average_within_seed_before_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
    )
    records = matrix_records(
        experiment,
        auc_values(
            experiment,
            chem=(10.0, 11.0),
            gpbo=(9.0, 10.0),
            legacy=(8.0, 9.0),
        ),
    )
    for record in records:
        if record["composition"] != "chem_lgbo":
            continue
        for artifact in record["guidance_artifacts"]:
            artifact["counterfactual_indices"] = []
        if record["seed"] == 100:
            record["guidance_artifacts"][0]["counterfactual_indices"] = [50]
            record["guidance_artifacts"][1]["counterfactual_indices"] = [50]
        else:
            record["guidance_artifacts"][0]["counterfactual_indices"] = [52]
            record["guidance_artifacts"][1]["counterfactual_indices"] = [52]

    oracle = np.ones(100, dtype=float)
    oracle[0] = 0.0
    oracle[1] = 2.0
    oracle[52] = -1.0
    monkeypatch.setattr(
        "chem_lgbo_experiment.DATA_LOADERS",
        {"buchwald_sub4": lambda: {"test_y": oracle}},
    )

    analysis = experiment.analyze(records=records)

    assert analysis["counterfactual"]["buchwald_sub4"]["seed_values"] == {
        "100": 0.5,
        "200": 1.0,
    }
    assert analysis["counterfactual"]["buchwald_sub4"]["mean"] == pytest.approx(
        0.75
    )
    assert analysis["counterfactual"]["buchwald_sub4"]["n_seeds"] == 2


def test_invalid_or_incomplete_matrix_cannot_promote(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4", "suzuki"],
        seeds=[100, 200],
    )
    records = matrix_records(
        experiment,
        auc_values(
            experiment,
            chem=(20.0, 21.0),
            gpbo=(1.0, 2.0),
            legacy=(3.0, 4.0),
        ),
    )

    analysis = experiment.analyze(records=records[:-1])

    assert analysis["matrix_valid"] is False
    assert analysis["verdict"] == "keep_legacy_lgbo"


def test_provenance_contains_source_hashes_and_runtime_config(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )

    provenance = experiment.provenance()

    assert provenance["config"] == experiment._config()
    assert provenance["model"] == experiment.chat_engine
    assert provenance["runtime"]["python"]
    assert provenance["runtime"]["numpy"]
    assert provenance["sources"][
        "packages/bo-core/bo_core/optimization/chem_lgbo.py"
    ]
    assert all(
        len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        for digest in provenance["sources"].values()
    )


def test_runtime_config_records_chem_tool_call_protocol(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )

    assert experiment._config()["temperature"] == 0.2
    assert experiment._config()["response_mode"] == "tool_call_react"
    assert experiment._config()["max_react_retries"] == 1


def test_legacy_provenance_without_tool_protocol_remains_readable(
    tmp_path: Path,
) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )
    stored = experiment.provenance()
    for field in ("temperature", "response_mode", "max_react_retries"):
        stored["config"].pop(field)

    experiment._assert_provenance({"records": [], "provenance": stored})

def test_legacy_provenance_with_records_cannot_resume_new_protocol(
    tmp_path: Path,
) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )
    stored = experiment.provenance()
    for field in ("temperature", "response_mode", "max_react_retries"):
        stored["config"].pop(field)

    with pytest.raises(ValueError, match="provenance does not match"):
        experiment._assert_provenance({"records": [{}], "provenance": stored})


def test_preflight_checks_llm_environment_before_running(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )

    def unavailable() -> None:
        raise RuntimeError("DeepSeek is not configured")

    experiment.verify_environment = unavailable  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="not configured"):
        experiment.preflight()


def test_preflight_persists_passed_stage(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "preflight.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=1,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]

    analysis = experiment.preflight()

    assert analysis["matrix_valid"] is True
    artifact = json.loads(experiment.output_path.read_text(encoding="utf-8"))
    assert artifact["stage"] == "preflight"
    assert artifact["verdict"] == "preflight_passed"
    assert artifact["analysis"]["matrix_valid"] is True


def test_run_screening_stops_and_persists_nonpositive_gate(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "screening.json",
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]

    analysis = experiment.run_screening()

    assert analysis["screening_passed"] is False
    artifact = json.loads(experiment.output_path.read_text(encoding="utf-8"))
    assert artifact["stage"] == "screening_stopped"
    assert artifact["screening"]["verdict"] == "keep_legacy_lgbo"


def test_report_defaults_to_stopped_screening_analysis(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "screening.json",
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]
    expected = experiment.run_screening()

    assert experiment.report() == expected


def test_default_model_uses_environment_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_FLASH_MODEL", "configured-model")

    experiment = ChemLGBOExperiment(tmp_path / "chem_lgbo.json")

    assert experiment.chat_engine == "configured-model"


def test_pending_groups_reruns_type_invalid_record(tmp_path: Path) -> None:
    output = tmp_path / "chem_lgbo.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100],
    )
    records = [
        valid_record(experiment, composition)
        for composition in experiment.COMPOSITIONS
    ]
    records[-1].pop("metrics")
    artifact = experiment._empty_artifact()
    artifact["records"] = records
    output.write_text(json.dumps(artifact), encoding="utf-8")

    assert experiment.pending_groups() == [("buchwald_sub4", 100)]


def test_analysis_reports_llm_failure_rates(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
    )
    records = [
        valid_record(experiment, composition)
        for composition in experiment.COMPOSITIONS
    ]
    legacy = next(
        record for record in records if record["composition"] == "legacy_lgbo"
    )
    for row in legacy["trajectory"][:4]:
        row["guidance_status"] = "fallback"
        row["guidance_reason"] = "llm_error"
    legacy["diagnostics"].update(
        llm_successes=36,
        llm_fallbacks=4,
        fallback_reasons={"llm_error": 4},
    )

    analysis = experiment.analyze(records=records)

    assert analysis["llm_failures"]["legacy_lgbo"]["buchwald_sub4"] == {
        "attempts": 40,
        "fallbacks": 4,
        "rate": 0.1,
    }


def test_resume_rejects_provenance_mismatch_before_calls(tmp_path: Path) -> None:
    output = tmp_path / "chem_lgbo.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.run_group("buchwald_sub4", 100)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    artifact["provenance"] = experiment.provenance()
    artifact["provenance"]["sources"][
        "packages/bo-core/bo_core/optimization/chem_lgbo.py"
    ] = "0" * 64
    output.write_text(json.dumps(artifact), encoding="utf-8")
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="provenance"):
        experiment.run_screening()


def test_run_full_requires_successful_screening(tmp_path: Path) -> None:
    experiment = ChemLGBOExperiment(
        tmp_path / "chem_lgbo.json",
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="screening"):
        experiment.run_full()


def test_run_full_preserves_screening_evidence(tmp_path: Path) -> None:
    output = tmp_path / "chem_lgbo.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    screening = {
        "matrix_valid": True,
        "screening_passed": True,
        "verdict": "keep_legacy_lgbo",
    }
    artifact = experiment._empty_artifact()
    artifact.update(
        provenance=experiment.provenance(),
        stage="screening_passed",
        screening=screening,
        verdict="keep_legacy_lgbo",
    )
    output.write_text(json.dumps(artifact), encoding="utf-8")
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]

    experiment.run_full()

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["stage"] == "complete"
    assert persisted["screening"] == screening


def test_report_rejects_tampered_screening_analysis(tmp_path: Path) -> None:
    output = tmp_path / "chem_lgbo.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100, 200],
        n_iters=4,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]
    experiment.run_screening()
    artifact = json.loads(output.read_text(encoding="utf-8"))
    artifact["screening"]["record_count"] = -1
    output.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="stored screening"):
        experiment.report(screening=True)


def test_preflight_verification_rejects_stale_provenance(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=1,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]
    expected = experiment.preflight()

    assert experiment.verify_preflight() == expected

    artifact = json.loads(output.read_text(encoding="utf-8"))
    artifact["provenance"]["runtime"]["python"] = "stale"
    output.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        experiment.verify_preflight()


def test_preflight_rejects_records_without_provenance(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    experiment = ChemLGBOExperiment(
        output,
        datasets=["buchwald_sub4"],
        seeds=[100],
        n_iters=1,
        backend="sklearn",
        engine_factory=make_factory([], []),
    )
    artifact = experiment._empty_artifact()
    artifact["records"] = [{}]
    output.write_text(json.dumps(artifact), encoding="utf-8")
    experiment.verify_environment = lambda: None  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="provenance"):
        experiment.preflight()


def test_main_preflight_uses_exact_smoke_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[tuple[Path, dict[str, Any]]] = []

    class FakeExperiment:
        def __init__(self, output_path: Path, **kwargs: Any) -> None:
            constructions.append((output_path, kwargs))

        def preflight(self) -> dict[str, Any]:
            return {"matrix_valid": True}

    output = tmp_path / "preflight.json"
    monkeypatch.setattr(experiment_module, "ChemLGBOExperiment", FakeExperiment)
    monkeypatch.setattr(experiment_module, "PREFLIGHT_PATH", output)

    assert experiment_module.main(["preflight"]) == 0
    assert constructions == [
        (
            output,
            {
                "datasets": ["buchwald_sub4", "suzuki"],
                "seeds": [100],
                "n_iters": 1,
            },
        )
    ]


def test_main_report_uses_screening_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[tuple[Path, dict[str, Any]]] = []

    class FakeExperiment:
        def __init__(self, output_path: Path, **kwargs: Any) -> None:
            constructions.append((output_path, kwargs))

        def report(self) -> dict[str, Any]:
            return {"verdict": "keep_legacy_lgbo"}

    output = tmp_path / "screening.json"
    output.write_text(json.dumps({"stage": "screening_stopped"}), encoding="utf-8")
    monkeypatch.setattr(experiment_module, "ChemLGBOExperiment", FakeExperiment)
    monkeypatch.setattr(experiment_module, "FULL_PATH", output)

    assert experiment_module.main(["report"]) == 0
    assert constructions == [
        (
            output,
            {"seeds": [100, 200, 300, 400, 500]},
        )
    ]
