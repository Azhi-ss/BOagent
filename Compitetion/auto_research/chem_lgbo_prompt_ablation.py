"""Paired prompt-ablation helpers for Chem-LGBO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from bo_core.benchmark.data_loader import DATA_LOADERS
from bo_core.llm_client import DeepSeekClient
from bo_core.optimization.chem_lgbo import (
    ChemLGBOEngine,
    build_subspace_mask,
)
from bo_core.optimization.chem_lgbo_prompt import PreviousGuidanceOutcome

_PHASES = ("1-10", "11-20", "21-30", "31-40")
_VARIANTS = ("control", "treatment")


def phase_for_step(step: int) -> str:
    """Return the fixed ten-round phase for one 1-based BO step."""
    if not 1 <= step <= 40:
        raise ValueError("step must be in [1, 40]")
    return _PHASES[(step - 1) // 10]


def classify_previous_outcome(
    previous_row: Mapping[str, Any], incumbent_before: float
) -> str:
    """Classify how the previous guidance was exercised."""
    if previous_row.get("guidance_status") != "applied":
        return "fallback"
    if previous_row.get("selected_in_subspace") is not True:
        return "selected_outside"
    observed = float(previous_row["observed_yield"])
    return "tested_improved" if observed > incumbent_before else "tested_nonimproving"


def same_mask(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two candidate-membership masks are identical."""
    left_array = np.asarray(left, dtype=bool)
    right_array = np.asarray(right, dtype=bool)
    if left_array.shape != right_array.shape:
        raise ValueError("mask shape mismatch")
    return bool(np.array_equal(left_array, right_array))


def select_pre_screen_states(
    records: Sequence[Mapping[str, Any]],
    *,
    per_phase: int = 2,
    per_extra_stratum: int = 2,
) -> list[tuple[str, int, int]]:
    """Select a deterministic, stratified subset of transition states."""
    if per_phase < 0 or per_extra_stratum < 0:
        raise ValueError("sample counts must be non-negative")

    selected: set[tuple[str, int, int]] = set()
    ordered_records = sorted(
        records, key=lambda item: (str(item["dataset"]), int(item["seed"]))
    )
    for record in ordered_records:
        dataset = str(record["dataset"])
        seed = int(record["seed"])
        trajectory = list(record["trajectory"])
        if len(trajectory) < 2:
            continue

        incumbent = float(record.get("prior_best", _fixed_prior_best(dataset)))
        candidates: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, previous in enumerate(trajectory[:-1]):
            previous_incumbent = incumbent
            observed = float(previous["observed_yield"])
            incumbent = max(incumbent, observed)
            next_step = int(trajectory[index + 1]["step"])
            stratum = classify_previous_outcome(previous, previous_incumbent)
            candidates[phase_for_step(next_step)][stratum].append(next_step)

        for phase in _PHASES:
            nonimproving = candidates[phase]["tested_nonimproving"]
            for step in nonimproving[:per_phase]:
                selected.add((dataset, seed, step))

        for stratum in ("tested_improved", "selected_outside"):
            steps = [
                step
                for phase in _PHASES
                for step in candidates[phase][stratum]
            ]
            for step in steps[:per_extra_stratum]:
                selected.add((dataset, seed, step))

    return sorted(selected)


def _fixed_prior_best(dataset: str) -> float:
    data = DATA_LOADERS[dataset]()
    return float(np.max(np.asarray(data["train_y"], dtype=float)))


def _state_keys(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, object]]:
    counters: dict[tuple[str, int, str], int] = defaultdict(int)
    keys: list[tuple[str, int, object]] = []
    for record in records:
        dataset = str(record["dataset"])
        seed = int(record["seed"])
        variant = str(record["variant"])
        if variant not in _VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        state_key = record.get("state_key")
        if state_key is None:
            counter_key = (dataset, seed, variant)
            state_key = counters[counter_key]
            counters[counter_key] += 1
        keys.append((dataset, seed, state_key))
    return keys


def aggregate_paired_results(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate state pairs within seed before crossing seeds."""
    pairs: dict[tuple[str, int, object], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for key, record in zip(_state_keys(records), records, strict=True):
        variant = str(record["variant"])
        if variant in pairs[key]:
            raise ValueError(f"duplicate paired record: {key!r}/{variant}")
        pairs[key][variant] = record

    by_dataset_seed: dict[tuple[str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for (dataset, seed, state_key), variants in pairs.items():
        if set(variants) != set(_VARIANTS):
            raise ValueError(f"incomplete pair: {(dataset, seed, state_key)!r}")
        by_dataset_seed[(dataset, seed)].append(
            (variants["control"], variants["treatment"])
        )

    datasets = sorted({dataset for dataset, _seed in by_dataset_seed})
    summary: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        seed_deltas: dict[str, float] = {}
        treatment_vs_gp: list[float] = []
        control_vs_gp: list[float] = []
        fallback_values = {variant: [] for variant in _VARIANTS}
        coverage_values = {variant: [] for variant in _VARIANTS}
        counterfactual_values = {variant: [] for variant in _VARIANTS}
        repeat_values = {variant: [] for variant in _VARIANTS}
        incumbent_values = {variant: [] for variant in _VARIANTS}
        repeat_by_stratum: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: {variant: [] for variant in _VARIANTS}
        )

        for (seed_dataset, seed), seed_pairs in sorted(by_dataset_seed.items()):
            if seed_dataset != dataset:
                continue
            deltas = []
            seed_treatment_vs_gp = []
            seed_control_vs_gp = []
            for control, treatment in seed_pairs:
                control_yield = float(control["selected_yield"])
                treatment_yield = float(treatment["selected_yield"])
                gp_yield = float(control["gp_yield"])
                if float(treatment["gp_yield"]) != gp_yield:
                    raise ValueError("paired records must use the same GP choice")
                deltas.append(treatment_yield - control_yield)
                seed_treatment_vs_gp.append(treatment_yield - gp_yield)
                seed_control_vs_gp.append(control_yield - gp_yield)
                for variant, record in (("control", control), ("treatment", treatment)):
                    fallback_values[variant].append(bool(record["fallback"]))
                    coverage = record.get("coverage")
                    if coverage is not None:
                        coverage_values[variant].append(float(coverage))
                    percentile = record.get("counterfactual_percentile")
                    if percentile is not None:
                        counterfactual_values[variant].append(float(percentile))
                    repeated = bool(record.get("same_previous_mask", False))
                    repeat_values[variant].append(repeated)
                    incumbent_values[variant].append(
                        bool(record.get("improved_incumbent", False))
                    )
                    stratum = str(
                        record.get("previous_outcome_stratum", "fallback")
                    )
                    repeat_by_stratum[stratum][variant].append(repeated)
            seed_deltas[str(seed)] = float(np.mean(deltas))
            treatment_vs_gp.append(float(np.mean(seed_treatment_vs_gp)))
            control_vs_gp.append(float(np.mean(seed_control_vs_gp)))

        values = np.asarray(list(seed_deltas.values()), dtype=float)
        summary[dataset] = {
            "seed_deltas": seed_deltas,
            "mean_delta": float(np.mean(values)),
            "treatment_vs_gp": float(np.mean(treatment_vs_gp)),
            "control_vs_gp": float(np.mean(control_vs_gp)),
            "control_fallback_rate": float(np.mean(fallback_values["control"])),
            "treatment_fallback_rate": float(np.mean(fallback_values["treatment"])),
            "control_mean_coverage": _optional_mean(coverage_values["control"]),
            "treatment_mean_coverage": _optional_mean(
                coverage_values["treatment"]
            ),
            "control_counterfactual_percentile": _optional_mean(
                counterfactual_values["control"]
            ),
            "treatment_counterfactual_percentile": _optional_mean(
                counterfactual_values["treatment"]
            ),
            "control_repeat_rate": _optional_mean(repeat_values["control"]),
            "treatment_repeat_rate": _optional_mean(
                repeat_values["treatment"]
            ),
            "control_incumbent_rate": _optional_mean(
                incumbent_values["control"]
            ),
            "treatment_incumbent_rate": _optional_mean(
                incumbent_values["treatment"]
            ),
            "repeat_by_stratum": {
                stratum: {
                    variant: float(np.mean(values))
                    for variant, values in variants.items()
                    if values
                }
                for stratum, variants in sorted(repeat_by_stratum.items())
            },
        }
    return summary


def _optional_mean(values: Sequence[float | bool]) -> float | None:
    return float(np.mean(values)) if values else None


def pre_screen_passes(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered prompt-only pre-screen gate."""
    reasons: list[str] = []
    required = {"buchwald_sub4", "suzuki"}
    if set(summary) != required:
        return {"passed": False, "reasons": ["both datasets are required"]}

    buchwald = summary["buchwald_sub4"]
    suzuki = summary["suzuki"]
    if float(buchwald["mean_delta"]) <= 0:
        reasons.append("buchwald treatment does not improve control")
    if float(buchwald["treatment_vs_gp"]) <= float(buchwald["control_vs_gp"]):
        reasons.append("buchwald treatment does not reduce GP loss")
    if float(suzuki["mean_delta"]) < 0:
        reasons.append("suzuki treatment harms control")

    for dataset, values in summary.items():
        if float(values["treatment_fallback_rate"]) > float(
            values["control_fallback_rate"]
        ):
            reasons.append(f"{dataset} treatment fallback rate increased")
        coverage = values.get("treatment_mean_coverage")
        if coverage is not None and float(coverage) >= 1.0:
            reasons.append(f"{dataset} treatment coverage is uninformative")

    return {"passed": not reasons, "reasons": reasons}


def _state_key(dataset: str, seed: int, step: int) -> str:
    return f"{dataset}:{seed}:{step}"


def replay_engine_to_step(
    record: Mapping[str, Any], target_step: int
) -> ChemLGBOEngine:
    """Reconstruct state with the ablation protocol's fixed 100 counterfactuals."""
    trajectory = list(record["trajectory"])
    if not 2 <= target_step <= len(trajectory):
        raise ValueError("target_step must identify a stored transition state")

    engine = ChemLGBOEngine(
        str(record["dataset"]),
        seed=int(record["seed"]),
        use_llm=False,
        n_iters=0,
        backend=str(record.get("backend", "botorch")),
        n_restarts=int(record.get("n_restarts", 10)),
        n_counterfactuals=100,
        outcome_feedback=False,
    )
    prior_best = float(np.max(engine.y_obs))
    artifacts = list(record.get("guidance_artifacts", []))
    engine.guidance_artifacts = []

    for row in trajectory[: target_step - 1]:
        query_index = int(row["query_index"])
        if not 0 <= query_index < engine.M or query_index in engine.queried:
            raise ValueError("stored query index is invalid or duplicated")
        observed_yield = float(row["observed_yield"])
        engine.trajectory.append(dict(row))
        engine.X_obs = np.vstack(
            [engine.X_obs, engine.pool_X[query_index : query_index + 1]]
        )
        engine.y_obs = np.append(engine.y_obs, observed_yield)
        engine.queried.add(query_index)
        engine.iteration += 1
        prior_best = max(prior_best, observed_yield)

    engine.guidance_artifacts = [
        dict(artifact)
        for artifact in artifacts
        if int(artifact.get("step", 0)) < target_step
    ]
    previous = trajectory[target_step - 2]
    previous_incumbent = float(
        np.max(engine.y_obs[: len(engine.y_obs) - 1])
    )
    if (
        previous.get("guidance_status") == "applied"
        and previous.get("subspace")
    ):
        engine.previous_outcome = PreviousGuidanceOutcome(
            proposed_subspace={
                str(field): [str(value) for value in values]
                for field, values in dict(previous["subspace"]).items()
            },
            selected_condition={
                str(field): str(value)
                for field, value in dict(previous["condition"]).items()
            },
            selected_in_subspace=bool(previous["selected_in_subspace"]),
            observed_yield=float(previous["observed_yield"]),
            incumbent_before=previous_incumbent,
        )
    else:
        engine.previous_outcome = None
    return engine


def _posterior_state(engine: ChemLGBOEngine) -> tuple[int, float, str]:
    surrogate = engine._fit_gp()
    mean, std = engine._predict_pool(surrogate)
    remaining = np.ones(engine.M, dtype=bool)
    if engine.queried:
        remaining[list(engine.queried)] = False
    acquisition = engine._expected_improvement(
        mean, std, float(np.max(engine.y_obs))
    )
    acquisition = np.where(remaining & np.isfinite(acquisition), acquisition, -np.inf)
    if not np.any(np.isfinite(acquisition)):
        raise ValueError("stored state has no finite GP acquisition")
    gp_index = int(np.argmax(acquisition))
    digest = hashlib.sha256()
    digest.update(np.asarray(mean, dtype=np.float64).tobytes())
    digest.update(np.asarray(std, dtype=np.float64).tobytes())
    digest.update(remaining.tobytes())
    return gp_index, float(engine.pool_yield[gp_index]), digest.hexdigest()


def evaluate_state(
    record: Mapping[str, Any],
    step: int,
    variant: str,
    client: Any,
) -> dict[str, Any]:
    """Evaluate one prompt variant on a reconstructed, identical BO state."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    engine = replay_engine_to_step(record, step)
    gp_index, gp_yield, posterior_hash = _posterior_state(engine)
    previous_outcome = engine.previous_outcome
    previous_mask = (
        build_subspace_mask(
            engine.test_df.loc[:, engine.feature_cols],
            previous_outcome.proposed_subspace,
        )
        if previous_outcome is not None
        else None
    )
    engine.outcome_feedback = variant == "treatment"
    engine.use_llm = True
    engine._client = client

    incumbent = float(np.max(engine.y_obs))
    row = engine.step()
    artifact = engine.guidance_artifacts[-1]
    selected_index = int(row["query_index"])
    selected_yield = float(engine.pool_yield[selected_index])
    selected_mask = (
        build_subspace_mask(
            engine.test_df.loc[:, engine.feature_cols], row["subspace"]
        )
        if row["subspace"]
        else None
    )
    counterfactual_indices = [
        int(index) for index in artifact["counterfactual_indices"]
    ]
    counterfactual_yields = [
        float(engine.pool_yield[index]) for index in counterfactual_indices
    ]
    counterfactual_percentile = (
        _midrank_percentile(selected_yield, counterfactual_yields)
        if counterfactual_yields
        else None
    )
    return {
        "state_key": _state_key(str(record["dataset"]), int(record["seed"]), step),
        "dataset": str(record["dataset"]),
        "seed": int(record["seed"]),
        "step": step,
        "phase": phase_for_step(step),
        "variant": variant,
        "previous_outcome_stratum": (
            _outcome_stratum(previous_outcome) if previous_outcome else "fallback"
        ),
        "posterior_hash": posterior_hash,
        "gp_index": gp_index,
        "gp_yield": gp_yield,
        "selected_index": selected_index,
        "selected_yield": selected_yield,
        "improved_incumbent": selected_yield > incumbent,
        "selected_in_subspace": row["selected_in_subspace"],
        "fallback": row["guidance_status"] == "fallback",
        "parse_reason": artifact["parser_reason"],
        "subspace": row["subspace"],
        "same_previous_mask": (
            same_mask(previous_mask, selected_mask)
            if previous_mask is not None and selected_mask is not None
            else False
        ),
        "mask_size": row["mask_size"],
        "coverage": row["coverage"],
        "counterfactual_count": len(counterfactual_indices),
        "counterfactual_percentile": counterfactual_percentile,
        "raw_response": artifact["raw_response"],
    }


def _outcome_stratum(outcome: PreviousGuidanceOutcome) -> str:
    if not outcome.selected_in_subspace:
        return "selected_outside"
    return (
        "tested_improved"
        if outcome.observed_yield > outcome.incumbent_before
        else "tested_nonimproving"
    )


def _midrank_percentile(selected: float, values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    wins = np.count_nonzero(array < selected)
    ties = np.count_nonzero(array == selected)
    return float((wins + 0.5 * ties) / len(array))


class PromptAblationExperiment:
    """Run and atomically persist paired prompt evaluations."""

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        *,
        client_factory: Callable[[str, dict[str, Any], int], Any],
        state_keys: Sequence[tuple[str, int, int]] | None = None,
    ) -> None:
        self.source_path = source_path
        self.output_path = output_path
        self.client_factory = client_factory
        source = json.loads(source_path.read_text(encoding="utf-8"))
        records = source.get("records")
        if not isinstance(records, list):
            raise TypeError("source artifact records are invalid")
        self.source_config = dict(source.get("config") or {})
        self.source_records = [
            dict(record)
            for record in records
            if record.get("composition") == "chem_lgbo"
        ]
        self.state_keys = list(
            state_keys or select_pre_screen_states(self.source_records)
        )

    def _load(self) -> dict[str, Any]:
        source_sha256 = _file_sha256(self.source_path)
        manifest = [list(key) for key in self.state_keys]
        if not self.output_path.exists():
            return {
                "schema_version": 1,
                "source_sha256": source_sha256,
                "state_manifest": manifest,
                "model_config": {
                    "chat_engine": self.source_config.get("chat_engine"),
                    "llm_max_tokens": self.source_config.get("llm_max_tokens"),
                    "reasoning_effort": self.source_config.get("reasoning_effort"),
                    "temperature": self.source_config.get("temperature", 0.2),
                    "response_mode": self.source_config.get(
                        "response_mode", "tool_call_react"
                    ),
                    "max_react_retries": self.source_config.get(
                        "max_react_retries", 1
                    ),
                },
                "provenance": _provenance(source_sha256),
                "record_count": 0,
                "records": [],
                "analysis": {},
                "gate": {"passed": False, "reasons": ["analysis incomplete"]},
            }
        artifact = json.loads(self.output_path.read_text(encoding="utf-8"))
        if artifact.get("source_sha256") != source_sha256:
            raise ValueError("source artifact provenance does not match")
        if artifact.get("state_manifest") != manifest:
            raise ValueError("state manifest does not match")
        if not isinstance(artifact.get("records"), list):
            raise TypeError("ablation artifact records are invalid")
        return artifact

    def run(self) -> dict[str, Any]:
        artifact = self._load()
        existing = {
            (str(record["state_key"]), str(record["variant"]))
            for record in artifact["records"]
        }
        source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for record in self.source_records:
            key = (str(record["dataset"]), int(record["seed"]))
            if key in source_by_key:
                raise ValueError(f"duplicate source record for {key!r}")
            source_by_key[key] = record
        changed = False
        for pair_index, (dataset, seed, step) in enumerate(self.state_keys):
            source_record = source_by_key[(dataset, seed)]
            variants = _VARIANTS if pair_index % 2 == 0 else tuple(reversed(_VARIANTS))
            for variant in variants:
                key = (_state_key(dataset, seed, step), variant)
                if key in existing:
                    continue
                client = self.client_factory(variant, source_record, step)
                artifact["records"].append(
                    evaluate_state(source_record, step, variant, client)
                )
                existing.add(key)
                changed = True
                self._persist(artifact)

        artifact["records"] = sorted(
            artifact["records"],
            key=lambda record: (str(record["state_key"]), str(record["variant"])),
        )
        artifact["record_count"] = len(artifact["records"])
        artifact["analysis"] = aggregate_paired_results(artifact["records"])
        artifact["gate"] = pre_screen_passes(artifact["analysis"])
        if changed:
            self._persist(artifact)
        return artifact

    def _persist(self, artifact: Mapping[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        os.replace(temporary, self.output_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(source_sha256: str) -> dict[str, Any]:
    source_paths = (
        Path(__file__),
        Path(__file__).resolve().parents[2]
        / "packages/bo-core/bo_core/optimization/chem_lgbo.py",
        Path(__file__).resolve().parents[2]
        / "packages/bo-core/bo_core/optimization/chem_lgbo_prompt.py",
        Path(__file__).resolve().parents[2]
        / "packages/bo-core/bo_core/optimization/chem_lgbo_parser.py",
    )
    return {
        "source_artifact_sha256": source_sha256,
        "sources": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): _file_sha256(path)
            for path in source_paths
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "report"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text(encoding="utf-8"))
    records = [
        record
        for record in source.get("records", [])
        if record.get("composition") == "chem_lgbo"
    ]
    state_keys = select_pre_screen_states(records)
    if args.mode == "preflight":
        result = {
            "source_sha256": _file_sha256(args.source),
            "record_count": len(records),
            "state_count": len(state_keys),
        }
    elif args.mode == "report":
        result = PromptAblationExperiment(
            args.source,
            args.output,
            client_factory=lambda _variant, _record, _step: None,
            state_keys=state_keys,
        )._load()
        result["record_count"] = len(result["records"])
        result["analysis"] = aggregate_paired_results(result["records"])
        result["gate"] = pre_screen_passes(result["analysis"])
    else:
        client = DeepSeekClient.from_env()
        if not client.is_configured():
            raise RuntimeError(
                "DeepSeek is not configured; set the API key in the project .env"
            )
        source.setdefault("config", {})["chat_engine"] = client.model

        def client_factory(
            _variant: str, _record: dict[str, Any], _step: int
        ) -> DeepSeekClient:
            variant_client = DeepSeekClient.from_env()
            variant_client.model = client.model
            variant_client.timeout_s = 120
            return variant_client

        experiment = PromptAblationExperiment(
            args.source,
            args.output,
            client_factory=client_factory,
            state_keys=state_keys,
        )
        experiment.source_config["chat_engine"] = client.model
        result = experiment.run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
