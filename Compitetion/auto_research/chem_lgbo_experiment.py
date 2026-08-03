"""Resumable serial comparison of GPBO, Legacy LGBO, and Chem-LGBO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = ROOT / "Compitetion" / "auto_research" / "history" / "experiments" / "chem_lgbo_v1_preflight.json"
FULL_PATH = ROOT / "Compitetion" / "auto_research" / "history" / "experiments" / "chem_lgbo_v1.json"
SOURCE_PATHS = (
    "packages/bo-core/bo_core/optimization/lgbo.py",
    "packages/bo-core/bo_core/optimization/chem_lgbo.py",
    "packages/bo-core/bo_core/optimization/chem_lgbo_parser.py",
    "packages/bo-core/bo_core/optimization/chem_lgbo_prompt.py",
    "packages/bo-core/bo_core/optimization/categorical.py",
    "packages/bo-core/bo_core/optimization/surrogate.py",
    "packages/bo-core/bo_core/benchmark/lgbo_runner.py",
    "packages/bo-core/bo_core/benchmark/data_loader.py",
    "packages/bo-core/bo_core/llm_client.py",
    "Compitetion/submission/code/bo_core/optimization/lgbo.py",
    "Compitetion/submission/code/bo_core/optimization/chem_lgbo.py",
    "Compitetion/submission/code/bo_core/optimization/chem_lgbo_parser.py",
    "Compitetion/submission/code/bo_core/optimization/chem_lgbo_prompt.py",
    "Compitetion/submission/code/bo_core/optimization/categorical.py",
    "Compitetion/submission/code/bo_core/optimization/surrogate.py",
    "Compitetion/submission/code/bo_core/benchmark/lgbo_runner.py",
    "Compitetion/submission/code/bo_core/benchmark/data_loader.py",
    "Compitetion/submission/code/bo_core/llm_client.py",
    "Compitetion/auto_research/chem_lgbo_experiment.py",
    "Compitetion/auto_research/analyze.py",
)
DATA_PATHS = tuple(
    f"datasets/chemical_reactions/{dataset}/{filename}"
    for dataset in ("buchwald_sub4", "suzuki")
    for filename in ("options.json", "train.csv", "test_features.csv", "test.csv")
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


from analyze import aggregate_results, composite_score
from bo_core.benchmark.data_loader import DATA_LOADERS
from bo_core.benchmark.lgbo_runner import GLOBAL_BEST, compute_metrics
from bo_core.optimization.chem_lgbo import ChemLGBOEngine
from bo_core.optimization.lgbo import LGBOEngine

RunKey = tuple[str, str, int]
EngineFactory = Callable[..., Any]


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    rng: np.random.RandomState,
    n_resamples: int = 10_000,
) -> tuple[float, float]:
    """Return a percentile-bootstrap CI for seed-paired differences."""
    values = np.asarray(differences, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a non-empty finite array")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    sample_indices = rng.randint(
        0,
        values.size,
        size=(n_resamples, values.size),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
    return float(lower), float(upper)


def counterfactual_percentile(
    selected_yield: float,
    counterfactual_yields: Sequence[float],
) -> float:
    """Compute the mid-rank percentile of a selected yield."""
    selected = float(selected_yield)
    values = np.asarray(list(counterfactual_yields), dtype=float)
    if values.size == 0 or not math.isfinite(selected) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("selected and counterfactual yields must be finite")
    wins = np.count_nonzero(values < selected)
    ties = np.count_nonzero(values == selected)
    return float((wins + 0.5 * ties) / values.size)


class ChemLGBOExperiment:
    """Run complete three-arm seed groups and persist only atomic groups."""

    COMPOSITIONS: ClassVar[tuple[str, ...]] = (
        "gpbo",
        "legacy_lgbo",
        "chem_lgbo",
    )
    DATASETS: ClassVar[tuple[str, ...]] = ("buchwald_sub4", "suzuki")
    SEEDS: ClassVar[tuple[int, ...]] = tuple(range(100, 2001, 100))
    PRIOR_PROTOCOL = "fixed_train_prior"

    def __init__(
        self,
        output_path: Path,
        *,
        datasets: list[str] | None = None,
        seeds: list[int] | None = None,
        n_iters: int = 40,
        backend: str = "botorch",
        chat_engine: str | None = None,
        llm_max_tokens: int = 8192,
        reasoning_effort: str = "low",
        llm_temperature: float = 0.2,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.datasets = list(datasets if datasets is not None else self.DATASETS)
        self.seeds = list(seeds if seeds is not None else self.SEEDS)
        self.n_iters = int(n_iters)
        self.backend = backend
        if chat_engine is None:
            from bo_core.llm_client import DeepSeekClient

            chat_engine = DeepSeekClient.from_env().model
        self.chat_engine = chat_engine
        self.llm_max_tokens = int(llm_max_tokens)
        self.reasoning_effort = reasoning_effort
        self.llm_temperature = float(llm_temperature)
        self._engine_factory = engine_factory or self._build_engine

    @staticmethod
    def verify_environment() -> None:
        """Fail before spending an experiment if the LLM is unavailable."""
        from bo_core.llm_client import DeepSeekClient

        if not DeepSeekClient.from_env().is_configured():
            raise RuntimeError(
                "DeepSeek is not configured; set the API key in the project .env"
            )

    def provenance(self) -> dict[str, Any]:
        paths = (*SOURCE_PATHS, *DATA_PATHS)
        missing = [path for path in paths if not (ROOT / path).is_file()]
        if missing:
            raise FileNotFoundError(f"provenance inputs are missing: {missing}")
        return {
            "sources": {path: _sha256(ROOT / path) for path in paths},
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": _package_version("torch"),
                "botorch": _package_version("botorch"),
            },
            "model": self.chat_engine,
            "config": self._config(),
        }

    def _assert_provenance(self, artifact: dict[str, Any]) -> None:
        stored = artifact.get("provenance")
        if artifact.get("records") and not stored:
            raise ValueError("artifact records are missing provenance")
        if stored:
            expected = self.provenance()
            stored = deepcopy(stored)
            stored_config = stored.get("config")
            if not artifact.get("records") and isinstance(stored_config, dict):
                for field in ("temperature", "response_mode", "max_react_retries"):
                    stored_config.setdefault(field, expected["config"][field])
            if stored != expected:
                raise ValueError("artifact provenance does not match current experiment")

    def preflight(self) -> dict[str, Any]:
        """Run and validate one real iteration for every configured dataset."""
        self.verify_environment()
        self._assert_provenance(self._load_artifact())
        self._run_pending_groups()
        analysis = self.analyze()
        if not analysis["matrix_valid"]:
            raise ValueError(analysis.get("validation_error", "preflight failed"))
        artifact = self._load_artifact()
        artifact.update(
            provenance=self.provenance(),
            stage="preflight",
            analysis=analysis,
            verdict="preflight_passed",
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._atomic_write_json(self.output_path, artifact)
        return analysis

    def verify_preflight(self) -> dict[str, Any]:
        artifact = self._load_artifact()
        if artifact.get("stage") != "preflight":
            raise RuntimeError("preflight has not passed")
        self._assert_provenance(artifact)
        analysis = artifact.get("analysis")
        if not isinstance(analysis, dict) or not analysis.get("matrix_valid"):
            raise ValueError("stored preflight is incomplete")
        self.validate_matrix(artifact["records"])
        return analysis

    def run_screening(self) -> dict[str, Any]:
        """Run the configured screening matrix and persist its mean gate."""
        self.verify_environment()
        artifact = self._load_artifact()
        self._assert_provenance(artifact)
        self._run_pending_groups()
        analysis = self.analyze(screening=True)
        stage = (
            "screening_passed"
            if analysis["screening_passed"]
            else "screening_stopped"
        )
        self._persist_analysis(stage=stage, screening=analysis)
        return analysis

    def run_full(self) -> dict[str, Any]:
        """Resume the configured full matrix and persist its final verdict."""
        self.verify_environment()
        artifact = self._load_artifact()
        self._assert_provenance(artifact)
        screening = artifact.get("screening")
        if artifact.get("stage") != "screening_passed" or not isinstance(
            screening, dict
        ) or not screening.get("screening_passed"):
            raise RuntimeError("screening must pass before the full run")
        self._run_pending_groups()
        analysis = self.analyze()
        self._persist_analysis(
            stage="complete", screening=screening, analysis=analysis
        )
        return analysis

    def report(self, *, screening: bool = False) -> dict[str, Any]:
        artifact = self._load_artifact()
        self._assert_provenance(artifact)
        if not screening and artifact.get("stage") in {
            "screening_passed",
            "screening_stopped",
        }:
            screening = True
        key = "screening" if screening else "analysis"
        stored = artifact.get(key)
        if stored is None:
            raise ValueError(f"stored {key} analysis is missing")
        if not isinstance(stored, dict):
            raise TypeError(f"stored {key} analysis is invalid")
        current = self.analyze(screening=screening)
        if current != stored:
            raise ValueError(f"stored {key} analysis does not match records")
        return stored

    def _run_pending_groups(self) -> None:
        for dataset, seed in self.pending_groups():
            self.run_group(dataset, seed)

    def _persist_analysis(
        self,
        *,
        stage: str,
        screening: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        artifact = self._load_artifact()
        result = analysis if analysis is not None else screening
        artifact.update(
            provenance=self.provenance(),
            stage=stage,
            screening=screening,
            analysis=analysis,
            verdict=result["verdict"] if result else "keep_legacy_lgbo",
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._atomic_write_json(self.output_path, artifact)

    def pending_groups(self) -> list[tuple[str, int]]:
        latest = self._latest_records()
        return [
            (dataset, seed)
            for dataset in self.datasets
            for seed in self.seeds
            if not self._complete_group(latest, dataset, seed)
        ]

    def run_group(self, dataset: str, seed: int) -> list[dict[str, Any]]:
        latest = self._latest_records()
        if self._complete_group(latest, dataset, seed):
            return [latest[(arm, dataset, seed)] for arm in self.COMPOSITIONS]

        engines = {
            arm: self._engine_factory(
                composition=arm,
                dataset=dataset,
                seed=seed,
                use_llm=arm != "gpbo",
                n_iters=self.n_iters,
                backend=self.backend,
                chat_engine=self.chat_engine,
                llm_max_tokens=self.llm_max_tokens,
                reasoning_effort=self.reasoning_effort,
                llm_temperature=(self.llm_temperature if arm == "chem_lgbo" else 0.0),
                n_counterfactuals=100 if arm == "chem_lgbo" else 0,
            )
            for arm in self.COMPOSITIONS
        }
        for iteration in range(self.n_iters):
            engines["gpbo"].step()
            llm_order = (
                ("legacy_lgbo", "chem_lgbo")
                if iteration % 2 == 0
                else ("chem_lgbo", "legacy_lgbo")
            )
            for arm in llm_order:
                engines[arm].step()

        records = [
            self._record(arm, dataset, seed, engines[arm])
            for arm in self.COMPOSITIONS
        ]
        self._persist_group(dataset, seed, records)
        return records

    def _build_engine(self, **kwargs: Any) -> LGBOEngine:
        composition = kwargs.pop("composition")
        kwargs.pop("n_counterfactuals")
        if composition == "chem_lgbo":
            kwargs["n_counterfactuals"] = 100
            return ChemLGBOEngine(**kwargs)
        return LGBOEngine(**kwargs)

    def _record(
        self,
        composition: str,
        dataset: str,
        seed: int,
        engine: Any,
    ) -> dict[str, Any]:
        trajectory = list(engine.trajectory)
        fallback_reasons: dict[str, int] = {}
        for row in trajectory:
            if row.get("guidance_status") == "fallback":
                reason = str(row.get("guidance_reason"))
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        attempts = 0 if composition == "gpbo" else len(trajectory)
        fallbacks = sum(fallback_reasons.values())
        diagnostics = {
            "llm_attempts": attempts,
            "llm_successes": attempts - fallbacks,
            "llm_fallbacks": fallbacks,
            "fallback_reasons": fallback_reasons,
            **getattr(engine, "health", {}),
        }
        return {
            "composition": composition,
            "dataset": dataset,
            "seed": seed,
            "backend": self.backend,
            "config": self._config(),
            "prior_protocol": self.PRIOR_PROTOCOL,
            "n_train_prior": len(engine.train_df),
            "initial_indices": list(range(len(engine.train_df))),
            "pool_size": int(engine.M),
            "trajectory": trajectory,
            "metrics": compute_metrics(engine, GLOBAL_BEST[dataset]),
            "diagnostics": diagnostics,
            "guidance_artifacts": list(
                getattr(engine, "guidance_artifacts", [])
            ),
            "status": "ok",
        }

    def _config(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_iters": self.n_iters,
            "chat_engine": self.chat_engine,
            "llm_max_tokens": self.llm_max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.llm_temperature,
            "response_mode": "tool_call_react",
            "max_react_retries": 1,
            "n_counterfactuals": 100,
        }

    def validate_record(self, record: dict[str, Any]) -> None:
        """Reject any record that cannot enter the paired comparison."""
        if record.get("status") != "ok":
            raise ValueError("record status is not ok")

        composition = record.get("composition")
        dataset = record.get("dataset")
        seed = record.get("seed")
        if composition not in self.COMPOSITIONS:
            raise ValueError(f"unknown composition: {composition!r}")
        if dataset not in self.datasets:
            raise ValueError(f"unknown dataset: {dataset!r}")
        if seed not in self.seeds:
            raise ValueError(f"unknown seed: {seed!r}")
        if record.get("backend") != self.backend:
            raise ValueError("record backend does not match experiment")
        if record.get("config") != self._config():
            raise ValueError("record config does not match experiment")
        if record.get("prior_protocol") != self.PRIOR_PROTOCOL:
            raise ValueError("record prior protocol is invalid")

        expected_prior = {"buchwald_sub4": 35, "suzuki": 29}[dataset]
        if record.get("n_train_prior") != expected_prior:
            raise ValueError("record prior count is invalid")
        if record.get("initial_indices") != list(range(expected_prior)):
            raise ValueError("record prior indices are invalid")

        pool_size = record.get("pool_size")
        if (
            not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or pool_size <= 0
        ):
            raise ValueError("record pool size is invalid")
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != self.n_iters:
            raise ValueError("record trajectory length is invalid")

        compact_fields = {
            "guidance_status",
            "guidance_reason",
            "subspace",
            "mask_size",
            "remaining_pool_size",
            "coverage",
            "counterfactual_seed",
            "selected_in_subspace",
        }
        query_indices: list[int] = []
        fallback_reasons: Counter[str] = Counter()
        applied = 0
        for step_number, row in enumerate(trajectory, start=1):
            if not isinstance(row, dict) or row.get("step") != step_number:
                raise ValueError(f"trajectory step {step_number} is invalid")
            if not compact_fields <= row.keys():
                raise ValueError(
                    f"trajectory step {step_number} lacks compact diagnostics"
                )
            query_index = row.get("query_index")
            if (
                not isinstance(query_index, int)
                or isinstance(query_index, bool)
                or not 0 <= query_index < pool_size
            ):
                raise ValueError(
                    f"trajectory step {step_number} query is invalid"
                )
            query_indices.append(query_index)
            for field in ("observed_yield", "predicted_yield"):
                value = row.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"trajectory step {step_number} {field} is invalid"
                    )

            expected_remaining = pool_size - step_number + 1
            if row.get("remaining_pool_size") != expected_remaining:
                raise ValueError(
                    f"trajectory step {step_number} remaining pool is invalid"
                )
            status = row.get("guidance_status")
            reason = row.get("guidance_reason")
            if composition == "gpbo":
                if status != "disabled" or reason != "use_llm_false":
                    raise ValueError("GPBO guidance diagnostics are invalid")
            elif status == "applied":
                applied += 1
            elif status == "fallback" and isinstance(reason, str):
                fallback_reasons[reason] += 1
            else:
                raise ValueError(
                    f"trajectory step {step_number} LLM outcome is invalid"
                )

            if composition != "chem_lgbo":
                if row.get("selected_in_subspace") is not None:
                    raise ValueError(
                        "non-Chem record has subspace selection state"
                    )
                continue
            if status == "applied":
                subspace = row.get("subspace")
                mask_size = row.get("mask_size")
                coverage = row.get("coverage")
                if not isinstance(subspace, dict) or not subspace:
                    raise ValueError("Chem applied row lacks a subspace")
                if not all(
                    isinstance(values, list) and values
                    for values in subspace.values()
                ):
                    raise ValueError("Chem applied subspace is invalid")
                if (
                    not isinstance(mask_size, int)
                    or isinstance(mask_size, bool)
                    or not 0 < mask_size < expected_remaining
                ):
                    raise ValueError("Chem applied mask size is invalid")
                if (
                    not isinstance(coverage, (int, float))
                    or isinstance(coverage, bool)
                    or not math.isfinite(float(coverage))
                    or not math.isclose(
                        float(coverage),
                        mask_size / expected_remaining,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("Chem applied coverage is invalid")
                condition = row.get("condition")
                if not isinstance(condition, dict):
                    raise ValueError("Chem applied condition is invalid")
                expected_selected = all(
                    str(condition.get(field)) in values
                    for field, values in subspace.items()
                )
                if row.get("selected_in_subspace") is not expected_selected:
                    raise ValueError(
                        "Chem selected_in_subspace is inconsistent"
                    )
                expected_seed = int(seed) * 1000 + step_number - 1
                if row.get("counterfactual_seed") != expected_seed:
                    raise ValueError("Chem counterfactual seed is invalid")

        if len(set(query_indices)) != self.n_iters:
            raise ValueError("record contains duplicate queries")

        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("record metrics are missing")
        for name in (
            "best_found",
            "initial_round_found_best",
            "t95",
            "AUC_best_so_far",
        ):
            value = metrics.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"metric {name} is invalid")
        t95 = metrics["t95"]
        if (
            not isinstance(t95, int)
            or isinstance(t95, bool)
            or not 1 <= t95 <= self.n_iters + 1
        ):
            raise ValueError("metric t95 is outside the experiment budget")

        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise TypeError("record diagnostics are missing")
        expected_attempts = 0 if composition == "gpbo" else self.n_iters
        fallbacks = sum(fallback_reasons.values())
        if diagnostics.get("llm_attempts") != expected_attempts:
            raise ValueError("LLM attempt count is invalid")
        if diagnostics.get("llm_successes") != applied:
            raise ValueError("LLM success count is invalid")
        if diagnostics.get("llm_fallbacks") != fallbacks:
            raise ValueError("LLM fallback count is invalid")
        if diagnostics.get("fallback_reasons") != dict(fallback_reasons):
            raise ValueError("LLM fallback reasons are invalid")
        if expected_attempts and applied + fallbacks != self.n_iters:
            raise ValueError("LLM outcomes do not cover every iteration")
        if expected_attempts and fallbacks / self.n_iters > 0.10:
            raise ValueError("LLM fallback rate exceeds 10%")
        for name in (
            "gp_fit_fallbacks",
            "gp_predict_fallbacks",
            "acquisition_fallbacks",
            "nonfinite_acquisition_scores",
            "duplicate_queries",
        ):
            if diagnostics.get(name) != 0:
                raise ValueError(f"health counter {name} is not zero")

        artifacts = record.get("guidance_artifacts")
        if composition != "chem_lgbo":
            if artifacts != []:
                raise ValueError(
                    "non-Chem record contains guidance artifacts"
                )
            return
        if not isinstance(artifacts, list) or len(artifacts) != self.n_iters:
            raise ValueError("Chem guidance artifacts are incomplete")
        for step_number, (row, artifact) in enumerate(
            zip(trajectory, artifacts), start=1
        ):
            if (
                not isinstance(artifact, dict)
                or artifact.get("step") != step_number
            ):
                raise ValueError("Chem guidance artifact step is invalid")
            for field in (
                "subspace",
                "mask_size",
                "remaining_pool_size",
                "counterfactual_seed",
            ):
                if artifact.get(field) != row.get(field):
                    raise ValueError(
                        f"Chem artifact {field} is inconsistent"
                    )
            if artifact.get("parser_reason") != row.get("guidance_reason"):
                raise ValueError(
                    "Chem artifact parser reason is inconsistent"
                )
            if artifact.get("selected_index") != row.get("query_index"):
                raise ValueError(
                    "Chem artifact selected index is inconsistent"
                )
            indices = artifact.get("counterfactual_indices")
            if not isinstance(indices, list) or len(indices) > 100:
                raise ValueError("Chem counterfactual indices are invalid")
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < pool_size
                for index in indices
            ):
                raise ValueError("Chem counterfactual index is out of range")
            if (
                row["guidance_status"] == "fallback"
                and (artifact.get("counterfactual_seed") is not None or indices)
            ):
                raise ValueError("Chem fallback has counterfactual data")

    def validate_matrix(self, records: list[dict[str, Any]]) -> None:
        """Require the exact configured dataset/seed/arm record key set."""
        expected = {
            (arm, dataset, seed)
            for dataset in self.datasets
            for seed in self.seeds
            for arm in self.COMPOSITIONS
        }
        actual: list[RunKey] = []
        for record in records:
            self.validate_record(record)
            actual.append(self._record_key(record))
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(
                "experiment matrix keys are incomplete or duplicated"
            )

    def analyze(
        self,
        records: list[dict[str, Any]] | None = None,
        *,
        screening: bool = False,
    ) -> dict[str, Any]:
        """Validate and analyze one exact experiment matrix."""
        if records is None:
            records = list(self._load_artifact()["records"])
        result: dict[str, Any] = {
            "matrix_valid": False,
            "record_count": len(records),
            "screening_passed": False,
            "ci_gate_passed": False,
            "verdict": "keep_legacy_lgbo",
        }
        try:
            self.validate_matrix(records)
        except (TypeError, ValueError) as exc:
            result["validation_error"] = str(exc)
            return result

        result["matrix_valid"] = True
        summary = aggregate_results(records)
        paired: dict[str, dict[str, dict[str, Any]]] = {}
        rng = np.random.RandomState(20260801)
        by_key = {self._record_key(record): record for record in records}
        for dataset in self.datasets:
            paired[dataset] = {}
            for baseline in ("gpbo", "legacy_lgbo"):
                differences = np.asarray(
                    [
                        float(
                            by_key[("chem_lgbo", dataset, seed)]["metrics"][
                                "AUC_best_so_far"
                            ]
                        )
                        - float(
                            by_key[(baseline, dataset, seed)]["metrics"][
                                "AUC_best_so_far"
                            ]
                        )
                        for seed in self.seeds
                    ],
                    dtype=float,
                )
                ci = paired_bootstrap_ci(differences, rng=rng)
                paired[dataset][baseline] = {
                    "differences": differences.tolist(),
                    "mean": float(np.mean(differences)),
                    "ci95": [float(ci[0]), float(ci[1])],
                }

        screening_passed = all(
            pair["mean"] > 0
            for dataset_pairs in paired.values()
            for pair in dataset_pairs.values()
        )
        ci_gate_passed = all(
            pair["ci95"][0] > 0
            for dataset_pairs in paired.values()
            for pair in dataset_pairs.values()
        )

        coverage: dict[str, dict[str, Any]] = {}
        fallback_reasons: dict[str, dict[str, int]] = {}
        llm_failures: dict[str, dict[str, dict[str, float | int]]] = {}
        for dataset in self.datasets:
            coverage_values = [
                float(row["coverage"])
                for record in records
                if record["dataset"] == dataset
                and record["composition"] == "chem_lgbo"
                for row in record["trajectory"]
                if row["guidance_status"] == "applied"
            ]
            coverage[dataset] = {
                "values": coverage_values,
                "mean": float(np.mean(coverage_values))
                if coverage_values
                else None,
                "n": len(coverage_values),
            }
        for composition in ("legacy_lgbo", "chem_lgbo"):
            counts: Counter[str] = Counter()
            llm_failures[composition] = {}
            for dataset in self.datasets:
                attempts = sum(
                    record["diagnostics"]["llm_attempts"]
                    for record in records
                    if record["composition"] == composition
                    and record["dataset"] == dataset
                )
                fallbacks = sum(
                    record["diagnostics"]["llm_fallbacks"]
                    for record in records
                    if record["composition"] == composition
                    and record["dataset"] == dataset
                )
                llm_failures[composition][dataset] = {
                    "attempts": attempts,
                    "fallbacks": fallbacks,
                    "rate": fallbacks / attempts if attempts else 0.0,
                }
            for record in records:
                if record["composition"] == composition:
                    counts.update(record["diagnostics"]["fallback_reasons"])
            fallback_reasons[composition] = dict(counts)

        result.update(
            {
                "official_summary": summary,
                "composite_scores": composite_score(summary),
                "paired_auc": paired,
                "mask_coverage": coverage,
                "fallback_reasons": fallback_reasons,
                "llm_failures": llm_failures,
                "counterfactual": self._counterfactual_summary(records),
                "screening_passed": screening_passed,
                "ci_gate_passed": ci_gate_passed,
                "stage": "screening" if screening else "full",
                "verdict": (
                    "promote_chem_lgbo"
                    if not screening and ci_gate_passed
                    else "keep_legacy_lgbo"
                ),
            }
        )
        return result

    def _counterfactual_summary(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        by_key = {self._record_key(record): record for record in records}
        summary: dict[str, dict[str, Any]] = {}
        for dataset in self.datasets:
            data = DATA_LOADERS[dataset]()
            oracle = np.asarray(data["test_y"], dtype=float)
            seed_values: dict[str, float] = {}
            for seed in self.seeds:
                record = by_key[("chem_lgbo", dataset, seed)]
                round_values: list[float] = []
                for artifact in record["guidance_artifacts"]:
                    indices = artifact["counterfactual_indices"]
                    if not indices:
                        continue
                    selected_index = int(artifact["selected_index"])
                    if selected_index >= len(oracle) or any(
                        int(index) >= len(oracle) for index in indices
                    ):
                        raise ValueError(
                            f"oracle index is out of range for {dataset}"
                        )
                    round_values.append(
                        counterfactual_percentile(
                            float(oracle[selected_index]),
                            [float(oracle[index]) for index in indices],
                        )
                    )
                if round_values:
                    seed_values[str(seed)] = float(np.mean(round_values))
            values = np.asarray(list(seed_values.values()), dtype=float)
            if values.size:
                ci = paired_bootstrap_ci(
                    values - 0.5,
                    rng=np.random.RandomState(20260801),
                )
                summary[dataset] = {
                    "seed_values": seed_values,
                    "mean": float(np.mean(values)),
                    "ci95_vs_0.5": [float(ci[0]), float(ci[1])],
                    "n_seeds": int(values.size),
                }
            else:
                summary[dataset] = {
                    "seed_values": {},
                    "mean": None,
                    "ci95_vs_0.5": None,
                    "n_seeds": 0,
                }
        return summary

    def _complete_group(
        self,
        latest: dict[RunKey, dict[str, Any]],
        dataset: str,
        seed: int,
    ) -> bool:
        records = [
            latest.get((arm, dataset, seed)) for arm in self.COMPOSITIONS
        ]
        if any(record is None for record in records):
            return False
        try:
            for record in records:
                self.validate_record(record)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _record_key(record: dict[str, Any]) -> RunKey:
        return str(record["composition"]), str(record["dataset"]), int(record["seed"])

    def _load_artifact(self) -> dict[str, Any]:
        if not self.output_path.exists():
            return self._empty_artifact()
        artifact = json.loads(self.output_path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict) or not isinstance(artifact.get("records"), list):
            raise TypeError(f"Invalid experiment artifact: {self.output_path}")
        return artifact

    def _latest_records(self) -> dict[RunKey, dict[str, Any]]:
        return {
            self._record_key(record): record
            for record in self._load_artifact()["records"]
        }

    def _persist_group(
        self,
        dataset: str,
        seed: int,
        records: list[dict[str, Any]],
    ) -> None:
        artifact = self._load_artifact()
        if not artifact.get("provenance"):
            artifact["provenance"] = self.provenance()
        artifact["records"] = [
            record
            for record in artifact["records"]
            if not (record.get("dataset") == dataset and record.get("seed") == seed)
        ] + records
        artifact["config"] = self._config()
        artifact["updated_at"] = datetime.now(UTC).isoformat()
        self._atomic_write_json(self.output_path, artifact)

    def _empty_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "chem_lgbo_v1",
            "config": self._config(),
            "provenance": {},
            "stage": "not_started",
            "records": [],
            "screening": None,
            "analysis": None,
            "verdict": None,
            "updated_at": None,
        }

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "screening", "full", "report")
    )
    mode = parser.parse_args(argv).mode
    datasets = ["buchwald_sub4", "suzuki"]
    screening_seeds = [100, 200, 300, 400, 500]

    if mode == "preflight":
        result = ChemLGBOExperiment(
            PREFLIGHT_PATH,
            datasets=datasets,
            seeds=[100],
            n_iters=1,
        ).preflight()
    elif mode == "screening":
        ChemLGBOExperiment(
            PREFLIGHT_PATH,
            datasets=datasets,
            seeds=[100],
            n_iters=1,
        ).verify_preflight()
        result = ChemLGBOExperiment(
            FULL_PATH,
            datasets=datasets,
            seeds=screening_seeds,
            n_iters=40,
        ).run_screening()
    else:
        kwargs: dict[str, Any] = {}
        if mode == "report":
            artifact = json.loads(FULL_PATH.read_text(encoding="utf-8"))
            if artifact.get("stage") in {"screening_passed", "screening_stopped"}:
                kwargs["seeds"] = screening_seeds
        experiment = ChemLGBOExperiment(FULL_PATH, **kwargs)
        result = experiment.run_full() if mode == "full" else experiment.report()

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
