"""Resumable fixed-prior comparison of the Manifold and DKL LGBO hybrids."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import components.library  # noqa: F401  # force component registration
from analyze import (
    GLOBAL_BEST,
    aggregate_results,
    assert_seed_completeness,
    composite_score,
    trajectory_analysis,
    write_report,
)
from components.protocol import Composition
from compositions.base import get_base_compositions
from engine import HybridEngine, compute_metrics

RunKey = tuple[str, str, int]
EngineFactory = Callable[..., HybridEngine]
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _configure_numerical_threads(n_threads: int) -> None:
    """Apply the H365 thread budget at the process boundary."""
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(n_threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n_threads)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(n_threads)
    except ImportError:
        pass


class HybridComparisonRunner:
    """Run and validate the exact 20-seed hybrid comparison matrix."""

    COMPOSITIONS: ClassVar[list[str]] = ["lgbo_manifold", "lgbo_dkl"]
    DATASETS: ClassVar[list[str]] = ["buchwald_sub4", "suzuki"]
    SEEDS: ClassVar[list[int]] = [i * 100 for i in range(1, 21)]
    N_ITERS: ClassVar[int] = 40
    MATRIX_SIZE: ClassVar[int] = 80
    PRIOR_PROTOCOL: ClassVar[str] = "fixed_train_prior"
    EXPECTED_N_TRAIN: ClassVar[dict[str, int]] = {
        "buchwald_sub4": 35,
        "suzuki": 29,
    }
    MAX_LLM_FAILURE_RATE: ClassVar[float] = 0.10

    def __init__(
        self,
        output_path: Path,
        n_iters: int = N_ITERS,
        seeds: list[int] | None = None,
        datasets: list[str] | None = None,
        status_path: Path | None = None,
        workers: int = 1,
    ) -> None:
        self.output_path = output_path
        self.status_path = status_path
        self.workers = max(1, int(workers))
        self.N_ITERS = int(n_iters)
        self.SEEDS = list(seeds if seeds is not None else type(self).SEEDS)
        self.DATASETS = list(datasets if datasets is not None else type(self).DATASETS)
        self.COMPOSITIONS = list(type(self).COMPOSITIONS)
        self.MATRIX_SIZE = len(self.COMPOSITIONS) * len(self.DATASETS) * len(self.SEEDS)
        self._engine_factory: EngineFactory | None = None
        self._compositions = {
            composition.name: composition for composition in get_base_compositions()
        }
        missing = set(self.COMPOSITIONS) - set(self._compositions)
        if missing:
            raise ValueError(f"Missing hybrid compositions: {sorted(missing)}")

    @classmethod
    def preflight(
        cls,
        output_path: Path,
        status_path: Path | None = None,
    ) -> HybridComparisonRunner:
        """Build a one-seed, one-iteration real-data preflight runner."""
        return cls(
            output_path=output_path,
            n_iters=1,
            seeds=[100],
            status_path=status_path,
        )

    @staticmethod
    def verify_environment() -> None:
        """Fail before an experiment if the DeepSeek client is not configured."""
        from bo_core.llm_client import DeepSeekClient

        if not DeepSeekClient.from_env().is_configured():
            raise RuntimeError(
                "DeepSeek is not configured; set the API key in the project .env"
            )

    def pending_keys(self) -> list[RunKey]:
        latest = self._latest_records()
        return [
            key for key in self._matrix_keys()
            if not self._is_valid_record(latest.get(key))
        ]

    def run_one(
        self,
        composition: Composition,
        dataset: str,
        seed: int,
    ) -> dict[str, Any]:
        """Run one matrix unit and persist its latest result atomically."""
        started = time.monotonic()
        engine: HybridEngine | None = None
        try:
            factory = self._engine_factory or HybridEngine
            engine = factory(
                composition,
                dataset,
                seed=seed,
                n_iters=self.N_ITERS,
            )
            trajectory = engine.run()
            record: dict[str, Any] = {
                "composition": composition.name,
                "dataset": dataset,
                "seed": seed,
                "prior_protocol": self.PRIOR_PROTOCOL,
                "n_train_prior": len(engine.initial_indices),
                "initial_indices": list(engine.initial_indices),
                "elapsed_s": round(time.monotonic() - started, 3),
                "metrics": compute_metrics(trajectory, GLOBAL_BEST[dataset]),
                "trajectory": trajectory,
                "diagnostics": engine.diagnostics,
                "status": "ok",
            }
            validation_error = self._record_validation_error(record)
            if validation_error is not None:
                record = {
                    **record,
                    "status": "failed",
                    "error": f"Validation gate failed: {validation_error}",
                }
        except Exception as exc:  # noqa: BLE001 - persist failed experiment units
            record = {
                "composition": composition.name,
                "dataset": dataset,
                "seed": seed,
                "prior_protocol": self.PRIOR_PROTOCOL,
                "n_train_prior": len(engine.initial_indices) if engine is not None else 0,
                "initial_indices": (
                    list(engine.initial_indices) if engine is not None else []
                ),
                "elapsed_s": round(time.monotonic() - started, 3),
                "diagnostics": engine.diagnostics if engine is not None else None,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        self._persist_latest(record)
        return record

    def run_all(self) -> None:
        """Run each currently pending matrix unit once."""
        pending = self.pending_keys()
        self._write_status(
            "running",
            pending_before=len(pending),
            workers=self.workers,
        )
        if self.workers == 1:
            self._run_serial(pending)
        else:
            self._run_parallel(pending)
        self._write_status("complete" if self.is_complete() else "incomplete")

    def _run_serial(self, pending: list[RunKey]) -> None:
        for composition_name, dataset, seed in pending:
            record = self.run_one(
                self._compositions[composition_name],
                dataset,
                seed,
            )
            self._log_record(record)
            self._write_status(
                "running",
                last_key=[composition_name, dataset, seed],
                workers=self.workers,
            )

    def _run_parallel(self, pending: list[RunKey]) -> None:
        """Fan out by (dataset, seed) groups; keep Manifold→DKL order inside each."""
        groups = self._pending_seed_groups(pending)
        if not groups:
            return
        n_threads = 1  # multi-worker H365 policy
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_configure_numerical_threads,
            initargs=(n_threads,),
        ) as pool:
            futures = {
                pool.submit(
                    _run_seed_group_worker,
                    str(self.output_path),
                    str(self.status_path) if self.status_path is not None else None,
                    self.N_ITERS,
                    self.SEEDS,
                    self.DATASETS,
                    dataset,
                    seed,
                    compositions,
                ): (dataset, seed, compositions)
                for dataset, seed, compositions in groups
            }
            for future in as_completed(futures):
                dataset, seed, compositions = futures[future]
                try:
                    records = future.result()
                except Exception as exc:  # noqa: BLE001 - surface worker crash
                    print(
                        f"[FAILED] worker {dataset}/seed{seed}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    self._write_status(
                        "running",
                        last_key=[compositions[0], dataset, seed],
                        workers=self.workers,
                        worker_error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                for record in records:
                    self._log_record(record)
                    self._write_status(
                        "running",
                        last_key=[
                            record["composition"],
                            record["dataset"],
                            record["seed"],
                        ],
                        workers=self.workers,
                    )

    def _pending_seed_groups(
        self,
        pending: list[RunKey],
    ) -> list[tuple[str, int, list[str]]]:
        """Collapse pending keys into ordered (dataset, seed, compositions) groups."""
        groups: list[tuple[str, int, list[str]]] = []
        current: tuple[str, int] | None = None
        compositions: list[str] = []
        for composition, dataset, seed in pending:
            key = (dataset, seed)
            if current is None:
                current = key
                compositions = [composition]
                continue
            if key == current:
                compositions.append(composition)
                continue
            groups.append((current[0], current[1], compositions))
            current = key
            compositions = [composition]
        if current is not None:
            groups.append((current[0], current[1], compositions))
        return groups

    @staticmethod
    def _log_record(record: dict[str, Any]) -> None:
        metrics = record.get("metrics", {})
        if record["status"] == "ok":
            detail = (
                f"best={metrics.get('best_found'):.2f} "
                f"t95={metrics.get('t95')}"
            )
        else:
            detail = record.get("error", "unknown failure")
        print(
            f"[{record['status'].upper()}] {record['composition']}/"
            f"{record['dataset']}/seed{record['seed']}: {detail}",
            flush=True,
        )

    def aggregate(self) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
        """Aggregate the latest records that pass all protocol gates."""
        valid = self.valid_records()
        if not valid:
            raise ValueError("No valid hybrid comparison records to aggregate")
        return aggregate_results(valid)

    def valid_records(self) -> list[dict[str, Any]]:
        latest = self._latest_records()
        return [
            latest[key]
            for key in self._matrix_keys()
            if self._is_valid_record(latest.get(key))
        ]

    def is_complete(self) -> bool:
        return len(self.valid_records()) == self.MATRIX_SIZE

    def write_final_report(self, output_dir: Path) -> Path:
        """Validate the complete matrix, then write the comparison report."""
        valid = self.valid_records()
        expected = set(self._matrix_keys())
        actual = {self._record_key(record) for record in valid}
        if actual != expected:
            missing = sorted(expected - actual)
            raise ValueError(
                f"Hybrid comparison incomplete: {len(actual)}/{len(expected)} valid; "
                f"missing={missing[:5]}"
            )
        assert_seed_completeness(valid, self.SEEDS, self.DATASETS)
        summary = aggregate_results(valid)
        scores = composite_score(summary)
        analyses = {
            composition: self._diagnostic_totals(valid, composition)
            for composition in self.COMPOSITIONS
        }
        return write_report(output_dir, summary, scores, analyses)

    @staticmethod
    def is_valid_record(record: dict[str, Any]) -> bool:
        """Validate a record against the canonical 40-iteration protocol."""
        return HybridComparisonRunner._record_validation_error_for(
            record,
            HybridComparisonRunner.N_ITERS,
        ) is None

    @staticmethod
    def _diagnostics_ok(diagnostics: dict[str, Any]) -> bool:
        """Validate LLM diagnostics for the canonical 40-iteration protocol."""
        return HybridComparisonRunner._diagnostics_error_for(
            diagnostics,
            HybridComparisonRunner.N_ITERS,
        ) is None

    def _is_valid_record(self, record: dict[str, Any] | None) -> bool:
        return record is not None and self._record_validation_error(record) is None

    def _record_validation_error(self, record: dict[str, Any]) -> str | None:
        return self._record_validation_error_for(record, self.N_ITERS)

    @classmethod
    def _record_validation_error_for(
        cls,
        record: dict[str, Any],
        n_iters: int,
    ) -> str | None:
        if record.get("status") != "ok":
            return "status is not ok"
        if record.get("prior_protocol") != cls.PRIOR_PROTOCOL:
            return "prior_protocol is not fixed_train_prior"
        dataset = record.get("dataset")
        if dataset not in cls.EXPECTED_N_TRAIN:
            return f"unknown dataset {dataset!r}"
        if record.get("composition") not in cls.COMPOSITIONS:
            return f"unknown composition {record.get('composition')!r}"
        expected_prior = cls.EXPECTED_N_TRAIN[dataset]
        if record.get("n_train_prior") != expected_prior:
            return f"n_train_prior is not {expected_prior}"
        if record.get("initial_indices") != list(range(expected_prior)):
            return "initial_indices do not cover the complete fixed prior"
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            return "metrics are missing"
        for name in (
            "best_found",
            "initial_round_found_best",
            "t95",
            "AUC_best_so_far",
        ):
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return f"metric {name} is missing or non-finite"
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != n_iters:
            return f"trajectory does not contain {n_iters} iterations"
        for index, step in enumerate(trajectory, start=1):
            if not isinstance(step, dict):
                return f"trajectory step {index} is invalid"
            observed = step.get("observed_yield")
            if not isinstance(observed, (int, float)) or not math.isfinite(observed):
                return f"trajectory step {index} has invalid observed_yield"
            if step.get("step") != index:
                return f"trajectory step {index} has invalid step number"
            if not isinstance(step.get("acquisition"), str):
                return f"trajectory step {index} has invalid acquisition"
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return "diagnostics are missing"
        return cls._diagnostics_error_for(diagnostics, n_iters)

    @classmethod
    def _diagnostics_error_for(
        cls,
        diagnostics: dict[str, Any],
        n_iters: int,
    ) -> str | None:
        summary = diagnostics.get("summary")
        if not isinstance(summary, dict):
            return "diagnostics summary is missing"
        for name in ("iterations_recorded", "iterations_completed", "llm_attempts"):
            if summary.get(name) != n_iters:
                return f"{name} is not {n_iters}"
        if summary.get("llm_not_configured") != 0:
            return "LLM was not configured"
        if summary.get("llm_diagnostics_missing") != 0:
            return "LLM diagnostics are missing"
        for name in (
            "crashed_iterations",
            "gp_fit_failures",
            "acquisition_fallbacks",
            "degenerate_score_iterations",
            "nonfinite_acquisition_scores",
            "mean_shift_failures",
            "surrogate_fallback_fits",
        ):
            if summary.get(name) != 0:
                return f"{name} is not zero"

        successes = summary.get("llm_successes")
        failures = summary.get("llm_failures")
        if not isinstance(successes, int) or not isinstance(failures, int):
            return "LLM success/failure counts are invalid"
        if successes + failures != n_iters:
            return "LLM outcomes do not account for every iteration"
        if failures / n_iters > cls.MAX_LLM_FAILURE_RATE:
            return "LLM failure rate exceeds 10%"
        return None

    def _matrix_keys(self) -> list[RunKey]:
        return [
            (composition, dataset, seed)
            for dataset in self.DATASETS
            for seed in self.SEEDS
            for composition in self.COMPOSITIONS
        ]

    @staticmethod
    def _record_key(record: dict[str, Any]) -> RunKey:
        composition = record.get("composition")
        dataset = record.get("dataset")
        seed = record.get("seed")
        if not isinstance(composition, str) or not isinstance(dataset, str):
            raise TypeError(f"Invalid result record key: {record!r}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"Invalid result record key: {record!r}")
        return composition, dataset, seed

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.output_path.exists():
            return []
        loaded = json.loads(self.output_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(isinstance(record, dict) for record in loaded):
            raise ValueError(f"Invalid result file schema: {self.output_path}")
        return loaded

    def _latest_records(self) -> dict[RunKey, dict[str, Any]]:
        return {self._record_key(record): record for record in self._load_records()}

    def _persist_latest(self, record: dict[str, Any]) -> None:
        with self._file_lock(self.output_path):
            latest = self._latest_records()
            latest[self._record_key(record)] = record
            matrix_order = {key: index for index, key in enumerate(self._matrix_keys())}
            records = sorted(
                latest.values(),
                key=lambda item: matrix_order.get(
                    self._record_key(item),
                    len(matrix_order),
                ),
            )
            self._atomic_write_json(self.output_path, records)

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    @contextmanager
    def _file_lock(path: Path) -> Iterator[None]:
        """Exclusive lock around read-modify-write of the shared result file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write_status(self, state: str, **extra: Any) -> None:
        if self.status_path is None:
            return
        with self._file_lock(self.status_path):
            valid_count = len(self.valid_records())
            try:
                result_file = str(self.output_path.relative_to(ROOT))
            except ValueError:
                result_file = str(self.output_path)
            payload = {
                "state": state,
                "mode": "preflight" if self.N_ITERS == 1 else "full",
                "valid_runs": valid_count,
                "required_runs": self.MATRIX_SIZE,
                "pending_runs": self.MATRIX_SIZE - valid_count,
                "result_file": result_file,
                "workers": self.workers,
                "updated_at": datetime.now(UTC).isoformat(),
                **extra,
            }
            self._atomic_write_json(self.status_path, payload)

    @staticmethod
    def _diagnostic_totals(
        records: list[dict[str, Any]],
        composition: str,
    ) -> dict[str, Any]:
        selected = [record for record in records if record["composition"] == composition]
        per_run = [trajectory_analysis(record["trajectory"]) for record in selected]
        action_types = sorted({
            action
            for analysis in per_run
            for action in analysis["llm_action_types"]
        })
        return {
            "n_improvements": sum(analysis["n_improvements"] for analysis in per_run),
            "llm_action_count": sum(analysis["llm_action_count"] for analysis in per_run),
            "llm_action_types": action_types,
            "acq_switches": sum(analysis["acq_switches"] for analysis in per_run),
            "final_best": max(analysis["final_best"] for analysis in per_run),
        }


def _run_seed_group_worker(
    output_path: str,
    status_path: str | None,
    n_iters: int,
    seeds: list[int],
    datasets: list[str],
    dataset: str,
    seed: int,
    compositions: list[str],
) -> list[dict[str, Any]]:
    """Worker entry: run Manifold then DKL for one (dataset, seed), persist each."""
    runner = HybridComparisonRunner(
        output_path=Path(output_path),
        n_iters=n_iters,
        seeds=seeds,
        datasets=datasets,
        status_path=Path(status_path) if status_path else None,
        workers=1,
    )
    records: list[dict[str, Any]] = []
    for composition_name in compositions:
        record = runner.run_one(
            runner._compositions[composition_name],
            dataset,
            seed,
        )
        records.append(record)
    return records


def _default_paths() -> tuple[Path, Path, Path, Path]:
    experiments = ROOT / "history" / "experiments"
    return (
        experiments / "lgbo_hybrid_fixed_prior_preflight.json",
        experiments / "lgbo_hybrid_fixed_prior_comparison.json",
        ROOT / "hybrid_comparison_status.json",
        ROOT / "reports" / "hybrid_fixed_prior",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "report", "status"))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process count for seed-parallel execution (1=serial, multi uses 1 thread/process).",
    )
    args = parser.parse_args()
    preflight_path, full_path, status_path, report_dir = _default_paths()

    if args.mode == "status":
        if status_path.exists():
            print(status_path.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"state": "not_started"}, indent=2))
        return 0

    preflight_runner = HybridComparisonRunner.preflight(
        preflight_path,
        status_path,
    )
    if args.mode == "preflight":
        HybridComparisonRunner.verify_environment()
        preflight_runner.run_all()
        return 0 if preflight_runner.is_complete() else 1

    if not preflight_runner.is_complete():
        raise RuntimeError("Real-data preflight must pass before the full comparison")

    workers = max(1, int(args.workers))
    if workers > 1:
        _configure_numerical_threads(1)
    else:
        _configure_numerical_threads(10)

    runner = HybridComparisonRunner(
        full_path,
        status_path=status_path,
        workers=workers,
    )
    if args.mode == "run":
        HybridComparisonRunner.verify_environment()
        runner.run_all()
        if not runner.is_complete():
            return 1
    report_path = runner.write_final_report(report_dir)
    try:
        report_file = str(report_path.relative_to(ROOT))
    except ValueError:
        report_file = str(report_path)
    runner._write_status("done", report_file=report_file)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
