"""Verify competition metrics from generated optimization trajectories."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from bo_core.benchmark.data_loader import load_dataset

CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_ROOT.parent / "results"
TRAJECTORIES_DIR = RESULTS_DIR / "optimization_trajectories"
DATASETS = ("buchwald_sub4", "suzuki")
SEEDS = tuple(range(100, 2001, 100))
METHOD = "lgbo"
N_STEPS = 40
METRICS = (
    "best_found",
    "initial_round_found_best",
    "t95",
    "AUC_best_so_far",
)


def _load_trajectory(
    path: Path,
    *,
    dataset: str,
    seed: int,
    backend: str,
    oracle_yields: np.ndarray,
) -> list[float]:
    try:
        data = torch.load(path, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        data = torch.load(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: payload must be a dictionary")

    expected_metadata = {
        "dataset": dataset,
        "seed": seed,
        "method": METHOD,
        "backend": backend,
    }
    for key, expected in expected_metadata.items():
        if data.get(key) != expected:
            raise ValueError(
                f"{path}: metadata {key!r} is {data.get(key)!r}, expected {expected!r}"
            )

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list):
        raise TypeError(f"{path}: trajectory must be a list")
    if len(trajectory) != N_STEPS:
        raise ValueError(f"{path}: trajectory must contain exactly {N_STEPS} rows")

    query_indices: set[int] = set()
    observed_yields: list[float] = []
    for expected_step, row in enumerate(trajectory, 1):
        if not isinstance(row, dict):
            raise TypeError(f"{path}: trajectory row {expected_step} must be a dictionary")
        if row.get("step") != expected_step:
            raise ValueError(
                f"{path}: trajectory steps must be exactly 1 through {N_STEPS}"
            )
        query_index = row.get("query_index")
        if isinstance(query_index, bool) or not isinstance(query_index, int):
            raise TypeError(f"{path}: query_index at step {expected_step} must be an integer")
        if not 0 <= query_index < len(oracle_yields):
            raise ValueError(f"{path}: query_index at step {expected_step} is out of range")
        if query_index in query_indices:
            raise ValueError(f"{path}: query_index values must be unique")
        query_indices.add(query_index)

        try:
            observed_yield = float(row["observed_yield"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: invalid observed_yield at step {expected_step}"
            ) from exc
        if not math.isfinite(observed_yield):
            raise ValueError(f"{path}: observed_yield at step {expected_step} is not finite")
        if observed_yield != float(oracle_yields[query_index]):
            raise ValueError(
                f"{path}: observed_yield at step {expected_step} does not match "
                f"dataset query_index {query_index}"
            )
        observed_yields.append(observed_yield)
    return observed_yields


def _compute_metrics(yields: list[float], global_best: float) -> dict[str, float]:
    best_so_far = np.maximum.accumulate(np.asarray(yields, dtype=float))
    target = 0.95 * global_best
    reached = np.flatnonzero(best_so_far >= target)
    return {
        "best_found": float(best_so_far[-1]),
        "initial_round_found_best": float(best_so_far[0]),
        "t95": float(reached[0] + 1) if reached.size else float(N_STEPS + 1),
        "AUC_best_so_far": float(np.mean(best_so_far)),
    }


def _stats(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return mean, std, ci95




def evaluate_all(
    trajectories_path: Path = TRAJECTORIES_DIR,
    results_path: Path = RESULTS_DIR,
    backend: str = "botorch",
    data_root: Path | None = None,
) -> Path:
    """Validate the complete submission matrix, then atomically write its summary."""
    trajectories_path = Path(trajectories_path)
    data_root = Path(data_root) if data_root is not None else None
    results_path = Path(results_path)
    metrics_by_dataset: dict[str, list[dict[str, float]]] = {}

    expected_paths = {
        Path(backend) / dataset / METHOD / f"seed_{seed}.pt"
        for dataset in DATASETS
        for seed in SEEDS
    }
    actual_paths = {
        path.relative_to(trajectories_path)
        for path in trajectories_path.rglob("*.pt")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(map(str, expected_paths - actual_paths))
        extra = sorted(map(str, actual_paths - expected_paths))
        raise ValueError(
            "Trajectory files do not match the required dataset/seed matrix; "
            f"missing={missing}, extra={extra}"
        )

    for dataset in DATASETS:
        dataset_dir = trajectories_path / backend / dataset / METHOD

        bundle = load_dataset(
            dataset,
            data_dir=data_root / dataset if data_root is not None else None,
        )
        global_best = float(bundle.global_best)
        if not math.isfinite(global_best):
            raise ValueError(f"{dataset}: global_best must be finite")
        oracle_yields = bundle.test[bundle.spec.target].to_numpy(dtype=float)
        if not np.isfinite(oracle_yields).all():
            raise ValueError(f"{dataset}: test yields must be finite")
        metrics_by_dataset[dataset] = [
            _compute_metrics(
                _load_trajectory(
                    dataset_dir / f"seed_{seed}.pt",
                    dataset=dataset,
                    seed=seed,
                    backend=backend,
                    oracle_yields=oracle_yields,
                ),
                global_best,
            )
            for seed in SEEDS
        ]

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": METHOD,
            "backend": backend,
        }
        for metric in METRICS:
            mean, std, ci95 = _stats(
                [run[metric] for run in metrics_by_dataset[dataset]]
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)

    results_path.mkdir(parents=True, exist_ok=True)
    output_path = results_path / "summary_metrics.csv"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=results_path,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        pd.DataFrame(rows).to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"Verified {len(DATASETS) * len(SEEDS)} trajectories: {output_path}")
    return output_path


if __name__ == "__main__":
    evaluate_all()
