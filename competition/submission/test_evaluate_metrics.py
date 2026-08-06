from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "bo-core"))
MODULE_PATH = Path(__file__).parent / "code" / "scripts" / "evaluate_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _oracle_bundle() -> SimpleNamespace:
    test = pd.DataFrame({"yield": np.arange(1.0, 101.0)})
    return SimpleNamespace(
        global_best=100.0,
        test=test,
        spec=SimpleNamespace(target="yield"),
    )


def _load_dataset_stub(_dataset: str, data_dir: Path | None = None) -> SimpleNamespace:
    del data_dir
    return _oracle_bundle()


def _payload(dataset: str, seed: int, backend: str = "botorch") -> dict[str, Any]:
    offset = seed // 100 - 1
    return {
        "dataset": dataset,
        "seed": seed,
        "method": "lgbo",
        "backend": backend,
        "trajectory": [
            {
                "step": step,
                "query_index": query_index,
                "observed_yield": float(query_index + 1),
            }
            for step in range(1, 41)
            for query_index in [step - 1 + offset]
        ],
    }


def _write_matrix(trajectories: Path, backend: str = "botorch") -> None:
    for dataset in MODULE.DATASETS:
        directory = trajectories / backend / dataset / MODULE.METHOD
        directory.mkdir(parents=True)
        for seed in MODULE.SEEDS:
            torch.save(_payload(dataset, seed, backend), directory / f"seed_{seed}.pt")


def _evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    monkeypatch.setattr(MODULE, "load_dataset", _load_dataset_stub)
    return MODULE.evaluate_all(trajectories, results, "botorch")


def test_valid_full_matrix_writes_exact_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _evaluate(tmp_path, monkeypatch)

    summary = pd.read_csv(output)
    expected_columns = ["dataset", "method", "backend"] + [
        f"{metric}_{stat}"
        for metric in MODULE.METRICS
        for stat in ("mean", "std", "ci95")
    ]
    assert summary.columns.tolist() == expected_columns
    assert summary["dataset"].tolist() == list(MODULE.DATASETS)
    assert summary["method"].tolist() == [MODULE.METHOD] * 2
    assert summary["backend"].tolist() == ["botorch"] * 2

    expected_runs = [
        MODULE._compute_metrics(
            [
                row["observed_yield"]
                for row in _payload(MODULE.DATASETS[0], seed)["trajectory"]
            ],
            100.0,
        )
        for seed in MODULE.SEEDS
    ]
    for row in summary.itertuples(index=False):
        for metric in MODULE.METRICS:
            values = [run[metric] for run in expected_runs]
            mean, std, ci95 = MODULE._stats(values)
            assert getattr(row, f"{metric}_mean") == pytest.approx(mean)
            assert getattr(row, f"{metric}_std") == pytest.approx(std)
            assert getattr(row, f"{metric}_ci95") == pytest.approx(ci95)


def test_metric_formulas() -> None:
    yields = [10.0, 95.0, 20.0] + [30.0] * 37

    assert MODULE._compute_metrics(yields, 100.0) == {
        "best_found": 95.0,
        "initial_round_found_best": 10.0,
        "t95": 2.0,
        "AUC_best_so_far": pytest.approx((10.0 + 39 * 95.0) / 40),
    }


def test_default_data_root_uses_registered_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    data_dirs: list[Path | None] = []

    def load_dataset_through_registry(
        _dataset: str, data_dir: Path | None = None
    ) -> SimpleNamespace:
        data_dirs.append(data_dir)
        return _oracle_bundle()

    monkeypatch.setattr(MODULE, "load_dataset", load_dataset_through_registry)

    MODULE.evaluate_all(trajectories, results, "botorch")

    assert data_dirs == [None, None]


@pytest.mark.parametrize("missing", ["dataset", "seed"])
def test_missing_matrix_entry_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    if missing == "dataset":
        directory = trajectories / "botorch" / MODULE.DATASETS[1] / MODULE.METHOD
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
    else:
        (trajectories / "botorch" / MODULE.DATASETS[0] / MODULE.METHOD / "seed_100.pt").unlink()
    monkeypatch.setattr(MODULE, "load_dataset", _load_dataset_stub)

    with pytest.raises(ValueError, match="required dataset/seed matrix"):
        MODULE.evaluate_all(trajectories, results, "botorch")

    assert not (results / "summary_metrics.csv").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["trajectory"][1].update(query_index=0), "unique"),
        (lambda payload: payload["trajectory"][0].update(query_index=100), "out of range"),
        (lambda payload: payload["trajectory"].pop(), "exactly 40"),
        (lambda payload: payload.update(dataset="suzuki"), "metadata 'dataset'"),
        (lambda payload: payload.update(seed=200), "metadata 'seed'"),
        (lambda payload: payload.update(method="gpbo"), "metadata 'method'"),
        (lambda payload: payload.update(backend="sklearn"), "metadata 'backend'"),
        (
            lambda payload: payload["trajectory"][0].update(observed_yield=float("nan")),
            "not finite",
        ),
        (
            lambda payload: payload["trajectory"][0].update(observed_yield=999.0),
            "does not match",
        ),
    ],
)
def test_invalid_trajectory_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    path = trajectories / "botorch" / MODULE.DATASETS[0] / MODULE.METHOD / "seed_100.pt"
    payload = torch.load(path, weights_only=True)
    mutation(payload)
    torch.save(payload, path)
    monkeypatch.setattr(MODULE, "load_dataset", _load_dataset_stub)

    with pytest.raises(ValueError, match=message):
        MODULE.evaluate_all(trajectories, results, "botorch")

    assert not (results / "summary_metrics.csv").exists()


def test_unexpected_or_duplicate_seed_file_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    source = trajectories / "botorch" / MODULE.DATASETS[0] / MODULE.METHOD / "seed_100.pt"
    unexpected = trajectories / "botorch" / MODULE.DATASETS[0] / MODULE.METHOD / "seed_999.pt"
    unexpected.write_bytes(source.read_bytes())
    monkeypatch.setattr(MODULE, "load_dataset", _load_dataset_stub)

    with pytest.raises(ValueError, match="extra=.*seed_999.pt"):
        MODULE.evaluate_all(trajectories, results, "botorch")

    assert not (results / "summary_metrics.csv").exists()


def test_corrupt_file_fails_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    path = trajectories / "botorch" / MODULE.DATASETS[0] / MODULE.METHOD / "seed_100.pt"
    path.write_bytes(b"not a torch payload")
    monkeypatch.setattr(MODULE, "load_dataset", _load_dataset_stub)

    with pytest.raises((EOFError, RuntimeError, pickle.UnpicklingError)):
        MODULE.evaluate_all(trajectories, results, "botorch")

    assert not (results / "summary_metrics.csv").exists()


def test_torch_load_falls_back_for_older_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "seed_100.pt"
    payload = _payload(MODULE.DATASETS[0], 100)
    calls: list[dict[str, Any]] = []

    def compatible_load(_path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if kwargs:
            raise TypeError("weights_only is unavailable")
        return payload

    monkeypatch.setattr(MODULE.torch, "load", compatible_load)

    assert MODULE._load_trajectory(
        path,
        dataset=MODULE.DATASETS[0],
        seed=100,
        backend="botorch",
        oracle_yields=_oracle_bundle().test["yield"].to_numpy(),
    ) == [row["observed_yield"] for row in payload["trajectory"]]
    assert calls == [{"weights_only": True}, {}]


def test_output_is_deferred_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = tmp_path / "trajectories"
    results = tmp_path / "results"
    _write_matrix(trajectories)
    monkeypatch.setattr(MODULE, "load_dataset", _load_dataset_stub)

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)

    with pytest.raises(OSError, match="synthetic write failure"):
        MODULE.evaluate_all(trajectories, results, "botorch")

    assert not (results / "summary_metrics.csv").exists()
    assert list(results.iterdir()) == []
