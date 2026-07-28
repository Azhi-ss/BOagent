from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bo_core.benchmark.lgbo_runner import (
    THREAD_ENV_VARS,
    _aggregate,
    _configure_numerical_threads,
    main,
    run_one,
)


@pytest.mark.parametrize(("workers", "expected_threads"), [(1, 10), (4, 1)])
def test_main_configures_process_thread_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    expected_threads: int,
):
    result = {
        "dataset": "buchwald_sub4",
        "method": "gpbo",
        "backend": "sklearn",
        "seed": 100,
        "best_found": 80.0,
        "initial_round_found_best": 80.0,
        "t95": 2,
        "AUC_best_so_far": 80.0,
    }
    future = MagicMock()
    future.result.return_value = result
    executor = MagicMock()
    executor.__enter__.return_value.submit.return_value = future
    monkeypatch.setattr(
        "sys.argv",
        [
            "lgbo_runner",
            "--datasets",
            "buchwald_sub4",
            "--methods",
            "gpbo",
            "--seeds",
            "100",
            "--workers",
            str(workers),
            "--output_dir",
            str(tmp_path),
        ],
    )

    with (
        patch("bo_core.benchmark.lgbo_runner._configure_numerical_threads") as configure,
        patch(
            "bo_core.benchmark.lgbo_runner.ProcessPoolExecutor",
            return_value=executor,
        ) as pool,
        patch("bo_core.benchmark.lgbo_runner.as_completed", return_value=[future]),
    ):
        main()

    configure.assert_called_once_with(expected_threads)
    assert pool.call_args.kwargs["initializer"] is configure
    assert pool.call_args.kwargs["initargs"] == (expected_threads,)


def test_configure_numerical_threads_updates_all_runtimes(
    monkeypatch: pytest.MonkeyPatch,
):
    for name in THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with (
        patch("threadpoolctl.threadpool_limits") as threadpool_limits,
        patch("torch.set_num_threads") as torch_set_num_threads,
    ):
        _configure_numerical_threads(3)

    assert all(os.environ[name] == "3" for name in THREAD_ENV_VARS)
    threadpool_limits.assert_called_once_with(limits=3)
    torch_set_num_threads.assert_called_once_with(3)


def test_aggregate_rejects_mixed_backends():
    result = {
        "dataset": "buchwald_sub4",
        "method": "gpbo",
        "seed": 100,
        "best_found": 80.0,
        "initial_round_found_best": 70.0,
        "t95": 2,
        "AUC_best_so_far": 75.0,
    }

    with pytest.raises(ValueError, match="single backend"):
        _aggregate(
            [
                {**result, "backend": "sklearn"},
                {**result, "backend": "botorch", "seed": 200},
            ]
        )


def test_run_one_isolates_backend_outputs_and_records_metadata(tmp_path: Path):
    engine = MagicMock()
    engine.trajectory = [{"observed_yield": 80.0}]
    engine.train_df = list(range(35))
    engine.encoder.dim = 32

    with (
        patch("bo_core.benchmark.lgbo_runner.LGBOEngine", return_value=engine) as engine_cls,
        patch("bo_core.benchmark.lgbo_runner._save_pt") as save_pt,
    ):
        result = run_one(
            dataset="buchwald_sub4",
            method="gpbo",
            seed=100,
            n_iters=1,
            output_dir=tmp_path,
            n_restarts=1,
            backend="botorch",
        )

    save_dir = tmp_path / "botorch" / "buchwald_sub4" / "gpbo"
    assert engine_cls.call_args.kwargs["backend"] == "botorch"
    assert engine_cls.call_args.kwargs["failure_log"] == str(
        save_dir / "seed_100_llm_failures.log"
    )
    assert (save_dir / "seed_100.csv").exists()
    assert save_pt.call_args.args[0] == save_dir / "seed_100.pt"
    pt_payload = save_pt.call_args.args[1]
    assert pt_payload["backend"] == "botorch"
    assert pt_payload["prior_protocol"] == "fixed_train_prior"
    assert pt_payload["n_train_prior"] == 35
    assert pt_payload["encoder_dim"] == 32
    assert result["backend"] == "botorch"
    assert result["prior_protocol"] == "fixed_train_prior"
    assert result["n_train_prior"] == 35
    assert result["encoder_dim"] == 32
