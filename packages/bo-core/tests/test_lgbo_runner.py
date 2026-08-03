from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from bo_core.benchmark.data_loader import DATA_LOADERS
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


@pytest.mark.parametrize(
    ("method", "expected_engine", "use_llm"),
    [
        ("gpbo", "legacy", False),
        ("lgbo", "legacy", True),
        ("chem_lgbo", "chem", True),
    ],
)
def test_run_one_routes_methods_and_saves_compact_payload(
    tmp_path: Path,
    method: str,
    expected_engine: str,
    use_llm: bool,
):
    engine = MagicMock()
    engine.trajectory = [
        {
            "step": 1,
            "observed_yield": 80.0,
            "guidance": {"status": "applied", "reason": "accepted"},
        }
    ]
    engine.train_df = list(range(35))
    engine.encoder.dim = 32
    engine.guidance_artifacts = [
        {
            "raw_response": "private",
            "prompt": "private",
            "mask": [True],
            "counterfactual_indices": [0],
        }
    ]

    with (
        patch(
            "bo_core.benchmark.lgbo_runner.LGBOEngine", return_value=engine
        ) as legacy_cls,
        patch(
            "bo_core.benchmark.lgbo_runner.ChemLGBOEngine",
            return_value=engine,
            create=True,
        ) as chem_cls,
        patch("bo_core.benchmark.lgbo_runner._save_pt") as save_pt,
    ):
        result = run_one(
            dataset="buchwald_sub4",
            method=method,
            seed=100,
            n_iters=1,
            output_dir=tmp_path,
            n_restarts=1,
            backend="botorch",
        )

    selected_cls = legacy_cls if expected_engine == "legacy" else chem_cls
    other_cls = chem_cls if expected_engine == "legacy" else legacy_cls
    selected_cls.assert_called_once()
    other_cls.assert_not_called()
    assert selected_cls.call_args.kwargs["use_llm"] is use_llm

    save_dir = tmp_path / "botorch" / "buchwald_sub4" / method
    assert selected_cls.call_args.kwargs["backend"] == "botorch"
    assert selected_cls.call_args.kwargs["failure_log"] == str(
        save_dir / "seed_100_llm_failures.log"
    )
    assert (save_dir / "seed_100.csv").exists()
    assert save_pt.call_args.args[0] == save_dir / "seed_100.pt"
    pt_payload = save_pt.call_args.args[1]
    assert set(pt_payload) == {
        "seed",
        "dataset",
        "method",
        "backend",
        "prior_protocol",
        "n_train_prior",
        "encoder_dim",
        "trajectory",
    }
    assert pt_payload["trajectory"] is engine.trajectory
    assert engine.guidance_artifacts not in pt_payload.values()
    assert result["backend"] == "botorch"
    assert result["prior_protocol"] == "fixed_train_prior"
    assert result["n_train_prior"] == 35
    assert result["encoder_dim"] == 32


def test_run_one_rejects_unknown_method_before_engine_construction(
    tmp_path: Path,
):
    with (
        patch("bo_core.benchmark.lgbo_runner.LGBOEngine") as engine_cls,
        pytest.raises(ValueError, match="Unknown method"),
    ):
        run_one(
            dataset="buchwald_sub4",
            method="typo",
            seed=100,
            n_iters=1,
            output_dir=tmp_path,
            backend="sklearn",
        )

    engine_cls.assert_not_called()


def test_run_one_smokes_real_gpbo_and_fake_chem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = DATA_LOADERS["buchwald_sub4"]()
    field = data["feature_cols"][0]
    value = str(data["test_df"][field].iloc[0])
    client = MagicMock()
    client.is_configured.return_value = True
    client.chat.return_value = SimpleNamespace(
        status="success",
        content="",
        error=None,
        usage={},
        tool_calls=[
            {
                "id": "call-smoke",
                "type": "function",
                "function": {
                    "name": "propose_sparse_subspace",
                    "arguments": f'{{"subspace":{{"{field}":["{value}"]}}}}',
                },
            }
        ],
    )
    monkeypatch.setattr(
        "bo_core.llm_client.DeepSeekClient.from_env", lambda: client
    )

    for method in ("gpbo", "chem_lgbo"):
        run_one(
            dataset="buchwald_sub4",
            method=method,
            seed=100,
            n_iters=1,
            output_dir=tmp_path,
            n_restarts=1,
            backend="sklearn",
        )
        save_dir = tmp_path / "sklearn" / "buchwald_sub4" / method
        assert (save_dir / "seed_100.csv").is_file()
        payload = torch.load(save_dir / "seed_100.pt", weights_only=False)
        assert set(payload) == {
            "seed",
            "dataset",
            "method",
            "backend",
            "prior_protocol",
            "n_train_prior",
            "encoder_dim",
            "trajectory",
        }
        row = payload["trajectory"][0]
        query_index = row["query_index"]
        assert 0 <= query_index < len(data["test_df"])
        assert row["condition"] == {
            name: str(data["test_df"][name].iloc[query_index])
            for name in data["feature_cols"]
        }
        assert not {
            "raw_response",
            "prompt",
            "mask",
            "counterfactual_indices",
        } & row.keys()

    assert client.chat.call_count == 1
