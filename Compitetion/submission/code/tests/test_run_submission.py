"""Tests for the competition submission entry point."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from main import run_submission


def test_main_runs_lgbo_with_40_query_budget(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run_one(
        dataset,
        method,
        seed,
        n_iters,
        output_dir,
        *,
        backend,
    ):
        calls.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "n_iters": n_iters,
                "output_dir": output_dir,
                "backend": backend,
            }
        )
        return {"best_found": 80.0}

    monkeypatch.setattr(run_submission, "SEEDS", [100])
    monkeypatch.setattr(run_submission, "DATASETS", ["buchwald_sub4"])
    monkeypatch.setattr(run_submission, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(run_submission, "run_one", fake_run_one)

    run_submission.main()

    assert calls == [
        {
            "dataset": "buchwald_sub4",
            "method": "lgbo",
            "seed": 100,
            "n_iters": 40,
            "output_dir": tmp_path,
            "backend": "botorch",
        }
    ]
