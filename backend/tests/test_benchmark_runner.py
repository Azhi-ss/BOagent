from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from benchmark.data_loader import BAND_ALIGNMENT_FEATURES, TARGET_COL
from benchmark.runner import BenchmarkRunner


def _create_test_excel() -> Path:
    """Create a minimal test Excel file with 30 rows of band_alignment data."""
    rng = np.random.RandomState(42)
    n_rows = 30
    data = {}
    for col in BAND_ALIGNMENT_FEATURES:
        data[col] = rng.uniform(1.0, 5.0, n_rows)
    data[TARGET_COL] = rng.uniform(15.0, 25.0, n_rows)
    df = pd.DataFrame(data)

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    df.to_excel(tmp.name, index=False)
    return Path(tmp.name)


class MockPVKBO:
    """Minimal mock of PVKBO that records calls and returns fake data."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame()
        self.llm_query_cost: list[float] = []
        self.llm_query_time: list[float] = []
        self.acq_func = None

    def optimize(self, test_metric="generalization_score"):
        # Return fake optimization trace
        feature_cols = self.kwargs["task_context"]["feature_cols"]
        configs = pd.DataFrame(
            [{col: 4.0 for col in feature_cols}] * 5
        )
        fvals = pd.DataFrame(
            {
                "score": [22.0, 22.5, 23.0, 23.5, 24.0],
                "generalization_score": [21.0, 21.5, 22.0, 22.5, 23.0],
            }
        )
        self.llm_query_cost = [0.01] * 5
        self.llm_query_time = [0.5] * 5
        self.observed_configs = configs
        self.observed_fvals = fvals
        return configs, fvals


class TestBenchmarkRunner:
    def test_init_validates_task_id(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            BenchmarkRunner(task_id="invalid_task")

    def test_init_validates_sm_mode(self):
        with pytest.raises(ValueError, match="Unknown sm_mode"):
            BenchmarkRunner(task_id="band_alignment", sm_mode="invalid")

    def test_run_produces_result_dict(self, monkeypatch):
        path = _create_test_excel()
        tmp_dir = Path(tempfile.mkdtemp())

        import sys
        try:
            # Mock the PVKBO import
            monkeypatch.setattr(
                "benchmark.runner._resolve_pvk_root",
                lambda: Path("/fake/pvk"),
            )

            with patch(
                "benchmark.runner.BayesianOptimizer",
                autospec=True,
            ) as mock_opt_class:
                mock_opt = MagicMock()
                # Mock suggest result
                from optimization.knowledge import SuggestionResult
                mock_opt.suggest.return_value = SuggestionResult(
                    suggestions=[{col: 4.0 for col in BAND_ALIGNMENT_FEATURES}],
                    analysis="mock",
                    prompt="mock"
                )
                mock_opt_class.return_value = mock_opt

                # Inject mock PVKBO module
                sys.modules["pvk_bo"] = MagicMock()
                sys.modules["pvk_bo.pvk_bo"] = MagicMock()
                sys.modules["pvk_bo.pvk_bo"].PVKBO = MockPVKBO

                runner = BenchmarkRunner(
                    task_id="band_alignment",
                    n_initial=5,
                    n_trials=3,
                    seed=42,
                    output_dir=tmp_dir,
                    data_path=path,
                )

                result = runner.run()

                assert result["task_id"] == "band_alignment"
                assert result["seed"] == 42
                assert "best_score" in result
                assert "best_generalization_score" in result
                assert isinstance(result["search_history"], pd.DataFrame)
                assert isinstance(result["best_config"], dict)
        finally:
            path.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            # Clean up mock modules
            for mod in ["pvk_bo.pvk_bo", "pvk_bo"]:
                sys.modules.pop(mod, None)
    def test_save_results_creates_output_files(self):
        path = _create_test_excel()
        tmp_dir = Path(tempfile.mkdtemp())

        try:
            runner = BenchmarkRunner(
                task_id="band_alignment",
                n_initial=5,
                n_trials=3,
                seed=42,
                output_dir=tmp_dir,
                data_path=path,
            )

            result = {
                "task_id": "band_alignment",
                "seed": 42,
                "search_history": pd.DataFrame({"score": [22.0, 23.0]}),
                "best_config": {"CHI_PVK": 4.0},
                "best_score": 23.0,
                "best_generalization_score": 22.0,
                "llm_query_cost": [0.01, 0.01],
                "llm_query_time": [0.5, 0.5],
                "fvals": pd.DataFrame({
                    "score": [22.0, 23.0],
                    "generalization_score": [21.0, 22.0],
                }),
            }

            runner.save_results(result)

            save_dir = tmp_dir / "results_discriminative" / "band_alignment"
            assert save_dir.exists()
            assert (save_dir / "42.csv").exists()
            assert (save_dir / "42_search_info.json").exists()
            assert (save_dir / "42_summary.json").exists()

            # Verify summary content
            with open(save_dir / "42_summary.json") as f:
                summary = json.load(f)
            assert summary["best_score"] == 23.0
            assert summary["n_trials"] == 3
            assert summary["convergence_curve"] == [22.0, 23.0]
        finally:
            path.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_comparison_runner_aggregate_does_not_drop_mean(self):
        from benchmark.comparison import ComparisonRunner
        
        runner = ComparisonRunner(
            task_id="band_alignment",
            n_initial=3,
            n_trials=5,
            seeds=[42, 7],
            traditional={"acquisition": "ei", "xi": 0.01, "kappa": 2.576},
            llmbo={"acquisition": "ucb", "xi": 0.01, "kappa": 2.576},
        )
        
        trajectories = [
            {
                "trad_best": [10.0, 15.0, 20.0, 25.0],
                "trad_gen": [10.0, 15.0, 20.0, 25.0],
                "llm_best": [10.0, 15.0, 20.0, 26.0],
                "llm_gen": [10.0, 15.0, 20.0, 26.0],
            },
            {
                "trad_best": [8.0],
                "trad_gen": [8.0],
                "llm_best": [8.0],
                "llm_gen": [8.0],
            }
        ]
        
        points = runner._aggregate(trajectories)
        
        assert len(points) == 6
        assert points[0]["trad_best_mean"] == 9.0
        assert points[3]["trad_best_mean"] == 16.5
        assert points[3]["llm_best_mean"] == 17.0
        assert points[5]["trad_best_mean"] == 16.5
        assert points[5]["llm_best_mean"] == 17.0

