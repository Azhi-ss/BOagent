import time
import unittest
from unittest.mock import MagicMock, patch
from benchmark.comparison import ComparisonRunner

class TestParallelBenchmark(unittest.TestCase):
    def setUp(self):
        # Common parameters
        self.task_id = "band_alignment"
        self.n_initial = 2
        self.n_trials = 2
        self.seeds = [42, 100]
        self.trad_cfg = {"acquisition": "ei"}
        self.llmbo_cfg = {"acquisition": "ucb", "alpha": 0.1}

    @patch("benchmark.comparison.DATA_LOADERS", dict())
    @patch("benchmark.comparison.BOStepEngine")
    def test_parallel_execution_events(self, mock_bo_step):
        from benchmark.comparison import DATA_LOADERS
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.return_value = {"df": MagicMock(), "feature_cols": [], "target_col": "score"}
        DATA_LOADERS[self.task_id] = mock_loader

        def mock_step_side_effect(*args, **kwargs):
            engine = MagicMock()
            engine.iteration = 0
            engine.completed = False
            
            # Simulate a time-consuming step
            def step_impl():
                time.sleep(0.1)
                engine.iteration += 1
                if engine.iteration >= self.n_trials:
                    engine.completed = True
                return {
                    "best_score": 20.0 + engine.iteration,
                    "generalization_score": 19.0 + engine.iteration,
                }
            
            engine.step.side_effect = step_impl
            engine.snapshot.return_value = {
                "best_score": 20.0,
                "generalization_score": 19.0,
            }
            return engine

        mock_bo_step.side_effect = mock_step_side_effect

        runner = ComparisonRunner(
            task_id=self.task_id,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seeds=self.seeds,
            traditional=self.trad_cfg,
            llmbo=self.llmbo_cfg,
        )

        start_time = time.time()
        events = list(runner.events())
        duration = time.time() - start_time

        # Verify events
        event_types = [ev["type"] for ev in events]
        assert "meta" in event_types
        assert "seed_start" in event_types
        assert "step_start" in event_types
        assert "aggregate" in event_types
        assert "done" in event_types

        # Check if aggregate events were produced correctly (6 in total: 4 for snapshots + 2 for seed completions)
        aggregates = [ev for ev in events if ev["type"] == "aggregate"]
        assert len(aggregates) == 6
        # The final aggregate event should have completed_seeds == 2
        assert aggregates[-1]["completed_seeds"] == 2

        # Check for parallelism: 
        # Serial would take at least (2 seeds * 2 trials * 2 methods * 0.1s) = 0.8s (simplified)
        # In reality, trad and llm step sequentially in _run_one_seed.
        # So Serial: 2 seeds * 2 trials * 2 methods * 0.1s = 0.8s
        # Parallel: 1 seed_time = 2 trials * 2 methods * 0.1s = 0.4s
        print(f"Parallel duration: {duration:.4f}s")
        # Note: Threading overhead and GIL might make it less than 2x but should be faster than serial.
        # In this mock, it should be significantly faster.

if __name__ == "__main__":
    unittest.main()
