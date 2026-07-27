#!/usr/bin/env python
"""Main Submission Entry Script for Competition Reproducibility.

Executes LGBO algorithm across 20 seeds (seeds 100, 200, ..., 2000) for Buchwald-sub4
and Suzuki-Miyaura reaction optimization datasets.
Saves standard .pt trajectories and outputs metric summaries.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add code root to Python path
code_root = Path(__file__).resolve().parents[1]
if str(code_root) not in sys.path:
    sys.path.insert(0, str(code_root))

from bo_core.benchmark.runner import BenchmarkRunner

SEEDS = [i * 100 for i in range(1, 21)]
DATASETS = ["buchwald_sub4", "suzuki"]
OUTPUT_DIR = code_root.parent / "results" / "optimization_trajectories"


def main():
    print("=" * 70)
    print("  RUNNING COMPETITION SUBMISSION BENCHMARK SWEEP (20 SEEDS)")
    print("=" * 70)

    for task_id in DATASETS:
        ds_name = "buchwald" if "buchwald" in task_id else "suzuki"
        ds_out = OUTPUT_DIR / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Running Dataset: {task_id} (Target dir: {ds_out})")

        for seed in SEEDS:
            print(f"  [Task={task_id}] Seed={seed}...")
            runner = BenchmarkRunner(
                task_id=task_id,
                n_initial=5,
                n_trials=40,
                seed=seed,
                backend="botorch",
                output_dir=ds_out,
            )
            try:
                res = runner.run()
                runner.save_results(res)
                print(f"    -> Done! Best Score: {res['best_score']:.4f}")
            except Exception as e:
                print(f"    -> Error in seed {seed}: {e}")

    print("\n" + "=" * 70)
    print("  SUBMISSION RUN COMPLETE. Results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
