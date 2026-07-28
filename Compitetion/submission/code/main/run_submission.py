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

from bo_core.benchmark.lgbo_runner import run_one

SEEDS = [i * 100 for i in range(1, 21)]
DATASETS = ["buchwald_sub4", "suzuki"]
OUTPUT_DIR = code_root.parent / "results" / "optimization_trajectories"


def main():
    print("=" * 70)
    print("  RUNNING COMPETITION SUBMISSION BENCHMARK SWEEP (20 SEEDS)")
    print("=" * 70)

    for task_id in DATASETS:
        print(f"\n>>> Running Dataset: {task_id} (Target dir: {OUTPUT_DIR})")

        for seed in SEEDS:
            print(f"  [Task={task_id}] Seed={seed}...")
            try:
                result = run_one(
                    dataset=task_id,
                    method="lgbo",
                    seed=seed,
                    n_iters=40,
                    output_dir=OUTPUT_DIR,
                    backend="botorch",
                )
                print(f"    -> Done! Best Score: {result['best_found']:.4f}")
            except Exception as exc:  # noqa: BLE001 - continue remaining seeds
                print(f"    -> Error in seed {seed}: {exc}")

    print("\n" + "=" * 70)
    print("  SUBMISSION RUN COMPLETE. Results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
