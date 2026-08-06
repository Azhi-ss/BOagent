#!/usr/bin/env python
"""Main Submission Entry Script for Competition Reproducibility.

Executes LGBO algorithm across 20 seeds (seeds 100, 200, ..., 2000) for Buchwald-sub4
and Suzuki-Miyaura reaction optimization datasets.
Saves standard .pt trajectories and outputs metric summaries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.evaluate_metrics import evaluate_all

try:
    from bo_core.benchmark.lgbo_runner import run_one
except ImportError as exc:
    raise SystemExit(
        "bo-core is not installed; run from the repository root or an exported "
        "competition snapshot"
    ) from exc

SEEDS = [i * 100 for i in range(1, 21)]
DATASETS = ["buchwald_sub4", "suzuki"]
OUTPUT_DIR = CODE_ROOT.parent / "results" / "optimization_trajectories"


def _is_complete_matrix(datasets: list[str], seeds: list[int], n_iters: int) -> bool:
    return (
        n_iters == 40
        and len(datasets) == len(DATASETS)
        and set(datasets) == set(DATASETS)
        and len(seeds) == len(SEEDS)
        and set(seeds) == set(SEEDS)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the competition submission")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--n-iters", type=int, default=40)
    parser.add_argument("--backend", choices=("botorch", "sklearn"), default="botorch")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    datasets = [value for value in args.datasets.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]

    print("=" * 70)
    print(f"  RUNNING COMPETITION SUBMISSION ({len(seeds)} SEEDS)")
    print("=" * 70)

    for task_id in datasets:
        print(f"\n>>> Running Dataset: {task_id} (Target dir: {args.output_dir})")
        for seed in seeds:
            print(f"  [Task={task_id}] Seed={seed}...")
            result = run_one(
                dataset=task_id,
                method="lgbo",
                seed=seed,
                n_iters=args.n_iters,
                output_dir=args.output_dir,
                backend=args.backend,
            )
            print(f"    -> Done! Best Score: {result['best_found']:.4f}")

    print("\n" + "=" * 70)
    if _is_complete_matrix(datasets, seeds, args.n_iters):
        summary_path = evaluate_all(
            trajectories_path=args.output_dir,
            results_path=args.output_dir.parent,
            backend=args.backend,
        )
        print("  VERIFIED SUBMISSION RUN COMPLETE. Summary saved to:", summary_path)
    else:
        print("  SUBSET RUN COMPLETE. No summary_metrics.csv was generated.")
        print("  Trajectories saved to:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
