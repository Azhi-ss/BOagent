"""CLI entry point for PVKBO benchmark.

Usage:
    cd packages/bo-core
    python -m bo_core.benchmark.cli \
        --task band_alignment \
        --engine deepseek-v4-flash \
        --sm_mode discriminative \
        --n_trials 20 \
        --n_initial 5 \
        --seed 42 \
        --output_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package root is on sys.path when invoked as python -m bo_core.benchmark.cli
_backend_dir = Path(__file__).resolve().parent.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PVKBO Benchmark — GP+LLM acquisition function evaluation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="band_alignment",
        choices=["band_alignment", "defects_doping"],
        help="Task to benchmark (default: band_alignment)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="deepseek-v4-flash",
        help="LLM chat engine model name (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--sm_mode",
        type=str,
        default="discriminative",
        choices=["discriminative", "generative"],
        help="Surrogate model mode (default: discriminative)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of BO trials (default: 20)",
    )
    parser.add_argument(
        "--n_initial",
        type=int,
        default=5,
        help="Number of initial samples (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed run (e.g. '42,123,456')",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=10,
        help="Number of candidate points per trial (default: 10)",
    )
    parser.add_argument(
        "--n_templates",
        type=int,
        default=2,
        help="Number of LLM prompt templates (default: 2)",
    )
    parser.add_argument(
        "--n_gens",
        type=int,
        default=5,
        help="Number of LLM generations (default: 5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="UCB exploration parameter (default: 0.1)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k GP candidates to send to LLM (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to Excel data file (default: PVK-LLM/custom_perovskite_dataset/)",
    )

    args = parser.parse_args()

    from bo_core.benchmark.runner import BenchmarkRunner, run_multi_seed

    common_kwargs = {
        "task_id": args.task,
        "n_initial": args.n_initial,
        "n_trials": args.n_trials,
        "sm_mode": args.sm_mode,
        "chat_engine": args.engine,
        "n_candidates": args.n_candidates,
        "n_templates": args.n_templates,
        "n_gens": args.n_gens,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "output_dir": args.output_dir,
        "data_path": args.data_path,
    }

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        results = run_multi_seed(seeds=seeds, **common_kwargs)
        print(f"\nMulti-seed complete. {len(results)} runs finished.")
        for r in results:
            print(
                f"  seed={r['seed']}: best={r['best_score']:.4f}, "
                f"gen={r['best_generalization_score']:.4f}"
            )
    else:
        runner = BenchmarkRunner(seed=args.seed, **common_kwargs)
        result = runner.run()
        runner.save_results(result)
        print(f"\nBenchmark complete.")
        print(f"  seed={result['seed']}")
        print(f"  best_score={result['best_score']:.4f}")
        print(f"  best_generalization_score={result['best_generalization_score']:.4f}")
        print(f"  results saved to: {runner.output_dir / f'results_{args.sm_mode}' / args.task}")


if __name__ == "__main__":
    main()
