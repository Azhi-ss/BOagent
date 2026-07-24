"""Benchmark runner for LGBO vs GPBO on the chemical-reaction datasets.

Drives N seeds x {buchwald_sub4, suzuki} x {lgbo, gpbo} using the LGBOEngine,
computes the competition metrics (best_found, initial_round_found_best, t95,
AUC_best_so_far), and writes per-run CSV + .pt trajectory files plus an
aggregate summary.json. .pt output uses torch if available, else falls back to
pickle (.pkl) with a warning.

Usage:
    cd packages/bo-core
    python -m bo_core.benchmark.lgbo_runner \\
        --datasets buchwald_sub4,suzuki --methods lgbo,gpbo \\
        --seeds 100,200,300 --n_iters 40 --workers 4
"""

from __future__ import annotations

import argparse
import os
# Restrict BLAS to 1 thread before importing numpy/sklearn to avoid
# CPU oversubscription when the ThreadPoolExecutor runs multiple GP fits in
# parallel (otherwise 4 workers × 16 BLAS threads = thread thrashing).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import json
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bo_core.benchmark.data_loader import DATA_LOADERS
from bo_core.optimization.lgbo import LGBOEngine

# Per-dataset global best (max Yield in test.csv), from the dataset READMEs.
GLOBAL_BEST = {"buchwald_sub4": 86.60, "suzuki": 99.90}


def compute_metrics(engine: LGBOEngine, global_best: float) -> dict[str, float | int]:
    """Competition metrics over the optimization-budget queries (the 40 steps).

    Per the README, ``best_found`` is the highest target value *discovered within
    the optimization budget* - i.e. the 40 queried test points, NOT the given
    train prior. The prior (incl. cross-product rows) is context, not discovery;
    excluding it avoids the prior max masking the optimization's contribution.
    LGBO and GPBO share the same prior, so the comparison stays fair.
    """
    yields = [t["observed_yield"] for t in engine.trajectory]
    if not yields:
        return {"best_found": float("-inf"), "initial_round_found_best": float("-inf"),
                "t95": math.inf, "AUC_best_so_far": 0.0}
    best_so_far = []
    cur = float("-inf")
    for y in yields:
        cur = max(cur, y)
        best_so_far.append(cur)
    target = 0.95 * global_best
    t95 = next((i for i, b in enumerate(best_so_far, 1) if b >= target), len(yields) + 1)
    return {
        "best_found": float(best_so_far[-1]),
        "initial_round_found_best": float(best_so_far[0]),
        "t95": int(t95),
        "AUC_best_so_far": float(np.mean(best_so_far)),
    }


def _save_pt(path: Path, obj: dict[str, Any]) -> None:
    """Save trajectory in competition .pt format (torch if available, else pickle)."""
    try:
        import torch  # type: ignore
        torch.save(obj, path)
    except ImportError:
        pkl = path.with_suffix(".pkl")
        with open(pkl, "wb") as f:
            pickle.dump(obj, f)
        print(f"[runner] torch not installed; saved {pkl.name} (pickle) instead of .pt")


def run_one(
    dataset: str,
    method: str,
    seed: int,
    n_iters: int,
    output_dir: Path,
    n_restarts: int = 10,
) -> dict[str, Any]:
    """Run one (dataset, method, seed) configuration; save CSV+.pt; return metrics."""
    use_llm = method == "lgbo"
    engine = LGBOEngine(
        dataset=dataset,
        seed=seed,
        use_llm=use_llm,
        n_iters=n_iters,
        n_restarts=n_restarts,
    )
    t0 = time.time()
    engine.run()
    elapsed = time.time() - t0
    metrics = compute_metrics(engine, GLOBAL_BEST[dataset])

    save_dir = output_dir / dataset / method
    save_dir.mkdir(parents=True, exist_ok=True)
    # CSV: human-readable trajectory.
    pd.DataFrame(engine.trajectory).to_csv(save_dir / f"seed_{seed}.csv", index=False)
    # .pt: competition submission format.
    pt_obj = {
        "seed": seed,
        "dataset": dataset,
        "method": method,
        "trajectory": engine.trajectory,
    }
    _save_pt(save_dir / f"seed_{seed}.pt", pt_obj)

    result = {"dataset": dataset, "method": method, "seed": seed, "elapsed_s": elapsed, **metrics}
    print(f"[runner] {dataset}/{method}/seed_{seed}: best={metrics['best_found']:.2f} "
          f"t95={metrics['t95']} AUC={metrics['AUC_best_so_far']:.2f} ({elapsed:.0f}s)")
    return result


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± std + 95% CI per (dataset, method) across seeds."""
    summary: dict[str, Any] = {}
    for dataset in sorted({r["dataset"] for r in results}):
        summary[dataset] = {}
        for method in sorted({r["method"] for r in results if r["dataset"] == dataset}):
            group = [r for r in results if r["dataset"] == dataset and r["method"] == method]
            seeds = sorted({r["seed"] for r in group})
            entry: dict[str, Any] = {"seeds": seeds, "n": len(group)}
            for metric in ("best_found", "initial_round_found_best", "t95", "AUC_best_so_far"):
                vals = np.array([r[metric] for r in group], dtype=float)
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                entry[metric] = {"mean": mean, "std": std, "ci95": ci}
            summary[dataset][method] = entry
    return summary


def _print_comparison(summary: dict[str, Any]) -> None:
    for dataset, methods in summary.items():
        print(f"\n=== {dataset} (global best {GLOBAL_BEST[dataset]}) ===")
        print(f"{'method':<8} {'best_mean':>10} {'best_std':>9} {'t95_mean':>9} {'AUC_mean':>9}")
        for method, e in methods.items():
            print(f"{method:<8} {e['best_found']['mean']:>10.2f} "
                  f"{e['best_found']['std']:>9.2f} {e['t95']['mean']:>9.1f} "
                  f"{e['AUC_best_so_far']['mean']:>9.2f}")
        if "lgbo" in methods and "gpbo" in methods:
            diff = methods["lgbo"]["best_found"]["mean"] - methods["gpbo"]["best_found"]["mean"]
            tag = "LGBO better" if diff > 0 else "GPBO better"
            print(f"  -> best_found LGBO - GPBO = {diff:+.2f} ({tag})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LGBO vs GPBO benchmark on chemical datasets")
    parser.add_argument("--datasets", default="buchwald_sub4,suzuki",
                        help="comma-separated dataset names (default: both)")
    parser.add_argument("--methods", default="lgbo,gpbo",
                        help="comma-separated: lgbo,gpbo (default: both)")
    parser.add_argument("--seeds", default="100,200,300",
                        help="comma-separated seeds (default: 100,200,300)")
    parser.add_argument("--n_iters", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--output_dir", default="results/lgbo")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)

    configs = [(d, m, s) for d in datasets for m in methods for s in seeds]
    print(f"Running {len(configs)} configs: {len(datasets)} datasets x "
          f"{len(methods)} methods x {len(seeds)} seeds, workers={args.workers}")

    results: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, d, m, s, args.n_iters, output_dir, args.n_restarts): (d, m, s)
            for (d, m, s) in configs
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                d, m, s = futures[fut]
                print(f"[runner] FAILED {d}/{m}/seed_{s}: {exc}")
    print(f"\nAll configs done in {time.time() - t0:.0f}s")

    summary = _aggregate(results)
    _print_comparison(summary)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
