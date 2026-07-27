#!/usr/bin/env python
"""Auto-research loop: component-level iteration for competition BO.

Usage:
  python loop.py --dry-run                    # list components & compositions
  python loop.py --smoke --max-comps 3        # 3-seed on top-3 compositions
  python loop.py --tier confirm --top 2       # 5-seed on top-2 from smoke
  python loop.py --tier full --top 1          # 20-seed on champion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

AUTO_ROOT = Path(__file__).resolve().parent
if str(AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_ROOT))

from analyze import GLOBAL_BEST, aggregate_results, assert_seed_completeness, composite_score, trajectory_analysis, write_report  # noqa: E402
from compositions.base import get_base_compositions  # noqa: E402
import components.library  # noqa: E402,F401  # force component registration
from components.protocol import Composition, list_components  # noqa: E402
from engine import DEFAULT_N_INITIAL, HybridEngine, compute_metrics  # noqa: E402
from mutate import generate_neighbors  # noqa: E402


SEED_TIERS = {
    "smoke": [100, 200, 300],
    "confirm": [100, 200, 300, 400, 500],
    "full": [i * 100 for i in range(1, 21)],
}
DATASETS = ["buchwald_sub4", "suzuki"]


def run_one(
    comp: Composition,
    dataset: str,
    seed: int,
    n_iters: int = 40,
) -> dict[str, Any]:
    """Run one (composition, dataset, seed) and return metrics."""
    engine = HybridEngine(
        comp,
        dataset,
        seed=seed,
        n_iters=n_iters,
        n_initial=DEFAULT_N_INITIAL,
    )
    t0 = time.time()
    trajectory = engine.run()
    elapsed = time.time() - t0
    metrics = compute_metrics(trajectory, GLOBAL_BEST[dataset])
    analysis = trajectory_analysis(trajectory)
    return {
        "composition": comp.name,
        "dataset": dataset,
        "seed": seed,
        "prior_protocol": "seeded_subsample",
        "n_initial": DEFAULT_N_INITIAL,
        "initial_indices": list(engine.initial_indices),
        "elapsed_s": elapsed,
        "metrics": metrics,
        "analysis": analysis,
        "trajectory": trajectory,
    }


def run_composition(
    comp: Composition,
    datasets: list[str],
    seeds: list[int],
    n_iters: int = 40,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Run one composition across datasets × seeds."""
    configs = [(comp, ds, s) for ds in datasets for s in seeds]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, c, d, s, n_iters): (c.name, d, s)
            for c, d, s in configs
        }
        for fut in as_completed(futures):
            name, ds, seed = futures[fut]
            try:
                results.append(fut.result())
                print(f"  [done] {name}/{ds}/seed_{seed}")
            except Exception as exc:
                print(f"  [FAIL] {name}/{ds}/seed_{seed}: {exc}")
    return results


def save_results(results: list[dict[str, Any]], tag: str) -> Path:
    out = AUTO_ROOT / "history" / f"{tag}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip trajectories for compact summary; keep in separate file
    slim = [{k: v for k, v in r.items() if k != "trajectory"} for r in results]
    out.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    return out


def print_leaderboard(scores: dict[str, float], title: str = "Leaderboard") -> None:
    print(f"\n=== {title} ===")
    for i, (comp, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1]), 1):
        marker = "*" if i == 1 else " "
        print(f"{marker} {i:2d}. {comp:30s} {score:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Component-level auto research")
    parser.add_argument("--tier", choices=["smoke", "confirm", "full"], default="smoke")
    parser.add_argument("--max-comps", type=int, default=None)
    parser.add_argument("--top", type=int, default=3, help="Top-N to advance from previous tier")
    parser.add_argument("--n-iters", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prev-results", type=str, default=None, help="JSON file from previous tier")
    args = parser.parse_args()

    comps = get_base_compositions()
    if args.max_comps:
        comps = comps[: args.max_comps]

    print("=" * 70)
    print("  COMPITETION AUTO-RESEARCH: Component-Level Iteration")
    print("=" * 70)
    print("Components available:", list_components())
    print(f"Base compositions: {[c.name for c in comps]}")
    print(f"Tier: {args.tier} -> seeds {SEED_TIERS[args.tier]}")

    if args.dry_run:
        print("\n[dry-run] would run:")
        for comp in comps:
            print(f"  - {comp.describe()}")
        return 0

    # Load previous tier results to select top-N
    if args.prev_results:
        prev = json.loads(Path(args.prev_results).read_text())
        prev_summary = aggregate_results(prev)
        prev_scores = composite_score(prev_summary)
        top_comps = sorted(prev_scores, key=prev_scores.get, reverse=True)[: args.top]
        comps = [c for c in comps if c.name in top_comps]
        print(f"Advanced from previous tier: {top_comps}")

    all_results: list[dict[str, Any]] = []
    for comp in comps:
        print(f"\n>>> Running {comp.name} ({comp.describe()})")
        results = run_composition(
            comp,
            DATASETS,
            SEED_TIERS[args.tier],
            n_iters=args.n_iters,
            workers=args.workers,
        )
        all_results.extend(results)
        path = save_results(results, f"{args.tier}_{comp.name}")
        print(f"  saved: {path}")

    if not all_results:
        print("[BLOCKED] No results completed.")
        return 2

    try:
        assert_seed_completeness(all_results, SEED_TIERS[args.tier], DATASETS)
    except ValueError as exc:
        print(f"[BLOCKED] Seed completeness failure:\n{exc}")
        return 2

    # Aggregate + score
    summary = aggregate_results(all_results)
    scores = composite_score(summary)
    print_leaderboard(scores, f"{args.tier.upper()} Tier Results")

    # Per-composition analysis
    analyses: dict[str, Any] = {}
    for r in all_results:
        analyses.setdefault(r["composition"], []).append(r["analysis"])
    # Average analysis
    avg_analyses = {}
    for comp, a_list in analyses.items():
        avg_analyses[comp] = {
            "n_improvements": float(np.mean([a["n_improvements"] for a in a_list])),
            "llm_action_count": float(np.mean([a["llm_action_count"] for a in a_list])),
            "acq_switches": float(np.mean([a["acq_switches"] for a in a_list])),
            "final_best": float(np.mean([a["final_best"] for a in a_list if a["final_best"] is not None])),
        }

    report_path = write_report(
        AUTO_ROOT / "reports",
        summary,
        scores,
        avg_analyses,
    )
    print(f"\nReport: {report_path}")

    # Save tier results for next tier
    tier_out = save_results(all_results, f"{args.tier}_all")
    print(f"Tier results: {tier_out}")

    # Suggest next mutations
    print("\n--- Suggested next mutations (neighbors of top-2) ---")
    top2 = sorted(scores, key=scores.get, reverse=True)[:2]
    for comp_name in top2:
        comp = next(c for c in comps if c.name == comp_name)
        neighbors = generate_neighbors(comp, max_neighbors=3)
        for n in neighbors:
            print(f"  {n.describe()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
