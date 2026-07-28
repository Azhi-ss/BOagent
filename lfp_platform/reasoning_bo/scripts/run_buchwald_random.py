#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Random-search baseline on Buchwald_sub4 (same protocol as run_buchwald_bo.py).

Mirrors the BO runner's protocol exactly so the comparison is fair:
  - same 20 competition seeds,
  - same 7-row seed trial (this product's labeled train rows),
  - same 40 iterations, 1 query per round,
  - same offline oracle (BuchwaldMetric lookup),
  - same .pt trajectory format.

The ONLY difference: each round picks a uniformly random unqueried pool row
(no GP, no acquisition function). This gives the "no-model" lower bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.buchwald import BUCHWALD_PARAM_NAMES as PARAM_NAMES, BuchwaldMetric  # noqa: E402

SUB4_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"
DEFAULT_SEEDS = [
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
]


def resolve_dataset(dir_name: str, prefix: str) -> dict:
    """Locate the benchmark dataset and return its file paths.

    Resolution order:
      1. Repo-shared layout: ``<repo>/datasets/chemical_reactions/<dir_name>``
         with unprefixed CSVs (``options.json``, ``test.csv``, ``train.csv``,
         ``searchspace.csv``).
      2. Legacy local layout: ``data/<Dir_Name>`` with prefixed CSVs
         (``<prefix>_test.csv``, ...), auto-extracting ``<Dir_Name>.zip``.
    """
    for candidate in (dir_name.lower(), dir_name):
        repo_dir = REPO_ROOT / "datasets" / "chemical_reactions" / candidate
        if (repo_dir / "test.csv").exists():
            return {
                "dir": repo_dir,
                "options": repo_dir / "options.json",
                "test": repo_dir / "test.csv",
                "train": repo_dir / "train.csv",
                "searchspace": repo_dir / "searchspace.csv",
            }
    legacy_dir = ROOT / "data" / dir_name
    zip_path = ROOT / "data" / f"{dir_name}.zip"
    if not legacy_dir.exists() and zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ROOT / "data")
    if not (legacy_dir / f"{prefix}_test.csv").exists():
        raise FileNotFoundError(
            f"Dataset {dir_name!r} not found under "
            f"{REPO_ROOT}/datasets/chemical_reactions or {ROOT}/data"
        )
    return {
        "dir": legacy_dir,
        "options": legacy_dir / "options.json",
        "test": legacy_dir / f"{prefix}_test.csv",
        "train": legacy_dir / f"{prefix}_train.csv",
        "searchspace": legacy_dir / f"{prefix}_searchspace.csv",
    }


def run_one_seed(
    seed: int,
    paths: dict,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    num_iterations: int,
    out_dir: Path,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    metric = BuchwaldMetric(
        name="buchwald",
        test_csv=paths["test"],
        train_csv=paths["train"],
    )

    pool_keys = [
        tuple(str(row[c]) for c in PARAM_NAMES) for _, row in test_df.iterrows()
    ]
    pool_index_map = {k: i for i, k in enumerate(pool_keys)}

    # Seed trial: this product's 7 labeled train rows (same as BO runner).
    init_rows = train_df[train_df["Product"] == SUB4_PRODUCT]
    queried: set[tuple[str, ...]] = set()
    for _, row in init_rows.iterrows():
        k = tuple(str(row[c]) for c in PARAM_NAMES)
        if k in pool_index_map:
            queried.add(k)

    trajectory: list[dict] = []
    best_so_far = -float("inf")

    # Best from seed trial (for fair best-so-far starting point).
    for k in queried:
        if k in metric.yield_map:
            best_so_far = max(best_so_far, float(metric.yield_map[k]))

    unqueried = [k for k in pool_keys if k not in queried]

    for step in range(1, num_iterations + 1):
        if not unqueried:
            print(f"[seed {seed}] pool exhausted at step {step}")
            break
        pick = rng.choice(unqueried)
        unqueried.remove(pick)
        queried.add(pick)
        query_index = pool_index_map[pick]
        observed_yield = float(metric.yield_map[pick])
        best_so_far = max(best_so_far, observed_yield)
        trajectory.append(
            {
                "step": step,
                "query_index": int(query_index),
                "condition": {n: pick[i] for i, n in enumerate(PARAM_NAMES)},
                "observed_yield": observed_yield,
            }
        )
        print(
            f"[seed {seed}] step {step:2d} | idx={query_index:3d} | "
            f"yield={observed_yield:6.2f} | best={best_so_far:6.2f}"
        )

    payload = {
        "seed": seed,
        "dataset": "Buchwald_sub4",
        "num_iterations": num_iterations,
        "method": "random",
        "trajectory": trajectory,
        "best_found": float(best_so_far),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / f"seed_{seed}.pt")

    with open(out_dir / f"seed_{seed}_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "query_index", "observed_yield", "best_so_far"])
        bsf = -float("inf")
        for rec in trajectory:
            bsf = max(bsf, rec["observed_yield"])
            writer.writerow([rec["step"], rec["query_index"], rec["observed_yield"], bsf])

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Buchwald_sub4 random-search baseline")
    parser.add_argument("--num_iterations", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Default: data/results/buchwald_sub4_random",
    )
    args = parser.parse_args()

    paths = resolve_dataset("Buchwald_sub4", prefix="buchwald_sub4")
    test_df = pd.read_csv(paths["test"])
    train_df = pd.read_csv(paths["train"])

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    out_dir = Path(
        args.result_dir
        if args.result_dir
        else str(ROOT / "data" / "results" / "buchwald_sub4_random")
    )

    searchspace_csv = paths["searchspace"]
    global_best = t95_threshold = None
    if searchspace_csv.exists():
        global_best = float(pd.read_csv(searchspace_csv)["Yield"].max())
        t95_threshold = 0.95 * global_best
        print(f"Global best = {global_best:.3f} | 95% threshold = {t95_threshold:.3f}")

    print("=" * 70)
    print(f"Buchwald_sub4 RANDOM baseline | iters={args.num_iterations} | seeds={seeds}")
    print(f"result_dir={out_dir}")
    print("=" * 70)

    summary = []
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        payload = run_one_seed(
            seed=seed,
            paths=paths,
            test_df=test_df,
            train_df=train_df,
            num_iterations=args.num_iterations,
            out_dir=out_dir,
        )
        t95 = None
        if t95_threshold is not None:
            bsf = -float("inf")
            for rec in payload["trajectory"]:
                bsf = max(bsf, rec["observed_yield"])
                if bsf >= t95_threshold:
                    t95 = rec["step"]
                    break
        summary.append(
            {"seed": seed, "best_found": payload["best_found"], "t95": t95, "reached_95": t95 is not None}
        )

    print("\n" + "=" * 70)
    if global_best is not None:
        print(f"Global best = {global_best:.3f} | 95% threshold = {t95_threshold:.3f}")
    print("RANDOM | Summary (best_found / t95 per seed):")
    for row in summary:
        t95_str = f"t95={row['t95']}" if row["reached_95"] else "t95=not-reached"
        print(f"  seed {row['seed']:5d}: best={row['best_found']:.3f} | {t95_str}")

    best_vals = [r["best_found"] for r in summary]
    t95_vals = [r["t95"] for r in summary if r["reached_95"]]
    if best_vals:
        print(
            f"best_found: mean={np.mean(best_vals):.3f} std={np.std(best_vals):.3f} "
            f"min={np.min(best_vals):.3f} max={np.max(best_vals):.3f}"
        )
    if t95_vals:
        print(
            f"t95: mean={np.mean(t95_vals):.1f} std={np.std(t95_vals):.1f} "
            f"min={np.min(t95_vals)} max={np.max(t95_vals)} "
            f"(reached {len(t95_vals)}/{len(summary)} seeds)"
        )
    else:
        print("t95: no seed reached 95% of global best within the budget")
    print(f"Saved {len(seeds)} .pt files -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
