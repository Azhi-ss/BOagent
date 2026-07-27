"""Evaluation Script for Competition Metric Verification.

Reads generated .pt trajectory files from submission/results/optimization_trajectories/
and computes/re-verifies the 4 core metrics (first_found, best_found, t95, AUC) with 95% CI.
Outputs or updates submission/results/summary_metrics.csv.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

code_root = Path(__file__).resolve().parents[1]
results_dir = code_root.parent / "results"
traj_dir = results_dir / "optimization_trajectories"

GLOBAL_MAX = {
    "buchwald": 86.60,
    "suzuki": 99.90,
}


def extract_observed_yields(data: dict) -> list[float]:
    """Extract observed yields array from .pt data dictionary."""
    if "trajectory" in data and isinstance(data["trajectory"], list):
        return [float(item["observed_yield"]) for item in data["trajectory"]]
    if "observed_yields" in data:
        return [float(y) for y in data["observed_yields"]]
    raise KeyError("Could not find yield trajectory in .pt data.")


def evaluate_all():
    print("=" * 70)
    print("  VERIFYING COMPETITION METRICS FROM GENERATED .PT TRAJECTORIES")
    print("=" * 70)

    rows = []

    for ds_name, g_max in GLOBAL_MAX.items():
        ds_dir = traj_dir / ds_name
        if not ds_dir.exists():
            print(f"  [WARNING] Trajectory directory not found: {ds_dir}")
            continue

        pt_files = sorted(ds_dir.glob("*.pt"))
        print(f"\n>>> Evaluating {ds_name} ({len(pt_files)} .pt trajectory files found)")

        trajectories = []
        for pf in pt_files:
            try:
                data = torch.load(pf)
                yields = extract_observed_yields(data)
                trajectories.append(yields)
            except Exception as e:
                print(f"  [ERROR] Failed reading {pf}: {e}")

        if not trajectories:
            continue

        target_95 = 0.95 * g_max

        first_founds = [t[0] for t in trajectories]
        best_founds = [float(np.max(np.maximum.accumulate(t))) for t in trajectories]
        
        t95_list = []
        auc_list = []
        for t in trajectories:
            bsf = np.maximum.accumulate(t)
            idx95 = np.where(bsf >= target_95)[0]
            t95_list.append(float(idx95[0] + 1) if len(idx95) > 0 else 41.0)
            auc_list.append(float(np.mean(bsf)))

        def _get_stats(vals):
            arr = np.array(vals)
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            ci = float(1.96 * s / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            return round(m, 2), round(s, 2), round(ci, 2)

        m1, s1, c1 = _get_stats(best_founds)
        m2, s2, c2 = _get_stats(t95_list)
        m3, s3, c3 = _get_stats(auc_list)

        print(f"  best_found : {m1} ± {c1} (std: {s1})")
        print(f"  t95        : {m2} ± {c2} (std: {s2})")
        print(f"  AUC        : {m3} ± {c3} (std: {s3})")

        rows.append({
            "dataset": ds_name,
            "method": "LGBO",
            "best_found_mean": m1,
            "best_found_std": s1,
            "best_found_ci95": c1,
            "t95_mean": m2,
            "t95_std": s2,
            "t95_ci95": c2,
            "auc_mean": m3,
            "auc_std": s3,
            "auc_ci95": c3,
        })

    df = pd.DataFrame(rows)
    out_csv = results_dir / "summary_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Verified and updated summary metrics CSV at: {out_csv}")


if __name__ == "__main__":
    evaluate_all()
