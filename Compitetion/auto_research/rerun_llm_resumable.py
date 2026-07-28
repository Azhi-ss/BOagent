"""Resumable rerun of LLM methods under the fixed train-prior protocol."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dministrator/project/BOagent/Compitetion/auto_research")

import components.library  # noqa: F401
from analyze import GLOBAL_BEST
from compositions.base import get_base_compositions
from engine import HybridEngine, compute_metrics

DATASETS = ["buchwald_sub4", "suzuki"]
SEEDS = [i * 100 for i in range(1, 21)]

BASE = {c.name: c for c in get_base_compositions()}
TO_RUN = [
    ("gpbo_cake", BASE["gpbo_cake"]),
    ("lgbo_mean_shift", BASE["lgbo_mean_shift"]),
    ("lgbo_cake", BASE["lgbo_cake"]),
]

OUT_DIR = Path("/home/dministrator/project/BOagent/Compitetion/auto_research/history/experiments")

for comp_name, comp in TO_RUN:
    out_file = OUT_DIR / f"{comp_name}_fixed_prior.json"
    results = []
    if out_file.exists():
        loaded_results = json.loads(out_file.read_text())
        latest_by_key = {
            (result.get("dataset"), result.get("seed")): result
            for result in loaded_results
        }
        results = list(latest_by_key.values())
        done_keys = {(r["dataset"], r["seed"]) for r in results if r.get("status") == "ok"}
        print(f"[{comp_name}] Resuming: {len(done_keys)} already done", flush=True)
    else:
        done_keys = set()
        print(f"[{comp_name}] Starting fresh", flush=True)

    for ds in DATASETS:
        for seed in SEEDS:
            if (ds, seed) in done_keys:
                continue
            t0 = time.time()
            eng = None
            try:
                eng = HybridEngine(comp, ds, seed=seed, n_iters=40)
                traj = eng.run()
                m = compute_metrics(traj, GLOBAL_BEST[ds])
                r = {
                    "composition": comp_name,
                    "dataset": ds,
                    "seed": seed,
                    "prior_protocol": "fixed_train_prior",
                    "n_train_prior": len(eng.initial_indices),
                    "initial_indices": list(eng.initial_indices),
                    "elapsed_s": round(time.time() - t0, 1),
                    "metrics": m,
                    "diagnostics": eng.diagnostics,
                    "status": "ok",
                }
                print(
                    f"[OK] {comp_name}/{ds}/seed{seed}: best={m['best_found']:.2f} t95={m['t95']} ({r['elapsed_s']}s)",
                    flush=True,
                )
            except Exception as exc:
                r = {
                    "composition": comp_name,
                    "dataset": ds,
                    "seed": seed,
                    "prior_protocol": "fixed_train_prior",
                    "n_train_prior": (
                        len(eng.initial_indices) if eng is not None else 0
                    ),
                    "initial_indices": (
                        list(eng.initial_indices) if eng is not None else []
                    ),
                    "elapsed_s": round(time.time() - t0, 1),
                    "diagnostics": eng.diagnostics if eng is not None else None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[FAIL] {comp_name}/{ds}/seed{seed}: {type(exc).__name__}: {exc}", flush=True)

            results = [
                previous
                for previous in results
                if (previous.get("dataset"), previous.get("seed")) != (ds, seed)
            ]
            results.append(r)
            out_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[{comp_name}] COMPLETE: {ok}/{len(results)} ok -> {out_file.name}", flush=True)

print("\n=== ALL DONE ===", flush=True)
