"""Resumable rerun of LLM methods under the fixed train-prior protocol."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/home/dministrator/project/BOagent/Compitetion/auto_research")

import components.library  # noqa: F401
from analyze import (
    GLOBAL_BEST,
    aggregate_results,
    assert_seed_completeness,
    composite_score,
)
from compositions.base import get_base_compositions
from engine import HybridEngine, compute_metrics

DATASETS = ["buchwald_sub4", "suzuki"]
SEEDS = [i * 100 for i in range(1, 21)]

BASE = {c.name: c for c in get_base_compositions()}
METRICS = ["BIC", "AIC", "MARGINAL_LIKELIHOOD"]
TO_RUN = [(metric, BASE["gpbo_cake"]) for metric in METRICS]

OUT_DIR = Path("/home/dministrator/project/BOagent/Compitetion/auto_research/history/experiments")

for metric, comp in TO_RUN:
    os.environ["CAKE_FITNESS_METRIC"] = metric
    comp_name = f"gpbo_cake_{metric.lower()}"
    out_file = OUT_DIR / "cake_fitness_metrics.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else []
    latest_by_key = {
        (result.get("fitness_metric"), result.get("dataset"), result.get("seed")): result
        for result in results
    }
    results = list(latest_by_key.values())
    done_keys = {
        (r["dataset"], r["seed"])
        for r in results
        if r.get("fitness_metric") == metric and r.get("status") == "ok"
    }
    print(f"[{comp_name}] Resuming: {len(done_keys)} already done", flush=True)

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
                populations = [
                    fit.get("active_kernels", [])
                    for fit in eng.diagnostics.get("surrogate", {}).get("fits", [])
                ]
                composite_survival = (
                    sum(any("+" in kernel or "*" in kernel for kernel in population) for population in populations)
                    / len(populations)
                    if populations else 0.0
                )
                r = {
                    "composition": comp_name,
                    "fitness_metric": metric,
                    "dataset": ds,
                    "seed": seed,
                    "prior_protocol": "fixed_train_prior",
                    "n_train_prior": len(eng.initial_indices),
                    "initial_indices": list(eng.initial_indices),
                    "elapsed_s": round(time.time() - t0, 1),
                    "metrics": m,
                    "composite_kernel_survival": composite_survival,
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
                    "fitness_metric": metric,
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
                if (
                    previous.get("fitness_metric"),
                    previous.get("dataset"),
                    previous.get("seed"),
                ) != (metric, ds, seed)
            ]
            results.append(r)
            out_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[{comp_name}] COMPLETE: {ok}/{len(results)} ok -> {out_file.name}", flush=True)

complete = [result for result in results if result.get("status") == "ok"]
if len(complete) == len(METRICS) * len(DATASETS) * len(SEEDS):
    assert_seed_completeness(complete, SEEDS, DATASETS)
    summary = aggregate_results(complete)
    scores = composite_score(summary)
    survival = {
        metric: sum(
            result["composite_kernel_survival"]
            for result in complete
            if result["fitness_metric"] == metric
        ) / (len(DATASETS) * len(SEEDS))
        for metric in METRICS
    }
    analysis_file = OUT_DIR / "cake_fitness_metrics_analysis.json"
    analysis_file.write_text(
        json.dumps({"summary": summary, "composite_scores": scores, "composite_kernel_survival": survival}, indent=2),
        encoding="utf-8",
    )
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "event": "cake_fitness_metrics_complete",
        "n_seeds": len(SEEDS),
        "datasets": DATASETS,
        "metrics": METRICS,
        "composite_scores": scores,
        "composite_kernel_survival": survival,
        "analysis": str(analysis_file),
    }
    with (OUT_DIR.parent / "ledger.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

print("\n=== ALL DONE ===", flush=True)
