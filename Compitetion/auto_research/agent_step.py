#!/usr/bin/env python
"""One agent step of the GOAL loop: read status → run queue → judge → update.

Usage:
  python agent_step.py           # run next units from status.json
  python agent_step.py --status  # print status only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import components.library  # noqa: F401
from analyze import (
    GLOBAL_BEST,
    aggregate_results,
    assert_seed_completeness,
    composite_score,
    write_report,
)
from components.protocol import Composition
from compositions.base import get_base_compositions
from engine import HybridEngine, compute_metrics
from mutate import generate_neighbors

SEED_TIERS = {
    "smoke": [100, 200, 300],
    "confirm": [100, 200, 300, 400, 500],
    "full": [i * 100 for i in range(1, 21)],
}
DATASETS = ["buchwald_sub4", "suzuki"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_status() -> dict[str, Any]:
    return json.loads((ROOT / "status.json").read_text(encoding="utf-8"))


def _save_status(status: dict[str, Any]) -> None:
    status["updated_at"] = _now()
    (ROOT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def _ledger(event: dict[str, Any]) -> None:
    event = {"ts": _now(), **event}
    with (ROOT / "history" / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False))


def _resolve_compositions(names: list[str]) -> list[Composition]:
    base = {c.name: c for c in get_base_compositions()}
    out: list[Composition] = []
    for name in names:
        if name in base:
            out.append(base[name])
            continue
        # neighbor names like gpbo_ei_acquisition_pi
        # rebuild from mutate if possible
        found = None
        for b in base.values():
            for n in generate_neighbors(b, max_neighbors=20):
                if n.name == name:
                    found = n
                    break
            if found:
                break
        if found:
            out.append(found)
        else:
            print(f"[warn] unknown composition {name}, skip")
    return out


def run_one(comp_dict: dict[str, Any], dataset: str, seed: int, n_iters: int) -> dict[str, Any]:
    comp = Composition(**comp_dict)
    t0 = time.time()
    engine = None
    try:
        engine = HybridEngine(
            comp,
            dataset,
            seed=seed,
            n_iters=n_iters,
        )
        traj = engine.run()
    except Exception as exc:
        return {
            "composition": comp.name,
            "dataset": dataset,
            "seed": seed,
            "prior_protocol": "fixed_train_prior",
            "n_train_prior": len(engine.initial_indices) if engine is not None else 0,
            "initial_indices": list(engine.initial_indices) if engine is not None else [],
            "elapsed_s": time.time() - t0,
            "diagnostics": engine.diagnostics if engine is not None else None,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    metrics = compute_metrics(traj, GLOBAL_BEST[dataset])
    return {
        "composition": comp.name,
        "dataset": dataset,
        "seed": seed,
        "prior_protocol": "fixed_train_prior",
        "n_train_prior": len(engine.initial_indices),
        "initial_indices": list(engine.initial_indices),
        "elapsed_s": time.time() - t0,
        "metrics": metrics,
        "diagnostics": engine.diagnostics,
        "status": "ok",
    }


def _comp_to_dict(c: Composition) -> dict[str, Any]:
    return {
        "name": c.name,
        "surrogate": c.surrogate,
        "acquisition": c.acquisition,
        "selector": c.selector,
        "llm_strategy": c.llm_strategy,
        "params": c.params,
    }


def run_tier(
    comps: list[Composition],
    tier: str,
    n_iters: int = 40,
    workers: int = 8,
) -> list[dict[str, Any]]:
    seeds = SEED_TIERS[tier]
    configs = [(c, d, s) for c in comps for d in DATASETS for s in seeds]
    results: list[dict[str, Any]] = []
    failures_path = ROOT / "history" / "failures.jsonl"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(run_one, _comp_to_dict(c), d, s, n_iters): (c.name, d, s)
            for c, d, s in configs
        }
        for fut in as_completed(futs):
            name, d, s = futs[fut]
            try:
                r = fut.result()
                if r.get("status") != "ok":
                    row = {"ts": _now(), "tier": tier, **r}
                    with failures_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, default=str) + "\n")
                    _ledger({
                        "event": "error",
                        **{
                            k: row[k]
                            for k in ("composition", "dataset", "seed", "tier", "error")
                        },
                    })
                    continue
                results.append(r)
                m = r["metrics"]
                print(
                    f"[ok] {name}/{d}/seed_{s}: best={m['best_found']:.2f} "
                    f"t95={m['t95']} AUC={m['AUC_best_so_far']:.2f} ({r['elapsed_s']:.0f}s)"
                )
            except Exception as exc:
                row = {
                    "ts": _now(),
                    "composition": name,
                    "dataset": d,
                    "seed": s,
                    "tier": tier,
                    "error": f"{type(exc).__name__}: {exc}",
                    "tb": traceback.format_exc(),
                }
                with failures_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                _ledger({"event": "error", **{k: row[k] for k in ("composition", "dataset", "seed", "tier", "error")}})
    return results


def baseline_gpbo_score() -> float:
    baseline = json.loads((ROOT / "baselines" / "seed_baseline.json").read_text())
    baseline_as = {
        "baseline_gpbo": {ds: baseline["summary"][ds]["gpbo"] for ds in baseline["summary"]}
    }
    return float(composite_score(baseline_as)["baseline_gpbo"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-iters", type=int, default=40)
    args = parser.parse_args()

    status = _load_status()
    if args.status:
        print(json.dumps(status, indent=2))
        return 0

    if status["state"] in ("done", "blocked_env", "blocked_human_gate"):
        print(f"Loop not runnable: state={status['state']} stop={status.get('stop_reason')}")
        return 0

    tier = status.get("phase", "smoke")
    if tier not in SEED_TIERS:
        if tier == "report":
            status["state"] = "done"
            status["stop_reason"] = status.get("stop_reason") or "done_confirm_only"
            _save_status(status)
            return 0
        print(f"Unknown phase {tier}")
        return 1

    queue = list(status.get("queue") or [])
    if not queue:
        status["state"] = "done"
        status["stop_reason"] = "done_no_improvement"
        _save_status(status)
        _ledger({"event": "stop", "reason": "empty_queue"})
        return 0

    comps = _resolve_compositions(queue)
    if not comps:
        status["state"] = "blocked_env"
        status["stop_reason"] = "blocked_env"
        status["last_event"] = "queue names unresolved"
        _save_status(status)
        return 2

    status["state"] = "running"
    status["last_event"] = f"run {tier} on {[c.name for c in comps]}"
    _save_status(status)
    _ledger({"event": "plan", "tier": tier, "compositions": [c.name for c in comps], "seeds": SEED_TIERS[tier]})

    t0 = time.time()
    results = run_tier(comps, tier, n_iters=args.n_iters, workers=args.workers)
    wall = time.time() - t0
    status["budget"]["wall_hours_used"] = float(status["budget"].get("wall_hours_used", 0)) + wall / 3600

    tag = f"{tier}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out = ROOT / "history" / f"{tag}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _ledger({"event": f"{tier}_saved", "path": str(out), "n_ok": len(results), "wall_s": wall})

    if not results:
        status["state"] = "blocked_env"
        status["stop_reason"] = "blocked_env"
        status["last_event"] = f"{tier} all failed"
        _save_status(status)
        return 2

    try:
        assert_seed_completeness(results, SEED_TIERS[tier], DATASETS)
    except ValueError as exc:
        status["state"] = "blocked_env"
        status["stop_reason"] = "incomplete_seeds"
        status["last_event"] = str(exc).split("\n")[0]
        _save_status(status)
        _ledger({"event": "incomplete_seeds", "error": str(exc)})
        print(f"[BLOCKED] Seed completeness failure:\n{exc}")
        return 2

    summary = aggregate_results(results)
    scores = composite_score(summary)
    base_ref = baseline_gpbo_score()
    winner = max(scores, key=scores.get)
    delta = scores[winner] - base_ref
    write_report(ROOT / "reports", summary, scores, {c: {} for c in scores})

    evaluated = set(status.get("evaluated") or [])
    evaluated.update(scores.keys())
    status["evaluated"] = sorted(evaluated)
    status["budget"]["compositions_evaluated"] = len(status["evaluated"])
    status[f"{tier}_scores"] = scores

    print(f"\n=== {tier.upper()} SCORES ===")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:.6f}")
    print(f"baseline_gpbo={base_ref:.6f} winner={winner} delta={delta:+.6f}")

    # Promotion logic
    if tier == "smoke":
        if delta > 1e-6:
            status["champion"] = {
                "source": f"smoke:{winner}",
                "composition": winner,
                "score": scores[winner],
                "tier": "smoke",
                "delta_vs_baseline_gpbo": delta,
                "summary": summary[winner],
            }
            status["phase"] = "confirm"
            status["queue"] = [winner]
            status["state"] = "running"
            verdict = "promote_confirm"
        else:
            status["champion"] = {
                "source": "seed_baseline",
                "composition": "baseline_gpbo",
                "score": base_ref,
                "tier": "baseline",
            }
            # try neighbors
            base_map = {c.name: c for c in get_base_compositions()}
            neigh = []
            if winner in base_map:
                neigh = [n.name for n in generate_neighbors(base_map[winner], max_neighbors=4)
                         if n.name not in evaluated]
            if neigh:
                status["phase"] = "smoke"
                status["queue"] = neigh[:3]
                status["state"] = "running"
                verdict = "try_neighbors"
            else:
                status["phase"] = "report"
                status["queue"] = []
                status["state"] = "done"
                status["stop_reason"] = "done_no_improvement"
                verdict = "done_no_improvement"

    elif tier == "confirm":
        if delta > 1e-6:
            status["champion"] = {
                "source": f"confirm:{winner}",
                "composition": winner,
                "score": scores[winner],
                "tier": "confirm",
                "delta_vs_baseline_gpbo": delta,
                "summary": summary[winner],
            }
            if status["budget"]["full_runs"] < status["budget"]["max_full_runs"]:
                status["phase"] = "full"
                status["queue"] = [winner]
                status["state"] = "running"
                verdict = "promote_full"
            else:
                status["phase"] = "report"
                status["queue"] = []
                status["state"] = "done"
                status["stop_reason"] = "done_confirm_only"
                verdict = "done_confirm_only"
        else:
            status["champion"] = {
                "source": "seed_baseline",
                "composition": "baseline_gpbo",
                "score": base_ref,
                "tier": "baseline",
                "note": "confirm failed to beat baseline; demote",
            }
            status["phase"] = "report"
            status["queue"] = []
            status["state"] = "done"
            status["stop_reason"] = "done_no_improvement"
            verdict = "confirm_reject"

    elif tier == "full":
        status["budget"]["full_runs"] = int(status["budget"].get("full_runs", 0)) + 1
        if delta > 1e-6:
            status["champion"] = {
                "source": f"full:{winner}",
                "composition": winner,
                "score": scores[winner],
                "tier": "full",
                "delta_vs_baseline_gpbo": delta,
                "summary": summary[winner],
                "ship_ready": True,
            }
            status["state"] = "done"
            status["stop_reason"] = "done_full_champion"
            status["phase"] = "report"
            status["queue"] = []
            verdict = "champion_promoted"
        else:
            status["champion"] = {
                "source": "seed_baseline",
                "composition": "baseline_gpbo",
                "score": base_ref,
                "tier": "baseline",
                "note": "full worse than baseline; force baseline",
            }
            status["state"] = "done"
            status["stop_reason"] = "done_no_improvement"
            status["phase"] = "report"
            status["queue"] = []
            verdict = "full_reject"
    else:
        verdict = "noop"

    # budget stop
    if status["budget"]["wall_hours_used"] >= status["budget"]["max_wall_hours"]:
        status["state"] = "done"
        status["stop_reason"] = "done_budget"
        status["queue"] = []

    status["last_event"] = f"judge_{tier} winner={winner} verdict={verdict}"
    (ROOT / "history" / "champion.json").write_text(
        json.dumps(status["champion"], indent=2), encoding="utf-8"
    )
    _save_status(status)
    _ledger({
        "event": f"judge_{tier}",
        "scores": scores,
        "baseline_gpbo": base_ref,
        "winner": winner,
        "delta": delta,
        "verdict": verdict,
        "state": status["state"],
        "stop_reason": status.get("stop_reason"),
        "next_queue": status.get("queue"),
    })

    if status["state"] == "done":
        final = ROOT / "reports" / "FINAL.md"
        final.write_text(
            f"# FINAL\n\n"
            f"- state: `{status['state']}`\n"
            f"- stop_reason: `{status.get('stop_reason')}`\n"
            f"- champion:\n\n```json\n{json.dumps(status['champion'], indent=2)}\n```\n",
            encoding="utf-8",
        )
        print("Wrote", final)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
