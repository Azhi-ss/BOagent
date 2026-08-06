"""Component comparator: metrics + trajectory analysis.

Scoring follows the competition README §4 — four core metrics per dataset,
each reported as mean ± std with 95% CI over 20 independent runs:

  - best_found            (higher better)  final best yield in 40 rounds
  - initial_round_found_best (higher better)  best yield after round 1
  - t95                   (lower better)  first round reaching 95% of global best
  - AUC_best_so_far       (higher better)  mean of best-so-far trajectory

The competition does NOT publish a single composite formula. ``composite_score``
is an *internal* accept/reject signal for the auto-research loop, built so that:

  1. every term is normalized to [0,1] (cross-dataset comparable)
  2. lower confidence bound (mean − CI95) is used — CI95 already embeds 1.96·se
  3. datasets are weighted equally (no suzuki dominance from raw yield scale)
  4. no single metric can dominate by scale (the old t95 bug)

``score_composition`` returns the composite; ``composite_score`` is kept as a
thin alias for backward compatibility with agent_step.py.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from bo_core.benchmark.data_loader import load_dataset

# Competition budget: 40 rounds per run. t95 beyond this is a failure.
N_ITERS = 40

# Default metric weights inside one dataset (sum = 1.0).
# best_found is the headline metric (README §6.2 "核心指标"); AUC captures the
# whole trajectory; t95 captures convergence speed; initial captures warm-start.
DEFAULT_METRIC_WEIGHTS = {
    "best_found": 0.40,
    "AUC_best_so_far": 0.30,
    "t95": 0.20,
    "initial_round_found_best": 0.10,
}

# Default per-dataset weights (sum = 1.0). Equal weight: README does not rank
# datasets; both are "core optimization" tasks.
DEFAULT_DATASET_WEIGHTS = {
    "buchwald_sub4": 0.5,
    "suzuki": 0.5,
}

# z used when building ci95 = z * std / sqrt(n) in aggregate_results.
# _normalize_metric must NOT multiply by this again: ci95 is already the half-width.
LCB_Z = 1.96


def assert_seed_completeness(
    results: list[dict[str, Any]],
    expected_seeds: list[int],
    datasets: list[str],
    compositions: list[str],
) -> None:
    """Require exactly one result for every expected matrix cell."""
    if not results:
        raise ValueError("Seed completeness check requires results")
    if not expected_seeds:
        raise ValueError("Seed completeness check requires expected seeds")
    if not datasets:
        raise ValueError("Seed completeness check requires datasets")
    if not compositions:
        raise ValueError("Seed completeness check requires compositions")

    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Expected seeds contain duplicates")
    if len(set(datasets)) != len(datasets):
        raise ValueError("Datasets contain duplicates")
    if len(set(compositions)) != len(compositions):
        raise ValueError("Compositions contain duplicates")

    expected = {
        (composition, dataset, seed)
        for composition in compositions
        for dataset in datasets
        for seed in expected_seeds
    }
    counts = Counter(
        (str(result["composition"]), str(result["dataset"]), int(result["seed"]))
        for result in results
    )
    actual = set(counts)
    problems = [
        *(f"missing {'/'.join(map(str, cell))}" for cell in sorted(expected - actual)),
        *(
            f"unexpected {'/'.join(map(str, cell))}"
            for cell in sorted(actual - expected)
        ),
        *(
            f"duplicate {'/'.join(map(str, cell))} ({count} results)"
            for cell, count in sorted(counts.items())
            if count > 1
        ),
    ]
    if problems:
        raise ValueError(
            "Seed completeness check failed (unfair comparison):\n  "
            + "\n  ".join(problems)
        )


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    expected_seeds: list[int],
    datasets: list[str],
    compositions: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-(composition, dataset) across seeds.

    Returns ``summary[comp][dataset][metric] = {mean, std, ci95, min, max, n}``.
    """
    if not results:
        raise ValueError("aggregate_results: results must not be empty")

    protocols = {
        str(r.get("prior_protocol", "legacy_full_prior"))
        for r in results
    }
    if len(protocols) > 1:
        raise ValueError(
            f"Cannot aggregate mixed prior_protocol values: {sorted(protocols)}"
        )
    assert_seed_completeness(results, expected_seeds, datasets, compositions)

    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for r in results:
        grouped.setdefault((r["composition"], r["dataset"]), []).append(r["metrics"])

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for (comp, ds), metrics_list in grouped.items():
        entry: dict[str, dict[str, float]] = {}
        n = len(metrics_list)
        for metric in ("best_found", "initial_round_found_best", "t95", "AUC_best_so_far"):
            vals = np.array([m[metric] for m in metrics_list], dtype=float)
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            entry[metric] = {
                "mean": float(np.mean(vals)),
                "std": std,
                "ci95": float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": n,
            }
        summary.setdefault(comp, {})[ds] = entry
    return summary


def _normalize_metric(metric: str, mean: float, ci95: float, dataset: str) -> float:
    """Normalize one metric to [0,1] with a lower-confidence-bound penalty.

    ``ci95`` is already the half-width ``1.96 * std / sqrt(n)`` from
    ``aggregate_results``. The LCB is therefore ``mean - ci95`` (not
    ``mean - 1.96 * ci95``).

    - Yield metrics (best_found, AUC, initial): (mean - ci95) / registered global best.
    - t95 (lower better): 1 - min(t95, N_ITERS) / N_ITERS, minus ci95 / N_ITERS.
      t95 = 0  -> 1.0 (instant convergence)
      t95 >= 40 -> 0.0 (never reached within budget)
      t95 = 41 (not reached, README penalty) -> clamped to 0.0
    """
    gbest = load_dataset(dataset).global_best
    if metric == "t95":
        # Convergence speed: 1.0 at t95=0, 0.0 at t95>=N_ITERS.
        raw = 1.0 - min(mean, float(N_ITERS)) / N_ITERS
        # Penalize uncertainty: high ci95 on t95 means inconsistent convergence.
        return max(0.0, raw - ci95 / N_ITERS)
    # Yield metrics: LCB of mean, normalized by global best.
    raw = (mean - ci95) / gbest
    return max(0.0, min(1.0, raw))


def score_composition(
    ds_map: dict[str, dict[str, dict[str, float]]],
    metric_weights: dict[str, float] | None = None,
    dataset_weights: dict[str, float] | None = None,
) -> float:
    """Score one composition across datasets. Returns composite in [0,1].

    For each dataset:
        ds_score = Σ_metric w_m · normalize(metric, LCB)
    Composite:
        Σ_dataset w_ds · ds_score
    """
    mw = metric_weights or DEFAULT_METRIC_WEIGHTS
    dw = dataset_weights or DEFAULT_DATASET_WEIGHTS

    total = 0.0
    weight_sum = 0.0
    for ds, metrics in ds_map.items():
        ds_score = 0.0
        for metric, w in mw.items():
            if metric not in metrics:
                continue
            m = metrics[metric]
            ds_score += w * _normalize_metric(metric, m["mean"], m["ci95"], ds)
        w_ds = dw.get(ds, 0.5)
        total += w_ds * ds_score
        weight_sum += w_ds
    return total / weight_sum if weight_sum else float("-inf")


def composite_score(
    summary: dict[str, dict[str, dict[str, float]]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Backward-compatible alias: returns ``{comp: score}`` for all comps.

    ``weights`` (if given) is interpreted as dataset weights, matching the old
    call sites in agent_step.py.
    """
    if weights is None:
        dw = None
    else:
        # Old callers passed a dict mixing metric & dataset weights; we only
        # honor dataset keys here to avoid the old t95-scale bug.
        dw = {k: v for k, v in weights.items() if k in DEFAULT_DATASET_WEIGHTS}
    return {
        comp: score_composition(ds_map, dataset_weights=dw)
        for comp, ds_map in summary.items()
    }


def trajectory_analysis(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze convergence speed, LLM usage, acquisition switching."""
    yields = [t["observed_yield"] for t in trajectory]
    best_so_far = []
    cur = float("-inf")
    for y in yields:
        cur = max(cur, y)
        best_so_far.append(cur)

    improvements = sum(1 for i in range(1, len(best_so_far)) if best_so_far[i] > best_so_far[i - 1])
    llm_actions = [t.get("llm_action") for t in trajectory if t.get("llm_action")]
    acq_switches = sum(
        1 for i in range(1, len(trajectory))
        if trajectory[i].get("acquisition") != trajectory[i - 1].get("acquisition")
    )

    return {
        "n_iters": len(trajectory),
        "n_improvements": improvements,
        "improvement_rate": improvements / max(len(trajectory), 1),
        "llm_action_count": len(llm_actions),
        "llm_action_types": list(set(llm_actions)),
        "acq_switches": acq_switches,
        "final_best": best_so_far[-1] if best_so_far else None,
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    scores: dict[str, float],
    analyses: dict[str, Any],
) -> Path:
    """Write markdown comparison report following README §4 layout.

    Reports all four metrics with mean ± std and 95% CI per dataset, exactly
    as the competition requires; plus the normalized composite for ranking.
    """
    lines = [
        "# Auto-Research Component Comparison Report",
        "",
        "## Composite Scores (normalized [0,1], higher = better)",
        "",
        "Composite = Σ_dataset w_ds · Σ_metric w_m · normalize_LCB(metric)",
        "",
        "| Composition | Score |",
        "|-------------|-------|",
    ]
    for comp, score in sorted(scores.items(), key=lambda x: -x[1]):
        lines.append(f"| {comp} | {score:.4f} |")

    lines.extend([
        "",
        "## Per-Dataset Metrics (mean ± std, 95% CI)",
        "",
    ])
    for comp, ds_map in summary.items():
        lines.append(f"### {comp}")
        lines.append(
            "| Dataset | best_found | initial_round_found_best | t95 | AUC_best_so_far |"
        )
        lines.append("|---------|------------|--------------------------|-----|------------------|")
        for ds, m in ds_map.items():
            lines.append(
                f"| {ds} "
                f"| {m['best_found']['mean']:.2f}±{m['best_found']['std']:.2f} "
                  f"(CI {m['best_found']['ci95']:.2f}) "
                f"| {m['initial_round_found_best']['mean']:.2f}±{m['initial_round_found_best']['std']:.2f} "
                  f"(CI {m['initial_round_found_best']['ci95']:.2f}) "
                f"| {m['t95']['mean']:.1f}±{m['t95']['std']:.1f} "
                  f"(CI {m['t95']['ci95']:.2f}) "
                f"| {m['AUC_best_so_far']['mean']:.2f}±{m['AUC_best_so_far']['std']:.2f} "
                  f"(CI {m['AUC_best_so_far']['ci95']:.2f}) |"
            )
        lines.append("")

    lines.extend([
        "## Trajectory Analysis",
        "",
        "| Composition | Improvements | LLM Actions | Acq Switches | Final Best |",
        "|-------------|--------------|-------------|--------------|------------|",
    ])
    for comp, a in analyses.items():
        lines.append(
            f"| {comp} | {a.get('n_improvements', 'n/a')} "
            f"| {a.get('llm_action_count', 'n/a')} "
            f"| {a.get('acq_switches', 'n/a')} "
            f"| {a.get('final_best', 'n/a')} |"
        )

    path = output_dir / "comparison_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
