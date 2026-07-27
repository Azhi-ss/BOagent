"""Measure sklearn and BoTorch surrogate costs on chemical candidate pools."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bo_core.benchmark.lgbo_runner import _configure_numerical_threads
from bo_core.optimization.lgbo import LGBOEngine
from bo_core.optimization.surrogate import (
    LBFGSB_MAX_LINE_SEARCH_STEPS,
    BackendName,
    create_surrogate,
)

BenchmarkMode = Literal["sklearn", "botorch_cold", "botorch_warm"]
MODES: tuple[BenchmarkMode, ...] = (
    "sklearn",
    "botorch_cold",
    "botorch_warm",
)
_TIMING_FIELDS = (
    "fit_s",
    "predict_s",
    "covariance_s",
    "total_s",
    "peak_rss_mb",
    "optimization_warning_count",
)


def _new_surrogate(
    mode: BenchmarkMode,
    *,
    seed: int,
    n_restarts: int,
    max_fit_iterations: int,
):
    backend: BackendName = "sklearn" if mode == "sklearn" else "botorch"
    return create_surrogate(
        backend,
        seed=seed,
        n_restarts=n_restarts,
        alpha=1e-2,
        jitter_levels=(1e-2, 1e-1, 1.0),
        max_fit_iterations=max_fit_iterations,
    )


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def run_case(
    dataset: str,
    mode: BenchmarkMode,
    seed: int,
    n_steps: int,
    n_threads: int,
    n_restarts: int = 0,
    max_fit_iterations: int = 100,
    workers: int = 1,
) -> dict[str, Any]:
    """Measure one deterministic growing-data surrogate workload."""
    _configure_numerical_threads(n_threads)
    backend: BackendName = "sklearn" if mode == "sklearn" else "botorch"
    optimization_warning_type: type[Warning] | None = None
    if backend == "botorch":
        from botorch.exceptions.warnings import OptimizationWarning

        optimization_warning_type = OptimizationWarning
    engine = LGBOEngine(
        dataset,
        seed=seed,
        use_llm=False,
        n_iters=0,
        n_restarts=n_restarts,
        backend=backend,
    )
    query_indices = np.random.RandomState(seed).permutation(engine.M)[:n_steps]
    X_obs = np.array(engine.X_obs, copy=True)
    y_obs = np.array(engine.y_obs, copy=True)
    warm_surrogate = (
        _new_surrogate(
            mode,
            seed=seed,
            n_restarts=n_restarts,
            max_fit_iterations=max_fit_iterations,
        )
        if mode in ("sklearn", "botorch_warm")
        else None
    )

    fit_s = 0.0
    predict_s = 0.0
    covariance_s = 0.0
    optimization_warning_count = 0
    total_start = time.perf_counter()
    for index in query_indices:
        surrogate = warm_surrogate or _new_surrogate(
            mode,
            seed=seed,
            n_restarts=n_restarts,
            max_fit_iterations=max_fit_iterations,
        )

        start = time.perf_counter()
        if optimization_warning_type is None:
            surrogate.fit(X_obs, y_obs)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", optimization_warning_type)
                surrogate.fit(X_obs, y_obs)
            optimization_warning_count += sum(
                issubclass(item.category, optimization_warning_type)
                for item in caught
            )
        fit_s += time.perf_counter() - start

        start = time.perf_counter()
        surrogate.predict(engine.pool_X)
        predict_s += time.perf_counter() - start

        distances = np.sum((engine.pool_X - engine.pool_X[index]) ** 2, axis=1)
        grid_size = min(50, engine.M)
        grid_indices = np.argpartition(distances, grid_size - 1)[:grid_size]
        X_grid = engine.pool_X[grid_indices]
        start = time.perf_counter()
        surrogate.posterior_covariance(X_grid)
        surrogate.posterior_cross_covariance(engine.pool_X, X_grid)
        covariance_s += time.perf_counter() - start

        X_obs = np.vstack([X_obs, engine.pool_X[index : index + 1]])
        y_obs = np.append(y_obs, engine.pool_yield[index])

    total_s = time.perf_counter() - total_start
    return {
        "dataset": dataset,
        "mode": mode,
        "backend": backend,
        "seed": seed,
        "n_steps": n_steps,
        "workers": workers,
        "n_threads": n_threads,
        "n_restarts": n_restarts,
        "max_fit_iterations": max_fit_iterations,
        "prior_rows": len(engine.X_obs),
        "pool_rows": int(engine.M),
        "dimensions": int(engine.pool_X.shape[1]),
        "query_indices": [int(index) for index in query_indices],
        "fit_s": fit_s,
        "predict_s": predict_s,
        "covariance_s": covariance_s,
        "total_s": total_s,
        "peak_rss_mb": _peak_rss_mb(),
        "optimization_warning_count": optimization_warning_count,
    }


def summarize(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated seeds without mixing worker configurations."""
    keys = sorted(
        {
            (case["dataset"], case["mode"], case["workers"], case["n_threads"])
            for case in cases
        }
    )
    summary: list[dict[str, Any]] = []
    for dataset, mode, workers, n_threads in keys:
        group = [
            case
            for case in cases
            if (
                case["dataset"],
                case["mode"],
                case["workers"],
                case["n_threads"],
            )
            == (dataset, mode, workers, n_threads)
        ]
        entry: dict[str, Any] = {
            "dataset": dataset,
            "mode": mode,
            "workers": workers,
            "n_threads": n_threads,
            "n": len(group),
        }
        for field in _TIMING_FIELDS:
            values = np.asarray([case[field] for case in group], dtype=float)
            entry[f"{field}_mean"] = float(np.mean(values))
            entry[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary.append(entry)
    return summary


def _run_worker_mode(
    configs: list[tuple[str, BenchmarkMode, int]],
    *,
    workers: int,
    n_steps: int,
    n_restarts: int,
    max_fit_iterations: int,
) -> list[dict[str, Any]]:
    n_threads = 10 if workers == 1 else 1
    results: list[dict[str, Any]] = []
    executor_kwargs: dict[str, Any] = {
        "max_workers": workers,
        "initializer": _configure_numerical_threads,
        "initargs": (n_threads,),
        "max_tasks_per_child": 1,
    }
    with ProcessPoolExecutor(**executor_kwargs) as pool:
        futures = {
            pool.submit(
                run_case,
                dataset,
                mode,
                seed,
                n_steps,
                n_threads,
                n_restarts,
                max_fit_iterations,
                workers,
            ): (dataset, mode, seed)
            for dataset, mode, seed in configs
        }
        for future in as_completed(futures):
            dataset, mode, seed = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[{workers}x{n_threads}] {dataset}/{mode}/seed_{seed}: "
                f"total={result['total_s']:.2f}s peak={result['peak_rss_mb']:.1f} MiB"
            )
    return results


def _interpretation(report: dict[str, Any]) -> list[str]:
    summary = report["summary"]
    sklearn_by_workload = {
        (row["dataset"], row["workers"], row["n_threads"]): row
        for row in summary
        if row["mode"] == "sklearn"
    }
    lines: list[str] = []
    for row in summary:
        if row["mode"] != "botorch_warm":
            continue
        key = (row["dataset"], row["workers"], row["n_threads"])
        sklearn = sklearn_by_workload.get(key)
        if sklearn is None or row["total_s_mean"] <= 0:
            continue
        speed_ratio = sklearn["total_s_mean"] / row["total_s_mean"]
        if speed_ratio >= 1.0:
            comparison = f"{speed_ratio:.2f}x faster"
        else:
            comparison = f"{1.0 / speed_ratio:.2f}x slower"
        lines.append(
            f"- {row['dataset']} / {row['workers']} x {row['n_threads']}: "
            f"warm-start BoTorch was {comparison} than sklearn."
        )

    botorch_cases = [case for case in report["cases"] if case["backend"] == "botorch"]
    fit_count = sum(case["n_steps"] for case in botorch_cases)
    warning_count = sum(
        case["optimization_warning_count"] for case in botorch_cases
    )
    if fit_count:
        warning_rate = 100.0 * warning_count / fit_count
        convergence_note = (
            "The cases completed without this convergence warning."
            if warning_count == 0
            else (
                "The cases completed, but this convergence signal must be resolved "
                "before making BoTorch the default backend."
            )
        )
        lines.append(
            f"- BoTorch emitted {warning_count} `OptimizationWarning` instances "
            f"across {fit_count} fits ({warning_rate:.1f}%). {convergence_note}"
        )
    lines.append(
        "- BoTorch/GPyTorch is the default backend. sklearn is available as a "
        "compatibility backend via explicit selection."
    )
    return lines


def _markdown(report: dict[str, Any]) -> str:
    max_line_search_steps = report["configuration"].get(
        "max_line_search_steps", LBFGSB_MAX_LINE_SEARCH_STEPS
    )
    lines = [
        "# H365 sklearn vs BoTorch Backend Benchmark",
        "",
        "This report measures surrogate work only; it excludes LLM calls and data loading.",
        "Times are means across seeds. Peak RSS is the per-case process high-water mark.",
        "",
        "| Dataset | Mode | Workers x threads | Fit (s) | Predict (s) | Covariance (s) | Total (s) | Peak RSS (MiB) | Opt. warnings/case |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['dataset']} | {row['mode']} | "
            f"{row['workers']} x {row['n_threads']} | "
            f"{row['fit_s_mean']:.3f} | {row['predict_s_mean']:.3f} | "
            f"{row['covariance_s_mean']:.3f} | {row['total_s_mean']:.3f} | "
            f"{row['peak_rss_mb_mean']:.1f} | "
            f"{row['optimization_warning_count_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *_interpretation(report),
            "",
            "## Configuration",
            "",
            f"- Steps per case: {report['configuration']['n_steps']}",
            f"- Seeds: {report['configuration']['seeds']}",
            f"- sklearn optimizer restarts: {report['configuration']['n_restarts']}",
            f"- BoTorch maximum SciPy iterations: {report['configuration']['max_fit_iterations']}",
            f"- BoTorch L-BFGS-B line-search steps per iteration: {max_line_search_steps}",
            "- `n_restarts` configures only sklearn's kernel optimizer; the BoTorch path performs one bounded deterministic SciPy fit.",
            "- Covariance workload: one 50 x 50 posterior block and one pool x 50 cross block per step.",
            "- Peak RSS is the high-water mark of one case process, not aggregate memory across concurrent workers.",
            "- `botorch_warm` reuses fitted hyperparameters; `botorch_cold` creates a fresh model each step.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="buchwald_sub4,suzuki")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seeds", default="100,200,300")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--worker-modes", default="1,4")
    parser.add_argument("--n-restarts", type=int, default=0)
    parser.add_argument("--max-fit-iterations", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/hardware/h365_backend_benchmark.json")
    )
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    invalid_modes = sorted(set(modes) - set(MODES))
    if invalid_modes:
        parser.error(f"unsupported modes: {invalid_modes}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    worker_modes = [
        int(item) for item in args.worker_modes.split(",") if item.strip()
    ]
    configs = [
        (dataset, mode, seed)
        for dataset in datasets
        for mode in modes
        for seed in seeds
    ]

    cases: list[dict[str, Any]] = []
    for workers in worker_modes:
        cases.extend(
            _run_worker_mode(
                configs,
                workers=workers,
                n_steps=args.steps,
                n_restarts=args.n_restarts,
                max_fit_iterations=args.max_fit_iterations,
            )
        )

    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "configuration": {
            "datasets": datasets,
            "modes": modes,
            "seeds": seeds,
            "worker_modes": worker_modes,
            "n_steps": args.steps,
            "n_restarts": args.n_restarts,
            "max_fit_iterations": args.max_fit_iterations,
            "max_line_search_steps": LBFGSB_MAX_LINE_SEARCH_STEPS,
        },
        "cases": sorted(
            cases,
            key=lambda case: (
                case["workers"],
                case["dataset"],
                case["mode"],
                case["seed"],
            ),
        ),
        "summary": summarize(cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
