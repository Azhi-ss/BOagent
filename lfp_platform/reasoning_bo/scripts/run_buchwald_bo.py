#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run discrete BO on the Buchwald_sub4 benchmark and dump a competition-format
``.pt`` trajectory per random seed.

Pipeline:
  1. Resolve the dataset (repo-shared ``datasets/chemical_reactions/buchwald_sub4``
     or legacy ``data/Buchwald_sub4`` with zip extraction) and load
     train/test/options.
  2. Build an Ax search space of 4 ChoiceParameters (Reactant2/Ligand/Additive/Base).
  3. Seed the experiment with this product's 7 labeled train rows.
  4. For each of ``--num_iterations`` rounds, use the Ax/BoTorch qLogNEI engine
     (BOModel.gen) to propose one combo; snap it to a valid, not-yet-queried
     test-pool row (falling back to the GP-best unqueried candidate when the
     proposal is invalid or a duplicate); query the offline oracle for its Yield.
  5. Save ``seed_<seed>.pt`` with a 40-step trajectory (step, query_index,
     condition, observed_yield) plus a best-so-far summary CSV.

Usage (from reasoning_bo repo root):

    python scripts/run_buchwald_bo.py --num_iterations 40 --seeds 100 200
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ax import (
    Arm,
    ChoiceParameter,
    Experiment,
    GeneratorRun,
    Objective,
    OptimizationConfig,
    ParameterType,
    Runner,
    SearchSpace,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bo.models import BOModel  # noqa: E402
from src.bo.pool_bo import PoolBO  # noqa: E402
from src.tasks.buchwald import BUCHWALD_PARAM_NAMES as PARAM_NAMES, BuchwaldMetric  # noqa: E402

SUB4_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"
DEFAULT_SEEDS = [
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
]


class SimpleRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}


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


def build_search_space(options: dict) -> SearchSpace:
    return SearchSpace(
        parameters=[
            ChoiceParameter(
                name=name,
                parameter_type=ParameterType.STRING,
                values=list(options[name]),
            )
            for name in PARAM_NAMES
        ]
    )


def best_unqueried_by_gp(
    bo_model: BOModel,
    unqueried_keys: list[tuple[str, ...]],
    sample: int = 150,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    """Pick the unqueried pool row with the highest GP posterior mean.

    Falls back to a random unqueried row if the model has no posterior yet.
    """
    if not unqueried_keys:
        raise RuntimeError("No unqueried pool rows left")
    pool = unqueried_keys
    if rng is not None and len(pool) > sample:
        pool = rng.sample(pool, sample)
    preds = bo_model.predict_posterior(
        [dict(zip(PARAM_NAMES, k)) for k in pool]
    )
    means = [p["mean"] for p in preds]
    if any(m is None for m in means):
        return rng.choice(unqueried_keys) if rng is not None else unqueried_keys[0]
    best_i = int(np.nanargmax(means))
    return pool[best_i]


def run_one_seed(
    seed: int,
    paths: dict,
    options: dict,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    num_iterations: int,
    out_dir: Path,
    *,
    acq: str = "pool",
    surrogate: str = "single_task",
    nei_restarts: int = 5,
    nei_raw_samples: int = 128,
    mle_maxiter: int = 50,
    n_restarts: int = 5,
    mcmc_warmup: int = 128,
    mcmc_samples: int = 64,
    mcmc_thinning: int = 8,
    device: str = "cpu",
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
    search_space = build_search_space(options)
    optimization_config = OptimizationConfig(
        objective=Objective(metric=metric, minimize=False)
    )
    experiment = Experiment(
        name="buchwald_sub4",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SimpleRunner(),
    )

    # Valid test-pool rows (the only allowed query targets).
    pool_keys = [
        tuple(str(row[c]) for c in PARAM_NAMES) for _, row in test_df.iterrows()
    ]
    pool_index_map = {k: i for i, k in enumerate(pool_keys)}
    queried: set[tuple[str, ...]] = set()
    queried_mask = np.zeros(len(pool_keys), dtype=bool)

    # Pool-based fast path precomputes one-hot features once.
    pool_bo: PoolBO | None = None
    if acq == "pool":
        pool_bo = PoolBO(
            pool_keys=pool_keys,
            options_per_var=[options[n] for n in PARAM_NAMES],
            param_names=PARAM_NAMES,
            maxiter=mle_maxiter,
            surrogate=surrogate,
            mcmc_warmup=mcmc_warmup,
            mcmc_samples=mcmc_samples,
            mcmc_thinning=mcmc_thinning,
            device=device,
        )

    # ---- Seed trial: this product's 7 labeled train rows ----
    init_rows = train_df[train_df["Product"] == SUB4_PRODUCT]
    init_arms = [
        Arm(parameters={n: str(row[n]) for n in PARAM_NAMES})
        for _, row in init_rows.iterrows()
    ]
    if init_arms:
        trial = experiment.new_batch_trial(generator_run=GeneratorRun(arms=init_arms))
        trial.run()
        trial.mark_completed()

    trajectory: list[dict] = []
    best_so_far = -float("inf")
    iter_times: list[float] = []

    # ---- BO loop: 1 query per round ----
    for step in range(1, num_iterations + 1):
        unqueried = [k for k in pool_keys if k not in queried]
        if not unqueried:
            print(f"[seed {seed}] pool exhausted at step {step}")
            break

        t0 = time.time()
        chosen_key: tuple[str, ...] | None = None
        try:
            if acq == "pool":
                obs_keys = [
                    tuple(str(row[c]) for c in PARAM_NAMES)
                    for _, row in init_rows.iterrows()
                ] + [k for k in queried]
                obs_y = [float(metric.yield_map[k]) for k in obs_keys]
                pick = pool_bo.suggest(
                    obs_keys, obs_y, queried_mask, q=1
                )[0]
                chosen_key = pool_keys[pick]
            else:
                from ax.modelbridge.registry import Models
                from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
                from ax.models.torch.botorch_modular.utils import ModelConfig
                from botorch.acquisition.logei import qLogNoisyExpectedImprovement
                from botorch.models.gp_regression import SingleTaskGP

                mb = Models.BOTORCH_MODULAR(
                    experiment=experiment,
                    data=experiment.fetch_data(),
                    surrogate_spec=SurrogateSpec(
                        model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
                    ),
                    botorch_acqf_class=qLogNoisyExpectedImprovement,
                )
                gr = mb.gen(
                    n=1,
                    model_gen_options={
                        "acquisition_optimizer_kwargs": {
                            "num_restarts": nei_restarts,
                            "raw_samples": nei_raw_samples,
                        }
                    },
                )
                proposed = gr.arms[0].parameters
                proposed_key = tuple(str(proposed[n]) for n in PARAM_NAMES)
                if proposed_key in pool_index_map and proposed_key not in queried:
                    chosen_key = proposed_key
                else:
                    bo_model = BOModel(experiment)
                    bo_model.model_bridge = mb  # reuse the NEI-fitted GP, not a fresh None model
                    chosen_key = best_unqueried_by_gp(bo_model, unqueried, rng=rng)
        except Exception as e:  # noqa: BLE001
            print(f"[seed {seed}] step {step} suggest failed ({e}); random fallback")
            chosen_key = rng.choice(unqueried)

        dt = time.time() - t0
        iter_times.append(dt)

        queried.add(chosen_key)
        query_index = pool_index_map[chosen_key]
        queried_mask[query_index] = True
        arm = Arm(parameters=dict(zip(PARAM_NAMES, chosen_key)))
        trial = experiment.new_trial(generator_run=GeneratorRun(arms=[arm]))
        trial.run()
        trial.mark_completed()

        observed_yield = float(metric.yield_map[chosen_key])
        best_so_far = max(best_so_far, observed_yield)
        trajectory.append(
            {
                "step": step,
                "query_index": int(query_index),
                "condition": {n: chosen_key[i] for i, n in enumerate(PARAM_NAMES)},
                "observed_yield": observed_yield,
                "iter_seconds": round(dt, 3),
            }
        )
        print(
            f"[seed {seed}] step {step:2d} | idx={query_index:3d} | "
            f"yield={observed_yield:6.2f} | best={best_so_far:6.2f} | "
            f"{dt:5.1f}s"
        )

    avg_iter = float(np.mean(iter_times)) if iter_times else 0.0
    payload = {
        "seed": seed,
        "dataset": "Buchwald_sub4",
        "num_iterations": num_iterations,
        "acq": acq,
        "trajectory": trajectory,
        "best_found": float(best_so_far),
        "avg_iter_seconds": round(avg_iter, 3),
        "total_seconds": round(float(np.sum(iter_times)), 3),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / f"seed_{seed}.pt")

    summary_path = out_dir / f"seed_{seed}_trajectory.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "query_index", "observed_yield", "best_so_far"])
        bsf = -float("inf")
        for rec in trajectory:
            bsf = max(bsf, rec["observed_yield"])
            writer.writerow([rec["step"], rec["query_index"], rec["observed_yield"], bsf])

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Buchwald_sub4 discrete BO runner")
    parser.add_argument("--num_iterations", type=int, default=40)
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="Random seeds (default: the 20 competition seeds 100..2000).",
    )
    parser.add_argument(
        "--acq", choices=["pool", "nei"], default="pool",
        help="pool: batch qLogEI over the discrete pool (fast). "
             "nei: Ax qLogNoisyExpectedImprovement with reduced restarts.",
    )
    parser.add_argument("--nei_restarts", type=int, default=5)
    parser.add_argument("--nei_raw_samples", type=int, default=128)
    parser.add_argument("--mle_maxiter", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5, help="Random restarts for SingleTaskGP MLE.")
    parser.add_argument(
        "--surrogate", choices=["single_task", "saas"], default="single_task",
        help="GP surrogate for the pool path: single_task (SingleTaskGP + MLE) "
             "or saas (SAASBO + NUTS MCMC, horseshoe prior). Ignored for --acq nei.",
    )
    parser.add_argument("--mcmc_warmup", type=int, default=128,
                         help="NUTS warmup steps (saas only).")
    parser.add_argument("--mcmc_samples", type=int, default=64,
                         help="NUTS num_samples (saas only).")
    parser.add_argument("--mcmc_thinning", type=int, default=8,
                         help="NUTS thinning (saas only).")
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Default: data/results/buchwald_sub4_bo_<acq>[_<surrogate>]",
    )
    parser.add_argument("--device", default=None,
                        help="torch device for GP fitting (default: cuda if available else cpu).")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = resolve_dataset("Buchwald_sub4", prefix="buchwald_sub4")
    options = json.load(open(paths["options"], encoding="utf-8"))
    test_df = pd.read_csv(paths["test"])
    train_df = pd.read_csv(paths["train"])

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    suffix = f"_{args.acq}" if args.acq != "pool" else ""
    if args.acq == "pool" and args.surrogate != "single_task":
        suffix = f"_{args.acq}_{args.surrogate}"
    out_dir = Path(
        args.result_dir
        if args.result_dir
        else str(ROOT / "data" / "results" / f"buchwald_sub4_bo{suffix}")
    )

    # Global best from the full search space (for t95 / round-to-95%-global-best).
    searchspace_csv = paths["searchspace"]
    global_best = None
    t95_threshold = None
    if searchspace_csv.exists():
        ss_df = pd.read_csv(searchspace_csv)
        global_best = float(ss_df["Yield"].max())
        t95_threshold = 0.95 * global_best
        print(f"Global best (searchspace) = {global_best:.3f} | 95% threshold = {t95_threshold:.3f}")

    print("=" * 70)
    print(f"Buchwald_sub4 BO | acq={args.acq} | surrogate={args.surrogate} | iters={args.num_iterations} | seeds={seeds}")
    print(f"data_dir={paths['dir']}")
    print(f"result_dir={out_dir}")
    print(f"pool={len(test_df)} rows | train(merged)={len(train_df)} rows")
    print("=" * 70)

    summary = []
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        payload = run_one_seed(
            seed=seed,
            paths=paths,
            options=options,
            test_df=test_df,
            train_df=train_df,
            num_iterations=args.num_iterations,
            out_dir=out_dir,
            acq=args.acq,
            surrogate=args.surrogate,
            nei_restarts=args.nei_restarts,
            nei_raw_samples=args.nei_raw_samples,
            mle_maxiter=args.mle_maxiter,
            n_restarts=args.n_restarts,
            mcmc_warmup=args.mcmc_warmup,
            mcmc_samples=args.mcmc_samples,
            mcmc_thinning=args.mcmc_thinning,
            device=args.device,
        )
        # round-to-95%-global-best (t95): first step whose best-so-far >= threshold.
        t95 = None
        if t95_threshold is not None:
            bsf = -float("inf")
            for rec in payload["trajectory"]:
                bsf = max(bsf, rec["observed_yield"])
                if bsf >= t95_threshold:
                    t95 = rec["step"]
                    break
        summary.append(
            {
                "seed": seed,
                "best_found": payload["best_found"],
                "t95": t95,
                "reached_95": t95 is not None,
                "avg_iter_seconds": payload.get("avg_iter_seconds", 0.0),
                "total_seconds": payload.get("total_seconds", 0.0),
            }
        )

    print("\n" + "=" * 70)
    if global_best is not None:
        print(f"Global best = {global_best:.3f} | 95% threshold = {t95_threshold:.3f}")
    print(f"acq={args.acq} | Summary (best_found / t95 / avg_iter_s per seed):")
    for row in summary:
        t95_str = f"t95={row['t95']}" if row["reached_95"] else "t95=not-reached"
        print(
            f"  seed {row['seed']:5d}: best={row['best_found']:.3f} | {t95_str} "
            f"| avg_iter={row['avg_iter_seconds']:.2f}s"
        )
    best_vals = [r["best_found"] for r in summary]
    t95_vals = [r["t95"] for r in summary if r["reached_95"]]
    iter_vals = [r["avg_iter_seconds"] for r in summary]
    if best_vals:
        print(
            f"best_found: mean={np.mean(best_vals):.3f} std={np.std(best_vals):.3f} "
            f"min={np.min(best_vals):.3f} max={np.max(best_vals):.3f}"
        )
    if iter_vals:
        print(
            f"avg_iter_seconds: mean={np.mean(iter_vals):.2f}s "
            f"(total BO time={np.sum([r['total_seconds'] for r in summary]):.1f}s)"
        )
    if t95_vals:
        print(
            f"t95 (rounds to 95% global best): "
            f"mean={np.mean(t95_vals):.1f} std={np.std(t95_vals):.1f} "
            f"min={np.min(t95_vals)} max={np.max(t95_vals)} "
            f"(reached {len(t95_vals)}/{len(summary)} seeds)"
        )
    else:
        print("t95: no seed reached 95% of global best within the budget")
    print(f"Saved {len(seeds)} .pt files -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
