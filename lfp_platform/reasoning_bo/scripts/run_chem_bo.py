#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""General discrete chemistry BO runner (Buchwald_sub4 / Suzuki).

Same pipeline as ``run_buchwald_bo.py`` but dataset-aware:
  - resolve the dataset (repo-shared ``datasets/chemical_reactions/<name>`` or
    legacy ``data/<dir>`` with zip extraction)
  - build an Ax search space of ChoiceParameters from ``options.json``
  - seed the experiment with the dataset's labeled train rows
    (Buchwald_sub4: filter the merged train by this product; Suzuki: all rows)
  - per round: pool-based qLogEI (with random restarts) picks one valid,
    not-yet-queried test-pool row; query the offline oracle for its Yield
  - save ``seed_<seed>.pt`` (competition format) + trajectory CSV

Usage (from reasoning_bo repo root):

    python scripts/run_chem_bo.py --dataset Suzuki --num_iterations 40
    python scripts/run_chem_bo.py --dataset Buchwald_sub4 --acq nei --seeds 100
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
from src.prompts.base import PromptManager  # noqa: E402
from src.tasks.buchwald import DiscreteChemMetric  # noqa: E402

DEFAULT_SEEDS = [
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
]

DATASETS = {
    "Buchwald_sub4": {
        "dir": "Buchwald_sub4",
        "prefix": "buchwald_sub4",
        "param_names": ["Reactant2", "Ligand", "Additive", "Base"],
        "init_product": "N-(4-ethylphenyl)-4-methylaniline",
        "chemistry_context": (
            "反应类型：Buchwald-Hartwig C-N 胺化偶联（钯催化芳基卤代物与胺偶联形成 C-N 键）。\n"
            "固定组分：Reactant1=4-methylaniline(胺)，Solvent=methylsulfinylmethane(DMSO)，Catalyst=palladium(2+) diacetate。\n"
            "决策变量：Reactant2(芳基卤代物，Cl/Br/I)、Ligand(磷配体)、Additive(异噁唑类添加剂)、Base。\n"
            "机理要点：氧化加成→胺配位→还原消除；联芳基大位阻膦配体(BrettPhos/XPhos 类)通常利于芳基卤代物胺化；"
            "强且位阻大的碱有利；不同异噁唑添加剂显著调节产率；芳基碘/溴通常比氯更活泼。"
        ),
    },
    "Suzuki": {
        "dir": "Suzuki",
        "prefix": "suzuki",
        "param_names": ["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"],
        "init_product": None,
        "chemistry_context": (
            "反应类型：Suzuki-Miyaura C-C 偶联（钯催化芳基卤代物与芳基硼试剂偶联形成 C-C 键）。\n"
            "固定组分：Catalyst=palladium(2+) diacetate，产物固定。\n"
            "决策变量：Electrophile(芳基卤代物/三氟甲磺酸酯)、Nucleophile(芳基硼试剂：硼酸/硼酸酯/三氟硼酸盐)、"
            "Ligand、Base、Solvent。\n"
            "机理要点：氧化加成→转金属→还原消除；硼试剂形式影响转金属效率(三氟硼酸盐/硼酸酯稳定性高)；"
            "碱(K3PO4/KF/Cs2CO3/KOH)与溶剂搭配影响活性物种生成；位阻大的底物倾向 SPhos/XPhos 类配体；"
            "'Nothing' 表示不加入该试剂。"
        ),
    },
}


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


def build_search_space(options: dict, param_names: list[str]) -> SearchSpace:
    return SearchSpace(
        parameters=[
            ChoiceParameter(
                name=name,
                parameter_type=ParameterType.STRING,
                values=list(options[name]),
            )
            for name in param_names
        ]
    )


def best_unqueried_by_gp(bo_model, unqueried_keys, param_names, sample=150, rng=None):
    if not unqueried_keys:
        raise RuntimeError("No unqueried pool rows left")
    pool = unqueried_keys
    if rng is not None and len(pool) > sample:
        pool = rng.sample(pool, sample)
    preds = bo_model.predict_posterior(
        [dict(zip(param_names, k)) for k in pool]
    )
    means = [p["mean"] for p in preds]
    if any(m is None for m in means):
        return rng.choice(unqueried_keys) if rng is not None else unqueried_keys[0]
    return pool[int(np.nanargmax(means))]


def _render_parameter_definitions(options: dict, param_names: list[str]) -> str:
    lines = []
    for n in param_names:
        vals = options[n]
        lines.append(f"- {n}（{len(vals)} 个候选）: " + "; ".join(vals))
    return "\n".join(lines)


def _render_history(observed_keys, observed_y, param_names, limit=20) -> str:
    if not observed_keys:
        return "（暂无已观测实验）"
    items = list(zip(observed_keys, observed_y))
    # show most recent `limit`, most-recent last
    shown = items[-limit:]
    lines = []
    for i, (k, y) in enumerate(shown, start=max(1, len(items) - limit + 1)):
        cond = ", ".join(f"{n}={k[j]}" for j, n in enumerate(param_names))
        lines.append(f"{i}. {cond} -> {y:.2f}%")
    if len(items) > limit:
        lines.append(f"...（共 {len(items)} 条，仅显示最近 {limit} 条）")
    return "\n".join(lines)


def _render_bo_recommendations(records: list[dict], param_names) -> str:
    if not records:
        return "（BO 本轮无可用候选）"
    lines = []
    for i, r in enumerate(records, 1):
        cond = ", ".join(f"{n}={r['params'][n]}" for n in param_names)
        lines.append(
            f"[候选 {i}] pool_index={r['index']} | {cond} | "
            f"predicted_mean={r['mean']:.2f} | predicted_std={r['std']:.2f} | "
            f"EI={r['ei']:.4f} | role={r['role']}"
        )
    return "\n".join(lines)


def _parse_llm_choice(content: str):
    """Extract chosen_index from LLM JSON output (tolerate markdown wrapping)."""
    import re
    text = content.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
    # find the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start:end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # salvage chosen_index via regex
        m = re.search(r'"chosen_index"\s*:\s*([0-9]+)', blob)
        return {"chosen_index": int(m.group(1))} if m else None
    return obj


def llm_pick(
    llm_client, prompt_manager, cfg, options, param_names,
    iteration, num_iterations, best_so_far,
    observed_keys, observed_y, bo_records,
):
    """Ask the LLM to pick one pool row. Returns (chosen_index, reasoning) or None."""
    prompt = prompt_manager.format(
        "chem_optimization_loop", lang="zh",
        chemistry_context=cfg["chemistry_context"],
        target="最大化 Yield (%)",
        iteration=iteration,
        num_iterations=num_iterations,
        best_so_far=f"{best_so_far:.2f}",
        parameter_definitions=_render_parameter_definitions(options, param_names),
        history=_render_history(observed_keys, observed_y, param_names),
        bo_recommendations=_render_bo_recommendations(bo_records, param_names),
    )
    content, _ = llm_client.generate(prompt, json_output=True)
    obj = _parse_llm_choice(content)
    if not obj or "chosen_index" not in obj:
        print(f"[LLM] failed to parse chosen_index from: {content[:200]}")
        return None
    try:
        idx = int(obj["chosen_index"])
    except (TypeError, ValueError):
        return None
    return idx, str(obj.get("reasoning", ""))[:300]


def run_one_seed(
    seed, paths, options, test_df, train_df, cfg, num_iterations, out_dir,
    *, acq="pool", surrogate="single_task", nei_restarts=5, nei_raw_samples=128, mle_maxiter=50, n_restarts=5,
    mcmc_warmup=128, mcmc_samples=64, mcmc_thinning=8,
    use_llm=False, llm_top_k=5, llm_backend="deepseek", device="cpu",
):
    param_names = cfg["param_names"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    metric = DiscreteChemMetric(
        name=cfg["dir"],
        param_names=param_names,
        test_csv=paths["test"],
        train_csv=paths["train"],
    )
    search_space = build_search_space(options, param_names)
    optimization_config = OptimizationConfig(objective=Objective(metric=metric, minimize=False))
    experiment = Experiment(
        name=cfg["dir"], search_space=search_space,
        optimization_config=optimization_config, runner=SimpleRunner(),
    )

    pool_keys = [tuple(str(row[c]) for c in param_names) for _, row in test_df.iterrows()]
    pool_index_map = {k: i for i, k in enumerate(pool_keys)}
    queried = set()
    queried_mask = np.zeros(len(pool_keys), dtype=bool)

    pool_bo = None
    if acq == "pool":
        pool_bo = PoolBO(
            pool_keys=pool_keys,
            options_per_var=[options[n] for n in param_names],
            param_names=param_names,
            maxiter=mle_maxiter, n_restarts=n_restarts,
            surrogate=surrogate,
            mcmc_warmup=mcmc_warmup, mcmc_samples=mcmc_samples, mcmc_thinning=mcmc_thinning,
            device=device,
        )

    # ---- LLM reasoner (optional) ----
    llm_client = None
    prompt_manager = None
    if use_llm:
        if llm_backend == "deepseek":
            from src.llms.deepseek import DeepSeekClient
            llm_client = DeepSeekClient()
        elif llm_backend == "qwq":
            from src.llms.qwq import QWQClient
            llm_client = QWQClient()
        else:
            raise ValueError(f"unknown llm_backend {llm_backend}")
        # Cap API latency so a stalled call falls back to BO instead of hanging.
        try:
            llm_client.client = llm_client.client.with_options(timeout=180.0)
        except Exception as e:  # noqa: BLE001
            print(f"[LLM] could not set client timeout ({e})")
        prompt_manager = PromptManager()

    # ---- Seed trial: labeled train rows ----
    if cfg.get("init_product"):
        init_rows = train_df[train_df["Product"] == cfg["init_product"]]
    else:
        init_rows = train_df
    init_arms = [Arm(parameters={n: str(row[n]) for n in param_names}) for _, row in init_rows.iterrows()]
    if init_arms:
        trial = experiment.new_batch_trial(generator_run=GeneratorRun(arms=init_arms))
        trial.run(); trial.mark_completed()

    trajectory = []
    best_so_far = -float("inf")
    iter_times = []

    for step in range(1, num_iterations + 1):
        unqueried = [k for k in pool_keys if k not in queried]
        if not unqueried:
            print(f"[seed {seed}] pool exhausted at step {step}")
            break

        t0 = time.time()
        chosen_key = None
        llm_reasoning = ""
        try:
            if acq == "pool":
                obs_keys = [tuple(str(row[c]) for c in param_names) for _, row in init_rows.iterrows()] + list(queried)
                obs_y = [float(metric.yield_map[k]) for k in obs_keys]
                if use_llm and llm_client is not None:
                    records, _ = pool_bo.rank_pool(obs_keys, obs_y, queried_mask, k=llm_top_k)
                    if not records:
                        pick = pool_bo.suggest(obs_keys, obs_y, queried_mask, q=1)[0]
                    else:
                        pick = records[0]["index"]  # BO fallback
                        try:
                            res = llm_pick(
                                llm_client, prompt_manager, cfg, options, param_names,
                                step, num_iterations, best_so_far,
                                obs_keys, obs_y, records,
                            )
                            if res is not None:
                                idx, llm_reasoning = res
                                if 0 <= idx < len(pool_keys) and not queried_mask[idx]:
                                    pick = idx
                                else:
                                    print(f"[LLM] chosen_index={idx} invalid/queried; fallback BO top-1")
                        except Exception as e:  # noqa: BLE001
                            print(f"[LLM] call failed ({e}); fallback BO top-1")
                else:
                    pick = pool_bo.suggest(obs_keys, obs_y, queried_mask, q=1)[0]
                chosen_key = pool_keys[pick]
            else:
                from ax.modelbridge.registry import Models
                from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
                from ax.models.torch.botorch_modular.utils import ModelConfig
                from botorch.acquisition.logei import qLogNoisyExpectedImprovement
                from botorch.models.gp_regression import SingleTaskGP
                mb = Models.BOTORCH_MODULAR(
                    experiment=experiment, data=experiment.fetch_data(),
                    surrogate_spec=SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]),
                    botorch_acqf_class=qLogNoisyExpectedImprovement,
                )
                gr = mb.gen(n=1, model_gen_options={
                    "acquisition_optimizer_kwargs": {"num_restarts": nei_restarts, "raw_samples": nei_raw_samples}})
                proposed = gr.arms[0].parameters
                proposed_key = tuple(str(proposed[n]) for n in param_names)
                if proposed_key in pool_index_map and proposed_key not in queried:
                    chosen_key = proposed_key
                else:
                    bo_model = BOModel(experiment); bo_model.model_bridge = mb  # reuse the NEI-fitted GP
                    chosen_key = best_unqueried_by_gp(bo_model, unqueried, param_names, rng=rng)
        except Exception as e:  # noqa: BLE001
            print(f"[seed {seed}] step {step} suggest failed ({e}); random fallback")
            chosen_key = rng.choice(unqueried)

        dt = time.time() - t0
        iter_times.append(dt)
        queried.add(chosen_key)
        query_index = pool_index_map[chosen_key]
        queried_mask[query_index] = True
        arm = Arm(parameters=dict(zip(param_names, chosen_key)))
        trial = experiment.new_trial(generator_run=GeneratorRun(arms=[arm]))
        trial.run(); trial.mark_completed()

        observed_yield = float(metric.yield_map[chosen_key])
        best_so_far = max(best_so_far, observed_yield)
        trajectory.append({
            "step": step, "query_index": int(query_index),
            "condition": {n: chosen_key[i] for i, n in enumerate(param_names)},
            "observed_yield": observed_yield, "iter_seconds": round(dt, 3),
            "llm_reasoning": llm_reasoning,
        })
        src_tag = "LLM" if (use_llm and llm_reasoning) else ("BO " if not use_llm else "BO*")
        print(f"[seed {seed}] step {step:2d} | {src_tag} | idx={query_index:4d} | "
              f"yield={observed_yield:6.2f} | best={best_so_far:6.2f} | {dt:5.1f}s")

    avg_iter = float(np.mean(iter_times)) if iter_times else 0.0
    payload = {
        "seed": seed, "dataset": cfg["dir"], "num_iterations": num_iterations,
        "acq": acq, "trajectory": trajectory, "best_found": float(best_so_far),
        "avg_iter_seconds": round(avg_iter, 3), "total_seconds": round(float(np.sum(iter_times)), 3),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / f"seed_{seed}.pt")
    with open(out_dir / f"seed_{seed}_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "query_index", "observed_yield", "best_so_far"])
        bsf = -float("inf")
        for r in trajectory:
            bsf = max(bsf, r["observed_yield"])
            w.writerow([r["step"], r["query_index"], r["observed_yield"], bsf])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Discrete chemistry BO runner")
    parser.add_argument("--dataset", choices=list(DATASETS), default="Suzuki")
    parser.add_argument("--num_iterations", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Random seeds (default: the 20 competition seeds 100..2000).")
    parser.add_argument("--acq", choices=["pool", "nei"], default="pool")
    parser.add_argument(
        "--surrogate", choices=["single_task", "saas"], default="single_task",
        help="GP surrogate for the pool path: single_task (SingleTaskGP + MLE) "
             "or saas (SAASBO + NUTS MCMC, horseshoe prior). Ignored for --acq nei.",
    )
    parser.add_argument("--nei_restarts", type=int, default=5)
    parser.add_argument("--nei_raw_samples", type=int, default=128)
    parser.add_argument("--mle_maxiter", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--mcmc_warmup", type=int, default=128, help="NUTS warmup steps (saas only).")
    parser.add_argument("--mcmc_samples", type=int, default=64, help="NUTS num_samples (saas only).")
    parser.add_argument("--mcmc_thinning", type=int, default=8, help="NUTS thinning (saas only).")
    parser.add_argument("--use_llm", action="store_true",
                        help="Let an LLM re-rank BO's top-k pool candidates each round.")
    parser.add_argument("--llm_top_k", type=int, default=5,
                        help="Number of BO candidates passed to the LLM each round.")
    parser.add_argument("--llm_backend", choices=["deepseek", "qwq"], default="deepseek")
    parser.add_argument("--result_dir", default=None,
                        help="Default: data/results/<dataset>_bo_<acq>[_llm]")
    parser.add_argument("--device", default=None,
                        help="torch device for GP fitting (default: cuda if available else cpu).")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DATASETS[args.dataset]
    paths = resolve_dataset(cfg["dir"], cfg["prefix"])
    options = json.load(open(paths["options"], encoding="utf-8"))
    test_df = pd.read_csv(paths["test"])
    train_df = pd.read_csv(paths["train"])

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    suffix = f"_{args.acq}" + ("_llm" if args.use_llm else "")
    out_dir = Path(args.result_dir) if args.result_dir else (
        ROOT / "data" / "results" / f"{args.dataset}_bo{suffix}")

    searchspace_csv = paths["searchspace"]
    global_best = t95_threshold = None
    if searchspace_csv.exists():
        global_best = float(pd.read_csv(searchspace_csv)["Yield"].max())
        t95_threshold = 0.95 * global_best
        print(f"Global best (searchspace) = {global_best:.3f} | 95% threshold = {t95_threshold:.3f}")

    print("=" * 70)
    llm_tag = f" | LLM={args.llm_backend}(top-{args.llm_top_k})" if args.use_llm else ""
    print(f"{args.dataset} BO | acq={args.acq}{llm_tag} | iters={args.num_iterations} | seeds={seeds} | device={args.device}")
    print(f"data_dir={paths['dir']}")
    print(f"result_dir={out_dir}")
    print(f"pool={len(test_df)} rows | train={len(train_df)} rows | params={cfg['param_names']}")
    print("=" * 70)
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        payload = run_one_seed(
            seed, paths, options, test_df, train_df, cfg,
            args.num_iterations, out_dir, acq=args.acq, surrogate=args.surrogate,
            nei_restarts=args.nei_restarts, nei_raw_samples=args.nei_raw_samples,
            mle_maxiter=args.mle_maxiter, n_restarts=args.n_restarts,
            mcmc_warmup=args.mcmc_warmup, mcmc_samples=args.mcmc_samples, mcmc_thinning=args.mcmc_thinning,
            use_llm=args.use_llm, llm_top_k=args.llm_top_k, llm_backend=args.llm_backend,
            device=args.device)
        t95 = None
        if t95_threshold is not None:
            bsf = -float("inf")
            for r in payload["trajectory"]:
                bsf = max(bsf, r["observed_yield"])
                if bsf >= t95_threshold:
                    t95 = r["step"]; break
        summary.append({"seed": seed, "best_found": payload["best_found"], "t95": t95,
                        "reached_95": t95 is not None,
                        "avg_iter_seconds": payload.get("avg_iter_seconds", 0.0),
                        "total_seconds": payload.get("total_seconds", 0.0)})

    print("\n" + "=" * 70)
    if global_best is not None:
        print(f"Global best = {global_best:.3f} | 95% threshold = {t95_threshold:.3f}")
    print(f"{args.dataset} | acq={args.acq} | Summary (best_found / t95 / avg_iter_s per seed):")
    for row in summary:
        t95s = f"t95={row['t95']}" if row["reached_95"] else "t95=not-reached"
        print(f"  seed {row['seed']:5d}: best={row['best_found']:.3f} | {t95s} | avg_iter={row['avg_iter_seconds']:.2f}s")
    best_vals = [r["best_found"] for r in summary]
    t95_vals = [r["t95"] for r in summary if r["reached_95"]]
    iter_vals = [r["avg_iter_seconds"] for r in summary]
    if best_vals:
        print(f"best_found: mean={np.mean(best_vals):.3f} std={np.std(best_vals):.3f} "
              f"min={np.min(best_vals):.3f} max={np.max(best_vals):.3f}")
    if iter_vals:
        print(f"avg_iter_seconds: mean={np.mean(iter_vals):.2f}s "
              f"(total BO time={np.sum([r['total_seconds'] for r in summary]):.1f}s)")
    if t95_vals:
        print(f"t95 (rounds to 95% global best): mean={np.mean(t95_vals):.1f} "
              f"std={np.std(t95_vals):.1f} min={np.min(t95_vals)} max={np.max(t95_vals)} "
              f"(reached {len(t95_vals)}/{len(summary)} seeds)")
    else:
        print("t95: no seed reached 95% of global best within the budget")
    print(f"Saved {len(seeds)} .pt files -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
