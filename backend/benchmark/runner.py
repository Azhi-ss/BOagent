from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.data_loader import DATA_LOADERS, build_task_context
from gp_llm_acq import GPLLM_ACQ


class BenchmarkRunner:
    """Execute a single-seed PVKBO benchmark run with GP+LLM ACQ."""

    def __init__(
        self,
        task_id: str,
        n_initial: int = 5,
        n_trials: int = 20,
        seed: int = 42,
        sm_mode: str = "discriminative",
        chat_engine: str = "deepseek-v4-flash",
        n_candidates: int = 10,
        n_templates: int = 2,
        n_gens: int = 5,
        alpha: float = 0.1,
        top_k: int = 20,
        output_dir: str | Path = "results",
        data_path: str | Path | None = None,
    ) -> None:
        self.task_id = task_id
        self.n_initial = n_initial
        self.n_trials = n_trials
        self.seed = seed
        self.sm_mode = sm_mode
        self.chat_engine = chat_engine
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.top_k = top_k
        self.output_dir = Path(output_dir)
        self.data_path = Path(data_path) if data_path else None

        if task_id not in DATA_LOADERS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {list(DATA_LOADERS)}"
            )
        if sm_mode not in ("discriminative", "generative"):
            raise ValueError(
                f"Unknown sm_mode: {sm_mode}. Use 'discriminative' or 'generative'"
            )

    def run(self) -> dict[str, Any]:
        """Execute the benchmark and return results dict."""
        # 1. Load data
        loader = DATA_LOADERS[self.task_id]
        data = loader(
            file_path=self.data_path, n_train=self.n_initial, seed=self.seed
        )
        task_context = build_task_context(self.task_id, data)

        # 2. Initialize PVKBO components
        # Add PVK-LLM to path if needed
        pvk_root = _resolve_pvk_root()
        if pvk_root and str(pvk_root) not in sys.path:
            sys.path.insert(0, str(pvk_root))

        try:
            from pvk_bo.pvk_bo import PVKBO
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import PVKBO from PVK-LLM. Ensure PVK-LLM is installed "
                f"and accessible at {pvk_root or 'the configured path'}. "
                f"Import error: {exc}"
            ) from exc

        top_pct = 0.25 if self.sm_mode == "generative" else None

        # Build init_f from train set
        def init_f(n_samples: int) -> list[dict[str, float]]:
            rng = np.random.RandomState(self.seed)
            indices = rng.choice(
                len(data["train_x"]), min(n_samples, len(data["train_x"])),
                replace=False,
            )
            configs: list[dict[str, float]] = []
            for idx in indices:
                config = {}
                for j, col in enumerate(data["feature_cols"]):
                    config[col] = float(data["train_x"][idx, j])
                configs.append(config)
            return configs

        # Build bbox_eval_f
        def bbox_eval_f(
            candidate_config: dict[str, Any],
        ) -> tuple[dict[str, float], dict[str, float]]:
            config = {
                col: float(candidate_config[col])
                for col in data["feature_cols"]
            }
            X = np.array(
                [[config[col] for col in data["feature_cols"]]]
            )
            # Search full dataset for exact or nearest match
            all_X = data["df"][data["feature_cols"]].values.astype(float)
            all_y = data["df"][data["target_col"]].values.astype(float)
            distances = np.sqrt(np.sum((all_X - X) ** 2, axis=1))
            nearest_idx = int(np.argmin(distances))
            score = float(all_y[nearest_idx])

            # Compute generalization score on test set
            test_X = data["test_x"]
            test_y = data["test_y"]
            test_distances = np.sqrt(np.sum((test_X - X) ** 2, axis=1))
            test_nearest_idx = int(np.argmin(test_distances))
            gen_score = float(test_y[test_nearest_idx])

            return config, {
                "score": score,
                "generalization_score": gen_score,
            }

        # Instantiate PVKBO with GPLLM_ACQ
        pvkbo = PVKBO(
            task_context=task_context,
            sm_mode=self.sm_mode,
            n_candidates=self.n_candidates,
            n_templates=self.n_templates,
            n_gens=self.n_gens,
            alpha=self.alpha,
            n_initial_samples=self.n_initial,
            n_trials=self.n_trials,
            init_f=init_f,
            bbox_eval_f=bbox_eval_f,
            chat_engine=self.chat_engine,
            top_pct=top_pct,
        )

        # Replace acq_func with GPLLM_ACQ
        pvkbo.acq_func = GPLLM_ACQ(
            task_context=task_context,
            n_candidates=self.n_candidates,
            n_templates=self.n_templates,
            lower_is_better=task_context["lower_is_better"],
            chat_engine=self.chat_engine,
            top_k=self.top_k,
            alpha=self.alpha,
        )

        # 3. Run optimization
        configs, fvals = pvkbo.optimize(test_metric="generalization_score")

        # 4. Find best
        best_idx = fvals["score"].idxmax()
        best_config = configs.iloc[best_idx].to_dict()
        best_score = float(fvals.iloc[best_idx]["score"])
        best_gen_score = float(fvals.iloc[best_idx]["generalization_score"])

        # 5. Build search history
        search_history = pd.concat([configs, fvals], axis=1)

        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "configs": configs,
            "fvals": fvals,
            "search_history": search_history,
            "best_config": best_config,
            "best_score": best_score,
            "best_generalization_score": best_gen_score,
            "llm_query_cost": pvkbo.llm_query_cost,
            "llm_query_time": pvkbo.llm_query_time,
        }

    def save_results(self, result: dict[str, Any]) -> None:
        """Persist benchmark results to output_dir."""
        save_dir = (
            self.output_dir
            / f"results_{self.sm_mode}"
            / self.task_id
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # CSV: search history
        result["search_history"].to_csv(
            save_dir / f"{self.seed}.csv", index=False
        )

        # JSON: search info (cost/time metadata)
        search_info = {
            "llm_query_cost_breakdown": result["llm_query_cost"],
            "llm_query_time_breakdown": result["llm_query_time"],
            "llm_query_cost": sum(result["llm_query_cost"]),
            "llm_query_time": sum(result["llm_query_time"]),
        }
        with open(save_dir / f"{self.seed}_search_info.json", "w") as f:
            json.dump(search_info, f, indent=2)

        # JSON: summary
        summary = {
            "task_id": result["task_id"],
            "seed": result["seed"],
            "best_config": result["best_config"],
            "best_score": result["best_score"],
            "best_generalization_score": result["best_generalization_score"],
            "n_trials": self.n_trials,
            "n_initial": self.n_initial,
            "sm_mode": self.sm_mode,
            "chat_engine": self.chat_engine,
            "convergence_curve": result["fvals"]["score"].tolist(),
            "generalization_curve": result["fvals"]["generalization_score"].tolist(),
        }
        with open(save_dir / f"{self.seed}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def run_multi_seed(
    task_id: str,
    seeds: list[int],
    output_dir: str | Path = "results",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run benchmark across multiple seeds sequentially.

    Returns:
        List of result dicts, one per seed.
    """
    results: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"Running {task_id} benchmark — seed {seed}")
        print(f"{'=' * 80}")
        runner = BenchmarkRunner(
            task_id=task_id,
            seed=seed,
            output_dir=output_dir,
            **kwargs,
        )
        result = runner.run()
        runner.save_results(result)
        results.append(result)
        print(
            f"Seed {seed}: best_score={result['best_score']:.4f}, "
            f"best_gen={result['best_generalization_score']:.4f}"
        )
    return results


def _resolve_pvk_root() -> Path | None:
    """Resolve PVK-LLM project root from env or default location."""
    env_root = os.environ.get("PVK_LLM_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    # Default: sibling to BOagent
    default = Path(__file__).resolve().parent.parent.parent.parent / "PVK-LLM"
    if default.exists():
        return default
    return None