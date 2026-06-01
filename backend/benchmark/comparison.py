from __future__ import annotations

import os
import sys
import concurrent.futures
import queue
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from benchmark.bo_step import BOStepEngine
from benchmark.data_loader import DATA_LOADERS


class ComparisonRunner:
    """Drive traditional BO and LLMBO across multiple seeds, yielding aggregate
    convergence events (per-iteration mean ± std) for SSE streaming.

    Both methods share the same loaded data, n_initial, and n_trials for each
    seed so their convergence curves are directly comparable. After each seed
    completes, an ``aggregate`` event carries the running mean/std across all
    seeds finished so far, so the variance band tightens as seeds accumulate.
    """

    def __init__(
        self,
        task_id: str,
        n_initial: int,
        n_trials: int,
        seeds: list[int],
        traditional: dict[str, Any],
        llmbo: dict[str, Any],
        data_path: str | Path | None = None,
    ) -> None:
        if task_id not in DATA_LOADERS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {list(DATA_LOADERS)}"
            )
        if not seeds:
            raise ValueError("seeds must be a non-empty list")
        self.task_id = task_id
        self.n_initial = n_initial
        self.n_trials = n_trials
        self.seeds = seeds
        self.traditional_cfg = traditional
        self.llmbo_cfg = llmbo
        self.data_path = Path(data_path) if data_path else None

    def _build_llm_acq(self, data: dict[str, Any]) -> Any | None:
        """Construct the GP+LLM acquisition for the LLMBO engine."""
        try:
            from benchmark.data_loader import build_task_context
            from optimization.optimizer import BayesianOptimizer
            from optimization.space import DiscreteSearchSpace
            from optimization.knowledge import KnowledgeEngine

            chat_engine = (
                self.llmbo_cfg.get("chat_engine")
                or os.environ.get("DEEPSEEK_FLASH_MODEL")
                or os.environ.get("DEEPSEEK_MODEL")
                or "deepseek-v4-flash"
            )
            knowledge = KnowledgeEngine(chat_engine=chat_engine)
            space = DiscreteSearchSpace(data["df"], data["feature_cols"])
            optimizer = BayesianOptimizer(
                space=space,
                target_name=data["target_col"],
                knowledge_engine=knowledge,
                n_restarts_optimizer=10
            )
            # Tag the optimizer with properties expected by BOStepEngine's bridging logic
            optimizer.chat_engine = chat_engine
            optimizer.n_candidates = int(self.llmbo_cfg.get("n_candidates", 5))
            optimizer.alpha = float(self.llmbo_cfg.get("alpha", 0.1))
            
            return optimizer
        except Exception:
            # LLM acquisition is best-effort; LLMBO falls back to GP argmax.
            return None

    def _run_one_seed(
        self, seed: int, seed_index: int, total_seeds: int
    ) -> Iterator[dict[str, Any]]:
        """Run both engines for one seed, yielding progress events and finally
        a ``_seed_done`` event carrying the per-iteration best/gen trajectories.
        """
        yield {
            "type": "seed_start",
            "seed": seed,
            "seed_index": seed_index,
            "total_seeds": total_seeds,
        }
        
        loader = DATA_LOADERS[self.task_id]
        data = loader(
            file_path=self.data_path, n_train=self.n_initial, seed=seed
        )

        traditional = BOStepEngine(
            method="traditional",
            data=data,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seed=seed,
            acquisition=self.traditional_cfg.get("acquisition", "ei"),
            xi=float(self.traditional_cfg.get("xi", 0.01)),
            kappa=float(self.traditional_cfg.get("kappa", 2.576)),
        )
        
        llmbo = BOStepEngine(
            method="llmbo",
            data=data,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seed=seed,
            acquisition=self.llmbo_cfg.get("acquisition", "ucb"),
            xi=float(self.llmbo_cfg.get("xi", 0.01)),
            kappa=float(self.llmbo_cfg.get("kappa", 2.576)),
            chat_engine=self.llmbo_cfg.get("chat_engine") or os.environ.get("DEEPSEEK_FLASH_MODEL") or "deepseek-v4-flash",
            n_candidates=int(self.llmbo_cfg.get("n_candidates", 5)),
        )

        # Trajectories indexed by iteration (0 = post-init baseline).
        trad_best: list[float] = []
        trad_gen: list[float] = []
        llm_best: list[float] = []
        llm_gen: list[float] = []

        snap_t = traditional.snapshot(candidate_score=None)
        snap_l = llmbo.snapshot(candidate_score=None)
        trad_best.append(snap_t["best_score"])
        trad_gen.append(snap_t["generalization_score"])
        llm_best.append(snap_l["best_score"])
        llm_gen.append(snap_l["generalization_score"])

        for _ in range(self.n_trials):
            if not traditional.completed:
                yield {
                    "type": "step_start",
                    "method": "traditional",
                    "seed_index": seed_index,
                    "total_seeds": total_seeds,
                    "iteration": traditional.iteration + 1,
                }
                s = traditional.step()
                trad_best.append(s["best_score"])
                trad_gen.append(s["generalization_score"])
            if not llmbo.completed:
                yield {
                    "type": "step_start",
                    "method": "llmbo",
                    "seed_index": seed_index,
                    "total_seeds": total_seeds,
                    "iteration": llmbo.iteration + 1,
                }
                s = llmbo.step()
                llm_best.append(s["best_score"])
                llm_gen.append(s["generalization_score"])
            # Emit a per-iteration snapshot so events() can push incremental aggregates.
            # This makes the chart update every ~6s instead of waiting ~60s for a full seed.
            yield {
                "type": "_iter_snapshot",
                "seed_index": seed_index,
                "trad_best": list(trad_best),
                "trad_gen": list(trad_gen),
                "llm_best": list(llm_best),
                "llm_gen": list(llm_gen),
            }
            if traditional.completed and llmbo.completed:
                break

        yield {
            "type": "_seed_done",
            "seed_index": seed_index,
            "trad_best": trad_best,
            "trad_gen": trad_gen,
            "llm_best": llm_best,
            "llm_gen": llm_gen,
        }

    @staticmethod
    def _pad(traj: list[float], length: int) -> list[float]:
        """Right-pad a trajectory with its last value to a fixed length."""
        if not traj:
            return [0.0] * length
        if len(traj) >= length:
            return traj[:length]
        return traj + [traj[-1]] * (length - len(traj))

    def _aggregate(
        self, trajectories: list[dict[str, list[float]]]
    ) -> list[dict[str, float]]:
        """Compute per-iteration mean/std across completed and active seeds.
        
        Active trajectories are padded with their latest values to prevent
        downward jumps in average scores as slower seeds catch up.
        """
        length = self.n_trials + 1
        keys = ["trad_best", "trad_gen", "llm_best", "llm_gen"]
        points: list[dict[str, float]] = []
        
        for i in range(length):
            point: dict[str, float] = {"iteration": i}
            for k in keys:
                vals = []
                for t in trajectories:
                    traj_list = t[k]
                    if len(traj_list) > i:
                        vals.append(traj_list[i])
                    elif traj_list:
                        vals.append(traj_list[-1])
                
                if vals:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
                else:
                    mean_val, std_val = 0.0, 0.0
                
                point[f"{k}_mean"] = mean_val
                point[f"{k}_std"] = std_val
            points.append(point)
        return points

    def events(self) -> Iterator[dict[str, Any]]:
        """Yield comparison events: meta, per-seed progress + aggregate, done."""
        total = len(self.seeds)
        yield {
            "type": "meta",
            "task_id": self.task_id,
            "n_trials": self.n_trials,
            "n_initial": self.n_initial,
            "seeds": self.seeds,
            "total_seeds": total,
        }

        # completed_trajectories: full seed results; partial_trajectories: per-seed
        # latest snapshot used for incremental aggregation between seed completions.
        trajectories: list[dict[str, list[float]]] = []
        partial_trajectories: dict[int, dict[str, list[float]]] = {}
        event_queue = queue.Queue()

        def run_seed_task(seed: int, idx: int):
            try:
                for ev in self._run_one_seed(seed, idx, total):
                    event_queue.put(ev)
            finally:
                event_queue.put({"type": "_done_signal"})

        # Start all seeds in parallel
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=total)
        for idx, seed in enumerate(self.seeds, start=1):
            executor.submit(run_seed_task, seed, idx)

        completed_seeds_count = 0
        while completed_seeds_count < total:
            try:
                ev = event_queue.get(timeout=1.0)
                if ev["type"] == "_done_signal":
                    completed_seeds_count += 1
                elif ev["type"] == "_seed_done":
                    trajectories.append(
                        {
                            "trad_best": ev["trad_best"],
                            "trad_gen": ev["trad_gen"],
                            "llm_best": ev["llm_best"],
                            "llm_gen": ev["llm_gen"],
                        }
                    )
                    # Remove this seed's partial entry now that it's finalised.
                    partial_trajectories.pop(ev.get("seed_index"), None)
                    yield {
                        "type": "aggregate",
                        "completed_seeds": len(trajectories),
                        "total_seeds": total,
                        "points": self._aggregate(trajectories),
                    }
                elif ev["type"] == "_iter_snapshot":
                    # Update this seed's live partial trajectory and re-aggregate
                    # so the frontend chart updates after every iteration.
                    partial_trajectories[ev["seed_index"]] = {
                        "trad_best": ev["trad_best"],
                        "trad_gen": ev["trad_gen"],
                        "llm_best": ev["llm_best"],
                        "llm_gen": ev["llm_gen"],
                    }
                    all_traj = list(trajectories) + list(partial_trajectories.values())
                    if all_traj:
                        yield {
                            "type": "aggregate",
                            "completed_seeds": len(trajectories),
                            "total_seeds": total,
                            "points": self._aggregate(all_traj),
                        }
                else:
                    yield ev
            except queue.Empty:
                continue

        executor.shutdown()

        # Final summary: mean ± std of final best_score across seeds.
        length = self.n_trials + 1
        trad_finals = np.array([self._pad(t["trad_best"], length)[-1] for t in trajectories])
        llm_finals = np.array([self._pad(t["llm_best"], length)[-1] for t in trajectories])
        yield {
            "type": "done",
            "total_seeds": total,
            "summary": {
                "traditional": {
                    "best_mean": float(np.mean(trad_finals)),
                    "best_std": float(np.std(trad_finals)),
                },
                "llmbo": {
                    "best_mean": float(np.mean(llm_finals)),
                    "best_std": float(np.std(llm_finals)),
                },
            },
        }


