from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

from benchmark.bo_step import BOStepEngine
from benchmark.data_loader import DATA_LOADERS


class ComparisonRunner:
    """Drive traditional BO and LLMBO over a shared train/test split,
    yielding per-iteration events for SSE streaming.

    Both engines share the same loaded data, seed, n_initial, and n_trials so
    their convergence curves are directly comparable. Each ``events()`` tick
    advances both engines by one step and yields two event dicts.
    """

    def __init__(
        self,
        task_id: str,
        n_initial: int,
        n_trials: int,
        seed: int,
        traditional: dict[str, Any],
        llmbo: dict[str, Any],
        data_path: str | Path | None = None,
    ) -> None:
        if task_id not in DATA_LOADERS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {list(DATA_LOADERS)}"
            )
        self.task_id = task_id
        self.n_initial = n_initial
        self.n_trials = n_trials
        self.seed = seed
        self.traditional_cfg = traditional
        self.llmbo_cfg = llmbo
        self.data_path = Path(data_path) if data_path else None

    def _build_llm_acq(self, data: dict[str, Any]) -> Any | None:
        """Construct the GP+LLM acquisition for the LLMBO engine."""
        try:
            from benchmark.data_loader import build_task_context
            from gp_llm_acq import GPLLM_ACQ

            task_context = build_task_context(self.task_id, data)
            chat_engine = (
                self.llmbo_cfg.get("chat_engine")
                or os.environ.get("DEEPSEEK_FLASH_MODEL")
                or os.environ.get("DEEPSEEK_MODEL")
                or "deepseek-v4-flash"
            )
            return GPLLM_ACQ(
                task_context=task_context,
                n_candidates=int(self.llmbo_cfg.get("n_candidates", 5)),
                n_templates=int(self.llmbo_cfg.get("n_templates", 2)),
                lower_is_better=False,
                chat_engine=chat_engine,
                top_k=int(self.llmbo_cfg.get("top_k", 20)),
                alpha=float(self.llmbo_cfg.get("alpha", 0.1)),
            )
        except Exception:
            # LLM acquisition is best-effort; LLMBO falls back to GP argmax.
            return None

    def events(self) -> Iterator[dict[str, Any]]:
        """Yield comparison events: meta, per-iteration snapshots, and done."""
        _ensure_pvk_path()

        loader = DATA_LOADERS[self.task_id]
        data = loader(
            file_path=self.data_path, n_train=self.n_initial, seed=self.seed
        )

        traditional = BOStepEngine(
            method="traditional",
            data=data,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seed=self.seed,
            acquisition=self.traditional_cfg.get("acquisition", "ei"),
            xi=float(self.traditional_cfg.get("xi", 0.01)),
            kappa=float(self.traditional_cfg.get("kappa", 2.576)),
        )
        llmbo = BOStepEngine(
            method="llmbo",
            data=data,
            n_initial=self.n_initial,
            n_trials=self.n_trials,
            seed=self.seed,
            acquisition=self.llmbo_cfg.get("acquisition", "ei"),
            xi=float(self.llmbo_cfg.get("xi", 0.01)),
            kappa=float(self.llmbo_cfg.get("kappa", 2.576)),
            llm_acq=self._build_llm_acq(data),
        )

        yield {
            "type": "meta",
            "task_id": self.task_id,
            "n_trials": self.n_trials,
            "n_initial": self.n_initial,
            "seed": self.seed,
            "feature_cols": data["feature_cols"],
            "target_col": data["target_col"],
        }

        # Emit iteration-0 baselines (post-initialization best)
        yield {"type": "iteration", **traditional.snapshot(candidate_score=None)}
        yield {"type": "iteration", **llmbo.snapshot(candidate_score=None)}

        for _ in range(self.n_trials):
            if not traditional.completed:
                yield {
                    "type": "step_start",
                    "method": "traditional",
                    "iteration": traditional.iteration + 1,
                }
                yield {"type": "iteration", **traditional.step()}
            if not llmbo.completed:
                yield {
                    "type": "step_start",
                    "method": "llmbo",
                    "iteration": llmbo.iteration + 1,
                }
                yield {"type": "iteration", **llmbo.step()}
            if traditional.completed and llmbo.completed:
                break

        yield {
            "type": "done",
            "traditional": traditional.snapshot(candidate_score=None),
            "llmbo": llmbo.snapshot(candidate_score=None),
        }


def _ensure_pvk_path() -> None:
    env_root = os.environ.get("PVK_LLM_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(
        Path(__file__).resolve().parent.parent.parent.parent / "PVK-LLM"
    )
    for root in candidates:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
            break
