from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pvk_tools import (
    DEFAULT_DEMO_CSV_PATH,
    DEFAULT_PVK_PROJECT_ROOT,
    build_task_context,
    evaluate_candidate,
    generate_candidate_points,
    initialize_observations,
    load_pvk_dataset_or_demo,
    select_query_point,
    summarize_session,
)
from pvk_llm_bo_runtime import (
    REAL_TASK_SPECS,
    RealPvkBoRuntime,
    RealPvkBoSessionRequest,
)


@dataclass(frozen=True)
class OptimizationSessionRequest:
    task_id: str = "passivation_demo"
    n_initial: int = 5
    n_trials: int = 5
    seed: int = 42
    use_llm: bool = False
    language: str = "zh"


class OptimizationSessionRuntime:
    def __init__(
        self,
        pvk_project_root: str | Path = DEFAULT_PVK_PROJECT_ROOT,
        demo_csv_path: str | Path = DEFAULT_DEMO_CSV_PATH,
        real_pvk_runtime: RealPvkBoRuntime | None = None,
    ) -> None:
        self.pvk_project_root = Path(pvk_project_root)
        self.demo_csv_path = Path(demo_csv_path)
        self.real_pvk_runtime = real_pvk_runtime or RealPvkBoRuntime()
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, request: OptimizationSessionRequest) -> dict[str, Any]:
        if request.task_id in REAL_TASK_SPECS:
            return self.real_pvk_runtime.create_session(
                RealPvkBoSessionRequest(
                    task_id=request.task_id,
                    n_initial=request.n_initial,
                    n_trials=request.n_trials,
                    seed=request.seed,
                    use_llm=True,
                    language=request.language,
                )
            )
        if request.task_id != "passivation_demo":
            raise ValueError(f"Unsupported task_id: {request.task_id}")

        dataset = load_pvk_dataset_or_demo(self.pvk_project_root, self.demo_csv_path)
        task_context = build_task_context(dataset, language=request.language)
        observed_configs, observed_fvals = initialize_observations(
            dataset, n_initial=request.n_initial, seed=request.seed
        )
        session_id = f"pvk_session_{uuid4().hex[:12]}"
        best_result = _best_result(observed_configs, observed_fvals)
        session = {
            "session_id": session_id,
            "status": "running" if request.n_trials > 0 else "completed",
            "task": {
                "task_id": request.task_id,
                "name": "Passivation formulation demo",
                "context": task_context,
                "data_source": dataset["data_source"],
                "data_boundary": dataset["data_boundary"],
                "source_path": dataset.get("source_path"),
                "record_count": len(dataset["records"]),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_step": 0,
            "best_result": best_result,
            "observed_configs": observed_configs,
            "observed_fvals": observed_fvals,
            "candidate_points": [],
            "tool_trace": [
                {
                    "step": "PVKBO.initialize",
                    "detail": f"Initialized {len(observed_configs)} observations from {dataset['data_source']}.",
                }
            ],
            "guardrails": {
                "mode": "demo",
                "llm_enabled": bool(request.use_llm),
                "language": request.language,
                "max_trials": max(0, int(request.n_trials)),
                "data_source": dataset["data_source"],
                "data_boundary": dataset["data_boundary"],
                "claim_boundary": "Deterministic/mock acquisition for product demo only; not experimental validation.",
            },
            "_request": request,
            "_dataset": dataset,
        }
        self._sessions[session_id] = session
        return self._public_session(session)

    def get_session(self, session_id: str) -> dict[str, Any]:
        if session_id.startswith("pvk_real_"):
            return self.real_pvk_runtime.get_session(session_id)
        return self._public_session(self._require_session(session_id))

    def run_step(self, session_id: str) -> dict[str, Any]:
        if session_id.startswith("pvk_real_"):
            return self.real_pvk_runtime.run_step(session_id)
        session = self._require_session(session_id)
        request: OptimizationSessionRequest = session["_request"]
        if session["status"] == "completed":
            return self._public_session(session)

        dataset = session["_dataset"]
        task_context = session["task"]["context"]
        candidate_points = generate_candidate_points(
            task_context,
            dataset,
            session["observed_configs"],
            session["observed_fvals"],
            seed=request.seed,
            current_step=session["current_step"],
        )
        session["tool_trace"].append(
            {
                "step": "LLM_ACQ.generate_candidate_points",
                "detail": f"Generated {len(candidate_points)} deterministic candidate points.",
            }
        )

        selected = select_query_point(
            candidate_points,
            session["observed_configs"],
            session["observed_fvals"],
        )
        session["tool_trace"].append(
            {
                "step": "LLM_SURROGATE.select_query_point",
                "detail": f"Selected {selected['candidate_id']} with mock acquisition score {selected['mock_acquisition_score']}.",
            }
        )

        evaluation = evaluate_candidate(selected, dataset)
        session["tool_trace"].append(
            {
                "step": "black_box.evaluate_candidate",
                "detail": f"Evaluated {selected['experiment_id']} -> {evaluation['metric']}={evaluation['score']}.",
            }
        )

        observed_config = {
            key: value
            for key, value in selected.items()
            if key
            not in {
                "candidate_id",
                "mock_acquisition_score",
                "acquisition_function",
            }
        }
        session["observed_configs"].append(observed_config)
        session["observed_fvals"].append(evaluation["score"])
        session["candidate_points"] = candidate_points
        session["current_step"] += 1
        session["best_result"] = _best_result(
            session["observed_configs"], session["observed_fvals"], evaluation
        )
        session["tool_trace"].append(
            {
                "step": "PVKBO.update_observations",
                "detail": f"Stored {len(session['observed_configs'])} observations; best score is {session['best_result']['score']}.",
            }
        )

        if session["current_step"] >= request.n_trials:
            session["status"] = "completed"
        return self._public_session(session)

    def get_artifacts(self, session_id: str) -> dict[str, Any]:
        if session_id.startswith("pvk_real_"):
            return self.real_pvk_runtime.get_artifacts(session_id)
        session = self._require_session(session_id)
        summary = summarize_session(self._public_session(session))
        return {
            "session_id": session["session_id"],
            "data_source": session["task"]["data_source"],
            "data_boundary": session["task"]["data_boundary"],
            "summary": summary,
            "observed_configs": deepcopy(session["observed_configs"]),
            "observed_fvals": list(session["observed_fvals"]),
            "candidate_points": deepcopy(session["candidate_points"]),
            "tool_trace": deepcopy(session["tool_trace"]),
            "guardrails": deepcopy(session["guardrails"]),
        }

    def _require_session(self, session_id: str) -> dict[str, Any]:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown optimization session: {session_id}") from exc

    def _public_session(self, session: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value for key, value in session.items() if not key.startswith("_")
        }
        return deepcopy(public)


def _best_result(
    observed_configs: list[dict[str, Any]],
    observed_fvals: list[float],
    latest_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not observed_configs or not observed_fvals:
        return {"score": 0.0, "config": {}, "result": {}}

    best_index = max(range(len(observed_fvals)), key=lambda index: observed_fvals[index])
    result = {
        "score": observed_fvals[best_index],
        "metric": "PCE_percent",
        "config": deepcopy(observed_configs[best_index]),
    }
    if (
        latest_evaluation
        and best_index == len(observed_fvals) - 1
        and observed_fvals[best_index] == latest_evaluation["score"]
    ):
        result["result"] = deepcopy(latest_evaluation["result"])
    else:
        result["result"] = {"evaluation_mode": "initial_observation"}
    return result
