from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from llm_client import DEFAULT_ENV_PATH, load_env_file


BOAGENT_ROOT = Path(__file__).resolve().parent
DEFAULT_PVK_REFERENCE_ROOT = Path(
    os.environ.get("PVK_LLM_ROOT", str(BOAGENT_ROOT.parent / "PVK-LLM"))
)
DEFAULT_REAL_DATA_ROOT = Path(
    os.environ.get(
        "PVK_DATA_ROOT",
        str(BOAGENT_ROOT.parent / "PVK-LLM" / "custom_perovskite_dataset"),
    )
)


class RealPvkBoUnavailableError(RuntimeError):
    """Raised when the real PVKBO path cannot run in the local environment."""


@dataclass(frozen=True)
class RealPvkBoSessionRequest:
    task_id: str = "band_alignment"
    n_initial: int = 5
    n_trials: int = 5
    seed: int = 42
    use_llm: bool = True
    language: str = "zh"
    sm_mode: str = "discriminative"
    n_candidates: int = 6
    n_templates: int = 6
    n_gens: int = 6
    alpha: float = -0.2
    chat_engine: str | None = None
    top_pct: float | None = None


REAL_TASK_SPECS: dict[str, dict[str, Any]] = {
    "band_alignment": {
        "task_id": "band_alignment",
        "name": "Band alignment optimization",
        "objective": "Maximize eta over PVK/HTL/ETL band-alignment features.",
        "workbook": "bandAlignment.xlsx",
        "model": "band_alignment",
        "feature_cols": ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"],
        "target_col": "eta",
    },
    "defects_doping": {
        "task_id": "defects_doping",
        "name": "Defects and doping optimization",
        "objective": "Maximize eta over defect-density and doping features.",
        "workbook": "defectsAndDoping.xlsx",
        "model": "defects_doping",
        "feature_cols": [
            "Nt_PVK/ETL",
            "Nt_HTL/PVK",
            "Na_PVK",
            "Nd_PVK",
            "Na_HTL",
            "Nd_HTL",
            "Na_ETL",
            "Nd_ETL",
        ],
        "target_col": "eta",
    },
}


def build_real_task_context(task_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    spec = _require_task_spec(task_id)
    feature_cols = list(spec["feature_cols"])
    target_col = str(spec["target_col"])
    missing_columns = [col for col in [*feature_cols, target_col] if col not in frame.columns]
    if missing_columns:
        raise RealPvkBoUnavailableError(
            f"PVK-LLM dataset for '{task_id}' is missing columns: {missing_columns}"
        )

    task_context = {
        "model": spec["model"],
        "task": "regression",
        "metric": "neg_mean_squared_error",
        "num_classes": 1,
        "n_classes": 1,
        "lower_is_better": False,
        "num_samples": int(len(frame)),
        "tot_feats": len(feature_cols),
        "cat_feats": 0,
        "num_feats": len(feature_cols),
        "feature_cols": feature_cols,
        "hyperparameter_constraints": {},
        "df": frame,
        "target_col": target_col,
    }
    for col in feature_cols:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        if values.empty:
            raise RealPvkBoUnavailableError(
                f"PVK-LLM dataset for '{task_id}' has no numeric values in '{col}'."
            )
        task_context["hyperparameter_constraints"][col] = [
            "float",
            "linear",
            [float(values.min()), float(values.max())],
        ]
    return task_context


class RealPvkBoRuntime:
    def __init__(
        self,
        pvk_reference_root: str | Path = DEFAULT_PVK_REFERENCE_ROOT,
        data_root: str | Path = DEFAULT_REAL_DATA_ROOT,
        env_path: str | Path = DEFAULT_ENV_PATH,
        pvk_bo_class: type | None = None,
    ) -> None:
        self.pvk_reference_root = Path(pvk_reference_root)
        self.data_root = _normalize_data_root(Path(data_root))
        self.env_path = Path(env_path)
        self._pvk_bo_class = pvk_bo_class
        self._sessions: dict[str, dict[str, Any]] = {}

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks = []
        for task_id, spec in REAL_TASK_SPECS.items():
            workbook_path = self._workbook_path(task_id)
            data_available = workbook_path.exists()
            tasks.append(
                {
                    "task_id": task_id,
                    "name": spec["name"],
                    "objective": spec["objective"],
                    "data_source": "PVK-LLM custom_perovskite_dataset",
                    "source_path": str(workbook_path),
                    "record_count": None,
                    "data_available": data_available,
                    "data_boundary": {
                        "notes": (
                            "Real PVK-LLM Excel dataset is available for live LLM-driven BO."
                            if data_available
                            else "Real PVK-LLM Excel dataset is missing; creating a real PVKBO session will fail fast."
                        ),
                        "dataset": spec["workbook"],
                        "source": str(workbook_path),
                        "warnings": []
                        if data_available
                        else [
                            f"Expected workbook not found: {workbook_path}",
                            "Demo CSV is not used as a substitute for real PVKBO tasks.",
                        ],
                        "constraints": [
                            "Requires OpenAI-compatible chat completions API.",
                            "Black-box evaluation uses exact or nearest-neighbor lookup from the workbook.",
                        ],
                    },
                }
            )
        return tasks

    def create_session(self, request: RealPvkBoSessionRequest) -> dict[str, Any]:
        spec = _require_task_spec(request.task_id)
        if request.sm_mode not in {"discriminative", "generative"}:
            raise ValueError(f"Unsupported sm_mode: {request.sm_mode}")
        if request.sm_mode == "generative" and request.top_pct is None:
            raise ValueError("top_pct is required for generative PVKBO surrogate mode.")

        workbook_path = self._workbook_path(request.task_id)
        if not workbook_path.exists():
            raise RealPvkBoUnavailableError(
                f"Missing real PVK-LLM workbook '{spec['workbook']}' at {self.data_root}. "
                f"Expected path: {workbook_path}"
            )

        chat_engine = configure_openai_compatible_env(
            env_path=self.env_path,
            requested_model=request.chat_engine,
            require_api_key=self._pvk_bo_class is None,
        )
        frame = pd.read_excel(workbook_path)
        task_context = build_real_task_context(request.task_id, frame)
        dataset = _PvkWorkbookBlackBox(task_context, seed=request.seed)
        pvk_bo_class = self._resolve_pvk_bo_class()
        pvkbo = pvk_bo_class(
            task_context=task_context,
            sm_mode=request.sm_mode,
            n_candidates=request.n_candidates,
            n_templates=request.n_templates,
            n_gens=request.n_gens,
            alpha=request.alpha,
            n_initial_samples=request.n_initial,
            n_trials=request.n_trials,
            init_f=dataset.generate_initialization,
            bbox_eval_f=dataset.evaluate_point,
            chat_engine=chat_engine,
            top_pct=request.top_pct,
        )

        init_cost, init_time = pvkbo._initialize()
        session_id = f"pvk_real_{uuid4().hex[:12]}"
        session = {
            "session_id": session_id,
            "status": "running" if request.n_trials > 0 else "completed",
            "task": {
                "task_id": request.task_id,
                "name": spec["name"],
                "context": _serializable_task_context(task_context),
                "data_source": f"PVK-LLM:{spec['workbook']}",
                "data_boundary": {
                    "notes": "Real PVK-LLM Excel data with live LLM_ACQ and LLM surrogate selection.",
                    "dataset": spec["workbook"],
                    "source": str(workbook_path),
                    "rows": int(len(frame)),
                    "warnings": [
                        "Scores are workbook black-box lookups, not new wet-lab experiments.",
                    ],
                    "constraints": [
                        f"sm_mode={request.sm_mode}",
                        f"chat_engine={chat_engine}",
                    ],
                },
                "source_path": str(workbook_path),
                "record_count": int(len(frame)),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_step": 0,
            "best_result": _best_result_from_frames(
                pvkbo.observed_configs, pvkbo.observed_fvals
            ),
            "observed_configs": _frame_to_records(pvkbo.observed_configs),
            "observed_fvals": _score_values(pvkbo.observed_fvals),
            "candidate_points": [],
            "tool_trace": [
                {
                    "step": "PVKBO.initialize",
                    "detail": (
                        f"Initialized {len(pvkbo.observed_configs)} observations from {spec['workbook']} "
                        f"in {init_time:.2f}s."
                    ),
                    "cost": init_cost,
                    "duration_s": init_time,
                }
            ],
            "guardrails": {
                "mode": "real_pvk_llm_bo",
                "llm_enabled": True,
                "language": request.language,
                "max_trials": max(0, int(request.n_trials)),
                "data_source": f"PVK-LLM:{spec['workbook']}",
                "data_boundary": "Real PVKBO runtime; workbook lookup only, not experimental validation.",
                "claim_boundary": "Live LLM BO over PVK-LLM workbook data; no claim of wet-lab validation.",
            },
            "_request": request,
            "_pvkbo": pvkbo,
        }
        self._sessions[session_id] = session
        return self._public_session(session)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._public_session(self._require_session(session_id))

    def run_step(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        request: RealPvkBoSessionRequest = session["_request"]
        pvkbo = session["_pvkbo"]
        if session["status"] == "completed":
            return self._public_session(session)

        started_at = time.time()
        candidate_points, acq_cost, acq_time = pvkbo.acq_func.get_candidate_points(
            pvkbo.observed_configs,
            pvkbo.observed_fvals[["score"]],
            alpha=request.alpha,
        )
        session["tool_trace"].append(
            {
                "step": "LLM_ACQ.get_candidate_points",
                "detail": f"Generated {len(candidate_points)} candidate points with live LLM acquisition.",
                "cost": acq_cost,
                "duration_s": acq_time,
            }
        )

        selected_candidate, sm_cost, sm_time = pvkbo.surrogate_model.select_query_point(
            pvkbo.observed_configs,
            pvkbo.observed_fvals[["score"]],
            candidate_points,
        )
        session["tool_trace"].append(
            {
                "step": "LLM_SURROGATE.select_query_point",
                "detail": "Selected the next query point with the live LLM surrogate.",
                "cost": sm_cost,
                "duration_s": sm_time,
            }
        )

        evaluated_config, evaluated_fval = pvkbo._evaluate_config(selected_candidate)
        score = float(evaluated_fval["score"].iloc[0])
        session["tool_trace"].append(
            {
                "step": "black_box.evaluate_candidate",
                "detail": f"Evaluated selected candidate with workbook lookup -> eta={score:.4f}.",
                "duration_s": time.time() - started_at,
            }
        )

        pvkbo._update_observations(evaluated_config, evaluated_fval)
        session["current_step"] += 1
        session["observed_configs"] = _frame_to_records(pvkbo.observed_configs)
        session["observed_fvals"] = _score_values(pvkbo.observed_fvals)
        session["candidate_points"] = _candidate_records(
            candidate_points, session["current_step"]
        )
        session["best_result"] = _best_result_from_frames(
            pvkbo.observed_configs, pvkbo.observed_fvals
        )
        session["tool_trace"].append(
            {
                "step": "PVKBO.update_observations",
                "detail": (
                    f"Stored {len(pvkbo.observed_configs)} observations; "
                    f"best eta is {session['best_result']['score']:.4f}."
                ),
            }
        )
        if session["current_step"] >= request.n_trials:
            session["status"] = "completed"
        return self._public_session(session)

    def get_artifacts(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        public_session = self._public_session(session)
        summary = {
            "session_id": session["session_id"],
            "status": session["status"],
            "task_id": session["task"]["task_id"],
            "data_source": session["task"]["data_source"],
            "data_boundary": session["task"]["data_boundary"],
            "current_step": session["current_step"],
            "best_score": session["best_result"]["score"],
            "best_result": session["best_result"],
            "observed_count": len(session["observed_configs"]),
            "observed_fvals": list(session["observed_fvals"]),
            "candidate_count": len(session["candidate_points"]),
        }
        return {
            "session_id": session["session_id"],
            "data_source": session["task"]["data_source"],
            "data_boundary": session["task"]["data_boundary"],
            "summary": summary,
            "observed_configs": public_session["observed_configs"],
            "observed_fvals": public_session["observed_fvals"],
            "candidate_points": public_session["candidate_points"],
            "tool_trace": public_session["tool_trace"],
            "guardrails": public_session["guardrails"],
        }

    def _resolve_pvk_bo_class(self) -> type:
        if self._pvk_bo_class is not None:
            return self._pvk_bo_class
        if not self.pvk_reference_root.exists():
            raise RealPvkBoUnavailableError(
                f"PVK-LLM reference repo not found: {self.pvk_reference_root}"
            )
        reference_root = str(self.pvk_reference_root)
        if reference_root not in sys.path:
            sys.path.insert(0, reference_root)
        _install_langchain_prompt_compat()
        _install_pandas_series_int_position_compat()
        _install_openai_single_completion_compat()
        try:
            from pvk_bo.pvk_bo import PVKBO
        except (ImportError, ModuleNotFoundError) as exc:
            raise RealPvkBoUnavailableError(
                "Cannot import PVKBO from the PVK-LLM reference repo. "
                f"Install compatible dependencies from PVK-LLM README first. Import error: {exc}"
            ) from exc
        return PVKBO

    def _workbook_path(self, task_id: str) -> Path:
        spec = _require_task_spec(task_id)
        return self.data_root / str(spec["workbook"])

    def _require_session(self, session_id: str) -> dict[str, Any]:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown real PVKBO session: {session_id}") from exc

    def _public_session(self, session: dict[str, Any]) -> dict[str, Any]:
        return deepcopy({key: value for key, value in session.items() if not key.startswith("_")})


class _PvkWorkbookBlackBox:
    def __init__(self, task_context: dict[str, Any], seed: int) -> None:
        self.task_context = task_context
        self.seed = seed
        self.feature_cols = list(task_context["feature_cols"])
        self.target_col = str(task_context["target_col"])
        self.frame: pd.DataFrame = task_context["df"]

    def generate_initialization(self, n_samples: int) -> list[dict[str, float]]:
        if n_samples <= 0:
            return []
        sample_size = min(n_samples, len(self.frame))
        sampled = self.frame.sample(n=sample_size, random_state=self.seed, replace=False)
        return [
            {col: float(row[col]) for col in self.feature_cols}
            for row in sampled.to_dict("records")
        ]

    def evaluate_point(self, candidate_config: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        config = {col: float(candidate_config[col]) for col in self.feature_cols}
        matrix = self.frame[self.feature_cols].astype(float).to_numpy()
        vector = np.array([config[col] for col in self.feature_cols], dtype=float)
        distances = np.sqrt(np.sum((matrix - vector) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        score = float(self.frame.iloc[nearest_index][self.target_col])
        return config, {
            "score": score,
            "generalization_score": score,
            "lookup_mode": "exact_match"
            if float(distances[nearest_index]) <= 1e-8
            else "nearest_neighbor",
        }


def configure_openai_compatible_env(
    *,
    env_path: str | Path = DEFAULT_ENV_PATH,
    requested_model: str | None = None,
    require_api_key: bool = True,
) -> str:
    load_env_file(Path(env_path))
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("DEEPSEEK_BASE_URL")
    model = (
        requested_model
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("DEEPSEEK_FLASH_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or "deepseek-v4-flash"
    )
    if require_api_key and (
        not api_key or api_key.strip() == "your_deepseek_api_key_here"
    ):
        raise RealPvkBoUnavailableError(
            "OPENAI_API_KEY or DEEPSEEK_API_KEY is required for real PVKBO LLM calls."
        )
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if api_base:
        os.environ.setdefault("OPENAI_API_BASE", api_base.rstrip("/"))
    return model


def _install_langchain_prompt_compat() -> None:
    try:
        import langchain
        from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
    except ImportError:
        return
    if not hasattr(langchain, "FewShotPromptTemplate"):
        setattr(langchain, "FewShotPromptTemplate", FewShotPromptTemplate)
    if not hasattr(langchain, "PromptTemplate"):
        setattr(langchain, "PromptTemplate", PromptTemplate)


def _install_pandas_series_int_position_compat() -> None:
    if getattr(pd.Series, "_boagent_legacy_int_position_compat", False):
        return

    original_getitem = pd.Series.__getitem__

    def getitem_with_legacy_int_position(self: pd.Series, key: Any) -> Any:
        try:
            return original_getitem(self, key)
        except KeyError:
            if isinstance(key, int) and key not in self.index:
                return self.iloc[key]
            raise

    pd.Series.__getitem__ = getitem_with_legacy_int_position
    pd.Series._boagent_legacy_int_position_compat = True


def _install_openai_single_completion_compat() -> None:
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        return
    if getattr(AsyncCompletions, "_boagent_force_single_completion", False):
        return

    original_create = AsyncCompletions.create

    async def create_with_single_completion(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "").lower()
        if model.startswith("deepseek"):
            if kwargs.get("n", 1) != 1:
                kwargs["n"] = 1
            if int(kwargs.get("max_tokens") or 0) < 512:
                kwargs["max_tokens"] = 512
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
        return await original_create(self, *args, **kwargs)

    AsyncCompletions.create = create_with_single_completion
    AsyncCompletions._boagent_force_single_completion = True


def _normalize_data_root(path: Path) -> Path:
    if path.name == "custom_perovskite_dataset":
        return path
    return path / "custom_perovskite_dataset"


def _require_task_spec(task_id: str) -> dict[str, Any]:
    try:
        return REAL_TASK_SPECS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported real PVKBO task_id: {task_id}") from exc


def _serializable_task_context(task_context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in task_context.items()
        if key != "df"
    }


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for raw_record in frame.to_dict("records"):
        records.append({str(key): _json_value(value) for key, value in raw_record.items()})
    return records


def _score_values(frame: pd.DataFrame) -> list[float]:
    if frame.empty or "score" not in frame.columns:
        return []
    return [float(value) for value in frame["score"].tolist()]


def _candidate_records(frame: pd.DataFrame, step: int) -> list[dict[str, Any]]:
    records = _frame_to_records(frame)
    for index, record in enumerate(records, start=1):
        record.setdefault("candidate_id", f"PVK-CAND-{step:02d}-{index:02d}")
        record.setdefault("acquisition_function", "PVK-LLM LLM_ACQ")
    return records


def _best_result_from_frames(configs: pd.DataFrame, fvals: pd.DataFrame) -> dict[str, Any]:
    scores = _score_values(fvals)
    if not scores:
        return {"score": 0.0, "metric": "eta", "config": {}, "result": {}}
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    config = _frame_to_records(configs.iloc[[best_index]])[0]
    result = _frame_to_records(fvals.iloc[[best_index]])[0]
    return {
        "score": float(scores[best_index]),
        "metric": "eta",
        "config": config,
        "result": result,
    }


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
