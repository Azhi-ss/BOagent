from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from llm_client import DeepSeekClient, generate_llm_notes
from pvk_demo import (
    DataSummary,
    Recommendation,
    build_agent_pipeline,
    build_data_summary,
    generate_recommendations,
    load_experiment_data,
    simulate_feedback,
)


@dataclass(frozen=True)
class AgentRunRequest:
    task_text: str
    recommendation_count: int = 5
    language: str = "en"
    use_llm: bool = False


@dataclass(frozen=True)
class StageResult:
    stage_name: str
    status: str
    summary: str
    artifact_name: str


@dataclass(frozen=True)
class AgentRun:
    run_id: str
    status: str
    created_at: str
    task_text: str
    language: str
    stage_results: list[StageResult]
    data_summary: DataSummary
    recommendations: list[Recommendation]
    feedback: dict[str, Any]
    llm_notes: dict[str, Any] | None
    guardrails: dict[str, Any]
    artifacts: dict[str, Any]


class AgentRunStore:
    def __init__(self, llm_client: DeepSeekClient | None = None) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._llm_client = llm_client

    def create_run(self, request: AgentRunRequest) -> AgentRun:
        data = load_experiment_data()
        summary = build_data_summary(data)
        recommendations = generate_recommendations(
            data, summary, n=request.recommendation_count
        )
        feedback = simulate_feedback(summary, recommendations)
        llm_notes = None
        if request.use_llm:
            llm_result = generate_llm_notes(
                task_text=request.task_text,
                summary=summary,
                recommendations=recommendations,
                language=request.language,
                client=self._llm_client,
            )
            llm_notes = {
                "status": llm_result.status,
                "provider": llm_result.provider,
                "model": llm_result.model,
                "content": llm_result.content,
                "usage": llm_result.usage,
                "error": llm_result.error,
            }
        run_id = f"run_{uuid4().hex[:12]}"
        stages = [
            StageResult(
                stage_name=stage.name,
                status="success",
                summary=stage.output_summary,
                artifact_name=_artifact_name(stage.name),
            )
            for stage in build_agent_pipeline()
        ]
        artifacts = {
            "data-summary": _summary_dict(summary),
            "recommendations": [_recommendation_dict(item) for item in recommendations],
            "simulated-feedback": feedback,
            "agent-stages": [asdict(stage) for stage in stages],
        }
        if llm_notes is not None:
            artifacts["llm-notes"] = llm_notes
        run = AgentRun(
            run_id=run_id,
            status="completed",
            created_at=datetime.now(UTC).isoformat(),
            task_text=request.task_text,
            language=request.language,
            stage_results=stages,
            data_summary=summary,
            recommendations=recommendations,
            feedback=feedback,
            llm_notes=llm_notes,
            guardrails={
                "mode": "demo",
                "llm_enabled": request.use_llm,
                "bo_claim": "rules/mock-acquisition only",
                "science_claim": "hypothesis, not experimentally validated",
                "pipdi_boundary": "demo-only exploration until real samples exist",
            },
            artifacts=artifacts,
        )
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def get_artifact(self, run_id: str, artifact_name: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        artifact = run.artifacts.get(artifact_name)
        if artifact is None:
            return None
        return {
            "run_id": run_id,
            "artifact_name": artifact_name,
            "content": artifact,
        }


def serialize_run(run: AgentRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "created_at": run.created_at,
        "task_text": run.task_text,
        "language": run.language,
        "stage_results": [asdict(stage) for stage in run.stage_results],
        "data_summary": _summary_dict(run.data_summary),
        "recommendations": [_recommendation_dict(item) for item in run.recommendations],
        "feedback": run.feedback,
        "llm_notes": run.llm_notes,
        "guardrails": run.guardrails,
        "artifacts": list(run.artifacts.keys()),
    }


def _summary_dict(summary: DataSummary) -> dict[str, Any]:
    return asdict(summary)


def _recommendation_dict(recommendation: Recommendation) -> dict[str, Any]:
    return asdict(recommendation)


def _artifact_name(stage_name: str) -> str:
    return stage_name.lower().replace(" ", "-")
