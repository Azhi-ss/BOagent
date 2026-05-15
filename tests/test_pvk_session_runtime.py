import json
from pathlib import Path

from pvk_session_runtime import OptimizationSessionRequest, OptimizationSessionRuntime
from pvk_tools import list_tasks, load_pvk_dataset_or_demo


DATA_PATH = Path(__file__).resolve().parents[1] / "demo_optimization_table.csv"


def test_list_tasks_exposes_passivation_demo_with_data_boundary():
    tasks = list_tasks(pvk_project_root=Path("/tmp/pvk-llm-without-dataset"))

    passivation_task = next(task for task in tasks if task["task_id"] == "passivation_demo")
    assert passivation_task["data_source"] == "demo_optimization_table.csv"
    assert "fallback" in passivation_task["data_boundary"].lower()


def test_missing_original_dataset_falls_back_to_demo_csv():
    dataset = load_pvk_dataset_or_demo(
        pvk_project_root=Path("/tmp/pvk-llm-without-dataset"),
        demo_csv_path=DATA_PATH,
    )

    assert dataset["task_id"] == "passivation_demo"
    assert dataset["data_source"] == "demo_optimization_table.csv"
    assert "PVK-LLM" in dataset["data_boundary"]
    assert dataset["records"]


def test_fallback_loader_handles_missing_passivator_indicator_columns(tmp_path):
    sparse_csv = tmp_path / "sparse_demo.csv"
    sparse_csv.write_text(
        "experiment_id,passivator_combo,PCE_percent\nSPARSE-1,EDAI2,21.5\n",
        encoding="utf-8",
    )

    dataset = load_pvk_dataset_or_demo(
        pvk_project_root=tmp_path / "missing-pvk",
        demo_csv_path=sparse_csv,
    )

    assert dataset["records"][0]["score"] == 21.5
    assert dataset["records"][0]["has_EDAI2"] == 0
    assert dataset["records"][0]["has_PipDI"] == 0


def test_create_session_returns_serializable_contract():
    runtime = OptimizationSessionRuntime(
        pvk_project_root=Path("/tmp/pvk-llm-without-dataset"),
        demo_csv_path=DATA_PATH,
    )

    session = runtime.create_session(OptimizationSessionRequest(n_initial=3, n_trials=2))

    assert session["session_id"].startswith("pvk_session_")
    assert session["status"] == "running"
    assert session["task"]["task_id"] == "passivation_demo"
    assert session["task"]["data_source"] == "demo_optimization_table.csv"
    assert session["current_step"] == 0
    assert len(session["observed_configs"]) == 3
    assert len(session["observed_fvals"]) == 3
    assert session["best_result"]["score"] == max(session["observed_fvals"])
    assert session["candidate_points"] == []
    assert session["tool_trace"]
    assert session["guardrails"]["llm_enabled"] is False
    assert "data_boundary" in session["guardrails"]
    json.dumps(session, allow_nan=False)


def test_run_step_executes_pvk_style_optimization_step():
    runtime = OptimizationSessionRuntime(
        pvk_project_root=Path("/tmp/pvk-llm-without-dataset"),
        demo_csv_path=DATA_PATH,
    )
    session = runtime.create_session(OptimizationSessionRequest(n_initial=3, n_trials=2, seed=7))

    stepped = runtime.run_step(session["session_id"])

    assert stepped["current_step"] == 1
    assert stepped["status"] == "running"
    assert len(stepped["observed_configs"]) == 4
    assert len(stepped["observed_fvals"]) == 4
    assert stepped["candidate_points"]
    assert stepped["best_result"]["score"] == max(stepped["observed_fvals"])
    trace_steps = [event["step"] for event in stepped["tool_trace"]]
    assert "LLM_ACQ.generate_candidate_points" in trace_steps
    assert "LLM_SURROGATE.select_query_point" in trace_steps
    assert "black_box.evaluate_candidate" in trace_steps
    assert "PVKBO.update_observations" in trace_steps


def test_run_step_marks_session_completed_after_requested_trials():
    runtime = OptimizationSessionRuntime(
        pvk_project_root=Path("/tmp/pvk-llm-without-dataset"),
        demo_csv_path=DATA_PATH,
    )
    session = runtime.create_session(OptimizationSessionRequest(n_initial=2, n_trials=1))

    completed = runtime.run_step(session["session_id"])

    assert completed["status"] == "completed"
    assert completed["current_step"] == 1


def test_get_artifacts_returns_summary_and_trace():
    runtime = OptimizationSessionRuntime(
        pvk_project_root=Path("/tmp/pvk-llm-without-dataset"),
        demo_csv_path=DATA_PATH,
    )
    session = runtime.create_session(OptimizationSessionRequest(n_initial=3, n_trials=1))
    runtime.run_step(session["session_id"])

    artifacts = runtime.get_artifacts(session["session_id"])

    assert artifacts["session_id"] == session["session_id"]
    assert artifacts["data_source"] == "demo_optimization_table.csv"
    assert "data_boundary" in artifacts
    assert artifacts["summary"]["best_score"] == max(artifacts["summary"]["observed_fvals"])
    assert artifacts["tool_trace"]
