from pathlib import Path

import pandas as pd
import pytest

from pvk_demo import (
    build_agent_pipeline,
    build_data_summary,
    generate_recommendations,
    load_experiment_data,
    simulate_feedback,
    ui_text,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "demo_optimization_table.csv"


def test_load_experiment_data_normalizes_current_csv_schema():
    data = load_experiment_data(DATA_PATH)

    assert len(data) > 0
    assert "PCE" in data.columns
    assert "passivator_system" in data.columns
    assert data["PCE"].max() == pytest.approx(26.3)


def test_build_data_summary_exposes_demo_data_health():
    data = load_experiment_data(DATA_PATH)
    summary = build_data_summary(data)

    assert summary.total_records == len(data)
    assert summary.best_experiment["experiment_id"]
    assert summary.best_pce == pytest.approx(26.3)
    assert summary.pipdi_real_sample_count == 0
    assert any("PipDI" in note for note in summary.data_health_notes)
    assert "EDAI2" in summary.passivator_counts


def test_generate_recommendations_returns_required_agent_team_outputs():
    data = load_experiment_data(DATA_PATH)
    summary = build_data_summary(data)
    recommendations = generate_recommendations(data, summary, n=5)

    assert len(recommendations) == 5
    assert {"Exploitation", "Exploration", "Control"}.issubset(
        {recommendation.recommendation_type for recommendation in recommendations}
    )
    assert all(recommendation.steps for recommendation in recommendations)
    assert all(recommendation.evidence_level for recommendation in recommendations)
    assert all(recommendation.validation_required for recommendation in recommendations)

    pipdi_recommendations = [
        recommendation
        for recommendation in recommendations
        if "PipDI" in recommendation.passivator_combination
    ]
    assert pipdi_recommendations
    assert all(
        recommendation.data_boundary == "demo-only exploration"
        for recommendation in pipdi_recommendations
    )


def test_generate_recommendations_respects_smaller_demo_batch_size():
    data = load_experiment_data(DATA_PATH)
    summary = build_data_summary(data)
    recommendations = generate_recommendations(data, summary, n=3)

    assert len(recommendations) == 3
    assert {recommendation.recommendation_type for recommendation in recommendations} == {
        "Exploitation",
        "Exploration",
        "Control",
    }


def test_simulate_feedback_never_claims_real_algorithm_performance():
    data = load_experiment_data(DATA_PATH)
    summary = build_data_summary(data)
    recommendations = generate_recommendations(data, summary, n=3)
    feedback = simulate_feedback(summary, recommendations)

    assert feedback["caption"].startswith("Synthetic walkthrough")
    assert feedback["rounds"] == ["Initial data", "Recommended batch", "Simulated feedback"]
    assert feedback["best_so_far"][0] == summary.best_pce
    assert feedback["best_so_far"][-1] >= summary.best_pce


def test_build_agent_pipeline_documents_claw_style_roles():
    pipeline = build_agent_pipeline()

    assert [stage.name for stage in pipeline] == [
        "Data Agent",
        "Domain Agent",
        "Optimizer Agent",
        "Critic Agent",
        "Experiment Planner Agent",
        "Reporter Agent",
    ]
    assert all(stage.output_summary for stage in pipeline)


def test_ui_text_supports_chinese_and_english_fallbacks():
    assert ui_text("app_title", "en") == "PVK-BO Agent Demo"
    assert ui_text("app_title", "zh") == "PVK-BO Agent 演示"
    assert ui_text("missing_key", "zh") == "missing_key"
    assert ui_text("app_title", "unsupported") == "PVK-BO Agent Demo"


def test_summary_handles_empty_dataframe_without_crashing():
    empty = pd.DataFrame(columns=["experiment_id", "PCE", "passivator_system", "data_type"])
    summary = build_data_summary(empty)

    assert summary.total_records == 0
    assert summary.best_pce == 0.0
    assert summary.best_experiment == {}
    assert any("No records" in note for note in summary.data_health_notes)


def test_summary_handles_nonstandard_nonempty_dataframe_without_crashing():
    raw = pd.DataFrame(
        [
            {
                "experiment_id": "RAW-1",
                "PCE_percent": 21.5,
                "passivator_combo": "EDAI2",
            }
        ]
    )

    summary = build_data_summary(raw)

    assert summary.total_records == 1
    assert summary.best_experiment["experiment_id"] == "RAW-1"
    assert summary.passivator_counts["EDAI2"] == 0
