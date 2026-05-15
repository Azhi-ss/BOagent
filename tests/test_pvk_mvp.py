import pytest

from pvk_mvp import (
    build_passivation_target,
    compute_bo_curve,
    generate_screening_analysis,
    handle_mvp_chat_turn,
)


def test_build_passivation_target_exposes_target_card_and_pipdi_boundary():
    target = build_passivation_target()

    assert target["title"] == "钙钛矿钝化配方优化 Target"
    assert "PCE" in target["objective"]
    assert set(target["passivators"]) == {"3MTPAI", "PDAI2", "EDAI2", "PipDI"}
    assert target["passivators"]["PipDI"]["risk"] == "高风险"
    assert "无真实样本" in target["passivators"]["PipDI"]["evidence_level"]
    assert "strategy/combination" in target["data_boundary"]
    assert target["champion_threshold"]["metric"] == "PCE"


def test_compute_bo_curve_returns_only_pvk_bo_series_and_boundary():
    curve = compute_bo_curve()

    assert curve["curve_boundary"] == "session best-so-far only, not benchmark"
    assert set(curve["series"]) == {"pvk_bo"}
    for series in curve["series"].values():
        assert series["label"]
        assert all({"iteration", "pce"} == set(point) for point in series["points"])


def test_compute_bo_curve_uses_demo_series_for_empty_observations():
    curve = compute_bo_curve([])

    assert curve["series"]["pvk_bo"]["points"]
    assert curve["series"]["pvk_bo"]["boundary"] == "demo walkthrough values, not benchmark"


def test_compute_bo_curve_uses_best_so_far_for_observed_fvals():
    curve = compute_bo_curve([20.0, 19.5, 21.2, 20.8, 22.0])
    pvk_points = curve["series"]["pvk_bo"]["points"]

    assert [point["iteration"] for point in pvk_points] == [1, 2, 3, 4, 5]
    assert [point["pce"] for point in pvk_points] == pytest.approx(
        [20.0, 20.0, 21.2, 21.2, 22.0]
    )
    assert curve["series"]["pvk_bo"]["boundary"] == "observed best-so-far from supplied fvals"


def test_generate_screening_analysis_prioritizes_grounded_candidates():
    target = build_passivation_target()
    analysis = generate_screening_analysis(target)

    assert analysis["phase"] == "Screening"
    assert analysis["prior_knowledge"]
    assert "小样本" in analysis["small_sample_induction"]
    assert "语言层" in analysis["language_level_explanation"]
    assert [item["passivator"] for item in analysis["candidate_rankings"]] == [
        "EDAI2",
        "PDAI2",
        "3MTPAI",
        "PipDI",
    ]
    pipdi = analysis["candidate_rankings"][-1]
    assert pipdi["risk"] == "高风险"
    assert "真实样本" in pipdi["rationale"]


def test_handle_mvp_chat_turn_returns_tool_calls_and_artifacts():
    response = handle_mvp_chat_turn("请开始 Screening，并解释为什么 PipDI 要谨慎")

    assert response["phase"] == "Screening"
    assert "PipDI" in response["assistant_message"]
    assert "不是真实 molar ratio BO" in response["assistant_message"]
    assert response["tool_calls"]
    assert {call["name"] for call in response["tool_calls"]}.issuperset(
        {"build_passivation_target", "generate_screening_analysis"}
    )
    assert "target" in response["artifacts"]
    assert "screening_analysis" in response["artifacts"]
    screening_call = next(
        call for call in response["tool_calls"] if call["name"] == "generate_screening_analysis"
    )
    assert set(screening_call["arguments"]) == {"target", "session"}


def test_handle_mvp_chat_turn_supports_initialization_and_optimization_paths():
    initialization = handle_mvp_chat_turn("帮我初始化这个配方优化任务")
    optimization = handle_mvp_chat_turn(
        "进入 Optimization 并给出曲线",
        session={"observed_fvals": [20.0, 21.0, 20.5]},
    )

    assert initialization["phase"] == "Initialization"
    assert "target" in initialization["artifacts"]
    assert initialization["tool_calls"][0]["arguments"] == {"session_or_artifacts": "session"}

    assert optimization["phase"] == "Optimization"
    assert "bo_curve" in optimization["artifacts"]
    assert optimization["artifacts"]["bo_curve"]["curve_boundary"] == "session best-so-far only, not benchmark"
    compute_call = next(
        call for call in optimization["tool_calls"] if call["name"] == "compute_bo_curve"
    )
    assert compute_call["arguments"] == {"observed_fvals": [20.0, 21.0, 20.5]}

