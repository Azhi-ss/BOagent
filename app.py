from __future__ import annotations

import pandas as pd
import streamlit as st

from pvk_demo import (
    DEFAULT_DATA_PATH,
    DataSummary,
    Recommendation,
    build_agent_pipeline,
    build_data_summary,
    generate_recommendations,
    load_experiment_data,
    simulate_feedback,
    ui_text,
)


st.set_page_config(
    page_title="PVK-BO Agent Demo",
    layout="wide",
)


def main() -> None:
    language = _select_language()
    t = lambda key: ui_text(key, language)

    st.title(t("app_title"))
    st.caption(t("app_caption"))
    st.warning(t("demo_warning"))

    data = _load_data_with_fallback()
    summary = build_data_summary(data)

    _render_sidebar(summary, language)

    setup_tab, data_tab, recommendation_tab, feedback_tab = st.tabs(
        [
            t("setup_tab"),
            t("data_tab"),
            t("recommendation_tab"),
            t("feedback_tab"),
        ]
    )

    with setup_tab:
        _render_setup(language)

    with data_tab:
        _render_data_analysis(data, summary, language)

    with recommendation_tab:
        recommendations = _render_recommendations(data, summary, language)

    with feedback_tab:
        _render_feedback(summary, recommendations, language)


def _select_language() -> str:
    choice = st.sidebar.selectbox(
        ui_text("language", "en"),
        options=["中文", "English"],
        index=0,
    )
    return "zh" if choice == "中文" else "en"


@st.cache_data(show_spinner=False)
def _load_demo_data() -> pd.DataFrame:
    return load_experiment_data(DEFAULT_DATA_PATH)


def _load_data_with_fallback() -> pd.DataFrame:
    try:
        return _load_demo_data()
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        st.error(f"Could not load demo CSV: {exc}")
        return pd.DataFrame()


def _render_sidebar(summary: DataSummary, language: str) -> None:
    t = lambda key: ui_text(key, language)
    st.sidebar.header(t("agent_team"))
    for index, stage in enumerate(build_agent_pipeline(), start=1):
        with st.sidebar.expander(f"{index}. {stage.name}", expanded=index <= 2):
            st.write(stage.responsibility)
            st.caption(stage.output_summary)

    st.sidebar.header(t("data_boundary"))
    st.sidebar.metric(t("records"), summary.total_records)
    st.sidebar.metric(t("best_pce"), f"{summary.best_pce:.2f}%")
    st.sidebar.info(t("sidebar_boundary_note"))


def _render_setup(language: str) -> None:
    t = lambda key: ui_text(key, language)
    st.subheader(t("task_setup"))
    st.text_area(
        t("natural_language_task"),
        value=t("default_task"),
        height=100,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(t("primary_objective"), t("maximize_pce"))
    col_b.metric(t("candidates"), "3MTPAI / PDAI2 / EDAI2 / PipDI")
    col_c.metric(t("demo_mode"), t("csv_rules"))

    st.info(t("setup_info"))


def _render_data_analysis(
    data: pd.DataFrame, summary: DataSummary, language: str
) -> None:
    t = lambda key: ui_text(key, language)
    st.subheader(t("data_grounding"))
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(t("total_records"), summary.total_records)
    col_b.metric(t("best_pce"), f"{summary.best_pce:.2f}%")
    col_c.metric(t("mean_pce"), f"{summary.mean_pce:.2f}%")
    col_d.metric(t("real_pipdi_samples"), summary.pipdi_real_sample_count)

    st.markdown(f"#### {t('data_health')}")
    for note in summary.data_health_notes:
        st.warning(note)

    if summary.best_experiment:
        st.markdown(f"#### {t('current_best_evidence')}")
        best = summary.best_experiment
        st.write(
            {
                "experiment_id": best.get("experiment_id"),
                "passivator_system": best.get("passivator_system"),
                "PCE": best.get("PCE"),
                "data_type": best.get("data_type"),
            }
        )
        with st.expander(t("evidence_text")):
            st.write(best.get("evidence_text", t("no_evidence")))

    st.markdown(f"#### {t('passivator_distribution')}")
    count_frame = pd.DataFrame(
        {
            "passivator": list(summary.passivator_counts.keys()),
            "count": list(summary.passivator_counts.values()),
        }
    ).set_index("passivator")
    st.bar_chart(count_frame)

    st.markdown(f"#### {t('demo_table_preview')}")
    preview_columns = [
        column
        for column in [
            "experiment_id",
            "passivator_system",
            "PCE",
            "Voc_V",
            "Jsc_mA_cm2",
            "FF_percent",
            "data_type",
            "recommendation_role",
        ]
        if column in data.columns
    ]
    st.dataframe(data[preview_columns], use_container_width=True, hide_index=True)


def _render_recommendations(
    data: pd.DataFrame, summary: DataSummary, language: str
) -> list[Recommendation]:
    t = lambda key: ui_text(key, language)
    st.subheader(t("recommendation_plan"))
    st.caption(t("recommendation_caption"))

    count = st.slider(t("recommendation_count"), min_value=3, max_value=5, value=5)
    if (
        "recommendations" not in st.session_state
        or st.session_state.get("recommendation_count") != count
    ):
        st.session_state.recommendations = generate_recommendations(data, summary, n=count)
        st.session_state.recommendation_count = count

    if st.button(t("generate_next"), type="primary"):
        with st.spinner(t("running_pipeline")):
            st.session_state.recommendations = generate_recommendations(data, summary, n=count)
            st.session_state.recommendation_count = count

    recommendations = st.session_state.recommendations
    if not recommendations:
        st.error(t("no_recommendations"))
        return []

    for recommendation in recommendations:
        with st.container(border=True):
            st.markdown(
                f"### {recommendation.experiment_id}: {recommendation.passivator_combination}"
            )
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric(t("type"), recommendation.recommendation_type)
            col_b.metric(t("risk"), recommendation.risk_level)
            col_c.metric(t("mock_score"), f"{recommendation.mock_acquisition_score:.2f}")
            col_d.metric(t("validation"), t("required"))
            st.caption(f"{t('data_boundary_label')}: {recommendation.data_boundary}")
            if recommendation.risk_level == "High":
                st.error(t("high_risk_warning"))
            else:
                st.info(recommendation.evidence_level)

            with st.expander(t("experiment_plan_evidence"), expanded=True):
                st.write(t("process_assumptions"), recommendation.process)
                st.markdown(f"**{t('steps')}**")
                for step in recommendation.steps:
                    st.write(f"- {step}")
                st.markdown(f"**{t('reason')}**")
                st.write(recommendation.reason)
                st.markdown(f"**{t('risks')}**")
                for risk in recommendation.risks:
                    st.write(f"- {risk}")
                st.markdown(f"**{t('hypothesis_to_validate')}**")
                st.write(recommendation.hypothesis)
                st.markdown(f"**{t('supporting_records')}**")
                st.write(
                    ", ".join(recommendation.supporting_records)
                    or t("demo_only_hypothesis")
                )

    return recommendations


def _render_feedback(
    summary: DataSummary, recommendations: list[Recommendation], language: str
) -> None:
    t = lambda key: ui_text(key, language)
    st.subheader(t("feedback_loop"))
    st.caption(t("feedback_caption"))

    if st.button(t("simulate_feedback")):
        st.session_state.feedback = simulate_feedback(summary, recommendations)

    if "feedback" not in st.session_state:
        st.session_state.feedback = simulate_feedback(summary, recommendations)
    feedback = st.session_state.feedback
    chart_data = pd.DataFrame(
        {
            "round": feedback["rounds"],
            "best_so_far": feedback["best_so_far"],
        }
    ).set_index("round")
    st.line_chart(chart_data)
    st.warning(feedback["caption"])

    with st.expander(t("round_labels")):
        for round_name, label in zip(feedback["rounds"], feedback["uncertainty_label"]):
            st.write(f"- {round_name}: {label}")


if __name__ == "__main__":
    main()
