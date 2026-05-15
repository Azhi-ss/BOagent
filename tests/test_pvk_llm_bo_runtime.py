from pathlib import Path

import pandas as pd
import pytest

from pvk_llm_bo_runtime import (
    RealPvkBoRuntime,
    RealPvkBoSessionRequest,
    RealPvkBoUnavailableError,
    _install_openai_single_completion_compat,
    _install_pandas_series_int_position_compat,
    build_real_task_context,
)


def _band_alignment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CHI_PVK": 3.8,
                "Eg_HTL": 2.1,
                "CHI_HTL": 2.4,
                "Eg_ETL": 3.0,
                "CHI_ETL": 4.1,
                "eta": 22.4,
            },
            {
                "CHI_PVK": 3.9,
                "Eg_HTL": 2.0,
                "CHI_HTL": 2.5,
                "Eg_ETL": 3.1,
                "CHI_ETL": 4.0,
                "eta": 23.2,
            },
            {
                "CHI_PVK": 4.0,
                "Eg_HTL": 2.2,
                "CHI_HTL": 2.6,
                "Eg_ETL": 3.2,
                "CHI_ETL": 4.2,
                "eta": 21.9,
            },
        ]
    )


class FakePvkBo:
    def __init__(
        self,
        task_context,
        sm_mode,
        n_candidates,
        n_templates,
        n_gens,
        alpha,
        n_initial_samples,
        n_trials,
        init_f,
        bbox_eval_f,
        chat_engine,
        top_pct=None,
        use_input_warping=False,
        prompt_setting=None,
        shuffle_features=False,
    ):
        del sm_mode, n_templates, n_gens, alpha, n_trials, chat_engine
        del top_pct, use_input_warping, prompt_setting, shuffle_features
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.init_f = init_f
        self.bbox_eval_f = bbox_eval_f
        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame()
        self.llm_query_cost = []
        self.llm_query_time = []

    def _initialize(self):
        configs = self.init_f(2)
        rows = []
        fvals = []
        for config in configs:
            evaluated_config, result = self.bbox_eval_f(config)
            rows.append(evaluated_config)
            fvals.append(result)
        self.observed_configs = pd.DataFrame(rows)
        self.observed_fvals = pd.DataFrame(fvals)
        return 0.0, 0.01

    def _evaluate_config(self, config):
        return pd.DataFrame([config.to_dict("records")[0]]), pd.DataFrame(
            [{"score": 24.1, "generalization_score": 24.1}]
        )

    def _update_observations(self, new_config, new_fval):
        self.observed_configs = pd.concat(
            [self.observed_configs, new_config], ignore_index=True
        )
        self.observed_fvals = pd.concat([self.observed_fvals, new_fval], ignore_index=True)

    @property
    def acq_func(self):
        class Acquisition:
            def get_candidate_points(_, observed_configs, observed_fvals, alpha):
                del observed_configs, observed_fvals, alpha
                return (
                    pd.DataFrame(
                        [
                            {
                                "CHI_PVK": 4.0,
                                "Eg_HTL": 2.2,
                                "CHI_HTL": 2.6,
                                "Eg_ETL": 3.2,
                                "CHI_ETL": 4.2,
                            }
                        ]
                    ),
                    0.11,
                    0.22,
                )

        return Acquisition()

    @property
    def surrogate_model(self):
        class Surrogate:
            def select_query_point(_, observed_configs, observed_fvals, candidate_points):
                del observed_configs, observed_fvals
                return candidate_points.iloc[[0]], 0.33, 0.44

        return Surrogate()


def test_missing_real_excel_fails_fast_with_expected_path():
    runtime = RealPvkBoRuntime(data_root=Path("/tmp/missing-pvk-dataset"))

    with pytest.raises(RealPvkBoUnavailableError) as exc:
        runtime.create_session(RealPvkBoSessionRequest(task_id="band_alignment"))

    assert "bandAlignment.xlsx" in str(exc.value)
    assert "/tmp/missing-pvk-dataset" in str(exc.value)


def test_build_real_task_context_uses_pvk_feature_constraints():
    context = build_real_task_context("band_alignment", _band_alignment_frame())

    assert context["model"] == "band_alignment"
    assert context["lower_is_better"] is False
    assert context["feature_cols"] == [
        "CHI_PVK",
        "Eg_HTL",
        "CHI_HTL",
        "Eg_ETL",
        "CHI_ETL",
    ]
    assert context["target_col"] == "eta"
    assert context["tot_feats"] == 5
    assert context["cat_feats"] == 0
    assert context["num_feats"] == 5
    assert context["hyperparameter_constraints"]["CHI_PVK"] == [
        "float",
        "linear",
        [3.8, 4.0],
    ]


def test_runtime_runs_one_real_pvkbo_step_with_serializable_artifacts(monkeypatch, tmp_path):
    data_root = tmp_path / "custom_perovskite_dataset"
    data_root.mkdir()
    workbook = data_root / "bandAlignment.xlsx"
    workbook.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("pvk_llm_bo_runtime.pd.read_excel", lambda _: _band_alignment_frame())

    runtime = RealPvkBoRuntime(data_root=data_root, pvk_bo_class=FakePvkBo)
    session = runtime.create_session(
        RealPvkBoSessionRequest(task_id="band_alignment", n_initial=2, n_trials=1)
    )
    stepped = runtime.run_step(session["session_id"])
    artifacts = runtime.get_artifacts(session["session_id"])

    assert stepped["status"] == "completed"
    assert stepped["current_step"] == 1
    assert stepped["task"]["data_source"] == "PVK-LLM:bandAlignment.xlsx"
    assert stepped["best_result"]["score"] == pytest.approx(24.1)
    assert stepped["candidate_points"][0]["candidate_id"] == "PVK-CAND-01-01"
    assert [event["step"] for event in stepped["tool_trace"]] == [
        "PVKBO.initialize",
        "LLM_ACQ.get_candidate_points",
        "LLM_SURROGATE.select_query_point",
        "black_box.evaluate_candidate",
        "PVKBO.update_observations",
    ]
    assert artifacts["summary"]["best_score"] == pytest.approx(24.1)
    assert artifacts["candidate_points"]


def test_pandas_series_int_position_compat_restores_legacy_pvk_indexing():
    original_getitem = pd.Series.__getitem__
    had_flag = hasattr(pd.Series, "_boagent_legacy_int_position_compat")
    original_flag = getattr(pd.Series, "_boagent_legacy_int_position_compat", None)
    try:
        if had_flag:
            delattr(pd.Series, "_boagent_legacy_int_position_compat")
        _install_pandas_series_int_position_compat()
        row = pd.Series({"CHI_PVK": 3.8, "Eg_HTL": 2.1})

        assert row[0] == 3.8
        assert row[1] == 2.1
    finally:
        pd.Series.__getitem__ = original_getitem
        if hasattr(pd.Series, "_boagent_legacy_int_position_compat"):
            delattr(pd.Series, "_boagent_legacy_int_position_compat")
        if had_flag:
            pd.Series._boagent_legacy_int_position_compat = original_flag


def test_openai_single_completion_compat_clamps_unsupported_n(monkeypatch):
    from openai.resources.chat.completions import AsyncCompletions

    captured = {}

    async def fake_create(self, *args, **kwargs):
        del self, args
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(AsyncCompletions, "create", fake_create)
    if hasattr(AsyncCompletions, "_boagent_force_single_completion"):
        monkeypatch.delattr(AsyncCompletions, "_boagent_force_single_completion")
    _install_openai_single_completion_compat()

    import asyncio

    result = asyncio.run(
        AsyncCompletions.create(object(), model="deepseek-v4-flash", messages=[], n=4)
    )

    assert result == "ok"
    assert captured["n"] == 1
    assert captured["max_tokens"] == 512
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_single_completion_compat_leaves_other_models_untouched(monkeypatch):
    from openai.resources.chat.completions import AsyncCompletions

    captured = {}

    async def fake_create(self, *args, **kwargs):
        del self, args
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(AsyncCompletions, "create", fake_create)
    if hasattr(AsyncCompletions, "_boagent_force_single_completion"):
        monkeypatch.delattr(AsyncCompletions, "_boagent_force_single_completion")
    _install_openai_single_completion_compat()

    import asyncio

    result = asyncio.run(
        AsyncCompletions.create(object(), model="gpt-4.1", messages=[], n=4)
    )

    assert result == "ok"
    assert captured == {"model": "gpt-4.1", "messages": [], "n": 4}
