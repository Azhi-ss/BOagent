from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from bo_core.optimization import chem_lgbo
from bo_core.optimization.chem_lgbo import (
    ChemLGBOEngine,
    build_subspace_mask,
    generate_counterfactual_indices,
    masked_mean_shift,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ligand": ["L1", "L1", "L2", "L3"],
            "Base": ["B1", "B2", "B2", "B1"],
            "Solvent": ["S1", "S2", "S1", "S2"],
        }
    )


def test_subspace_mask_uses_joint_membership_and_omitted_fields() -> None:
    candidates = _candidates()

    assert build_subspace_mask(
        candidates,
        {"Ligand": ["L1", "L2"], "Base": ["B2"]},
    ).tolist() == [False, True, True, False]
    assert build_subspace_mask(candidates, {"Ligand": ["L2"]}).tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert build_subspace_mask(candidates, {}).tolist() == [True, True, True, True]
    assert build_subspace_mask(candidates, {"Base": ["missing"]}).tolist() == [
        False,
        False,
        False,
        False,
    ]


def test_masked_mean_shift_adds_one_sigma_without_mutating_inputs() -> None:
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = np.array([0.1, 0.2, 0.3, 0.4])
    mask = np.array([False, True, True, False])
    mu_before = mu.copy()
    sigma_before = sigma.copy()

    shifted = masked_mean_shift(mu, sigma, mask)

    np.testing.assert_array_equal(shifted[mask], mu[mask] + sigma[mask])
    np.testing.assert_array_equal(shifted[~mask], mu[~mask])
    np.testing.assert_array_equal(mu, mu_before)
    np.testing.assert_array_equal(sigma, sigma_before)
    assert not np.shares_memory(shifted, mu)


@pytest.mark.parametrize(
    ("mu", "sigma", "mask", "error"),
    [
        (
            np.array([[1.0, 2.0]]),
            np.array([0.1, 0.2]),
            np.array([True, False]),
            ValueError,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.1]),
            np.array([True, False]),
            ValueError,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            np.array([True]),
            ValueError,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            np.array([1, 0]),
            TypeError,
        ),
        (
            np.array([1, 2]),
            np.array([0.1, 0.2]),
            np.array([True, False]),
            TypeError,
        ),
        (
            np.array([1.0, np.nan]),
            np.array([0.1, 0.2]),
            np.array([True, False]),
            ValueError,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.1, np.inf]),
            np.array([True, False]),
            ValueError,
        ),
    ],
)
def test_masked_mean_shift_rejects_invalid_arrays(
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        masked_mean_shift(mu, sigma, mask)


def _large_non_cartesian_pool() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    ligands = [f"L{i}" for i in range(12)]
    bases = [f"B{i}" for i in range(12)]
    rows = [
        (ligand, base)
        for ligand in ligands
        for base in bases
        if (ligand, base) not in {("L0", "B11"), ("L11", "B0")}
    ]
    return pd.DataFrame(rows, columns=["Ligand", "Base"]), {
        "Ligand": ligands,
        "Base": bases,
    }


def test_counterfactuals_are_matched_legal_deterministic_and_local_rng_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, options = _large_non_cartesian_pool()
    queried = np.zeros(len(candidates), dtype=bool)
    queried[0] = True
    mu = np.zeros(len(candidates), dtype=float)
    sigma = np.ones(len(candidates), dtype=float)
    accepted: list[tuple[dict[str, list[str]], np.ndarray]] = []
    latest: dict[str, object] = {}
    real_build_mask = build_subspace_mask

    def record_mask(
        frame: pd.DataFrame, proposal: dict[str, list[str]]
    ) -> np.ndarray:
        mask = real_build_mask(frame, proposal)
        latest["proposal"] = {key: list(values) for key, values in proposal.items()}
        latest["mask"] = mask
        return mask

    def fake_ei(
        shifted: np.ndarray, posterior_sigma: np.ndarray, best_f: float
    ) -> np.ndarray:
        del posterior_sigma, best_f
        accepted.append((latest["proposal"], shifted > mu))  # type: ignore[arg-type]
        return shifted

    def fail_global_choice(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("module-level RNG must not be used")

    monkeypatch.setattr(chem_lgbo, "build_subspace_mask", record_mask)
    monkeypatch.setattr(np.random, "choice", fail_global_choice)
    kwargs = {
        "candidate_features": candidates,
        "feature_options": options,
        "subspace": {"Ligand": ["L0"], "Base": ["B0"]},
        "queried_mask": queried,
        "mu": mu,
        "sigma": sigma,
        "best_f": 0.0,
        "expected_improvement": fake_ei,
        "count": 100,
    }

    first = generate_counterfactual_indices(
        **kwargs, rng=np.random.RandomState(123)
    )
    first_accepted = accepted.copy()
    accepted.clear()
    second = generate_counterfactual_indices(
        **kwargs, rng=np.random.RandomState(123)
    )

    assert first == second
    assert len(first) == len(first_accepted) == 100
    remaining = ~queried
    for index, (proposal, effective_mask) in zip(first, first_accepted):
        assert list(proposal) == ["Ligand", "Base"]
        assert len(proposal["Ligand"]) == len(proposal["Base"]) == 1
        assert proposal["Ligand"][0] in options["Ligand"]
        assert proposal["Base"][0] in options["Base"]
        assert 0 < np.count_nonzero(effective_mask) < np.count_nonzero(remaining)
        assert 0 <= index < len(candidates)
        assert remaining[index] and effective_mask[index]


def test_counterfactuals_stop_at_attempt_bound_without_padding_duplicates() -> None:
    class CountingRandomState:
        def __init__(self) -> None:
            self.inner = np.random.RandomState(7)
            self.calls = 0

        def choice(self, *args: object, **kwargs: object) -> np.ndarray:
            self.calls += 1
            assert self.calls <= 200
            return self.inner.choice(*args, **kwargs)

    candidates = pd.DataFrame({"Ligand": ["L1", "L1", "L2"]})
    rng = CountingRandomState()

    indices = generate_counterfactual_indices(
        candidate_features=candidates,
        feature_options={"Ligand": ["L1"]},
        subspace={"Ligand": ["L1"]},
        queried_mask=np.zeros(3, dtype=bool),
        mu=np.zeros(3),
        sigma=np.ones(3),
        best_f=0.0,
        expected_improvement=lambda shifted, _sigma, _best: shifted,
        rng=rng,  # type: ignore[arg-type]
        count=2,
    )

    assert indices == [0]
    assert rng.calls == 200


def test_counterfactual_api_does_not_accept_oracle_values() -> None:
    candidates = pd.DataFrame({"Ligand": ["L1", "L2"]})
    with pytest.raises(TypeError, match="pool_yield"):
        generate_counterfactual_indices(
            candidate_features=candidates,
            feature_options={"Ligand": ["L1", "L2"]},
            subspace={"Ligand": ["L1"]},
            queried_mask=np.zeros(2, dtype=bool),
            mu=np.zeros(2),
            sigma=np.ones(2),
            best_f=0.0,
            expected_improvement=lambda shifted, _sigma, _best: shifted,
            rng=np.random.RandomState(1),
            count=1,
            pool_yield=np.array([10.0, 20.0]),  # type: ignore[call-arg]
        )


class _ChemPosterior:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, _x: np.ndarray, _y: np.ndarray) -> _ChemPosterior:
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(x) == len(self.mean)
        return self.mean.copy(), self.std.copy()


class _ChemClient:
    def __init__(
        self,
        content: str = "",
        *,
        responses: list[SimpleNamespace] | None = None,
        status: str = "success",
        error: str | None = None,
        raises: bool = False,
    ) -> None:
        self.content = content
        self.responses = responses
        self.status = status
        self.error = error
        self.raises = raises
        self.calls = 0
        self.messages: list[list[dict[str, object]]] = []
        self.kwargs: list[dict[str, object]] = []

    def is_configured(self) -> bool:
        return True

    def chat(
        self, messages: list[dict[str, object]], **kwargs: object
    ) -> SimpleNamespace:
        self.calls += 1
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        if self.raises:
            raise RuntimeError("transport failed")
        if self.responses is not None:
            return self.responses[self.calls - 1]
        return _tool_result(
            self.content,
            call_id=f"call-{self.calls}",
            status=self.status,
            error=self.error,
        )


def _tool_result(
    arguments: str,
    *,
    call_id: str = "call-1",
    name: str = "propose_sparse_subspace",
    status: str = "success",
    error: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        content="",
        error=error,
        usage={},
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    )


def _chem_engine(
    dataset: str = "suzuki",
    *,
    n_counterfactuals: int = 0,
    outcome_feedback: bool = False,
) -> ChemLGBOEngine:
    engine = ChemLGBOEngine(
        dataset,
        seed=100,
        use_llm=False,
        n_iters=0,
        n_restarts=0,
        n_counterfactuals=n_counterfactuals,
        outcome_feedback=outcome_feedback,
    )
    engine.use_llm = True
    return engine


def _proposal(field: str, values: list[str]) -> str:
    return "Thinking: fixed chemistry rationale\n" + json.dumps(
        {"subspace": {field: values}}
    )


def _posterior_for(engine: ChemLGBOEngine) -> tuple[np.ndarray, np.ndarray]:
    best = float(np.max(engine.y_obs))
    return np.full(engine.M, best - 4.0), np.ones(engine.M)


def _direct_gp_index(
    engine: ChemLGBOEngine, mean: np.ndarray, std: np.ndarray
) -> int:
    acquisition = engine._expected_improvement(
        mean, std, float(np.max(engine.y_obs))
    )
    remaining = np.ones(engine.M, dtype=bool)
    for index in engine.queried:
        remaining[index] = False
    return int(np.argmax(np.where(remaining, acquisition, -np.inf)))


def test_chem_engine_applies_exact_shift_but_ei_can_escape_subspace() -> None:
    engine = _chem_engine(n_counterfactuals=3)
    value = str(engine.test_df["Ligand"].iloc[0])
    subspace = {"Ligand": [value]}
    mask = build_subspace_mask(engine.test_df[engine.feature_cols], subspace)
    outside_index = int(np.flatnonzero(~mask)[0])
    mean, std = _posterior_for(engine)
    mean[outside_index] = float(np.max(engine.y_obs)) + 5.0
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient(_proposal("Ligand", [value]))
    engine._client = client

    row = engine.step()

    expected_mean = masked_mean_shift(mean, std, mask)
    expected_index = int(
        np.argmax(
            engine._expected_improvement(
                expected_mean, std, float(np.max(engine.y_obs[:-1]))
            )
        )
    )
    assert client.calls == 1
    assert row["query_index"] == expected_index == outside_index
    assert row["predicted_yield"] == pytest.approx(expected_mean[expected_index])
    assert row["guidance_status"] == "applied"
    assert row["guidance_reason"] == "accepted"
    assert row["subspace"] == subspace
    assert row["mask_size"] == int(mask.sum())
    assert row["remaining_pool_size"] == engine.M
    assert row["coverage"] == pytest.approx(mask.mean())
    assert row["selected_in_subspace"] is False
    assert row["counterfactual_seed"] == 100_000
    assert not ({"raw_response", "mask", "counterfactual_indices"} & row.keys())

    artifact = engine.guidance_artifacts[0]
    assert artifact["raw_response"] == client.content
    assert artifact["parser_reason"] == "accepted"
    assert artifact["subspace"] == subspace
    assert artifact["selected_index"] == row["query_index"]
    assert len(artifact["counterfactual_indices"]) == 3


def test_treatment_prompt_receives_only_completed_previous_guidance_outcome() -> None:
    engine = _chem_engine(outcome_feedback=True)
    value = str(engine.test_df["Ligand"].iloc[0])
    mean, std = _posterior_for(engine)
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient(_proposal("Ligand", [value]))
    engine._client = client
    incumbent_before = float(np.max(engine.y_obs))

    first = engine.step()
    engine.step()

    first_user = client.messages[0][1]["content"]
    second_user = client.messages[1][1]["content"]
    assert "[Previous guidance outcome]\n- (none)" in first_user
    assert json.dumps(first["subspace"], sort_keys=True) in second_user
    assert json.dumps(first["condition"], sort_keys=True) in second_user
    assert (
        "Selected point was inside the proposed subspace: "
        + str(first["selected_in_subspace"]).lower()
    ) in second_user
    assert (
        f"Observed {engine.meta.target_name}: {first['observed_yield']:.4f}"
        in second_user
    )
    assert f"Incumbent before this trial: {incumbent_before:.4f}" in second_user
    assert (
        f"Improvement over incumbent: {first['observed_yield'] - incumbent_before:.4f}"
        in second_user
    )


def test_previous_outcome_keeps_pretrial_incumbent_when_trial_improves() -> None:
    engine = _chem_engine(outcome_feedback=True)
    value = str(engine.test_df["Ligand"].iloc[0])
    mean, std = _posterior_for(engine)
    selected = _direct_gp_index(engine, mean, std)
    incumbent_before = float(np.max(engine.y_obs))
    engine.pool_yield = engine.pool_yield.copy()
    engine.pool_yield[selected] = incumbent_before + 10.0
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient(_proposal("Ligand", [value]))
    engine._client = client

    first = engine.step()
    engine.step()

    second_user = client.messages[1][1]["content"]
    assert first["observed_yield"] > incumbent_before
    assert f"Incumbent before this trial: {incumbent_before:.4f}" in second_user
    assert f"Improvement over incumbent: {10.0:.4f}" in second_user


def test_control_prompt_does_not_add_guidance_outcome_block() -> None:
    engine = _chem_engine()
    value = str(engine.test_df["Ligand"].iloc[0])
    mean, std = _posterior_for(engine)
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient(_proposal("Ligand", [value]))
    engine._client = client

    engine.step()
    engine.step()

    assert all(
        "[Previous guidance outcome]" not in call[1]["content"]
        for call in client.messages
    )


def test_treatment_does_not_create_outcome_from_fallback_guidance() -> None:
    engine = _chem_engine(outcome_feedback=True)
    mean, std = _posterior_for(engine)
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient("not-json")
    engine._client = client

    first = engine.step()
    client.content = _proposal("Ligand", [str(engine.test_df["Ligand"].iloc[0])])
    engine.step()

    assert first["guidance_status"] == "fallback"
    assert "[Previous guidance outcome]\n- (none)" in client.messages[1][1]["content"]


def test_chem_engine_accepts_one_point_joint_mask() -> None:
    engine = _chem_engine()
    subspace = {
        field: [str(engine.test_df[field].iloc[0])] for field in engine.feature_cols
    }
    mask = build_subspace_mask(engine.test_df[engine.feature_cols], subspace)
    assert mask.sum() == 1
    mean, std = _posterior_for(engine)
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient("Thinking\n" + json.dumps({"subspace": subspace}))
    engine._client = client

    row = engine.step()

    assert row["guidance_status"] == "applied"
    assert row["mask_size"] == 1
    assert client.calls == 1


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("", "empty_response"),
        ("not-json", "invalid_json"),
        ('{"subspace":{}}', "invalid_schema"),
        ('{"subspace":{"Unknown":["x"]}}', "unknown_field"),
        ('{"subspace":{"Ligand":["not-an-option"]}}', "unknown_value"),
        ('{"subspace":{"Ligand":["L1","L1"]}}', "duplicate_value"),
        ('{"subspace":{"Ligand":[]}}', "empty_choice"),
    ],
)
def test_chem_parser_fallbacks_select_same_index_as_gpbo(
    content: str, reason: str
) -> None:
    engine = _chem_engine()
    mean, std = _posterior_for(engine)
    mean[-1] += 2.0
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient(content)
    engine._client = client
    expected_index = _direct_gp_index(engine, mean, std)

    row = engine.step()

    assert row["query_index"] == expected_index
    assert row["predicted_yield"] == pytest.approx(mean[expected_index])
    assert row["guidance_status"] == "fallback"
    assert row["guidance_reason"] == reason
    assert row["selected_in_subspace"] is None
    assert client.calls == 2


@pytest.mark.parametrize(
    ("client", "reason", "calls"),
    [
        (None, "llm_unavailable", 0),
        (_ChemClient(raises=True), "llm_error", 1),
        (_ChemClient(status="error", error="down"), "llm_error", 1),
    ],
)
def test_chem_transport_fallbacks_select_same_index_as_gpbo(
    client: _ChemClient | None, reason: str, calls: int
) -> None:
    engine = _chem_engine()
    mean, std = _posterior_for(engine)
    mean[-1] += 2.0
    engine._surrogate = _ChemPosterior(mean, std)
    engine._client = client
    expected_index = _direct_gp_index(engine, mean, std)

    row = engine.step()

    assert row["query_index"] == expected_index
    assert row["guidance_status"] == "fallback"
    assert row["guidance_reason"] == reason
    assert (client.calls if client else 0) == calls


def _illegal_suzuki_pair(engine: ChemLGBOEngine) -> tuple[str, str]:
    legal = set(
        engine.test_df[["Electrophile", "Nucleophile"]].itertuples(
            index=False, name=None
        )
    )
    for electrophile in engine.options_json["Electrophile"]:
        for nucleophile in engine.options_json["Nucleophile"]:
            if (electrophile, nucleophile) not in legal:
                return electrophile, nucleophile
    raise AssertionError("Suzuki pool unexpectedly contains the full Cartesian product")


@pytest.mark.parametrize(
    "case",
    ["empty_intersection", "already_queried_only", "uninformative_full_pool"],
)
def test_chem_mask_fallbacks_select_same_index_as_gpbo(case: str) -> None:
    engine = _chem_engine()
    if case == "empty_intersection":
        electrophile, nucleophile = _illegal_suzuki_pair(engine)
        subspace = {
            "Electrophile": [electrophile],
            "Nucleophile": [nucleophile],
        }
    elif case == "already_queried_only":
        subspace = {
            field: [str(engine.test_df[field].iloc[0])]
            for field in engine.feature_cols
        }
        raw_mask = build_subspace_mask(engine.test_df[engine.feature_cols], subspace)
        engine.queried.update(np.flatnonzero(raw_mask).tolist())
    else:
        subspace = {"Ligand": list(engine.options_json["Ligand"])}

    mean, std = _posterior_for(engine)
    mean[-1] += 2.0
    engine._surrogate = _ChemPosterior(mean, std)
    client = _ChemClient("Thinking\n" + json.dumps({"subspace": subspace}))
    engine._client = client
    expected_index = _direct_gp_index(engine, mean, std)

    row = engine.step()

    assert row["query_index"] == expected_index
    assert row["predicted_yield"] == pytest.approx(mean[expected_index])
    assert row["guidance_status"] == "fallback"
    assert row["guidance_reason"] == case
    assert client.calls == 2


def _partial_subspace(
    engine: ChemLGBOEngine, *, exclude: set[int] | None = None
) -> dict[str, list[str]]:
    exclude = exclude or set()
    for field in engine.feature_cols:
        for value in engine.options_json[field]:
            mask = build_subspace_mask(
                engine.test_df[engine.feature_cols], {field: [value]}
            )
            remaining_hits = set(np.flatnonzero(mask)) - exclude
            if remaining_hits and len(remaining_hits) < engine.M - len(exclude):
                return {field: [value]}
    raise AssertionError("expected a proper non-empty subspace")

def test_chem_react_recovers_missing_tool_call() -> None:
    engine = _chem_engine()
    accepted = _partial_subspace(engine)
    client = _ChemClient(
        responses=[
            SimpleNamespace(
                status="success", content="", error=None, usage={}, tool_calls=None
            ),
            _tool_result(json.dumps({"subspace": accepted}), call_id="recovered"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    _, _, diagnostics, _ = engine._llm_mean_shift(
        _ChemPosterior(mean, std), mean, std
    )

    assert diagnostics["guidance_reason"] == "accepted"
    assert client.calls == 2
    assert client.messages[1][-1]["role"] == "user"
    assert engine.guidance_artifacts[-1]["react_first_reason"] == "empty_response"



def test_chem_react_recovers_unknown_value_and_records_artifact() -> None:
    engine = _chem_engine()
    accepted = _partial_subspace(engine)
    first = '{"subspace":{"Ligand":["not-an-option"]}}'
    second = json.dumps({"subspace": accepted})
    client = _ChemClient(
        responses=[
            _tool_result(first, call_id="bad-call"),
            _tool_result(second, call_id="good-call"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    shifted, _, diagnostics, _ = engine._llm_mean_shift(
        _ChemPosterior(mean, std), mean, std
    )

    assert client.calls == 2
    assert diagnostics["guidance_reason"] == "accepted"
    assert not np.array_equal(shifted, mean)
    artifact = engine.guidance_artifacts[-1]
    assert artifact == {
        **artifact,
        "raw_response": second,
        "parser_reason": "accepted",
        "react_retried": True,
        "react_first_reason": "unknown_value",
        "llm_attempts": 2,
        "tool_call_id": "good-call",
    }


def test_chem_react_recovers_already_queried_only() -> None:
    engine = _chem_engine()
    queried_subspace = {
        field: [str(engine.test_df[field].iloc[0])] for field in engine.feature_cols
    }
    queried_mask = build_subspace_mask(
        engine.test_df[engine.feature_cols], queried_subspace
    )
    engine.queried.update(np.flatnonzero(queried_mask).tolist())
    accepted = _partial_subspace(engine, exclude=engine.queried)
    client = _ChemClient(
        responses=[
            _tool_result(json.dumps({"subspace": queried_subspace})),
            _tool_result(json.dumps({"subspace": accepted}), call_id="recovered"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    _, _, diagnostics, mask = engine._llm_mean_shift(
        _ChemPosterior(mean, std), mean, std
    )

    assert diagnostics["guidance_reason"] == "accepted"
    assert mask is not None and np.any(mask)
    assert engine.guidance_artifacts[-1]["react_first_reason"] == "already_queried_only"


def test_chem_react_recovers_uninformative_full_pool() -> None:
    engine = _chem_engine()
    accepted = _partial_subspace(engine)
    client = _ChemClient(
        responses=[
            _tool_result(json.dumps({"subspace": {"Ligand": engine.options_json["Ligand"]}})),
            _tool_result(json.dumps({"subspace": accepted}), call_id="partial"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    _, _, diagnostics, _ = engine._llm_mean_shift(
        _ChemPosterior(mean, std), mean, std
    )

    assert diagnostics["guidance_reason"] == "accepted"
    assert engine.guidance_artifacts[-1]["react_first_reason"] == "uninformative_full_pool"


def test_chem_react_second_failure_falls_back_once_with_final_reason() -> None:
    engine = _chem_engine()
    client = _ChemClient(
        responses=[
            _tool_result("not-json", call_id="first"),
            _tool_result('{"subspace":{"Ligand":["missing"]}}', call_id="second"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    shifted, _, diagnostics, mask = engine._llm_mean_shift(
        _ChemPosterior(mean, std), mean, std
    )

    np.testing.assert_array_equal(shifted, mean)
    assert mask is None
    assert diagnostics["guidance_reason"] == "unknown_value"
    assert len(engine.guidance_artifacts) == 1
    artifact = engine.guidance_artifacts[0]
    assert artifact["parser_reason"] == "unknown_value"
    assert artifact["react_first_reason"] == "invalid_json"
    assert artifact["react_retried"] is True
    assert artifact["llm_attempts"] == 2
    assert artifact["tool_call_id"] == "second"


def test_chem_retry_uses_matching_tool_call_id_and_stays_local() -> None:
    engine = _chem_engine()
    engine.trajectory.append({"condition": {"marker": "trajectory"}, "observed_yield": 1.0})
    trajectory_before = list(engine.trajectory)
    accepted = _partial_subspace(engine)
    client = _ChemClient(
        responses=[
            _tool_result('{"subspace":{"Ligand":["missing"]}}', call_id="retry-me"),
            _tool_result(json.dumps({"subspace": accepted}), call_id="final"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    engine._llm_mean_shift(_ChemPosterior(mean, std), mean, std)

    retry_messages = client.messages[1]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-2]["tool_calls"][0]["id"] == "retry-me"  # type: ignore[index]
    assert retry_messages[-1]["role"] == "tool"
    assert retry_messages[-1]["tool_call_id"] == "retry-me"
    assert json.loads(str(retry_messages[-1]["content"]))["reason"] == "unknown_value"
    assert engine.trajectory == trajectory_before
    assert len(client.messages[0]) == 2
    assert all(message.get("role") not in {"assistant", "tool"} for message in client.messages[0])

def test_chem_retry_without_tool_call_id_uses_user_feedback() -> None:
    engine = _chem_engine()
    accepted = _partial_subspace(engine)
    missing_id = _tool_result('{"subspace":{"Ligand":["missing"]}}')
    del missing_id.tool_calls[0]["id"]
    client = _ChemClient(
        responses=[
            missing_id,
            _tool_result(json.dumps({"subspace": accepted}), call_id="final"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    engine._llm_mean_shift(_ChemPosterior(mean, std), mean, std)

    retry_messages = client.messages[1]
    assert retry_messages[-1]["role"] == "user"
    assert all(message.get("role") != "tool" for message in retry_messages)


def test_chem_retry_multiple_tool_calls_uses_user_feedback() -> None:
    engine = _chem_engine()
    accepted = _partial_subspace(engine)
    multiple = _tool_result('{"subspace":{"Ligand":["missing"]}}')
    multiple.tool_calls.append(
        {
            "id": "extra",
            "type": "function",
            "function": {
                "name": "propose_sparse_subspace",
                "arguments": json.dumps({"subspace": accepted}),
            },
        }
    )
    client = _ChemClient(
        responses=[
            multiple,
            _tool_result(json.dumps({"subspace": accepted}), call_id="final"),
        ]
    )
    engine._client = client
    mean, std = _posterior_for(engine)

    engine._llm_mean_shift(_ChemPosterior(mean, std), mean, std)

    retry_messages = client.messages[1]
    assert retry_messages[-1]["role"] == "user"
    assert all(message.get("role") != "tool" for message in retry_messages)


def test_chem_forces_tool_call_and_defaults_temperature() -> None:
    engine = _chem_engine()
    assert engine.llm_temperature == pytest.approx(0.2)
    accepted = _partial_subspace(engine)
    client = _ChemClient(json.dumps({"subspace": accepted}))
    engine._client = client
    mean, std = _posterior_for(engine)

    engine._llm_mean_shift(_ChemPosterior(mean, std), mean, std)

    assert client.kwargs[0]["temperature"] == pytest.approx(0.2)
    extra_body = client.kwargs[0]["extra_body"]
    assert extra_body["tools"] == [chem_lgbo.PROPOSE_SUBSPACE_TOOL]  # type: ignore[index]
    assert extra_body["tool_choice"]["function"]["name"] == "propose_sparse_subspace"  # type: ignore[index]


def test_chem_guidance_is_invariant_to_unqueried_oracle_yield() -> None:
    engines = [_chem_engine(n_counterfactuals=5), _chem_engine(n_counterfactuals=5)]
    engines[1].test_df.loc[:, engines[1].target_col] = np.linspace(
        -10_000.0, 10_000.0, engines[1].M
    )
    engines[1].pool_yield = engines[1].test_df[
        engines[1].target_col
    ].to_numpy()
    value = str(engines[0].test_df["Ligand"].iloc[0])
    mean, std = _posterior_for(engines[0])
    outputs = []
    for engine in engines:
        client = _ChemClient(_proposal("Ligand", [value]))
        engine._client = client
        output = engine._llm_mean_shift(
            _ChemPosterior(mean, std), mean.copy(), std.copy()
        )
        outputs.append((output, client.messages, engine.guidance_artifacts))

    np.testing.assert_array_equal(outputs[0][0][0], outputs[1][0][0])
    np.testing.assert_array_equal(outputs[0][0][3], outputs[1][0][3])
    assert outputs[0][0][2] == outputs[1][0][2]
    assert outputs[0][1] == outputs[1][1]
    assert outputs[0][2][0]["counterfactual_indices"] == outputs[1][2][0][
        "counterfactual_indices"
    ]


def test_chem_engine_end_to_end_three_real_gp_steps() -> None:
    engine = ChemLGBOEngine(
        "suzuki",
        seed=100,
        use_llm=False,
        n_iters=3,
        n_restarts=0,
        backend="sklearn",
    )
    value = str(engine.test_df["Ligand"].iloc[0])
    client = _ChemClient(_proposal("Ligand", [value]))
    engine.use_llm = True
    engine._client = client

    trajectory = engine.run()

    indices = [row["query_index"] for row in trajectory]
    assert len(indices) == len(set(indices)) == 3
    assert client.calls == 3
    assert all(row["guidance_status"] == "applied" for row in trajectory)
    for row in trajectory:
        index = row["query_index"]
        assert row["observed_yield"] == pytest.approx(engine.pool_yield[index])
        assert row["condition"] == {
            field: str(engine.test_df[field].iloc[index])
            for field in engine.feature_cols
        }
