from __future__ import annotations

import json

import pandas as pd
from bo_core.optimization.chem_lgbo_prompt import (
    PreviousGuidanceOutcome,
    build_compatibility_pairs,
    build_system_prompt,
    build_user_prompt,
)
from bo_core.optimization.lgbo_prompt import DatasetMeta


def _meta() -> DatasetMeta:
    return DatasetMeta(
        dataset="suzuki",
        feature_cols=["Electrophile", "Nucleophile", "Ligand", "Base"],
        options={
            "Electrophile": ["E1", "E2"],
            "Nucleophile": ["N1", "N2"],
            "Ligand": ["L1", "L2"],
            "Base": ["B1", "B2"],
        },
    )


def _candidate_features(yields: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Electrophile": ["E1", "E2", "E1"],
            "Nucleophile": ["N1", "N2", "N1"],
            "Ligand": ["L1", "L2", "L2"],
            "Base": ["B1", "B2", "B2"],
            "Yield": yields,
        }
    )


def test_compatibility_pairs_are_candidate_rows_in_first_occurrence_order() -> None:
    pairs = build_compatibility_pairs(
        _candidate_features([1.0, 2.0, 3.0]),
        ("Electrophile", "Nucleophile"),
    )

    assert pairs == [("E1", "N1"), ("E2", "N2")]
    assert ("E1", "N2") not in pairs
    assert ("E2", "N1") not in pairs


def test_compatibility_pairs_ignore_yield_and_missing_fields() -> None:
    low = _candidate_features([0.0, 0.0, 0.0])
    high = _candidate_features([99.9, 98.8, 97.7])

    assert build_compatibility_pairs(
        low, ("Electrophile", "Nucleophile")
    ) == build_compatibility_pairs(high, ("Electrophile", "Nucleophile"))
    assert build_compatibility_pairs(low, ("missing", "Nucleophile")) == []


def test_prompts_define_sparse_guidance_without_candidate_yield_leakage() -> None:
    meta = _meta()
    features = _candidate_features([99.9, 98.8, 97.7])
    pairs = {
        ("Electrophile", "Nucleophile"): build_compatibility_pairs(
            features, ("Electrophile", "Nucleophile")
        )
    }

    system = build_system_prompt(meta)
    user = build_user_prompt(
        meta,
        history=[
            (
                {
                    "Electrophile": "E2",
                    "Nucleophile": "N2",
                    "Ligand": "L2",
                    "Base": "B2",
                },
                42.5,
            )
        ],
        compatibility_pairs=pairs,
        prev_thinking="Test a less hindered ligand.",
    )

    assert '{"subspace":{"Ligand":["L1"],"Base":["B1","B2"]}}' in system
    assert "omitted fields are unconstrained" in system
    assert "non-empty array" in system
    assert "Do not emit Point, Region, bounds, or confidence" in system
    assert "guides the posterior mean" in system
    assert "does not hard-filter" in system
    assert "E1" in user and "N1" in user and "L1" in user and "B1" in user
    assert "(E1, N1)" in user and "(E2, N2)" in user
    assert "(E1, N2)" not in user and "(E2, N1)" not in user
    assert "99.9000" not in user and "98.8000" not in user and "97.7000" not in user
    assert "Yield=42.5000" in user
    assert "Test a less hindered ligand." in user


def test_user_prompt_is_invariant_to_unqueried_candidate_yields() -> None:
    meta = _meta()
    pair = ("Electrophile", "Nucleophile")
    prompts = []
    for yields in ([0.0, 0.0, 0.0], [99.9, 98.8, 97.7]):
        pairs = {pair: build_compatibility_pairs(_candidate_features(yields), pair)}
        prompts.append(build_user_prompt(meta, [], pairs))

    assert prompts[0] == prompts[1]


def test_default_prompts_remain_byte_identical_to_v1() -> None:
    meta = _meta()

    assert build_system_prompt(meta) == (
        "You are a senior chemist optimizing Suzuki-Miyaura C-C coupling for Yield.\n"
        "Use mechanism and the observed history to propose a promising sparse categorical subspace.\n"
        "Choose one or more declared feature fields; omitted fields are unconstrained.\n"
        "Each chosen field maps to a non-empty array of exact literal option strings.\n"
        "Final Answer must contain exactly this JSON shape and nothing after it:\n"
        '{"subspace":{"Ligand":["L1"],"Base":["B1","B2"]}}\n'
        "Do not emit Point, Region, bounds, or confidence. Do not emit empty arrays.\n"
        "The suggestion guides the posterior mean; it does not hard-filter the experiment pool.\n"
        "Compatibility pairs are feature-only hints from the unlabeled candidate pool, not validation rules."
    )
    assert build_user_prompt(meta, [], {}) == (
        "[Background]\n"
        "- Reaction: Suzuki-Miyaura C-C coupling\n"
        "- Mechanism: Pd-catalyzed coupling of an aryl electrophile (Electrophile) with an aryl boron nucleophile (Nucleophile) to form a C-C bond. Product fixed: 6-(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)quinoline. Catalyst fixed: palladium(2+) diacetate. Ligand, Base, and Solvent govern transmetalation rate and protodeboronation side paths; Electrophile leaving group (Cl/Br/I/OTf) and Nucleophile form (boronic acid/ester/BF3K) trade reactivity and stability.\n"
        "- Literal options:\n"
        "  - Electrophile: ['E1', 'E2']\n"
        "  - Nucleophile: ['N1', 'N2']\n"
        "  - Ligand: ['L1', 'L2']\n"
        "  - Base: ['B1', 'B2']\n"
        "\n[Review]\n"
        "- Historical observations: (none yet)\n"
        "- Previous thinking: (none)"
    )


def test_treatment_prompt_renders_no_previous_outcome_without_changing_schema() -> None:
    meta = _meta()

    system = build_system_prompt(meta, outcome_feedback=True)
    user = build_user_prompt(meta, [], {}, outcome_feedback=True)

    assert "one selected point, not proof" in system
    assert "selected point was outside" in system
    assert "do not discard it merely because one point failed" in system
    assert "Return only the required JSON" in system
    assert '[Previous guidance outcome]\n- (none)' in user
    assert '{"subspace":{"Ligand":["L1"],"Base":["B1","B2"]}}' in system


def test_treatment_prompt_renders_stable_previous_guidance_outcome() -> None:
    outcome = PreviousGuidanceOutcome(
        proposed_subspace={"Ligand": ["L2", "L1"], "Base": ["B2"]},
        selected_condition={
            "Electrophile": "E2",
            "Nucleophile": "N1",
            "Ligand": "L2",
            "Base": "B2",
        },
        selected_in_subspace=True,
        observed_yield=42.1,
        incumbent_before=68.9,
    )

    user = build_user_prompt(
        _meta(),
        [],
        {},
        previous_outcome=outcome,
        outcome_feedback=True,
    )

    assert json.dumps(outcome.proposed_subspace, sort_keys=True) in user
    assert json.dumps(outcome.selected_condition, sort_keys=True) in user
    assert "Selected point was inside the proposed subspace: true" in user
    assert "Observed Yield: 42.1000" in user
    assert "Incumbent before this trial: 68.9000" in user
    assert "Improvement over incumbent: -26.8000" in user


def test_treatment_prompt_preserves_outside_subspace_semantics() -> None:
    outcome = PreviousGuidanceOutcome(
        proposed_subspace={"Ligand": ["L1"]},
        selected_condition={
            "Electrophile": "E1",
            "Nucleophile": "N1",
            "Ligand": "L2",
            "Base": "B1",
        },
        selected_in_subspace=False,
        observed_yield=70.0,
        incumbent_before=68.0,
    )

    user = build_user_prompt(
        _meta(), [], {}, previous_outcome=outcome, outcome_feedback=True
    )

    assert "Selected point was inside the proposed subspace: false" in user
    assert "Improvement over incumbent: 2.0000" in user
