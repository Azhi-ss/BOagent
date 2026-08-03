from __future__ import annotations

import json

import pytest
from bo_core.optimization.chem_lgbo_parser import parse_subspace_response

FEATURES = ["Reactant2", "Ligand", "Additive", "Base"]
OPTIONS = {
    "Reactant2": ["R1", "R2"],
    "Ligand": ["L1", "L2"],
    "Additive": [
        "3-phenyl-1,2-oxazole",
        "dicyclohexyl-[2-[2,4,6-tri(propan-2-yl)phenyl]phenyl]phosphane",
    ],
    "Base": ["B1", "B2"],
}


def test_accepts_sparse_final_json_after_rationale() -> None:
    text = (
        'Thinking: I also considered {"subspace":{"Ligand":["L2"]}}.\n'
        'Final Answer:\n{"subspace":{"Ligand":["L1"],"Base":["B1","B2"]}}'
    )

    subspace, reason = parse_subspace_response(text, FEATURES, OPTIONS)

    assert reason == "accepted"
    assert subspace == {"Ligand": ["L1"], "Base": ["B1", "B2"]}


def test_preserves_literal_values_with_commas_and_brackets() -> None:
    tricky = OPTIONS["Additive"][1]
    text = json.dumps({"subspace": {"Additive": [tricky]}})

    subspace, reason = parse_subspace_response(text, FEATURES, OPTIONS)

    assert reason == "accepted"
    assert subspace == {"Additive": [tricky]}


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty_response"),
        ("   \n\t", "empty_response"),
        ("no JSON here", "invalid_json"),
        ('{"subspace":{"Ligand":["L1"]}', "invalid_json"),
        ('{"subspace":{"Ligand":["L1"]}} trailing', "invalid_json"),
        (
            '{"subspace":{"Ligand":["L1"]}}\n{"subspace":',
            "invalid_json",
        ),
        ('{"subspace":{"Ligand":["L1"]},"mode":"point"}', "invalid_schema"),
        ('{"mode":"point"}', "invalid_schema"),
        ('{"subspace":{}}', "invalid_schema"),
        ('{"subspace":[]}', "invalid_schema"),
        ('{"subspace":{"Unknown":["x"]}}', "unknown_field"),
        ('{"subspace":{"Ligand":"L1"}}', "invalid_schema"),
        ('{"subspace":{"Ligand":[1]}}', "invalid_schema"),
        ('{"subspace":{"Ligand":[]}}', "empty_choice"),
        ('{"subspace":{"Ligand":["L1","L1"]}}', "duplicate_value"),
        ('{"subspace":{"Ligand":["not-an-option"]}}', "unknown_value"),
    ],
)
def test_rejects_invalid_response_with_exact_reason(text: str, reason: str) -> None:
    assert parse_subspace_response(text, FEATURES, OPTIONS) == (None, reason)
