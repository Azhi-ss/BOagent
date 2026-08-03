from __future__ import annotations

import json

import pytest
from bo_core.optimization.lgbo_parser import parse_llm_response

FEATURES = ["Reactant2", "Ligand", "Additive", "Base"]
OPTIONS = {
    "Reactant2": ["1-bromo-4-ethylbenzene", "1-chloro-4-ethylbenzene", "1-ethyl-4-iodobenzene"],
    "Ligand": ["L1", "L2"],
    # Additive names deliberately contain commas and brackets (the hard case).
    "Additive": ["3-phenyl-1,2-oxazole", "dicyclohexyl-[2-[2,4,6-tri(propan-2-yl)phenyl]phenyl]phosphane"],
    "Base": ["B1", "B2"],
}


def _point(values, c):
    return json.dumps({"mode": "point", "values": values, "confidence": c})


def test_valid_point_with_prose():
    text = (
        "Thinking: Bromide is more reactive than chloride for oxidative addition; "
        "history supports this. Choosing point.\n"
        "Final Answer:\n"
        + _point(["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"], 0.7)
    )
    res = parse_llm_response(text, FEATURES, OPTIONS)
    assert res is not None
    mode, values, c = res
    assert mode == "point"
    assert values == ["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"]
    assert c == pytest.approx(0.7)


def test_values_with_commas_and_brackets_parse_correctly():
    """The key robustness case: IUPAC names containing commas/brackets."""
    tricky = "dicyclohexyl-[2-[2,4,6-tri(propan-2-yl)phenyl]phenyl]phosphane"
    text = "Final Answer:\n" + _point(
        ["1-chloro-4-ethylbenzene", "L2", tricky, "B2"], 0.55
    )
    res = parse_llm_response(text, FEATURES, OPTIONS)
    assert res is not None
    _, values, c = res
    assert values[2] == tricky
    assert c == pytest.approx(0.55)


def test_region_lb_equals_ub_is_accepted():
    text = json.dumps(
        {
            "mode": "region",
            "lb": ["1-ethyl-4-iodobenzene", "L1", "3-phenyl-1,2-oxazole", "B1"],
            "ub": ["1-ethyl-4-iodobenzene", "L1", "3-phenyl-1,2-oxazole", "B1"],
            "confidence": 0.4,
        }
    )
    res = parse_llm_response(text, FEATURES, OPTIONS)
    assert res is not None
    mode, values, c = res
    assert mode == "region"
    assert values == ["1-ethyl-4-iodobenzene", "L1", "3-phenyl-1,2-oxazole", "B1"]
    assert c == pytest.approx(0.4)


def test_region_with_lb_not_equal_ub_rejected():
    # A "range" over categorical variables is meaningless -> None.
    text = json.dumps(
        {
            "mode": "region",
            "lb": ["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"],
            "ub": ["1-chloro-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"],
            "confidence": 0.4,
        }
    )
    assert parse_llm_response(text, FEATURES, OPTIONS) is None


def test_missing_confidence_rejected():
    text = json.dumps({"mode": "point", "values": ["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"]})
    assert parse_llm_response(text, FEATURES, OPTIONS) is None


def test_confidence_clamped_to_unit_interval():
    text = _point(["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"], 1.5)
    _, _, c = parse_llm_response(text, FEATURES, OPTIONS)
    assert c == 1.0
    text2 = _point(["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"], -0.3)
    _, _, c2 = parse_llm_response(text2, FEATURES, OPTIONS)
    assert c2 == 0.0


def test_invalid_category_rejected():
    text = _point(["1-bromo-4-ethylbenzene", "L1", "this-additive-does-not-exist", "B1"], 0.5)
    assert parse_llm_response(text, FEATURES, OPTIONS) is None


def test_wrong_value_count_rejected():
    text = _point(["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole"], 0.5)  # d=3 != 4
    assert parse_llm_response(text, FEATURES, OPTIONS) is None


def test_malformed_json_rejected():
    text = 'Final Answer:\n{"mode": "point", "values": ["1-bromo-4-ethylbenzene", "L1", '  # truncated
    assert parse_llm_response(text, FEATURES, OPTIONS) is None


def test_picks_last_json_when_thinking_contains_braces():
    # Thinking mentions a stray {"foo": 1}; the Final Answer (last JSON) wins.
    text = (
        'Thinking: I considered {"foo": 1} as a counterfactual.\n'
        "Final Answer:\n"
        + _point(["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"], 0.6)
    )
    res = parse_llm_response(text, FEATURES, OPTIONS)
    assert res is not None
    assert res[1] == ["1-bromo-4-ethylbenzene", "L1", "3-phenyl-1,2-oxazole", "B1"]


def test_empty_or_garbage_rejected():
    assert parse_llm_response("", FEATURES, OPTIONS) is None
    assert parse_llm_response("I cannot help with that.", FEATURES, OPTIONS) is None
