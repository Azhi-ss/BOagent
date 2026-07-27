"""Tests for experiment protocol validation in result aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

from analyze import aggregate_results


_METRICS = {
    "best_found": 80.0,
    "initial_round_found_best": 60.0,
    "t95": 10,
    "AUC_best_so_far": 70.0,
}


def _result(protocol: str) -> dict[str, object]:
    return {
        "composition": "gpbo_ei",
        "dataset": "buchwald_sub4",
        "seed": 100,
        "prior_protocol": protocol,
        "metrics": dict(_METRICS),
    }


def test_aggregate_rejects_mixed_prior_protocols() -> None:
    results = [_result("legacy_full_prior"), _result("seeded_subsample")]

    with pytest.raises(ValueError, match="prior_protocol"):
        aggregate_results(results)


def test_aggregate_accepts_single_prior_protocol() -> None:
    summary = aggregate_results([_result("seeded_subsample")])

    assert summary["gpbo_ei"]["buchwald_sub4"]["best_found"]["mean"] == 80.0
