"""Tests for experiment protocol validation in result aggregation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

from analyze import GLOBAL_BEST, N_ITERS, _normalize_metric, aggregate_results


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


def test_normalize_uses_ci95_once_for_yield_lcb() -> None:
    """ci95 is already 1.96*se; LCB must be mean - ci95, not mean - 1.96*ci95."""
    mean = 86.0
    ci95 = 2.0
    gbest = GLOBAL_BEST["buchwald_sub4"]
    expected = (mean - ci95) / gbest
    got = _normalize_metric("best_found", mean, ci95, "buchwald_sub4")
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
    # Guard against the old double-count formula.
    buggy = mean / gbest - 1.96 * ci95 / gbest
    assert not math.isclose(got, buggy, rel_tol=0.0, abs_tol=1e-9)


def test_normalize_uses_ci95_once_for_t95() -> None:
    mean = 20.0
    ci95 = 4.0
    expected = 1.0 - mean / N_ITERS - ci95 / N_ITERS
    got = _normalize_metric("t95", mean, ci95, "suzuki")
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
