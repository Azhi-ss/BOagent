"""Tests for seed completeness and edge-case validation in runners."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import components.library  # noqa: F401
from compositions.base import get_base_compositions
from engine import HybridEngine


def _gpbo():
    return next(c for c in get_base_compositions() if c.name == "gpbo_ei")


def test_legacy_n_initial_argument_is_rejected() -> None:
    with pytest.raises(TypeError, match="n_initial"):
        HybridEngine(
            _gpbo(),
            "buchwald_sub4",
            seed=100,
            n_iters=1,
            n_initial=5,
        )


def test_same_fixed_prior_across_different_compositions() -> None:
    """Cross-method fairness: every method receives the same fixed prior."""
    manifold = next(c for c in get_base_compositions() if c.name == "gpbo_manifold")
    cake = next(c for c in get_base_compositions() if c.name == "gpbo_cake")

    e1 = HybridEngine(manifold, "buchwald_sub4", seed=100, n_iters=1)
    e2 = HybridEngine(cake, "buchwald_sub4", seed=100, n_iters=1)

    assert e1.initial_indices == e2.initial_indices == tuple(range(35))


def test_hybrid_lgbo_manifold_and_dkl_are_registered() -> None:
    """P0/P1 orthogonal hybrids: strong surrogate × LGBO mean-shift."""
    by_name = {c.name: c for c in get_base_compositions()}

    manifold = by_name["lgbo_manifold"]
    assert manifold.surrogate == "botorch_manifold"
    assert manifold.llm_strategy == "lgbo_mean_shift"
    assert manifold.params.get("use_llm") is True
    assert manifold.params.get("evolve_interval") == 5

    dkl = by_name["lgbo_dkl"]
    assert dkl.surrogate == "botorch_dkl"
    assert dkl.llm_strategy == "lgbo_mean_shift"
    assert dkl.params.get("use_llm") is True
    assert dkl.params.get("hidden_dim") == 16
    assert dkl.params.get("n_layers") == 2


def test_hybrid_params_reach_surrogate() -> None:
    """Composition params (hidden_dim / evolve_interval) must reach the surrogate."""
    by_name = {c.name: c for c in get_base_compositions()}

    eng_dkl = HybridEngine(by_name["lgbo_dkl"], "buchwald_sub4", seed=100, n_iters=1)
    assert eng_dkl.surrogate.hidden_dim == 16
    assert eng_dkl.surrogate.n_layers == 2

    eng_man = HybridEngine(by_name["lgbo_manifold"], "buchwald_sub4", seed=100, n_iters=1)
    assert eng_man.surrogate.evolve_interval == 5


def test_aggregate_rejects_empty_results() -> None:
    from analyze import aggregate_results

    with pytest.raises(ValueError, match="empty"):
        aggregate_results([])


def test_seed_completeness_passes_when_all_seeds_present() -> None:
    from analyze import assert_seed_completeness

    results = [
        {"composition": "gpbo_ei", "dataset": "buchwald_sub4", "seed": 100},
        {"composition": "gpbo_ei", "dataset": "buchwald_sub4", "seed": 200},
        {"composition": "gpbo_ei", "dataset": "suzuki", "seed": 100},
        {"composition": "gpbo_ei", "dataset": "suzuki", "seed": 200},
    ]
    assert_seed_completeness(results, [100, 200], ["buchwald_sub4", "suzuki"])


def test_seed_completeness_fails_when_seed_missing() -> None:
    from analyze import assert_seed_completeness

    results = [
        {"composition": "gpbo_ei", "dataset": "buchwald_sub4", "seed": 100},
        {"composition": "gpbo_ei", "dataset": "buchwald_sub4", "seed": 200},
        {"composition": "gpbo_ei", "dataset": "suzuki", "seed": 100},
        # suzuki seed 200 missing - simulates a failed run
    ]
    with pytest.raises(ValueError, match="suzuki missing seeds"):
        assert_seed_completeness(results, [100, 200], ["buchwald_sub4", "suzuki"])
