"""Tests for seed-controlled initial observations in HybridEngine."""
from __future__ import annotations

import sys
from pathlib import Path

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import components.library  # noqa: F401
from compositions.base import get_base_compositions
from engine import HybridEngine


def _gpbo() -> object:
    return next(comp for comp in get_base_compositions() if comp.name == "gpbo_ei")


def test_same_seed_selects_same_initial_observations() -> None:
    first = HybridEngine(_gpbo(), "buchwald_sub4", seed=100, n_iters=1, n_initial=5)
    second = HybridEngine(_gpbo(), "buchwald_sub4", seed=100, n_iters=1, n_initial=5)

    assert first.initial_indices == second.initial_indices
    assert len(first.initial_indices) == 5


def test_different_seeds_select_different_initial_observations() -> None:
    first = HybridEngine(_gpbo(), "buchwald_sub4", seed=100, n_iters=1, n_initial=5)
    second = HybridEngine(_gpbo(), "buchwald_sub4", seed=200, n_iters=1, n_initial=5)

    assert first.initial_indices != second.initial_indices


def test_none_uses_full_legacy_prior() -> None:
    engine = HybridEngine(_gpbo(), "buchwald_sub4", seed=100, n_iters=1, n_initial=None)

    assert engine.initial_indices == tuple(range(35))
    assert len(engine.train_df) == 35
