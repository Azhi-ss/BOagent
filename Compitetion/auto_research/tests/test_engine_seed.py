"""Tests for the fixed competition prior in HybridEngine."""
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


def test_buchwald_uses_all_fixed_initial_observations() -> None:
    engine = HybridEngine(_gpbo(), "buchwald_sub4", seed=100, n_iters=1)

    assert engine.initial_indices == tuple(range(35))
    assert len(engine.train_df) == 35
    assert engine.encoder.dim == 32
    assert engine.pool_X.shape == (783, 32)


def test_seed_does_not_change_fixed_initial_observations() -> None:
    first = HybridEngine(_gpbo(), "buchwald_sub4", seed=100, n_iters=1)
    second = HybridEngine(_gpbo(), "buchwald_sub4", seed=200, n_iters=1)

    assert first.initial_indices == second.initial_indices == tuple(range(35))
    assert first.train_df.equals(second.train_df)


def test_suzuki_uses_all_fixed_initial_observations() -> None:
    engine = HybridEngine(_gpbo(), "suzuki", seed=100, n_iters=1)

    assert engine.initial_indices == tuple(range(29))
    assert len(engine.train_df) == 29
    assert engine.encoder.dim == 35
