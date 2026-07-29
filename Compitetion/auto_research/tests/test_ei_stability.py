from __future__ import annotations

import sys
from pathlib import Path

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import numpy as np
from scipy.stats import norm

from components.library import _ei
from engine import _score_statistics


def _mirror_seed1800_step20() -> np.ndarray:
    """Reconstruct the lgbo_dkl/buchwald_sub4/seed1800 step20 EI array shape.

    The diagnostic record stored only aggregate stats: max=1.821902921587371e-11,
    std=6.506790785236766e-13, 37/783 nonzero. Reverse-engineering shows the
    distribution is a single spike (one candidate at max, the rest ~0): a lone
    spike of height h among N zeros has std = h/sqrt(N) = 1.822e-11/sqrt(783) =
    6.511e-13, matching the recorded std to 4 sig figs. So step20's EI is
    sparse-but-valid: argmax correctly selected idx 554 (the lone improvable
    point, observed yield 73.16), not an arbitrary tie.
    """
    arr = np.zeros(783, dtype=float)
    arr[554] = 1.821902921587371e-11
    return arr


def test_small_scale_ei_is_not_flagged_degenerate():
    """A sparse-but-valid EI array (one real spike among zeros) must not be
    flagged degenerate — argmax still locates the lone improvable candidate.

    This is the seed1800 false positive: the absolute 1e-12 constant-threshold
    flagged std=6.5e-13 as "constant", but the array has a real, argmax-meaningful
    spike at idx 554. Degeneracy must be judged by relative spread (std/max ~3.6%
    here), not absolute magnitude.
    """
    ei = _mirror_seed1800_step20()
    stats = _score_statistics(ei)
    assert stats["nonfinite_count"] == 0
    assert not stats["is_degenerate"], (
        f"Sparse-but-valid spike (std/max={stats['std'] / stats['max']:.3%}, "
        f"std={stats['std']:.3e}, max={stats['max']:.3e}) was flagged degenerate "
        f"by the absolute 1e-12 threshold — the seed1800 validation-gate false "
        f"positive. argmax correctly selected the lone spike."
    )


def test_truly_constant_scores_still_flagged_degenerate():
    """A genuinely constant score array (zero spread) must remain degenerate —
    the relative-threshold relaxation must not let a real degenerate case
    through."""
    constant = np.full(100, 5.0)
    stats = _score_statistics(constant)
    assert stats["is_degenerate"], "A truly constant array must stay degenerate."


def test_ei_recovers_true_high_improvement_candidate():
    """A candidate clearly above best_f must rank highest (monotone sanity check)."""
    mu = np.array([80.0, 85.0, 90.0])
    sigma = np.array([1.0, 1.0, 1.0])
    best_f = 82.0
    ei = _ei(mu, sigma, best_f, 0.01)
    assert np.argmax(ei) == 2
    assert np.all(ei >= 0)
