"""Base compositions: one per reference method family."""
from __future__ import annotations

from components.protocol import Composition

BASE_COMPOSITIONS: list[Composition] = [
    # GPBO baseline (no LLM)
    Composition(
        name="gpbo_ei",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False, "xi": 0.01},
    ),
    # H1: GPBO with kernel manifold evolution (no LLM)
    Composition(
        name="gpbo_manifold",
        surrogate="botorch_manifold",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False, "xi": 0.01, "evolve_interval": 5},
    ),
    # H2: GPBO with ALAS learnable alpha-stable kernel (no LLM)
    Composition(
        name="gpbo_alas",
        surrogate="botorch_alas",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False, "xi": 0.01, "mode": "alas", "init_alpha": 1.5},
    ),
    # H3: GPBO with Deep Kernel Learning (no LLM)
    Composition(
        name="gpbo_dkl",
        surrogate="botorch_dkl",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False, "xi": 0.01, "hidden_dim": 16, "n_layers": 2},
    ),
    # H4: GPBO with CAKE LLM kernel evolution (LLM at kernel structure)
    Composition(
        name="gpbo_cake",
        surrogate="botorch_cake",
        acquisition="ei",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False, "xi": 0.01, "evolve_interval": 5, "population_size": 6},
    ),
    Composition(
        name="gpbo_ucb",
        surrogate="botorch_matern",
        acquisition="ucb",
        selector="argmax",
        llm_strategy="none",
        params={"use_llm": False, "kappa": 2.576},
    ),
    # H6: CAKE kernel evolution + LGBO mean-shift (both LLM entry points on)
    Composition(
        name="lgbo_cake",
        surrogate="botorch_cake",
        acquisition="ei",
        selector="argmax",
        llm_strategy="lgbo_mean_shift",
        params={"use_llm": True, "xi": 0.01, "evolve_interval": 5, "population_size": 6},
    ),
    # LGBO (current champion)
    Composition(
        name="lgbo_mean_shift",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="lgbo_mean_shift",
        params={"use_llm": True, "xi": 0.01},
    ),
    # lmabo: adaptive acquisition
    Composition(
        name="lmabo_adaptive",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="lmabo_adaptive_acq",
        params={"use_llm": True, "xi": 0.01},
    ),
    # BORA: plateau-triggered LLM
    Composition(
        name="bora_adaptive",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="bora_adaptive",
        params={"use_llm": True, "plateau_window": 5},
    ),
    # LLM-in-the-Loop: direct pool pick
    Composition(
        name="llm_in_loop",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="argmax",
        llm_strategy="llm_in_loop_pick",
        params={"use_llm": True},
    ),
    # Hybrid candidates (mixed components)
    Composition(
        name="lgbo_softmax",
        surrogate="botorch_matern",
        acquisition="ei",
        selector="softmax_explore",
        llm_strategy="lgbo_mean_shift",
        params={"use_llm": True, "xi": 0.01},
    ),
    Composition(
        name="lmabo_ucb",
        surrogate="botorch_matern",
        acquisition="ucb",
        selector="argmax",
        llm_strategy="lmabo_adaptive_acq",
        params={"use_llm": True, "kappa": 2.576},
    ),
]


def get_base_compositions() -> list[Composition]:
    return list(BASE_COMPOSITIONS)
