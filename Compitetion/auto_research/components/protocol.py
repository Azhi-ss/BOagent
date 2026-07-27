"""Component protocol: pluggable BO algorithm building blocks.

A BO method is decomposed into 4 slots. Each slot has multiple implementations
extracted from references/. A Composition wires one implementation per slot.

Slots
-----
SurrogateFactory   : fit(X_obs, y_obs) -> SurrogateModel
AcquisitionFactory : score(surrogate, pool_X, best_f, state) -> np.ndarray (M,)
Selector           : select(scores, state) -> int (pool index)
LLMStrategy        : maybe_llm(history, state) -> Optional[LLMDecision]

LLMDecision is one of:
  - None                (no LLM action this round)
  - {"acq_type": str}   (lmabo: switch acquisition function)
  - {"point": dict, "confidence": float}  (LGBO: mean-shift)
  - {"pool_indices": list[int]}           (LLM-in-the-Loop: pick from pool)
  - {"action": str}     (BORA: a1=BO, a2=LLM, a3=hybrid)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


@dataclass
class StepContext:
    """Per-step state passed between components."""
    iteration: int
    n_iters: int
    feature_cols: list[str]
    options: dict[str, list[str]]
    history: list[tuple[dict[str, str], float]]  # (condition, yield)
    queried: set[int]
    best_f: float
    remaining: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Composition:
    """One concrete BO method = 4 component choices."""
    name: str
    surrogate: str          # component id
    acquisition: str        # component id
    selector: str           # component id
    llm_strategy: str       # component id
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.name}: sur={self.surrogate} acq={self.acquisition} "
            f"sel={self.selector} llm={self.llm_strategy}"
        )


class SurrogateFactory(Protocol):
    def __call__(
        self, backend: str, seed: int, **kwargs: Any
    ) -> Any: ...  # returns SurrogateModel


class AcquisitionFactory(Protocol):
    def __call__(
        self, surrogate: Any, pool_X: np.ndarray, best_f: float, ctx: StepContext
    ) -> np.ndarray: ...


class Selector(Protocol):
    def __call__(self, scores: np.ndarray, ctx: StepContext) -> int: ...


class LLMStrategy(Protocol):
    def __call__(self, ctx: StepContext) -> dict[str, Any] | None: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SURROGATES: dict[str, SurrogateFactory] = {}
ACQUISITIONS: dict[str, AcquisitionFactory] = {}
SELECTORS: dict[str, Selector] = {}
LLM_STRATEGIES: dict[str, LLMStrategy] = {}


def register_surrogate(name: str) -> Callable[[SurrogateFactory], SurrogateFactory]:
    def deco(fn: SurrogateFactory) -> SurrogateFactory:
        SURROGATES[name] = fn
        return fn
    return deco


def register_acquisition(name: str) -> Callable[[AcquisitionFactory], AcquisitionFactory]:
    def deco(fn: AcquisitionFactory) -> AcquisitionFactory:
        ACQUISITIONS[name] = fn
        return fn
    return deco


def register_selector(name: str) -> Callable[[Selector], Selector]:
    def deco(fn: Selector) -> Selector:
        SELECTORS[name] = fn
        return fn
    return deco


def register_llm(name: str) -> Callable[[LLMStrategy], LLMStrategy]:
    def deco(fn: LLMStrategy) -> LLMStrategy:
        LLM_STRATEGIES[name] = fn
        return fn
    return deco


def list_components() -> dict[str, list[str]]:
    return {
        "surrogates": sorted(SURROGATES),
        "acquisitions": sorted(ACQUISITIONS),
        "selectors": sorted(SELECTORS),
        "llm_strategies": sorted(LLM_STRATEGIES),
    }
