"""Composition mutator: swap components to create new variants."""
from __future__ import annotations

from typing import Any

from components.protocol import Composition, list_components
import components.library  # noqa: F401  # force registration


def mutate_composition(comp: Composition, slot: str, new_component: str, **param_overrides: Any) -> Composition:
    """Create a new composition by swapping one component."""
    params = dict(comp.params)
    params.update(param_overrides)
    return Composition(
        name=f"{comp.name}_{slot}_{new_component}",
        surrogate=new_component if slot == "surrogate" else comp.surrogate,
        acquisition=new_component if slot == "acquisition" else comp.acquisition,
        selector=new_component if slot == "selector" else comp.selector,
        llm_strategy=new_component if slot == "llm_strategy" else comp.llm_strategy,
        params=params,
    )


def generate_neighbors(comp: Composition, max_neighbors: int = 5) -> list[Composition]:
    """Generate single-component-swap neighbors of a composition."""
    comps = list_components()
    neighbors: list[Composition] = []
    current = {
        "surrogate": comp.surrogate,
        "acquisition": comp.acquisition,
        "selector": comp.selector,
        "llm_strategy": comp.llm_strategy,
    }
    slot_map = {
        "surrogate": "surrogates",
        "acquisition": "acquisitions",
        "selector": "selectors",
        "llm_strategy": "llm_strategies",
    }
    for slot, comp_key in slot_map.items():
        for candidate in comps[comp_key]:
            if candidate != current[slot]:
                neighbors.append(mutate_composition(comp, slot, candidate))
                if len(neighbors) >= max_neighbors:
                    return neighbors
    return neighbors


def crossover(a: Composition, b: Composition, name: str | None = None) -> Composition:
    """Combine components from two compositions."""
    import random

    rng = random.Random(42)
    return Composition(
        name=name or f"{a.name}_x_{b.name}",
        surrogate=rng.choice([a.surrogate, b.surrogate]),
        acquisition=rng.choice([a.acquisition, b.acquisition]),
        selector=rng.choice([a.selector, b.selector]),
        llm_strategy=rng.choice([a.llm_strategy, b.llm_strategy]),
        params={**a.params, **b.params},
    )
