"""Component library: reference algorithm building blocks.

All components operate on the competition's fully-categorical one-hot pool.
No file writes outside auto_research/; evaluation is read-only on references/.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy.stats import norm

from .protocol import (
    StepContext,
    register_acquisition,
    register_llm,
    register_selector,
    register_surrogate,
)

# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------

@register_surrogate("botorch_matern")
def _surrogate_botorch(backend: str, seed: int, **kw: Any) -> Any:
    from bo_core.optimization.surrogate import create_surrogate

    return create_surrogate(
        "botorch",
        seed=seed,
        n_restarts=kw.get("n_restarts", 10),
        alpha=kw.get("alpha", 1e-2),
    )


@register_surrogate("sklearn_matern")
def _surrogate_sklearn(backend: str, seed: int, **kw: Any) -> Any:
    from bo_core.optimization.surrogate import create_surrogate

    return create_surrogate(
        "sklearn",
        seed=seed,
        n_restarts=kw.get("n_restarts", 10),
        alpha=kw.get("alpha", 1e-2),
    )


@register_surrogate("botorch_manifold")
def _surrogate_botorch_manifold(backend: str, seed: int, **kw: Any) -> Any:
    from components.kernel_manifold import create_manifold_surrogate

    return create_manifold_surrogate(
        backend,
        seed=seed,
        alpha=kw.get("alpha", 1e-2),
        evolve_interval=kw.get("evolve_interval", 5),
        kernel_library=kw.get("kernel_library"),
    )


@register_surrogate("botorch_alas")
def _surrogate_botorch_alas(backend: str, seed: int, **kw: Any) -> Any:
    from components.alas import create_alas_surrogate

    return create_alas_surrogate(
        backend,
        seed=seed,
        alpha=kw.get("alpha", 1e-2),
        mode=kw.get("mode", "alas"),
        init_alpha=kw.get("init_alpha", 1.5),
    )


@register_surrogate("botorch_dkl")
def _surrogate_botorch_dkl(backend: str, seed: int, **kw: Any) -> Any:
    from components.dkl import create_dkl_surrogate

    return create_dkl_surrogate(
        backend,
        seed=seed,
        alpha=kw.get("alpha", 1e-2),
        hidden_dim=kw.get("hidden_dim", 16),
        n_layers=kw.get("n_layers", 2),
    )


@register_surrogate("botorch_cake")
def _surrogate_botorch_cake(backend: str, seed: int, **kw: Any) -> Any:
    from components.cake import create_cake_surrogate

    return create_cake_surrogate(
        backend,
        seed=seed,
        alpha=kw.get("alpha", 1e-2),
        population_size=kw.get("population_size", 6),
        evolve_interval=kw.get("evolve_interval", 5),
        chat_engine=kw.get("chat_engine", "deepseek-v4-flash"),
        reasoning_effort=kw.get("reasoning_effort", "low"),
    )


# ---------------------------------------------------------------------------
# Acquisitions (analytic EI/UCB/PI over pool)
# ---------------------------------------------------------------------------

def _ei(mu: np.ndarray, sigma: np.ndarray, best_f: float, xi: float) -> np.ndarray:
    imp = mu - best_f - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def _ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float) -> np.ndarray:
    return mu + kappa * sigma


def _pi(mu: np.ndarray, sigma: np.ndarray, best_f: float, xi: float) -> np.ndarray:
    imp = mu - best_f - xi
    z = imp / sigma
    return norm.cdf(z)


@register_acquisition("ei")
def _acq_ei(surrogate: Any, pool_X: np.ndarray, best_f: float, ctx: StepContext) -> np.ndarray:
    mu, sigma = surrogate.predict(pool_X)
    return _ei(mu, sigma, best_f, ctx.extra.get("xi", 0.01))


@register_acquisition("ucb")
def _acq_ucb(surrogate: Any, pool_X: np.ndarray, best_f: float, ctx: StepContext) -> np.ndarray:
    mu, sigma = surrogate.predict(pool_X)
    return _ucb(mu, sigma, ctx.extra.get("kappa", 2.576))


@register_acquisition("pi")
def _acq_pi(surrogate: Any, pool_X: np.ndarray, best_f: float, ctx: StepContext) -> np.ndarray:
    mu, sigma = surrogate.predict(pool_X)
    return _pi(mu, sigma, best_f, ctx.extra.get("xi", 0.01))


@register_acquisition("posterior_mean")
def _acq_pmean(surrogate: Any, pool_X: np.ndarray, best_f: float, ctx: StepContext) -> np.ndarray:
    mu, _ = surrogate.predict(pool_X)
    return mu


@register_acquisition("thompson")
def _acq_ts(surrogate: Any, pool_X: np.ndarray, best_f: float, ctx: StepContext) -> np.ndarray:
    """Simple Thompson sampling over the discrete pool."""
    mu, sigma = surrogate.predict(pool_X)
    rng = np.random.RandomState(ctx.extra.get("seed", 0) + ctx.iteration)
    return rng.normal(mu, np.maximum(sigma, 1e-9))


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

@register_selector("argmax")
def _sel_argmax(scores: np.ndarray, ctx: StepContext) -> int:
    mask = np.ones(len(scores), dtype=bool)
    for q in ctx.queried:
        if 0 <= q < len(scores):
            mask[q] = False
    masked = np.where(mask, scores, -np.inf)
    return int(np.argmax(masked))


@register_selector("softmax_explore")
def _sel_softmax(scores: np.ndarray, ctx: StepContext) -> int:
    mask = np.ones(len(scores), dtype=bool)
    for q in ctx.queried:
        if 0 <= q < len(scores):
            mask[q] = False
    masked = np.where(mask, scores, -np.inf)
    # Temperature decays with remaining iterations: more explore early.
    tau = max(0.1, 2.0 * ctx.remaining / max(ctx.n_iters, 1))
    z = masked - np.max(masked[np.isfinite(masked)])
    p = np.exp(z / tau)
    p[~mask] = 0.0
    p = p / p.sum()
    rng = np.random.RandomState(ctx.extra.get("seed", 0) + ctx.iteration)
    return int(rng.choice(len(scores), p=p))


# ---------------------------------------------------------------------------
# LLM Strategies
# ---------------------------------------------------------------------------

@register_llm("none")
def _llm_none(ctx: StepContext) -> None:
    return None


@register_llm("lgbo_mean_shift")
def _llm_lgbo(ctx: StepContext) -> dict[str, Any] | None:
    """LGBO: LLM proposes point+confidence, GP mean-shift."""
    from bo_core.llm_client import DeepSeekClient
    from bo_core.optimization.lgbo_parser import parse_llm_response
    from bo_core.optimization.lgbo_prompt import (
        DatasetMeta,
        build_system_prompt,
        build_user_prompt,
    )

    if not ctx.extra.get("use_llm", False):
        return None

    client = ctx.extra.get("_client")
    if client is None:
        client = DeepSeekClient.from_env()
        client.model = ctx.extra.get("chat_engine", "deepseek-v4-flash")
        client.timeout_s = 120
        ctx.extra["_client"] = client
    if not client.is_configured():
        return None

    meta = DatasetMeta(
        dataset=ctx.extra["dataset"],
        feature_cols=ctx.feature_cols,
        options=ctx.options,
        target_name=ctx.extra.get("target_col", "Yield"),
    )
    system = build_system_prompt(meta)
    user = build_user_prompt(meta, ctx.history, ctx.extra.get("prev_thinking"))
    try:
        result = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=8192,
            extra_body={"reasoning_effort": ctx.extra.get("reasoning_effort", "low")},
        )
    except Exception:  # noqa: BLE001 - LLM call is best-effort
        return None
    if getattr(result, "status", None) != "success" or not result.content:
        return None

    parsed = parse_llm_response(result.content, ctx.feature_cols, ctx.options)
    if parsed is None:
        return None
    _mode, values, confidence = parsed
    return {
        "action": "mean_shift",
        "point": dict(zip(ctx.feature_cols, values)),
        "confidence": confidence,
        "thinking": ctx.extra.get("prev_thinking"),
    }


@register_llm("lmabo_adaptive_acq")
def _llm_lmabo(ctx: StepContext) -> dict[str, Any] | None:
    """lmabo: LLM selects acquisition function from portfolio based on GP state."""
    from bo_core.llm_client import DeepSeekClient

    if not ctx.extra.get("use_llm", False):
        return None

    client = ctx.extra.get("_client")
    if client is None:
        client = DeepSeekClient.from_env()
        client.model = ctx.extra.get("chat_engine", "deepseek-v4-flash")
        client.timeout_s = 120
        ctx.extra["_client"] = client
    if not client.is_configured():
        return None

    # Build GP-state summary (lmabo style)
    acq_choices = ["EI", "UCB", "PI", "PosteriorMean", "Thompson"]
    prompt = f"""You are an expert in Bayesian Optimization. Current state:
- Iteration: {ctx.iteration}/{ctx.n_iters} (remaining: {ctx.remaining})
- Observed N: {len(ctx.history)}
- Best so far: {ctx.best_f:.2f}
- Dataset: {ctx.extra.get('dataset', 'unknown')}
- Feature dims: {len(ctx.feature_cols)} categorical (one-hot encoded)

Available acquisition functions: {', '.join(acq_choices)}

Select the acquisition function that will best balance exploration/exploitation
for the next step. Avoid repeating the same function if it failed to improve recently.

Respond strictly with one word from the list above."""
    try:
        result = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=32,
        )
    except Exception:  # noqa: BLE001 - LLM call is best-effort
        return None
    if getattr(result, "status", None) != "success" or not result.content:
        return None

    choice = result.content.strip().split()[0].strip('.,;:!?"')
    if choice not in acq_choices:
        choice = "EI"  # fallback
    return {"action": "switch_acq", "acq_type": choice}


@register_llm("bora_adaptive")
def _llm_bora(ctx: StepContext) -> dict[str, Any] | None:
    """BORA: plateau detection -> switch between BO-only / LLM-only / hybrid."""
    # Plateau detection: no improvement in last `window` iterations.
    window = ctx.extra.get("plateau_window", 5)
    recent = [y for _, y in ctx.history[-window:]]
    plateau = len(recent) >= window and max(recent) <= ctx.best_f + 1e-6

    uncertainty = ctx.extra.get("avg_sigma", 1.0)
    high_unc = uncertainty > ctx.extra.get("uncertainty_threshold", 0.5)

    if plateau or high_unc:
        # a2: full LLM intervention (exploration)
        return {"action": "llm_pick"}
    # a1: vanilla BO (exploitation)
    return {"action": "bo_pick"}


@register_llm("llm_in_loop_pick")
def _llm_llm_in_loop(ctx: StepContext) -> dict[str, Any] | None:
    """LLM-in-the-Loop: LLM picks top candidates from the pool directly."""
    from bo_core.llm_client import DeepSeekClient

    if not ctx.extra.get("use_llm", False):
        return None

    client = ctx.extra.get("_client")
    if client is None:
        client = DeepSeekClient.from_env()
        client.model = ctx.extra.get("chat_engine", "deepseek-v4-flash")
        client.timeout_s = 120
        ctx.extra["_client"] = client
    if not client.is_configured():
        return None

    # Build candidate summary from unqueried pool
    unqueried = [i for i in range(len(ctx.extra["pool_conditions"])) if i not in ctx.queried]
    if not unqueried:
        return None
    # Sample up to 20 for prompt
    rng = random.Random(ctx.extra.get("seed", 0) + ctx.iteration)
    sample_idx = rng.sample(unqueried, min(20, len(unqueried)))
    lines = []
    for i in sample_idx:
        cond = ctx.extra["pool_conditions"][i]
        cond_str = ", ".join(f"{k}={v}" for k, v in cond.items())
        lines.append(f"  {i}: {cond_str}")
    candidates_text = "\n".join(lines)

    hist_text = "\n".join(
        f"  {c} -> {y:.2f}" for c, y in ctx.history[-10:]
    )

    prompt = f"""You are optimizing a chemical reaction. Recent observations:
{hist_text}

Unexplored candidate conditions (index: conditions):
{candidates_text}

Pick the single most promising candidate index to try next.
Return ONLY the integer index, no explanation."""
    try:
        result = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=16,
        )
    except Exception:  # noqa: BLE001 - LLM call is best-effort
        return None
    if getattr(result, "status", None) != "success" or not result.content:
        return None

    try:
        idx = int(result.content.strip().split()[0])
    except (ValueError, IndexError):
        return None
    if idx not in unqueried:
        return None
    return {"action": "pool_pick", "pool_index": idx}
