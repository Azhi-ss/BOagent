#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pool-based discrete Bayesian optimization with BoTorch.

For benchmarks where the feasible set is a finite candidate pool (e.g. the
Buchwald discrete reaction-condition grid), it is both faster and cleaner to
fit a single GP per round and evaluate the acquisition *directly on the pool*
in one batch, instead of running BoTorch's multi-start continuous acquisition
optimizer over a one-hot relaxation. This module implements that fast path:

  - one-hot encode the 4 categorical variables over the pool,
  - fit a ``SingleTaskGP`` with a capped MLE budget,
  - batch-evaluate ``qLogExpectedImprovement`` over the whole pool and pick the
    best not-yet-queried candidate(s).

Compared with ``BOModel.gen`` (Ax ``qLogNoisyExpectedImprovement`` + continuous
multi-start optimization) this avoids the expensive acquisition optimization and
the invalid/duplicate-candidate snapping fallback.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

SURROGATES = ("single_task", "saas")


class OneHotEncoder:
    """One-hot encode a tuple of categorical values across fixed option lists."""

    def __init__(self, options_per_var: Sequence[Sequence[str]]):
        self.option_lists = [list(o) for o in options_per_var]
        self.index_maps = [
            {v: i for i, v in enumerate(opts)} for opts in self.option_lists
        ]
        self.dim = sum(len(o) for o in self.option_lists)

    def encode(self, keys: Sequence[Sequence[str]]) -> torch.Tensor:
        n = len(keys)
        X = torch.zeros(n, self.dim, dtype=torch.double)
        col = 0
        for j, idx_map in enumerate(self.index_maps):
            width = len(idx_map)
            for r, key in enumerate(keys):
                X[r, col + idx_map[key[j]]] = 1.0
            col += width
        return X


class PoolBO:
    """Greedy pool BO with qLogEI acquisition.

    ``surrogate`` selects the GP used to model the objective:
      - ``"single_task"`` (default): ``SingleTaskGP`` + Matern/RBF kernel, fit by
        capped MLE with random restarts. Fast, but treats every one-hot
        dimension equally — prone to over-fitting on high-dim sparse inputs.
      - ``"saas"``: ``SaasFullyBayesianSingleTaskGP`` (SAASBO), fit by NUTS MCMC.
        Puts a horseshoe prior on lengthscales so irrelevant dimensions are
        automatically shrunk. Better on high-dim one-hot + small sample, at the
        cost of slower fitting.
    """

    def __init__(
        self,
        pool_keys: Sequence[Sequence[str]],
        options_per_var: Sequence[Sequence[str]],
        param_names: Sequence[str] | None = None,
        maxiter: int = 50,
        n_restarts: int = 5,
        surrogate: str = "single_task",
        mcmc_warmup: int = 128,
        mcmc_samples: int = 64,
        mcmc_thinning: int = 8,
        device: str = "cpu",
    ) -> None:
        if surrogate not in SURROGATES:
            raise ValueError(f"surrogate must be one of {SURROGATES}, got {surrogate!r}")
        self.encoder = OneHotEncoder(options_per_var)
        self.pool_keys = list(pool_keys)
        self.param_names = list(param_names) if param_names is not None else None
        self.X_pool = self.encoder.encode(self.pool_keys)
        self.maxiter = maxiter
        self.n_restarts = n_restarts
        self.surrogate = surrogate
        self.mcmc_warmup = mcmc_warmup
        self.mcmc_samples = mcmc_samples
        self.mcmc_thinning = mcmc_thinning
        self.device = torch.device(device)
        self.X_pool = self.X_pool.to(self.device)

    def _fit_gp(self, X: torch.Tensor, Y: torch.Tensor):
        """Fit the configured surrogate on (X, Y) and return it.

        For ``single_task``: capped MLE with ``n_restarts`` random lengthscale
        inits (seed-dependent). For ``saas``: NUTS MCMC with horseshoe prior.
        """
        if self.surrogate == "saas":
            return self._fit_saas(X, Y)
        return self._fit_single_task(X, Y)

    def _fit_single_task(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        """Fit SingleTaskGP with a capped MLE budget and random restarts.

        Random restarts (random lengthscale initialization, driven by the
        caller's torch RNG / seed) make the fit seed-dependent so different
        experiment seeds produce different trajectories instead of identical
        deterministic ones. We keep the restart with the best marginal
        likelihood.
        """
        n_restarts = self.n_restarts
        best_gp: SingleTaskGP | None = None
        best_mll = -float("inf")
        for _ in range(n_restarts):
            gp = SingleTaskGP(X, Y)
            # Random lengthscale init so restarts (and seeds) diverge. The
            # default SingleTaskGP covar_module here is a plain RBFKernel.
            with torch.no_grad():
                cov = gp.covar_module
                cov.lengthscale = torch.rand_like(cov.lengthscale) * (self.encoder.dim ** 0.5)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={
                    "options": {"maxiter": self.maxiter, "ftol": 1e-3, "gtol": 1e-3}
                },
            )
            with torch.no_grad():
                val = float(mll(gp.likelihood(gp(X)), Y).sum())
            if val > best_mll:
                best_mll = val
                best_gp = gp
        return best_gp  # type: ignore[return-value]

    def _fit_saas(self, X: torch.Tensor, Y: torch.Tensor) -> SaasFullyBayesianSingleTaskGP:
        """Fit SAASBO GP via NUTS MCMC with a horseshoe prior on lengthscales.

        The horseshoe prior pushes most dimensions' lengthscales large
        (effectively ignoring them), keeping only the relevant subset active.
        This is the key difference from ``SingleTaskGP`` on high-dim one-hot
        inputs where most dimensions carry no signal.
        """
        gp = SaasFullyBayesianSingleTaskGP(X, Y)
        fit_fully_bayesian_model_nuts(
            gp,
            warmup_steps=self.mcmc_warmup,
            num_samples=self.mcmc_samples,
            thinning=self.mcmc_thinning,
            disable_progbar=True,
        )
        return gp

    def _pool_mean_std(self, gp) -> tuple[np.ndarray, np.ndarray]:
        """Return marginal (mean, std) over the whole pool, robust to GP type.

        ``SingleTaskGP`` posterior has shape [N, 1]. ``SaasFullyBayesianSingleTaskGP``
        returns a mixture posterior [N, S, 1, 1] over S MCMC samples; we marginalize
        over the sample axis using the law of total variance:
            Var = E[Var_s] + Var[E_s]
        so the reported std reflects both model uncertainty and MCMC spread.
        """
        with torch.no_grad():
            posterior = gp.posterior(self.X_pool)
            mean = posterior.mean  # [N, 1] or [N, S, 1, 1]
            var = posterior.variance
            if mean.dim() > 2:
                # Mixture case: collapse the MCMC-sample axis (dim 1 here).
                # mean: [N, S, 1, 1] -> [N, 1]
                m = mean.squeeze(-1).squeeze(-1)  # [N, S]
                v = var.squeeze(-1).squeeze(-1)   # [N, S]
                marginal_mean = m.mean(dim=-1)             # [N]
                marginal_var = v.mean(dim=-1) + m.var(dim=-1, unbiased=False)  # [N]
                means = marginal_mean.detach().cpu().numpy().reshape(-1)
                stds = marginal_var.clamp_min(0.0).sqrt().detach().cpu().numpy().reshape(-1)
            else:
                means = mean.detach().cpu().numpy().reshape(-1)
                stds = var.clamp_min(0.0).sqrt().detach().cpu().numpy().reshape(-1)
        return means, stds

    def rank_pool(
        self,
        observed_keys: Sequence[Sequence[str]],
        observed_y: Sequence[float],
        queried_pool_mask: np.ndarray,
        k: int = 5,
    ) -> tuple[list[dict], "object"]:
        """Fit GP once, score the whole pool by qLogEI, return top-k unqueried.

        Returns (top_k_records, gp). Each record is::
            {"index", "params" (dict), "mean", "std", "ei", "role"}
        sorted by EI descending. ``role`` is "explore" if std is large else
        "exploit". The fitted ``gp`` is returned so callers can reuse it.
        """
        X_obs = self.encoder.encode(list(observed_keys)).to(self.device)
        Y = torch.tensor(
            list(observed_y), dtype=torch.double, device=self.device
        ).unsqueeze(-1)
        gp = self._fit_gp(X_obs, Y)
        best_f = float(Y.max())

        with torch.no_grad():
            acq = qLogExpectedImprovement(gp, best_f=torch.tensor(best_f))
            ei = acq(self.X_pool.unsqueeze(-2)).detach().cpu().numpy().reshape(-1)
            means, stds = self._pool_mean_std(gp)

        std_arr = np.asarray(stds, dtype=float)
        std_threshold = float(np.quantile(std_arr[~queried_pool_mask], 0.75)) if (~queried_pool_mask).any() else 0.0
        ei_masked = np.where(queried_pool_mask, -np.inf, ei)
        order = np.argsort(-ei_masked)[:k]

        records: list[dict] = []
        for idx in order:
            if not np.isfinite(ei_masked[idx]):
                continue
            records.append({
                "index": int(idx),
                "params": ({n: self.pool_keys[idx][i] for i, n in enumerate(self.param_names)} if self.param_names else {f"var_{j}": v for j, v in enumerate(self.pool_keys[idx])}),
                "mean": float(means[idx]),
                "std": float(stds[idx]),
                "ei": float(ei[idx]),
                "role": "explore" if stds[idx] >= std_threshold else "exploit",
            })
        return records, gp

    def suggest(
        self,
        observed_keys: Sequence[Sequence[str]],
        observed_y: Sequence[float],
        queried_pool_mask: np.ndarray,
        q: int = 1,
    ) -> list[int]:
        """Return indices (into the pool) of the top-q unqueried candidates."""
        X_obs = self.encoder.encode(list(observed_keys)).to(self.device)
        Y = torch.tensor(
            list(observed_y), dtype=torch.double, device=self.device
        ).unsqueeze(-1)
        gp = self._fit_gp(X_obs, Y)
        best_f = float(Y.max())

        chosen: list[int] = []
        mask = queried_pool_mask.copy()
        with torch.no_grad():
            for _ in range(q):
                acq = qLogExpectedImprovement(gp, best_f=torch.tensor(best_f))
                vals = acq(self.X_pool.unsqueeze(-2))  # (N, 1)
                vals = vals.detach().cpu().numpy().reshape(-1)
                vals[mask] = -np.inf
                pick = int(np.argmax(vals))
                chosen.append(pick)
                mask[pick] = True
                # Greedy: update best_f with the model's marginal posterior
                # mean at the pick (keeps acquisition coherent for the next
                # greedy step). Works for both SingleTaskGP and the SAAS
                # mixture posterior (averages over MCMC samples).
                pick_mean = self._point_marginal_mean(gp, pick)
                best_f = max(best_f, pick_mean)
        return chosen

    def _point_marginal_mean(self, gp, pool_idx: int) -> float:
        """Marginal posterior mean at a single pool point, robust to GP type."""
        with torch.no_grad():
            posterior = gp.posterior(self.X_pool[pool_idx].unsqueeze(0))
            mean = posterior.mean
            if mean.dim() > 2:
                # mixture [1, S, 1, 1] -> average over samples
                return float(mean.mean())
            return float(mean)
