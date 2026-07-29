#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-04 10:03:29
@File: src/bo/models.py
@IDE: vscode
@Description:
    BO model
"""

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

import numpy as np
from ax.core.observation import ObservationFeatures


class BOModel:
    def __init__(self, experiment):
        self.experiment = experiment
        self.model_bridge = None

    def gen(self, n):
        print(
            f"Start using BO algorithms to generate bo_recommendations candidates..."
        )
        self.model_bridge = Models.BOTORCH_MODULAR(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(botorch_model_class=SingleTaskGP),
                ]
            ),
            botorch_acqf_class=qLogNoisyExpectedImprovement,
        )

        print("Done!\n")
        return self.model_bridge.gen(n=n)

    def predict_posterior(self, candidates_params):
        """Return GP posterior (mean, std) for each candidate parameterization.

        Requires ``self.model_bridge`` to be fitted — either by calling ``gen``
        on this object or by assigning an already-fitted bridge (e.g. one
        created via ``Models.BOTORCH_MODULAR``) to ``self.model_bridge``.
        Returns a list of ``{"params", "mean", "std"}`` dicts; ``None`` if the
        model is not yet fitted so callers can degrade gracefully.
        """
        if self.model_bridge is None:
            return [
                {"params": p, "mean": None, "std": None}
                for p in candidates_params
            ]
        obs_feats = [
            ObservationFeatures(parameters=p) for p in candidates_params
        ]
        f, cov = self.model_bridge.predict(obs_feats)
        metric = list(f.keys())[0]
        means = np.atleast_1d(f[metric])
        variances = np.atleast_1d(cov[metric][metric])
        results = []
        for p, m, v in zip(candidates_params, means, variances):
            results.append(
                {
                    "params": p,
                    "mean": float(m),
                    "std": float(np.sqrt(max(v, 0.0))),
                }
            )
        return results
