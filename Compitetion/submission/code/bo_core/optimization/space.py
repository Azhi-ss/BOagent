from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc


class SearchSpace(ABC):
    """Abstract interface for the search space of an optimization task."""
    
    @abstractmethod
    def get_unobserved(self, observed_configs: pd.DataFrame) -> pd.DataFrame:
        """Return a pool of candidate points that have not been observed yet."""

    @property
    @abstractmethod
    def feature_cols(self) -> list[str]:
        pass

class DiscreteSearchSpace(SearchSpace):
    """A search space defined by a fixed pool of candidates (e.g. from a CSV/Excel)."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        self.df = df
        self._feature_cols = feature_cols
        self.all_x = df[feature_cols].values.astype(float)

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    def get_unobserved(self, observed_configs: pd.DataFrame) -> pd.DataFrame:
        if observed_configs.empty:
            return self.df.copy()
        
        obs_values = observed_configs[self.feature_cols].values.astype(float)
        
        # Vectorized check for observed points
        # self.all_x is (N, D), obs_values is (M, D)
        # We want mask of size N where True if row in self.all_x is in obs_values
        
        # For small M, we can use a loop over M and vectorize over N
        is_observed = np.zeros(len(self.all_x), dtype=bool)
        for obs_row in obs_values:
            # Check which rows in all_x are close to this obs_row
            matches = np.all(np.isclose(self.all_x, obs_row, rtol=1e-5, atol=1e-8), axis=1)
            is_observed |= matches
            
        return self.df[~is_observed].copy()

class ContinuousSearchSpace(SearchSpace):
    """A search space defined by continuous variable bounds."""
    
    def __init__(self, variables: list[dict[str, Any]], n_samples: int = 2000, seed: int = 42):
        self.variables = variables
        self.n_samples = n_samples
        self.seed = seed
        self._feature_cols = [v["name"] for v in variables]

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    def get_unobserved(self, observed_configs: pd.DataFrame) -> pd.DataFrame:
        """Sample the continuous space using simplex Dirichlet sampling for mixture proportion variables,
        or LHS for standard unconstrained continuous variables.
        """
        d = len(self.variables)
        
        # Check if the variables represent a mixture (proportion ratios summing to 1)
        # Typically detected by name starting with 'x_'
        is_mixture = d > 1 and all(v["name"].startswith("x_") for v in self.variables)
        
        if is_mixture:
            rng = np.random.RandomState(self.seed)
            # Sample uniformly from the flat simplex using symmetric Dirichlet / Exponential normalization
            samples = rng.exponential(scale=1.0, size=(self.n_samples, d))
            samples /= samples.sum(axis=1, keepdims=True)
            pool_df = pd.DataFrame(samples, columns=self.feature_cols)
        else:
            sampler = qmc.LatinHypercube(d=d, seed=self.seed)
            sample = sampler.random(n=self.n_samples)
            l_bounds = [v["min"] for v in self.variables]
            u_bounds = [v["max"] for v in self.variables]
            scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
            pool_df = pd.DataFrame(scaled_sample, columns=self.feature_cols)
        
        if observed_configs.empty:
            return pool_df

        return pool_df
