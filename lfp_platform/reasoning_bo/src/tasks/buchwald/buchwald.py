#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discrete chemistry BO metric (offline lookup-table oracle).

Generic version of the Buchwald oracle: given a tuple of categorical reaction
variables, returns the recorded Yield (%) from a labeled test CSV. Combos absent
from the oracle yield NaN so the runner can fall back to a valid pool candidate.
Optionally merges labeled train rows into the yield map so the GP seed trial
gets finite targets instead of NaN.

``BuchwaldMetric`` is kept as a thin subclass with the Buchwald_sub4 variable
names defaulted, for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws

BUCHWALD_PARAM_NAMES = ("Reactant2", "Ligand", "Additive", "Base")


def _key_of(params: dict[str, Any], param_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(params[n]) for n in param_names)


class DiscreteChemMetric(Metric):
    """Lookup-table oracle over a labeled discrete chemistry test set."""

    def __init__(
        self,
        name: str,
        param_names: Sequence[str],
        *,
        test_csv: str | Path,
        train_csv: str | Path | None = None,
        target_column: str = "Yield",
        noiseless: bool = True,
        lower_is_better: bool = False,
    ) -> None:
        self.param_names = tuple(param_names)
        self.noiseless = noiseless
        self.target_column = target_column
        self.test_csv_path = Path(test_csv)
        self.train_csv_path = Path(train_csv) if train_csv is not None else None
        df = pd.read_csv(self.test_csv_path)
        self.oracle_df = df.reset_index(drop=True)
        keys = list(zip(*(df[c].astype(str) for c in self.param_names)))
        # query_index map is test-only (only test rows are valid query targets).
        self.index_map: dict[tuple[str, ...], int] = {
            k: i for i, k in enumerate(keys)
        }
        # yield lookup includes test rows plus any labeled train rows so the
        # GP seed trial (train combos) gets finite targets instead of NaN.
        self.yield_map: dict[tuple[str, ...], float] = dict(
            zip(keys, df[self.target_column].astype(float).tolist())
        )
        if self.train_csv_path is not None:
            train_df = pd.read_csv(self.train_csv_path)
            for _, row in train_df.iterrows():
                k = tuple(str(row[c]) for c in self.param_names)
                if k not in self.yield_map:
                    self.yield_map[k] = float(row[self.target_column])
        super().__init__(name=name, lower_is_better=lower_is_better)

    def clone(self) -> "DiscreteChemMetric":
        return self.__class__(
            name=self._name,
            param_names=self.param_names,
            test_csv=self.test_csv_path,
            train_csv=self.train_csv_path,
            target_column=self.target_column,
            noiseless=self.noiseless,
            lower_is_better=none_throws(self.lower_is_better),
        )

    def fetch_trial_data(
        self, trial: BaseTrial, **kwargs: Any
    ) -> MetricFetchResult:
        try:
            records = []
            for arm_name, arm in trial.arms_by_name.items():
                key = _key_of(arm.parameters, self.param_names)
                y = self.yield_map.get(key, float("nan"))
                records.append(
                    {
                        "arm_name": arm_name,
                        "metric_name": self.name,
                        "mean": y,
                        "sem": 0.0 if self.noiseless else None,
                        "trial_index": trial.index,
                    }
                )
            return Ok(Data(df=pd.DataFrame.from_records(records)))
        except Exception as e:
            return Err(MetricFetchE(message=f"Failed: {e}", exception=e))


class BuchwaldMetric(DiscreteChemMetric):
    """Buchwald_sub4 oracle (Reactant2/Ligand/Additive/Base)."""

    def __init__(
        self,
        name: str = "buchwald",
        *,
        param_names: Sequence[str] | None = None,
        test_csv: str | Path,
        train_csv: str | Path | None = None,
        target_column: str = "Yield",
        noiseless: bool = True,
        lower_is_better: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            param_names=param_names if param_names is not None else BUCHWALD_PARAM_NAMES,
            test_csv=test_csv,
            train_csv=train_csv,
            target_column=target_column,
            noiseless=noiseless,
            lower_is_better=lower_is_better,
        )
