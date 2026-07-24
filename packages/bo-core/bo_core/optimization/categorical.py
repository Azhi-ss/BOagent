"""One-hot encoding for fully-categorical feature spaces.

Categorical chemical-reaction datasets (Buchwald/Suzuki) carry IUPAC-name
features that the existing numeric BO pipeline cannot consume. This module
provides a small, dependency-light one-hot encoder keyed on a fixed option
schema, plus a helper to build that schema as the union of categories across
the data frames we actually encode (the merged ``train.csv`` contains
cross-product categories absent from a dataset's own ``options.json``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


def union_options(
    feature_cols: Sequence[str],
    *dfs: pd.DataFrame,
    options_json: Dict[str, Sequence[str]] | None = None,
) -> Dict[str, List[str]]:
    """Build the per-column option list as the sorted union of categories.

    Collects unique values from each provided DataFrame (over ``feature_cols``)
    and optionally merges a canonical ``options_json`` dict. The merged
    ``train.csv`` carries cross-product reagents not present in a dataset's own
    ``options.json``, so the union (not ``options_json`` alone) is the correct
    encoding schema.

    Returns a dict mapping each feature column to its sorted unique values.
    """
    sets: Dict[str, set] = {col: set() for col in feature_cols}
    if options_json:
        for col in feature_cols:
            sets[col].update(options_json.get(col, []))
    for df in dfs:
        for col in feature_cols:
            if col in df.columns:
                sets[col].update(df[col].dropna().astype(str).unique().tolist())
    return {col: sorted(sets[col]) for col in feature_cols}


class OneHotEncoder:
    """One-hot encoder over a fixed categorical schema.

    Each feature column maps to a contiguous one-hot block; a config is encoded
    by concatenating the blocks. Decoding takes the argmax within each block.
    """

    def __init__(self, feature_cols: Sequence[str], options: Dict[str, Sequence[str]]) -> None:
        self.feature_cols: List[str] = list(feature_cols)
        # Copy and validate: every declared column must have a non-empty option list.
        self.options: Dict[str, List[str]] = {
            col: list(options[col]) for col in self.feature_cols
        }
        for col, vals in self.options.items():
            if not vals:
                raise ValueError(f"Option list for column {col!r} is empty")
        self._cat_index: Dict[str, Dict[str, int]] = {
            col: {v: i for i, v in enumerate(vals)}
            for col, vals in self.options.items()
        }
        # Block offsets and sizes for slicing during decode.
        self._offsets: List[int] = []
        self._sizes: List[int] = []
        offset = 0
        for col in self.feature_cols:
            size = len(self.options[col])
            self._offsets.append(offset)
            self._sizes.append(size)
            offset += size
        self._dim = offset

    @property
    def dim(self) -> int:
        """Total one-hot width D = sum of option counts across columns."""
        return self._dim

    def encode_rows(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Encode a sequence of config dicts into an (N, D) float array."""
        n = len(rows)
        X = np.zeros((n, self._dim), dtype=float)
        for i, row in enumerate(rows):
            for j, col in enumerate(self.feature_cols):
                value = row[col]
                idx = self._cat_index[col].get(value)
                if idx is None:
                    raise ValueError(
                        f"Unknown category {value!r} for column {col!r}; "
                        f"not in encoder option list"
                    )
                X[i, self._offsets[j] + idx] = 1.0
        return X

    def encode_df(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a DataFrame's feature columns into an (N, D) float array."""
        rows = df[self.feature_cols].to_dict("records")
        return self.encode_rows(rows)

    def decode(self, vec: np.ndarray) -> Dict[str, str]:
        """Decode a single one-hot vector (D,) back into a config dict."""
        vec = np.asarray(vec)
        out: Dict[str, str] = {}
        for j, col in enumerate(self.feature_cols):
            offset, size = self._offsets[j], self._sizes[j]
            block = vec[offset:offset + size]
            out[col] = self.options[col][int(np.argmax(block))]
        return out

    def decode_many(self, X: np.ndarray) -> List[Dict[str, str]]:
        """Decode an (N, D) array into a list of config dicts."""
        return [self.decode(X[i]) for i in range(len(X))]
