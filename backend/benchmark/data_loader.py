from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default data paths (relative to PVK-LLM project root)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "PVK-LLM" / "custom_perovskite_dataset"

BAND_ALIGNMENT_FEATURES = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
DEFECTS_DOPING_FEATURES = [
    "Nt_PVK/ETL", "Nt_HTL/PVK", "Na_PVK", "Nd_PVK",
    "Na_HTL", "Nd_HTL", "Na_ETL", "Nd_ETL",
]
TARGET_COL = "eta"


def load_band_alignment_data(
    file_path: str | Path | None = None,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load band alignment dataset and split into train/test.

    Returns:
        dict with keys: train_x, train_y, test_x, test_y,
                        feature_cols, target_col, df
    """
    if file_path is None:
        file_path = DEFAULT_DATA_ROOT / "bandAlignment.xlsx"
    df = pd.read_excel(Path(file_path))
    return _split_data(df, BAND_ALIGNMENT_FEATURES, TARGET_COL, n_train, seed)


def load_defects_doping_data(
    file_path: str | Path | None = None,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load defects & doping dataset and split into train/test.

    Returns:
        dict with keys: train_x, train_y, test_x, test_y,
                        feature_cols, target_col, df
    """
    if file_path is None:
        file_path = DEFAULT_DATA_ROOT / "defectsAndDoping.xlsx"
    df = pd.read_excel(Path(file_path))
    return _split_data(df, DEFECTS_DOPING_FEATURES, TARGET_COL, n_train, seed)


def _split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_train: int,
    seed: int,
) -> dict[str, Any]:
    """Split DataFrame into train/test sets."""
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset. Available: {df.columns.tolist()}")

    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return {
        "train_x": train_df[feature_cols].values,
        "train_y": train_df[target_col].values,
        "test_x": test_df[feature_cols].values,
        "test_y": test_df[target_col].values,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "df": df,
    }


def build_task_context(
    task_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Build task_context dict from loaded data, matching PVK-LLM format.

    Args:
        task_id: "band_alignment" or "defects_doping"
        data: dict from load_*_data()
    """
    feature_cols = data["feature_cols"]
    df = data["df"]

    hyperparameter_constraints: dict[str, list[Any]] = {}
    for col in feature_cols:
        col_data = pd.to_numeric(df[col], errors="coerce").dropna()
        hyperparameter_constraints[col] = [
            "float",
            "linear",
            [float(col_data.min()), float(col_data.max())],
        ]

    return {
        "model": task_id,
        "task": "regression",
        "metric": "neg_mean_squared_error",
        "num_classes": 1,
        "n_classes": 1,
        "lower_is_better": False,
        "num_samples": int(len(df)),
        "tot_feats": len(feature_cols),
        "cat_feats": 0,
        "num_feats": len(feature_cols),
        "feature_cols": feature_cols,
        "target_col": data["target_col"],
        "hyperparameter_constraints": hyperparameter_constraints,
        "df": df,
    }


DATA_LOADERS = {
    "band_alignment": load_band_alignment_data,
    "defects_doping": load_defects_doping_data,
}
