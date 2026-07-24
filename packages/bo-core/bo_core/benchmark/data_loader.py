from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

# Project roots and default data paths
# Path(__file__) is BOagent/packages/bo-core/bo_core/benchmark/data_loader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
UNIFIED_DATASET_ROOT = PROJECT_ROOT / "datasets"
LEGACY_DATA_ROOT = PROJECT_ROOT / "references" / "PVK-LLM" / "custom_perovskite_dataset"
DEFAULT_DATA_ROOT = UNIFIED_DATASET_ROOT / "perovskite"



# Perovskite features & targets
BAND_ALIGNMENT_FEATURES = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
DEFECTS_DOPING_FEATURES = [
    "Nt_PVK/ETL", "Nt_HTL/PVK", "Na_PVK", "Nd_PVK",
    "Na_HTL", "Nd_HTL", "Na_ETL", "Nd_ETL",
]
PEROVSKITE_TARGET_COL = "eta"
TARGET_COL = PEROVSKITE_TARGET_COL

# Chemical reactions features & targets
BUCHWALD_SUB4_FEATURES = ["Reactant2", "Ligand", "Additive", "Base"]
SUZUKI_FEATURES = ["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"]
REACTION_TARGET_COL = "Yield"

# Battery features & targets
BATTERY_CATHODE_FEATURES = ["Precursor", "Sintering_Time", "Atmosphere", "Solvent"]
BATTERY_TARGET_COL = "Discharge_Capacity_mAh_g"


def _resolve_perovskite_file(filename: str, legacy_filename: str | None = None) -> Path:
    """Resolve file path prioritizing datasets/perovskite over legacy paths."""
    unified_path = UNIFIED_DATASET_ROOT / "perovskite" / filename
    if unified_path.exists():
        return unified_path
    
    if legacy_filename:
        unified_legacy = UNIFIED_DATASET_ROOT / "perovskite" / legacy_filename
        if unified_legacy.exists():
            return unified_legacy
            
    legacy_path = LEGACY_DATA_ROOT / (legacy_filename or filename)
    if legacy_path.exists():
        return legacy_path

    return unified_path


def load_band_alignment_data(
    file_path: str | Path | None = None,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load band alignment dataset and split into train/test."""
    if file_path is None:
        csv_dir = UNIFIED_DATASET_ROOT / "perovskite" / "band_alignment"
        if (csv_dir / "searchspace.csv").exists():
            df = pd.read_csv(csv_dir / "searchspace.csv")
            return _split_data(df, BAND_ALIGNMENT_FEATURES, PEROVSKITE_TARGET_COL, n_train, seed)
        file_path = _resolve_perovskite_file("band_alignment.xlsx", "bandAlignment.xlsx")

    path = Path(file_path)
    if path.is_dir():
        df = pd.read_csv(path / "searchspace.csv")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return _split_data(df, BAND_ALIGNMENT_FEATURES, PEROVSKITE_TARGET_COL, n_train, seed)


def load_defects_doping_data(
    file_path: str | Path | None = None,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load defects & doping dataset and split into train/test."""
    if file_path is None:
        csv_dir = UNIFIED_DATASET_ROOT / "perovskite" / "defects_doping"
        if (csv_dir / "searchspace.csv").exists():
            df = pd.read_csv(csv_dir / "searchspace.csv")
            return _split_data(df, DEFECTS_DOPING_FEATURES, PEROVSKITE_TARGET_COL, n_train, seed)
        file_path = _resolve_perovskite_file("defects_doping.xlsx", "defectsAndDoping.xlsx")

    path = Path(file_path)
    if path.is_dir():
        df = pd.read_csv(path / "searchspace.csv")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return _split_data(df, DEFECTS_DOPING_FEATURES, PEROVSKITE_TARGET_COL, n_train, seed)



def load_buchwald_sub4_data(
    data_dir: str | Path | None = None,
    use_presplit: bool = True,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load Buchwald-Hartwig coupling reaction dataset (Buchwald_sub4)."""
    if data_dir is None:
        data_dir = UNIFIED_DATASET_ROOT / "chemical_reactions" / "buchwald_sub4"
    else:
        data_dir = Path(data_dir)

    if use_presplit and (data_dir / "train.csv").exists() and (data_dir / "test.csv").exists():
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        full_df = pd.read_csv(data_dir / "searchspace.csv") if (data_dir / "searchspace.csv").exists() else pd.concat([train_df, test_df], ignore_index=True)
        return {
            "train_x": train_df[BUCHWALD_SUB4_FEATURES].values,
            "train_y": train_df[REACTION_TARGET_COL].values,
            "test_x": test_df[BUCHWALD_SUB4_FEATURES].values,
            "test_y": test_df[REACTION_TARGET_COL].values,
            "feature_cols": BUCHWALD_SUB4_FEATURES,
            "target_col": REACTION_TARGET_COL,
            "df": full_df,
            "train_df": train_df,
            "test_df": test_df,
        }

    searchspace_path = data_dir / "searchspace.csv" if data_dir.is_dir() else data_dir
    df = pd.read_csv(searchspace_path)
    return _split_data(df, BUCHWALD_SUB4_FEATURES, REACTION_TARGET_COL, n_train, seed)


def load_suzuki_data(
    data_dir: str | Path | None = None,
    use_presplit: bool = True,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load Suzuki-Miyaura cross-coupling reaction dataset."""
    if data_dir is None:
        data_dir = UNIFIED_DATASET_ROOT / "chemical_reactions" / "suzuki"
    else:
        data_dir = Path(data_dir)

    if use_presplit and (data_dir / "train.csv").exists() and (data_dir / "test.csv").exists():
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        full_df = pd.read_csv(data_dir / "searchspace.csv") if (data_dir / "searchspace.csv").exists() else pd.concat([train_df, test_df], ignore_index=True)
        return {
            "train_x": train_df[SUZUKI_FEATURES].values,
            "train_y": train_df[REACTION_TARGET_COL].values,
            "test_x": test_df[SUZUKI_FEATURES].values,
            "test_y": test_df[REACTION_TARGET_COL].values,
            "feature_cols": SUZUKI_FEATURES,
            "target_col": REACTION_TARGET_COL,
            "df": full_df,
            "train_df": train_df,
            "test_df": test_df,
        }

    searchspace_path = data_dir / "searchspace.csv" if data_dir.is_dir() else data_dir
    df = pd.read_csv(searchspace_path)
    return _split_data(df, SUZUKI_FEATURES, REACTION_TARGET_COL, n_train, seed)


def load_battery_cathode_data(
    data_dir: str | Path | None = None,
    use_presplit: bool = True,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load Battery Cathode Synthesis dataset (battery_cathode)."""
    if data_dir is None:
        data_dir = UNIFIED_DATASET_ROOT / "battery" / "battery_cathode"
    else:
        data_dir = Path(data_dir)

    if use_presplit and (data_dir / "train.csv").exists() and (data_dir / "test.csv").exists():
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        full_df = pd.read_csv(data_dir / "searchspace.csv") if (data_dir / "searchspace.csv").exists() else pd.concat([train_df, test_df], ignore_index=True)
        return {
            "train_x": train_df[BATTERY_CATHODE_FEATURES].values,
            "train_y": train_df[BATTERY_TARGET_COL].values,
            "test_x": test_df[BATTERY_CATHODE_FEATURES].values,
            "test_y": test_df[BATTERY_TARGET_COL].values,
            "feature_cols": BATTERY_CATHODE_FEATURES,
            "target_col": BATTERY_TARGET_COL,
            "df": full_df,
            "train_df": train_df,
            "test_df": test_df,
        }

    searchspace_path = data_dir / "searchspace.csv" if data_dir.is_dir() else data_dir
    df = pd.read_csv(searchspace_path)
    return _split_data(df, BATTERY_CATHODE_FEATURES, BATTERY_TARGET_COL, n_train, seed)


def validate_dataset_schema(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Validate DataFrame schema using pandera to ensure non-null and valid data."""
    columns_schema = {}
    for col in feature_cols:
        columns_schema[col] = pa.Column(nullable=False)

    if target_col in ["eta", "Yield", "Discharge_Capacity_mAh_g"]:
        columns_schema[target_col] = pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False)
    else:
        columns_schema[target_col] = pa.Column(nullable=False)

    schema = pa.DataFrameSchema(columns=columns_schema)
    return schema.validate(df)


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

    df = validate_dataset_schema(df, feature_cols, target_col)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
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


DATA_LOADERS = {
    "band_alignment": load_band_alignment_data,
    "defects_doping": load_defects_doping_data,
    "buchwald_sub4": load_buchwald_sub4_data,
    "suzuki": load_suzuki_data,
    "battery_cathode": load_battery_cathode_data,
}

