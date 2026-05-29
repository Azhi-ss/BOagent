from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.data_loader import (
    BAND_ALIGNMENT_FEATURES,
    DEFECTS_DOPING_FEATURES,
    DATA_LOADERS,
    TARGET_COL,
    build_task_context,
)


def _create_test_excel(feature_cols: list[str]) -> Path:
    """Create a minimal test Excel file with random data."""
    rng = np.random.RandomState(42)
    n_rows = 30
    data = {}
    for col in feature_cols:
        data[col] = rng.uniform(1.0, 5.0, n_rows)
    data[TARGET_COL] = rng.uniform(15.0, 25.0, n_rows)
    df = pd.DataFrame(data)

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    df.to_excel(tmp.name, index=False)
    return Path(tmp.name)


class TestBandAlignmentDataLoader:
    def test_load_returns_expected_structure(self):
        path = _create_test_excel(BAND_ALIGNMENT_FEATURES)
        try:
            data = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=42)
            assert "train_x" in data
            assert "train_y" in data
            assert "test_x" in data
            assert "test_y" in data
            assert data["feature_cols"] == BAND_ALIGNMENT_FEATURES
            assert data["target_col"] == TARGET_COL
            assert isinstance(data["df"], pd.DataFrame)
            assert data["train_x"].shape[0] == 10
            assert data["test_x"].shape[0] == 20
            assert data["train_x"].shape[1] == len(BAND_ALIGNMENT_FEATURES)
        finally:
            path.unlink(missing_ok=True)

    def test_different_seeds_produce_different_splits(self):
        path = _create_test_excel(BAND_ALIGNMENT_FEATURES)
        try:
            data1 = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=42)
            data2 = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=99)
            # Train sets should differ
            assert not np.array_equal(data1["train_x"], data2["train_x"])
        finally:
            path.unlink(missing_ok=True)

    def test_build_task_context_for_band_alignment(self):
        path = _create_test_excel(BAND_ALIGNMENT_FEATURES)
        try:
            data = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=42)
            ctx = build_task_context("band_alignment", data)
            assert ctx["model"] == "band_alignment"
            assert ctx["lower_is_better"] is False
            assert ctx["feature_cols"] == BAND_ALIGNMENT_FEATURES
            assert ctx["target_col"] == "eta"
            assert "hyperparameter_constraints" in ctx
            for col in BAND_ALIGNMENT_FEATURES:
                assert col in ctx["hyperparameter_constraints"]
                constraint = ctx["hyperparameter_constraints"][col]
                assert constraint[0] == "float"
                assert constraint[1] == "linear"
                assert len(constraint[2]) == 2
        finally:
            path.unlink(missing_ok=True)


class TestDefectsDopingDataLoader:
    def test_load_returns_expected_structure(self):
        path = _create_test_excel(DEFECTS_DOPING_FEATURES)
        try:
            data = DATA_LOADERS["defects_doping"](file_path=path, n_train=10, seed=42)
            assert data["train_x"].shape[0] == 10
            assert data["test_x"].shape[0] == 20
            assert data["train_x"].shape[1] == len(DEFECTS_DOPING_FEATURES)
            assert data["feature_cols"] == DEFECTS_DOPING_FEATURES
        finally:
            path.unlink(missing_ok=True)
