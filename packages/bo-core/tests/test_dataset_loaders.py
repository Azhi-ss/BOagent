from __future__ import annotations

import pytest
from bo_core.benchmark.data_loader import (
    load_band_alignment_data,
    load_defects_doping_data,
    load_buchwald_sub4_data,
    load_suzuki_data,
    load_battery_cathode_data,
    DATA_LOADERS,
)


def test_load_band_alignment_data():
    data = load_band_alignment_data()
    assert "train_x" in data
    assert "train_y" in data
    assert "test_x" in data
    assert "test_y" in data
    assert data["target_col"] == "eta"
    assert len(data["feature_cols"]) == 5


def test_load_defects_doping_data():
    data = load_defects_doping_data()
    assert "train_x" in data
    assert "train_y" in data
    assert "test_x" in data
    assert "test_y" in data
    assert data["target_col"] == "eta"
    assert len(data["feature_cols"]) == 8


def test_load_buchwald_sub4_data():
    data = load_buchwald_sub4_data()
    assert "train_x" in data
    assert "train_y" in data
    assert "test_x" in data
    assert "test_y" in data
    assert data["target_col"] == "Yield"
    assert data["feature_cols"] == ["Reactant2", "Ligand", "Additive", "Base"]
    assert len(data["train_y"]) == 35  # Pre-split train samples
    assert len(data["test_y"]) == 783  # Pre-split test samples


def test_load_suzuki_data():
    data = load_suzuki_data()
    assert "train_x" in data
    assert "train_y" in data
    assert "test_x" in data
    assert "test_y" in data
    assert data["target_col"] == "Yield"
    assert data["feature_cols"] == ["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"]
    assert len(data["train_y"]) == 29  # Pre-split train samples
    assert len(data["test_y"]) == 5731 # Pre-split test samples


def test_load_battery_cathode_data():
    data = load_battery_cathode_data()
    assert "train_x" in data
    assert "train_y" in data
    assert "test_x" in data
    assert "test_y" in data
    assert data["target_col"] == "Discharge_Capacity_mAh_g"
    assert data["feature_cols"] == ["Precursor", "Sintering_Time", "Atmosphere", "Solvent"]
    assert len(data["train_y"]) == 10  # Pre-split train samples
    assert len(data["test_y"]) == 539  # Pre-split test samples


def test_data_loaders_registry():
    assert "band_alignment" in DATA_LOADERS
    assert "defects_doping" in DATA_LOADERS
    assert "buchwald_sub4" in DATA_LOADERS
    assert "suzuki" in DATA_LOADERS
    assert "battery_cathode" in DATA_LOADERS

