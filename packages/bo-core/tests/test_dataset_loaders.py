from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from bo_core.benchmark import (
    DATASETS as public_datasets,
)
from bo_core.benchmark import (
    DatasetSpec as PublicDatasetSpec,
)
from bo_core.benchmark import (
    datasets,
)
from bo_core.benchmark import (
    get_dataset as public_get_dataset,
)
from bo_core.benchmark import (
    load_dataset as public_load_dataset,
)
from bo_core.benchmark.data_loader import load_dataset
from bo_core.benchmark.datasets import DATASETS, DatasetSpec, get_dataset

EXPECTED = {
    "band_alignment": ("perovskite", "eta", 5, 10, 989),
    "defects_doping": ("perovskite", "eta", 8, 10, 988),
    "buchwald_sub4": ("chemical_reactions", "Yield", 4, 35, 783),
    "suzuki": ("chemical_reactions", "Yield", 5, 29, 5731),
    "heck": ("chemical_reactions", "Yield", 5, 172, 1556),
    "battery_cathode": ("battery", "Discharge_Capacity_mAh_g", 4, 10, 539),
}
REQUIRED_FILES = (
    "searchspace.csv",
    "train.csv",
    "test.csv",
    "test_features.csv",
    "options.json",
)




def test_dataset_registry_is_the_single_source_of_dataset_facts() -> None:
    assert set(DATASETS) == set(EXPECTED)
    for dataset_id, (category, target, feature_count, _, _) in EXPECTED.items():
        spec = get_dataset(dataset_id)
        assert isinstance(spec, DatasetSpec)
        assert spec.category == category
        assert spec.target == target
        assert len(spec.features) == feature_count
        assert spec.objective == "maximize"
        assert spec.directory.name == dataset_id


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("missing")


def test_benchmark_package_exports_public_contract() -> None:
    assert public_datasets is DATASETS
    assert PublicDatasetSpec is DatasetSpec
    assert public_get_dataset is get_dataset
    assert public_load_dataset is load_dataset


def test_dataset_root_honors_complete_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "snapshot" / "datasets"
    directory = root / "chemical_reactions" / "suzuki"
    directory.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (directory / name).touch()
    monkeypatch.setenv("BOAGENT_DATA_ROOT", str(root))

    assert get_dataset("suzuki").directory == directory


def test_dataset_root_discovers_exported_snapshot_from_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "exported"
    root = snapshot / "datasets"
    directory = root / "chemical_reactions" / "suzuki"
    directory.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (directory / name).touch()
    workdir = snapshot / "code" / "main"
    workdir.mkdir(parents=True)
    installed_module = (
        tmp_path / "venv" / "site-packages" / "bo_core" / "benchmark" / "datasets.py"
    )
    monkeypatch.delenv("BOAGENT_DATA_ROOT", raising=False)
    monkeypatch.setattr(datasets, "__file__", str(installed_module))
    monkeypatch.chdir(workdir)

    assert get_dataset("suzuki").directory == root / "chemical_reactions" / "suzuki"


def test_explicit_dataset_root_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setenv("BOAGENT_DATA_ROOT", str(configured))

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = get_dataset("suzuki").directory

    message = str(exc_info.value)
    assert "BOAGENT_DATA_ROOT is incomplete" in message
    assert str(configured / "chemical_reactions" / "suzuki") in message
    assert "searchspace.csv" in message


def test_empty_explicit_dataset_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOAGENT_DATA_ROOT", "")

    with pytest.raises(FileNotFoundError, match="explicitly set but empty"):
        _ = get_dataset("suzuki").directory


def test_missing_dataset_root_lists_attempted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "empty" / "nested"
    workdir.mkdir(parents=True)
    installed_module = (
        tmp_path / "venv" / "site-packages" / "bo_core" / "benchmark" / "datasets.py"
    )
    monkeypatch.delenv("BOAGENT_DATA_ROOT", raising=False)
    monkeypatch.setattr(datasets, "__file__", str(installed_module))
    monkeypatch.chdir(workdir)

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = get_dataset("suzuki").directory

    message = str(exc_info.value)
    assert "Attempted roots:" in message
    assert str(tmp_path / "datasets") in message
    assert str(workdir / "datasets") in message


@pytest.mark.parametrize("dataset_id", sorted(EXPECTED))
def test_registered_dataset_loads_presplit_contract(dataset_id: str) -> None:
    _, target, _, train_count, test_count = EXPECTED[dataset_id]
    bundle = load_dataset(dataset_id)

    assert len(bundle.train) == train_count
    assert len(bundle.test) == test_count
    assert bundle.spec.target == target
    assert list(bundle.test_features.columns) == list(bundle.spec.features)
    pd.testing.assert_frame_equal(
        bundle.test.loc[:, list(bundle.spec.features)].reset_index(drop=True),
        bundle.test_features.reset_index(drop=True),
        check_dtype=False,
    )
    assert bundle.global_best == pytest.approx(float(bundle.test[target].max()))
    assert set(bundle.options) == set(bundle.spec.features)


def test_loader_returns_independent_frames() -> None:
    first = load_dataset("suzuki")
    second = load_dataset("suzuki")
    first.train.iloc[0, 0] = "changed"
    assert second.train.iloc[0, 0] != "changed"
