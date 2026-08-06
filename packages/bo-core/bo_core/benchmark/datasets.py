from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

Objective = Literal["maximize", "minimize"]
_REQUIRED_DATASET_FILES = (
    "searchspace.csv",
    "train.csv",
    "test.csv",
    "test_features.csv",
    "options.json",
)


def _dataset_root(category: str, dataset_id: str) -> Path:
    """Find the configured or discovered root containing the dataset."""
    configured = os.environ.get("BOAGENT_DATA_ROOT")
    if configured is not None:
        if not configured.strip():
            raise FileNotFoundError("BOAGENT_DATA_ROOT is explicitly set but empty")
        root = Path(configured).expanduser().resolve()
        directory = root / category / dataset_id
        missing = [
            name
            for name in _REQUIRED_DATASET_FILES
            if not (directory / name).is_file()
        ]
        if not missing:
            return root
        raise FileNotFoundError(
            "BOAGENT_DATA_ROOT is incomplete for registered dataset "
            f"{dataset_id!r} at {directory}; missing: {', '.join(missing)}"
        )

    candidates = [
        *(parent / "datasets" for parent in Path(__file__).resolve().parents),
        *(parent / "datasets" for parent in (Path.cwd(), *Path.cwd().parents)),
    ]
    attempted: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in attempted:
            continue
        attempted.append(resolved)
        directory = resolved / category / dataset_id
        if all((directory / name).is_file() for name in _REQUIRED_DATASET_FILES):
            return resolved

    paths = "\n".join(f"- {path}" for path in attempted)
    raise FileNotFoundError(
        f"Could not locate registered dataset {dataset_id!r}. Attempted roots:\n{paths}"
    )


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    category: str
    features: tuple[str, ...]
    target: str
    objective: Objective = "maximize"
    unit: str | None = None

    @property
    def directory(self) -> Path:
        return _dataset_root(self.category, self.id) / self.category / self.id


_DATASETS = {
    "band_alignment": DatasetSpec(
        id="band_alignment",
        category="perovskite",
        features=("CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"),
        target="eta",
        unit="%",
    ),
    "defects_doping": DatasetSpec(
        id="defects_doping",
        category="perovskite",
        features=(
            "Nt_PVK/ETL",
            "Nt_HTL/PVK",
            "Na_PVK",
            "Nd_PVK",
            "Na_HTL",
            "Nd_HTL",
            "Na_ETL",
            "Nd_ETL",
        ),
        target="eta",
        unit="%",
    ),
    "buchwald_sub4": DatasetSpec(
        id="buchwald_sub4",
        category="chemical_reactions",
        features=("Reactant2", "Ligand", "Additive", "Base"),
        target="Yield",
        unit="%",
    ),
    "suzuki": DatasetSpec(
        id="suzuki",
        category="chemical_reactions",
        features=("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"),
        target="Yield",
        unit="%",
    ),
    "heck": DatasetSpec(
        id="heck",
        category="chemical_reactions",
        features=("Base", "Ligand", "Solvent", "Concentration_M", "Temp_C"),
        target="Yield",
        unit="%",
    ),
    "battery_cathode": DatasetSpec(
        id="battery_cathode",
        category="battery",
        features=("Precursor", "Sintering_Time_Hours", "Atmosphere", "Solvent"),
        target="Discharge_Capacity_mAh_g",
        unit="mAh/g",
    ),
}

DATASETS = MappingProxyType(_DATASETS)


def get_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return DATASETS[dataset_id]
    except KeyError:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset: {dataset_id}. Available: {available}"
        ) from None
