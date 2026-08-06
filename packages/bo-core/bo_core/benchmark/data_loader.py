from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from bo_core.benchmark.datasets import DatasetSpec, get_dataset


@dataclass(frozen=True)
class DatasetBundle:
    spec: DatasetSpec
    searchspace: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    test_features: pd.DataFrame
    options: dict[str, list[Any]]

    @property
    def global_best(self) -> float:
        values = self.test[self.spec.target].to_numpy(dtype=float)
        reducer = np.max if self.spec.objective == "maximize" else np.min
        return float(reducer(values))

    def legacy(self) -> dict[str, Any]:
        features = list(self.spec.features)
        target = self.spec.target
        return {
            "train_x": self.train[features].to_numpy(),
            "train_y": self.train[target].to_numpy(),
            "test_x": self.test_features[features].to_numpy(),
            "test_y": self.test[target].to_numpy(),
            "feature_cols": features,
            "target_col": target,
            "df": self.searchspace,
            "train_df": self.train,
            "test_df": self.test,
            "options": self.options,
            "global_best": self.global_best,
            "spec": self.spec,
        }


def load_dataset(dataset_id: str, data_dir: str | Path | None = None) -> DatasetBundle:
    spec = get_dataset(dataset_id)
    directory = Path(data_dir) if data_dir is not None else spec.directory
    required = {
        "searchspace": directory / "searchspace.csv",
        "train": directory / "train.csv",
        "test": directory / "test.csv",
        "test_features": directory / "test_features.csv",
        "options": directory / "options.json",
    }
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Dataset {dataset_id} is missing required files: {', '.join(missing)}"
        )

    searchspace = _read_frame(required["searchspace"], spec, include_target=True)
    train = _read_frame(required["train"], spec, include_target=True)
    test = _read_frame(required["test"], spec, include_target=True)
    test_features = _read_frame(
        required["test_features"], spec, include_target=False
    )
    feature_columns = list(spec.features)
    expected_features = test.loc[:, feature_columns].reset_index(drop=True)
    candidate_features = test_features.loc[:, feature_columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            expected_features,
            candidate_features,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(
            f"Dataset {dataset_id} test_features.csv is not row-aligned with test.csv"
        ) from exc

    options = json.loads(required["options"].read_text(encoding="utf-8"))
    missing_options = set(spec.features) - set(options)
    if missing_options:
        missing_text = ", ".join(sorted(missing_options))
        raise ValueError(
            f"Dataset {dataset_id} options.json is missing features: {missing_text}"
        )
    feature_options = {feature: options[feature] for feature in spec.features}
    return DatasetBundle(
        spec=spec,
        searchspace=searchspace,
        train=train,
        test=test,
        test_features=candidate_features,
        options=feature_options,
    )


def load_dataset_legacy(
    dataset_id: str, data_dir: str | Path | None = None
) -> dict[str, Any]:
    return load_dataset(dataset_id, data_dir).legacy()


def _read_frame(path: Path, spec: DatasetSpec, *, include_target: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = list(spec.features)
    if include_target:
        columns.append(spec.target)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing registered columns: {', '.join(missing)}")

    schema_columns = {column: pa.Column(nullable=False) for column in spec.features}
    if include_target:
        schema_columns[spec.target] = pa.Column(
            float,
            pa.Check.greater_than_or_equal_to(0.0),
            nullable=False,
            coerce=True,
        )
    return pa.DataFrameSchema(columns=schema_columns, strict=False).validate(frame)

