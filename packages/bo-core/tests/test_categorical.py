from __future__ import annotations

import json

import numpy as np
import pytest
from bo_core.benchmark.data_loader import (
    UNIFIED_DATASET_ROOT,
    load_buchwald_sub4_data,
    load_suzuki_data,
)
from bo_core.optimization.categorical import OneHotEncoder, union_options


def _buchwald_frames():
    data = load_buchwald_sub4_data()
    return data, data["feature_cols"], data["train_df"], data["test_df"]


def _suzuki_frames():
    data = load_suzuki_data()
    return data, data["feature_cols"], data["train_df"], data["test_df"]


def _dataset_options(dataset: str, feature_cols: list[str]) -> dict[str, list[str]]:
    opts_path = (
        UNIFIED_DATASET_ROOT
        / "chemical_reactions"
        / dataset
        / "options.json"
    )
    options_json = json.loads(opts_path.read_text())
    return {col: options_json[col] for col in feature_cols}


def test_buchwald_target_options_are_32_dimensional_without_dead_pool_columns():
    _, feature_cols, _, test_df = _buchwald_frames()
    enc = OneHotEncoder(
        feature_cols,
        _dataset_options("buchwald_sub4", feature_cols),
    )

    X = enc.encode_df(test_df)

    assert enc.dim == 32
    assert X.shape == (len(test_df), 32)
    assert np.all(X.sum(axis=0) > 0)
    for j in range(len(feature_cols)):
        block = X[:, enc._offsets[j]:enc._offsets[j] + enc._sizes[j]]
        assert np.allclose(block.sum(axis=1), 1.0)

    decoded = enc.decode_many(X)
    assert decoded == test_df[feature_cols].to_dict("records")


def test_buchwald_full_prior_encodes_out_of_domain_reactants_as_zero_block():
    _, feature_cols, train_df, _ = _buchwald_frames()
    enc = OneHotEncoder(
        feature_cols,
        _dataset_options("buchwald_sub4", feature_cols),
    )

    X = enc.encode_df(train_df, allow_unknown=True)
    reactant_idx = feature_cols.index("Reactant2")
    offset = enc._offsets[reactant_idx]
    size = enc._sizes[reactant_idx]
    reactant_block = X[:, offset:offset + size]
    is_target_reactant = train_df["Reactant2"].isin(
        enc.options["Reactant2"]
    ).to_numpy()

    assert X.shape == (35, 32)
    assert int(is_target_reactant.sum()) == 7
    assert np.allclose(reactant_block[is_target_reactant].sum(axis=1), 1.0)
    assert np.allclose(reactant_block[~is_target_reactant], 0.0)

    for col in ("Ligand", "Additive", "Base"):
        j = feature_cols.index(col)
        block = X[:, enc._offsets[j]:enc._offsets[j] + enc._sizes[j]]
        assert np.allclose(block.sum(axis=1), 1.0)


def test_suzuki_roundtrip_and_dim():
    _, feature_cols, train_df, test_df = _suzuki_frames()
    enc = OneHotEncoder(
        feature_cols,
        _dataset_options("suzuki", feature_cols),
    )

    assert enc.dim == sum(len(enc.options[c]) for c in feature_cols)
    assert enc.dim == 35

    for df in (train_df, test_df):
        X = enc.encode_df(df)
        assert X.shape == (len(df), enc.dim)
        decoded = enc.decode_many(X)
        for i, row in enumerate(df[feature_cols].to_dict("records")):
            assert decoded[i] == row


def test_encode_unknown_category_raises():
    _, feature_cols, train_df, _ = _buchwald_frames()
    enc = OneHotEncoder(
        feature_cols,
        _dataset_options("buchwald_sub4", feature_cols),
    )
    bad_row = {col: train_df[col].iloc[0] for col in feature_cols}
    bad_row["Reactant2"] = "this-is-not-a-real-reagent"

    with pytest.raises(ValueError, match="Unknown category"):
        enc.encode_rows([bad_row])


def test_options_json_is_superset_merged_into_union():
    """The sub4 options.json categories must all survive into the union."""
    _, feature_cols, train_df, test_df = _buchwald_frames()
    opts_path = (
        UNIFIED_DATASET_ROOT
        / "chemical_reactions"
        / "buchwald_sub4"
        / "options.json"
    )
    options_json = json.loads(opts_path.read_text())
    opts = union_options(
        feature_cols,
        train_df,
        test_df,
        options_json=options_json,
    )
    for col in feature_cols:
        assert set(options_json[col]).issubset(set(opts[col]))
