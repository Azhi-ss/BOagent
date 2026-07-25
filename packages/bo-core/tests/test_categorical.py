from __future__ import annotations

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


def test_union_options_includes_cross_product_categories():
    data, feature_cols, train_df, test_df = _buchwald_frames()
    # options.json (sub4-only) would miss these cross-product train categories.
    cross_product = {"2-bromopyridine", "1-bromo-4-(trifluoromethyl)benzene"}
    opts = union_options(feature_cols, train_df, test_df)
    r2 = set(opts["Reactant2"])
    assert cross_product.issubset(r2), (
        f"union_options must include cross-product categories, missing {cross_product - r2}"
    )
    assert len(opts["Reactant2"]) > 3, "sub4 options.json has 3 Reactant2; union must be larger"


def test_buchwald_roundtrip_and_dim():
    data, feature_cols, train_df, test_df = _buchwald_frames()
    enc = OneHotEncoder(feature_cols, union_options(feature_cols, train_df, test_df))

    # Dim = sum of union option counts, must exceed the sub4-only 32.
    assert enc.dim == sum(len(enc.options[c]) for c in feature_cols)
    assert enc.dim > 32

    for df in (train_df, test_df):
        X = enc.encode_df(df)
        assert X.shape == (len(df), enc.dim)
        # Each row is a valid one-hot: one 1 per feature block.
        for j in range(len(feature_cols)):
            block = X[:, enc._offsets[j]:enc._offsets[j] + enc._sizes[j]]
            assert np.allclose(block.sum(axis=1), 1.0)
        # Round-trip: decode recovers the original IUPAC names.
        decoded = enc.decode_many(X)
        for i, row in enumerate(df[feature_cols].to_dict("records")):
            assert decoded[i] == row, f"round-trip mismatch at row {i}"


def test_suzuki_roundtrip_and_dim():
    data, feature_cols, train_df, test_df = _suzuki_frames()
    enc = OneHotEncoder(feature_cols, union_options(feature_cols, train_df, test_df))

    assert enc.dim == sum(len(enc.options[c]) for c in feature_cols)
    # Suzuki train.csv is single-product (29 rows, NOT merged across products
    # like Buchwald's 35), so train+test union == the sub option set = 35.
    assert enc.dim == 35

    for df in (train_df, test_df):
        X = enc.encode_df(df)
        assert X.shape == (len(df), enc.dim)
        decoded = enc.decode_many(X)
        for i, row in enumerate(df[feature_cols].to_dict("records")):
            assert decoded[i] == row


def test_encode_unknown_category_raises():
    _, feature_cols, train_df, _ = _buchwald_frames()
    enc = OneHotEncoder(feature_cols, union_options(feature_cols, train_df))
    bad_row = {col: train_df[col].iloc[0] for col in feature_cols}
    bad_row["Reactant2"] = "this-is-not-a-real-reagent"
    with pytest.raises(ValueError, match="Unknown category"):
        enc.encode_rows([bad_row])


def test_options_json_is_superset_merged_into_union():
    """The sub4 options.json categories must all survive into the union."""
    import json
    data, feature_cols, train_df, test_df = _buchwald_frames()
    opts_path = UNIFIED_DATASET_ROOT / "chemical_reactions" / "buchwald_sub4" / "options.json"
    options_json = json.loads(opts_path.read_text())
    opts = union_options(feature_cols, train_df, test_df, options_json=options_json)
    for col in feature_cols:
        assert set(options_json[col]).issubset(set(opts[col]))
