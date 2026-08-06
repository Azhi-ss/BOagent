from __future__ import annotations

import pytest
from bo_core.benchmark.data_loader import load_dataset
from bo_core.optimization.lgbo_prompt import DatasetMeta, build_system_prompt


def test_prompt_metadata_is_assembled_from_registered_dataset_facts() -> None:
    bundle = load_dataset("suzuki")
    meta = DatasetMeta(
        dataset=bundle.spec.id,
        feature_cols=list(bundle.spec.features),
        options=bundle.options,
        target_name=bundle.spec.target,
    )

    assert meta.feature_cols == list(bundle.spec.features)
    assert meta.options == bundle.options
    assert "Suzuki-Miyaura" in build_system_prompt(meta)


def test_prompt_strategy_rejects_dataset_without_registered_strategy_context() -> None:
    bundle = load_dataset("heck")
    meta = DatasetMeta(
        dataset=bundle.spec.id,
        feature_cols=list(bundle.spec.features),
        options=bundle.options,
        target_name=bundle.spec.target,
    )

    with pytest.raises(KeyError):
        build_system_prompt(meta)
