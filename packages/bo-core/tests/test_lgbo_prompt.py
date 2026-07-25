from __future__ import annotations

import json

import pytest

from bo_core.benchmark.data_loader import UNIFIED_DATASET_ROOT
from bo_core.optimization.lgbo_prompt import DatasetMeta, build_system_prompt, build_user_prompt


def _load_options(dataset: str) -> dict:
    opts_path = UNIFIED_DATASET_ROOT / "chemical_reactions" / dataset / "options.json"
    return json.loads(opts_path.read_text())


def _buchwald_meta() -> DatasetMeta:
    opts = _load_options("buchwald_sub4")
    return DatasetMeta(
        dataset="buchwald_sub4",
        feature_cols=["Reactant2", "Ligand", "Additive", "Base"],
        options=opts,
    )


def _suzuki_meta() -> DatasetMeta:
    opts = _load_options("suzuki")
    return DatasetMeta(
        dataset="suzuki",
        feature_cols=["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"],
        options=opts,
    )


def test_system_prompt_has_appendix_b_structure():
    s = build_system_prompt(_buchwald_meta())
    # Evidence hierarchy
    assert "Evidence hierarchy" in s
    assert "PRIMARY" in s and "SECONDARY" in s
    assert "SIDE WITH BACKGROUND" in s
    # Modes emitted as JSON with point/region + confidence
    assert '{"mode": "point"' in s
    assert '{"mode": "region"' in s
    assert "confidence" in s
    assert "ccc in [0,1]" in s
    # Categorical literal output rule (lb=ub for region)
    assert "literal IUPAC option string" in s
    assert "lb[i]=ub[i]" in s
    # Output protocol + anti-collapse
    assert "Thinking" in s and "Final Answer" in s
    assert "Anti-collapse" in s
    assert "Do NOT normalize" in s


def test_system_prompt_mentions_reaction():
    s_b = build_system_prompt(_buchwald_meta())
    assert "Buchwald-Hartwig C-N coupling" in s_b
    s_s = build_system_prompt(_suzuki_meta())
    assert "Suzuki-Miyaura C-C coupling" in s_s


def test_user_prompt_background_and_review():
    meta = _buchwald_meta()
    history = [
        ({"Reactant2": "1-bromo-4-ethylbenzene", "Ligand": "L1", "Additive": "A1", "Base": "B1"}, 12.3),
        ({"Reactant2": "1-chloro-4-ethylbenzene", "Ligand": "L2", "Additive": "A2", "Base": "B2"}, 45.6),
    ]
    u = build_user_prompt(meta, history, prev_thinking="Bromide gave higher yield than chloride.")

    # [Background]
    assert "[Background]" in u
    assert "Parameter order (d=4)" in u
    assert "Reactant2, Ligand, Additive, Base" in u
    assert "Maximize f(x) = Yield (%)" in u
    assert "literal IUPAC" in u
    # Options listed per variable
    for col in meta.feature_cols:
        assert col in u
    # [Review]: newest first -> 45.6 appears before 12.3
    assert "[Review]" in u
    idx_45 = u.find("45.6000")
    idx_12 = u.find("12.3000")
    assert idx_45 != -1 and idx_12 != -1 and idx_45 < idx_12, "history must be newest-first"
    assert "Bromide gave higher yield" in u  # prev thinking carried
    assert "Adoption note" in u


def test_user_prompt_empty_history_and_max_history_cap():
    meta = _suzuki_meta()
    u_empty = build_user_prompt(meta, history=[])
    assert "(none yet)" in u_empty
    assert "(none)" in u_empty  # prev thinking default

    # Build 15 history items with unique condition markers; only the last 10
    # should appear, newest first.
    history = [
        ({c: f"{c}_v{i}" for c in meta.feature_cols}, float(i))
        for i in range(15)
    ]
    u = build_user_prompt(meta, history, max_history=10)
    # Exactly 10 history entries shown.
    assert u.count("-> Yield=") == 10
    # Newest (index 14) shown first; index 0 (oldest) dropped.
    assert "Electrophile_v14" in u and "Electrophile_v5" in u
    assert "Electrophile_v0" not in u and "Electrophile_v4" not in u
    assert u.find("Electrophile_v14") < u.find("Electrophile_v5")  # newest first


def test_user_prompt_options_are_sub4_space_not_union():
    """The prompt lists the dataset's own options.json (valid pool space), not
    the encoder's cross-product union."""
    meta = _buchwald_meta()
    u = build_user_prompt(meta, history=[])
    # sub4 Reactant2 options (3) are present; a cross-product category is NOT.
    assert "1-bromo-4-ethylbenzene" in u
    assert "2-bromopyridine" not in u  # cross-product, not a valid sub4 option
