"""Contract tests for offline Chem-LGBO beta replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

from chem_lgbo_beta_replay import _verdict, main, replay_beta_sweep
from chem_lgbo_prompt_ablation import _posterior_state, replay_engine_to_step
from test_chem_lgbo_prompt_ablation import _real_record


def _sources(tmp_path: Path, *, fallback: bool = False) -> tuple[Path, Path]:
    state_record = _real_record()
    engine = replay_engine_to_step(state_record, 2)
    gp_index, gp_yield, posterior_hash = _posterior_state(engine)
    ligand = str(engine.test_df["Ligand"].iloc[0])
    guidance = {
        "state_key": "suzuki:100:2",
        "dataset": "suzuki",
        "seed": 100,
        "step": 2,
        "variant": "control",
        "posterior_hash": posterior_hash,
        "gp_index": gp_index,
        "gp_yield": gp_yield,
        "fallback": fallback,
        "subspace": {"Ligand": [ligand]},
    }
    state_path = tmp_path / "states.json"
    guidance_path = tmp_path / "guidance.json"
    state_path.write_text(json.dumps({"records": [state_record]}), encoding="utf-8")
    guidance_path.write_text(json.dumps({"records": [guidance]}), encoding="utf-8")
    return state_path, guidance_path


def test_replay_beta_zero_reproduces_gp_and_writes_aggregate(tmp_path: Path) -> None:
    state_path, guidance_path = _sources(tmp_path)

    result = replay_beta_sweep(
        state_path,
        guidance_path,
        tmp_path / "replay.json",
        betas=(0.0, 1.0),
    )

    zero = next(record for record in result["records"] if record["beta"] == 0.0)
    assert zero["selected_index"] == zero["gp_index"]
    assert zero["selected_yield"] == pytest.approx(zero["gp_yield"])
    summary = result["analysis"]["suzuki"]["control"]["0"]
    assert summary["count"] == 1
    assert summary["mean_delta_vs_gp"] == pytest.approx(0.0)
    assert json.loads((tmp_path / "replay.json").read_text(encoding="utf-8")) == result


def test_fallback_replays_pure_gp_for_every_beta(tmp_path: Path) -> None:
    state_path, guidance_path = _sources(tmp_path, fallback=True)

    result = replay_beta_sweep(
        state_path,
        guidance_path,
        tmp_path / "replay.json",
        betas=(0.0, 0.5, 1.0),
    )

    assert {record["selected_index"] for record in result["records"]} == {
        result["records"][0]["gp_index"]
    }
    assert all(record["selected_in_subspace"] is None for record in result["records"])


def test_replay_rejects_invalid_beta_duplicate_state_and_posterior_drift(
    tmp_path: Path,
) -> None:
    state_path, guidance_path = _sources(tmp_path)
    output = tmp_path / "replay.json"

    with pytest.raises(ValueError, match="beta"):
        replay_beta_sweep(state_path, guidance_path, output, betas=(-0.1,))

    guidance: dict[str, Any] = json.loads(guidance_path.read_text(encoding="utf-8"))
    guidance["records"].append(dict(guidance["records"][0]))
    guidance_path.write_text(json.dumps(guidance), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate guidance state"):
        replay_beta_sweep(state_path, guidance_path, output)

    guidance["records"] = [guidance["records"][0]]
    guidance["records"][0]["posterior_hash"] = "wrong"
    guidance_path.write_text(json.dumps(guidance), encoding="utf-8")
    with pytest.raises(ValueError, match="posterior hash"):
        replay_beta_sweep(state_path, guidance_path, output)


def test_replay_rejects_existing_output_with_different_provenance(tmp_path: Path) -> None:
    state_path, guidance_path = _sources(tmp_path)
    output = tmp_path / "replay.json"
    output.write_text(
        json.dumps(
            {
                "state_source_sha256": "wrong",
                "guidance_source_sha256": "wrong",
                "betas": [0.0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        replay_beta_sweep(state_path, guidance_path, output, betas=(0.0,))


def test_replay_rejects_saved_guidance_with_empty_remaining_mask(tmp_path: Path) -> None:
    state_path, guidance_path = _sources(tmp_path)
    guidance = json.loads(guidance_path.read_text(encoding="utf-8"))
    guidance["records"][0]["subspace"] = {"Ligand": ["not-in-the-pool"]}
    guidance_path.write_text(json.dumps(guidance), encoding="utf-8")

    with pytest.raises(ValueError, match="guidance mask"):
        replay_beta_sweep(state_path, guidance_path, tmp_path / "replay.json")


def test_verdict_preserves_dataset_variant_heterogeneity() -> None:
    analysis = {
        "buchwald_sub4": {
            "control": {"0.25": {"mean_delta_vs_gp": -1.0}},
        },
        "suzuki": {
            "control": {"0.25": {"mean_delta_vs_gp": 2.0}},
        },
    }

    verdict = _verdict(analysis, (0.0, 0.25))

    assert verdict["interpretation"] == "heterogeneous_reward_signal"
    assert verdict["groups"]["buchwald_sub4"]["control"]["has_positive_signal"] is False
    assert verdict["groups"]["suzuki"]["control"]["has_positive_signal"] is True


def test_cli_does_not_construct_llm_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path, guidance_path = _sources(tmp_path)
    monkeypatch.setattr(
        "chem_lgbo_prompt_ablation.DeepSeekClient.from_env",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be constructed")),
    )

    assert main(
        [
            "--states",
            str(state_path),
            "--guidance",
            str(guidance_path),
            "--output",
            str(tmp_path / "replay.json"),
        ]
    ) == 0
