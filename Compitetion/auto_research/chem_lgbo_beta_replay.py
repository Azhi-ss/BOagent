"""Offline reward-strength replay for saved Chem-LGBO subspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from bo_core.optimization.chem_lgbo import build_subspace_mask
from chem_lgbo_prompt_ablation import replay_engine_to_step

DEFAULT_BETAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, dict) else None
    if not isinstance(records, list):
        raise TypeError(f"{label} records are invalid")
    return [dict(record) for record in records]


def _validate_betas(betas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(beta) for beta in betas)
    if not values or any(not np.isfinite(beta) or beta < 0 for beta in values):
        raise ValueError("beta values must be finite and non-negative")
    if len(values) != len(set(values)):
        raise ValueError("beta values must be unique")
    return values


def _state_snapshot(record: Mapping[str, Any], step: int) -> dict[str, Any]:
    engine = replay_engine_to_step(record, step)
    surrogate = engine._fit_gp()
    mean, std = engine._predict_pool(surrogate)
    remaining = np.ones(engine.M, dtype=bool)
    if engine.queried:
        remaining[list(engine.queried)] = False
    acquisition = engine._expected_improvement(mean, std, float(np.max(engine.y_obs)))
    acquisition = np.where(remaining & np.isfinite(acquisition), acquisition, -np.inf)
    if not np.any(np.isfinite(acquisition)):
        raise ValueError("stored state has no finite GP acquisition")
    gp_index = int(np.argmax(acquisition))
    digest = hashlib.sha256()
    digest.update(np.asarray(mean, dtype=np.float64).tobytes())
    digest.update(np.asarray(std, dtype=np.float64).tobytes())
    digest.update(remaining.tobytes())
    return {
        "engine": engine,
        "mean": mean,
        "std": std,
        "remaining": remaining,
        "best_f": float(np.max(engine.y_obs)),
        "gp_index": gp_index,
        "gp_yield": float(engine.pool_yield[gp_index]),
        "posterior_hash": digest.hexdigest(),
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["dataset"]), str(record["variant"]), float(record["beta"]))].append(record)

    result: dict[str, Any] = {}
    for (dataset, variant, beta), rows in sorted(grouped.items()):
        deltas = np.asarray(
            [float(row["selected_yield"]) - float(row["gp_yield"]) for row in rows]
        )
        applicable = [row for row in rows if row["selected_in_subspace"] is not None]
        result.setdefault(dataset, {}).setdefault(variant, {})[f"{beta:g}"] = {
            "count": len(rows),
            "mean_delta_vs_gp": float(np.mean(deltas)),
            "wins": int(np.count_nonzero(deltas > 0)),
            "ties": int(np.count_nonzero(deltas == 0)),
            "losses": int(np.count_nonzero(deltas < 0)),
            "subspace_hit_rate": (
                float(np.mean([bool(row["selected_in_subspace"]) for row in applicable]))
                if applicable
                else None
            ),
            "incumbent_improvement_rate": float(
                np.mean([bool(row["improved_incumbent"]) for row in rows])
            ),
        }
    return result


def _verdict(analysis: Mapping[str, Any], betas: Sequence[float]) -> dict[str, Any]:
    positive = [beta for beta in betas if beta > 0]
    groups: dict[str, dict[str, Any]] = {}
    for dataset, variants in analysis.items():
        groups[str(dataset)] = {}
        for variant, summaries in variants.items():
            means = {
                f"{beta:g}": float(summaries[f"{beta:g}"]["mean_delta_vs_gp"])
                for beta in positive
            }
            best = max(means, key=means.get) if means else None
            groups[str(dataset)][str(variant)] = {
                "best_beta": float(best) if best is not None else None,
                "mean_delta_vs_gp": means,
                "has_positive_signal": any(delta > 0 for delta in means.values()),
            }
    signals = [
        bool(group["has_positive_signal"])
        for variants in groups.values()
        for group in variants.values()
    ]
    return {
        "groups": groups,
        "interpretation": (
            "heterogeneous_reward_signal"
            if signals and any(signals) and not all(signals)
            else (
                "positive_reward_signal_in_all_groups"
                if signals and all(signals)
                else "no_positive_reward_signal_in_any_group"
            )
        ),
        "scope": "saved categorical subspaces only; not paper covariance LGBO",
    }


def replay_beta_sweep(
    state_source_path: Path,
    guidance_source_path: Path,
    output_path: Path,
    *,
    betas: Sequence[float] = DEFAULT_BETAS,
) -> dict[str, Any]:
    """Replay saved subspaces across reward strengths without any LLM calls."""
    beta_values = _validate_betas(betas)
    state_hash = _sha256(state_source_path)
    guidance_hash = _sha256(guidance_source_path)
    identity = {
        "state_source_sha256": state_hash,
        "guidance_source_sha256": guidance_hash,
        "betas": list(beta_values),
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("existing output provenance does not match")

    state_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in _records(state_source_path, "state source"):
        if record.get("composition") != "chem_lgbo":
            continue
        key = (str(record.get("dataset")), int(record.get("seed")))
        if key in state_by_key:
            raise ValueError(f"duplicate state source record for {key!r}")
        state_by_key[key] = record

    guidance_records = _records(guidance_source_path, "guidance source")
    seen: set[tuple[str, str]] = set()
    snapshots: dict[tuple[str, int, int], dict[str, Any]] = {}
    replayed: list[dict[str, Any]] = []
    required = {"state_key", "dataset", "seed", "step", "variant", "posterior_hash", "gp_index", "gp_yield"}

    for guidance in guidance_records:
        missing = required - guidance.keys()
        if missing:
            raise ValueError(f"guidance record missing fields: {sorted(missing)}")
        guidance_key = (str(guidance["state_key"]), str(guidance["variant"]))
        if guidance_key in seen:
            raise ValueError(f"duplicate guidance state: {guidance_key!r}")
        seen.add(guidance_key)

        dataset, seed, step = str(guidance["dataset"]), int(guidance["seed"]), int(guidance["step"])
        state_key = (dataset, seed, step)
        if state_key not in snapshots:
            try:
                source_record = state_by_key[(dataset, seed)]
            except KeyError as exc:
                raise ValueError(f"missing state source for {(dataset, seed)!r}") from exc
            snapshots[state_key] = _state_snapshot(source_record, step)
        snapshot = snapshots[state_key]
        if snapshot["posterior_hash"] != guidance["posterior_hash"]:
            raise ValueError(f"posterior hash mismatch for {guidance_key!r}")
        if snapshot["gp_index"] != int(guidance["gp_index"]) or not np.isclose(
            snapshot["gp_yield"], float(guidance["gp_yield"]), rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"saved GP result mismatch for {guidance_key!r}")

        engine = snapshot["engine"]
        fallback = bool(guidance.get("fallback")) or not guidance.get("subspace")
        mask = None if fallback else build_subspace_mask(
            engine.test_df.loc[:, engine.feature_cols], dict(guidance["subspace"])
        ) & snapshot["remaining"]
        if mask is not None and (
            not np.any(mask) or np.array_equal(mask, snapshot["remaining"])
        ):
            raise ValueError(f"saved guidance mask is invalid for {guidance_key!r}")
        for beta in beta_values:
            shifted = np.asarray(snapshot["mean"], dtype=float).copy()
            if mask is not None:
                shifted[mask] += beta * snapshot["std"][mask]
            acquisition = engine._expected_improvement(shifted, snapshot["std"], snapshot["best_f"])
            acquisition = np.where(
                snapshot["remaining"] & np.isfinite(acquisition), acquisition, -np.inf
            )
            if not np.any(np.isfinite(acquisition)):
                raise ValueError(f"no finite acquisition for {guidance_key!r}, beta={beta:g}")
            selected = int(np.argmax(acquisition))
            if beta == 0 and selected != snapshot["gp_index"]:
                raise ValueError(f"beta=0 did not reproduce GP for {guidance_key!r}")
            selected_yield = float(engine.pool_yield[selected])
            replayed.append(
                {
                    "state_key": guidance["state_key"],
                    "dataset": dataset,
                    "seed": seed,
                    "step": step,
                    "variant": guidance["variant"],
                    "beta": beta,
                    "fallback": fallback,
                    "gp_index": snapshot["gp_index"],
                    "gp_yield": snapshot["gp_yield"],
                    "selected_index": selected,
                    "selected_yield": selected_yield,
                    "selected_in_subspace": bool(mask[selected]) if mask is not None else None,
                    "improved_incumbent": selected_yield > snapshot["best_f"],
                }
            )

    analysis = _aggregate(replayed)
    artifact = {
        "schema_version": 1,
        **identity,
        "state_count": len(snapshots),
        "guidance_record_count": len(guidance_records),
        "record_count": len(replayed),
        "records": replayed,
        "analysis": analysis,
        "verdict": _verdict(analysis, beta_values),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = replay_beta_sweep(args.states, args.guidance, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
