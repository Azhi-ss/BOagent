"""Offline evidence-gated top-5 reranking experiment for saved Chem-LGBO states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from bo_core.llm_client import DeepSeekClient
from bo_core.optimization.reranking import (
    ArtifactEvidenceGate,
    ChemGPShortlistAdapter,
    DeepSeekCandidateReranker,
    SelectCandidateUseCase,
)
from chem_lgbo_prompt_ablation import replay_engine_to_step

SHORTLIST_SIZE = 5
REQUIRED_DATASETS = {"buchwald_sub4", "suzuki"}
REQUIRED_SEEDS = {"100", "200", "300", "400", "500"}
MIN_RANKING_ACCURACY = 0.6
MIN_PAIRWISE_ACCURACY = 0.6
MAX_BRIER_SCORE = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, dict) else None
    if not isinstance(records, list):
        raise TypeError("state source records are invalid")
    return [dict(record) for record in records]


def _state_key(dataset: str, seed: int, step: int) -> str:
    return f"{dataset}:{seed}:{step}"


def _matched_random_seed(dataset: str, seed: int, step: int) -> int:
    return int.from_bytes(
        hashlib.sha256(_state_key(dataset, seed, step).encode()).digest()[:4], "big"
    )


def _ranking_metrics(
    shortlist: Sequence[int],
    yields: Mapping[int, float],
    confidence: float | None,
) -> tuple[float, float, float | None]:
    proposed = list(shortlist)
    ideal = sorted(proposed, key=lambda index: (-yields[index], index))
    ranking_accuracy = float(
        np.mean([left == right for left, right in zip(proposed, ideal, strict=True)])
    )
    total = correct = 0
    for left_pos, left in enumerate(proposed):
        for right in proposed[left_pos + 1 :]:
            total += 1
            if yields[left] >= yields[right]:
                correct += 1
    pairwise = float(correct / total) if total else 1.0
    if confidence is None:
        return ranking_accuracy, pairwise, None
    probability = float(np.clip(confidence, 0.0, 1.0))
    selected_best = float(proposed[0] == ideal[0])
    return ranking_accuracy, pairwise, float((probability - selected_best) ** 2)


def _bootstrap_lcb(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    rng = np.random.RandomState(20260804)
    means = np.mean(rng.choice(array, size=(10_000, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025))




def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["dataset"]), int(record["seed"]))].append(record)

    result: dict[str, Any] = {}
    for (dataset, seed), rows in sorted(grouped.items()):
        brier_scores = [row["brier_score"] for row in rows if row["brier_score"] is not None]
        seed_metrics = {
            "state_count": len(rows),
            "steps": sorted(int(row["step"]) for row in rows),
            "confidence_coverage": float(len(brier_scores) / len(rows)),
            "ranking_accuracy": float(np.mean([row["ranking_accuracy"] for row in rows])),
            "pairwise_accuracy": float(np.mean([row["pairwise_accuracy"] for row in rows])),
            "delta_vs_gp": float(np.mean([row["selected_yield"] - row["gp_yield"] for row in rows])),
            "delta_vs_random": float(np.mean([row["selected_yield"] - row["random_yield"] for row in rows])),
            "brier_score": float(np.mean(brier_scores)) if brier_scores else None,
            "failure_rate": float(np.mean([row["selection_source"] == "gp" for row in rows])),
        }
        dataset_result = result.setdefault(dataset, {"seeds": {}})
        dataset_result["seeds"][str(seed)] = seed_metrics

    for dataset_result in result.values():
        seeds = list(dataset_result["seeds"].values())
        gp_deltas = [float(seed["delta_vs_gp"]) for seed in seeds]
        random_deltas = [float(seed["delta_vs_random"]) for seed in seeds]
        brier_scores = [float(seed["brier_score"]) for seed in seeds if seed["brier_score"] is not None]
        dataset_result["seed_count"] = len(seeds)
        dataset_result["mean_delta_vs_gp"] = float(np.mean(gp_deltas))
        dataset_result["mean_delta_vs_random"] = float(np.mean(random_deltas))
        dataset_result["delta_vs_gp_lcb"] = _bootstrap_lcb(gp_deltas)
        dataset_result["delta_vs_random_lcb"] = _bootstrap_lcb(random_deltas)
        dataset_result["mean_ranking_accuracy"] = float(np.mean([seed["ranking_accuracy"] for seed in seeds]))
        dataset_result["mean_pairwise_accuracy"] = float(np.mean([seed["pairwise_accuracy"] for seed in seeds]))
        dataset_result["mean_brier_score"] = float(np.mean(brier_scores)) if brier_scores else None
        dataset_result["failure_rate"] = float(np.mean([seed["failure_rate"] for seed in seeds]))
        dataset_result["confidence_coverage"] = float(np.mean([seed["confidence_coverage"] for seed in seeds]))
        dataset_result["state_manifest"] = sorted(
            (seed, step)
            for seed, metrics in dataset_result["seeds"].items()
            for step in metrics["steps"]
        )
    return result

def _gate_verdict(analysis: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    missing = REQUIRED_DATASETS - analysis.keys()
    if missing:
        reasons.append(f"missing datasets: {', '.join(sorted(missing))}")
    seed_sets = {
        dataset: set(values["seeds"])
        for dataset, values in analysis.items()
        if dataset in REQUIRED_DATASETS
    }
    if len(seed_sets) == len(REQUIRED_DATASETS) and len({frozenset(seeds) for seeds in seed_sets.values()}) != 1:
        reasons.append("target datasets must use the same paired seed set")
    state_manifests = {
        tuple(map(tuple, values["state_manifest"]))
        for dataset, values in analysis.items()
        if dataset in REQUIRED_DATASETS
    }
    if len(state_manifests) > 1:
        reasons.append("target datasets must use the same paired seed/step manifest")
    for dataset, seeds in seed_sets.items():
        if seeds != REQUIRED_SEEDS:
            reasons.append(f"{dataset} must cover seeds 100, 200, 300, 400, and 500")
    for dataset, values in analysis.items():
        metrics = {
            "GP improvement lower bound": values["delta_vs_gp_lcb"],
            "matched-random improvement lower bound": values["delta_vs_random_lcb"],
            "failure rate": values["failure_rate"],
            "ranking accuracy": values["mean_ranking_accuracy"],
            "pairwise accuracy": values["mean_pairwise_accuracy"],
            "confidence coverage": values["confidence_coverage"],
        }
        nonfinite = {name for name, value in metrics.items() if not math.isfinite(float(value))}
        brier = values["mean_brier_score"]
        if brier is not None and not math.isfinite(float(brier)):
            nonfinite.add("Brier score")
        if nonfinite:
            reasons.append(f"{dataset} has non-finite metrics: {', '.join(sorted(nonfinite))}")
            continue
        if int(values["seed_count"]) < 2:
            reasons.append(f"{dataset} requires at least two paired seeds")
        if float(values["delta_vs_gp_lcb"]) <= 0:
            reasons.append(f"{dataset} GP improvement lower bound is not positive")
        if float(values["delta_vs_random_lcb"]) <= 0:
            reasons.append(f"{dataset} matched-random improvement lower bound is not positive")
        if float(values["failure_rate"]) != 0:
            reasons.append(f"{dataset} has reranking failures")
        if float(values["mean_ranking_accuracy"]) < MIN_RANKING_ACCURACY:
            reasons.append(f"{dataset} ranking accuracy is below threshold")
        if float(values["mean_pairwise_accuracy"]) < MIN_PAIRWISE_ACCURACY:
            reasons.append(f"{dataset} pairwise accuracy is below threshold")
        if float(values["confidence_coverage"]) != 1:
            reasons.append(f"{dataset} confidence coverage is incomplete")
        if brier is None or float(brier) > MAX_BRIER_SCORE:
            reasons.append(f"{dataset} confidence calibration is below threshold")
    return {"passed": not reasons, "reasons": reasons}


def evaluate_records(
    state_source_path: Path,
    output_path: Path,
    *,
    state_keys: Sequence[tuple[str, int, int]],
    client_factory: Callable[[], Any],
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    if len(state_keys) != len(set(state_keys)):
        raise ValueError("duplicate state key")
    source_hash = _sha256(state_source_path)
    identity = {
        "source_sha256": source_hash,
        "model": model,
        "prompt_version": prompt_version,
        "shortlist_size": SHORTLIST_SIZE,
        "state_manifest": [list(key) for key in state_keys],
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("existing output provenance does not match")

    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in _records(state_source_path):
        if record.get("composition") != "chem_lgbo":
            continue
        key = (str(record["dataset"]), int(record["seed"]))
        if key in source_by_key:
            raise ValueError(f"duplicate state source record for {key!r}")
        source_by_key[key] = record

    output_records: list[dict[str, Any]] = []
    for dataset, seed, step in state_keys:
        try:
            source_record = source_by_key[(dataset, seed)]
        except KeyError as exc:
            raise ValueError(f"missing state source for {(dataset, seed)!r}") from exc
        engine = replay_engine_to_step(source_record, step)
        shortlist = ChemGPShortlistAdapter(engine).shortlist()
        gp_winner = shortlist[0]
        gate_artifact = {
            **identity,
            "dataset": dataset,
            "passed": True,
        }
        expected = {**identity, "dataset": dataset}
        client = client_factory()
        if str(getattr(client, "model", "")) != model:
            raise ValueError("LLM client model does not match artifact provenance")
        reranker = DeepSeekCandidateReranker(client, prompt_version)

        class FixedShortlist:
            def __init__(self, candidates: tuple[Any, ...]) -> None:
                self._candidates = candidates

            def shortlist(self) -> tuple[Any, ...]:
                return self._candidates

        result = SelectCandidateUseCase(
            FixedShortlist(shortlist),
            reranker,
            ArtifactEvidenceGate(gate_artifact, expected),
        ).execute()
        selected_index = result.selected.pool_index
        selected_yield = float(engine.pool_yield[selected_index])
        gp_yield = float(engine.pool_yield[gp_winner.pool_index])
        random_seed = _matched_random_seed(dataset, seed, step)
        random_index = int(
            np.random.RandomState(random_seed).choice(
                [candidate.pool_index for candidate in shortlist]
            )
        )
        random_yield = float(engine.pool_yield[random_index])
        ranked_ids = (
            result.proposal.ordered_ids
            if result.proposal is not None
            else tuple(candidate.pool_index for candidate in shortlist)
        )
        oracle_yields = {
            candidate.pool_index: float(engine.pool_yield[candidate.pool_index])
            for candidate in shortlist
        }
        confidence = (
            result.proposal.confidence[0]
            if result.proposal is not None and result.proposal.confidence
            else None
        )
        ranking_accuracy, pairwise_accuracy, brier_score = _ranking_metrics(
            ranked_ids, oracle_yields, confidence
        )
        output_records.append(
            {
                "state_key": _state_key(dataset, seed, step),
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "shortlist_indices": [candidate.pool_index for candidate in shortlist],
                "shortlist_scores": [candidate.acquisition_score for candidate in shortlist],
                "gp_index": gp_winner.pool_index,
                "gp_yield": gp_yield,
                "selected_index": selected_index,
                "selected_yield": selected_yield,
                "random_index": random_index,
                "random_yield": random_yield,
                "selection_source": result.source,
                "gate_reason": result.gate_reason,
                "fallback_reason": result.fallback_reason,
                "ranking_accuracy": ranking_accuracy,
                "pairwise_accuracy": pairwise_accuracy,
                "brier_score": brier_score,
                "oracle_not_sent": True,
            }
        )

    analysis = _aggregate(output_records)
    gate = _gate_verdict(analysis)
    artifact = {
        "schema_version": 1,
        **identity,
        "datasets": sorted(analysis),
        "passed": gate["passed"],
        "gate": gate,
        "records": output_records,
        "analysis": analysis,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return artifact


def _parse_state(value: str) -> tuple[str, int, int]:
    dataset, seed, step = value.split(":", 2)
    return dataset, int(seed), int(step)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--states", type=Path, required=True)
    preflight.add_argument("--model", required=True)
    preflight.add_argument("--prompt-version", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--states", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--state", action="append", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--prompt-version", required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--artifact", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "preflight":
        records = _records(args.states)
        if not records:
            raise ValueError("state source is empty")
        print(json.dumps({"source_sha256": _sha256(args.states), "record_count": len(records)}))
        return 0
    if args.command == "report":
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        report_value = {key: artifact.get(key) for key in ("passed", "datasets", "analysis")}
        args.output.write_text(json.dumps(report_value, indent=2), encoding="utf-8")
        return 0

    client = DeepSeekClient.from_env()
    evaluate_records(
        args.states,
        args.output,
        state_keys=[_parse_state(value) for value in args.state],
        client_factory=lambda: client,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
