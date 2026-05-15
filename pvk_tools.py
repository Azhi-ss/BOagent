from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any


BOAGENT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_CSV_PATH = BOAGENT_ROOT / "demo_optimization_table.csv"
DEFAULT_PVK_PROJECT_ROOT = BOAGENT_ROOT.parent / "PVK-LLM"
PASSIVATORS = ("3MTPAI", "PDAI2", "EDAI2", "PipDI")


def list_tasks(
    pvk_project_root: str | Path = DEFAULT_PVK_PROJECT_ROOT,
    demo_csv_path: str | Path = DEFAULT_DEMO_CSV_PATH,
) -> list[dict[str, Any]]:
    dataset = load_pvk_dataset_or_demo(pvk_project_root, demo_csv_path)
    tasks = [
        {
            "task_id": "passivation_demo",
            "name": "Passivation formulation demo",
            "objective": "Maximize perovskite solar-cell PCE using a PVK-LLM-style BO loop.",
            "data_source": dataset["data_source"],
            "data_boundary": dataset["data_boundary"],
            "record_count": len(dataset["records"]),
        }
    ]

    xlsx_files = _find_pvk_xlsx_files(Path(pvk_project_root))
    if xlsx_files:
        tasks.extend(
            [
                {
                    "task_id": "band_alignment",
                    "name": "Band alignment optimization",
                    "objective": "Maximize eta over PVK/HTL/ETL band-alignment features.",
                    "data_source": "PVK-LLM custom_perovskite_dataset",
                    "data_boundary": "Original PVK-LLM Excel data is available; runtime still uses deterministic demo acquisition unless wired to live LLM calls.",
                    "record_count": None,
                },
                {
                    "task_id": "defects_doping",
                    "name": "Defects and doping optimization",
                    "objective": "Maximize eta over defect-density and doping features.",
                    "data_source": "PVK-LLM custom_perovskite_dataset",
                    "data_boundary": "Original PVK-LLM Excel data is available; runtime still uses deterministic demo acquisition unless wired to live LLM calls.",
                    "record_count": None,
                },
            ]
        )
    return tasks


def load_pvk_dataset_or_demo(
    pvk_project_root: str | Path = DEFAULT_PVK_PROJECT_ROOT,
    demo_csv_path: str | Path = DEFAULT_DEMO_CSV_PATH,
) -> dict[str, Any]:
    pvk_root = Path(pvk_project_root)
    xlsx_files = _find_pvk_xlsx_files(pvk_root)
    if xlsx_files:
        records = _load_xlsx_records(xlsx_files[0])
        return {
            "task_id": "passivation_demo",
            "records": records,
            "data_source": f"PVK-LLM:{xlsx_files[0].name}",
            "data_boundary": "Using original PVK-LLM custom_perovskite_dataset Excel data; demo runtime uses deterministic/mock acquisition, not live LLM BO.",
            "source_path": str(xlsx_files[0]),
        }

    demo_path = Path(demo_csv_path)
    records = _load_demo_csv_records(demo_path)
    return {
        "task_id": "passivation_demo",
        "records": records,
        "data_source": demo_path.name,
        "data_boundary": "PVK-LLM custom_perovskite_dataset/*.xlsx not found; fallback to BOagent demo_optimization_table.csv. Demo data is literature-extracted/mixed-system evidence and not validated BO performance.",
        "source_path": str(demo_path),
    }


def build_task_context(dataset: dict[str, Any], language: str = "zh") -> dict[str, Any]:
    return {
        "task_id": dataset["task_id"],
        "task": "passivation formulation optimization",
        "model": "PVK-LLM-demo-runtime",
        "metric": "PCE_percent",
        "lower_is_better": False,
        "language": language,
        "data_source": dataset["data_source"],
        "data_boundary": dataset["data_boundary"],
        "hyperparameter_constraints": {
            "passivator_combo": ["categorical", _unique_values(dataset["records"], "passivator_combo")],
            "solvent": ["categorical", _unique_values(dataset["records"], "solvent")],
            "spin_speed_rpm": ["int", _numeric_range(dataset["records"], "spin_speed_rpm")],
            "spin_time_s": ["float", _numeric_range(dataset["records"], "spin_time_s")],
            "anneal_temp_C": ["float", _numeric_range(dataset["records"], "anneal_temp_C")],
            "anneal_time_min": ["float", _numeric_range(dataset["records"], "anneal_time_min")],
        },
    }


def initialize_observations(
    dataset: dict[str, Any], n_initial: int = 5, seed: int = 42
) -> tuple[list[dict[str, Any]], list[float]]:
    records = list(dataset["records"])
    if not records:
        return [], []

    ordered = sorted(records, key=lambda record: record["score"], reverse=True)
    rng = random.Random(seed)
    top_count = min(len(ordered), max(1, min(2, n_initial)))
    top = ordered[:top_count]
    rest = ordered[top_count:]
    rng.shuffle(rest)
    selected = (top + rest)[: max(0, min(n_initial, len(records)))]
    configs = [_record_to_config(record) for record in selected]
    fvals = [float(record["score"]) for record in selected]
    return configs, fvals


def generate_candidate_points(
    task_context: dict[str, Any],
    dataset: dict[str, Any],
    observed_configs: list[dict[str, Any]],
    observed_fvals: list[float],
    seed: int = 42,
    current_step: int = 0,
    n_candidates: int = 5,
) -> list[dict[str, Any]]:
    del task_context
    observed_ids = {config.get("experiment_id") for config in observed_configs}
    rng = random.Random(seed + current_step)
    candidates = [
        record for record in dataset["records"] if record.get("experiment_id") not in observed_ids
    ]
    if not candidates:
        candidates = list(dataset["records"])
    rng.shuffle(candidates)

    best_seen = max(observed_fvals) if observed_fvals else 0.0
    ranked = []
    for rank, record in enumerate(sorted(candidates, key=lambda item: item["score"], reverse=True)):
        exploration_bonus = 0.03 * ((rank + current_step) % 3)
        improvement = max(0.0, float(record["score"]) - best_seen) / 100.0
        acquisition_score = round(float(record["score"]) / 100.0 + improvement + exploration_bonus, 4)
        point = _record_to_config(record)
        point.update(
            {
                "candidate_id": f"CAND-{current_step + 1:02d}-{rank + 1:02d}",
                "mock_acquisition_score": acquisition_score,
                "acquisition_function": "LLM_ACQ expected improvement + diversity mock",
            }
        )
        ranked.append(point)
    return sorted(ranked, key=lambda item: item["mock_acquisition_score"], reverse=True)[
        : max(1, n_candidates)
    ]


def select_query_point(
    candidate_points: list[dict[str, Any]],
    observed_configs: list[dict[str, Any]],
    observed_fvals: list[float],
) -> dict[str, Any]:
    del observed_configs, observed_fvals
    if not candidate_points:
        raise ValueError("candidate_points must not be empty")
    return max(candidate_points, key=lambda point: point["mock_acquisition_score"])


def evaluate_candidate(candidate: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    experiment_id = candidate.get("experiment_id")
    record = next(
        (
            item
            for item in dataset["records"]
            if item.get("experiment_id") == experiment_id
        ),
        None,
    )
    score = float(record["score"] if record else candidate.get("mock_acquisition_score", 0.0) * 25.0)
    return {
        "score": round(score, 4),
        "metric": "PCE_percent",
        "result": {
            "PCE_percent": round(score, 4),
            "Voc_V": _safe_float((record or {}).get("Voc_V")),
            "Jsc_mA_cm2": _safe_float((record or {}).get("Jsc_mA_cm2")),
            "FF_percent": _safe_float((record or {}).get("FF_percent")),
            "evaluation_mode": "deterministic_black_box_lookup",
            "data_boundary": dataset["data_boundary"],
        },
    }


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "task_id": session["task"]["task_id"],
        "data_source": session["task"]["data_source"],
        "data_boundary": session["task"]["data_boundary"],
        "current_step": session["current_step"],
        "best_score": session["best_result"]["score"],
        "best_result": session["best_result"],
        "observed_count": len(session["observed_configs"]),
        "observed_fvals": list(session["observed_fvals"]),
        "candidate_count": len(session["candidate_points"]),
    }


def _find_pvk_xlsx_files(pvk_project_root: Path) -> list[Path]:
    dataset_dir = pvk_project_root / "custom_perovskite_dataset"
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("*.xlsx"))


def _load_demo_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_normalize_record(row, index) for index, row in enumerate(reader)]


def _load_xlsx_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to load PVK-LLM Excel datasets") from exc

    frame = pd.read_excel(path)
    return [
        _normalize_record(
            {str(key): value for key, value in row.items()},
            index,
        )
        for index, row in enumerate(frame.to_dict("records"))
    ]


def _normalize_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    score = _safe_float(row.get("PCE_percent", row.get("PCE", row.get("eta"))), 0.0)
    experiment_id = str(row.get("experiment_id") or row.get("id") or f"PVK-{index + 1:04d}")
    passivator_combo = _clean_text(row.get("passivator_combo") or row.get("passivator_system"))
    normalized = {
        "experiment_id": experiment_id,
        "passivator_combo": passivator_combo or "Not specified",
        "perovskite_system": _clean_text(row.get("perovskite_system")),
        "device_config": _clean_text(row.get("device_config")),
        "solvent": _clean_text(row.get("solvent")) or "Not specified",
        "spin_speed_rpm": _safe_float(row.get("spin_speed_rpm")),
        "spin_time_s": _safe_float(row.get("spin_time_s")),
        "anneal_temp_C": _safe_float(row.get("anneal_temp_C")),
        "anneal_time_min": _safe_float(row.get("anneal_time_min")),
        "score": score,
        "Voc_V": _safe_float(row.get("Voc_V")),
        "Jsc_mA_cm2": _safe_float(row.get("Jsc_mA_cm2")),
        "FF_percent": _safe_float(row.get("FF_percent")),
        "evidence_text": _clean_text(row.get("evidence_text")),
    }
    for passivator in PASSIVATORS:
        normalized[f"has_{passivator}"] = int(
            (_safe_float(row.get(f"has_{passivator}"), 0.0) or 0.0) > 0
        )
    return normalized


def _record_to_config(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": record["experiment_id"],
        "passivator_combo": record["passivator_combo"],
        "perovskite_system": record["perovskite_system"],
        "device_config": record["device_config"],
        "solvent": record["solvent"],
        "spin_speed_rpm": record["spin_speed_rpm"],
        "spin_time_s": record["spin_time_s"],
        "anneal_temp_C": record["anneal_temp_C"],
        "anneal_time_min": record["anneal_time_min"],
    }


def _unique_values(records: list[dict[str, Any]], key: str) -> list[str]:
    values = sorted(
        {
            str(record.get(key))
            for record in records
            if record.get(key) not in (None, "", "Not specified")
        }
    )
    return values or ["Not specified"]


def _numeric_range(records: list[dict[str, Any]], key: str) -> list[float]:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return [0.0, 0.0]
    return [min(values), max(values)]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number
