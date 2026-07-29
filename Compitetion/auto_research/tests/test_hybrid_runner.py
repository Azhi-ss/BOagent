"""Failing tests for HybridComparisonRunner: resumable lgbo_manifold vs lgbo_dkl.

Matrix: 2 compositions x 2 datasets (buchwald_sub4, suzuki) x 20 seeds
(100..2000) = 80 runs, 40 iterations each, fixed_train_prior protocol.

These tests verify the runner contract WITHOUT executing real BO or hitting
the LLM API. HybridEngine and compute_metrics are monkeypatched to cheap
fakes. The runner must:
  1. persist each run atomically (temp file + atomic rename)
  2. resume by latest (composition, dataset, seed); retry failed only
  3. reject records with wrong prior_protocol, iteration count, or prior
  4. require successful-run diagnostics: every LLM outcome accounted for,
     zero not_configured / missing, and at most 10% failures
  5. validate the exact 80-run matrix
  6. aggregate / report only on valid records
  7. support a separately configurable preflight (1 seed x 1 iter)

Red phase: every test below FAILS until HybridComparisonRunner exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))

import components.library  # noqa: F401  # force component registration
from components.protocol import Composition
from compositions.base import get_base_compositions

# ---------------------------------------------------------------------------
# Fixtures: the matrix under test
# ---------------------------------------------------------------------------

EXPECTED_COMPOSITIONS = ["lgbo_manifold", "lgbo_dkl"]
EXPECTED_DATASETS = ["buchwald_sub4", "suzuki"]
EXPECTED_SEEDS = [i * 100 for i in range(1, 21)]  # 100..2000
EXPECTED_N_ITERS = 40
EXPECTED_MATRIX_SIZE = 80  # 2 comps * 2 datasets * 20 seeds


@pytest.fixture()
def comp_map() -> dict[str, Composition]:
    return {c.name: c for c in get_base_compositions()}


@pytest.fixture()
def tmp_outfile(tmp_path: Path) -> Path:
    """Per-test output path so no real history file is touched."""
    return tmp_path / "hybrid_comparison.json"


# ---------------------------------------------------------------------------
# Fakes: cheap stand-ins for HybridEngine + compute_metrics
# ---------------------------------------------------------------------------

def _make_diagnostics(
    n_iters: int,
    *,
    llm_attempts: int | None = None,
    llm_successes: int | None = None,
    llm_not_configured: int = 0,
    llm_diagnostics_missing: int = 0,
    llm_failures: int = 0,
    iterations_completed: int | None = None,
    **health_overrides: int,
) -> dict[str, Any]:
    """Build a diagnostics dict matching HybridEngine.diagnostics schema."""
    if llm_attempts is None:
        llm_attempts = n_iters
    if llm_successes is None:
        llm_successes = n_iters
    if iterations_completed is None:
        iterations_completed = n_iters
    health = {
        "crashed_iterations": 0,
        "gp_fit_failures": 0,
        "acquisition_fallbacks": 0,
        "constant_score_iterations": 0,
        "degenerate_score_iterations": 0,
        "nonfinite_acquisition_scores": 0,
        "mean_shift_actions": llm_successes,
        "mean_shift_failures": 0,
        "surrogate_fallback_fits": 0,
        **health_overrides,
    }
    return {
        "schema_version": 1,
        "composition": "fake",
        "dataset": "fake",
        "seed": 0,
        "summary": {
            "iterations_completed": iterations_completed,
            "iterations_recorded": n_iters,
            "llm_attempts": llm_attempts,
            "llm_successes": llm_successes,
            "llm_failures": llm_failures,
            "llm_not_configured": llm_not_configured,
            "llm_diagnostics_missing": llm_diagnostics_missing,
            **health,
        },
        "iterations": [{"step": i + 1, "status": "completed"} for i in range(n_iters)],
        "surrogate": None,
    }


def _make_trajectory(n_iters: int) -> list[dict[str, Any]]:
    return [
        {
            "step": i + 1,
            "observed_yield": float(60 + i),
            "acquisition": "ei",
            "llm_action": "mean_shift",
        }
        for i in range(n_iters)
    ]


def _make_ok_record(
    composition: str,
    dataset: str,
    seed: int,
    *,
    n_iters: int = EXPECTED_N_ITERS,
    prior_protocol: str = "fixed_train_prior",
    n_train_prior: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if n_train_prior is None:
        n_train_prior = 35 if dataset == "buchwald_sub4" else 29
    if diagnostics is None:
        diagnostics = _make_diagnostics(n_iters)
    return {
        "composition": composition,
        "dataset": dataset,
        "seed": seed,
        "prior_protocol": prior_protocol,
        "n_train_prior": n_train_prior,
        "initial_indices": list(range(n_train_prior)),
        "elapsed_s": 0.1,
        "metrics": {
            "best_found": 80.0,
            "initial_round_found_best": 60.0,
            "t95": 10,
            "AUC_best_so_far": 70.0,
        },
        "trajectory": _make_trajectory(n_iters),
        "diagnostics": diagnostics,
        "status": "ok",
    }


def _fake_engine_factory(n_iters: int, diagnostics_override: dict[str, Any] | None = None):
    """Return a fake HybridEngine constructor that yields a MagicMock engine."""
    diag = diagnostics_override or _make_diagnostics(n_iters)

    def _factory(composition, dataset, seed, n_iters=40, **kw):
        engine = MagicMock()
        engine.run.return_value = _make_trajectory(n_iters)
        engine.diagnostics = diag
        engine.initial_indices = tuple(range(35 if dataset == "buchwald_sub4" else 29))
        return engine

    return _factory


# ---------------------------------------------------------------------------
# 1. Import & class existence
# ---------------------------------------------------------------------------


def test_hybrid_comparison_runner_importable() -> None:
    """The runner module and class must exist and be importable."""
    from hybrid_runner import HybridComparisonRunner  # noqa: F401


def test_runner_exposes_matrix_configuration() -> None:
    """Runner must declare its fixed matrix constants."""
    from hybrid_runner import HybridComparisonRunner

    assert HybridComparisonRunner.COMPOSITIONS == EXPECTED_COMPOSITIONS
    assert HybridComparisonRunner.DATASETS == EXPECTED_DATASETS
    assert HybridComparisonRunner.SEEDS == EXPECTED_SEEDS
    assert HybridComparisonRunner.N_ITERS == EXPECTED_N_ITERS
    assert HybridComparisonRunner.MATRIX_SIZE == EXPECTED_MATRIX_SIZE
    assert HybridComparisonRunner.PRIOR_PROTOCOL == "fixed_train_prior"


# ---------------------------------------------------------------------------
# 2. Exact 80-run matrix validation
# ---------------------------------------------------------------------------


def test_matrix_covers_exact_80_runs() -> None:
    """Pending run set must be exactly 80 (composition, dataset, seed) tuples."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=Path("/tmp/nonexistent_matrix.json"))
    pending = runner.pending_keys()
    assert len(pending) == EXPECTED_MATRIX_SIZE
    # Every combination must be present
    for comp in EXPECTED_COMPOSITIONS:
        for ds in EXPECTED_DATASETS:
            for seed in EXPECTED_SEEDS:
                assert (comp, ds, seed) in pending


def test_matrix_is_paired_by_dataset_and_seed() -> None:
    """The two methods must run adjacently to limit API time drift."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=Path("/tmp/nonexistent_matrix.json"))
    pending = runner.pending_keys()

    for index in range(0, len(pending), len(EXPECTED_COMPOSITIONS)):
        pair = pending[index : index + len(EXPECTED_COMPOSITIONS)]
        assert [key[0] for key in pair] == EXPECTED_COMPOSITIONS
        assert len({(key[1], key[2]) for key in pair}) == 1


def test_matrix_empty_when_all_done(tmp_outfile: Path) -> None:
    """If all 80 records are persisted ok, pending_keys must be empty."""
    from hybrid_runner import HybridComparisonRunner

    all_records = [
        _make_ok_record(comp, ds, seed)
        for comp in EXPECTED_COMPOSITIONS
        for ds in EXPECTED_DATASETS
        for seed in EXPECTED_SEEDS
    ]
    tmp_outfile.write_text(json.dumps(all_records), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    assert runner.pending_keys() == []


# ---------------------------------------------------------------------------
# 3. Resume: only retry failed, deduplicate by latest key
# ---------------------------------------------------------------------------


def test_resume_skips_ok_records(tmp_outfile: Path) -> None:
    """Ok records from a prior run must not be retried."""
    from hybrid_runner import HybridComparisonRunner

    ok_records = [
        _make_ok_record("lgbo_manifold", "buchwald_sub4", 100),
        _make_ok_record("lgbo_manifold", "buchwald_sub4", 200),
    ]
    tmp_outfile.write_text(json.dumps(ok_records), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    pending = runner.pending_keys()
    assert ("lgbo_manifold", "buchwald_sub4", 100) not in pending
    assert ("lgbo_manifold", "buchwald_sub4", 200) not in pending
    # 80 - 2 = 78 remaining
    assert len(pending) == EXPECTED_MATRIX_SIZE - 2


def test_resume_retries_failed_records(tmp_outfile: Path) -> None:
    """Failed records must remain in the pending set for retry."""
    from hybrid_runner import HybridComparisonRunner

    failed = _make_ok_record("lgbo_dkl", "suzuki", 100)
    failed["status"] = "failed"
    failed["error"] = "RuntimeError: crash"
    failed.pop("metrics", None)
    tmp_outfile.write_text(json.dumps([failed]), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    assert ("lgbo_dkl", "suzuki", 100) in runner.pending_keys()


def test_resume_deduplicates_by_latest_key(tmp_outfile: Path) -> None:
    """When the same key appears twice, only the latest (last) record counts."""
    from hybrid_runner import HybridComparisonRunner

    ok_first = _make_ok_record("lgbo_manifold", "suzuki", 300)
    failed_retry = _make_ok_record("lgbo_manifold", "suzuki", 300)
    failed_retry["status"] = "failed"
    failed_retry["error"] = "crash"
    # Latest entry is failed -> must be retried
    tmp_outfile.write_text(json.dumps([ok_first, failed_retry]), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    assert ("lgbo_manifold", "suzuki", 300) in runner.pending_keys()


def test_resume_deduplicates_latest_ok_not_retried(tmp_outfile: Path) -> None:
    """When latest record for a key is ok, it must not be retried even if
    an earlier failed record exists for the same key."""
    from hybrid_runner import HybridComparisonRunner

    failed = _make_ok_record("lgbo_dkl", "buchwald_sub4", 500)
    failed["status"] = "failed"
    ok_retry = _make_ok_record("lgbo_dkl", "buchwald_sub4", 500)
    tmp_outfile.write_text(json.dumps([failed, ok_retry]), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    assert ("lgbo_dkl", "buchwald_sub4", 500) not in runner.pending_keys()


# ---------------------------------------------------------------------------
# 4. Atomic persistence
# ---------------------------------------------------------------------------


def test_run_persists_results_atomically(tmp_outfile: Path, comp_map) -> None:
    """Each completed run must be persisted via a temp file + atomic rename.
    No partial or corrupt JSON should be observable mid-write."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile, n_iters=1)
    runner._engine_factory = _fake_engine_factory(1)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_one(comp_map["lgbo_manifold"], "buchwald_sub4", seed=100)

    # File must exist and be valid JSON
    assert tmp_outfile.exists()
    data = json.loads(tmp_outfile.read_text())
    assert len(data) == 1
    assert data[0]["status"] == "ok"


def test_atomic_write_no_temp_file_left(tmp_outfile: Path, comp_map) -> None:
    """No leftover .tmp files after a successful run."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile, n_iters=1)
    runner._engine_factory = _fake_engine_factory(1)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_one(comp_map["lgbo_dkl"], "suzuki", seed=100)

    leftovers = list(tmp_outfile.parent.glob("*.tmp*"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# 5. Record validation: reject wrong protocol / iterations / prior
# ---------------------------------------------------------------------------


def test_validate_rejects_wrong_protocol() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    record["prior_protocol"] = "seeded_subsample"

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_wrong_iteration_count() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    record["diagnostics"] = _make_diagnostics(30)  # only 30, not 40

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_wrong_n_train_prior() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    record["n_train_prior"] = 5  # should be 35

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_suzuki_wrong_prior_size() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_dkl", "suzuki", 100)
    record["n_train_prior"] = 35  # should be 29

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_failed_status() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "suzuki", 200)
    record["status"] = "failed"

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_missing_diagnostics() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "suzuki", 200)
    record["diagnostics"] = None

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_trajectory_with_missing_fields() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "suzuki", 200)
    record["trajectory"][0].pop("observed_yield")

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_missing_trajectory() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "suzuki", 200)
    record.pop("trajectory")

    assert HybridComparisonRunner.is_valid_record(record) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gp_fit_failures", 1),
        ("acquisition_fallbacks", 1),
        ("degenerate_score_iterations", 1),
        ("nonfinite_acquisition_scores", 1),
        ("mean_shift_failures", 1),
        ("surrogate_fallback_fits", 1),
    ],
)
def test_validate_rejects_unhealthy_hybrid_evidence(field: str, value: int) -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_dkl", "buchwald_sub4", 100)
    record["diagnostics"] = _make_diagnostics(40, **{field: value})

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_missing_metric() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    record["metrics"].pop("AUC_best_so_far")

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_nonfinite_metric() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    record["metrics"]["best_found"] = float("nan")

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_accepts_correct_record() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_dkl", "buchwald_sub4", 100)
    assert HybridComparisonRunner.is_valid_record(record) is True


# ---------------------------------------------------------------------------
# 6. Diagnostics quality gate: 40 attempts / 40 successes / zero missing
# ---------------------------------------------------------------------------


def test_validate_rejects_insufficient_llm_attempts() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "suzuki", 100)
    record["diagnostics"] = _make_diagnostics(40, llm_attempts=30)

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_insufficient_llm_successes() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_dkl", "buchwald_sub4", 100)
    record["diagnostics"] = _make_diagnostics(40, llm_successes=30)

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_not_configured() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "suzuki", 100)
    record["diagnostics"] = _make_diagnostics(40, llm_not_configured=1)

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_diagnostics_missing() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_dkl", "suzuki", 100)
    record["diagnostics"] = _make_diagnostics(40, llm_diagnostics_missing=1)

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_high_failure_rate() -> None:
    """LLM failures above an acceptable threshold must be rejected."""
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    # 40 attempts, 30 successes, 10 failures = 25% failure rate
    record["diagnostics"] = _make_diagnostics(
        40, llm_attempts=40, llm_successes=30, llm_failures=10
    )

    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_accepts_acceptable_failure_rate() -> None:
    """A small number of LLM failures (below threshold) is acceptable."""
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_dkl", "suzuki", 100)
    # 40 attempts, 38 successes, 2 failures = 5% failure rate
    record["diagnostics"] = _make_diagnostics(
        40, llm_attempts=40, llm_successes=38, llm_failures=2
    )

    assert HybridComparisonRunner.is_valid_record(record) is True


# ---------------------------------------------------------------------------
# 7. Full run: all 80 with fakes, then aggregate only valid
# ---------------------------------------------------------------------------


def test_full_run_produces_80_valid_records(tmp_outfile: Path, comp_map) -> None:
    """A complete run over all 80 keys must produce 80 valid records."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    runner._engine_factory = _fake_engine_factory(EXPECTED_N_ITERS)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_all()

    data = json.loads(tmp_outfile.read_text())
    assert len(data) == EXPECTED_MATRIX_SIZE
    assert all(r["status"] == "ok" for r in data)
    assert all(HybridComparisonRunner.is_valid_record(r) for r in data)


def test_aggregate_only_uses_valid_records(tmp_outfile: Path, comp_map) -> None:
    """Aggregation must filter out invalid records before computing summary."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    runner._engine_factory = _fake_engine_factory(EXPECTED_N_ITERS)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_all()

    summary = runner.aggregate()
    # 2 compositions, 2 datasets each
    assert set(summary.keys()) == set(EXPECTED_COMPOSITIONS)
    for comp in EXPECTED_COMPOSITIONS:
        assert set(summary[comp].keys()) == set(EXPECTED_DATASETS)
        for ds in EXPECTED_DATASETS:
            entry = summary[comp][ds]
            assert entry["best_found"]["n"] == len(EXPECTED_SEEDS)


def test_aggregate_filters_invalid_records(tmp_outfile: Path, comp_map) -> None:
    """If some records are invalid, aggregation must exclude them (n < 20)."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    runner._engine_factory = _fake_engine_factory(EXPECTED_N_ITERS)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_all()

    # Corrupt one record: wrong iteration count
    data = json.loads(tmp_outfile.read_text())
    data[0]["diagnostics"] = _make_diagnostics(30)
    tmp_outfile.write_text(json.dumps(data), encoding="utf-8")

    runner2 = HybridComparisonRunner(output_path=tmp_outfile)
    summary = runner2.aggregate()
    # The corrupted record's (comp, dataset) should have n=19 instead of 20
    comp = data[0]["composition"]
    ds = data[0]["dataset"]
    assert summary[comp][ds]["best_found"]["n"] == len(EXPECTED_SEEDS) - 1


# ---------------------------------------------------------------------------
# 8. Preflight: separately configurable 1 seed x 1 iter
# ---------------------------------------------------------------------------


def test_preflight_uses_single_seed_single_iter(tmp_path: Path, comp_map) -> None:
    """Preflight mode must run exactly 1 seed x 1 iteration per (comp, dataset)."""
    from hybrid_runner import HybridComparisonRunner

    outfile = tmp_path / "preflight.json"
    runner = HybridComparisonRunner.preflight(output_path=outfile)

    assert runner.SEEDS == [100]
    assert runner.N_ITERS == 1
    assert runner.DATASETS == EXPECTED_DATASETS
    assert runner.COMPOSITIONS == EXPECTED_COMPOSITIONS
    # Preflight matrix = 2 comps * 2 datasets * 1 seed = 4
    assert runner.MATRIX_SIZE == 4


def test_preflight_executes_only_4_keys(tmp_path: Path, comp_map) -> None:
    from hybrid_runner import HybridComparisonRunner

    outfile = tmp_path / "preflight.json"
    runner = HybridComparisonRunner.preflight(output_path=outfile)
    runner._engine_factory = _fake_engine_factory(1)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_all()

    data = json.loads(outfile.read_text())
    assert len(data) == 4
    assert all(r["diagnostics"]["summary"]["iterations_recorded"] == 1 for r in data)


def test_preflight_does_not_pollute_full_matrix(tmp_path: Path, comp_map) -> None:
    """A preflight run must not be treated as completion of the full matrix."""
    from hybrid_runner import HybridComparisonRunner

    preflight_out = tmp_path / "preflight.json"
    full_out = tmp_path / "full.json"

    preflight_runner = HybridComparisonRunner.preflight(output_path=preflight_out)
    preflight_runner._engine_factory = _fake_engine_factory(1)
    with patch("hybrid_runner.HybridEngine", side_effect=preflight_runner._engine_factory):
        preflight_runner.run_all()

    full_runner = HybridComparisonRunner(output_path=full_out)
    pending = full_runner.pending_keys()
    assert len(pending) == EXPECTED_MATRIX_SIZE


# ---------------------------------------------------------------------------
# 9. Resume after partial: continue from where it left off
# ---------------------------------------------------------------------------


def test_resume_rejects_record_with_non_numeric_seed(
    tmp_outfile: Path,
    comp_map,
) -> None:
    """Malformed persisted records must fail with a schema error, not int()."""
    from hybrid_runner import HybridComparisonRunner

    malformed = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    malformed["seed"] = None
    tmp_outfile.write_text(json.dumps([malformed]), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)

    with pytest.raises(TypeError, match="Invalid result record key"):
        runner.pending_keys()


def test_resume_after_partial_run(tmp_outfile: Path, comp_map) -> None:
    """After completing 4 runs, resuming must only run the remaining 76."""
    from hybrid_runner import HybridComparisonRunner

    # Seed file with 4 ok records
    initial = [
        _make_ok_record("lgbo_manifold", "buchwald_sub4", 100),
        _make_ok_record("lgbo_manifold", "suzuki", 100),
        _make_ok_record("lgbo_dkl", "buchwald_sub4", 100),
        _make_ok_record("lgbo_dkl", "suzuki", 100),
    ]
    tmp_outfile.write_text(json.dumps(initial), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    runner._engine_factory = _fake_engine_factory(EXPECTED_N_ITERS)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_all()

    data = json.loads(tmp_outfile.read_text())
    assert len(data) == EXPECTED_MATRIX_SIZE
    ok_count = sum(1 for r in data if r["status"] == "ok")
    assert ok_count == EXPECTED_MATRIX_SIZE


def test_resume_appends_not_overwrites(tmp_outfile: Path, comp_map) -> None:
    """Resumed runs must append results, preserving prior ok records."""
    from hybrid_runner import HybridComparisonRunner

    prior_ok = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    tmp_outfile.write_text(json.dumps([prior_ok]), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    runner._engine_factory = _fake_engine_factory(EXPECTED_N_ITERS)

    with patch("hybrid_runner.HybridEngine", side_effect=runner._engine_factory):
        runner.run_all()

    data = json.loads(tmp_outfile.read_text())
    assert len(data) == EXPECTED_MATRIX_SIZE
    # The original record must still be present
    found = [r for r in data if r["seed"] == 100 and r["dataset"] == "buchwald_sub4"
             and r["composition"] == "lgbo_manifold"]
    assert len(found) == 1


# ---------------------------------------------------------------------------
# 10. Diagnostics validation helper
# ---------------------------------------------------------------------------


def test_diagnostics_check_requires_40_attempts() -> None:
    from hybrid_runner import HybridComparisonRunner

    diag = _make_diagnostics(40, llm_attempts=39, llm_successes=40)
    assert HybridComparisonRunner._diagnostics_ok(diag) is False


def test_diagnostics_check_requires_40_successes() -> None:
    from hybrid_runner import HybridComparisonRunner

    diag = _make_diagnostics(40, llm_attempts=40, llm_successes=39)
    assert HybridComparisonRunner._diagnostics_ok(diag) is False


def test_diagnostics_check_rejects_zero_not_configured() -> None:
    from hybrid_runner import HybridComparisonRunner

    diag = _make_diagnostics(40, llm_not_configured=1)
    assert HybridComparisonRunner._diagnostics_ok(diag) is False


def test_diagnostics_check_rejects_zero_missing() -> None:
    from hybrid_runner import HybridComparisonRunner

    diag = _make_diagnostics(40, llm_diagnostics_missing=1)
    assert HybridComparisonRunner._diagnostics_ok(diag) is False


def test_diagnostics_check_rejects_excessive_failures() -> None:
    from hybrid_runner import HybridComparisonRunner

    diag = _make_diagnostics(
        40, llm_attempts=40, llm_successes=30, llm_failures=10
    )
    assert HybridComparisonRunner._diagnostics_ok(diag) is False


def test_diagnostics_check_accepts_clean_run() -> None:
    from hybrid_runner import HybridComparisonRunner

    diag = _make_diagnostics(40)
    assert HybridComparisonRunner._diagnostics_ok(diag) is True


# ---------------------------------------------------------------------------
# 11. Environment, failure persistence, report, and status gates
# ---------------------------------------------------------------------------


def test_environment_gate_rejects_unconfigured_client(monkeypatch) -> None:
    from hybrid_runner import HybridComparisonRunner

    client = MagicMock()
    client.is_configured.return_value = False
    monkeypatch.setattr(
        "bo_core.llm_client.DeepSeekClient.from_env",
        lambda: client,
    )

    with pytest.raises(RuntimeError, match="not configured"):
        HybridComparisonRunner.verify_environment()


def test_run_one_persists_engine_failure(tmp_outfile: Path, comp_map) -> None:
    from hybrid_runner import HybridComparisonRunner

    def failing_factory(*args, **kwargs):
        raise RuntimeError("surrogate exploded")

    runner = HybridComparisonRunner(output_path=tmp_outfile, n_iters=1)
    runner._engine_factory = failing_factory

    record = runner.run_one(comp_map["lgbo_dkl"], "buchwald_sub4", seed=100)

    persisted = json.loads(tmp_outfile.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "surrogate exploded" in record["error"]
    assert persisted == [record]


def test_run_one_persists_trajectory(tmp_outfile: Path, comp_map) -> None:
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile, n_iters=1)
    runner._engine_factory = _fake_engine_factory(1)

    record = runner.run_one(comp_map["lgbo_manifold"], "buchwald_sub4", seed=100)

    assert record["trajectory"] == _make_trajectory(1)


def test_diagnostic_totals_use_persisted_trajectories() -> None:
    from hybrid_runner import HybridComparisonRunner

    records = [
        _make_ok_record("lgbo_manifold", "buchwald_sub4", 100, n_iters=3),
        _make_ok_record("lgbo_manifold", "suzuki", 100, n_iters=3),
    ]

    totals = HybridComparisonRunner._diagnostic_totals(records, "lgbo_manifold")

    assert totals["n_improvements"] == 4
    assert totals["llm_action_count"] == 6
    assert totals["acq_switches"] == 0
    assert totals["final_best"] == 62.0


def test_aggregate_rejects_when_no_valid_records(tmp_outfile: Path) -> None:
    from hybrid_runner import HybridComparisonRunner

    failed = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    failed["status"] = "failed"
    tmp_outfile.write_text(json.dumps([failed]), encoding="utf-8")

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    with pytest.raises(ValueError, match="No valid"):
        runner.aggregate()


def test_report_rejects_incomplete_matrix(tmp_outfile: Path) -> None:
    from hybrid_runner import HybridComparisonRunner

    tmp_outfile.write_text(
        json.dumps([_make_ok_record("lgbo_manifold", "buchwald_sub4", 100)]),
        encoding="utf-8",
    )

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    with pytest.raises(ValueError, match="incomplete"):
        runner.write_final_report(tmp_outfile.parent / "reports")


def test_status_reports_valid_and_pending_counts(tmp_path: Path) -> None:
    from hybrid_runner import HybridComparisonRunner

    outfile = tmp_path / "preflight.json"
    status_path = tmp_path / "status.json"
    runner = HybridComparisonRunner.preflight(outfile, status_path)
    runner._engine_factory = _fake_engine_factory(1)

    runner.run_all()

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    assert status["mode"] == "preflight"
    assert status["valid_runs"] == 4
    assert status["required_runs"] == 4
    assert status["pending_runs"] == 0


# ---------------------------------------------------------------------------
# 12. Diagnostics schema rejection
# ---------------------------------------------------------------------------


def test_diagnostics_check_rejects_missing_summary() -> None:
    from hybrid_runner import HybridComparisonRunner

    assert HybridComparisonRunner._diagnostics_ok({}) is False


def test_validate_rejects_unknown_composition() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("unknown", "buchwald_sub4", 100)
    assert HybridComparisonRunner.is_valid_record(record) is False


def test_validate_rejects_noncontiguous_fixed_prior() -> None:
    from hybrid_runner import HybridComparisonRunner

    record = _make_ok_record("lgbo_manifold", "buchwald_sub4", 100)
    record["initial_indices"] = list(range(1, 36))
    assert HybridComparisonRunner.is_valid_record(record) is False


def test_invalid_result_file_schema_is_rejected(tmp_outfile: Path) -> None:
    from hybrid_runner import HybridComparisonRunner

    tmp_outfile.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    runner = HybridComparisonRunner(output_path=tmp_outfile)

    with pytest.raises(ValueError, match="Invalid result file schema"):
        runner.pending_keys()


# ---------------------------------------------------------------------------
# 13. Multi-worker seed-parallel execution
# ---------------------------------------------------------------------------


def test_runner_defaults_to_single_worker(tmp_outfile: Path) -> None:
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    assert runner.workers == 1


def test_configure_numerical_threads_sets_env_and_torch(monkeypatch) -> None:
    from hybrid_runner import THREAD_ENV_VARS, _configure_numerical_threads

    calls: list[int] = []

    class _FakeTorch:
        @staticmethod
        def set_num_threads(n: int) -> None:
            calls.append(n)

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch)
    for name in THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    _configure_numerical_threads(1)

    for name in THREAD_ENV_VARS:
        assert __import__("os").environ[name] == "1"
    assert calls == [1]


def test_locked_persist_merges_concurrent_writes(tmp_outfile: Path) -> None:
    """Two concurrent writers must not clobber each other's latest records."""
    import threading

    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _write(composition: str, seed: int) -> None:
        try:
            barrier.wait(timeout=5)
            runner._persist_latest(
                _make_ok_record(composition, "buchwald_sub4", seed)
            )
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=_write, args=("lgbo_manifold", 100)),
        threading.Thread(target=_write, args=("lgbo_dkl", 100)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    records = json.loads(tmp_outfile.read_text(encoding="utf-8"))
    keys = {(r["composition"], r["dataset"], r["seed"]) for r in records}
    assert keys == {
        ("lgbo_manifold", "buchwald_sub4", 100),
        ("lgbo_dkl", "buchwald_sub4", 100),
    }


def test_run_all_with_workers_uses_process_pool(
    tmp_outfile: Path,
    comp_map,
    monkeypatch,
) -> None:
    """workers>1 must fan out pending seed groups via ProcessPoolExecutor."""
    from hybrid_runner import HybridComparisonRunner

    runner = HybridComparisonRunner(output_path=tmp_outfile, workers=4)
    runner._engine_factory = _fake_engine_factory(EXPECTED_N_ITERS)

    submitted: list[tuple[str, int, tuple[str, ...]]] = []

    class _FakeFuture:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def result(self) -> dict[str, Any]:
            return self._payload

    class _FakePool:
        def __init__(self, *args, **kwargs) -> None:
            self.max_workers = kwargs.get("max_workers")
            self.initializer = kwargs.get("initializer")
            self.initargs = kwargs.get("initargs")

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def submit(self, fn, *args):
            # args: output, status, n_iters, seeds, datasets, dataset, seed, compositions
            dataset = args[5]
            seed = args[6]
            compositions = tuple(args[7])
            submitted.append((dataset, seed, compositions))
            return _FakeFuture(
                [
                    {
                        "composition": name,
                        "dataset": dataset,
                        "seed": seed,
                        "status": "ok",
                        "metrics": {
                            "best_found": 1.0,
                            "t95": 1,
                        },
                    }
                    for name in compositions
                ]
            )

    def _fake_as_completed(futures):
        return list(futures)

    monkeypatch.setattr(
        "hybrid_runner.ProcessPoolExecutor",
        _FakePool,
    )
    monkeypatch.setattr(
        "hybrid_runner.as_completed",
        _fake_as_completed,
    )

    runner.run_all()

    assert submitted
    # First pending group for a fresh matrix: buchwald_sub4 / seed100 / both comps
    assert submitted[0] == (
        "buchwald_sub4",
        100,
        ("lgbo_manifold", "lgbo_dkl"),
    )
    assert len(submitted) == 40  # 2 datasets x 20 seeds


def test_cli_run_accepts_workers_flag(monkeypatch, tmp_path: Path) -> None:
    from hybrid_runner import main

    calls: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            calls["init_kwargs"] = kwargs
            self._complete = True

        @classmethod
        def preflight(cls, *args, **kwargs):
            return cls()

        @staticmethod
        def verify_environment() -> None:
            return None

        def is_complete(self) -> bool:
            return True

        def run_all(self) -> None:
            calls["ran"] = True

        def write_final_report(self, output_dir: Path) -> Path:
            path = output_dir / "report.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")
            return path

        def _write_status(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr("hybrid_runner.HybridComparisonRunner", _FakeRunner)
    monkeypatch.setattr(
        "hybrid_runner._default_paths",
        lambda: (
            tmp_path / "preflight.json",
            tmp_path / "full.json",
            tmp_path / "status.json",
            tmp_path / "reports",
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["hybrid_runner.py", "run", "--workers", "4"],
    )

    assert main() == 0
    assert calls.get("ran") is True
    assert calls["init_kwargs"].get("workers") == 4
