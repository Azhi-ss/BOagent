from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("export_snapshot.py")
SPEC = importlib.util.spec_from_file_location("export_snapshot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
export_snapshot = MODULE.export_snapshot
_check_destination_safety = MODULE._check_destination_safety
ROOT = MODULE.ROOT
RUN_SUBMISSION = MODULE.SUBMISSION / "code" / "main" / "run_submission.py"


def _load_run_submission(monkeypatch: pytest.MonkeyPatch):
    runner = types.ModuleType("bo_core.benchmark.lgbo_runner")
    runner.run_one = lambda **kwargs: {"best_found": 1.0}
    monkeypatch.setitem(sys.modules, "bo_core.benchmark.lgbo_runner", runner)
    spec = importlib.util.spec_from_file_location("run_submission", RUN_SUBMISSION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_full_matrix_detection_requires_exact_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_run_submission(monkeypatch)
    datasets = ["buchwald_sub4", "suzuki"]
    seeds = list(range(100, 2001, 100))
    assert module._is_complete_matrix(datasets, seeds, 40)
    assert not module._is_complete_matrix(datasets[:1], seeds, 40)
    assert not module._is_complete_matrix(datasets, seeds[:1], 40)
    assert not module._is_complete_matrix(datasets, seeds, 1)


def test_full_matrix_invokes_strict_evaluator_after_all_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_run_submission(monkeypatch)
    calls: list[tuple[str, int] | tuple[str, Path, Path, str]] = []
    monkeypatch.setattr(
        module,
        "run_one",
        lambda **kwargs: calls.append((kwargs["dataset"], kwargs["seed"]))
        or {"best_found": 1.0},
    )
    monkeypatch.setattr(
        module,
        "evaluate_all",
        lambda trajectories_path, results_path, backend: calls.append(
            ("verify", trajectories_path, results_path, backend)
        )
        or results_path / "summary_metrics.csv",
    )
    output_dir = tmp_path / "results" / "optimization_trajectories"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUN_SUBMISSION), "--backend", "sklearn", "--output-dir", str(output_dir)],
    )

    module.main()

    assert len(calls) == 41
    assert calls[-1] == ("verify", output_dir, output_dir.parent, "sklearn")



# ---------------------------------------------------------------------------
# Safety: dangerous destinations raise before any mutation
# ---------------------------------------------------------------------------


def test_destination_equal_to_repo_root_raises() -> None:
    with pytest.raises(ValueError, match="repository root"):
        _check_destination_safety(ROOT)


def test_destination_equal_to_repo_root_parent_raises() -> None:
    with pytest.raises(ValueError, match="refusing"):
        _check_destination_safety(ROOT.parent)


def test_destination_ancestor_of_repo_root_raises() -> None:
    # Find an ancestor that is NOT in the explicit named-path set so the
    # ancestor-of-root path is exercised rather than one of the named guards.
    named = {ROOT, ROOT.parent, Path.home(), Path("/")}
    ancestor = next((p for p in ROOT.parents if p not in named), ROOT.parent)
    with pytest.raises(ValueError, match="refusing"):
        _check_destination_safety(ancestor)


def test_export_snapshot_dangerous_destination_raises_before_mutation(
    tmp_path: Path,
) -> None:
    # Even when called through the public API, nothing on disk must change.
    sentinel = ROOT / ".snap_safety_canary_must_not_exist"
    assert not sentinel.exists(), "pre-existing canary — clean up manually"
    with pytest.raises(ValueError):
        export_snapshot(ROOT)
    assert not sentinel.exists()


def test_safe_destination_does_not_raise(tmp_path: Path) -> None:
    # A fresh subdir of tmp_path is always safe.
    _check_destination_safety(tmp_path / "snapshot")


# ---------------------------------------------------------------------------
# Atomic replacement: existing destination replaced only after full build
# ---------------------------------------------------------------------------


def test_existing_destination_replaced_atomically(tmp_path: Path) -> None:
    dest = tmp_path / "snapshot"
    # First export populates dest.
    export_snapshot(dest)
    assert dest.is_dir()
    # Plant a stale sentinel that must not survive the second export.
    stale = dest / "stale_marker.txt"
    stale.write_text("old")
    # Second export should atomically replace dest.
    export_snapshot(dest)
    assert dest.is_dir()
    assert not stale.exists(), "stale file survived — destination was not replaced"


# ---------------------------------------------------------------------------
# Excluded artifacts: generated / local state must not appear in the snapshot
# ---------------------------------------------------------------------------


@pytest.fixture()
def snapshot(tmp_path: Path) -> Path:
    return export_snapshot(tmp_path / "snapshot")


_EXCLUDED_NAMES = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "results",
    "tmp_results",
    "logs",
]


@pytest.mark.parametrize("name", _EXCLUDED_NAMES)
def test_generated_directory_excluded(snapshot: Path, name: str, tmp_path: Path) -> None:
    # Plant the artifact in the source tree temporarily, re-export, check absence.
    bo_core_src = ROOT / "packages" / "bo-core"
    planted = bo_core_src / name
    created = False
    if not planted.exists():
        planted.mkdir()
        created = True
    try:
        fresh = export_snapshot(tmp_path / f"snap_{name}")
        hits = list((fresh / "packages" / "bo-core").rglob(name))
        assert hits == [], f"Excluded artifact {name!r} found in snapshot: {hits}"
    finally:
        if created:
            planted.rmdir()


def test_pyc_files_excluded(tmp_path: Path) -> None:
    bo_core_src = ROOT / "packages" / "bo-core"
    fake_pyc = bo_core_src / "fake_module.pyc"
    fake_pyc.write_bytes(b"fake")
    try:
        snap = export_snapshot(tmp_path / "snapshot")
        hits = list((snap / "packages" / "bo-core").rglob("*.pyc"))
        assert hits == [], f".pyc files found in snapshot: {hits}"
    finally:
        fake_pyc.unlink(missing_ok=True)


def test_egg_info_excluded(tmp_path: Path) -> None:
    snap = export_snapshot(tmp_path / "snapshot")
    hits = list((snap / "packages" / "bo-core").rglob("*.egg-info"))
    assert hits == [], f"egg-info found in snapshot: {hits}"


def test_log_files_excluded(tmp_path: Path) -> None:
    bo_core_src = ROOT / "packages" / "bo-core"
    fake_log = bo_core_src / "test_run.log"
    fake_log.write_text("log content")
    try:
        snap = export_snapshot(tmp_path / "snapshot")
        hits = list((snap / "packages" / "bo-core").rglob("*.log"))
        assert hits == [], f".log files found in snapshot: {hits}"
    finally:
        fake_log.unlink(missing_ok=True)


def test_snapshot_exports_uv_dependency_contract(snapshot: Path) -> None:
    assert (snapshot / "pyproject.toml").read_bytes() == (ROOT / "pyproject.toml").read_bytes()
    assert (snapshot / "uv.lock").read_bytes() == (ROOT / "uv.lock").read_bytes()


def test_snapshot_install_and_container_contract_use_exported_manifest(snapshot: Path) -> None:
    pyproject = (snapshot / "pyproject.toml").read_text()
    lock = (snapshot / "uv.lock").read_text()
    dockerfile = (snapshot / "code" / "Dockerfile").read_text()

    assert 'members = ["packages/bo-core"]' in pyproject
    assert 'bo-core = { workspace = true }' in pyproject
    assert 'name = "boagent-workspace"' in lock
    assert 'name = "bo-core"' in lock
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "RUN uv sync --frozen --no-dev" in dockerfile
    assert 'CMD ["uv", "run", "--frozen", "--no-dev", "--no-sync"' in dockerfile




# ---------------------------------------------------------------------------
# Smoke: snapshot runs standalone without repository imports (existing contract)
# ---------------------------------------------------------------------------


def test_exported_snapshot_runs_without_repository_imports(tmp_path: Path) -> None:
    snapshot = export_snapshot(tmp_path / "snapshot")
    output_dir = snapshot / "results"
    completed = subprocess.run(
        [
            sys.executable,
            "code/main/run_submission.py",
            "--datasets",
            "buchwald_sub4",
            "--seeds",
            "100",
            "--n-iters",
            "1",
            "--backend",
            "sklearn",
            "--output-dir",
            str(output_dir),
        ],
        cwd=snapshot,
        env={"PYTHONPATH": str(snapshot / "packages" / "bo-core")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "sklearn" / "buchwald_sub4" / "lgbo" / "seed_100.pt").is_file()
    assert not (output_dir.parent / "summary_metrics.csv").exists()
    assert "SUBSET RUN COMPLETE" in completed.stdout
