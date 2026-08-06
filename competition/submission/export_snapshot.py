from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "competition" / "submission"
DATASETS = ("buchwald_sub4", "suzuki")

# Patterns excluded via copytree ignore — generated/local state that must not
# enter a reproducible snapshot.
_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    ".coverage",
    "*.log",
    "results",
    "tmp_results",
    "logs",
)


def _check_destination_safety(destination: Path) -> None:
    """Reject destinations that could overwrite repository source."""
    destination = destination.resolve()
    if destination == ROOT or destination in ROOT.parents:
        raise ValueError(
            f"Destination {destination} contains the repository root; refusing."
        )
    if destination.is_relative_to(ROOT):
        raise ValueError(
            f"Destination {destination} is inside the repository root; refusing."
        )


def export_snapshot(destination: Path) -> Path:
    destination = destination.resolve()
    _check_destination_safety(destination)

    # Build completely before replacing the destination. Moving an existing
    # destination aside keeps it recoverable if the final rename fails.
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(dir=parent, prefix=f".{destination.name}.tmp-")
    )
    backup = parent / f".{destination.name}.backup-{uuid4().hex}"
    try:
        _populate(tmp_dir)
        if destination.exists():
            destination.rename(backup)
        try:
            tmp_dir.rename(destination)
        except Exception:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return destination


def _populate(dest: Path) -> None:
    """Copy all reproducible source inputs into *dest*."""
    shutil.copytree(
        ROOT / "packages" / "bo-core",
        dest / "packages" / "bo-core",
        ignore=_IGNORE,
    )
    shutil.copy2(ROOT / "pyproject.toml", dest / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", dest / "uv.lock")
    shutil.copytree(SUBMISSION / "code", dest / "code", ignore=_IGNORE)
    for dataset in DATASETS:
        shutil.copytree(
            ROOT / "datasets" / "chemical_reactions" / dataset,
            dest / "datasets" / "chemical_reactions" / dataset,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standalone competition snapshot"
    )
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(export_snapshot(args.destination))


if __name__ == "__main__":
    main()
