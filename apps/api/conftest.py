"""Pytest config for apps/api: make `api` and `bo_core` importable without installing."""

import sys
from pathlib import Path

_api_dir = Path(__file__).resolve().parent
if str(_api_dir) not in sys.path:
    sys.path.insert(0, str(_api_dir))

# In dev, expose the local bo-core package without requiring `pip install -e`.
_pkg_root = _api_dir.parent.parent / "packages" / "bo-core"
if _pkg_root.is_dir() and str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
