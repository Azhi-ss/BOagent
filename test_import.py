import sys
from pathlib import Path

# Add packages/bo-core to sys.path so `bo_core` is importable without install.
_pkg_root = Path(__file__).resolve().parent / "packages" / "bo-core"
sys.path.insert(0, str(_pkg_root))

try:
    import bo_core
    print("Successfully imported bo_core")
    import bo_core.optimization.optimizer
    print("Successfully imported bo_core.optimization.optimizer")
    import bo_core.benchmark.runner
    print("Successfully imported bo_core.benchmark.runner")
except ImportError as e:
    print(f"Failed to import: {e}")
    print(f"sys.path: {sys.path}")
