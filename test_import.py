import sys
from pathlib import Path
backend_path = Path('backend').absolute()
sys.path.insert(0, str(backend_path))
try:
    import optimization
    print("Successfully imported optimization")
    import optimization.optimizer
    print("Successfully imported optimization.optimizer")
except ImportError as e:
    print(f"Failed to import: {e}")
    print(f"sys.path: {sys.path}")
