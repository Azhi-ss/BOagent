import sys
from pathlib import Path

# Add backend directory to sys.path so tests can find 'api', 'optimization', and 'benchmark'
backend_dir = Path(__file__).parent.absolute()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
