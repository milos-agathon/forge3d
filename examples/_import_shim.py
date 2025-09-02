# Shared import shim: put "from _import_shim import ensure_repo_import" at top of examples
import sys
from pathlib import Path

def ensure_repo_import():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
