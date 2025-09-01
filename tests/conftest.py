# Auto-bootstrap forge3d native module for `pytest -q` from a fresh clone.
# If `import forge3d` fails, we build it once via `maturin develop --release`.
# Set FORGE3D_NO_BOOTSTRAP=1 to disable this behavior (e.g., in CI where the wheel is preinstalled).
import os
import sys
import importlib
import subprocess
from pathlib import Path

def _build_with_maturin():
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # Ensure maturin is available
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "maturin"], check=True)
    # Build & develop-install in the current venv
    subprocess.run(["maturin", "develop", "--release"], cwd=str(repo), check=True, env=env)

def pytest_sessionstart(session):
    if os.environ.get("FORGE3D_NO_BOOTSTRAP") == "1":
        return
    try:
        import forge3d  # noqa: F401
        return
    except ModuleNotFoundError:
        _build_with_maturin()
        importlib.invalidate_caches()
        import forge3d  # noqa: F401