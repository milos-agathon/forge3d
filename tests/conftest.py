# Ensure `import forge3d` works from a fresh clone:
# 1) Put repo/python on sys.path so the package is importable without prior install.
# 2) If the native extension is missing, auto-build once via maturin develop --release.
# Set FORGE3D_NO_BOOTSTRAP=1 to disable autobuild (e.g., in CI with preinstalled wheel).
import os
import sys
import subprocess
import importlib
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _ensure_python_path():
    repo = _repo_root()
    pkg_dir = repo / "python"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))

def _install_maturin():
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "maturin"], check=True)

def _maturin_develop():
    repo = _repo_root()
    env = os.environ.copy()
    # Keep output concise in CI logs
    subprocess.run(
        ["maturin", "develop", "--release"],
        cwd=str(repo),
        env=env,
        check=True,
    )

def _needs_build_from_exc(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        isinstance(exc, ModuleNotFoundError) and "forge3d" in msg
    ) or (
        isinstance(exc, ImportError) and ("_forge3d" in msg or "forge3d._forge3d" in msg)
    )

def pytest_configure(config):
    """Register custom markers for Workstream I tasks."""
    config.addinivalue_line(
        "markers", "viewer: tests for interactive viewer functionality (Workstream I1)"
    )
    config.addinivalue_line(
        "markers", "offscreen: tests for offscreen rendering and Jupyter integration (Workstream I2)"
    )

def pytest_sessionstart(session):
    if os.environ.get("FORGE3D_NO_BOOTSTRAP") == "1":
        return
    _ensure_python_path()
    try:
        import forge3d  # noqa: F401
        return
    except Exception as exc:
        if not _needs_build_from_exc(exc):
            # Unexpected error - rethrow so pytest shows the real issue.
            raise
        try:
            _install_maturin()
            _maturin_develop()
            importlib.invalidate_caches()
            _ensure_python_path()
            import forge3d  # noqa: F401
            print("forge3d bootstrap: built via maturin", flush=True)
        except Exception as build_exc:
            # Provide a clear, actionable message then rethrow.
            raise RuntimeError(
                "forge3d bootstrap failed. Ensure Rust toolchain and (on Windows) "
                "MSVC Build Tools are installed. Original error: {}".format(build_exc)
            ) from build_exc