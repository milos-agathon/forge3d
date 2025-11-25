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
    """Register custom markers for Workstream I tasks and P5.7."""
    config.addinivalue_line(
        "markers", "viewer: tests for interactive viewer functionality (Workstream I1)"
    )
    config.addinivalue_line(
        "markers", "offscreen: tests for offscreen rendering and Jupyter integration (Workstream I2)"
    )
    config.addinivalue_line(
        "markers", "opbr: tests for PBR rendering"
    )
    config.addinivalue_line(
        "markers", "olighting: tests for lighting"
    )
    config.addinivalue_line(
        "markers", "interactive_viewer: tests requiring interactive viewer"
    )
    config.addinivalue_line(
        "markers", "slow: slow tests"
    )


# Track P5.7 test results for artifact generation
_p57_results = {}


def pytest_runtest_logreport(report):
    """Track P5.7 test results."""
    if report.when == "call":
        # Check if this is a P5.7 test
        if "test_p5_ssao" in report.nodeid or "test_p5_ssgi" in report.nodeid or "test_p5_ssr" in report.nodeid:
            _p57_results[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    """Write p5_PASS.txt if all P5.7 tests passed."""
    import hashlib
    import json
    from datetime import datetime
    
    if not _p57_results:
        return
    
    # Check if all P5.7 tests passed (excluding skipped)
    passed = [k for k, v in _p57_results.items() if v == "passed"]
    failed = [k for k, v in _p57_results.items() if v == "failed"]
    skipped = [k for k, v in _p57_results.items() if v == "skipped"]
    
    repo_root = _repo_root()
    report_dir = repo_root / "reports" / "p5"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    if failed:
        # Write FAIL file
        fail_file = report_dir / "p5_FAIL.txt"
        with open(fail_file, "w") as f:
            f.write(f"P5.7 Acceptance Tests FAILED\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Failed: {len(failed)}\n")
            f.write(f"Skipped: {len(skipped)}\n")
            f.write("\nFailed tests:\n")
            for t in failed:
                f.write(f"  - {t}\n")
    elif passed:
        # Write PASS file with hashed metrics
        metrics = {
            "passed_count": len(passed),
            "skipped_count": len(skipped),
            "timestamp": datetime.now().isoformat(),
            "tests": passed,
        }
        metrics_hash = hashlib.sha256(
            json.dumps(metrics, sort_keys=True).encode()
        ).hexdigest()
        
        pass_file = report_dir / "p5_PASS.txt"
        with open(pass_file, "w") as f:
            f.write(f"P5.7 Acceptance Tests PASSED\n")
            f.write(f"Timestamp: {metrics['timestamp']}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Skipped: {len(skipped)}\n")
            f.write(f"Metrics hash: {metrics_hash}\n")
            f.write("\nPassed tests:\n")
            for t in passed:
                f.write(f"  - {t}\n")


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