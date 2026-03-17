# Ensure `import forge3d` works from a fresh clone:
# 1) Put repo/python on sys.path so the package is importable without prior install.
# 2) If the native extension is missing, auto-build once via maturin develop --release.
# Set FORGE3D_NO_BOOTSTRAP=1 to disable autobuild (e.g., in CI with preinstalled wheel).
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_python_path():
    repo = _repo_root()
    pkg_dir = repo / "python"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    # Also make test helpers (e.g. _license_test_keys) importable.
    tests_dir = repo / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))


def _install_maturin():
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "maturin"], check=True)


def _maturin_develop():
    repo = _repo_root()
    env = os.environ.copy()
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


def _license_api():
    module = importlib.import_module("forge3d._license")
    return module._reset_license_state, module.set_license_key


@pytest.fixture
def pro_license():
    """Enable a deterministic test license for Pro-gated APIs."""
    from _license_test_keys import sign_test_key

    _reset_license_state, set_license_key = _license_api()
    _reset_license_state()
    set_license_key(sign_test_key("PRO", "20991231"))
    try:
        yield
    finally:
        set_license_key("")


def pytest_configure(config):
    """Register custom markers for Workstream I tasks and P5.7."""

    config.addinivalue_line(
        "markers", "viewer: tests for interactive viewer functionality (Workstream I1)"
    )
    config.addinivalue_line(
        "markers", "offscreen: tests for offscreen rendering and Jupyter integration (Workstream I2)"
    )
    config.addinivalue_line("markers", "opbr: tests for PBR rendering")
    config.addinivalue_line("markers", "olighting: tests for lighting")
    config.addinivalue_line(
        "markers", "interactive_viewer: tests requiring interactive viewer"
    )
    config.addinivalue_line("markers", "pro: tests requiring a Pro license")
    config.addinivalue_line("markers", "slow: slow tests")


def pytest_collection_modifyitems(config, items):
    """Auto-tag tests that require the Pro license fixture."""

    del config

    for item in items:
        if "pro_license" in item.fixturenames:
            item.add_marker(pytest.mark.pro)


# Track P5.7 test results for artifact generation
_p57_results = {}


def pytest_runtest_logreport(report):
    """Track P5.7 test results."""

    if report.when == "call":
        if (
            "test_p5_ssao" in report.nodeid
            or "test_p5_ssgi" in report.nodeid
            or "test_p5_ssr" in report.nodeid
        ):
            _p57_results[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    """Write p5_PASS.txt if all P5.7 tests passed."""

    import hashlib
    import json
    from datetime import datetime

    del session, exitstatus

    if not _p57_results:
        return

    passed = [k for k, v in _p57_results.items() if v == "passed"]
    failed = [k for k, v in _p57_results.items() if v == "failed"]
    skipped = [k for k, v in _p57_results.items() if v == "skipped"]

    repo_root = _repo_root()
    report_dir = repo_root / "reports" / "p5"
    report_dir.mkdir(parents=True, exist_ok=True)

    if failed:
        fail_file = report_dir / "p5_FAIL.txt"
        with open(fail_file, "w", encoding="utf-8") as f:
            f.write("P5.7 Acceptance Tests FAILED\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Failed: {len(failed)}\n")
            f.write(f"Skipped: {len(skipped)}\n")
            f.write("\nFailed tests:\n")
            for test_name in failed:
                f.write(f"  - {test_name}\n")
    elif passed:
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
        with open(pass_file, "w", encoding="utf-8") as f:
            f.write("P5.7 Acceptance Tests PASSED\n")
            f.write(f"Timestamp: {metrics['timestamp']}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Skipped: {len(skipped)}\n")
            f.write(f"Metrics hash: {metrics_hash}\n")
            f.write("\nPassed tests:\n")
            for test_name in passed:
                f.write(f"  - {test_name}\n")


def pytest_sessionstart(session):
    del session

    # Always make test helpers importable, even in no-bootstrap mode.
    tests_dir = _repo_root() / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    if os.environ.get("FORGE3D_NO_BOOTSTRAP") == "1":
        return

    _ensure_python_path()
    try:
        import forge3d  # noqa: F401
        return
    except Exception as exc:
        if not _needs_build_from_exc(exc):
            raise
        try:
            _install_maturin()
            _maturin_develop()
            importlib.invalidate_caches()
            _ensure_python_path()
            import forge3d  # noqa: F401
            print("forge3d bootstrap: built via maturin", flush=True)
        except Exception as build_exc:
            raise RuntimeError(
                "forge3d bootstrap failed. Ensure Rust toolchain and (on Windows) "
                f"MSVC Build Tools are installed. Original error: {build_exc}"
            ) from build_exc
