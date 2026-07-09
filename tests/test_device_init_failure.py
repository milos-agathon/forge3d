# tests/test_device_init_failure.py
# CENSOR Task 3: device-init failures surface as catchable Python exceptions,
# never a process abort/panic. A bad WGPU_BACKENDS pin must reach the strict
# env parser in try_ctx() and raise a RuntimeError-derived exception.
import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_bad_backend_env_raises_python_exception_not_abort():
    """An unrecognized WGPU_BACKENDS value is a catchable Python error.

    The subprocess must exit cleanly (no panic/abort => returncode 0) and it
    must report that a *Python* exception was caught rather than crashing.
    """
    code = (
        "import os\n"
        "os.environ['WGPU_BACKENDS'] = 'nonexistent'\n"
        "import forge3d as f3d\n"
        "try:\n"
        "    f3d.capabilities()\n"
        "    print('NO_ERROR')\n"
        "except BaseException as e:\n"
        "    print('PYEXC', type(e).__name__)\n"
    )
    r = _run(code)
    assert r.returncode == 0, (
        f"process aborted (returncode={r.returncode}); stderr:\n{r.stderr}"
    )
    assert "PYEXC" in r.stdout, (
        f"expected a caught Python exception, got stdout:\n{r.stdout}\n"
        f"stderr:\n{r.stderr}"
    )


def test_bad_backend_env_via_session_raises_not_abort():
    """Session() is another try_ctx() entry point; same honesty contract."""
    code = (
        "import os\n"
        "os.environ['WGPU_BACKENDS'] = 'nonexistent'\n"
        "import forge3d as f3d\n"
        "try:\n"
        "    f3d.Session()\n"
        "    print('NO_ERROR')\n"
        "except BaseException as e:\n"
        "    print('PYEXC', type(e).__name__)\n"
    )
    r = _run(code)
    assert r.returncode == 0, (
        f"process aborted (returncode={r.returncode}); stderr:\n{r.stderr}"
    )
    assert "PYEXC" in r.stdout, (
        f"expected a caught Python exception, got stdout:\n{r.stdout}\n"
        f"stderr:\n{r.stderr}"
    )


def test_typed_gpu_exceptions_importable():
    """The typed exception classes import and are RuntimeError subclasses."""
    import forge3d as f3d

    assert issubclass(f3d.MemoryBudgetExceeded, RuntimeError)
    assert issubclass(f3d.DegradedCapability, RuntimeError)
