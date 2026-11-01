"""Publication regressions must be selected by the real hosted test profiles."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_fast_profile_executes_publication_regressions():
    sys.path.insert(0, str(ROOT / "tests"))
    spec = importlib.util.spec_from_file_location("c06_ci_lane", ROOT / "scripts/ci_pytest_lane.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_pytest_args("fast", [])
    assert "tests/test_publication.py" in args
    assert "tests/test_publication_ci.py" in args
    assert "tests/test_scale_bar_projected_units.py" in args
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    fast_job = workflow.split("  test-fast-contract:", 1)[1].split("  test-orbis-wasm:", 1)[0]
    assert "--profile fast" in fast_job
    assert "pip install -U pip maturin pytest numpy pillow pyproj" in fast_job


def test_physical_gpu_lane_executes_publication_with_zero_skips():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    step = workflow.split("- name: Run C06 publication physical GPU gate", 1)[1].split("- name:", 1)[0]
    assert "FORGE3D_RUN_GPU_ANAMNESIS: '1'" in step
    assert "pytest tests/test_publication.py tests/test_publication_gpu.py" in step
    assert 'assert_junit_zero_skips.py "$junit"' in step
