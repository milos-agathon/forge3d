# A1.7-BEGIN:pytest-backends
import os, json, subprocess, sys, shutil, pathlib, pytest

ENABLED = os.environ.get("VF_TEST_BACKENDS", "").strip() != ""

@pytest.mark.skipif(not ENABLED, reason="Set VF_TEST_BACKENDS=1 to enable cross-backend test")
def test_cross_backend_runner(tmp_path):
    out_dir = tmp_path / "bk_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "python/tools/backends_runner.py",
        "--width", "96", "--height", "96", "--runs", "2",
        "--out-dir", str(out_dir),
    ]
    subprocess.check_call(cmd)
    rep = json.loads((out_dir / "backends_report.json").read_text())
    assert rep["at_least_one_ok"] is True
# A1.7-END:pytest-backends