# A1.10-BEGIN:pytest-perf
import json, os, subprocess, sys, tempfile, pathlib, pytest

ENABLED = os.environ.get("VF_TEST_PERF", "").strip() == "1"

@pytest.mark.skipif(not ENABLED, reason="Set VF_TEST_PERF=1 to enable perf sanity in CI")
def test_perf_sanity_runs(tmp_path):
    out_json = tmp_path / "perf.json"
    subprocess.check_call([
        sys.executable, "python/tools/perf_sanity.py",
        "--width", "96", "--height", "96", "--runs", "10", "--warmups", "2",
        "--json", str(out_json),
    ])
    rep = json.loads(out_json.read_text())
    assert "steady" in rep and "p95_ms" in rep["steady"]
# A1.10-END:pytest-perf