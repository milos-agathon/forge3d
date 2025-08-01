# A1.9-BEGIN:pytest-diagnostics
import json, sys, subprocess

def test_enumerate_and_probe(tmp_path):
    out = tmp_path / "diag.json"
    subprocess.check_call([sys.executable, "python/tools/device_diagnostics.py", "--json", str(out)])
    rep = json.loads(out.read_text())
    assert "adapters" in rep and "probes" in rep
# A1.9-END:pytest-diagnostics
