import pytest
import forge3d as f3d
from forge3d.diagnostics import capabilities


def test_capabilities_reports_requested_and_granted():
    if not f3d.has_gpu():
        pytest.skip("no GPU adapter")
    caps = capabilities()
    assert set(caps) >= {"requested", "granted", "limits"}
    assert "timestamp_query" in caps["requested"]
    assert set(caps["granted"]) <= set(caps["requested"])


def test_absent_capability_is_recorded_not_fatal():
    if not f3d.has_gpu():
        pytest.skip("no GPU adapter")
    caps = capabilities()
    from forge3d._forge3d import native_degradations

    degs = [d for d in native_degradations() if d["kind"] == "capability_absent"]
    absent = set(caps["requested"]) - set(caps["granted"])
    assert absent == {d["name"] for d in degs}
