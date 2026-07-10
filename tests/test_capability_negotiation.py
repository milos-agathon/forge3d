import pytest
import forge3d as f3d
from forge3d.diagnostics import capabilities
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_production_device_paths_do_not_request_empty_features():
    for relative in (
        "src/viewer/init/device_init.rs",
        "src/terrain/spike/constructor.rs",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "required_features: wgpu::Features::empty()" not in source, relative


def test_viewer_gpu_timing_readback_is_non_blocking():
    source = (
        ROOT / "src/viewer/state/viewer_helpers/gi/reexecute.rs"
    ).read_text(encoding="utf-8")
    assert "Maintain::Wait" not in source
    assert "pollster::block_on(timer.get_results())" not in source
    assert "timer.try_get_results()" in source
