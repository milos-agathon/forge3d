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


def test_no_default_device_descriptor_requests_in_production_code():
    """`request_device(&wgpu::DeviceDescriptor::default(), ...)` is
    `Features::empty()` in disguise: a device created that way bypasses
    capability negotiation entirely (audit finding F-08, extrude_polygon_gpu_py).
    Test modules are exempt (heuristic: a hit after the file's first
    `#[cfg(test)]` marker, or a `tests.rs` file, is test code)."""
    hits = []
    for path in (ROOT / "src").rglob("*.rs"):
        if path.name == "tests.rs" or path.parent.name == "tests":
            continue
        text = path.read_text(encoding="utf-8")
        cfg_test = text.find("#[cfg(test)]")
        for i, line in enumerate(text.splitlines(), 1):
            if "request_device(&wgpu::DeviceDescriptor::default()" in line:
                offset = sum(
                    len(l) + 1 for l in text.splitlines()[: i - 1]
                )
                if cfg_test != -1 and offset > cfg_test:
                    continue  # inside the trailing test module
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{i}")
    assert hits == [], (
        f"un-negotiated DeviceDescriptor::default() device requests: {hits}"
    )


def test_viewer_gpu_timing_readback_is_non_blocking():
    source = (
        ROOT / "src/viewer/state/viewer_helpers/gi/reexecute.rs"
    ).read_text(encoding="utf-8")
    assert "Maintain::Wait" not in source
    assert "pollster::block_on(timer.get_results())" not in source
    assert "timer.try_get_results()" in source
