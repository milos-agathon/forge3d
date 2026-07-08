from __future__ import annotations

import pytest

import forge3d as f3d


def _scene_with_vt(families: list[str]) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={
                "width": 16,
                "height": 16,
                "asset_status": "fixture",
                "virtual_texture": {
                    "enabled": True,
                    "families": families,
                },
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=500.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=32, height=32, format="png"),
    )


def _vt_summary(report: f3d.ValidationReport) -> f3d.LayerSummary:
    return next(summary for summary in report.layer_summaries if summary.layer_id == "terrain.vt")


def test_albedo_only_vt_scene_validates_and_renders(tmp_path):
    scene = _scene_with_vt(["albedo"])

    report = scene.validate()

    assert report.status == "ok"
    assert report.diagnostics == ()
    assert report.supported_features["vt.albedo"] == "supported"
    assert "vt.normal" not in report.unsupported_features
    summary = _vt_summary(report)
    assert summary.support_level == "supported"
    assert summary.details["families"] == ["albedo"]
    assert summary.details["native_supported_families"] == ["albedo"]

    # The fixture DEM has no renderable heightmap: render must block with a
    # structured diagnostic instead of writing a CPU placeholder.
    output_path = tmp_path / "albedo-vt.png"
    with pytest.raises(f3d.MapSceneNativeUnavailable):
        scene.render(str(output_path))
    assert not output_path.exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "ok"


def test_non_albedo_vt_families_are_deterministic_supported_families():
    first = _scene_with_vt(["albedo", "normal", "mask"]).validate()
    second = _scene_with_vt(["mask", "albedo", "normal"]).validate()

    assert first.status == "ok"
    assert second.status == "ok"
    assert first.to_dict() == second.to_dict()

    diagnostics = [diagnostic for diagnostic in first.diagnostics if diagnostic.code == "vt_unsupported_family"]
    assert diagnostics == []

    summary = _vt_summary(first)
    assert summary.support_level == "supported"
    assert summary.diagnostic_codes == ()
    assert summary.details["families"] == ["albedo", "mask", "normal"]
    assert summary.details["native_supported_families"] == ["albedo", "mask", "normal"]
    assert first.supported_features["vt.albedo"] == "supported"
    assert first.supported_features["vt.mask"] == "supported"
    assert first.supported_features["vt.normal"] == "supported"
    assert first.unsupported_features == {}


def test_non_albedo_vt_request_does_not_block_validation(tmp_path):
    scene = _scene_with_vt(["albedo", "normal"])
    output_path = tmp_path / "silently-skipped-normal.png"

    # Non-albedo VT families no longer block validation; rendering itself is
    # blocked only because the fixture terrain has no native heightmap.
    with pytest.raises(f3d.MapSceneNativeUnavailable):
        scene.render(str(output_path))

    assert not output_path.exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "ok"
    assert "vt.normal" not in scene.last_validation_report.unsupported_features
