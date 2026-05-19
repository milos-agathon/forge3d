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
    assert summary.details["native_supported_family"] == "albedo"

    output_path = tmp_path / "albedo-vt.png"
    render_report = scene.render(str(output_path))

    assert output_path.exists()
    assert render_report.status == "ok"


def test_non_albedo_vt_families_are_deterministic_unsupported_diagnostics():
    first = _scene_with_vt(["albedo", "normal", "mask"]).validate()
    second = _scene_with_vt(["mask", "albedo", "normal"]).validate()

    assert first.status == "error"
    assert second.status == "error"
    assert first.to_dict() == second.to_dict()

    diagnostics = [diagnostic for diagnostic in first.diagnostics if diagnostic.code == "vt_unsupported_family"]
    assert [diagnostic.details["family"] for diagnostic in diagnostics] == ["mask", "normal"]
    assert [diagnostic.object_id for diagnostic in diagnostics] == ["vt.mask", "vt.normal"]
    assert {diagnostic.layer_id for diagnostic in diagnostics} == {"terrain.vt"}
    assert {diagnostic.support_level for diagnostic in diagnostics} == {"unsupported"}

    summary = _vt_summary(first)
    assert summary.support_level == "unsupported"
    assert summary.diagnostic_codes == ("vt_unsupported_family", "vt_unsupported_family")
    assert summary.details["families"] == ["albedo", "mask", "normal"]
    assert first.supported_features["vt.albedo"] == "supported"
    assert first.unsupported_features["vt.mask"] == "unsupported"
    assert first.unsupported_features["vt.normal"] == "unsupported"


def test_non_albedo_vt_request_blocks_render_without_silent_skip(tmp_path):
    scene = _scene_with_vt(["albedo", "normal"])
    output_path = tmp_path / "silently-skipped-normal.png"

    with pytest.raises(RuntimeError, match="blocking diagnostics"):
        scene.render(str(output_path))

    assert not output_path.exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "error"
    diagnostic = scene.last_validation_report.diagnostics[0]
    assert diagnostic.code == "vt_unsupported_family"
    assert diagnostic.object_id == "vt.normal"
    assert diagnostic.details == {"family": "normal", "supported_family": "albedo"}
