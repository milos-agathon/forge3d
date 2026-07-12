from __future__ import annotations

import forge3d as f3d


def _scene(layer: f3d.MapSceneBuildingLayer) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "fixture-dem"}),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[layer],
    )


def _diagnostics(report: f3d.ValidationReport) -> dict[tuple[str, str | None], f3d.Diagnostic]:
    return {(diagnostic.code, diagnostic.object_id): diagnostic for diagnostic in report.diagnostics}


def test_missing_uv_path_format_and_scalar_fallback_are_diagnosed(tmp_path):
    missing_texture = tmp_path / "missing_facade.ktx2"
    layer = f3d.MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="buildings.texture.errors",
        support_level="underdeveloped",
        geometry_count=2,
        metadata={
            "source_id": "fixture-buildings",
            "textured_materials": [
                {
                    "material_id": "facade",
                    "object_id": "building-1",
                    "albedo_texture": str(missing_texture),
                    "texture_format": "ktx2",
                    "uv_available": False,
                    "scalar_fallback": True,
                }
            ],
        },
    )

    report = _scene(layer).validate()
    diagnostics = _diagnostics(report)

    assert report.status == "error"
    assert diagnostics[("missing_texture_path", "building-1")].details["material_id"] == "facade"
    assert diagnostics[("missing_uvs", "building-1")].details["material_id"] == "facade"
    assert diagnostics[("unsupported_texture_format", "building-1")].details["format"] == "ktx2"
    assert diagnostics[("placeholder_fallback", "building-1")].details["feature"] == (
        "building textured material scalar fallback"
    )
    summary = next(s for s in report.layer_summaries if s.layer_id == "buildings.texture.errors")
    assert summary.details["textured_material_status"] == "placeholder/fallback"
    assert summary.support_level == "placeholder/fallback"


def test_pro_gated_textured_import_remains_explicitly_pro_gated(tmp_path):
    layer = f3d.MapSceneBuildingLayer.from_cityjson(
        tmp_path / "buildings.city.json",
        layer_id="buildings.texture.pro",
        support_level="Pro-gated",
        geometry_count=1,
        metadata={
            "source_id": "fixture-buildings",
            "asset_status": "fixture",
            "textured_materials": [
                {
                    "material_id": "facade",
                    "object_id": "building-1",
                    "albedo_texture": "facade.png",
                    "uv_available": True,
                }
            ],
        },
    )

    report = _scene(layer).validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert "pro_gated_path" in codes
    assert "unsupported_feature" not in codes
    assert report.unsupported_features["buildings.pro_gated_path"] == "Pro-gated"
    assert report.supported_features["buildings.textured_pbr"] == "supported"
