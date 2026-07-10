from __future__ import annotations

import pytest

import forge3d as f3d


def _scene(layer: f3d.MapSceneBuildingLayer) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "fixture-dem"}),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[layer],
    )


def test_textured_building_fixture_reports_supported_native_material_path(tmp_path):
    texture = tmp_path / "facade.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    layer = f3d.MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="buildings.textured.fixture",
        support_level="underdeveloped",
        geometry_count=1,
        metadata={
            "source_id": "fixture-buildings",
            "asset_status": "fixture",
            "textured_materials": [
                {
                    "material_id": "facade",
                    "object_id": "building-1",
                    "albedo_texture": str(texture),
                    "texture_format": "png",
                    "uv_available": True,
                }
            ],
        },
    )

    report = _scene(layer).validate()

    assert report.status == "ok"
    assert not report.diagnostics
    assert report.supported_features["buildings.textured_pbr"] == "supported"
    summary = next(s for s in report.layer_summaries if s.layer_id == "buildings.textured.fixture")
    assert summary.details["textured_material_status"] == "supported"
    assert summary.details["textured_materials"][0]["uv_available"] is True


def test_textured_building_render_never_writes_output_when_terrain_backend_is_unavailable(tmp_path):
    texture = tmp_path / "facade.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    layer = f3d.MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="buildings.textured.blocked",
        support_level="underdeveloped",
        geometry_count=1,
        metadata={
            "source_id": "fixture-buildings",
            "asset_status": "fixture",
            "textured_materials": [
                {
                    "material_id": "facade",
                    "object_id": "building-1",
                    "albedo_texture": str(texture),
                    "uv_available": True,
                }
            ],
        },
    )
    output = tmp_path / "textured-building.png"

    scene = _scene(layer)
    assert scene.validate().supported_features["buildings.textured_pbr"] == "supported"
    with pytest.raises(f3d.MapSceneNativeUnavailable, match="native rendering unavailable"):
        scene.render(str(output))

    assert not output.exists()
