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


def test_textured_building_fixture_is_explicitly_diagnostic_only(tmp_path):
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

    assert report.status == "error"
    diagnostic = next(d for d in report.diagnostics if d.code == "unsupported_feature")
    assert diagnostic.layer_id == "buildings.textured.fixture"
    assert diagnostic.object_id == "building-1"
    assert diagnostic.details["feature"] == "building textured PBR render path"
    summary = next(s for s in report.layer_summaries if s.layer_id == "buildings.textured.fixture")
    assert summary.support_level == "unsupported"
    assert summary.details["textured_material_status"] == "unsupported"
    assert summary.details["textured_materials"][0]["uv_available"] is True


def test_textured_building_render_is_blocked_without_silent_scalar_success(tmp_path):
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

    with pytest.raises(RuntimeError, match="blocking diagnostics"):
        _scene(layer).render(str(output))

    assert not output.exists()
