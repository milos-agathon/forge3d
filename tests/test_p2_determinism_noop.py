from __future__ import annotations

import pytest

import forge3d as f3d


def _base_scene(*, terrain_metadata=None, layers=(), diagnostics_policy=None) -> f3d.MapScene:
    metadata = {"source_id": "terrain", "width": 8, "height": 8}
    metadata.update(terrain_metadata or {})
    return f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata=metadata),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=16, height=16),
        diagnostics_policy=diagnostics_policy,
        layers=list(layers),
    )


def test_p2_validation_reports_are_deterministic_for_diagnostic_paths(tmp_path):
    texture = tmp_path / "facade.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    layer = f3d.MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="buildings",
        support_level="underdeveloped",
        geometry_count=1,
        metadata={
            "source_id": "buildings",
            "asset_status": "fixture",
            "textured_materials": [
                {"material_id": "facade", "object_id": "b1", "albedo_texture": str(texture), "uv_available": True}
            ],
            "instancing": {"requested": True, "support_level": "unsupported", "path": "building instances"},
        },
    )
    scene = _base_scene(
        terrain_metadata={"virtual_texture": {"enabled": True, "families": ["albedo", "normal"]}},
        layers=[layer],
        diagnostics_policy={"large_scene_summary": True},
    )

    assert scene.validate().to_dict() == scene.validate().to_dict()


def test_p2_noop_success_paths_block_render(tmp_path):
    texture = tmp_path / "facade.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    scene = _base_scene(
        terrain_metadata={"virtual_texture": {"enabled": True, "families": ["normal"]}},
        layers=[
            f3d.MapSceneBuildingLayer.from_geojson(
                tmp_path / "buildings.geojson",
                layer_id="buildings",
                support_level="underdeveloped",
                geometry_count=1,
                metadata={
                    "source_id": "buildings",
                    "asset_status": "fixture",
                    "textured_materials": [
                        {
                            "material_id": "facade",
                            "object_id": "b1",
                            "albedo_texture": str(texture),
                            "uv_available": True,
                        }
                    ],
                },
            )
        ],
        diagnostics_policy={"large_scene_summary": True},
    )
    output = tmp_path / "noop.png"

    report = scene.validate()
    assert report.status == "ok"
    assert report.supported_features["vt.normal"] == "supported"
    assert report.supported_features["buildings.textured_pbr"] == "supported"

    with pytest.raises(f3d.MapSceneNativeUnavailable, match="native rendering unavailable"):
        scene.render(str(output))

    assert not output.exists()


def test_advanced_label_diagnostics_are_deterministic_not_silent_success():
    labels = [
        {
            "id": "curved",
            "text": "River",
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [20, 5]]},
            "curved_text": True,
        },
        {
            "id": "repeat",
            "text": "Road",
            "geometry": {"type": "LineString", "coordinates": [[0, 20], [80, 20]]},
            "repeat_distance": 40,
        },
    ]

    first = f3d.LabelPlan.compile(labels=labels, camera={}, viewport=(100, 100), seed=9)
    second = f3d.LabelPlan.compile(labels=labels, camera={}, viewport=(100, 100), seed=9)

    assert first.to_dict() == second.to_dict()
    assert [label.label_id for label in first.accepted] == ["repeat"]
    assert any(diagnostic.code == "experimental_feature" for diagnostic in first.diagnostics)
