from __future__ import annotations

import forge3d as f3d


def test_p2_quickstart_vt_textured_building_and_large_scene_diagnostics(tmp_path):
    texture = tmp_path / "facade.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=None,
            crs="EPSG:4326",
            metadata={
                "source_id": "terrain",
                "width": 8,
                "height": 8,
                "virtual_texture": {"enabled": True, "families": ["albedo", "normal", "mask"]},
            },
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=16, height=16),
        diagnostics_policy={"large_scene_summary": True, "gpu_memory_budget_bytes": 1},
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
    )

    report = scene.validate()
    codes = {diagnostic.code for diagnostic in report.diagnostics}

    assert {"vt_unsupported_family", "unsupported_feature", "estimated_gpu_memory"}.issubset(codes)
    assert next(s for s in report.layer_summaries if s.layer_id == "large_scene.resources")


def test_p2_quickstart_advanced_labels_compile_or_diagnose():
    plan = f3d.LabelPlan.compile(
        labels=[
            {
                "id": "road",
                "text": "Road",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
                "repeat_distance": 50,
            },
            {
                "id": "curved",
                "text": "River",
                "geometry": {"type": "LineString", "coordinates": [[0, 20], [50, 30]]},
                "curved_text": True,
            },
        ],
        camera={},
        viewport=(120, 80),
    )

    assert [label.label_id for label in plan.accepted] == ["road"]
    assert any(diagnostic.code == "experimental_feature" for diagnostic in plan.diagnostics)
