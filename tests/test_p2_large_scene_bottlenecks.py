from __future__ import annotations

import forge3d as f3d


def _resource_summary(report: f3d.ValidationReport) -> f3d.LayerSummary:
    return next(summary for summary in report.layer_summaries if summary.layer_id == "large_scene.resources")


def test_bottleneck_layer_types_are_deterministic_from_known_memory_and_counts():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=None,
            crs="EPSG:4326",
            metadata={"source_id": "terrain", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=8, height=8),
        diagnostics_policy={"large_scene_summary": True},
        layers=[
            f3d.PointCloudLayer(layer_id="points", point_count=100, metadata={"source_id": "points"}),
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source={"source_id": "buildings", "source_format": "geojson"},
                support_level="underdeveloped",
                geometry_count=5,
            ),
        ],
    )

    first = scene.validate().to_dict()
    second = scene.validate().to_dict()
    assert first == second

    summary = _resource_summary(f3d.ValidationReport.from_dict(first))
    assert summary.details["bottleneck_layer_types"][:2] == ["point_cloud_layer", "building_layer"]
    assert summary.details["bottleneck_layers"][0]["layer_id"] == "points"
