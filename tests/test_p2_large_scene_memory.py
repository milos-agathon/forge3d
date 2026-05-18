from __future__ import annotations

import forge3d as f3d


def _resource_summary(report: f3d.ValidationReport) -> f3d.LayerSummary:
    return next(summary for summary in report.layer_summaries if summary.layer_id == "large_scene.resources")


def test_large_scene_memory_summary_uses_known_metadata_and_budget_diagnostic():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=None,
            crs="EPSG:4326",
            metadata={"source_id": "terrain", "width": 10, "height": 10},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=20, height=20),
        diagnostics_policy={"gpu_memory_budget_bytes": 1, "large_scene_summary": True},
        layers=[
            f3d.RasterOverlay(
                layer_id="raster",
                path=None,
                crs="EPSG:4326",
                metadata={"source_id": "raster", "width": 4, "height": 4},
            ),
            f3d.PointCloudLayer(layer_id="points", point_count=5, metadata={"source_id": "points"}),
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source={"source_id": "buildings", "source_format": "geojson"},
                support_level="underdeveloped",
                geometry_count=3,
            ),
            f3d.Tiles3DLayer(
                layer_id="tiles",
                source={"source_id": "tiles", "source_format": "tileset.json"},
                metadata={"source_id": "tiles", "memory_estimate_bytes": 4096},
            ),
        ],
    )

    report = scene.validate()
    summary = _resource_summary(report)

    assert report.estimated_gpu_memory_bytes is not None
    assert any(diagnostic.code == "estimated_gpu_memory" for diagnostic in report.diagnostics)
    estimates = {item["layer_id"]: item["memory_estimate_bytes"] for item in summary.details["memory_estimates"]}
    assert estimates["terrain"] == 400
    assert estimates["raster"] == 64
    assert estimates["points"] == 120
    assert estimates["buildings"] == 288
    assert estimates["tiles"] == 4096
    assert summary.details["total_estimated_gpu_memory_bytes"] == report.estimated_gpu_memory_bytes


def test_large_scene_does_not_invent_memory_when_metadata_is_unavailable():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "terrain"}),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=8, height=8),
        diagnostics_policy={"large_scene_summary": True},
        layers=[f3d.PointCloudLayer(layer_id="points", metadata={"source_id": "points"})],
    )

    report = scene.validate()
    summary = _resource_summary(report)
    estimates = {item["layer_id"] for item in summary.details["memory_estimates"]}

    assert "points" not in estimates
    assert summary.details["unavailable_stats"]
