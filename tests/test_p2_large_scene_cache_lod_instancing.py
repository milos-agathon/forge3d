from __future__ import annotations

import forge3d as f3d


def _scene(*layers) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=None,
            crs="EPSG:4326",
            metadata={
                "source_id": "terrain",
                "width": 4,
                "height": 4,
                "unavailable_cache_lod_stats": ["cache"],
            },
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=8, height=8),
        diagnostics_policy={"large_scene_summary": True},
        layers=list(layers),
    )


def _resource_summary(report: f3d.ValidationReport) -> f3d.LayerSummary:
    return next(summary for summary in report.layer_summaries if summary.layer_id == "large_scene.resources")


def test_cache_lod_and_instancing_status_are_normalized_by_layer_type():
    scene = _scene(
        f3d.PointCloudLayer(
            layer_id="points",
            point_count=10,
            metadata={"source_id": "points", "unavailable_cache_lod_stats": ["lod"]},
        ),
        f3d.Tiles3DLayer(
            layer_id="tiles",
            source={"source_id": "tiles", "source_format": "tileset.json"},
            cache_budget=8,
            cache_stats={"entries": 2, "bytes": 1024},
            lod={"sse": 16, "max_depth": 3},
            metadata={"source_id": "tiles"},
        ),
        f3d.MapSceneBuildingLayer(
            layer_id="buildings",
            source={"source_id": "buildings", "source_format": "geojson"},
            support_level="underdeveloped",
            geometry_count=5,
            metadata={
                "source_id": "buildings",
                "instancing": {"requested": True, "support_level": "unsupported", "path": "building instances"},
            },
        ),
    )

    report = scene.validate()
    summary = _resource_summary(report)

    assert any(d.code == "unavailable_cache_lod_stats" and d.layer_id == "points" for d in report.diagnostics)
    assert any(d.code == "unsupported_instancing_path" and d.layer_id == "buildings" for d in report.diagnostics)
    by_layer = {item["layer_id"]: item for item in summary.details["cache_lod_status"]}
    assert by_layer["tiles"]["cache_stats"] == {"bytes": 1024, "entries": 2}
    assert by_layer["tiles"]["lod"] == {"max_depth": 3, "sse": 16}
    assert by_layer["points"]["unavailable_cache_lod_stats"] == ["lod"]
    assert summary.details["instancing_status"]["buildings"]["support_level"] == "unsupported"


def test_unavailable_terrain_cache_lod_stats_are_reported_before_render():
    report = _scene().validate()
    summary = _resource_summary(report)

    assert any(d.code == "unavailable_cache_lod_stats" and d.layer_id == "terrain" for d in report.diagnostics)
    assert {"layer_id": "terrain", "stats": ["cache"]} in summary.details["unavailable_stats"]
