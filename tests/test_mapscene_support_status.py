from pathlib import Path

import pytest

import forge3d as f3d


def _scene(*, layers=(), terrain_metadata=None) -> f3d.MapScene:
    metadata = {"width": 8, "height": 8, "asset_status": "fixture"}
    if terrain_metadata:
        metadata.update(terrain_metadata)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata=metadata,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=500.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=32, height=32, format="png"),
        layers=list(layers),
    )


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def test_product_building_intent_reports_support_status_diagnostics():
    scene = _scene(
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="pro-buildings",
                source="fixtures/buildings.geojson",
                support_level="Pro-gated",
            ),
            f3d.MapSceneBuildingLayer(
                layer_id="fallback-buildings",
                source="fixtures/fallback.geojson",
                support_level="placeholder/fallback",
                geometry_count=0,
            ),
            f3d.MapSceneBuildingLayer(
                layer_id="experimental-buildings",
                source="fixtures/experimental.city.json",
                support_level="experimental",
                geometry_count=3,
            ),
            f3d.MapSceneBuildingLayer(
                layer_id="unsupported-buildings",
                source="fixtures/unsupported.obj",
                support_level="unsupported",
            ),
        ]
    )

    report = scene.validate()

    assert report.status == "error"
    assert _codes(report) == [
        "placeholder_fallback",
        "pro_gated_path",
        "unsupported_feature",
        "experimental_feature",
    ]
    by_layer = {diagnostic.layer_id: diagnostic for diagnostic in report.diagnostics}
    assert by_layer["pro-buildings"].support_level == "Pro-gated"
    assert by_layer["fallback-buildings"].support_level == "placeholder/fallback"
    assert by_layer["experimental-buildings"].support_level == "experimental"
    assert by_layer["unsupported-buildings"].support_level == "unsupported"
    assert by_layer["unsupported-buildings"].severity == "error"


def test_terrain_vt_and_3dtiles_intent_are_diagnosed_before_render():
    scene = _scene(
        terrain_metadata={
            "width": 8,
            "height": 8,
            "virtual_texture": {"enabled": True, "families": ["albedo", "normal", "mask"]},
        },
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="tiles-buildings",
                source={"kind": "3dtiles", "path": "fixtures/tileset.json"},
                support_level="underdeveloped",
            )
        ],
    )

    report = scene.validate()

    assert report.status == "error"
    assert _codes(report) == [
        "python_public_3dtiles_incomplete",
        "vt_unsupported_family",
        "vt_unsupported_family",
    ]
    assert sorted(
        diagnostic.details["family"]
        for diagnostic in report.diagnostics
        if diagnostic.code == "vt_unsupported_family"
    ) == ["mask", "normal"]
    assert report.diagnostics[0].layer_id == "tiles-buildings"


def test_point_cloud_layer_reports_underdeveloped_render_path_before_render():
    scene = _scene(
        layers=[
            f3d.PointCloudLayer(
                layer_id="points",
                path="fixtures/points.las",
                crs="EPSG:32610",
                point_count=24,
                metadata={"asset_status": "fixture"},
            )
        ]
    )

    report = scene.validate()

    assert report.status == "error"
    assert _codes(report) == ["placeholder_fallback"]
    diagnostic = report.diagnostics[0]
    assert diagnostic.layer_id == "points"
    assert diagnostic.details["feature"] == "point cloud MapScene render path"
    summary = next(summary for summary in report.layer_summaries if summary.layer_id == "points")
    assert summary.support_level == "placeholder/fallback"


def test_unknown_recipe_layer_is_not_silently_ignored():
    class UnknownLayer:
        layer_id = "mystery"

        def to_dict(self):
            return {"kind": "unknown_layer", "layer_id": self.layer_id}

    scene = _scene(layers=[UnknownLayer()])

    report = scene.validate()

    assert report.status == "error"
    assert _codes(report) == ["unsupported_layer_type"]
    assert report.diagnostics[0].layer_id == "mystery"
    assert report.diagnostics[0].support_level == "unsupported"
    summaries_by_id = {summary.layer_id: summary for summary in report.layer_summaries}
    assert summaries_by_id["mystery"].support_level == "unsupported"


def test_render_and_save_bundle_do_not_hide_blocking_validation(tmp_path):
    scene = _scene(
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="pro-buildings",
                source="fixtures/buildings.geojson",
                support_level="Pro-gated",
            )
        ]
    )
    image_path = tmp_path / "blocked.png"
    bundle_path = tmp_path / "blocked.forge3d"

    with pytest.raises(RuntimeError, match="blocking diagnostics"):
        scene.render(str(image_path))
    assert not image_path.exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "error"
    assert scene.last_validation_report.render_blocked() is True

    bundle_report = scene.save_bundle(str(bundle_path))
    assert Path(bundle_path).exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "error"
    assert bundle_report.status == "error"
