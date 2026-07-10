from __future__ import annotations


def _scene_with_layer(layer):
    from forge3d.map_scene import LightingPreset, MapScene, OrbitCamera, OutputSpec, TerrainSource

    return MapScene(
        terrain=TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "fixture-dem"}),
        camera=OrbitCamera(),
        lighting=LightingPreset(),
        layers=[layer],
        output=OutputSpec(width=64, height=64),
    )


def _summary(report, layer_id: str):
    return next(summary for summary in report.layer_summaries if summary.layer_id == layer_id)


def test_building_validation_distinguishes_support_statuses(tmp_path):
    from forge3d.map_scene import MapSceneBuildingLayer

    building_path = tmp_path / "buildings.geojson"
    building_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    cases = [
        (
            MapSceneBuildingLayer.from_geojson(building_path, layer_id="b.supported", support_level="supported"),
            "placeholder_fallback",
        ),
        (
            MapSceneBuildingLayer.from_geojson(building_path, layer_id="b.pro", support_level="Pro-gated"),
            "pro_gated_path",
        ),
        (
            MapSceneBuildingLayer.from_geojson(building_path, layer_id="b.fallback", support_level="placeholder/fallback"),
            "placeholder_fallback",
        ),
        (
            MapSceneBuildingLayer.from_mesh(building_path, layer_id="b.unsupported", support_level="unsupported"),
            "unsupported_feature",
        ),
    ]

    for layer, expected_code in cases:
        report = _scene_with_layer(layer).validate()
        assert expected_code in [diagnostic.code for diagnostic in report.diagnostics], layer.layer_id
        assert _summary(report, layer.layer_id).layer_type == "building_layer"


def test_cityjson_and_geojson_building_render_prep_stays_diagnostic_bearing(tmp_path):
    from forge3d.map_scene import MapSceneBuildingLayer

    cityjson = tmp_path / "fixture.city.json"
    geojson = tmp_path / "fixture.geojson"
    cityjson.write_text('{"type":"CityJSON","version":"1.1","CityObjects":{}}', encoding="utf-8")
    geojson.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    scene_layers = [
        MapSceneBuildingLayer.from_cityjson(cityjson, layer_id="b.cityjson", support_level="supported"),
        MapSceneBuildingLayer.from_geojson(geojson, layer_id="b.geojson", support_level="supported"),
    ]

    for layer in scene_layers:
        report = _scene_with_layer(layer).validate()
        codes = [diagnostic.code for diagnostic in report.diagnostics]
        assert "placeholder_fallback" in codes
        assert report.unsupported_features["buildings.placeholder_fallback"] == "placeholder/fallback"


def test_building_summary_includes_geometry_count_bounds_and_material_status(tmp_path):
    from forge3d.map_scene import MapSceneBuildingLayer

    layer = MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="b.summary",
        support_level="underdeveloped",
        geometry_count=2,
        bounds=[0.0, 1.0, 2.0, 3.0],
        material_status="scalar_pbr_supported",
        metadata={"source_id": "fixture-buildings"},
    )

    report = _scene_with_layer(layer).validate()
    summary = _summary(report, "b.summary")

    assert summary.object_count == 2
    assert summary.bounds == (0.0, 1.0, 2.0, 3.0)
    assert summary.details["material_status"] == "scalar_pbr_supported"
    assert summary.details["source_kind"] == "geojson"


def test_textured_pbr_buildings_are_explicitly_unsupported_when_not_implemented(tmp_path):
    from forge3d.map_scene import MapSceneBuildingLayer

    layer = MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="b.textured",
        support_level="underdeveloped",
        material_status="textured_pbr_unsupported",
        metadata={"source_id": "fixture-buildings"},
    )

    report = _scene_with_layer(layer).validate()
    diagnostics = {(diagnostic.layer_id, diagnostic.code, diagnostic.details.get("feature")) for diagnostic in report.diagnostics}

    assert ("b.textured", "unsupported_feature", "building textured PBR") in diagnostics


def test_zero_geometry_building_fallback_cannot_be_mistaken_for_success(tmp_path):
    from forge3d.map_scene import MapSceneBuildingLayer

    layer = MapSceneBuildingLayer.from_geojson(
        tmp_path / "empty.geojson",
        layer_id="b.empty",
        support_level="supported",
        geometry_count=0,
        metadata={"source_id": "fixture-empty-buildings"},
    )

    report = _scene_with_layer(layer).validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert "placeholder_fallback" in codes
    assert _summary(report, "b.empty").support_level == "placeholder/fallback"


def test_building_docs_record_scalar_pbr_and_textured_pbr_status():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / "docs/guides/data_and_scene_workflows.md").read_text(encoding="utf-8")

    assert "scalar PBR" in workflow
    assert "textured PBR" in workflow
    assert "unsupported_feature" in workflow
