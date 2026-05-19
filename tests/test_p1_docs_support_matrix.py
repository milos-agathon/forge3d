from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_p1_docs_state_local_style_scope_and_no_runtime_parity_claims():
    workflow = _read("docs/guides/data_and_scene_workflows.md")
    feature_map = _read("docs/guides/feature_map.md")

    assert "local/provided feature styling" in workflow
    assert "fill, line, and circle" in workflow
    assert "unsupported_style_field" in workflow
    assert "unsupported_style_layer_type" in workflow
    assert "not full Cesium runtime parity" in workflow
    assert "not full Mapbox GL parity" in workflow
    assert "underdeveloped" in workflow

    assert "MapScene" in feature_map
    assert "MapSceneBuildingLayer" in feature_map
    assert "Tiles3DLayer" in feature_map
    assert "underdeveloped" in feature_map


def test_p1_api_docs_record_product_api_support_and_compatibility_alias():
    api_reference = _read("docs/api/api_reference.rst")

    for marker in (
        "LabelLayer.from_features",
        "LabelLayer.from_geodataframe",
        "LabelLayer.from_style_layer",
        "BuildingLayer.from_geojson",
        "BuildingLayer.from_cityjson",
        "BuildingLayer.from_mesh",
        "Tiles3DLayer.from_tileset_json",
        "Tiles3DLayer.from_b3dm",
        "MapScene.load_bundle",
        "MapScene.save_bundle",
        "MapSceneBuildingLayer",
        "legacy ``forge3d.BuildingLayer``",
    ):
        assert marker in api_reference

    for diagnostic in (
        "missing_label_field",
        "unicode_coverage_gap",
        "unsupported_tile_format",
        "unsupported_tile_feature",
        "missing_external_asset",
        "unavailable_terrain_sampler",
    ):
        assert diagnostic in api_reference
