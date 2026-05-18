from __future__ import annotations


def test_p1_product_contracts_are_owned_by_p0_modules():
    import forge3d as f3d
    from forge3d import buildings
    from forge3d import diagnostics, label_plan, map_scene

    assert f3d.Diagnostic is diagnostics.Diagnostic
    assert f3d.ValidationReport is diagnostics.ValidationReport
    assert f3d.LabelPlan is label_plan.LabelPlan
    assert f3d.MapScene is map_scene.MapScene
    assert f3d.LabelLayer is map_scene.LabelLayer
    assert f3d.MapSceneBuildingLayer is map_scene.BuildingLayer
    assert f3d.BuildingLayer is buildings.BuildingLayer
    assert f3d.BuildingLayer is not map_scene.BuildingLayer


def test_p1_public_mapscene_asset_api_shape_exists():
    from forge3d.map_scene import BuildingLayer, LabelLayer, MapScene, Tiles3DLayer

    for name in ("from_features", "from_geodataframe", "from_style_layer", "compile_labels"):
        assert hasattr(LabelLayer, name), f"LabelLayer missing {name}"

    for name in ("from_geojson", "from_cityjson", "from_mesh"):
        assert hasattr(BuildingLayer, name), f"BuildingLayer missing {name}"

    for name in ("from_tileset_json", "from_b3dm"):
        assert hasattr(Tiles3DLayer, name), f"Tiles3DLayer missing {name}"

    assert hasattr(MapScene, "save_bundle")
    assert hasattr(MapScene, "load_bundle")


def test_p1_public_api_does_not_require_viewer_ipc():
    from forge3d.map_scene import BuildingLayer, LabelLayer, Tiles3DLayer

    assert LabelLayer.__module__ == "forge3d.map_scene"
    assert BuildingLayer.__module__ == "forge3d.map_scene"
    assert Tiles3DLayer.__module__ == "forge3d.map_scene"


def test_p1_building_api_compatibility_decision_is_explicit():
    import forge3d as f3d
    from forge3d.map_scene import BuildingLayer as ProductBuildingLayer

    assert hasattr(ProductBuildingLayer, "from_geojson")
    assert hasattr(ProductBuildingLayer, "from_cityjson")
    assert hasattr(ProductBuildingLayer, "from_mesh")
    assert f3d.MapSceneBuildingLayer is ProductBuildingLayer
    assert f3d.BuildingLayer is not ProductBuildingLayer
