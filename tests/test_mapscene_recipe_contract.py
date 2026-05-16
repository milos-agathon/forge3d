import forge3d as f3d
from forge3d.label_plan import KeepoutRegion, PriorityClass


def _recipe_parts():
    terrain = f3d.TerrainSource(
        path="fixtures/dem.tif",
        crs="EPSG:32610",
        metadata={"width": 4, "height": 3},
        elevation_sampling_available=True,
    )
    raster = f3d.RasterOverlay(
        layer_id="ortho",
        path="fixtures/ortho.tif",
        crs="EPSG:32610",
        opacity=0.75,
        metadata={"width": 4, "height": 3},
    )
    vector = f3d.VectorOverlay(
        layer_id="roads",
        features=[
            {
                "id": "road-1",
                "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                "properties": {"class": "primary"},
            }
        ],
        crs="EPSG:32610",
        style={"layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#ffffff"}}]},
    )
    labels = f3d.LabelLayer(
        layer_id="labels",
        labels=[
            {
                "id": "city-a",
                "text": "Alpha",
                "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 0.0)},
                "priority_class": "cities",
            }
        ],
        glyph_atlas={"glyphs": list("Aahlp")},
        priority_rules=[PriorityClass("cities", rank=10)],
    )
    point_cloud = f3d.PointCloudLayer(
        layer_id="points",
        path="fixtures/points.las",
        crs="EPSG:32610",
        point_count=42,
        metadata={"format": "las"},
    )
    building = f3d.MapSceneBuildingLayer(
        layer_id="buildings",
        source="fixtures/buildings.city.json",
        support_level="Pro-gated",
        geometry_count=0,
        bounds=None,
        material_status="scalar_pbr",
    )
    furniture = f3d.MapFurnitureLayer(
        title="Harbor overview",
        legend={"items": ["roads", "labels"]},
        scale_bar={"units": "m"},
        north_arrow={"style": "simple"},
        keepouts=[KeepoutRegion("title", "title", (0, 0, 320, 64), priority=100)],
    )
    camera = f3d.OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=2500.0,
        azimuth_deg=35.0,
        elevation_deg=55.0,
        fov_deg=45.0,
    )
    lighting = f3d.LightingPreset(name="daylight", sun_direction=(0.2, 0.7, 0.4), intensity=1.5)
    output = f3d.OutputSpec(width=800, height=600, format="png", path="map.png")
    return terrain, raster, vector, labels, point_cloud, building, furniture, camera, lighting, output


def test_mapscene_recipe_components_are_public_and_serializable():
    terrain, raster, vector, labels, point_cloud, building, furniture, camera, lighting, output = _recipe_parts()

    recipe = f3d.SceneRecipe(
        terrain=terrain,
        camera=camera,
        lighting=lighting,
        layers=[raster, vector, labels, point_cloud, building],
        output=output,
        map_furniture=furniture,
        render_policy=f3d.RenderFailurePolicy.CONTINUE_ON_WARNING,
        diagnostics_policy={"missing_metadata": "warning"},
    )
    scene = f3d.MapScene(recipe=recipe)

    assert scene.recipe is recipe
    assert scene.last_validation_report is None
    assert scene.compiled_label_plans == {}
    assert hasattr(scene, "validate")
    assert hasattr(scene, "render")
    assert hasattr(scene, "save_bundle")

    payload = recipe.to_dict()
    assert payload["terrain"]["kind"] == "terrain_source"
    assert [layer["kind"] for layer in payload["layers"]] == [
        "raster_overlay",
        "vector_overlay",
        "label_layer",
        "point_cloud_layer",
        "building_layer",
    ]
    assert payload["layers"][2]["priority_rules"][0]["name"] == "cities"
    assert payload["map_furniture"]["keepouts"][0]["kind"] == "title"
    assert payload["camera"]["kind"] == "orbit_camera"
    assert payload["lighting"]["kind"] == "lighting_preset"
    assert payload["output"] == {
        "kind": "output_spec",
        "width": 800,
        "height": 600,
        "format": "png",
        "path": "map.png",
        "metadata": {},
    }


def test_mapscene_constructor_accepts_recipe_keywords_without_raw_ipc():
    terrain, raster, vector, labels, point_cloud, building, furniture, camera, lighting, output = _recipe_parts()

    scene = f3d.MapScene(
        terrain=terrain,
        camera=camera,
        lighting=lighting,
        layers=[raster, vector, labels, point_cloud, building],
        output=output,
        map_furniture=furniture,
    )

    assert isinstance(scene.recipe, f3d.SceneRecipe)
    assert scene.recipe.to_dict()["layers"][0]["layer_id"] == "ortho"
    assert "MapScene" in f3d.__all__
    assert "SceneRecipe" in f3d.__all__
    assert "MapSceneBuildingLayer" in f3d.__all__
    assert f3d.MapSceneBuildingLayer is not f3d.BuildingLayer
