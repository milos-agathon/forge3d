from dataclasses import replace
import json

import numpy as np
import pytest

import forge3d as f3d
from forge3d.geospatial_materials import GeospatialMaterialRecipe, MaterialRasterStyle


def test_sample_normal_direction_is_independent_of_arbitrary_scale():
    mask=f3d.MaterialMask("range",field="slope",units="degrees",lower=40,upper=50)
    material=f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("base",(0,0,0)),f3d.MaterialLayer("east-facing slope",(255,255,255),mask)))
    result=material.evaluate_samples([(500000,4600000)]*3,crs="EPSG:32632",units="m",
        normals_enu=[(1,0,1),(1e300,0,1e300),(1e-300,0,1e-300)])
    assert result.masks["east-facing slope"].all()
    np.testing.assert_array_equal(result.rgba,np.full((3,4),255,dtype=np.uint8))


def recipe(categorical=None):
    return GeospatialMaterialRecipe((f3d.MaterialLayer("base", (120, 100, 80)),
        f3d.MaterialLayer("high", (230, 220, 210), mask=f3d.MaterialMask("range", field="value", units="kg", lower=10, upper=20))), categorical=categorical)


def features():
    return [{"type": "Feature", "id": str(i), "properties": {"height": 20 + i},
             "geometry": {"type": "Polygon", "coordinates": [[[i, 0], [i + .4, 0], [i + .4, .4], [i, .4], [i, 0]]]}} for i in range(2)]


def test_same_recipe_raster_vector_and_building_albedo_meaning():
    authored = recipe()
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    raster = authored.prepare_raster(np.array([[5, 15], [5, 15]], np.float32), grid, units="kg")
    values = {"value": (np.array([5, 15]), "kg")}
    anchors = [[500000, 4600000], [500002, 4600000]]
    vector = f3d.VectorOverlay("v", features=features(), crs=grid.crs)
    bound_vector = authored.bind_vector(vector, anchors, units="m", fields=values)
    building = f3d.MapSceneBuildingLayer("b", features=features(), metadata={"crs": grid.crs})
    bound_building = authored.bind_buildings(building, anchors, units="m", fields=values)
    for index in range(2):
        expected = raster.rgba[0, index].tolist()
        assert bound_vector.features[index]["properties"]["forge3d:material"]["rgba"] == expected
        assert bound_building.features[index]["properties"]["forge3d:material"]["rgba"] == expected
    assert vector.features[0]["properties"] == {"height": 20}
    assert bound_building.features[0]["id"] == "0"
    from forge3d.map_scene import _native_building_mesh_batches_for_layers
    batches = _native_building_mesh_batches_for_layers([bound_building])
    assert batches and len(batches) == 2
    from forge3d.terrain_materials import _linear
    np.testing.assert_array_equal(batches[1]["color"], [*_linear(raster.rgba[0, 1, :3] / 255), 1])


def test_shared_publication_legend_and_replacement():
    palette = f3d.CategoricalRasterStyle({1: (120, 100, 80, 255), 2: (230, 220, 210, 255)}, {1: "Low", 2: "High"})
    material = GeospatialMaterialRecipe((f3d.MaterialLayer("base", (120, 100, 80)),), categorical=palette)
    vector = f3d.VectorOverlay("v", features=features(), crs="EPSG:32632")
    vector = material.bind_vector(vector, [[500000,4600000],[500002,4600000]], units="m", classes=[1,2])
    scene = f3d.SceneRecipe(f3d.TerrainSource(data=np.zeros((4,4),np.float32), crs="EPSG:32632"),
        f3d.OrbitCamera(distance=30),f3d.LightingPreset("custom"), output=f3d.OutputSpec(width=384,height=256))
    first = material.attach(scene, vector)
    second = material.attach(first, vector)
    assert len(second.layers) == 1
    assert second.map_furniture.legend["entries"] == [dict(entry, interpretation="Exact display swatch", binding="v")
                                                     for entry in palette.legend()["entries"]]


def test_sample_coordinates_normals_and_units():
    authored = GeospatialMaterialRecipe((f3d.MaterialLayer("base", (100, 100, 100)),
        f3d.MaterialLayer("roof", (255, 255, 255), mask=f3d.MaterialMask("range", field="slope", units="degrees", lower=0, upper=10))))
    result = authored.evaluate_samples([[500000, 4600000, 100], [500002, 4600000, 110]],
        crs="EPSG:32632", units="m", normals_enu=[[0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(result.rgba[:, :3], [[255, 255, 255], [100, 100, 100]])
    with pytest.raises(ValueError, match="missing"):
        authored.evaluate_samples([[500000, 4600000]], crs="EPSG:32632", units="m")
    with pytest.raises(ValueError, match="units"):
        authored.evaluate_samples([[500000, 4600000]], crs="EPSG:32632", units="feet")


def test_raster_binding_roundtrip_and_missing_observations():
    authored = recipe()
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    values = np.array([[5, 15], [np.nan, 5]], np.float32)
    style = MaterialRasterStyle(authored, grid, value_units="kg")
    from forge3d.raster_style import raster_style_from_dict
    decoded = raster_style_from_dict(style.to_dict())
    assert json.dumps(style.to_dict(), sort_keys=True) == json.dumps(decoded.to_dict(), sort_keys=True)
    assert not style.apply(values).rgba[1, 0].any()
    mask = f3d.MaterialMask("invert", children=(f3d.MaterialMask("range", field="value", units="kg", lower=10, upper=20),))
    assert not mask.evaluate({"value": (values, "kg")}, None, np.ones((2, 2), bool))[1, 0]


def test_unsupported_channels_and_modes_raise_without_fallback():
    vector = f3d.VectorOverlay("v", features=features(), crs="EPSG:32632")
    building = f3d.MapSceneBuildingLayer("b", features=features(), metadata={"crs": "EPSG:32632"})
    values = {"value": ([5, 15], "kg")}
    anchors = [[500000, 4600000], [500002, 4600000]]
    protected = recipe().bind_buildings(building, anchors, units="m", fields=values, presentation="protected")
    assert protected.metadata["material_presentation"] == "protected building swatches"
    assert all(f["properties"]["forge3d:material"]["presentation"] == "protected" for f in protected.features)
    physical = replace(recipe(), presentation="physical").bind_vector(vector, anchors, units="m", fields=values)
    assert physical.metadata["material_presentation"] == "physical terrain albedo"
    with pytest.raises(ValueError, match="repeating surface textures"):
        recipe().with_layer("base", texture=f3d.MaterialTexture("asset.png", (16, 16), "srgb", "albedo")).bind_vector(vector, anchors, units="m", fields=values)
    assert not hasattr(vector, "material_placeholder")


def test_generic_public_exports_and_canonical_authoring():
    assert GeospatialMaterialRecipe.from_dict(recipe().to_dict()).to_dict() == recipe().to_dict()
    for name in ("GeospatialMaterialRecipe", "MaterialLayer", "MaterialEvaluation", "MaterialRasterStyle"):
        assert name in f3d.__all__
