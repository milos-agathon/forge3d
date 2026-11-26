"""C01 authoring contracts independent of GPU sampling."""
import numpy as np
import pytest
from dataclasses import replace
from PIL import Image

import forge3d as f3d
from test_c01_review_gpu import _scene


def test_one_legend_preserves_each_binding_even_when_swatches_match(monkeypatch):
    grid, base = _scene()
    palette = f3d.CategoricalRasterStyle({1: (40, 60, 20, 255)}, {1: "same class"})
    material = f3d.GeospatialMaterialRecipe.land_cover(palette, albedo=(150, 140, 130))
    classes = np.ones((64, 64), np.int16)
    scene = material.bind(base, grid, classes=classes)
    raster = material.bind_raster(base, classes, grid, units="class", classes=classes,
                                  layer_id="raster").layers[-1]
    feature = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [500032, 4599968]}}
    vector = material.bind_vector(f3d.VectorOverlay("vector", features=[feature], crs=grid.crs),
                                  [[500032, 4599968]], units="m", classes=[1])
    ring = [[500020, 4599980], [500044, 4599980], [500044, 4599956],
            [500020, 4599956], [500020, 4599980]]
    building = material.bind_buildings(f3d.MapSceneBuildingLayer("building", metadata={"crs": grid.crs},
        features=[replace_feature(feature, ring)]), [[500032, 4599968]], units="m", classes=[1])
    attached = material.attach(scene, raster, vector, building)
    entries = attached.map_furniture.legend["entries"]
    assert {entry["binding"] for entry in entries} == {"terrain", "raster", "vector", "building"}
    assert all(entry["interpretation"] == ("Surface albedo; lit pixels differ"
               if entry["binding"] == "building" else "Exact display swatch") for entry in entries)
    from forge3d import _map_scene_render as render
    captions = []
    monkeypatch.setattr(render, "_draw_text", lambda image, text, *args, **kwargs: captions.append(text))
    render._draw_simple_legend(np.zeros((720, 1280, 4), np.uint8), attached.map_furniture.legend)
    assert all(any(caption.startswith(binding + ": ") for caption in captions)
               for binding in ("terrain", "raster", "vector", "building"))
    positioned = np.zeros((720, 1280, 4), np.uint8)
    render._draw_simple_legend(positioned, {**attached.map_furniture.legend, "origin_px": [12, 42]})
    np.testing.assert_array_equal(positioned[42, 12], (255, 255, 255, 255))


def replace_feature(feature, ring):
    return {**feature, "properties": {"height": 25},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


@pytest.mark.parametrize("metadata_key", ("material_maps", "materials"))
def test_albedo_only_recipe_preserves_existing_native_surface_channels(tmp_path, metadata_key):
    from forge3d._map_scene_rasters import material_surface_maps
    grid, base = _scene()
    path = tmp_path / "legacy-roughness.png"
    Image.new("RGB", (2, 2), (128, 128, 128)).save(path)
    base = replace(base, terrain=replace(base.terrain,
        metadata={metadata_key: {"roughness_path": str(path)}}))
    material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer("ground", (150, 140, 130)),),
                                       presentation="physical")
    assert not material.has_surface_channels
    bound = material.bind(base, grid)
    with material_surface_maps(bound) as native:
        assert native.terrain.metadata["material_maps"]["roughness_path"] == str(path)
        assert native.terrain.metadata["material_maps"]["physical_normals"] is True


def test_material_stack_composes_native_channels_and_raster_nodata(tmp_path):
    from forge3d._map_scene_rasters import material_surface_maps

    grid, base = _scene()
    path = tmp_path / "tangent.png"
    Image.new("RGB", (2, 2), (128, 204, 230)).save(path)
    tangent = f3d.MaterialTexture(str(path), (8, 8), "linear", "normal", normal_convention="opengl")
    ground = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer(
        "ground", (150, 140, 130), normal=tangent, roughness=0.6),), presentation="physical")
    cover = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer(
        "cover", (150, 140, 130), roughness=0.2),), presentation="physical")
    values = np.ones((64, 64), np.float32)
    terrain = ground.bind(base, grid)
    with material_surface_maps(terrain) as native:
        ground_normal = np.array(Image.open(native.terrain.metadata["material_maps"]["normal_path"]))

    values[:, :32] = np.nan
    stack = cover.bind_raster(terrain, values, grid, units="class")
    with material_surface_maps(stack) as native:
        maps = native.terrain.metadata["material_maps"]
        assert maps["physical_normals"] is True
        np.testing.assert_array_equal(Image.open(maps["normal_path"]), ground_normal)
        roughness = np.array(Image.open(maps["roughness_path"]))[..., 0]
        np.testing.assert_array_equal(roughness[:, :32], 153)
        np.testing.assert_array_equal(roughness[:, 32:], 51)
        assert "mask_path" not in maps

    albedo_only = replace(ground, layers=(replace(ground.layers[0], normal=None, roughness=None),))
    stack = cover.bind_raster(albedo_only.bind(base, grid), values, grid, units="class")
    with material_surface_maps(stack) as native:
        maps = native.terrain.metadata["material_maps"]
        np.testing.assert_array_equal(np.array(Image.open(maps["mask_path"]))[..., 0],
                                      np.isfinite(values).astype(np.uint8) * 255)
        np.testing.assert_array_equal(np.array(Image.open(maps["roughness_path"]))[:, 32:, 0], 51)

    ground_normal_only = replace(ground, layers=(replace(ground.layers[0], roughness=None),))
    stack = cover.bind_raster(ground_normal_only.bind(base, grid), values, grid, units="class")
    with pytest.raises(ValueError, match="different coverage.*shared native material mask"):
        with material_surface_maps(stack):
            pytest.fail("unequal channel masks cannot be uploaded together")

    values[:] = 1
    stack = cover.bind_raster(ground_normal_only.bind(base, grid), values, grid, units="class")
    with material_surface_maps(stack) as native:
        maps = native.terrain.metadata["material_maps"]
        np.testing.assert_array_equal(Image.open(maps["normal_path"]), ground_normal)
        np.testing.assert_array_equal(np.array(Image.open(maps["roughness_path"]))[..., 0], 51)
        assert "mask_path" not in maps
