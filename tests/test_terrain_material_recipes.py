"""C01 contract: source meaning, ground scale and the native scene seam."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
from forge3d.terrain_materials import (
    MaterialGrid, MaterialMask, MaterialTexture, TerrainMaterialLayer,
    TerrainMaterialRecipe, TerrainMaterialStyle,
)


def grid(origin=(500000.0, 4600000.0), spacing=(2.0, 3.0)):
    return MaterialGrid("EPSG:32632", origin, spacing, "m", "m")


def recipe(**kwargs):
    return TerrainMaterialRecipe((TerrainMaterialLayer("ground", (170, 125, 80)),), **kwargs)


def test_scale_slope_aspect_composition_and_overrides():
    yy, xx = np.mgrid[:6, :7]
    heights = xx * 2.0 + yy * 3.0
    mask = MaterialMask("all", children=(MaterialMask("range", field="slope", units="degrees", lower=50, upper=60),
        MaterialMask("range", field="value", units="kg", lower=10, upper=12)))
    authored = replace(recipe(), layers=(*recipe().layers, TerrainMaterialLayer("rule", (255, 255, 255), mask)))
    result = authored.prepare(heights, grid(), fields={"value": (np.full_like(heights, 11), "kg")})
    np.testing.assert_allclose(result.slope, np.degrees(np.arctan(np.sqrt(2))))
    np.testing.assert_allclose(result.aspect, 315)
    assert result.masks["rule"].all()
    np.testing.assert_array_equal(result.rgba[..., :3], np.full((*heights.shape, 3), 255))
    assert authored.with_layer("rule", opacity=0).layers[1].opacity == 0
    with pytest.raises(ValueError, match="units"):
        authored.prepare(heights, grid(), fields={"value": (heights, "g")})


def test_adjacent_extents_keep_texture_phase_and_ground_scale(tmp_path):
    path = tmp_path / "texture.png"
    Image.fromarray(np.array([[[255, 10, 20], [20, 255, 10]], [[10, 20, 255], [255, 255, 255]]], np.uint8)).save(path)
    texture = MaterialTexture(str(path), (8, 12), "srgb", "albedo", (500000, 4600000), variation_seed=None)
    authored = TerrainMaterialRecipe((TerrainMaterialLayer("ground", (255, 255, 255), texture=texture),))
    heights = np.zeros((6, 10))
    whole = authored.prepare(heights, grid()).rgba
    left = authored.prepare(heights[:, :5], grid()).rgba
    right = authored.prepare(heights[:, 5:], grid(origin=(500010, 4600000))).rgba
    np.testing.assert_array_equal(np.concatenate((left, right), axis=1), whole)
    np.testing.assert_array_equal(whole[:, :4], whole[:, 4:8])


@pytest.mark.parametrize("mode", ["single", "multidirectional"])
def test_true_neighbours_preserve_relief_and_slope_across_tile_seam(mode):
    yy, xx = np.mgrid[:8, :12]
    source = xx ** 2 + yy ** 2
    authored = recipe(relief_strength=.8, relief_mode=mode)
    full = authored.prepare(source[1:-1, 1:-1], grid(), derivative_halo=source)
    left = authored.prepare(source[1:-1, 1:6], grid(), derivative_halo=source[:, :7])
    right = authored.prepare(source[1:-1, 6:-1], grid(origin=(500010, 4600000)), derivative_halo=source[:, 5:])
    np.testing.assert_array_equal(full.rgba, np.concatenate((left.rgba, right.rgba), axis=1))
    np.testing.assert_array_equal(full.slope, np.concatenate((left.slope, right.slope), axis=1))
    with pytest.raises(ValueError, match="derivative_halo"):
        authored.prepare(source[1:-1, 1:-1], grid(), derivative_halo=np.zeros_like(source))


@pytest.mark.parametrize("mode", ["single", "multidirectional"])
def test_protected_classes_legend_and_relief_changes(mode):
    palette = f3d.CategoricalRasterStyle({2: (80, 140, 50, 255), 7: (20, 90, 220, 255)},
                                       {2: "forest", 7: "water"}, nodata=-1)
    heights = np.arange(36).reshape(6, 6)
    classes = np.full((6, 6), -1)
    classes[:2] = 2
    classes[2:4] = 7
    first = recipe(categorical=palette, relief_strength=1, relief_mode=mode).prepare(heights, grid(), classes=classes)
    second = recipe(categorical=palette, relief_strength=1, relief_azimuth=135, relief_mode=mode).prepare(heights, grid(), classes=classes)
    np.testing.assert_array_equal(first.rgba[:4], second.rgba[:4])
    np.testing.assert_array_equal(first.rgba[0, 0], palette.palette[2])
    assert first.legend == palette.legend() == second.legend
    assert np.any(first.rgba[4:] != second.rgba[4:])


def test_multidirectional_relief_on_known_planar_normals_and_roundtrip():
    from forge3d.widgets import _multidirectional_hillshade
    # One metre east raises the plane one metre; its unit normal is
    # (-1, 0, 1)/sqrt(2) in ENU. At 45-degree altitude, the four Lambert
    # responses are (1-sin(azimuth))/2, with USGS aspect weights.
    heights = np.tile(np.arange(7) * 2.0, (6, 1))
    angles = np.deg2rad([225, 270, 315, 360])
    weights = np.sin(np.deg2rad(270) - angles) ** 2
    responses = np.clip(.18 + .82 * (1 - np.sin(angles)) / 2, 0, 1)
    expected = np.sum(weights * responses) / np.sum(weights)
    actual = _multidirectional_hillshade(heights, 315, 45, dx_m=2, dy_m=3)
    # The existing hillshade computes normals in float32.
    np.testing.assert_allclose(actual, expected, rtol=np.finfo(np.float32).eps)
    flat = _multidirectional_hillshade(np.zeros_like(heights), 315, 45)
    np.testing.assert_allclose(flat, .18 + .82 / np.sqrt(2), rtol=np.finfo(np.float32).eps)
    authored = recipe(relief_strength=.6, relief_mode="multidirectional")
    assert TerrainMaterialRecipe.from_dict(authored.to_dict()) == authored
    assert "relief_mode" not in recipe().to_dict()
    with pytest.raises(ValueError, match="relief_mode"):
        recipe(relief_mode="unknown")


def test_masks_nodata_and_missing_inputs():
    heights = np.zeros((4, 4))
    heights[0, 0] = np.nan
    authored = replace(recipe(), layers=(*recipe().layers, TerrainMaterialLayer("class", (0, 0, 0), MaterialMask("classes", class_ids=(2,)))))
    with pytest.raises(ValueError, match="categorical"):
        authored.prepare(heights, grid())
    result = authored.prepare(heights, grid(), classes=np.full_like(heights, 2))
    assert not result.rgba[0, 0].any()
    assert np.isnan(result.aspect[-1, -1])
    for mask in [MaterialMask("range", field="aspect", units="degrees", lower=350, upper=10),
                 MaterialMask("invert", children=(MaterialMask("classes", class_ids=(2,)),))]:
        assert not mask.evaluate({"aspect": (result.aspect, "degrees")}, np.full_like(heights, 2), result.valid_mask).any()


def test_one_sided_ranges_reuse_thresholds_without_invented_extents():
    values = np.array([100, 300, 100000, np.nan])
    valid = np.ones(values.shape, bool)
    mask = MaterialMask("range", field="elevation", units="m", lower=300)
    np.testing.assert_array_equal(mask.evaluate({"elevation": (values, "m")}, None, valid), [False, True, True, False])
    mask = MaterialMask("range", field="elevation", units="m", upper=300)
    np.testing.assert_array_equal(mask.evaluate({"elevation": (values, "m")}, None, valid), [True, True, False, False])


def test_smooth_material_weights_compose_without_changing_class_meaning():
    heights = np.tile(np.array([0., 10., 20., 30.]), (3, 1))
    ramp = MaterialMask("ramp", field="elevation", units="m", lower=10, upper=30)
    inverse = MaterialMask("invert", children=(ramp,))
    fields = {"elevation": (np.array([10., 20., 30., np.nan]), "m")}
    valid = np.ones(4, bool)
    np.testing.assert_array_equal(ramp.evaluate(fields, None, valid), [0, .5, 1, 0])
    np.testing.assert_array_equal(inverse.evaluate(fields, None, valid), [1, .5, 0, 0])
    all_mask = MaterialMask("all", children=(ramp, MaterialMask("range",field="elevation",units="m",upper=20)))
    np.testing.assert_array_equal(all_mask.evaluate(fields, None, valid), [0, .5, 0, 0])
    authored = TerrainMaterialRecipe((TerrainMaterialLayer("ground", (0,0,0)), TerrainMaterialLayer("snow", (255,255,255), ramp)))
    result = authored.prepare(heights, grid())
    # A half-weight white layer over black is half linear radiance, sRGB 188.
    np.testing.assert_array_equal(result.rgba[0,:,0], [0,0,188,255])
    palette = f3d.CategoricalRasterStyle({1:(20,90,220,255)}, {1:"Observed class"})
    protected = replace(authored, categorical=palette).prepare(heights,grid(),classes=np.ones_like(heights))
    np.testing.assert_array_equal(protected.rgba, np.broadcast_to(palette.palette[1],protected.rgba.shape))
    assert TerrainMaterialRecipe.from_dict(authored.to_dict()) == authored
    with pytest.raises(ValueError, match="increasing bounds"):
        MaterialMask("ramp",field="elevation",units="m",lower=30,upper=10)


def test_inverse_class_masks_do_not_turn_declared_nodata_into_observation():
    palette = f3d.CategoricalRasterStyle({1:(20,90,220,255),2:(176,67,33,255)}, {1:"Observation",2:"Other observation"}, nodata=0)
    inverted = MaterialMask("invert",children=(MaterialMask("classes",class_ids=(1,)),))
    authored = TerrainMaterialRecipe((TerrainMaterialLayer("ground",(0,0,0)),
        TerrainMaterialLayer("other observed class",(255,255,255),inverted)),categorical=palette)
    classes = np.tile([0,1,2],(3,1))
    prepared = authored.prepare(np.zeros((3,3)),grid(),classes=classes)
    np.testing.assert_array_equal(prepared.masks["other observed class"][0],[False,False,True])
    samples = f3d.GeospatialMaterialRecipe.from_dict(authored.to_dict()).evaluate_samples(
        [(500000,4600000),(500002,4600000),(500004,4600000)],crs="EPSG:32632",units="m",classes=classes[0])
    np.testing.assert_array_equal(samples.rgba,prepared.rgba[0])
    for ambiguous in (np.array([1,2,3],dtype=object),np.array(['1','2','3']),np.array([1j,2j,3j])):
        with pytest.raises(ValueError,match="real numeric"):
            authored.prepare(np.zeros((3,3)),grid(),classes=np.tile(ambiguous,(3,1)))


@pytest.mark.parametrize("factory", [
    lambda: MaterialGrid("EPSG:4326", (0, 0), (1, 1), "m", "m"),
    lambda: MaterialGrid("EPSG:32632", (0, 0), (1, 1), "", "m"),
    lambda: MaterialTexture("x.png", (1, 1), "linear", "normal"),
    lambda: MaterialTexture("x.png", (float("nan"), 1), "srgb", "albedo"),
    lambda: MaterialMask("range", field="slope", units="degrees", lower=20, upper=10),
    lambda: recipe(relief_strength=float("nan")),
    lambda: recipe(presentation="physical", relief_strength=1),
    lambda: recipe(categorical=f3d.CategoricalRasterStyle({1: (1, 2, 3, 100)})),
])
def test_ambiguous_or_nonfinite_inputs_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_recipe_and_bound_style_canonical_roundtrip(tmp_path):
    palette = f3d.CategoricalRasterStyle({2: (80, 140, 50, 255)}, {2: "forest"})
    authored = recipe(categorical=palette)
    path = authored.save(tmp_path / "recipe.json")
    loaded = TerrainMaterialRecipe.load(path)
    second = loaded.save(tmp_path / "second.json")
    assert path.read_bytes() == second.read_bytes()
    style = TerrainMaterialStyle(authored, grid(), np.full((4, 4), 2), {"value": (np.zeros((4, 4)), "kg")})
    from forge3d.raster_style import raster_style_from_dict
    decoded = raster_style_from_dict(style.to_dict())
    assert json.dumps(style.to_dict(), sort_keys=True) == json.dumps(decoded.to_dict(), sort_keys=True)
    np.testing.assert_array_equal(style.apply(np.zeros((4, 4))).rgba, decoded.apply(np.zeros((4, 4))).rgba)


def test_relative_texture_recipe_reloads_without_rewriting_reference(tmp_path, monkeypatch):
    Image.new("RGB", (2, 2), "red").save(tmp_path / "texture.png")
    texture = MaterialTexture("texture.png", (4, 4), "srgb", "albedo")
    authored = TerrainMaterialRecipe((TerrainMaterialLayer("ground", (255, 255, 255), texture=texture),))
    path = authored.save(tmp_path / "recipe.json")
    loaded = TerrainMaterialRecipe.load(path)
    assert loaded.save(tmp_path / "second.json").read_bytes() == path.read_bytes()
    monkeypatch.chdir(tmp_path.parent)
    assert loaded.prepare(np.zeros((3, 3)), grid()).rgba[0, 0, 0] == 255
    bound = loaded.bind(scene_recipe(), grid())
    assert Path(bound.terrain.style.recipe.layers[0].texture.path).is_absolute()
    relocated = tmp_path / "publication"
    relocated.mkdir()
    relocated_recipe = TerrainMaterialRecipe.load(loaded.save(relocated / "recipe.json"))
    np.testing.assert_array_equal(relocated_recipe.prepare(np.zeros((3, 3)), grid()).rgba,
                                  loaded.prepare(np.zeros((3, 3)), grid()).rgba)


def scene_recipe():
    return f3d.SceneRecipe(f3d.TerrainSource(data=np.zeros((8, 8), np.float32), crs="EPSG:32632"),
        f3d.OrbitCamera(distance=40), f3d.LightingPreset(name="custom"), output=f3d.OutputSpec(width=256, height=256))


def test_bound_recipe_native_params_and_legend():
    palette = f3d.CategoricalRasterStyle({2: (80, 140, 50, 255)}, {2: "forest"})
    bound = recipe(categorical=palette).bind(scene_recipe(), grid(), classes=np.full((8, 8), 2))
    assert bound.map_furniture.legend["entries"] == [dict(entry, interpretation="Exact display swatch", binding="terrain")
                                                   for entry in palette.legend()["entries"]]
    from forge3d.map_scene import _build_mapscene_terrain_params
    params = _build_mapscene_terrain_params(bound, bound.terrain.data, (256, 256))
    assert params is not None
    assert isinstance(bound.terrain.style, TerrainMaterialStyle)
    hdr_bound = replace(bound, output=f3d.OutputSpec(width=256, height=256, samples=4,
                                                    hdr=True, denoiser="atrous"))
    bound.terrain.style.validate_scene(hdr_bound)
    assert hdr_bound.output.hdr and hdr_bound.output.samples == 4
    # GPU tests also assert exact display swatches and float32 linear EXR values
    # after accumulation/denoise; acceptance is stronger than mere validation.


def test_native_ground_scale_rejects_square_stretch_of_rectangular_extent():
    base = replace(scene_recipe(), lighting=f3d.LightingPreset("custom", settings={"cli_params": {"camera_mode": "mesh:zup:north"}}))
    with pytest.raises(ValueError, match="square ground footprint"):
        recipe().bind(base, grid())  # 16 by 24 ground metres, not square.
    assert recipe().bind(base, grid(spacing=(2, 2))).terrain.style.grid.spacing == (2, 2)


def test_texture_bytes_participate_in_cache_identity(tmp_path):
    path = tmp_path / "texture.png"
    Image.new("RGB", (2, 2), "red").save(path)
    texture = MaterialTexture(str(path), (8, 8), "srgb", "albedo")
    authored = TerrainMaterialRecipe((TerrainMaterialLayer("ground", (255, 255, 255), texture=texture),))
    bound = authored.bind(scene_recipe(), grid())
    from forge3d._map_scene_rasters import raster_content_hashes
    before = raster_content_hashes(bound)
    Image.new("RGB", (2, 2), "blue").save(path)
    after = raster_content_hashes(bound)
    assert before != after


def test_public_exports():
    for name in ("MaterialGrid", "MaterialTexture", "MaterialMask", "TerrainMaterialLayer", "TerrainMaterialRecipe", "TerrainMaterialStyle", "PreparedTerrainMaterial"):
        assert name in f3d.__all__ and hasattr(f3d, name)


def test_prepared_raster_and_meaning_sidecar(tmp_path):
    palette = f3d.CategoricalRasterStyle({2: (80, 140, 50, 255)}, {2: "forest"})
    result = recipe(categorical=palette).prepare(np.zeros((3, 3)), grid(), classes=np.full((3, 3), 2))
    path = result.save_raster(tmp_path / "material.png")
    np.testing.assert_array_equal(np.asarray(Image.open(path)), result.rgba)
    sidecar = json.loads(path.with_suffix(".png.json").read_text())
    assert sidecar["display_transform"] == "sRGB"
    assert sidecar["legend"] == palette.legend()
    assert sidecar["grid"] == grid().to_dict()


def test_linear_channels_composition_scale_assets_and_legacy_bytes(tmp_path):
    normal_path = tmp_path / "normal.png"
    roughness_path = tmp_path / "rough.png"
    Image.fromarray(np.tile(np.array([[[218, 128, 218]]], dtype=np.uint8), (2, 2, 1))).save(normal_path)
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(roughness_path)
    normal = MaterialTexture("normal.png", (8, 8), "linear", "normal", normal_convention="opengl")
    rough = MaterialTexture("rough.png", (8, 8), "linear", "roughness", variation_seed=None)
    material = TerrainMaterialRecipe((
        TerrainMaterialLayer("ground", (120, 120, 120), roughness=.2),
        TerrainMaterialLayer("rock", (180, 180, 180), MaterialMask("ramp", "elevation", "m", 0, 2),
                             normal=normal, roughness_texture=rough),
    ), presentation="physical")
    path = material.save(tmp_path / "recipe.json")
    loaded = TerrainMaterialRecipe.load(path)
    heights = np.ones((4, 8))
    material_grid = grid((0, 0), (4, 4))
    result = loaded.prepare(heights, material_grid)
    # Ground and rock blend equally. Interpolation of roughness data stays
    # linear (unlike sRGB colour); the existing native normal basis consumes
    # the normalized direction after material-weight composition.
    np.testing.assert_allclose(result.roughness, .35)
    np.testing.assert_allclose(np.linalg.norm(result.normals, axis=-1), 1)
    assert np.all(result.normals[..., 0] > 0)
    np.testing.assert_array_equal(result.normals[:, :4], result.normals[:, 4:])
    assert loaded.to_dict() == material.to_dict()
    assert loaded.resolved().layers[1].normal.resolved_path == normal_path
    destination = tmp_path / "nested"
    destination.mkdir()
    reread = TerrainMaterialRecipe.load(loaded.save(destination / "recipe.json"))
    np.testing.assert_array_equal(reread.prepare(heights, material_grid).normals, result.normals)
    result.save_raster(tmp_path / "surface.png")
    sidecar = json.loads((tmp_path / "surface.png.json").read_text())
    assert sidecar["surface_channels"]["normal"]["color_space"] == "linear"
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "surface.roughness.png"))[..., 0],
                                  np.full(heights.shape, round(.35 * 255), np.uint8))
    legacy = recipe().to_dict()
    assert not any(key in legacy["layers"][0] for key in ("normal", "roughness", "roughness_texture"))
    assert TerrainMaterialRecipe.from_dict(legacy).to_dict() == legacy


def test_channel_contracts_reject_ambiguous_and_unconsumed_data(tmp_path):
    with pytest.raises(ValueError, match="linear"):
        MaterialTexture("x", (1, 1), "srgb", "roughness")
    with pytest.raises(ValueError, match="opengl"):
        MaterialTexture("x", (1, 1), "linear", "normal")
    combined = TerrainMaterialRecipe((TerrainMaterialLayer("ground", (1, 2, 3), roughness=.5),))
    assert combined.presentation == "protected" and combined.has_surface_channels
    with pytest.raises(ValueError, match="every layer"):
        TerrainMaterialRecipe((TerrainMaterialLayer("a", (1, 2, 3), roughness=.5),
                               TerrainMaterialLayer("b", (1, 2, 3))), presentation="physical")
    image = tmp_path / "ambiguous.png"
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(image)
    with pytest.raises(ValueError, match="grayscale"):
        MaterialTexture(str(image), (1, 1), "linear", "roughness").sample(grid(), (2, 2))
    with pytest.raises(ValueError, match="albedo channel"):
        TerrainMaterialLayer("a", (1, 2, 3), texture=MaterialTexture(str(image), (1, 1), "linear", "normal", normal_convention="opengl"))


def test_rock_snow_preset_retains_region_masks_and_uses_explicit_units():
    region = MaterialMask("classes", class_ids=(1,))
    recipe = f3d.GeospatialMaterialRecipe.rock_snow(
        TerrainMaterialLayer("ground", (100, 90, 80)),
        TerrainMaterialLayer("rock", (50, 60, 70), mask=region),
        TerrainMaterialLayer("snow", (250, 250, 250), mask=region),
        rock_slope_degrees=(20, 30), snow_elevation_m=(280, 320), snow_slope_degrees=(30, 45))
    assert isinstance(recipe, f3d.GeospatialMaterialRecipe)
    evaluated = recipe.evaluate_samples([[0, 0, 300], [1, 0, 300]], crs="EPSG:32632", units="m",
        normals_enu=[[0, 0, 1], [0, 0, 1]], classes=[1, 2])
    np.testing.assert_array_equal(evaluated.masks["snow"], [.5, 0])
    np.testing.assert_array_equal(evaluated.rgba[1, :3], [100, 90, 80])
    assert f3d.GeospatialMaterialRecipe.from_dict(recipe.to_dict()).to_dict() == recipe.to_dict()
