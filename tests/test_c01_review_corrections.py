"""Regressions from the 2026-10-15 C01 review, independent of old certificates."""
from dataclasses import replace

import numpy as np
from PIL import Image

import forge3d as f3d


def test_georeferenced_building_keeps_footprint_and_metre_height(monkeypatch):
    from forge3d import map_scene as ms
    grid, scene, material, _ = _fixture()
    bound = material.bind(scene,grid,classes=np.ones((4,4),np.int16))
    ring = [[500001,4599999],[500003,4599999],[500003,4599997],[500001,4599997],[500001,4599999]]
    buildings = f3d.MapSceneBuildingLayer('site',features=[{'type':'Feature','id':'site',
        'properties':{'height':20},'geometry':{'type':'Polygon','coordinates':[ring]}}],metadata={'crs':grid.crs})
    captured=[]
    original=ms._native_building_mesh_batches_for_layers
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        captured.extend(result)
        return result
    monkeypatch.setattr(ms,'_native_building_mesh_batches_for_layers',observe)
    result=ms._terrain_scatter_building_batches_for_recipe(replace(bound,layers=(buildings,)),scene.terrain.data)
    assert result[1]['building_coordinate_space']=='terrain_crs'
    batch=captured[0]
    expected=np.array([.25*1.7-.85,.5*1.7-.85],np.float32)
    np.testing.assert_array_equal(np.unique(batch['positions'][:,0]),expected)
    np.testing.assert_array_equal(np.unique(batch['positions'][:,2]),expected)
    assert batch['wall_height'] * (4/1.7) == 20


def test_terrain_material_and_class_raster_share_ground_and_legend():
    from forge3d.map_scene import _mapscene_thematic_surface
    grid, scene, material, _ = _fixture()
    ground = replace(material, categorical=None)
    bound = ground.bind(scene, grid)
    classes = np.zeros((4, 4), np.int16)
    classes[:, :2] = 1
    palette = f3d.CategoricalRasterStyle({1:(80,140,50,255)}, {1:'forest'}, nodata=0)
    raster = f3d.RasterOverlay('classes', data=classes, crs=grid.crs, style=palette,
        metadata={'material_legend':palette.legend(), 'material_presentation':'protected'})
    bound = material.attach(bound, raster)
    bound.terrain.style.validate_scene(bound)
    actual = _mapscene_thematic_surface(bound, (4,4))
    np.testing.assert_array_equal(actual.rgba[:, :2], np.tile((80,140,50,255),(4,2,1)))
    np.testing.assert_array_equal(actual.rgba[:, 2:], np.tile((150,140,130,255),(4,2,1)))
    assert actual.valid_mask.all()
    assert bound.map_furniture.legend['entries'][0]['binding'] == 'classes'


def _fixture():
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    scene = f3d.SceneRecipe(f3d.TerrainSource(data=np.zeros((4, 4), np.float32), crs=grid.crs),
        f3d.OrbitCamera(distance=30), f3d.LightingPreset("custom"),
        output=f3d.OutputSpec(width=384, height=256))
    palette = f3d.CategoricalRasterStyle({1: (80, 140, 50, 255)}, {1: "forest"})
    recipe = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("ground", (150, 140, 130)),), categorical=palette)
    features = [{"type": "Feature", "id": "0", "properties": {"height": 20},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [.4, 0], [.4, .4], [0, .4], [0, 0]]]}}]
    return grid, scene, recipe, features


def test_legends_keep_binding_specific_meaning():
    grid, scene, recipe, features = _fixture()
    anchors = [[500000, 4600000]]
    vector = recipe.bind_vector(f3d.VectorOverlay("v", features=features, crs=grid.crs), anchors, units="m", classes=[1])
    buildings = recipe.bind_buildings(f3d.MapSceneBuildingLayer("b", features=features, metadata={"crs": grid.crs}),
                                     anchors, units="m", classes=[1])
    classes = np.ones((4, 4), np.int16)
    physical = replace(recipe, presentation="physical")
    a = recipe.attach(physical.bind(scene, grid, classes=classes), vector)
    b = recipe.attach(recipe.bind(scene, grid, classes=classes), buildings)
    c = physical.bind_raster(scene, np.ones((4, 4)), grid, units="kg", classes=classes)
    for bound in (a, b):
        entries = bound.map_furniture.legend["entries"]
        # Same class/swatch needs two entries when the bindings interpret it differently.
        assert {e["interpretation"] for e in entries} == {"Exact display swatch", "Surface albedo; lit pixels differ"}
        assert len(entries) == 2
    assert c.map_furniture.legend["entries"][0]["interpretation"] == "Surface albedo; lit pixels differ"


def test_opengl_export_and_native_normal_frames_are_explicit(tmp_path):
    from forge3d._map_scene_rasters import material_surface_maps
    grid, scene, recipe, _ = _fixture()
    path = tmp_path / "north.png"
    Image.fromarray(np.full((2, 2, 3), [128, 171, 247], np.uint8)).save(path)
    normal = f3d.MaterialTexture(str(path), (16, 16), "linear", "normal", normal_convention="opengl")
    material = replace(recipe, categorical=None, presentation="physical",
                       layers=(f3d.MaterialLayer("ground", (150, 140, 130), normal=normal, roughness=1),))
    prepared = material.prepare(np.zeros((4, 4)), grid)
    assert (prepared.normals[..., 1] > 0).all()  # exported +G means north
    for bound in (material.bind(scene, grid), material.bind_raster(scene, np.ones((4, 4)), grid, units="kg")):
        with material_surface_maps(bound) as native_recipe:
            pixels = np.asarray(Image.open(native_recipe.terrain.metadata["material_maps"]["normal_path"]))
            assert (pixels[..., 1] < 128).all()  # native bitangent is south
            assert (pixels[..., 0] == 128).all()  # X remains east


def test_texture_grid_footprint_filters_linear_premultiplied_alpha(tmp_path):
    grid, _, _, _ = _fixture()
    path = tmp_path / "checker.png"
    pixels = np.zeros((64, 64, 4), np.uint8)
    pixels[:, :, :] = (255, 255, 255, 255)
    pixels[3::4, :, :] = (255, 0, 0, 0)  # transparent RGB must not bleed
    Image.fromarray(pixels).save(path)
    texture = f3d.MaterialTexture(str(path), (2, 2), "srgb", "albedo")
    rgb, alpha = texture.sample(grid, (4, 4))
    np.testing.assert_allclose(rgb, 1)
    np.testing.assert_allclose(alpha, .75)


def test_texture_breaks_32m_period_and_preserves_adjacent_and_halo_bytes(tmp_path):
    grid, _, _, _ = _fixture()
    path = tmp_path / "noise.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(path)
    texture = f3d.MaterialTexture(str(path), (32, 32), "srgb", "albedo")
    rgb, _ = texture.sample(grid, (32, 48))
    assert not np.array_equal(rgb[:, 16:], rgb[:, :-16])
    assert not np.array_equal(rgb[16:], rgb[:-16])
    assert "variation_seed" in texture.to_dict()
    left, _ = texture.sample(grid, (32, 24))
    right, _ = texture.sample(replace(grid, origin=(500048, 4600000)), (32, 24))
    np.testing.assert_array_equal(rgb, np.concatenate((left, right), axis=1))
    halo, _ = texture.sample(replace(grid, origin=(499998, 4600002)), (34, 50))
    np.testing.assert_array_equal(rgb, halo[1:-1, 1:-1])


def test_raster_without_elevation_uses_its_grid_footprint(tmp_path):
    grid, _, _, _ = _fixture()
    path = tmp_path / "opaque-stripes.png"
    pixels = np.full((64, 64, 3), 255, np.uint8)
    pixels[3::4] = 0
    Image.fromarray(pixels).save(path)
    texture = f3d.MaterialTexture(str(path), (2, 2), "srgb", "albedo")
    material = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("ground", (255, 255, 255), texture=texture),))
    terrain = material.prepare(np.zeros((4, 4)), grid)
    raster = material.prepare_raster(np.zeros((4, 4)), grid, units="kg")
    np.testing.assert_array_equal(terrain.rgba, raster.rgba)


def test_pre_seed_recipe_keeps_canonical_bytes():
    material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer("ground", (255, 255, 255),
        texture=f3d.MaterialTexture("texture.png", (32, 32), "srgb", "albedo")),))
    old = material.to_dict()
    old["layers"][0]["texture"].pop("variation_seed")
    restored = f3d.TerrainMaterialRecipe.from_dict(old)
    assert restored.to_dict() == old
    assert restored.layers[0].texture.variation_seed is None


def test_antitiling_variance_stays_in_measured_footprint_phase_range(tmp_path):
    from forge3d.terrain_materials import _linear
    grid, _, _, _ = _fixture()
    path = tmp_path / "variance.png"
    pixels = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    texture = f3d.MaterialTexture(str(path), (32, 32), "srgb", "albedo")
    actual, _ = texture.sample(grid, (512, 512))
    source = _linear(pixels/255)
    # Measure the range of true 8x8-texel footprint variance over every
    # integer phase. This source-derived range replaces an arbitrary tolerance.
    reference = []
    for y in range(8):
        for x in range(8):
            shifted = np.roll(source, (y, x), axis=(0, 1))
            reference.append(shifted.reshape(16, 8, 16, 8, 3).mean(axis=(1, 3)).var(axis=(0, 1)))
    variance = actual.var(axis=(0, 1))
    assert np.all(variance >= np.min(reference, axis=0)), (variance, np.min(reference, axis=0))
    assert np.all(variance <= np.max(reference, axis=0)), (variance, np.max(reference, axis=0))
