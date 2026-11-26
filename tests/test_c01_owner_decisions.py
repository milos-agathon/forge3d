"""Owner-approved family presentations, with physical hardware rendering."""
from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
import forge3d as f3d
from forge3d import map_scene as ms
from test_c01_review_gpu import _scene


def _material(presentation):
    return f3d.GeospatialMaterialRecipe.land_cover(
        f3d.CategoricalRasterStyle({1: (40, 60, 20, 255)}, {1: "protected class"}),
        albedo=(150, 140, 130)) if presentation == "protected" else f3d.GeospatialMaterialRecipe(
            (f3d.MaterialLayer("paint", (150, 100, 50)),), presentation="physical")


def _feature(identifier, x0, x1, *, height=25):
    ring = [[x0, 4599980], [x1, 4599980], [x1, 4599956], [x0, 4599956], [x0, 4599980]]
    return {"type": "Feature", "id": identifier, "properties": {"height": height},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_c01_triplanar_projection_follows_steep_surface_geometry():
    """The production sampler's diagnostic weights must use all three axes."""
    import copy
    from pathlib import Path
    rows, cols = np.mgrid[:128, :128]
    heights = (55 * np.sin(cols * 2 / 20) + 55 * np.cos(rows * 2 / 20)).astype("float32")
    grid, recipe = _scene()
    grid = replace(grid, spacing=(2, 2))
    recipe = replace(recipe, terrain=replace(recipe.terrain, data=heights), layers=())
    recipe = _material("physical").bind(recipe, grid)
    recipe = replace(recipe, camera=replace(recipe.camera, distance=435.2,
        elevation_deg=65, azimuth_deg=140), map_furniture=None,
        terrain=replace(recipe.terrain, metadata={**recipe.terrain.metadata,
            "material_maps": {"physical_normals": True}}))
    config = copy.copy(ms._build_mapscene_terrain_params(recipe, heights, (256, 256)).python_object)
    config.triplanar = replace(config.triplanar, normal_strength=1)
    config.aov = replace(config.aov, enabled=True, depth=True)
    # Mode 21 reports the weights used by the real albedo sampler. This is a
    # diagnostic render only; every MapScene product render keeps debug_mode=0.
    config.debug_mode = 21
    hdr, temporary = ms._native_ibl_path(recipe)
    try:
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        frame, aov = renderer.render_with_aov(f3d.MaterialSet.terrain_default(),
            f3d.IBL.from_hdr(hdr, intensity=1), f3d.TerrainRenderParams(config), heights)
        pixels = np.asarray(frame.to_numpy())
        terrain, _ = aov._visibility_depth()
        visible = terrain < 1
        visible[:8, :8] = False
        dominant = np.argmax(pixels[..., :3], axis=-1)
        counts = [int(np.count_nonzero(visible & (dominant == axis))) for axis in range(3)]
        assert all(counts), f"Steep C01 geometry must exercise X/Y/Z triplanar projections: {counts}"
    finally:
        if temporary:
            Path(hdr).unlink()


@pytest.mark.parametrize("span_scale", [1.0, 2.0])
def test_c01_building_scatter_uses_world_axes_and_declared_height(span_scale):
    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    base_span = ms._terrain_frame(recipe.terrain)[2]
    if span_scale != 1.0:
        recipe = replace(recipe, terrain=replace(recipe.terrain, metadata={
            **recipe.terrain.metadata, "terrain_span": float(base_span) * span_scale,
        }))
    layer = _material("protected").bind_buildings(
        f3d.MapSceneBuildingLayer("block", metadata={"crs": grid.crs},
            features=[_feature("block", 500015, 500032, height=25)]),
        [[500020, 4599970]], units="m", classes=[1], presentation="protected")
    heights = ms._load_native_heightmap(recipe.terrain)
    batches, _ = ms._terrain_scatter_building_batches_for_recipe(
        replace(recipe, layers=(layer,)), heights)
    batch = batches[0]
    # Native pack_instance_transforms maps translation through the contract,
    # then uniformly scales the mesh's already-world-oriented local axes.
    scale = max(1.0, ms._terrain_frame(recipe.terrain)[2] or ms._terrain_scene_diagonal(recipe.terrain)) / max(heights.shape)
    local = np.asarray(batch["levels"][0]["mesh"]["positions"]) * scale
    np.testing.assert_allclose(
        np.ptp(local[:, :2], axis=0), (17 * span_scale, 24 * span_scale), atol=1e-4
    )
    np.testing.assert_allclose(np.ptp(local[:, 2]), 25, atol=1e-4)
    assert np.min(local[:, 2]) == 0, "The building bottom must sit on sampled terrain"


@pytest.mark.parametrize("offline", (False, True), ids=("preview", "final"))
def test_protected_buildings_use_native_visible_batch_identity(tmp_path, monkeypatch, offline):
    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    protected = _material("protected").bind_buildings(
        f3d.MapSceneBuildingLayer("protected", metadata={"crs": grid.crs},
            features=[_feature("class-building", 500015, 500032)]),
        [[500020, 4599970]], units="m", classes=[1], presentation="protected")
    physical = _material("physical").bind_buildings(
        f3d.MapSceneBuildingLayer("physical", metadata={"crs": grid.crs},
            features=[_feature("lit-building", 500032, 500049)]),
        [[500040, 4599970]], units="m")
    recipe = replace(recipe, layers=(protected, physical), map_furniture=None,
        camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140),
        output=replace(recipe.output, samples=4 if offline else 1, hdr=offline,
                       denoiser="atrous" if offline else "none"))
    captures = []
    exr_arrays = []
    denoise_arrays = []
    if offline:
        from forge3d import offline as offline_module
        original_denoise = offline_module.atrous_denoise
        def observe_denoise(color, **kwargs):
            result = original_denoise(color, **kwargs)
            denoise_arrays.append((color.copy(), result.copy(), kwargs["normal"].copy()))
            return result
        monkeypatch.setattr(offline_module, "atrous_denoise", observe_denoise)
    original = ms._render_terrain_renderer_result
    original_exr = ms._write_exr_array
    def observe_exr(path, array, **kwargs):
        exr_arrays.append(np.asarray(array).copy())
        return original_exr(path, array, **kwargs)
    monkeypatch.setattr(ms, "_write_exr_array", observe_exr)
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append((np.asarray(result.aov_frame._building_ids()), result.metadata))
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    frames=[]
    with ms._shared_terrain_render_context():
        for index, sun in enumerate(((.8, .6, 0), (-.8, .6, 0))):
            scene=f3d.MapScene(recipe=replace(recipe, lighting=replace(recipe.lighting, sun_direction=sun)))
            scene.render(tmp_path/f"sun-{index}.png")
            frames.append(np.asarray(Image.open(tmp_path/f"sun-{index}.png")))
    ids, metadata = captures[0]
    class_pixels = ids == metadata["building_batch_ids"]["class-building"] + 1
    lit_pixels = ids == metadata["building_batch_ids"]["lit-building"] + 1
    assert class_pixels.any() and lit_pixels.any()
    for frame in frames:
        np.testing.assert_array_equal(frame[class_pixels], np.tile((40,60,20,255),(class_pixels.sum(),1)))
    if offline:
        from forge3d.terrain_materials import _linear
        expected = np.r_[_linear(np.array([40, 60, 20], np.float32)/255), np.float32(1)]
        assert len(exr_arrays) == 2
        for array in exr_arrays:
            np.testing.assert_array_equal(array[class_pixels], np.tile(expected, (class_pixels.sum(), 1)))
        for before, after, normal in denoise_arrays:
            # Scatter has no valid normal/albedo guide in the offline terrain
            # AOVs. Do not filter its accumulated colour using the ground below.
            np.testing.assert_array_equal(normal[lit_pixels], 0)
            np.testing.assert_array_equal(after[lit_pixels], before[lit_pixels])
    assert np.any(frames[0][lit_pixels] != frames[1][lit_pixels])
    assert not np.any(class_pixels & lit_pixels)


def test_physical_only_buildings_denoise_without_protected_bindings(tmp_path):
    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    building = _material("physical").bind_buildings(
        f3d.MapSceneBuildingLayer("lit", metadata={"crs": grid.crs},
            features=[_feature("lit", 500015, 500032)]),
        [[500020, 4599970]], units="m")
    scene = f3d.MapScene(recipe=replace(recipe, layers=(building,), map_furniture=None,
        camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140),
        output=replace(recipe.output, samples=4, hdr=True, denoiser="atrous")))
    scene.render(tmp_path/"physical-only.png")
    assert scene.last_render_backend == "gpu_terrain"
    assert scene.last_render_metadata["denoiser_used"] == "atrous"
    assert scene.last_render_metadata["denoiser_guide_coverage"] == (
        'terrain normals; accumulated scatter, sky and their sample-footprint terrain margins pass through without filtering')


def test_physical_vector_colour_is_lit_terrain_albedo(tmp_path):
    grid, recipe = _scene()
    ground=f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("ground",(150,140,130)),),presentation="physical")
    recipe = ground.bind(recipe, grid)
    vector = _material("physical").bind_vector(
        f3d.VectorOverlay("paint", crs=grid.crs, width_world=2,
            features=[_feature("paint", 500005, 500050)]),
        [[500020,4599970]], units="m")
    recipe = replace(recipe, layers=(vector,), map_furniture=None,
        camera=replace(recipe.camera,distance=110,elevation_deg=40,azimuth_deg=140))
    frames=[]
    controls=[]
    with ms._shared_terrain_render_context():
        for index,sun in enumerate(((0,1,0),(.8,.2,0))):
            scene=f3d.MapScene(recipe=replace(recipe,lighting=replace(recipe.lighting,sun_direction=sun)))
            scene.render(tmp_path/f"lit-{index}.png")
            frames.append(np.asarray(Image.open(tmp_path/f"lit-{index}.png")))
            assert scene.last_render_metadata["physical_vector_backend"] == "native_terrain_uv_albedo"
            control=f3d.MapScene(recipe=replace(scene.recipe,layers=()))
            control.render(tmp_path/f"unpainted-{index}.png")
            controls.append(np.asarray(Image.open(tmp_path/f"unpainted-{index}.png")))
    painted=np.any(frames[0] != controls[0],axis=-1)
    assert painted.any(), "The declared vector must actually change surface albedo"
    assert np.any(frames[0][painted] != frames[1][painted])
    assert not np.any(np.all(frames[0] == (150,100,50,255),axis=-1))


def test_protected_polygon_owns_pixel_centres_without_blended_edge_colours(tmp_path, monkeypatch):
    from forge3d._map_scene_projection import frame_visibility
    from forge3d._map_scene_render import _ring_contains
    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    feature = _feature("class", 500015, 500032)
    layer = _material("protected").bind_vector(
        f3d.VectorOverlay("class", crs=grid.crs, features=[feature]),
        [[500020, 4599970]], units="m", classes=[1])
    recipe = replace(recipe, layers=(layer,), map_furniture=None,
        camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140))
    visibility = []
    original = ms._render_terrain_renderer_result
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        visibility.append(frame_visibility(args[0], result.aov_frame))
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    scene = f3d.MapScene(recipe=recipe)
    scene.render(tmp_path/"exact-polygon.png")
    uv, terrain = visibility[0][1:]
    from forge3d._map_scene_projection import TerrainProjector
    projector = TerrainProjector(scene.recipe)
    ring = [projector.uv(p) for p in feature["geometry"]["coordinates"][0]]
    inside = terrain & _ring_contains(ring, uv[..., 0], uv[..., 1])
    assert inside.any()
    rgba = np.asarray(Image.open(tmp_path/"exact-polygon.png"))
    np.testing.assert_array_equal(rgba[inside], np.tile((40, 60, 20, 255), (inside.sum(), 1)))


def test_protected_stroke_has_exact_colour_and_does_not_paint_the_background(tmp_path):
    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    feature = {"type": "Feature", "id": "line", "properties": {},
        "geometry": {"type": "LineString", "coordinates": [[500010, 4599970], [500045, 4599960]]}}
    layer = _material("protected").bind_vector(
        f3d.VectorOverlay("line", crs=grid.crs, width_px=3, features=[feature]),
        [[500020, 4599970]], units="m", classes=[1])
    recipe = replace(recipe, map_furniture=None,
        camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140))
    with ms._shared_terrain_render_context():
        f3d.MapScene(recipe=recipe).render(tmp_path/"line-control.png")
        f3d.MapScene(recipe=replace(recipe, layers=(layer,))).render(tmp_path/"line-exact.png")
    before = np.asarray(Image.open(tmp_path/"line-control.png"))
    after = np.asarray(Image.open(tmp_path/"line-exact.png"))
    changed = np.any(before != after, axis=-1)
    assert changed.any() and not changed[0, 0]
    np.testing.assert_array_equal(after[changed], np.tile((40, 60, 20, 255), (changed.sum(), 1)))


def test_feature_report_update_preserves_values_and_snapshot_isolation():
    from forge3d.diagnostics import LayerSummary, ValidationReport
    grid, recipe = _scene()
    scene = f3d.MapScene(recipe=recipe)
    original = ValidationReport(layer_summaries=(LayerSummary("r", "raster", "supported",
        details={"classes": [[1, 2], [3, 4]]}),))
    expected = original.to_dict()
    expected["supported_features"]["vector"] = "supported"
    updated = scene._report_with_feature(original, "vector", "supported")
    assert updated.to_dict() == expected
    updated.layer_summaries[0].details["classes"][0][0] = 9
    assert original.layer_summaries[0].details["classes"][0][0] == 1


def test_c01_affine_and_bounds_use_the_same_pixel_area_geometry():
    from forge3d._map_scene_projection import TerrainProjector
    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    affine = (grid.origin[0]-.5, 1, 0, grid.origin[1]+.5, 0, -1)
    affine_recipe = replace(recipe, terrain=replace(recipe.terrain,
        metadata={**recipe.terrain.metadata, "transform": affine}))
    point = (500020, 4599970)
    np.testing.assert_allclose(TerrainProjector(recipe).uv(point),
                               TerrainProjector(affine_recipe).uv(point), atol=0, rtol=0)
    layer = _material("protected").bind_buildings(
        f3d.MapSceneBuildingLayer("block", metadata={"crs": grid.crs},
            features=[_feature("block", 500015, 500032)]),
        [point], units="m", classes=[1], presentation="protected")
    batches = [ms._terrain_scatter_building_batches_for_recipe(
        replace(candidate, layers=(layer,)), ms._load_native_heightmap(candidate.terrain))[0][0]
        for candidate in (recipe, affine_recipe)]
    for key in ("positions", "normals", "indices"):
        np.testing.assert_array_equal(batches[0]["levels"][0]["mesh"][key],
                                      batches[1]["levels"][0]["mesh"][key])
    np.testing.assert_array_equal(batches[0]["transforms"], batches[1]["transforms"])


def test_c01_materials_preserve_registered_screen_projection():
    from forge3d._map_scene_projection import TerrainProjector, project_vector_recipe

    grid, recipe = _scene()
    recipe = _material("physical").bind(recipe, grid)
    east, north = grid.origin
    metadata = {
        **recipe.terrain.metadata,
        "registration": "point",
        "transform": (east - 0.5, 1, 0, north + 0.5, 0, -1),
    }
    point = (east + 21, north - 42)
    vector = f3d.VectorOverlay(
        "station",
        crs=grid.crs,
        features=[{"type": "Feature", "properties": {},
                   "geometry": {"type": "Point", "coordinates": point}}],
    )
    protected = _material("protected").bind_vector(
        vector, [point], units="m", classes=[1],
    )
    recipe = replace(
        recipe, terrain=replace(recipe.terrain, metadata=metadata),
        lighting=replace(recipe.lighting, settings={"cli_params": {"camera_mode": "screen"}}),
        layers=(protected,),
    )

    projector = TerrainProjector(recipe)
    np.testing.assert_allclose(projector.uv(point), (1 / 3, 2 / 3))
    rendered = project_vector_recipe(recipe, projector=projector)
    coordinates = rendered.layers[0].features[0]["geometry"]["coordinates"]
    np.testing.assert_allclose(coordinates[:2], (128 / 3, 256 / 3), atol=1e-4)


def test_shared_surface_kind_excludes_scatter_at_equal_depth():
    from forge3d._map_scene_projection import frame_visibility

    _, recipe = _scene()
    recipe = replace(recipe, lighting=replace(recipe.lighting,
        settings={"cli_params": {"camera_mode": "screen"}}))
    shape = (recipe.output.height, recipe.output.width)
    depth = np.full(shape, 0.5, dtype=np.float32)
    kind = np.ones(shape, dtype=np.uint8)
    kind[:, shape[1] // 2:] = 2

    class SharedVisibility:
        def _visibility_depth(self):
            return depth, depth

        def _interactive_surface_kind(self):
            return kind

    _, uv, terrain = frame_visibility(recipe, SharedVisibility())
    np.testing.assert_array_equal(terrain, kind == 1)
    assert np.isnan(uv[kind == 2]).all()


@pytest.mark.parametrize("effect", ("clouds", "screen_space", "water"))
def test_surface_effects_change_lit_pixels_but_preserve_protected_classes(tmp_path, monkeypatch, effect):
    from forge3d._map_scene_projection import frame_visibility
    grid, recipe = _scene()
    yy, xx = np.mgrid[:64, :64]
    recipe = replace(recipe, terrain=replace(recipe.terrain, data=(xx*.5+yy*.8).astype(np.float32)),
        camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140))
    material = f3d.GeospatialMaterialRecipe.land_cover(f3d.CategoricalRasterStyle(
        {1: (40, 60, 20, 255)}, {1: "Protected"}, nodata=0), albedo=(150, 140, 130))
    recipe = material.bind(recipe, grid, classes=np.where(xx>=32, 1, 0))
    settings = ({"clouds": {"enabled": True, "shadows_enabled": True, "coverage": .72,
        "density": .48, "shadow_strength": .38, "quality": "high"}} if effect=="clouds" else
        {"screen_space": {"ssao": {"enabled": True, "radius": 2.6, "intensity": 1.35},
                          "ssgi": {"enabled": True, "intensity": .45},
                          "ssr": {"enabled": True, "intensity": .5}}})
    if effect == "water":
        settings = {"water": {"enabled": True, "auto_mask": True, "level": 32., "slope_threshold": 1.}}
    captures = []
    original = ms._render_terrain_renderer_result
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append(frame_visibility(args[0], result.aov_frame))
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    frames = []
    with ms._shared_terrain_render_context():
        for index, extras in enumerate(({}, settings)):
            scene = f3d.MapScene(recipe=replace(recipe, map_furniture=None,
                lighting=replace(recipe.lighting, settings={**recipe.lighting.settings, **extras})))
            scene.render(tmp_path/f"effect-{index}.png")
            frames.append(np.asarray(Image.open(tmp_path/f"effect-{index}.png")))
    _, uv, terrain = captures[-1]
    protected = terrain & (uv[..., 0]>=.5)
    lit = terrain & ~protected
    assert protected.any() and lit.any()
    np.testing.assert_array_equal(frames[-1][protected], np.tile((40, 60, 20, 255), (protected.sum(), 1)))
    assert np.any(frames[0][lit] != frames[1][lit])


@pytest.mark.parametrize("binding", ("surface", "vector", "building"))
def test_fullscreen_colour_uses_native_raster_centres_and_overlay_identity(tmp_path, monkeypatch, binding):
    grid, recipe = _scene()
    recipe = replace(recipe, map_furniture=None,
        camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140),
        lighting=replace(recipe.lighting, settings={"cli_params": {"camera_mode": "screen"}}))
    palette = f3d.CategoricalRasterStyle({1: (40, 60, 20, 255), 2: (110, 40, 80, 255)},
        {1: "Left", 2: "Right"})
    material = f3d.GeospatialMaterialRecipe.land_cover(palette, albedo=(150, 140, 130))
    protected = material.bind(recipe, grid, classes=np.where(np.indices((64, 64))[1]<32, 1, 2))
    if binding == "vector":
        feature = _feature("v", 500015, 500032)
        vector = _material("protected").bind_vector(
            f3d.VectorOverlay("v", crs=grid.crs, features=[feature]),
            [[500020, 4599970]], units="m", classes=[1])
        protected = replace(_material("physical").bind(recipe, grid), layers=(vector,))
    elif binding == "building":
        building = _material("physical").bind_buildings(
            f3d.MapSceneBuildingLayer("lit", metadata={"crs": grid.crs},
                features=[_feature("lit", 500015, 500032)]),
            [[500020, 4599970]], units="m")
        protected = replace(protected, layers=(building,))
    captures = []
    original = ms._render_terrain_renderer_result
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append(result)
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    scene = f3d.MapScene(recipe=replace(protected, map_furniture=None))
    scene.render(tmp_path/"fullscreen.png")
    pixels = np.asarray(Image.open(tmp_path/"fullscreen.png"))
    if binding == "vector":
        # Geographic polygon bounds map to native fullscreen pixel centres.
        np.testing.assert_array_equal(pixels[41:89, 31:65],
            np.tile((40, 60, 20, 255), (48, 34, 1)))
    else:
        ids = (np.asarray(captures[-1].aov_frame._building_ids()) if binding=="building"
               else np.zeros(pixels.shape[:2], np.uint32))
        for half, color in ((np.indices(ids.shape)[1]<64, palette.palette[1]),
                            (np.indices(ids.shape)[1]>=64, palette.palette[2])):
            visible = half & (ids==0)
            assert visible.any()
            np.testing.assert_array_equal(pixels[visible], np.tile(color, (visible.sum(), 1)))
        if binding == "building":
            assert np.any(ids>0)
            plain = _material("physical").bind(recipe, grid)
            f3d.MapScene(recipe=replace(plain, layers=(building,), map_furniture=None)).render(tmp_path/"lit-control.png")
            control = np.asarray(Image.open(tmp_path/"lit-control.png"))
            np.testing.assert_array_equal(pixels[ids>0], control[ids>0])


def test_plain_terrain_atrous_preserves_sky_samples4():
    recipe=f3d.SceneRecipe(
        f3d.TerrainSource(data=np.zeros((64,64),np.float32),crs="EPSG:32632",
                          metadata={"source_id":"plain-sky-regression"}),
        f3d.OrbitCamera(distance=110,elevation_deg=40,azimuth_deg=140),
        f3d.LightingPreset("custom",sun_direction=(.8,.6,0),
                           settings={"cli_params":{"camera_mode":"mesh:zup:north"}}),
        output=f3d.OutputSpec(width=128,height=128,samples=4,aovs=("normal",)))
    with ms._shared_terrain_render_context():
        ordinary=ms._render_terrain_renderer_result(recipe,recipe.terrain.data)
        filtered=ms._render_terrain_renderer_result(
            replace(recipe,output=replace(recipe.output,denoiser="atrous")),recipe.terrain.data)
    sky=np.linalg.norm(np.asarray(filtered.aov_frame.normal()),axis=-1)==0
    assert sky.any() and np.any(ordinary.rgba[sky,:3]>0)
    np.testing.assert_array_equal(filtered.rgba[sky],ordinary.rgba[sky])


def test_physical_vector_grid_coverage_and_linear_alpha():
    from forge3d._map_scene_materials import drape_physical_vectors
    from forge3d.terrain_materials import _linear, _display
    grid, recipe=_scene()
    ground=_material("physical")
    recipe=ground.bind(recipe,grid)
    vector=ground.bind_vector(f3d.VectorOverlay("half-opacity",crs=grid.crs,
        features=[_feature("paint",500005,500050)],
        style={"layers":[{"type":"fill","paint":{"fill-opacity":.5}}]}),
        [[500020,4599970]],units="m")
    # The ground and paint deliberately differ, so coverage cannot pass as a no-op.
    from forge3d.raster_style import RasterStyleResult
    background=np.full((64,64,4),(30,60,90,255),np.uint8)
    base=RasterStyleResult(background,np.ones((64,64),bool),{})
    result=drape_physical_vectors(replace(recipe,layers=(vector,)),(64,64),base)
    expected=_display((_linear(np.array((150,100,50))/255)+_linear(np.array((30,60,90))/255))/2)
    np.testing.assert_array_equal(result.rgba[30,20,:3],expected)
    np.testing.assert_array_equal(result.rgba[0,0],background[0,0])


def test_nearer_building_owns_identity_and_hides_lower_instance():
    grid, recipe=_scene()
    recipe=_material("physical").bind(recipe,grid)
    palette=f3d.CategoricalRasterStyle({1:(40,60,20,255),2:(220,80,40,255)},
                                     {1:"tall front instance",2:"occluded low instance"})
    material=f3d.GeospatialMaterialRecipe.land_cover(palette,albedo=(150,140,130))
    covered=_feature("covered",500024,500040,height=10)
    # Strictly interior geometry tests occlusion without coplanar wall ties.
    covered["geometry"]["coordinates"][0]=[
        [500024,4599976],[500040,4599976],[500040,4599960],
        [500024,4599960],[500024,4599976]]
    layer=material.bind_buildings(f3d.MapSceneBuildingLayer("overlap",metadata={"crs":grid.crs},
        features=[_feature("front",500020,500044,height=35),
                  covered]),
        [[500032,4599970],[500032,4599970]],units="m",classes=[1,2],presentation="protected")
    recipe=replace(recipe,layers=(layer,),map_furniture=None,
        camera=replace(recipe.camera,distance=110,elevation_deg=40,azimuth_deg=140))
    with ms._shared_terrain_render_context():
        frame=ms._render_terrain_renderer_result(recipe,recipe.terrain.data)
    ids=np.asarray(frame.aov_frame._building_ids())
    assert np.any(ids==frame.metadata["building_batch_ids"]["front"]+1)
    assert not np.any(ids==frame.metadata["building_batch_ids"]["covered"]+1)
    terrain,scene=frame.aov_frame._visibility_depth()
    assert np.all(scene[ids>0] < terrain[ids>0])
    assert np.all(ids[scene==1]==0)


def test_physical_and_protected_rasters_share_one_lit_scene(tmp_path,monkeypatch):
    grid,recipe=_scene()
    recipe=f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("ground",(150,140,130)),),
                                       presentation="physical").bind(recipe,grid)
    rows,cols=np.mgrid[:64,:64]
    left=np.where(cols<32,1.,np.nan)
    right=np.where(cols>=32,1.,np.nan)
    recipe=_material("physical").bind_raster(recipe,left,grid,units="class",layer_id="lit-raster")
    recipe=_material("protected").bind_raster(recipe,right,grid,units="class",
        classes=np.where(np.isfinite(right),1,np.nan),layer_id="exact-raster")
    recipe=replace(recipe,map_furniture=None,
        camera=replace(recipe.camera,distance=110,elevation_deg=40,azimuth_deg=140))
    captures=[]
    from forge3d._map_scene_projection import frame_visibility
    original=ms._render_terrain_renderer_result
    def observe(recipe,*args,**kwargs):
        result=original(recipe,*args,**kwargs)
        captures.append(frame_visibility(recipe,result.aov_frame))
        return result
    monkeypatch.setattr(ms,"_render_terrain_renderer_result",observe)
    frames=[]
    with ms._shared_terrain_render_context():
        for index,sun in enumerate(((.8,.6,0),(.2,.98,0))):
            scene=f3d.MapScene(recipe=replace(recipe,lighting=replace(recipe.lighting,sun_direction=sun)))
            scene.render(tmp_path/f"rasters-{index}.png")
            frames.append(np.asarray(Image.open(tmp_path/f"rasters-{index}.png")))
    _,uv,terrain=captures[0]
    protected=terrain & (uv[...,0]>=.5)
    physical=terrain & (uv[...,0]<.5)
    assert protected.any() and physical.any()
    for frame in frames:
        np.testing.assert_array_equal(frame[protected],np.tile((40,60,20,255),(protected.sum(),1)))
    assert np.any(frames[0][physical] != frames[1][physical])


@pytest.mark.parametrize("mode",("mesh:zup:north","clipmap:2:128:128:10:0.3:zup:north"))
def test_ridge_occludes_building_identity_but_elevated_building_survives(mode):
    grid,recipe=_scene()
    rows,cols=np.mgrid[:128,:128]
    heights=(80*np.exp(-((rows-64)/5)**2)).astype(np.float32)
    recipe=replace(recipe,terrain=replace(recipe.terrain,data=heights),
        camera=f3d.OrbitCamera(target=(0,0,0),distance=260,azimuth_deg=-90,elevation_deg=70,fov_deg=40),
        lighting=replace(recipe.lighting,settings={"cli_params":{"camera_mode":mode}}))
    recipe=_material("physical").bind(recipe,grid)
    ring=[[500060,4599950],[500068,4599950],[500068,4599944],
          [500060,4599944],[500060,4599950]]
    counts=[]
    with ms._shared_terrain_render_context():
        for height in (10,120):
            feature={"type":"Feature","id":"ridge-building","properties":{"height":height},
                     "geometry":{"type":"Polygon","coordinates":[ring]}}
            layer=_material("protected").bind_buildings(
                f3d.MapSceneBuildingLayer("ridge-building",features=[feature],metadata={"crs":grid.crs}),
                [[500064,4599947]],units="m",classes=[1],presentation="protected")
            frame=ms._render_terrain_renderer_result(replace(recipe,layers=(layer,)),heights)
            ids=np.asarray(frame.aov_frame._building_ids())
            counts.append(int(np.count_nonzero(ids)))
    assert counts[0]==0, "The ridge must hide the lower building"
    assert counts[1]>0, "The elevated building must remain visible, including against sky"
