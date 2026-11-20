"""Independent HDR-radiance and offline-scatter regressions from C01 R1-R4."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

import forge3d as f3d
from forge3d import map_scene as ms


CLOUDS = {"enabled": True, "shadows_enabled": True, "coverage": .72,
          "density": .48, "shadow_strength": .38, "quality": "high"}


def test_mapscene_accepts_native_triplanar_repeat_for_motion_fixture():
    from test_c01_review_gpu import _scene
    _, recipe = _scene()
    recipe = replace(recipe, lighting=replace(recipe.lighting,
        settings={**recipe.lighting.settings,
                  "triplanar": {"scale": 1/32, "blend_sharpness": 4, "normal_strength": 1}}))
    config = ms._build_mapscene_terrain_params(recipe, recipe.terrain.data, (128, 128)).python_object
    assert config.triplanar.scale == 1/32


def _retain(name, **arrays):
    if os.environ.get("FORGE3D_C01_ARTIFACT_DIR"):
        out = Path(os.environ["FORGE3D_C01_ARTIFACT_DIR"]) / name
        out.mkdir(parents=True, exist_ok=True)
        for key, value in arrays.items():
            np.save(out / (key + ".npy"), value)
            if value.dtype == np.uint8 and value.ndim == 3:
                Image.fromarray(value).save(out / (key + ".png"))


@pytest.mark.parametrize("protected", (False, True), ids=("legacy-clouds", "protected-and-lit-clouds"))
def test_cloud_hdr_keeps_radiance_outside_exact_paint(tmp_path, monkeypatch, protected):
    captured = {}
    render = ms._render_native_offscreen_result
    writer = ms._write_exr_array
    save = f3d.HdrFrame.save

    def observe(*args, **kwargs):
        result = render(*args, **kwargs)
        captured["hdr"] = np.asarray(result.hdr_frame.to_numpy_f32()).copy()
        captured["pixels"] = result.rgba.copy()
        captured["mask"] = result.hdr_overlay_mask.copy()
        return result

    def write(path, array, **kwargs):
        captured["exported"] = np.asarray(array).copy()
        return writer(path, array, **kwargs)

    def save_hdr(self, path):
        captured["exported"] = np.asarray(self.to_numpy_f32()).copy()
        return save(self, path)

    monkeypatch.setattr(ms, "_render_native_offscreen_result", observe)
    monkeypatch.setattr(ms, "_write_exr_array", write)
    monkeypatch.setattr(f3d.HdrFrame, "save", save_hdr)
    if protected:
        from test_c01_review_gpu import _scene
        grid, recipe = _scene()
        yy, xx = np.mgrid[:64, :64]
        recipe = replace(recipe, terrain=replace(recipe.terrain, data=(xx*.5+yy*.8).astype(np.float32)),
                         camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140))
        material = f3d.GeospatialMaterialRecipe.land_cover(f3d.CategoricalRasterStyle(
            {1: (40, 60, 20, 255)}, {1: "Protected"}, nodata=0), albedo=(150, 140, 130))
        recipe = material.bind(recipe, grid, classes=np.where(xx >= 32, 1, 0))
        recipe = replace(recipe, map_furniture=None, output=replace(recipe.output, hdr=True),
                         lighting=replace(recipe.lighting, settings={**recipe.lighting.settings, "clouds": CLOUDS}))
        scene = f3d.MapScene(recipe=recipe)
    else:
        from test_recipe_goldens import _base_scene
        dem = np.zeros((16, 16), np.float32)
        dem[5:11, 5:11] = .35
        scene = _base_scene(tmp_path, "clouds", width=128, height=80, heightmap=dem, hdr=True,
                            terrain_metadata={"width": 16, "height": 16, "source_id": "clouds", "clouds": CLOUDS})
    scene.render(tmp_path / "clouds.png")
    hdr, exported, pixels = (captured[k] for k in ("hdr", "exported", "pixels"))
    exact = np.all(pixels == (40, 60, 20, 255), axis=-1) if protected else np.zeros(hdr.shape[:2], bool)
    assert (~exact).any()
    _retain("hdr-" + str(protected), hdr=hdr, exported=exported, pixels=pixels, exact=exact)
    np.testing.assert_array_equal(exported[~exact], hdr[~exact])
    np.testing.assert_array_equal(captured["mask"], exact)
    if protected:
        from forge3d.terrain_materials import _linear
        assert exact.any()
        expected = _linear(np.array((40, 60, 20), np.float32)/255)
        np.testing.assert_array_equal(exported[exact, :3], np.tile(expected, (exact.sum(), 1)))
        assert "physical pixels retain scene radiance" in scene.last_render_metadata["hdr_overlay_encoding"]
    else:
        assert "hdr_overlay_encoding" not in scene.last_render_metadata
    assert (tmp_path / "clouds.exr").is_file()


def _sample_footprint(mask, render_scale):
    """Independent Pillow maximum-filter oracle; square output-sample support."""
    radius = max(1, int(np.ceil(1/render_scale)))
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).filter(
        ImageFilter.MaxFilter(2*radius+1)), dtype=bool)


@pytest.mark.parametrize("route,camera_mode,msaa,render_scale",
    [("mapscene", mode, 1, 1.0) for mode in ("mesh", "screen")]
    + [("direct", mode, msaa, scale) for mode in ("mesh", "screen")
       for msaa, scale in ((1,1.0),(1,.5),(4,1.0),(4,.5))])
def test_legacy_buildings_samples4_atrous_preserves_building_pixels(tmp_path, monkeypatch, route, camera_mode, msaa, render_scale):
    from test_recipe_goldens import _buildings
    recipe = (_visibility_fixture(buildings=True, protected=False)[1]
              if camera_mode == "mesh" else _buildings(tmp_path).recipe)
    recipe = replace(recipe, output=replace(recipe.output, samples=4),
        lighting=replace(recipe.lighting, settings={**recipe.lighting.settings,
            "cli_params": {**recipe.lighting.settings.get("cli_params", {}),
                           "camera_mode": camera_mode}}))
    heightmap = ms._load_native_heightmap(recipe.terrain)
    resources = ms._TerrainRenderResources(f3d) if route == "direct" else None

    def render(recipe, heights):
        if route == "mapscene":
            return ms._render_terrain_renderer_result(recipe, heights)
        # Sampling is a direct renderer contract, not a MapScene recipe option.
        config = ms._build_mapscene_terrain_params(recipe, heights,
            (recipe.output.width, recipe.output.height)).python_object
        params = f3d.TerrainRenderParams(replace(config, msaa_samples=msaa, render_scale=render_scale,
            aov=replace(config.aov, enabled=True, albedo=True, normal=True)))
        batches = ms._terrain_scatter_building_batches_for_recipe(recipe, heights)
        resources.renderer.set_scatter_batches(batches[0] if batches is not None else [])
        hdr_path, delete_hdr = ms._native_ibl_path(recipe)
        try:
            env_maps = resources.env_maps(hdr_path)
        finally:
            if delete_hdr:
                os.unlink(hdr_path)
        result = f3d.render_offline(resources.renderer, resources.material_set, env_maps,
            params, heights, settings=ms._mapscene_offline_settings(recipe.output))
        return ms._MapSceneNativeRenderResult(rgba=ms._frame_to_rgba(result.frame, recipe.output),
            hdr_frame=result.hdr_frame, aov_frame=result.aov_frame,
            metadata={**(batches[1] if batches is not None else {}), **result.metadata})
    from forge3d import offline
    references = []
    original = offline._scatter_terrain_reference
    def capture(*args, **kwargs):
        mask, reference = original(*args, **kwargs)
        references.append((mask, reference, args[3].python_object))
        return mask, reference
    monkeypatch.setattr(offline, "_scatter_terrain_reference", capture)
    with ms._shared_terrain_render_context():
        ordinary = render(recipe, heightmap)
        filtered = render(replace(recipe,
            output=replace(recipe.output, denoiser="atrous")), heightmap)
        terrain_only = render(replace(recipe,
            layers=tuple(layer for layer in recipe.layers if not isinstance(layer, f3d.MapSceneBuildingLayer))), heightmap)
    guide, reference, used_config = references[-1]
    before_hdr = np.asarray(ordinary.hdr_frame.to_numpy_f32()).copy()
    after_hdr = np.asarray(filtered.hdr_frame.to_numpy_f32()).copy()
    clean_hdr = np.asarray(terrain_only.hdr_frame.to_numpy_f32()).copy()
    # Keep cast-shadow changes out of the silhouette oracle: terrain shadows
    # legitimately change when geometry is removed and remain terrain samples.
    settings = recipe.lighting.settings
    renderer_config = settings.get("renderer_config", {})
    oracle_recipe = replace(recipe, lighting=replace(recipe.lighting, settings={**settings,
        "renderer_config": {**renderer_config, "shadows": {**renderer_config.get("shadows", {}), "enabled": False}}}))
    with ms._shared_terrain_render_context():
        oracle_on = render(oracle_recipe, heightmap)
        oracle_off = render(replace(oracle_recipe,
            layers=tuple(layer for layer in recipe.layers if not isinstance(layer, f3d.MapSceneBuildingLayer))), heightmap)
    oracle_on_hdr = np.asarray(oracle_on.hdr_frame.to_numpy_f32()).copy()
    oracle_off_hdr = np.asarray(oracle_off.hdr_frame.to_numpy_f32()).copy()
    contribution = np.any(oracle_on_hdr[...,:3] != oracle_off_hdr[...,:3], axis=-1)
    building = np.asarray(reference._building_ids()) > 0
    terrain_depth, scene_depth = reference._visibility_depth()
    native_terrain = (terrain_depth < 1) & (terrain_depth == scene_depth) & ~building
    expanded = _sample_footprint(building, render_scale)
    assert building.any()
    assert filtered.metadata["building_backend"] == "terrain_scatter_instanced_mesh"
    assert filtered.metadata["denoiser_used"] == "atrous"
    prefix = "legacy-buildings" if route == "direct" else "mapscene-buildings"
    _retain(f"{prefix}-{camera_mode}-msaa{msaa}-scale{render_scale}",
        ordinary=ordinary.rgba, filtered=filtered.rgba, building=building, expanded=expanded,
        guide=guide, native_terrain=native_terrain, before_hdr=before_hdr, after_hdr=after_hdr, terrain_only_hdr=clean_hdr,
        scatter_contribution=contribution, guided_leaks=guide & contribution,
        oracle_on_hdr=oracle_on_hdr, oracle_off_hdr=oracle_off_hdr)
    assert contribution.any()
    assert not (guide & contribution).any(), "Building-on/off oracle found buildings in terrain guides"
    np.testing.assert_array_equal(after_hdr[contribution,:3], before_hdr[contribution,:3])
    assert filtered.metadata["offline_effective_msaa_samples"] == 1
    assert filtered.metadata["offline_requested_msaa_samples"] == msaa
    assert used_config.msaa_samples == msaa and used_config.render_scale == render_scale
    assert guide.any(), "The building fixture must retain guided terrain"
    assert np.any(filtered.rgba[guide] != ordinary.rgba[guide]), "The building fixture must actually filter terrain"
    assert not guide[expanded].any(), "Scatter sample footprint must be excluded from terrain guides"
    np.testing.assert_array_equal(filtered.rgba[expanded], ordinary.rgba[expanded])
    np.testing.assert_array_equal(filtered.rgba[building], ordinary.rgba[building])
    assert "scatter" in filtered.metadata["denoiser_guide_coverage"]


@pytest.mark.parametrize("camera_mode", ("mesh", "screen"))
@pytest.mark.parametrize("msaa,render_scale", ((1,1.0),(1,.5),(4,1.0),(4,.5)))
def test_direct_offline_vegetation_atrous_preserves_scatter_pixels(msaa, render_scale, camera_mode, monkeypatch):
    from test_tv22_scatter_wind import _make_gpu_fixtures, _make_scatter_source, _place_instances, _scatter_mesh, ts
    from forge3d.terrain_params import DenoiseSettings
    renderer, materials, ibl, params, heights = _make_gpu_fixtures()
    batch = ts.TerrainScatterBatch(name="vegetation", color=(.22, .48, .24, 1),
        transforms=_place_instances(_make_scatter_source()),
        levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())])
    ts.apply_to_renderer(renderer, [batch])
    config = replace(params.python_object, aa_samples=4, camera_mode=camera_mode,
                     msaa_samples=msaa, render_scale=render_scale)
    config.aov.enabled = True
    config.aov.albedo = True
    config.aov.normal = True
    _, aov = renderer.render_with_aov(materials, ibl, f3d.TerrainRenderParams(
        replace(config, msaa_samples=1, render_scale=1.0,
                aov=replace(config.aov, building_id=True))), heights)
    scatter = np.asarray(aov._building_ids()) > 0
    terrain_depth, scene_depth = aov._visibility_depth()
    native_terrain = (terrain_depth < 1) & (terrain_depth == scene_depth) & ~scatter
    assert scatter.any(), "The direct fixture must draw vegetation"
    expanded = _sample_footprint(scatter, render_scale)
    from forge3d import offline
    normals = []
    original_denoise = offline.atrous_denoise
    def observe_denoise(*args, **kwargs):
        normals.append(kwargs["normal"].copy())
        return original_denoise(*args, **kwargs)
    monkeypatch.setattr(offline, "atrous_denoise", observe_denoise)
    settings = f3d.OfflineQualitySettings(enabled=True)
    ordinary = f3d.render_offline(renderer, materials, ibl, f3d.TerrainRenderParams(config), heights, settings=settings)
    denoised = f3d.render_offline(renderer, materials, ibl, f3d.TerrainRenderParams(replace(config,
        denoise=DenoiseSettings(enabled=True, method="atrous"))), heights, settings=settings)
    before, after = np.asarray(ordinary.frame.to_numpy()), np.asarray(denoised.frame.to_numpy())
    before_hdr = np.asarray(ordinary.hdr_frame.to_numpy_f32()).copy()
    after_hdr = np.asarray(denoised.hdr_frame.to_numpy_f32()).copy()
    renderer.set_scatter_batches([])
    terrain_only = f3d.render_offline(renderer, materials, ibl, f3d.TerrainRenderParams(config), heights, settings=settings)
    clean_hdr = np.asarray(terrain_only.hdr_frame.to_numpy_f32()).copy()
    # Isolate visible scatter samples from their legitimate terrain shadows.
    oracle_config = replace(config, shadows=replace(config.shadows, enabled=False))
    ts.apply_to_renderer(renderer, [batch])
    oracle_on = f3d.render_offline(renderer, materials, ibl, f3d.TerrainRenderParams(oracle_config), heights, settings=settings)
    renderer.set_scatter_batches([])
    oracle_off = f3d.render_offline(renderer, materials, ibl, f3d.TerrainRenderParams(oracle_config), heights, settings=settings)
    oracle_on_hdr = np.asarray(oracle_on.hdr_frame.to_numpy_f32()).copy()
    oracle_off_hdr = np.asarray(oracle_off.hdr_frame.to_numpy_f32()).copy()
    contribution = np.any(oracle_on_hdr[...,:3] != oracle_off_hdr[...,:3], axis=-1)
    guided = np.any(normals[-1] != 0, axis=-1)
    _retain(f"vegetation-{camera_mode}-msaa{msaa}-scale{render_scale}", ordinary=before, filtered=after, scatter=scatter, expanded=expanded,
            guide_normals=normals[-1], native_terrain=native_terrain, before_hdr=before_hdr, after_hdr=after_hdr,
            terrain_only_hdr=clean_hdr, scatter_contribution=contribution, guided_leaks=guided & contribution,
            oracle_on_hdr=oracle_on_hdr, oracle_off_hdr=oracle_off_hdr)
    assert guided.any() and np.any(after[guided] != before[guided]), "Terrain filtering must be exercised"
    assert contribution.any()
    # Independent measured support bound: jitter plus the linear frame resolve
    # reaches three output pixels at scale .5. No wrap to the opposite edge.
    from PIL import Image, ImageFilter
    support_radius = 1 if render_scale == 1 else 3
    support = np.asarray(Image.fromarray(scatter.astype(np.uint8)).filter(
        ImageFilter.MaxFilter(2*support_radius+1)), dtype=bool)
    assert not (contribution & ~support).any(), "Frame resolve wrapped scatter outside its local support"
    assert not (guided & contribution).any(), "Native scatter-on/off oracle found scatter in terrain guides"
    np.testing.assert_array_equal(after_hdr[contribution,:3], before_hdr[contribution,:3])
    np.testing.assert_array_equal(after[expanded], before[expanded])
    assert not np.any(normals[-1][expanded]), "Scatter footprint must not enter terrain neighbourhoods"
    assert denoised.metadata["denoiser_used"] == "atrous"
    np.testing.assert_array_equal(after[scatter], before[scatter])
    assert "scatter" in denoised.metadata["denoiser_guide_coverage"]


def test_hdr_paint_excludes_later_title_and_label(tmp_path, monkeypatch):
    from test_c01_review_gpu import _scene
    from forge3d.terrain_materials import _linear
    grid, recipe = _scene()
    swatch = (40,60,20,255)
    material = f3d.GeospatialMaterialRecipe.land_cover(f3d.CategoricalRasterStyle(
        {1:swatch},{1:"Protected"}), albedo=(150,140,130))
    recipe = material.bind(recipe, grid, classes=np.ones((64,64),np.int32))
    label = f3d.LabelLayer("labels", labels=[{"id":"label","text":"Paint label",
        "geometry":{"type":"Point","coordinates":[128,128,0]}}],
        glyph_atlas={"glyphs":sorted(set("Paint label"))})
    recipe = replace(recipe, layers=(label,),
        map_furniture=f3d.MapFurnitureLayer(title="PROTECTED HDR TITLE"),
        output=replace(recipe.output,width=256,height=256,hdr=True))
    captured = {}
    original_render, original_write = ms._render_native_offscreen_result, ms._write_exr_array
    original_label, original_furniture = ms._composite_native_label_layers, ms._compose_furniture
    def label_stage(base,*args,**kwargs):
        before=base.copy()
        result=original_label(base,*args,**kwargs)
        captured["label_changed"]=np.any(before!=result[0],axis=-1)
        return result
    def furniture_stage(base,*args,**kwargs):
        before=base.copy()
        result=original_furniture(base,*args,**kwargs)
        captured["furniture_changed"]=np.any(before!=base,axis=-1)
        captured["paint_before_overlays"]=np.all(before==swatch,axis=-1)|captured["label_changed"]
        return result
    def observe(*args,**kwargs):
        result=original_render(*args,**kwargs)
        captured.update(hdr=np.asarray(result.hdr_frame.to_numpy_f32()).copy(),
            rgba=result.rgba.copy(),mask=result.hdr_overlay_mask.copy())
        return result
    def write(path,array,**kwargs):
        captured["exr"]=np.asarray(array).copy()
        return original_write(path,array,**kwargs)
    monkeypatch.setattr(ms,"_composite_native_label_layers",label_stage)
    monkeypatch.setattr(ms,"_compose_furniture",furniture_stage)
    monkeypatch.setattr(ms,"_render_native_offscreen_result",observe)
    monkeypatch.setattr(ms,"_write_exr_array",write)
    scene=f3d.MapScene(recipe=recipe)
    scene.render(tmp_path/"overlaid.png")
    assert captured["label_changed"].any() and captured["furniture_changed"].any()
    exact=np.all(captured["rgba"]==swatch,axis=-1)
    overlays=(captured["label_changed"]|captured["furniture_changed"]) & ~exact
    assert (overlays & captured["paint_before_overlays"]).any() and exact.any()
    _retain("hdr-overlaid-paint",**{k:v for k,v in captured.items() if isinstance(v,np.ndarray)})
    np.testing.assert_array_equal(captured["exr"][overlays],captured["hdr"][overlays])
    np.testing.assert_array_equal(captured["mask"],exact)
    np.testing.assert_array_equal(captured["exr"][exact,:3],
        np.tile(_linear(np.array(swatch[:3],np.float32)/255),(exact.sum(),1)))
    assert "physical pixels retain scene radiance" in scene.last_render_metadata["hdr_overlay_encoding"]


def test_protected_paint_visibility_is_denoiser_invariant_with_ignored_sampling_keys(tmp_path, monkeypatch):
    from test_c01_review_gpu import _scene
    grid, recipe = _scene()
    swatch=(40,60,20,255)
    material=f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("ground",(150,140,130),roughness=.8),),
        categorical=f3d.CategoricalRasterStyle({1:swatch},{1:"Protected"}))
    ring=[[500024,4599976],[500040,4599976],[500040,4599960],[500024,4599960],[500024,4599976]]
    building=f3d.MapSceneBuildingLayer("buildings",features=[{"type":"Feature","id":"building",
        "properties":{"height":25},"geometry":{"type":"Polygon","coordinates":[ring]}}],metadata={"crs":grid.crs})
    recipe=replace(material.bind(recipe,grid,classes=np.ones((64,64),np.int32)),layers=(building,),
        camera=replace(recipe.camera,distance=110,elevation_deg=40,azimuth_deg=140),map_furniture=None,
        lighting=replace(recipe.lighting,settings={**recipe.lighting.settings,
            "cli_params":{"camera_mode":"mesh","msaa_samples":4,"render_scale":.5}}),
        output=replace(recipe.output,samples=4,width=256,height=256))
    frames=[]
    original=ms._render_native_offscreen_result
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        frames.append((result.rgba.copy(),result.hdr_overlay_mask.copy()))
        return result
    monkeypatch.setattr(ms,"_render_native_offscreen_result",observe)
    with ms._shared_terrain_render_context():
        for denoiser in ("none","atrous"):
            with pytest.warns(UserWarning, match="MapScene.*ignores"):
                f3d.MapScene(recipe=replace(recipe,output=replace(recipe.output,denoiser=denoiser))).render(tmp_path/(denoiser+".png"))
    (before,paint_before),(after,paint_after)=frames
    _retain("protected-fixed-sampling",ordinary=before,filtered=after,
        paint_before=paint_before,paint_after=paint_after)
    assert paint_after.any() and (~paint_after).any()
    np.testing.assert_array_equal(paint_before,paint_after)
    np.testing.assert_array_equal(before[paint_before|paint_after],after[paint_before|paint_after])


def _visibility_fixture(buildings=True, protected=True):
    from test_c01_review_gpu import _scene
    grid, recipe = _scene()
    yy, xx = np.mgrid[:64,:64]
    heights = (2*np.sin(xx/5)+3*np.cos(yy/7)+xx*.04).astype(np.float32)
    terrain = replace(recipe.terrain, data=heights, metadata={"width":64,"height":64,
        "resolution":1,"bounds":(499999.5,4599936.5,500063.5,4600000.5),"source_id":"visibility-fixture"})
    recipe = replace(recipe, terrain=terrain,
        camera=replace(recipe.camera,distance=110,elevation_deg=50,azimuth_deg=140),
        lighting=replace(recipe.lighting,sun_direction=(.5,.3,.8)),
        output=replace(recipe.output,width=256,height=256),map_furniture=None)
    if protected:
        material=f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("ground",(150,140,130),roughness=.8),),
            categorical=f3d.CategoricalRasterStyle({1:(40,60,20,255)},{1:"Protected"}))
        recipe=material.bind(recipe,grid,classes=np.ones((64,64),np.int32))
    if buildings:
        ring=[[500024,4599976],[500036,4599976],[500036,4599964],[500024,4599964],[500024,4599976]]
        building=f3d.MapSceneBuildingLayer("buildings",features=[{"type":"Feature","id":"building",
            "properties":{"height":8},"geometry":{"type":"Polygon","coordinates":[ring]}}],metadata={"crs":grid.crs})
        recipe=replace(recipe,layers=(building,))
    return grid,replace(recipe,map_furniture=None)


@pytest.mark.parametrize("samples,buildings", ((1,False),(1,True),(4,False),(4,True)))
@pytest.mark.parametrize("content", ("paint","vector"))
def test_mapscene_ignores_recipe_sampling_in_every_visibility_route(tmp_path,monkeypatch,samples,buildings,content):
    grid,recipe=_visibility_fixture(buildings=buildings,protected=content=="paint")
    if content=="vector":
        outline=[[500005,4599995],[500059,4599995],[500059,4599941],[500005,4599941],[500005,4599995]]
        vector=f3d.VectorOverlay("vector",features=[{"type":"Feature","id":"shape","properties":{},
            "geometry":{"type":"Polygon","coordinates":[outline]}}],crs=grid.crs,metadata={"coordinate_space":"world"})
        recipe=replace(recipe,layers=(*recipe.layers,vector))
    recipe=replace(recipe,lighting=replace(recipe.lighting,settings={**recipe.lighting.settings,
        "cli_params":{"camera_mode":"mesh","msaa_samples":4,"render_scale":.5}}),
        output=replace(recipe.output,samples=samples))
    frames=[]
    original=ms._render_native_offscreen_result
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        frames.append(result)
        return result
    monkeypatch.setattr(ms,"_render_native_offscreen_result",observe)
    aov_configs=[]
    original_aov=f3d.TerrainRenderer.render_with_aov
    def observe_aov(self,*args,**kwargs):
        params=kwargs.get("params",args[2] if len(args)>2 else None)
        aov_configs.append(params.python_object)
        return original_aov(self,*args,**kwargs)
    monkeypatch.setattr(f3d.TerrainRenderer,"render_with_aov",observe_aov)
    settings=recipe.lighting.settings
    control=replace(recipe,lighting=replace(recipe.lighting,settings={**settings,
        "cli_params":{name:value for name,value in settings["cli_params"].items()
                      if name not in ("msaa_samples","render_scale")}}))
    with ms._shared_terrain_render_context():
        with pytest.warns(UserWarning,match="MapScene.*ignores"):
            f3d.MapScene(recipe=recipe).render(tmp_path/"visible.png")
        assert len(aov_configs)==1, "Fixed sampling must not add a one-shot visibility draw"
        f3d.MapScene(recipe=control).render(tmp_path/"control.png")
    assert len(frames)==2 and len(aov_configs)==2
    assert all(c.msaa_samples==1 and c.render_scale==1.0 for c in aov_configs)
    np.testing.assert_array_equal(frames[0].rgba,frames[1].rgba)
    assert recipe.lighting.settings["cli_params"]["msaa_samples"]==4
    assert recipe.lighting.settings["cli_params"]["render_scale"]==.5
    depth,uv,terrain=__import__('forge3d._map_scene_projection',fromlist=['frame_visibility']).frame_visibility(recipe,frames[0].aov_frame)
    assert terrain.any() and np.isfinite(uv[terrain]).all()
    if content=="paint":
        assert np.all(frames[0].rgba[terrain] == (40,60,20,255))
    _retain(f"fixed-visibility-{content}-s{samples}-b{buildings}",rgba=frames[0].rgba,terrain=terrain,depth=depth)
