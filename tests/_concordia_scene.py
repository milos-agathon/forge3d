"""Deterministic CONCORDIA acceptance scene shared by tests and measurements."""
from __future__ import annotations

import numpy as np
import forge3d as f3d


def scene(*, width=256, height=192, effects=("ssao", "ssgi", "ssr", "taa", "cloud_shadows"),
          cartography=True, atmosphere=False, height_ao=True):
    x, y = np.meshgrid(np.linspace(-1, 1, 32, dtype=np.float32), np.linspace(-1, 1, 32, dtype=np.float32))
    dem = (1.5 + 2*np.exp(-8*(x+0.35)**2-3*y*y) + 0.25*x).astype(np.float32)
    settings = {
        "camera": {"camera_mode": "mesh:zup"},
        "screen_space": {name: {"enabled": name in effects} for name in ("ssao", "ssgi", "ssr", "taa")},
        "height_ao": {"enabled": height_ao},
        "clouds": {"enabled": "cloud_shadows" in effects, "shadows_enabled": "cloud_shadows" in effects,
                   "coverage": 0.5, "density": 0.5, "shadow_strength": 0.5},
        "sun": {"azimuth_deg": 135.0, "elevation_deg": 42.0},
    }
    if atmosphere:
        settings["fog"] = {"density": 0.5, "height_falloff": 0.1}
        settings["volumetrics"] = {"enabled": True, "density": 0.005, "mode": "height"}
    layers = [f3d.MapSceneBuildingLayer(
        layer_id="building", source={"source_id": "concordia-building", "asset_status": "fixture"},
        support_level="supported", geometry_count=3, material_status="scalar_pbr_underdeveloped",
        features=[{"id": name, "geometry": {"type": "Polygon", "coordinates": [[
            [a,b],[a+size,b],[a+size,b+size],[a,b+size],[a,b]]]},
            "properties": {"height": elevation, "roof:shape": "flat"}}
            for name,a,b,size,elevation in (("block",0.42,0.42,0.16,10.0),
                ("west",0.1,0.15,0.12,5.0),("east",0.78,0.73,0.12,5.0))],
        metadata={"source_id": "concordia-building", "asset_status": "fixture"})]
    if cartography:
        layers.extend([
            f3d.VectorOverlay(layer_id="road", crs="EPSG:32610", width_px=1.0,
                style={"version":8,"layers":[{"id":"road","type":"line","paint":{"line-color":"#ffcc22"}}]}, features=[{
                    "id": "road", "geometry": {"type": "LineString", "coordinates": [
                        [-12.0, -3.0],[-2.0,-3.0],[12.0,-3.0]]}}]),
            f3d.LabelLayer(layer_id="labels", labels=[
                {"id":"ridge", "text":"Ridge", "kind":"point",
                 "geometry":{"type":"Point","coordinates":[-10.0,7.0,5.0]}},
                {"id":"behind-building", "text":"Hidden", "kind":"point",
                 "geometry":{"type":"Point","coordinates":[0.0,0.0,2.5]}}],
                glyph_atlas={"glyphs": sorted(set("RidgeHidden"))},
                metadata={"coordinate_space":"world"})])
    return f3d.MapScene(
        terrain=f3d.TerrainSource(data=dem, crs="EPSG:32610", elevation_sampling_available=True,
            metadata={"source_id":"concordia-dem","width":32,"height":32,
                      "bounds":[-16.0,-16.0,16.0,16.0],"asset_status":"fixture"}),
        camera=f3d.OrbitCamera(target=(0.0,0.0,0.0),distance=45.0,azimuth_deg=30.0,elevation_deg=48.0),
        lighting=f3d.LightingPreset(name="daylight",intensity=1.0,settings=settings),
        output=f3d.OutputSpec(width=width,height=height,format="png"),layers=layers,
        reproducibility_profile=f3d.ReproducibilityProfile(seed=45))


def render(scene):
    import forge3d.map_scene as ms
    plan = scene.compile_plan()
    before = {name: value.to_dict() for name,value in plan.label_plans.items()}
    result = ms._render_native_offscreen_result(scene.recipe,plan)
    assert result is not None, "a physical native frame is required"
    assert before == {name: value.to_dict() for name,value in plan.label_plans.items()}
    return result


def require_gpu():
    import pytest
    from _terrain_runtime import terrain_rendering_available
    if not terrain_rendering_available():
        pytest.skip("CONCORDIA requires a terrain-capable physical GPU")


def capture(model, *, history=False):
    """Explicit measurement readbacks occur only after the beauty read."""
    import forge3d.map_scene as ms
    require_gpu()
    if not ms._mapscene_native_effects(model.recipe):
        # Validation needs the interactive AOV getter even for a height-only
        # ablation. Request a zero-radiance native GI pass; product routing is
        # deliberately unchanged and the caller's recipe is not mutated.
        import copy
        model = copy.deepcopy(model)
        model.recipe.lighting.settings.setdefault("screen_space", {})["ssgi"] = {
            "enabled": True, "intensity": 0.0}
    with ms._shared_terrain_render_context():
        if history:
            render(model)
        result = render(model)
        renderer = ms._ACTIVE_TERRAIN_RESOURCES.holder[0].renderer
        data = renderer._interactive_validation_data()
    pipeline = result.metadata["native_pipeline"]
    assert pipeline["image_readbacks_before_output"] == 0
    assert pipeline["beauty_readbacks"] == 1
    return result, data


def world_positions(data):
    depth = data["depth"]
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    ndc = np.stack(((xx+.5)*2/width-1, 1-(yy+.5)*2/height,
                    np.zeros_like(xx), np.ones_like(xx)), axis=-1)
    inv_proj = np.asarray(data["inv_proj"]).T
    near = ndc @ inv_proj.T
    near = near[..., :3] / near[..., 3:4]
    ndc[..., 2] = 1
    far = ndc @ inv_proj.T
    far = far[..., :3] / far[..., 3:4]
    view = near + (far-near)*((depth+near[..., 2])/(near[..., 2]-far[..., 2]))[..., None]
    return (np.concatenate((view, np.ones((*depth.shape, 1))), axis=-1)
            @ np.asarray(data["inv_view"]))[..., :3]


def height_visibility_at_pixels(data):
    world = world_positions(data)
    field = data["height_visibility"]
    size = np.asarray([field.shape[1], field.shape[0]])
    # The shared mesh world +Y follows increasing source row; north is a camera reflection.
    scale_bias = np.asarray(data["cloud_scale_bias"])
    uv = world[..., :2] * scale_bias[:2] + scale_bias[2:]
    pixel = np.minimum((np.clip(uv, 0, 1)*size).astype(int), size-1)
    return np.where(data["surface_kind"] == 1, field[pixel[..., 1], pixel[..., 0]], 1.0)


def reference_scene():
    """A 16m canyon floor, 6m ridges, and three near-field building contacts."""
    model=scene(effects=("ssao",),cartography=False)
    x,y=np.meshgrid(np.linspace(-1,1,32,dtype=np.float32),np.linspace(-1,1,32,dtype=np.float32))
    model.recipe.terrain.data=(1.5+6*(np.exp(-60*(x-.65)**2)+np.exp(-60*(x+.65)**2))).astype(np.float32)
    # A flat facet needs two adjacent grid vertices. Flatten only one cell
    # at each existing crest, preserving the original height range/camera.
    # These maximum-height facets are physically clear receivers in the same
    # canyon/building scene, not a separate all-clear-plane luminance test.
    profile=model.recipe.terrain.data[0]
    middle=len(profile)//2
    for peak in (int(np.argmax(profile[:middle])),middle+int(np.argmax(profile[middle:]))):
        adjacent=peak+(1 if peak<middle else -1)
        model.recipe.terrain.data[:,[peak,adjacent]]=profile.max()

    settings=model.recipe.lighting.settings
    settings["height_ao"].update(max_distance=32.0,steps=64,directions=16)
    # Near-field contact radius spans twice the actual tallest building's
    # render-space height. This is derived from the shared geometry contract,
    # rather than assuming the source height units survive normalization.
    from forge3d._map_scene_projection import TerrainProjector
    from forge3d._map_scene_native_inputs import building_world_triangles
    import forge3d.map_scene as ms
    projector=TerrainProjector(model.recipe)
    triangles=np.asarray(building_world_triangles(projector))
    batches=ms._terrain_scatter_building_batches_for_recipe(model.recipe,projector.heightmap,shared_frame=True)[0]
    heights=[]
    cursor=0
    for batch in batches:
        count=np.asarray(batch["levels"][0]["mesh"]["indices"]).size//3*len(batch["transforms"])
        vertices=triangles[cursor:cursor+count].reshape(-1,3);cursor+=count
        heights.append(float(np.ptp(vertices[:,2])))
    settings["screen_space"]["ssao"].update(radius=2*max(heights),intensity=1.0)
    return model
