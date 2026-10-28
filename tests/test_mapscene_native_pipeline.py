from __future__ import annotations

import ast
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as ms
from _concordia_scene import capture, render, require_gpu, scene


def test_native_path_has_no_cpu_surrogate_calls():
    for function in (ms._render_native_offscreen_result, ms._render_terrain_renderer_result_impl):
        tree = ast.parse(inspect.getsource(function))
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        assert not calls & {"_apply_mapscene_screen_space","_apply_mapscene_cloud_shadow"}
    directory=Path(__file__).parents[1]/"src/terrain/renderer/interactive"
    for path in directory.glob("*.rs"):
        if path.name == "validation.rs":
            continue  # Explicit after-beauty instrumentation is not a frame input.
        source=path.read_text()
        assert not any(token in source for token in ("map_async(","copy_texture_to_buffer(","read_texture(","readback_texture(")),path
    source=(directory/"execution.rs").read_text()
    assert source.count("self.queue.submit(")==1


def test_interactive_native_surface_registered_exported_and_stubbed():
    from forge3d._native import get_native_module
    assert f3d.TerrainRenderer is get_native_module().TerrainRenderer
    assert "TerrainRenderer" in f3d.__all__
    for name in ("render_interactive","interactive_pipeline_info"):
        assert callable(getattr(f3d.TerrainRenderer,name))
    module=ast.parse(Path(f3d.__file__).with_name("__init__.pyi").read_text())
    renderer=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=="TerrainRenderer")
    assert {"render_interactive","interactive_pipeline_info"} <= {node.name for node in renderer.body if isinstance(node,ast.FunctionDef)}


def test_interactive_cache_rejects_before_rendering_and_none_renders(tmp_path):
    require_gpu()
    from forge3d.determinism import (
        _canonical_params_config, canonical_heightmap, write_canonical_hdr,
    )

    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    inputs = dict(
        material_set=f3d.MaterialSet.terrain_default(),
        env_maps=f3d.IBL.from_hdr(str(hdr_path), intensity=1.0),
        params=f3d.TerrainRenderParams(_canonical_params_config()(64, 64)),
        heightmap=np.ascontiguousarray(canonical_heightmap(), dtype=np.float32),
    )
    before = renderer.interactive_pipeline_info()
    cache = tmp_path / "native-cache"
    with pytest.raises(RuntimeError, match="does not yet serialize the shared-depth HDR graph; use cache=None"):
        renderer.render_interactive(**inputs, cache=cache)
    assert not cache.exists()
    assert renderer.interactive_pipeline_info() == before
    assert renderer.last_anamnesis_cache_report["graph_command_submissions"] == 0

    rgba = renderer.render_interactive(**inputs, cache=None).to_numpy()
    assert rgba.shape == (64, 64, 4)
    assert np.any(rgba[..., :3])
    assert json.loads(renderer.interactive_pipeline_info())["beauty_readbacks"] == 1


@pytest.mark.parametrize("effects", [tuple(name for name,on in zip(
    ("ssao","ssgi","ssr","taa","cloud_shadows","volumetrics","height_ao","atmosphere"), flags) if on)
    for flags in itertools.product((False,True),repeat=8) if any(flags[:6])])
def test_every_enabled_subset_has_live_ordered_gpu_passes(effects, monkeypatch):
    require_gpu()
    def forbidden(*args,**kwargs):
        raise AssertionError("native rendering called a CPU surrogate")
    monkeypatch.setattr(ms,"_apply_mapscene_screen_space",forbidden)
    monkeypatch.setattr(ms,"_apply_mapscene_cloud_shadow",forbidden)
    model=scene(effects=effects,cartography=False,height_ao="height_ao" in effects,width=96,height=80)
    model.recipe.lighting.settings["fog"]={"density":0.5 if "atmosphere" in effects else 0.0}
    model.recipe.lighting.settings["volumetrics"]={"enabled":"volumetrics" in effects,"density":0.005,"mode":"height"}
    with ms._shared_terrain_render_context():
        result=render(model)
    pipeline=result.metadata["native_pipeline"]
    assert pipeline["image_readbacks_before_output"]==0
    assert pipeline["beauty_readbacks"]==1
    execution=pipeline["execution_report"]
    assert execution["adapter"]
    passes=execution["passes"]
    labels=[entry["label"] for entry in passes]
    expected=["concordia.resources"]
    if "cloud_shadows" in effects: expected.append("concordia.cloud_shadows")
    expected.extend(["concordia.shadows","concordia.geometry"])
    if set(effects)&{"ssao","ssgi","ssr"}: expected.append("concordia.depth_pyramid")
    if set(effects)&{"ssao","height_ao"}: expected.append("concordia.ao")
    if "ssgi" in effects: expected.append("concordia.ssgi")
    if "ssr" in effects: expected.append("concordia.ssr")
    if set(effects)&{"cloud_shadows","atmosphere"}: expected.append("concordia.atmosphere")
    if "volumetrics" in effects: expected.append("concordia.froxels")
    if "taa" in effects: expected.append("concordia.temporal")
    expected.append("concordia.output_transform")
    assert [label for label in labels if label.startswith("concordia.")] == expected
    if pipeline["gpu_timestamps_available"]:
        assert all(entry.get("gpu_ms") is not None and entry["gpu_ms"]>=0 for entry in passes if entry["label"] in expected)
    assert pipeline["input_size"]==pipeline["output_size"]==[96,80]
    assert "mapscene_numpy" not in json.dumps(result.metadata)
    assert np.isfinite(result.rgba).all() and np.any(result.rgba[...,:3])


def test_mixed_frame_preserves_compiled_labels_and_full_resolution():
    require_gpu()
    result=render(scene())
    assert result.rgba.shape==(192,256,4)
    pipeline=result.metadata["native_pipeline"]
    labels=[entry["label"] for entry in pipeline["execution_report"]["passes"]]
    assert pipeline["vector_batches"]>0
    assert labels.index("concordia.temporal")<labels.index("concordia.output_transform")<labels.index("concordia.vectors")
    if pipeline["label_batches"]:
        assert labels.index("concordia.output_transform")<labels.index("concordia.labels")


@pytest.mark.parametrize("alpha,edl", [(0,False),(128,False),(128,True)])
def test_transparent_points_preserve_opaque_depth(alpha, edl):
    base = scene(effects=("taa",), cartography=False, height_ao=False)
    before, opaque = capture(base)
    base.recipe.layers = [*base.recipe.layers, f3d.PointCloudLayer(layer_id="transparent", crs="EPSG:32610",
        point_count=3, support_level="supported", metadata={"coordinate_space":"world",
            "positions":[[-8.0,-8.0,5.0],[0.0,0.0,15.0],[8.0,8.0,-5.0]],
            "color":[255,40,20,alpha],"point_size":18.0,"edl":edl})]
    after, points = capture(base)
    np.testing.assert_array_equal(points["depth"], opaque["depth"])
    np.testing.assert_array_equal(points["surface_kind"], opaque["surface_kind"])
    assert points["point_depth_convention"] == "normalized_device_depth"
    assert len(points["point_depths"]) == 1
    representative = points["point_depths"][0]
    if alpha == 0:
        np.testing.assert_array_equal(before.rgba, after.rgba)
        assert np.all(representative == 1.0)
    else:
        assert np.any(representative < 1.0)
        assert np.any(before.rgba != after.rgba)
    assert after.metadata["native_pipeline"]["layer_stages"]["point_clouds"] == "post_resolve_depth_aware_unlit"


def test_shared_scatter_buildings_keep_the_y_up_mesh_contract():
    from forge3d._map_scene_projection import TerrainProjector

    model = scene()
    heightmap = TerrainProjector(model.recipe).heightmap
    ordinary, _ = ms._terrain_scatter_building_batches_for_recipe(
        model.recipe, heightmap, shared_frame=False
    )
    shared, _ = ms._terrain_scatter_building_batches_for_recipe(
        model.recipe, heightmap, shared_frame=True
    )
    assert ordinary and len(shared) == len(ordinary)
    for first, second in zip(ordinary, shared):
        # Native instance packing owns the Y-up to Z-up conversion for both paths.
        np.testing.assert_array_equal(first["transforms"], second["transforms"])
        a = first["levels"][0]["mesh"]
        b = second["levels"][0]["mesh"]
        np.testing.assert_array_equal(a["positions"], b["positions"])
        np.testing.assert_array_equal(a["normals"], b["normals"])
        # The shared path retains its existing front-face winding convention.
        np.testing.assert_array_equal(
            np.asarray(a["indices"]).reshape(-1, 3)[:, [0, 2, 1]],
            np.asarray(b["indices"]).reshape(-1, 3),
        )


def test_building_occlusion_is_frozen_before_rendering():
    require_gpu()
    model = scene()
    import copy
    from forge3d._map_scene_projection import TerrainProjector
    projector = TerrainProjector(model.recipe)
    assert 2.5 > projector.height(projector.uv([0.0,0.0]))
    terrain_only = copy.deepcopy(model)
    terrain_only.recipe.layers = [layer for layer in terrain_only.recipe.layers
                                if not isinstance(layer, f3d.MapSceneBuildingLayer)]
    assert "behind-building" in [item.label_id for item in terrain_only.compile_plan().label_plans["labels"].accepted]
    legacy = copy.deepcopy(model)
    legacy.recipe.lighting.settings["screen_space"] = {}
    legacy.recipe.lighting.settings["clouds"]["enabled"] = False
    assert "behind-building" in [item.label_id for item in legacy.compile_plan().label_plans["labels"].accepted]
    plan = model.compile_plan()
    labels = plan.label_plans["labels"]
    assert "behind-building" not in [item.label_id for item in labels.accepted]
    assert any(item.label_id == "behind-building" and item.reason == "terrain_occluded"
               for item in labels.rejected)
    before = labels.to_dict()
    assert ms._render_native_offscreen_result(model.recipe, plan) is not None
    assert labels.to_dict() == before
    hidden_only=scene()
    layer=next(layer for layer in hidden_only.recipe.layers if isinstance(layer,f3d.LabelLayer))
    layer.labels=[record for record in layer.labels if record["id"]=="behind-building"]
    _,occluded=capture(hidden_only)
    assert not np.any(occluded["label_coverage"])
    hidden_only.recipe.layers=[layer for layer in hidden_only.recipe.layers
                               if not isinstance(layer,f3d.MapSceneBuildingLayer)]
    _,visible=capture(hidden_only)
    assert np.any(visible["label_coverage"])


def test_gtao_alias_routes_one_authoritative_ao_pass():
    model=scene(effects=(),cartography=False,height_ao=False)
    model.recipe.lighting.settings["screen_space"]["gtao"]={"enabled":True,"radius":2.0}
    result,_=capture(model)
    labels=[item["label"] for item in result.metadata["native_pipeline"]["execution_report"]["passes"]]
    assert labels.count("concordia.ao")==1
    assert result.metadata["native_pipeline"]["effects"]["gtao"]


def test_conflicting_ao_aliases_are_diagnosed():
    model=scene(cartography=False)
    settings=model.recipe.lighting.settings["screen_space"]
    settings["ssao"]={"enabled":True,"radius":2.0}
    settings["gtao"]={"enabled":True,"radius":3.0}
    with pytest.raises(ValueError,match="conflicting radius"):
        ms._mapscene_screen_space_settings(model.recipe)


def test_ground_vector_is_hidden_by_nearer_building_depth():
    model=scene(effects=("taa",),cartography=False,height_ao=False)
    clear,geometry=capture(model)
    road=next(layer for layer in scene().recipe.layers if isinstance(layer,f3d.VectorOverlay))
    model.recipe.layers=[*model.recipe.layers,road]
    drawn,data=capture(model)
    np.testing.assert_array_equal(data["depth"],geometry["depth"])
    changed=np.any(drawn.rgba[...,:3]!=clear.rgba[...,:3],axis=-1)
    assert changed.any()
    assert not changed[data["surface_kind"]==2].any()


def test_clipmap_two_frames_clear_shared_geometry_attachments():
    require_gpu()
    model=scene(effects=("ssao","taa"),cartography=False,height_ao=False)
    model.recipe.lighting.settings["camera"]={"camera_mode":"screen"}
    model.recipe.terrain.metadata["clipmap"]={"enabled":True,"levels":3,"ring_resolution":16,
        "terrain_extent_m":32.0,"max_resident_height_bytes":3*16*16*4}
    with ms._shared_terrain_render_context():
        first=render(model)
        assert first.metadata["terrain_geometry_backend"]=="clipmap_indexed_pbr"
        model.recipe.camera.distance=90.0
        second=render(model)
        data=ms._ACTIVE_TERRAIN_RESOURCES.holder[0].renderer._interactive_validation_data()
    fresh,reference=capture(model,history=True)
    np.testing.assert_array_equal(data["surface_kind"],reference["surface_kind"])
    np.testing.assert_array_equal(data["depth"],reference["depth"])
    assert second.metadata["native_pipeline"]["image_readbacks_before_output"]==0
    assert np.any(data["surface_kind"]==1)


def test_post_resolve_gpu_cartography_coverage_is_byte_identical():
    """Compare submitted full-resolution draws, including their shared depth."""
    import hashlib
    import os
    from pathlib import Path
    from PIL import Image

    temporal, with_taa = capture(scene(effects=("taa",), height_ao=True), history=True)
    assert temporal.metadata["native_pipeline"]["temporal_history_used"]
    direct, without_taa = capture(scene(effects=(), height_ao=True))
    np.testing.assert_array_equal(with_taa["cartography_depth"],without_taa["cartography_depth"])
    hashes = {}
    for name in ("label_coverage", "vector_coverage"):
        assert with_taa[name].shape[:2] == (192, 256)
        assert np.any(with_taa[name]), f"{name} must exercise a real draw"
        np.testing.assert_array_equal(with_taa[name], without_taa[name])
        hashes[name] = hashlib.sha256(with_taa[name].tobytes()).hexdigest()
    directory = os.environ.get("FORGE3D_CONCORDIA_ARTIFACT_DIR")
    if directory:
        folder = Path(directory) / "cartography"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "gpu-coverage.npz", **{
            name: with_taa[name] for name in hashes})
        Image.fromarray(with_taa["label_coverage"]).save(folder / "label-coverage.png")
        Image.fromarray(with_taa["vector_coverage"]).save(folder / "vector-coverage.png")
        Image.fromarray(temporal.rgba).save(folder / "temporal.png")
        Image.fromarray(direct.rgba).save(folder / "without-temporal.png")
        (folder / "coverage.json").write_text(json.dumps({
            "output_size": [256, 192], "byte_identical": True, "sha256": hashes,
            "vector_storage": "rgba8unorm", "label_storage": "rgba8unorm",
            "temporal_mode": temporal.metadata["native_pipeline"]["temporal_mode"],
            "taau_available": temporal.metadata["native_pipeline"]["taau_available"],
            "scope": "explicit after-beauty GPU replay of the exact submitted full-resolution batches",
        }, indent=2))


def test_public_render_exercises_mixed_native_frame(tmp_path, monkeypatch):
    from PIL import Image
    require_gpu()
    model = scene(atmosphere=True)
    model.recipe.map_furniture = f3d.MapFurnitureLayer(
        title="CONCORDIA", north_arrow={"style": "simple", "size": 24})
    model.recipe.layers = [*model.recipe.layers, f3d.PointCloudLayer(
        layer_id="points", crs="EPSG:32610", point_count=1, support_level="supported",
        metadata={"coordinate_space": "world", "positions": [[-8.0, -8.0, 5.0]],
                  "color": [255, 40, 20, 128], "point_size": 12.0, "edl": True})]
    plan = model.compile_plan()
    frozen = {name: value.to_dict() for name, value in plan.label_plans.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("public rendering called a CPU surrogate")
    monkeypatch.setattr(ms, "_apply_mapscene_screen_space", forbidden)
    monkeypatch.setattr(ms, "_apply_mapscene_cloud_shadow", forbidden)
    output = tmp_path / "public-mixed.png"
    with ms._shared_terrain_render_context():
        model.render(str(tmp_path / "warmup.png"))
        report = model.render(str(output))
    image = np.asarray(Image.open(output).convert("RGBA"))
    assert image.shape == (192, 256, 4) and image[..., :3].any()
    pipeline = model.last_render_metadata["native_pipeline"]
    assert pipeline["image_readbacks_before_output"] == 0
    assert pipeline["beauty_readbacks"] == 1
    labels = [item["label"] for item in pipeline["execution_report"]["passes"]]
    assert labels.index("concordia.temporal") < labels.index("concordia.transparent_points")
    assert labels.index("concordia.transparent_points") < labels.index("concordia.vectors")
    assert labels.index("concordia.labels") < labels.index("concordia.furniture")
    assert pipeline["point_edl_layers"] == 1
    from forge3d import _forge3d as native
    assert pipeline["execution_report"] == json.loads(native.render_execution_report())
    recorded = json.loads(pipeline["execution_report"]["inputs"]["concordia.pipeline"])
    assert recorded["effects"] == pipeline["effects"]
    assert recorded["furniture_batches"] == pipeline["furniture_batches"] == 1
    for feature in ("mapscene.ssao", "mapscene.ssgi", "mapscene.ssr", "mapscene.taa",
                    "mapscene.cloud_shadows", "point_cloud.mapscene_render", "point_cloud.edl",
                    "mapscene.furniture_composite"):
        assert report.supported_features[feature] == "supported"
    assert frozen == {name: value.to_dict() for name, value in plan.label_plans.items()}


def test_screen_camera_has_native_shared_depth(tmp_path):
    require_gpu()
    model = scene(effects=("ssao","ssgi","ssr","taa","cloud_shadows"), cartography=False)
    model.recipe.lighting.settings["camera"] = {"camera_mode": "screen"}
    result, data = capture(model)
    assert (data["surface_kind"] == 1).any() and (data["surface_kind"] == 2).any()
    assert np.ptp(data["depth"][data["surface_kind"] > 0]) > 0
    assert result.metadata["native_pipeline"]["beauty_readbacks"] == 1
    assert np.isfinite(result.rgba).all()


def test_shared_motion_is_finite_uv_displacement_across_camera_turn():
    from _concordia_scene import world_positions
    require_gpu()
    model = scene(effects=("taa",), cartography=False, height_ao=False)
    with ms._shared_terrain_render_context():
        render(model)
        renderer = ms._ACTIVE_TERRAIN_RESOURCES.holder[0].renderer
        previous = renderer._interactive_validation_data()
        model.recipe.camera.azimuth_deg += 180.0
        current = render(model)
        data = renderer._interactive_validation_data()
    assert not current.metadata["native_pipeline"]["temporal_history_reset"]
    assert current.metadata["native_pipeline"]["velocity_convention"] == "current_minus_previous_uv_positive_y_down"
    world = world_positions(data)
    homogeneous = np.concatenate((world, np.ones((*world.shape[:2], 1))), axis=-1)
    clip = homogeneous @ np.asarray(previous["view"]) @ np.asarray(previous["proj"])
    previous_ndc = np.full((*world.shape[:2], 2), 3.0)
    valid = clip[..., 3] > 0
    previous_ndc[valid] = clip[valid, :2] / clip[valid, 3:4]
    yy, xx = np.indices(data["depth"].shape)
    current_ndc = np.stack(((xx+.5)*2/256-1, 1-(yy+.5)*2/192), axis=-1)
    jitter = current.metadata["native_pipeline"]["raster_jitter_px"]
    current_ndc -= np.asarray(jitter)/[256,192]*[2,-2]
    expected = np.clip((current_ndc-previous_ndc)*[.5,-.5], -1, 1)
    hit = data["surface_kind"] > 0
    assert np.isfinite(data["velocity"]).all()
    np.testing.assert_allclose(data["velocity"][hit], expected[hit],
                               atol=2*np.finfo(np.float16).eps, rtol=0)


@pytest.mark.parametrize("cap", ["butt", "round", "square"])
@pytest.mark.parametrize("join", ["miter", "round", "bevel"])
@pytest.mark.parametrize("dashed", [False, True])
def test_bounded_native_stroke_preserves_full_canvas_coverage(cap, join, dashed):
    from forge3d._map_scene_render import _draw_polyline
    # Include overlapping joins, a zero-length segment, and offscreen caps.
    points = [(-18, 6), (19, 31), (19, 31), (48, 9), (67, 70), (120, 45)]
    full = np.zeros((83, 107, 4), dtype=np.uint8)
    bounded = full.copy()
    options = dict(width_px=7.25, cap=cap, join=join,
                   dash_array=[7.5, 3.25] if dashed else None)
    _draw_polyline(full, points, (255, 128, 0, 173), **options)
    _draw_polyline(bounded, points, (255, 128, 0, 173),
                   clip_to_geometry=True, **options)
    # Alpha is the authored geometric coverage consumed by the GPU batch.
    np.testing.assert_array_equal(bounded[..., 3], full[..., 3])
    assert np.any(bounded[..., 3])


def test_bounded_native_polygon_and_disc_match_full_canvas_raster():
    from forge3d._map_scene_render import _draw_disc, _draw_polygon_fill
    rings = [[(-4.25, 5.5), (61.2, 8.75), (80.5, 67.25), (4.5, 72.5)],
             [(12.25, 17.75), (42.5, 19.5), (30.75, 51.25)]]
    for function, args in ((_draw_polygon_fill, (rings, (137, 83, 211, 128))),
                           (_draw_disc, (105.75, 1.5, (137, 83, 211, 128), 10.25)),
                           (_draw_disc, (-30.0, -20.0, (137, 83, 211, 128), 2.0))):
        full = np.zeros((83, 107, 4), dtype=np.uint8)
        bounded = full.copy()
        function(full, *args)
        function(bounded, *args, clip_to_geometry=True)
        np.testing.assert_array_equal(bounded, full)


def test_temporal_history_uses_jitter_and_changes_the_frame():
    require_gpu()
    model=scene(effects=("taa",),cartography=False,height_ao=False)
    with ms._shared_terrain_render_context():
        first=render(model)
        second=render(model)
    initial=first.metadata["native_pipeline"]; accumulated=second.metadata["native_pipeline"]
    assert initial["temporal_history_reset"] and not initial["temporal_history_used"]
    assert initial["raster_jitter_px"] == [0.0,0.0]
    assert not accumulated["temporal_history_reset"] and accumulated["temporal_history_used"]
    assert accumulated["raster_jitter_px"] != [0.0,0.0]
    assert np.any(first.rgba != second.rgba)


@pytest.mark.parametrize("change", ["sun","clouds","ibl","material"])
def test_temporal_history_resets_when_scene_radiance_changes(change):
    require_gpu()
    model=scene(effects=("taa","cloud_shadows"),cartography=False,height_ao=False)
    with ms._shared_terrain_render_context():
        render(model)
        assert render(model).metadata["native_pipeline"]["temporal_history_used"]
        settings=model.recipe.lighting.settings
        if change == "sun": model.recipe.lighting.sun_direction=(1.0,0.0,1.0)
        elif change == "clouds": settings["clouds"]["shadow_offset_x"] = 0.25
        elif change == "ibl":
            resources=ms._ACTIVE_TERRAIN_RESOURCES.holder[0]
            next(iter(resources._ibl_cache.values())).intensity=0.5
        else:
            ms._ACTIVE_TERRAIN_RESOURCES.holder[0].material_set=f3d.MaterialSet.custom((0.8,0.2,0.1),0.0,0.5)
        assert render(model).metadata["native_pipeline"]["temporal_history_reset"]


def test_opaque_cartographic_color_survives_output_transform():
    model=scene(effects=("taa",),height_ao=False)
    model.recipe.layers=[layer for layer in model.recipe.layers if isinstance(layer,f3d.VectorOverlay)]
    model.recipe.layers[0].width_px=8.0
    result,data=capture(model)
    opaque=data["vector_coverage"][...,3]==255
    assert opaque.any()
    np.testing.assert_array_equal(result.rgba[opaque,:3],np.broadcast_to([255,204,34],(opaque.sum(),3)))


@pytest.mark.parametrize("height_ao,fog,volume", itertools.product((False,True),repeat=3))
def test_height_atmosphere_and_froxel_stage_combinations(height_ao,fog,volume):
    model=scene(effects=("taa",),cartography=False,height_ao=height_ao,width=96,height=80)
    settings=model.recipe.lighting.settings
    settings["fog"]={"density":0.5 if fog else 0.0}
    settings["volumetrics"]={"enabled":volume,"density":0.005,"mode":"height"}
    result=render(model)
    pipeline=result.metadata["native_pipeline"]
    labels=[item["label"] for item in pipeline["execution_report"]["passes"]]
    assert ("concordia.ao" in labels)==height_ao
    assert ("concordia.atmosphere" in labels)==fog
    assert ("concordia.froxels" in labels)==volume
    assert pipeline["input_size"]==pipeline["output_size"]==[96,80]
    assert labels.index("concordia.geometry")<labels.index("concordia.temporal")<labels.index("concordia.output_transform")
    for name in ("concordia.ao","concordia.atmosphere","concordia.froxels"):
        if name in labels: assert labels.index("concordia.geometry")<labels.index(name)<labels.index("concordia.temporal")


def test_first_public_temporal_frame_is_reported_as_warmup(tmp_path):
    require_gpu()
    model=scene(effects=("taa",),cartography=False,height_ao=False)
    report=model.render(str(tmp_path/"warmup.png"))
    assert report.supported_features["mapscene.taa"]=="underdeveloped"
    assert not model.last_render_metadata["native_pipeline"]["temporal_history_used"]


def test_opaque_unlit_point_keeps_its_authored_display_color():
    model=scene(effects=("taa",),cartography=False,height_ao=False)
    model.recipe.layers=[f3d.PointCloudLayer(layer_id="color",crs="EPSG:32610",
        point_count=1,support_level="supported",metadata={"coordinate_space":"world",
        "positions":[[0.0,0.0,15.0]],"color":[255,204,34,255],"point_size":18.0})]
    result,data=capture(model)
    assert np.any(np.all(result.rgba[...,:3]==[255,204,34],axis=-1))
    assert np.any(data["point_depths"][0]<1.0)


def test_shared_screen_projection_is_additive_and_depth_invertible():
    from forge3d._map_scene_projection import TerrainProjector
    model=scene(effects=("taa",),cartography=False,height_ao=False,width=128,height=96)
    model.recipe.lighting.settings["camera"]={"camera_mode":"screen"}
    projector=TerrainProjector(model.recipe)
    params=projector.params
    ground=projector.height((0.5,0.5))
    points=[[0.5,0.5,ground,0.0],[0.5,0.5,ground,1.0]]
    assert params.project_terrain_points(points)==params.project_terrain_points(points,shared_frame=False)
    projected=params.project_terrain_points(points,shared_frame=True)
    np.testing.assert_array_equal(projected[0][:2],projected[1][:2])
    assert projected[0][2]!=projected[1][2]
    depth=params.project_terrain_depth(projector.heightmap,(128,96),shared_frame=True)
    uv=params.unproject_terrain_depth(depth,shared_frame=True)
    yy,xx=np.indices(depth.shape)
    expected=np.stack(((xx+.5)/128,1-(yy+.5)/96),axis=-1)
    hit=depth<1.0
    assert hit.any()
    np.testing.assert_allclose(uv[hit],expected[hit],atol=np.finfo(np.float32).eps,rtol=0)


def test_screen_landmark_uses_gpu_cartography_and_preserves_authored_pixels(tmp_path,monkeypatch):
    from test_recipe_goldens import _textured_gltf_landmark
    model=_textured_gltf_landmark(tmp_path)
    expected,drawn,_=ms._composite_textured_landmark_layers(
        np.zeros((88,128,4),dtype=np.uint8),model.recipe)
    assert drawn and np.any(expected[...,3])
    def forbidden(*args,**kwargs):
        raise AssertionError("native landmark called a post-readback CPU compositor")
    monkeypatch.setattr(ms,"_composite_textured_landmark_layers",forbidden)
    result,data=capture(model)
    np.testing.assert_array_equal(data["vector_coverage"],expected)
    assert result.metadata["gltf_textured_backend"]=="native_cartographic_landmark"
    assert result.metadata["gltf_layer_stages"]=={"textured-landmark":"post_resolve_cartographic"}
    assert "concordia.vectors" in [p["label"] for p in result.metadata["native_pipeline"]["execution_report"]["passes"]]
    report=model.render()
    assert report.supported_features["gltf.textured_mapscene_render"]=="supported"
    assert report.supported_features["buildings.textured_pbr"]=="underdeveloped"


def test_world_textured_geometry_still_requires_shared_uv_material_stream(tmp_path):
    from test_recipe_goldens import _textured_gltf_landmark
    model=_textured_gltf_landmark(tmp_path)
    model.recipe.layers[0].features=[{"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]},"properties":{"height":3}}]
    with pytest.raises(f3d.MapSceneNativeUnavailable,match="shared UV/material/velocity"):
        render(model)


def test_default_screen_landmark_rectangle_remains_renderable(tmp_path):
    from test_recipe_goldens import _textured_gltf_landmark
    model=_textured_gltf_landmark(tmp_path)
    model.recipe.layers[0].metadata.pop("screen_rect")
    expected,drawn,_=ms._composite_textured_landmark_layers(np.zeros((88,128,4),dtype=np.uint8),model.recipe)
    assert drawn
    result,data=capture(model)
    np.testing.assert_array_equal(data["vector_coverage"],expected)
    assert result.metadata["gltf_layer_stages"]["textured-landmark"]=="post_resolve_cartographic"
