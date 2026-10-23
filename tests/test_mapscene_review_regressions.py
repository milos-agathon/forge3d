"""Reproductions of the node-45 architecture review's concrete failures."""
from __future__ import annotations

import copy
import numpy as np
import pytest
import forge3d as f3d
import forge3d.map_scene as ms
from _concordia_scene import capture, render, require_gpu, scene


def test_polygon_parts_keep_layer_default_colors_and_input_geometry():
    from forge3d._map_scene_native_inputs import vector_batches
    from forge3d._map_scene_projection import project_vector_recipe
    model = scene(effects=("taa",), cartography=False, height_ao=False)
    model.recipe.layers = [f3d.VectorOverlay(layer_id="default-style", crs="EPSG:32610",
        features=[{"geometry": {"type": "Polygon", "coordinates": [[
            [-12., -12.], [-2., -12.], [-2., -2.], [-12., -2.], [-12., -12.]]]}}])]
    projected = project_vector_recipe(model.recipe)
    before = copy.deepcopy(projected.layers[0].features)
    payload = copy.deepcopy(projected.layers[0].to_dict())
    for feature in payload["features"]: feature.pop("_terrain_draped", None)
    expected = np.asarray(ms._render_rgb(payload, salt="vector"))
    batches = vector_batches(model.recipe, projected)
    assert len(batches) == 2
    assert projected.layers[0].features == before
    for batch in batches:
        alpha = batch["rgba"][..., 3]
        rgb = batch["rgba"][alpha == alpha.max(), :3]
        assert len(rgb)
        # Fully covered samples avoid repeated AA-edge quantization. CPU
        # UNORM conversion truncates after f32 division: one output step.
        np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), atol=1, rtol=0)


@pytest.mark.parametrize("kind", ["Polygon", "LineString"])
@pytest.mark.parametrize("terrain", ["flat", "relief"])
def test_draped_cartography_has_no_self_occlusion_holes(kind, terrain):
    from forge3d._map_scene_native_inputs import vector_batches
    from forge3d._map_scene_projection import project_vector_recipe
    model = scene(effects=("taa",), cartography=False, height_ao=False)
    if terrain == "flat": model.recipe.terrain.data[:] = 0.0
    coordinates = ([[-12., -12.], [-2., -12.], [-2., -2.], [-12., -2.], [-12., -12.]]
                   if kind == "Polygon" else [[-12., -12.], [-2., -12.]])
    model.recipe.layers = [f3d.VectorOverlay(
        layer_id="drape", crs="EPSG:32610", width_px=8.0,
        style={"version": 8, "layers": [{"id": "drape", "type": "fill" if kind == "Polygon" else "line",
            "paint": {"fill-color": "#ffcc22", "fill-opacity": 1.0, "line-color": "#ffcc22",
                      "line-opacity": 1.0}}] + ([{"id": "outline", "type": "line",
                          "paint": {"line-opacity": 0.0}}] if kind == "Polygon" else [])},
        features=[{"geometry": {"type": kind, "coordinates": [coordinates] if kind == "Polygon" else coordinates}}])]
    projected = project_vector_recipe(model.recipe)
    batches = vector_batches(model.recipe, projected)
    authored = batches[0]["rgba"][..., 3] == 255
    if terrain == "flat":
        raw, _ = ms._composite_python_precise_vector_layers(
            np.zeros((192, 256, 4), np.uint8), projected, projected.layers, clip_to_geometry=True)
        # A flat unobstructed plane cannot lose coverage during input authoring.
        np.testing.assert_array_equal(authored, raw[..., 3] == 255)
    assert authored.any()
    _, data = capture(model)
    assert np.all(data["surface_kind"][authored] == 1)
    np.testing.assert_array_equal(data["vector_coverage"][authored, 3], 255)
    np.testing.assert_array_equal(data["vector_coverage"][authored, :3],
                                  np.broadcast_to([255, 204, 34], (authored.sum(), 3)))


@pytest.mark.parametrize("camera", ["screen", "mesh:zup"])
def test_label_decisions_are_consistent_across_interactive_effects(camera):
    baseline = scene(effects=("taa",), height_ao=False)
    baseline.recipe.lighting.settings["camera"] = {"camera_mode": camera}
    expected = baseline.compile_plan().label_plans["labels"].to_dict()
    for effect in ("taa", "ssao", "ssgi", "ssr", "cloud_shadows"):
        enabled = scene(effects=(), height_ao=False)
        enabled.recipe.lighting.settings["camera"] = {"camera_mode": camera}
        settings = enabled.recipe.lighting.settings
        if effect == "cloud_shadows":
            settings["clouds"].update(enabled=True, shadows_enabled=True)
        else:
            settings["screen_space"][effect]["enabled"] = True
        assert enabled.compile_plan().label_plans["labels"].to_dict() == expected


@pytest.mark.parametrize("camera", ["screen", "mesh:zup"])
def test_effects_off_preserves_legacy_label_projection_and_occlusion(camera):
    from forge3d._map_scene_projection import TerrainProjector
    model = scene(effects=(), height_ao=False)
    model.recipe.lighting.settings["camera"] = {"camera_mode": camera}
    assert not TerrainProjector(model.recipe).shared_frame
    labels = model.compile_plan().label_plans["labels"]
    if camera == "screen":
        assert not labels.accepted
        assert any(item.reason == "outside_view" for item in labels.rejected)
    else:
        assert "behind-building" in [item.label_id for item in labels.accepted]


@pytest.mark.parametrize("size", [(48, 80), (80, 48), (48, 48)])
@pytest.mark.parametrize("effects", [("taa",), ("ssao",), ("ssgi",), ("ssr",), ("cloud_shadows",)])
def test_small_interactive_viewport_has_structured_diagnostic(size, effects):
    require_gpu()
    model = scene(effects=effects, cartography=False, height_ao=False)
    model.recipe.output.width, model.recipe.output.height = size
    with pytest.raises(f3d.MapSceneNativeUnavailable, match="at least 64x64") as error:
        render(model)
    assert f"{size[0]}x{size[1]}" in str(error.value)


def test_small_effects_off_viewport_still_renders():
    require_gpu()
    model = scene(effects=(), cartography=False, height_ao=False)
    model.recipe.output.width, model.recipe.output.height = 48, 32
    assert render(model).rgba.shape == (32, 48, 4)


def test_draped_fill_is_still_hidden_by_buildings():
    model = scene(effects=("taa",), cartography=False, height_ao=False)
    model.recipe.layers = [*model.recipe.layers, f3d.VectorOverlay(
        layer_id="fill", crs="EPSG:32610", width_px=1.0,
        style={"version": 8, "layers": [{"id": "fill", "type": "fill",
            "paint": {"fill-color": "#ffcc22"}}]},
        features=[{"geometry": {"type": "Polygon", "coordinates": [[
            [-14., -14.], [14., -14.], [14., 14.], [-14., 14.], [-14., -14.]]]}}])]
    _, data = capture(model)
    buildings = data["surface_kind"] == 2
    assert buildings.any() and data["vector_coverage"][..., 3].any()
    assert not data["vector_coverage"][buildings, 3].any()


@pytest.mark.parametrize("change", ["fog", "sky", "volumetrics", "water", "raster"])
def test_temporal_history_resets_for_all_pre_resolve_inputs(change, monkeypatch):
    require_gpu()
    model = scene(effects=("taa",), cartography=False, height_ao=False)
    settings = model.recipe.lighting.settings
    settings["fog"] = {"density": 0.2}
    settings["sky"] = {"enabled": True, "model": "preetham", "turbidity": 2.0}
    settings["volumetrics"] = {"enabled": True, "density": 0.005, "mode": "height"}
    mask = np.zeros((32, 32), np.float32)
    monkeypatch.setattr(ms, "_mapscene_water_mask", lambda *_args: mask)
    raster = np.zeros((32, 32, 4), np.uint8)
    raster[..., 0] = 255
    raster[..., 3] = 255
    # Raster inputs enter through the existing public native OverlayLayer seam.
    # Rebuilding identical LUT bytes must not reset an otherwise stable frame.
    constructor = f3d.TerrainRenderParams
    def with_raster(config):
        config.overlays = [f3d.OverlayLayer.from_raster_rgba(raster)]
        return constructor(config)
    monkeypatch.setattr(f3d, "TerrainRenderParams", with_raster)
    with ms._shared_terrain_render_context():
        render(model)
        assert render(model).metadata["native_pipeline"]["temporal_history_used"]
        if change == "fog": settings["fog"]["density"] = 0.9
        elif change == "sky": settings["sky"]["turbidity"] = 4.0
        elif change == "volumetrics": settings["volumetrics"]["density"] = 0.02
        elif change == "water": mask[:] = 1.0
        else: raster[..., :3] = [20, 80, 220]
        pipeline = render(model).metadata["native_pipeline"]
        assert pipeline["temporal_history_reset"] and not pipeline["temporal_history_used"]
        assert render(model).metadata["native_pipeline"]["temporal_history_used"]


def test_temporal_render_bypasses_image_cache_and_keeps_support_report(tmp_path):
    require_gpu()
    model = scene(effects=("taa",), cartography=False, height_ao=False)
    with ms._shared_terrain_render_context():
        first = model.render(str(tmp_path / "fresh.png"), cache=tmp_path / "cache")
        warmed = model.render(str(tmp_path / "warm.png"), cache=tmp_path / "cache")
    assert first.supported_features["mapscene.taa"] == "underdeveloped"
    assert warmed.supported_features["mapscene.taa"] == "supported"
    assert model.last_render_metadata["native_pipeline"]["temporal_history_used"]
    assert (tmp_path / "fresh.png").read_bytes() != (tmp_path / "warm.png").read_bytes()
    assert not (tmp_path / "cache").exists()


def test_interactive_msaa_request_is_diagnosed(monkeypatch):
    require_gpu()
    constructor = f3d.TerrainRenderParams
    def multisample(config):
        config.msaa_samples = 4
        return constructor(config)
    monkeypatch.setattr(f3d, "TerrainRenderParams", multisample)
    with pytest.raises(f3d.MapSceneNativeUnavailable, match="msaa_samples=1"):
        render(scene(effects=("taa",), cartography=False, height_ao=False))


@pytest.mark.parametrize("effects", [(), ("taa",)])
def test_screen_image_landmark_never_claims_textured_pbr(effects, tmp_path):
    from test_recipe_goldens import _textured_gltf_landmark
    require_gpu()
    model = _textured_gltf_landmark(tmp_path)
    model.recipe.lighting.settings["screen_space"] = {"taa": {"enabled": "taa" in effects}}
    # Clear every other route trigger so the first case exercises legacy output.
    model.recipe.lighting.settings["clouds"] = {"enabled": False}
    report = model.render(str(tmp_path / "landmark.png"))
    assert report.supported_features["gltf.textured_mapscene_render"] == "supported"
    assert report.supported_features["buildings.textured_pbr"] == "underdeveloped"


def test_shared_scene_cloud_shader_runs_on_gpu():
    require_gpu()
    model = f3d.Scene(96, 80)
    model.enable_cloud_shadows("low")
    model.set_cloud_coverage(0.5)
    model.set_cloud_density(0.5)
    model.set_cloud_shadow_intensity(0.5)
    image = model.render_rgba()
    assert image.shape == (80, 96, 4) and np.isfinite(image).all()
    assert model.is_cloud_shadows_enabled()
    from forge3d import _forge3d as native
    import json
    report = json.loads(native.render_execution_report())
    assert "cloud_shadow_compute_shader" in json.dumps(report)
