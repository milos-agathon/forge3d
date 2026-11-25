"""Compatibility regressions for the C01 same-camera material compositor."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
from forge3d import map_scene as ms
from test_c01_review_gpu import _scene


@pytest.mark.parametrize("settings", ({"samples": 4}, {"hdr": True},
                                      {"denoiser": "atrous"},
                                      {"width": 32, "height": 32, "samples": 4, "hdr": True}),
                         ids=("accumulation", "hdr", "denoise", "small-output"))
def test_world_vectors_keep_offline_output_paths(tmp_path, settings, monkeypatch):
    exported = []
    original_writer = ms._write_exr_array
    def write_exr(path, array, **kwargs):
        exported.append(np.asarray(array).copy())
        return original_writer(path, array, **kwargs)
    monkeypatch.setattr(ms, "_write_exr_array", write_exr)
    grid, recipe = _scene()
    material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer("ground", (150, 140, 130)),),
                                       presentation="physical")
    recipe = material.bind(recipe, grid)
    recipe = replace(recipe, camera=replace(recipe.camera, distance=110, elevation_deg=40, azimuth_deg=140),
        lighting=replace(recipe.lighting, settings={"cli_params": {"camera_mode": "mesh:zup:north"}}))
    ring = [[500001, 4599999], [500062, 4599999], [500062, 4599938],
            [500001, 4599938], [500001, 4599999]]
    vector = f3d.VectorOverlay("paint", crs=grid.crs,
        metadata={"coordinate_space": "world"},
        features=[{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}],
        style={"layers": [{"type": "fill", "paint": {"fill-color": "#283c14"}}]})
    paint = f3d.GeospatialMaterialRecipe.land_cover(
        f3d.CategoricalRasterStyle({1: (40, 60, 20, 255)}, {1: "protected vector"}), albedo=(150, 140, 130))
    vector = paint.bind_vector(vector, [[500032, 4599968]], units="m", classes=[1])
    recipe = replace(recipe, layers=(vector,), map_furniture=None,
                     output=replace(recipe.output, **settings))
    scene = f3d.MapScene(recipe=recipe)
    assert not scene.validate().render_blocked(), [d.message for d in scene.validate().diagnostics]
    scene.render(tmp_path / "offline.png")
    assert scene.last_render_backend == "gpu_terrain"
    pixels = np.asarray(Image.open(tmp_path / "offline.png"))
    assert np.any(np.all(pixels == (40, 60, 20, 255), axis=-1))
    if settings.get("hdr"):
        from forge3d.terrain_materials import _linear
        mask = np.all(pixels == (40, 60, 20, 255), axis=-1)
        expected = _linear(np.array((40, 60, 20), dtype=np.float32) / 255)
        assert len(exported) == 1 and (tmp_path / "offline.exr").is_file()
        np.testing.assert_array_equal(exported[0][mask, :3], np.tile(expected, (mask.sum(), 1)))
        assert "linear sRGB" in scene.last_render_metadata["hdr_overlay_encoding"]


@pytest.mark.parametrize("settings", ({}, {"samples": 4, "hdr": True, "denoiser": "atrous"}),
                         ids=("display", "offline-hdr"))
def test_building_beauty_is_independent_of_aov_route(settings):
    grid, recipe = _scene()
    material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer("ground", (150, 140, 130)),),
                                       presentation="physical")
    recipe = material.bind(recipe, grid)
    recipe = replace(recipe, camera=replace(recipe.camera, distance=110,
                     elevation_deg=40, azimuth_deg=140))
    ring = [[500020, 4599980], [500044, 4599980], [500044, 4599956],
            [500020, 4599956], [500020, 4599980]]
    buildings = f3d.MapSceneBuildingLayer("buildings", metadata={"crs": grid.crs},
        features=[{"type": "Feature", "properties": {"height": 25},
                   "geometry": {"type": "Polygon", "coordinates": [ring]}}])
    recipe = replace(recipe, layers=(buildings,), output=replace(recipe.output, **settings))
    vector = f3d.VectorOverlay("visibility", crs=grid.crs,
        features=[{"type": "Feature", "geometry": {"type": "LineString",
                   "coordinates": [[500020, 4599980], [500044, 4599980]]}}])
    with ms._shared_terrain_render_context():
        ordinary = ms._render_terrain_renderer_result(recipe, recipe.terrain.data)
        captured = ms._render_terrain_renderer_result(
            replace(recipe, layers=(buildings, vector), output=replace(recipe.output, aovs=("depth",))),
            recipe.terrain.data)
        empty = ms._render_terrain_renderer_result(replace(recipe, layers=()), recipe.terrain.data)
    assert ordinary is not None and captured is not None
    assert empty is not None
    assert ordinary.metadata["building_scatter_stats"]["visible_instances"] > 0
    assert np.any(ordinary.rgba != empty.rgba), "Buildings must contribute visible beauty pixels"
    np.testing.assert_array_equal(ordinary.rgba, captured.rgba)


@pytest.mark.parametrize("axis", ("north", "east"))
@pytest.mark.parametrize("sun_sign", (1, -1), ids=("toward", "away"))
def test_normal_map_matches_real_tilted_geometry(tmp_path, monkeypatch, axis, sun_sign):
    from forge3d._map_scene_projection import frame_visibility
    captures = []
    original = ms._render_terrain_renderer_result
    def observe(recipe, *args, **kwargs):
        result = original(recipe, *args, **kwargs)
        captures.append((np.asarray(result.aov_frame.normal()),
                         frame_visibility(recipe, result.aov_frame)[2]))
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    grid, base = _scene()
    sun = (0, .6, sun_sign * .8) if axis == "north" else (sun_sign * .8, .6, 0)
    base = replace(base, camera=replace(base.camera, distance=110, elevation_deg=35),
                   lighting=replace(base.lighting, sun_direction=sun,
                       settings={"cli_params": {"camera_mode": "mesh:zup:north"},
                           "renderer_config": {"shadows": {"enabled": False}},
                           "height_ao": {"enabled": False}, "sun_visibility": {"enabled": False},
                           "fog": {"density": 0}, "pom": False}),
                   output=replace(base.output, aovs=("normal",)))
    angle = np.radians(20)
    vector = np.array((0, np.sin(angle), np.cos(angle)) if axis == "north"
                      else (np.sin(angle), 0, np.cos(angle)))
    rows, cols = np.mgrid[:64, :64]
    # Native terrain spans 64 metres between endpoint samples. Match the
    # geometry's actual world slope rather than assuming sample spacing here.
    geometry = (rows if axis == "north" else -cols) * (64 / 63) * np.tan(angle)
    frames, normals, masks = [], [], []
    for name, dem, normal in (("flat", np.zeros((64, 64)), (0, 0, 1)),
                              ("mapped", np.zeros((64, 64)), vector),
                              ("geometry", geometry, (0, 0, 1))):
        path = tmp_path / (name + "-normal.png")
        encoded = np.rint((np.asarray(normal) + 1) * 127.5).astype(np.uint8)
        Image.fromarray(np.tile(encoded, (2, 2, 1))).save(path)
        texture = f3d.MaterialTexture(str(path), (16, 16), "linear", "normal",
                                     normal_convention="opengl")
        material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer(
            "ground", (150, 150, 150), normal=texture, roughness=1),), presentation="physical")
        recipe = material.bind(replace(base, terrain=replace(base.terrain, data=dem.astype(np.float32))), grid)
        scene = f3d.MapScene(recipe=replace(recipe, map_furniture=None))
        scene.render(tmp_path / (name + ".png"))
        frames.append(np.asarray(Image.open(tmp_path / (name + ".png"))).astype(float))
        normals.append(captures[-1][0])
        masks.append(captures[-1][1])
    common = np.logical_and.reduce(masks)
    assert common.any()
    mapped = np.mean(normals[1][common], axis=0)
    tilted = np.mean(normals[2][common], axis=0)
    # Both maps are quantized uint8 direction data. Two encodings plus the
    # half-float AOV's unit-scale epsilon bound the component difference.
    tolerance = 4 / 255 + np.finfo(np.float16).eps
    np.testing.assert_allclose(mapped, tilted, atol=tolerance, rtol=0)
    delta = float(np.mean(frames[1][common, :3] - frames[0][common, :3]))
    geometry_delta = float(np.mean(frames[2][common, :3] - frames[0][common, :3]))
    print(axis, "normal", mapped, "geometry", tilted, "delta", delta, "geometry_delta", geometry_delta)
    assert abs(delta) > 1, "Declared normal must exceed the output's one-code rounding floor"
    assert delta * geometry_delta > 0, "Mapped and real geometric tilt must respond in the same direction"
    assert delta * sun_sign > 0, "The declared ENU tilt must face its corresponding sun"
    if os.environ.get("FORGE3D_C01_ARTIFACT_DIR"):
        destination = Path(os.environ["FORGE3D_C01_ARTIFACT_DIR"]) / f"normal-{axis}-{sun_sign}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("flat", "mapped", "geometry"):
            shutil.copyfile(tmp_path / (name + ".png"), destination / (name + ".png"))
        (destination / "measurement.json").write_text(json.dumps({
            "normal_aov_mean": mapped.tolist(), "geometry_aov_mean": tilted.tolist(),
            "mapped_display_delta": delta, "geometry_display_delta": geometry_delta,
            "component_tolerance_basis": "two uint8 direction encodings plus float16 epsilon",
            "component_tolerance": float(tolerance), "camera_polar_degrees": 35,
            "shadow_ao_pom_disabled_for_directional_comparison": True}, indent=2))


def test_original_normal_probe_crop_dilutes_terrain_with_sky(tmp_path, monkeypatch):
    """Measure coverage instead of treating a mixed sky crop as surface response."""
    from forge3d._map_scene_projection import frame_visibility
    captured = []
    original = ms._render_terrain_renderer_result
    def observe(recipe, *args, **kwargs):
        result = original(recipe, *args, **kwargs)
        captured.append(frame_visibility(recipe, result.aov_frame)[2])
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    _, base = _scene()
    base = replace(base, camera=replace(base.camera, distance=260, elevation_deg=89),
        lighting=replace(base.lighting, sun_direction=(0, .6, .8),
                         settings={"cli_params": {"camera_mode": "mesh:zup:north"}}),
        output=replace(base.output, width=192, height=192, aovs=("depth",)))
    frames = []
    tilt = np.radians(20)
    for name, normal in (("flat", (0, 0, 1)), ("tilted", (0, np.sin(tilt), np.cos(tilt)))):
        path = tmp_path / (name + "-normal.png")
        Image.fromarray(np.tile(np.rint((np.asarray(normal) + 1) * 127.5).astype(np.uint8), (2, 2, 1))).save(path)
        texture = f3d.MaterialTexture(str(path), (16, 16), "linear", "normal", normal_convention="opengl")
        material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer(
            "ground", (150, 150, 150), normal=texture, roughness=1),), presentation="physical")
        scene = f3d.MapScene(recipe=replace(material.bind(base, grid), map_furniture=None))
        scene.render(tmp_path / (name + ".png"))
        frames.append(np.asarray(Image.open(tmp_path / (name + ".png"))).astype(float)[..., :3])
    np.testing.assert_array_equal(captured[0], captured[1])
    np.testing.assert_array_equal(frames[0][~captured[0]], frames[1][~captured[0]])
    coverage = captured[0][72:120, 72:120]
    delta = (frames[1] - frames[0])[72:120, 72:120]
    assert coverage.any() and not coverage.all()
    visible_delta, crop_delta = float(delta[coverage].mean()), float(delta.mean())
    assert visible_delta > 1
    assert np.isclose(crop_delta, visible_delta * coverage.mean())
    print("original probe crop", "terrain_fraction", coverage.mean(),
          "visible_delta", visible_delta, "whole_crop_delta", crop_delta)


@pytest.mark.parametrize("settings", ({"samples": 4},
    {"samples": 4, "hdr": True, "denoiser": "atrous"},
    {"samples": 4, "format": "exr", "denoiser": "atrous"}),
    ids=("accumulated-display", "hdr-denoised", "exr-denoised"))
def test_protected_terrain_survives_offline_display_and_linear_export(tmp_path, monkeypatch, settings):
    from test_c01_review_gpu import _capture_visibility
    from forge3d.terrain_materials import _linear
    captures = _capture_visibility(monkeypatch)
    exported = []
    writer = ms._write_exr_array
    def write_exr(path, array, **kwargs):
        exported.append(np.asarray(array).copy())
        return writer(path, array, **kwargs)
    monkeypatch.setattr(ms, "_write_exr_array", write_exr)
    grid, base = _scene()
    swatch = (40, 60, 20, 255)
    material = f3d.TerrainMaterialRecipe.land_cover(
        f3d.CategoricalRasterStyle({3: swatch}, {3: "protected terrain"}), albedo=(150, 140, 130))
    base = replace(base, output=replace(base.output, **settings))
    scene = f3d.MapScene(recipe=replace(material.bind(base, grid,
        classes=np.full((64, 64), 3)), map_furniture=None))
    exr = settings.get("format") == "exr"
    path = tmp_path / ("protected.exr" if exr else "protected.png")
    scene.render(path)
    terrain = captures[-1][2]
    assert terrain.any() and path.is_file()
    if not exr:
        pixels = np.asarray(Image.open(path))
        np.testing.assert_array_equal(pixels[terrain], np.tile(swatch, (terrain.sum(), 1)))
    if exr or settings.get("hdr"):
        assert len(exported) == 1
        expected = _linear(np.array(swatch[:3], dtype=np.float32) / 255)
        np.testing.assert_array_equal(exported[0][terrain, :3], np.tile(expected, (terrain.sum(), 1)))
        if os.environ.get("FORGE3D_C01_ARTIFACT_DIR"):
            destination = Path(os.environ["FORGE3D_C01_ARTIFACT_DIR"]) / ("protected-exr" if exr else "protected-hdr")
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path if exr else path.with_suffix(".exr"), destination / "beauty.exr")
            np.save(destination / "expected.npy", exported[0])
            np.save(destination / "terrain.npy", terrain)


def test_shared_renderer_restores_fullscreen_wind_after_3d_depth_mode():
    import test_tv22_scatter_wind as wind_tests
    renderer, materials, ibl, params, heights = wind_tests._make_gpu_fixtures()
    source = wind_tests._make_scatter_source()
    batch = wind_tests.ts.TerrainScatterBatch(
        levels=[wind_tests.ts.TerrainScatterLevel(mesh=wind_tests._scatter_mesh())],
        transforms=wind_tests._place_instances(source),
        wind=wind_tests.ts.ScatterWindSettings(enabled=True, amplitude=2.0, speed=1.0))
    before = wind_tests._render_frame(renderer, materials, ibl, params, heights, [batch], time_seconds=0)
    mesh = f3d.TerrainRenderParams(replace(params.python_object, camera_mode="mesh:zup:north"))
    wind_tests._render_frame(renderer, materials, ibl, mesh, heights, [batch], time_seconds=0)
    after = wind_tests._render_frame(renderer, materials, ibl, params, heights, [batch], time_seconds=0)
    animated = wind_tests._render_frame(renderer, materials, ibl, params, heights, [batch], time_seconds=.37)
    np.testing.assert_array_equal(before, after)
    assert not np.array_equal(after, animated)
