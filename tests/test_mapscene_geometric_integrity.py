"""SUTURA geometric integrity: 3D MapScene draws things in the right place.

SUTURA guarantees that every layer is drawn for real or blocks. These checks
add the missing half: in the 3D (``mesh:zup``) camera modes the map is
north-up and not mirrored, the sun lights the flank that faces it,
``camera.target`` moves the view, declared lighting settings are validated,
and terrain-UV rasters, vectors and world Point labels follow that camera.
Unsupported world inputs block before pixels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
import forge3d.map_scene as map_scene

from _sutura_recipes import _write_raster_overlay
from _terrain_runtime import terrain_rendering_available

N = 256
_ZUP = {"camera_mode": "mesh:zup"}


def _gaussian(cx: float, cy: float, sigma: float, height: float) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    return (height * np.exp(-(((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))).astype(np.float32)


def _compass_sun(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """MapScene sun_direction: compass azimuth 0 = +Z (north), 90 = +X (east)."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e))


def _top_down_north_up() -> "f3d.OrbitCamera":
    # Geographic frame: eye a little south of the target, looking north and down.
    return f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=420.0, azimuth_deg=-90.0, elevation_deg=0.5, fov_deg=40.0)


def _scene(dem: np.ndarray, *, camera=None, sun=(0.0, 0.8, 0.6), settings=None, layers=(), camera_mode="mesh:zup") -> "f3d.MapScene":
    lighting = {"colormap": "9a9a9a,9a9a9a", "hue_variation_strength": 0.0, "cli_params": {"camera_mode": camera_mode}}
    lighting.update(settings or {})
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=dem,
            crs="EPSG:32633",
            metadata={"width": N, "height": N, "resolution": [1.0, 1.0], "source_id": "geometric-integrity"},
            elevation_sampling_available=True,
        ),
        camera=camera or _top_down_north_up(),
        lighting=f3d.LightingPreset(name="outdoorsun", sun_direction=sun, intensity=1.0, settings=lighting),
        output=f3d.OutputSpec(width=256, height=256, format="png", samples=1),
        layers=list(layers),
    )


def _render_luma(scene: "f3d.MapScene", path: Path) -> np.ndarray:
    scene.render(str(path))
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _feature_position(luma: np.ndarray) -> tuple[int, int]:
    """Strongest shading deviation inside the terrain footprint (not the background)."""
    background = luma[0, 0]
    terrain = np.abs(luma - background) > 4.0
    assert terrain.any(), "no terrain pixels rendered"
    reference = np.median(luma[terrain])
    contrast = np.where(terrain, np.abs(luma - reference), 0.0)
    row, col = np.unravel_index(np.argmax(contrast), contrast.shape)
    return int(row), int(col)


requires_gpu = pytest.mark.skipif(
    not terrain_rendering_available(), reason="needs a terrain-capable GPU adapter"
)


# ---------------------------------------------------------------- no GPU


def test_mapscene_adds_north_option_to_zup_modes_only():
    assert map_scene._mapscene_north_up_camera_mode("mesh:zup") == "mesh:zup:north"
    assert map_scene._mapscene_north_up_camera_mode("mesh:zup:north") == "mesh:zup:north"
    assert map_scene._mapscene_north_up_camera_mode("clipmap:7:32:32:10:0.3:zup").endswith(":zup:north")
    assert map_scene._mapscene_north_up_camera_mode("screen") == "screen"
    assert map_scene._mapscene_north_up_camera_mode("mesh") == "mesh"


def test_lighting_extras_are_validated_not_dropped():
    extras = map_scene._mapscene_lighting_extras({"sky": {"enabled": True, "aerial_density": 0.002}})
    assert type(extras["sky"]).__name__ == "SkySettings"
    with pytest.raises(ValueError, match="sky"):
        map_scene._mapscene_lighting_extras({"sky": {"enabld": True}})
    with pytest.raises(TypeError, match="height_ao"):
        map_scene._mapscene_lighting_extras({"height_ao": True})


def test_ibl_intensity_requires_ibl_gi_mode():
    assert map_scene._mapscene_ibl_intensity({"intensity": 0.3}, ["ibl", "ssao"]) == pytest.approx(0.3)
    assert map_scene._mapscene_ibl_intensity({}, []) == 1.0
    with pytest.raises(ValueError, match="gi.modes includes 'ibl'"):
        map_scene._mapscene_ibl_intensity({"intensity": 0.3}, ["ssao"])


def test_camera_target_must_be_finite():
    scene = _scene(np.zeros((N, N), dtype=np.float32), camera=f3d.OrbitCamera(target=(float("nan"), 0.0, 0.0)))
    with pytest.raises(ValueError, match="camera.target"):
        map_scene._mapscene_camera_target(scene.recipe)


@pytest.mark.parametrize("layer_kind", ["raster", "vector"])
def test_screen_space_layers_block_in_3d_modes(tmp_path, monkeypatch, layer_kind):
    def fake_terrain(_recipe, _heightmap, **_kwargs):
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    if layer_kind == "raster":
        layer = f3d.RasterOverlay(layer_id="ortho", path=str(_write_raster_overlay(tmp_path)), crs="EPSG:32633",
                                  metadata={"width": 8, "height": 8, "source_id": "overlay"})
        required = "terrain-UV raster drape"
    else:
        layer = f3d.VectorOverlay(layer_id="roads", features=[{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 10]]}, "properties": {}}],
                                  crs="EPSG:32633")
        required = "vector projection through the terrain camera"
    scene = _scene(np.zeros((N, N), dtype=np.float32), layers=[layer])
    out = tmp_path / "blocked.png"
    with pytest.raises(map_scene.MapSceneNativeUnavailable) as excinfo:
        scene.render(str(out))
    blocks = [b for b in excinfo.value.diagnostics if b.get("layer") == layer.layer_id]
    assert blocks and required in blocks[0]["required_native"], excinfo.value.diagnostics
    assert not out.exists(), "a blocked render must not write pixels"


def test_chronos_camera_hash_covers_target():
    if not hasattr(f3d, "compile_frame"):
        pytest.skip("CHRONOS native surface unavailable")
    from forge3d.chronos import _camera_json, _scene_json

    dem = np.zeros((N, N), dtype=np.float32)
    a = _scene(dem, camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=300.0))
    b = _scene(dem, camera=f3d.OrbitCamera(target=(50.0, 0.0, 0.0), distance=300.0))
    hash_a = json.loads(f3d.compile_frame(0, 1, 1, _camera_json(a), _scene_json(a, [])).to_json())["camera_hash"]
    hash_b = json.loads(f3d.compile_frame(0, 1, 1, _camera_json(b), _scene_json(b, [])).to_json())["camera_hash"]
    assert hash_a != hash_b


# ---------------------------------------------------------------- GPU


@requires_gpu
def test_3d_view_is_north_up_and_not_mirrored(tmp_path):
    # Row 0 is north and column 0 is west: one peak in the north-east quadrant.
    luma = _render_luma(_scene(_gaussian(200, 56, 20, 100.0)), tmp_path / "ne_peak.png")
    row, col = _feature_position(luma)
    assert row < N // 2 and col >= N // 2, f"north-east peak rendered at row {row}, col {col}"


@requires_gpu
@pytest.mark.parametrize("facing, sun_from, sun_opposite", [
    ("west", 270.0, 90.0), ("east", 90.0, 270.0), ("north", 0.0, 180.0), ("south", 180.0, 0.0),
])
def test_2d_sun_lights_the_slope_facing_it(tmp_path, facing, sun_from, sun_opposite):
    """Default 2D map: a uniform slope is brighter under the sun it faces.

    A plane needs no framing assumptions (the 2D view crops the DEM), so this
    measures the shading frame alone. Rows run north -> south, columns west -> east.
    """
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    rises_to = {"west": 0.6 * xx, "east": 0.6 * (N - xx), "north": 0.6 * yy, "south": 0.6 * (N - yy)}
    dem = rises_to[facing].astype(np.float32)

    def mean_luma(azimuth: float) -> float:
        scene = _scene(dem, sun=_compass_sun(azimuth, 30.0), camera_mode="screen",
                       camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=300.0, azimuth_deg=0.0, elevation_deg=90.0))
        return float(_render_luma(scene, tmp_path / f"plane_{facing}_{int(azimuth)}.png")[64:192, 64:192].mean())

    lit, unlit = mean_luma(sun_from), mean_luma(sun_opposite)
    assert lit > unlit + 10.0, f"{facing}-facing slope: sun it faces {lit:.1f}, opposite sun {unlit:.1f}"


@requires_gpu
@pytest.mark.parametrize(
    "azimuth, lit, shaded",
    [(90.0, "east", "west"), (270.0, "west", "east"), (180.0, "south", "north"), (0.0, "north", "south")],
)
def test_sun_lights_the_flank_facing_it(tmp_path, azimuth, lit, shaded):
    # The 3D north-up camera shows north up and east right, so image flanks
    # are compass flanks.
    scene = _scene(_gaussian(128, 128, 45, 120.0), sun=_compass_sun(azimuth, 25.0))
    luma = _render_luma(scene, tmp_path / f"sun_{int(azimuth)}.png")
    c = N // 2
    flanks = {
        "north": luma[c - 60 : c - 30, c - 15 : c + 15].mean(),
        "south": luma[c + 30 : c + 60, c - 15 : c + 15].mean(),
        "west": luma[c - 15 : c + 15, c - 60 : c - 30].mean(),
        "east": luma[c - 15 : c + 15, c + 30 : c + 60].mean(),
    }
    assert flanks[lit] > flanks[shaded] + 5.0, f"compass sun {azimuth}: {flanks}"


@requires_gpu
def test_overhead_sun_lights_all_flanks_alike(tmp_path):
    """A zenith sun has no preferred flank: shading must be symmetric."""
    luma = _render_luma(_scene(_gaussian(128, 128, 45, 120.0), sun=_compass_sun(0.0, 89.9)), tmp_path / "zenith.png")
    c = N // 2
    flanks = [
        luma[c - 60 : c - 30, c - 15 : c + 15].mean(),
        luma[c + 30 : c + 60, c - 15 : c + 15].mean(),
        luma[c - 15 : c + 15, c - 60 : c - 30].mean(),
        luma[c - 15 : c + 15, c + 30 : c + 60].mean(),
    ]
    assert max(flanks) - min(flanks) < 8.0, flanks


@requires_gpu
def test_camera_target_moves_the_view(tmp_path):
    dem = _gaussian(128, 128, 12, 80.0)
    base = _render_luma(_scene(dem), tmp_path / "target0.png")
    moved_camera = f3d.OrbitCamera(target=(60.0, 0.0, 0.0), distance=420.0, azimuth_deg=-90.0, elevation_deg=0.5, fov_deg=40.0)
    moved = _render_luma(_scene(dem, camera=moved_camera), tmp_path / "target_east.png")

    # Looking 60 m further east puts the centred peak further left on screen.
    base_col, moved_col = _feature_position(base)[1], _feature_position(moved)[1]
    assert moved_col < base_col - 10, (base_col, moved_col)


@requires_gpu
@pytest.mark.parametrize(
    "key, value",
    [
        ("sky", {"enabled": True, "aerial_density": 0.02}),
        ("height_ao", {"enabled": True, "strength": 1.0, "max_distance": 60.0}),
        ("sun_visibility", {"enabled": True, "mode": "hard", "max_distance": 300.0}),
    ],
)
def test_declared_lighting_settings_change_the_image(tmp_path, key, value):
    dem = _gaussian(110, 120, 25, 90.0) + _gaussian(160, 140, 18, 60.0)
    sun = _compass_sun(120.0, 12.0)
    plain = _render_luma(_scene(dem, sun=sun), tmp_path / "plain.png")
    with_setting = _render_luma(_scene(dem, sun=sun, settings={key: value}), tmp_path / f"{key}.png")
    assert np.abs(with_setting - plain).mean() > 0.5, f"settings[{key!r}] had no effect on the render"

# Stages 2-3: physical terrain-UV footprint and shared terrain-camera alignment.
_CAMERA_MODES = ("mesh:zup", "clipmap:2:32:32:10:0.3:zup")


def _drape_fixture(tmp_path, *, mode, pose):
    from forge3d.helpers.offscreen import save_png_deterministic
    dem = _gaussian(200, 56, 20, 100.0) + _gaussian(80, 180, 30, 35.0)
    camera = _top_down_north_up() if pose == "overhead" else f3d.OrbitCamera(
        target=(24.0, -16.0, 0.0), distance=420.0, azimuth_deg=-65.0, elevation_deg=30.0, fov_deg=40.0,
    )
    scene = _scene(dem, camera=camera, camera_mode=mode)
    # Bundles persist file-backed terrain assets; inline array fingerprints do
    # not serialize the DEM samples in the existing recipe format.
    dem_path = tmp_path / "asymmetric.npy"
    np.save(dem_path, dem)
    scene.recipe.terrain.path = str(dem_path)
    scene.recipe.terrain.data = None
    span = map_scene._terrain_scene_diagonal(scene.recipe.terrain)
    world = (span * 0.125, span * 0.125)
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[..., 0] = 255
    pixels[..., 3] = 255
    path = tmp_path / "footprint.png"
    save_png_deterministic(path, pixels)
    raster = f3d.RasterOverlay(layer_id="ortho", path=str(path), crs="EPSG:32633",
        metadata={"source_id": "footprint", "width": 32, "height": 32,
                  "bounds": [world[0] - 16, world[1] - 16, world[0] + 16, world[1] + 16]})
    return scene, raster, world


@requires_gpu
@pytest.mark.parametrize("mode", _CAMERA_MODES)
@pytest.mark.parametrize("pose", ("overhead", "oblique-target"))
def test_native_drape_vectors_and_world_labels_follow_camera(tmp_path, mode, pose, monkeypatch, record_property, line_join="miter"):
    from forge3d._map_scene_projection import TerrainProjector
    from forge3d import recipe_manifest as rm
    scene, raster, world = _drape_fixture(tmp_path, mode=mode, pose=pose)
    vector = f3d.VectorOverlay(layer_id="marker", crs="EPSG:32633", width_px=3, line_join=line_join,
        features=[
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": world}, "properties": {}},
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates":
                [[world[0] - 8, world[1]], [world[0] + 8, world[1]]]}, "properties": {}},
        ],
        style={"layers": [{"type": "line", "paint": {"line-color": "#ffffff"}}]})
    labels = f3d.LabelLayer(layer_id="labels", occlusion="none",
        labels=[{"id": "peak", "kind": "point", "text": "P", "geometry": {"type": "Point", "coordinates": world}}],
        glyph_atlas={"glyphs": ["P"]}, metadata={"source_id": "peak-label", "seed": 7, "coordinate_space": "world"})
    with map_scene._shared_terrain_render_context():
        bare_path = tmp_path / "bare.png"
        scene.render(str(bare_path))
        bare = np.asarray(Image.open(bare_path).convert("RGBA"))
        scene.recipe.layers = [raster]
        drape_path = tmp_path / "drape.png"
        scene.render(str(drape_path))
        draped = np.asarray(Image.open(drape_path).convert("RGBA"))
        assert scene.last_render_metadata["raster_overlay_backend"] == "native_terrain_uv"
        assert scene.last_render_metadata["raster_overlay_layer_count"] == 1
        expected = TerrainProjector(scene.recipe).project(world)
        red = (draped[..., 0] > draped[..., 1] + draped[..., 2].astype(np.int32))
        assert red.any(), "the physical terrain pass must sample the red footprint"
        row, col = int(expected[1]), int(expected[0])
        assert red[row, col], (mode, pose, expected)
        changed = np.any(bare != draped, axis=2)
        assert changed.any() and not changed[0, 0], "a UV footprint must not paint the frame background"
        if pose == "overhead":
            assert row < N // 2 and col > N // 2, "north-east raster footprint must remain north-east"
        scene.recipe.layers = [raster, vector]
        vector_path = tmp_path / "vector.png"
        vector_report = scene.render(str(vector_path))
        vector_pixels = np.asarray(Image.open(vector_path).convert("RGBA"))
        expected_backend = "native_oit" if line_join == "round" else "python_precise_raster"
        assert scene.last_render_metadata["vector_backend"] == expected_backend
        if line_join != "round":
            assert vector_report.supported_features["mapscene.vector_precise_raster_composite"] == "supported"
        vector_mask = np.any(vector_pixels != draped, axis=2)
        assert vector_mask.any()
        assert np.all(red[vector_mask]), "projected vector marker must sit inside its world raster footprint"
        drawn_y, drawn_x = np.nonzero(vector_mask)
        projected_path = TerrainProjector(scene.recipe).path(vector.features[1]["geometry"]["coordinates"])
        for sample in projected_path:
            distance = np.hypot(drawn_x + 0.5 - sample[0], drawn_y + 0.5 - sample[1])
            assert float(distance.min()) <= vector.width_px, "every draped line segment must produce pixels"
        record_property("drawn_line_samples", len(projected_path))
        scene.recipe.layers = [raster, vector, labels]
        compiled = scene.compile_plan()
        assert compiled.label_plans["labels"].accepted
        label = compiled.label_plans["labels"].accepted[0]
        center = next(candidate for candidate in label.candidates if candidate.candidate_type == "center")
        np.testing.assert_allclose(center.anchor, expected, rtol=0, atol=np.finfo(np.float32).eps * N)
        frozen = rm.manifest_to_json(compiled.manifest)
        def fail_recompile(*args, **kwargs):
            raise AssertionError("render must consume the compiled label plan")
        monkeypatch.setattr(map_scene, "_label_plan_from_layer", fail_recompile)
        final_path = tmp_path / "labels.png"
        scene.render(str(final_path))
        final = np.asarray(Image.open(final_path).convert("RGBA"))
        assert np.any(final != vector_pixels), "native label pixels must be drawn"
        glyph_y, glyph_x = np.nonzero(np.any(final != vector_pixels,axis=2))
        bounds = label.screen_bounds
        halo = float(label.typography.get("halo_width_px",0))
        # One pixel covers integer rasterization at the glyph/halo edge.
        assert glyph_x.min() >= math.floor(bounds[0]-halo)-1
        assert glyph_x.max() <= math.ceil(bounds[2]+halo)+1
        assert glyph_y.min() >= math.floor(bounds[1]-halo)-1
        assert glyph_y.max() <= math.ceil(bounds[3]+halo)+1

        assert rm.manifest_to_json(compiled.manifest) == frozen
        record_property("raster_footprint_pixels", int(red.sum()))
        record_property("vector_pixels", int(vector_mask.sum()))
        record_property("projected_anchor", list(expected))


@requires_gpu
@pytest.mark.parametrize("mode", ("screen",) + _CAMERA_MODES)
def test_transparent_drape_preserves_identical_pixels(tmp_path, mode):
    from forge3d.helpers.offscreen import save_png_deterministic
    scene, raster, _ = _drape_fixture(tmp_path, mode=mode, pose="overhead")
    save_png_deterministic(raster.path, np.zeros((32, 32, 4), dtype=np.uint8))
    with map_scene._shared_terrain_render_context():
        plain_path, transparent_path = tmp_path / "plain.png", tmp_path / "transparent.png"
        scene.render(str(plain_path))
        scene.recipe.layers = [raster]
        scene.render(str(transparent_path))
    np.testing.assert_array_equal(np.asarray(Image.open(plain_path)), np.asarray(Image.open(transparent_path)))


@requires_gpu
def test_raster_content_invalidates_chronos_and_pixel_cache(tmp_path, record_property):
    from forge3d.chronos import _camera_json, _scene_json
    from forge3d.helpers.offscreen import save_png_deterministic
    scene, raster, _ = _drape_fixture(tmp_path, mode="mesh:zup", pose="overhead")
    scene.recipe.layers = [raster]
    def scene_hash():
        return json.loads(f3d.compile_frame(0, 1, 1, _camera_json(scene), _scene_json(scene, [])).to_json())["scene_hash"]
    first_hash = scene_hash()
    cache = tmp_path / "cache"
    with map_scene._shared_terrain_render_context():
        scene.render(str(tmp_path / "first.png"), cache=cache)
        assert not scene.last_render_metadata["cache_hit"]
        scene.render(str(tmp_path / "replayed.png"), cache=cache)
        assert scene.last_render_metadata["cache_hit"]
        pixels = np.zeros((32, 32, 4), dtype=np.uint8)
        pixels[..., 1] = 255
        pixels[..., 3] = 255
        save_png_deterministic(raster.path, pixels)
        second_hash = scene_hash()
        assert first_hash != second_hash
        scene.render(str(tmp_path / "changed.png"), cache=cache)
        assert not scene.last_render_metadata["cache_hit"]
    assert (tmp_path / "first.png").read_bytes() != (tmp_path / "changed.png").read_bytes()
    record_property("original_scene_hash", first_hash)
    record_property("changed_scene_hash", second_hash)


@pytest.mark.parametrize("bad_source", ("unreadable", "invalid-png", "invalid-bounds"))
def test_unsupported_drape_blocks_before_terrain_pixels(tmp_path, monkeypatch, bad_source):
    scene, raster, _ = _drape_fixture(tmp_path, mode="mesh:zup", pose="overhead")
    if bad_source == "unreadable":
        raster.path = str(tmp_path / "missing.png")
    elif bad_source == "invalid-png":
        Path(raster.path).write_bytes(b"invalid image bytes")
    else:
        raster.metadata = dict(raster.metadata, bounds=[0, 0, 0, 1])
    scene.recipe.layers = [raster]
    def no_pixels(*args, **kwargs):
        raise AssertionError("unsupported raster must block before the native terrain draw")
    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result_impl", no_pixels)
    output = tmp_path / "blocked.png"
    if bad_source == "unreadable":
        report = scene.validate()
        assert any(item.layer_id == "ortho" and item.severity == "error" for item in report.diagnostics)
        with pytest.raises(RuntimeError, match="blocked by blocking diagnostics"):
            scene.render(str(output))
    else:
        with pytest.raises(f3d.MapSceneNativeUnavailable) as excinfo:
            scene.render(str(output))
        assert excinfo.value.diagnostics[0]["layer"] == "ortho"
    assert not output.exists()


@requires_gpu
@pytest.mark.parametrize("mode", _CAMERA_MODES)
def test_world_overlay_bundle_freezes_manifest_and_report(tmp_path, mode, monkeypatch, record_property):
    from forge3d import recipe_manifest as rm
    from test_mapscene_sutura_integrity import _ssim, _report_bytes
    scene, raster, world = _drape_fixture(tmp_path, mode=mode, pose="oblique-target")
    scene.recipe.layers = [raster, f3d.LabelLayer(layer_id="labels", occlusion="none",
        labels=[{"id": "peak", "text": "P", "geometry": {"type": "Point", "coordinates": world}}],
        glyph_atlas={"glyphs": ["P"]}, metadata={"source_id": "world-label", "coordinate_space": "world"})]
    with map_scene._shared_terrain_render_context():
        first_path, second_path = tmp_path / "first.png", tmp_path / "second.png"
        first_report = scene.render(str(first_path))
        first_manifest = rm.manifest_to_json(scene.compiled_plan.manifest)
        scene.save_bundle(tmp_path / "bundle")
        with monkeypatch.context() as patch:
            def no_recompile(*args, **kwargs):
                raise AssertionError("v3 bundle load must consume frozen label/depth decisions")
            patch.setattr(map_scene, "_label_plan_from_layer", no_recompile)
            loaded = f3d.MapScene.load_bundle(scene.last_bundle_path)
        assert rm.manifest_to_json(loaded.compiled_plan.manifest) == first_manifest
        second_report = loaded.render(str(second_path))
    score = _ssim(np.asarray(Image.open(first_path)), np.asarray(Image.open(second_path)))
    assert score >= 0.99
    assert _report_bytes(first_report) == _report_bytes(second_report)
    assert rm.manifest_to_json(loaded.compiled_plan.manifest) == first_manifest
    record_property("roundtrip_ssim", score)

@requires_gpu
def test_screen_drape_keeps_north_up_uv_footprint(tmp_path, record_property):
    scene, raster, _ = _drape_fixture(tmp_path, mode="screen", pose="overhead")
    scene.recipe.layers = [raster]
    path = tmp_path / "screen.png"
    scene.render(str(path))
    pixels = np.asarray(Image.open(path).convert("RGBA"))
    red = pixels[..., 0].astype(np.int32) > pixels[..., 1].astype(np.int32) + pixels[..., 2].astype(np.int32)
    yy, xx = np.nonzero(red)
    assert len(xx)
    assert float(xx.mean()) > N / 2 and float(yy.mean()) < N / 2
    assert not red[0, 0] and not red[-1, -1]
    record_property("footprint_centroid", [float(xx.mean()), float(yy.mean())])


def test_projection_preserves_distinct_f64_positions_until_screen_space():
    scene = _scene(np.zeros((N, N), dtype=np.float32))
    params = map_scene._build_mapscene_terrain_params(scene.recipe, scene.recipe.terrain.data, (N, N))
    left = float(np.float32(0.5))
    adjacent = float(np.nextafter(np.float32(0.5), np.float32(1.0)))
    right = (left + adjacent) / 2.0
    assert np.float32(left) == np.float32(right)
    projected = params.project_terrain_points([[left, 0.5, 0.0, 0.0], [right, 0.5, 0.0, 0.0]])
    assert projected[0] is not None and projected[1] is not None
    assert projected[0][:2] != projected[1][:2]


def test_projection_surface_registration_and_finite_contract():
    scene = _scene(np.zeros((N, N), dtype=np.float32))
    params = map_scene._build_mapscene_terrain_params(scene.recipe, scene.recipe.terrain.data, (N, N))
    assert callable(params.project_terrain_points)
    assert callable(params.project_terrain_depth)
    assert callable(params.unproject_terrain_depth)
    assert params.project_terrain_points([[0.5, 0.5, 0.0, 0.0]])[0] is not None
    with pytest.raises(ValueError, match="finite"):
        params.project_terrain_points([[float("nan"), 0.5, 0.0, 0.0]])
    depth=params.project_terrain_depth(scene.recipe.terrain.data,(N,N))
    assert depth.shape==(N,N) and depth.dtype==np.float32
    uv=params.unproject_terrain_depth(depth)
    assert uv.shape==(N,N,2)
    assert np.isfinite(uv[depth<1]).all()
    with pytest.raises(ValueError,match="finite"):
        params.project_terrain_depth(np.full((2,2),np.nan,np.float32),(N,N))


@requires_gpu
def test_world_label_depth_is_decided_in_compile_plan(tmp_path, monkeypatch, record_property):
    from forge3d._map_scene_projection import TerrainProjector
    scene, _, world = _drape_fixture(tmp_path, mode="mesh:zup", pose="overhead")
    projector = TerrainProjector(scene.recipe)
    ground = projector.height(projector.uv(world))
    labels = f3d.LabelLayer(layer_id="depth-labels", occlusion="terrain",
        glyph_atlas={"glyphs": ["P"]}, metadata={"source_id": "depth-labels", "seed": 7, "coordinate_space": "world"})
    plans = []
    with map_scene._shared_terrain_render_context():
        scene.render(str(tmp_path / "bare.png"))
        bare = np.asarray(Image.open(tmp_path / "bare.png"))
        for name, elevation in (("below", ground - 30), ("ground", ground), ("above", ground + 30)):
            labels.labels = [{"id": name, "text": "P", "kind": "point", "geometry":
                {"type": "Point", "coordinates": [*world, elevation]}}]
            scene.recipe.layers = [labels]
            compiled = scene.compile_plan()
            plan = compiled.label_plans["depth-labels"]
            plans.append(plan)
            with monkeypatch.context() as patch:
                def no_decision(*args, **kwargs):
                    raise AssertionError("depth decisions must be frozen by compile_plan")
                patch.setattr(map_scene, "_label_plan_from_layer", no_decision)
                scene.render(str(tmp_path / (name + ".png")))
        assert not plans[0].accepted and plans[1].accepted and plans[2].accepted
        np.testing.assert_array_equal(bare, np.asarray(Image.open(tmp_path / "below.png")))
        assert np.any(bare != np.asarray(Image.open(tmp_path / "above.png")))
    record_property("below_terrain_accepted", len(plans[0].accepted))
    record_property("on_terrain_accepted", len(plans[1].accepted))
    record_property("above_terrain_accepted", len(plans[2].accepted))


@requires_gpu
@pytest.mark.parametrize("mode", ("screen", *_CAMERA_MODES))
def test_geotiff_mask_and_world_grid_drape_on_physical_terrain(tmp_path, mode):
    import rasterio
    from rasterio.transform import from_origin
    from forge3d._map_scene_projection import TerrainProjector
    scene, png, world = _drape_fixture(tmp_path, mode=mode, pose="overhead")
    transform = from_origin(500000.0, 5800000.0, 1.0, 1.0)
    scene.recipe.terrain.metadata = dict(scene.recipe.terrain.metadata,
        geotransform=list(transform.to_gdal()))
    # The same NE patch expressed in a real UTM raster grid; transparent
    # surroundings and an asymmetric alpha notch remain transparent.
    pixels = np.zeros((N, N, 4), dtype=np.uint8)
    pixels[40:72, 184:216] = [255, 0, 0, 255]
    pixels[40:48, 184:192] = 0
    path = tmp_path / "ortho.tif"
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=4,
        dtype="uint8", crs="EPSG:32633", transform=transform) as dst:
        dst.write(np.moveaxis(pixels, -1, 0))
    raster = f3d.RasterOverlay(layer_id="ortho", path=str(path), crs="EPSG:32633")
    with map_scene._shared_terrain_render_context():
        scene.render(str(tmp_path / "bare-geotiff.png"))
        bare = np.asarray(Image.open(tmp_path / "bare-geotiff.png"))
        scene.recipe.layers = [raster]
        scene.render(str(tmp_path / "draped-geotiff.png"))
        rgba = np.asarray(Image.open(tmp_path / "draped-geotiff.png"))
        red = rgba[..., 0].astype(np.int32) > rgba[..., 1].astype(np.int32) + rgba[..., 2].astype(np.int32)
        assert red.any()
        if mode == "screen":
            row, col = 56, 200
        else:
            east, north = transform * (200.5, 56.5)
            anchor = TerrainProjector(scene.recipe).project((east, north))
            row, col = int(anchor[1]), int(anchor[0])
        assert red[row, col]
        assert row < N // 2 and col > N // 2
        assert not np.any(bare[0, 0] != rgba[0, 0])
        projector = None if mode == "screen" else TerrainProjector(scene.recipe)
        def pixel_at(col, row):
            if projector is None:
                return int(row), int(col)
            east, north = transform * (col + 0.5, row + 0.5)
            anchor = projector.project((east, north))
            return int(anchor[1]), int(anchor[0])
        # All four footprint edges and both notch axes, beyond the filter edge.
        for col, row in ((200,42),(214,56),(200,70),(186,56),(194,44),(188,50)):
            y,x = pixel_at(col,row)
            assert red[y,x], (mode,col,row,x,y)
        for col, row in ((200,36),(220,56),(200,76),(180,56),(188,44)):
            y,x = pixel_at(col,row)
            assert not red[y,x], (mode,col,row,x,y)
            np.testing.assert_array_equal(rgba[y,x],bare[y,x])



@pytest.mark.parametrize("input_kind", ("outside-world-width", "outside-terrain", "world-line-label"))
def test_unsupported_world_overlays_block_before_pixels(tmp_path, monkeypatch, input_kind):
    scene, _, world = _drape_fixture(tmp_path, mode="mesh:zup", pose="overhead")
    if input_kind == "world-line-label":
        layer = f3d.LabelLayer(layer_id="unsupported", occlusion="none", metadata={"coordinate_space": "world"}, labels=[
            {"id": "line", "kind": "line", "text": "P", "geometry":
             {"type": "LineString", "coordinates": [list(world), [world[0]+1, world[1]+1]]}}])
    else:
        coords = (world[0]*8, world[1]*8)
        layer = f3d.VectorOverlay(layer_id="unsupported", crs="EPSG:32633",
            width_world=1.0 if input_kind == "outside-world-width" else None,
            features=[{"type": "Feature", "geometry": {"type": "Point", "coordinates": coords}}])
    scene.recipe.layers = [layer]
    def no_pixels(*args, **kwargs):
        raise AssertionError("unsupported world inputs must block before the terrain draw")
    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result_impl", no_pixels)
    output = tmp_path / "unsupported.png"
    report = scene.validate()
    assert report.render_blocked()
    assert scene.compile_plan().validation_report.render_blocked()
    with pytest.raises(RuntimeError, match="blocking diagnostics"):
        scene.render(str(output))
    assert any(diagnostic.layer_id == "unsupported" for diagnostic in report.diagnostics)
    assert not output.exists()


@requires_gpu
@pytest.mark.parametrize("mode", _CAMERA_MODES)
def test_camera_alignment_on_raised_asymmetric_peak(tmp_path, mode, monkeypatch, record_property):
    fixture = _drape_fixture
    def peak_fixture(*args, **kwargs):
        scene, raster, _ = fixture(*args, **kwargs)
        span = map_scene._terrain_scene_diagonal(scene.recipe.terrain)
        world = (span * (200 / (N - 1) - 0.5), span * (0.5 - 56 / (N - 1)))
        raster.metadata = dict(raster.metadata,
            bounds=[world[0]-16, world[1]-16, world[0]+16, world[1]+16])
        # The original offset view puts this summit on the terrain silhouette,
        # where a finite-width stroke legitimately reaches the background.
        # Aim at the raised peak, retaining the strict full-footprint assertion
        # and the native camera's oblique polar angle (measured from vertical).
        heightmap = map_scene._load_native_heightmap(scene.recipe.terrain)
        center = sum(map_scene._heightmap_domain(heightmap)) * 0.5
        cell_spacing = span / (N - 1)
        dy, dx = np.gradient(heightmap, cell_spacing)
        max_slope = float(np.hypot(dx, dy).max())
        # Half the measured steepest-slope grazing angle keeps this fixture
        # away from folds without changing the production camera or mesh.
        polar_angle = math.degrees(math.atan(1.0 / max_slope)) * 0.5
        scene.recipe.camera = f3d.OrbitCamera(
            target=(world[0], world[1], float(heightmap[56, 200]) - center),
            distance=420.0, azimuth_deg=-65.0, elevation_deg=polar_angle, fov_deg=40.0)
        record_property("camera_polar_angle_deg", polar_angle)
        from forge3d._map_scene_projection import TerrainProjector
        projector = TerrainProjector(scene.recipe)
        anchor = projector.project(world)
        flat = projector.project((*world, 0.0))
        assert np.linalg.norm(np.asarray(anchor[:2]) - np.asarray(flat[:2])) > 3.0
        record_property("height_projection_pixel_displacement", float(
            np.linalg.norm(np.asarray(anchor[:2]) - np.asarray(flat[:2]))))
        ground = projector.height(projector.uv(world))
        assert ground == pytest.approx(float(map_scene._load_native_heightmap(scene.recipe.terrain)[56, 200]))
        record_property("terrain_height_at_anchor", ground)
        return scene, raster, world
    monkeypatch.setitem(globals(), "_drape_fixture", peak_fixture)
    test_native_drape_vectors_and_world_labels_follow_camera(
        tmp_path, mode, "oblique-target", monkeypatch, record_property, line_join="round")


def test_path_only_world_vector_blocks_before_pixels(tmp_path, monkeypatch):
    scene, _, _ = _drape_fixture(tmp_path, mode="mesh:zup", pose="overhead")
    source = tmp_path / "points.geojson"
    source.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}}]}), encoding="utf-8")
    scene.recipe.layers = [f3d.VectorOverlay(layer_id="path-vector", path=str(source), crs="EPSG:32633")]
    def no_pixels(*args, **kwargs):
        raise AssertionError("path-only world vectors must block before the terrain draw")
    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result_impl", no_pixels)
    output = tmp_path / "blocked-vector.png"
    # Preserve the existing path-loader validation diagnostic and exception.
    with pytest.raises(RuntimeError, match="blocking diagnostics"):
        scene.render(str(output))
    assert scene.last_validation_report.render_blocked()
    assert any(diagnostic.layer_id == "path-vector"
               for diagnostic in scene.last_validation_report.diagnostics)
    assert not output.exists()


@requires_gpu
@pytest.mark.parametrize("line_join", ("miter", "round"))
def test_projected_polygon_preserves_world_footprint_and_hole(tmp_path, line_join):
    from forge3d._map_scene_projection import TerrainProjector
    scene, raster, world = _drape_fixture(tmp_path, mode="mesh:zup", pose="oblique-target")
    def ring(radius):
        x, y = world
        return [[x-radius,y-radius],[x+radius,y-radius],[x+radius,y+radius],
                [x-radius,y+radius],[x-radius,y-radius]]
    polygon = f3d.VectorOverlay(layer_id="polygon", crs="EPSG:32633", line_join=line_join,
        width_px=1, features=[{"type":"Feature", "geometry":{"type":"Polygon",
            "coordinates":[ring(12), ring(4)]}, "properties":{}}],
        style={"layers":[{"type":"fill", "paint":{"fill-color":"#ffffff", "fill-opacity":1}},
                         {"type":"line", "paint":{"line-color":"#ffffff"}}]})
    with map_scene._shared_terrain_render_context():
        scene.recipe.layers=[raster]
        scene.render(str(tmp_path/"drape.png"))
        draped=np.asarray(Image.open(tmp_path/"drape.png"))
        scene.recipe.layers=[raster,polygon]
        polygon_report = scene.render(str(tmp_path/"polygon.png"))
        expected_backend = "native_oit" if line_join == "round" else "python_precise_raster"
        assert scene.last_render_metadata["vector_backend"] == expected_backend
        if line_join != "round":
            assert polygon_report.supported_features["mapscene.vector_precise_raster_composite"] == "supported"
        painted=np.asarray(Image.open(tmp_path/"polygon.png"))
        changed=np.any(painted!=draped,axis=2)
        red=draped[...,0].astype(np.int32)>draped[...,1].astype(np.int32)+draped[...,2].astype(np.int32)
        assert changed.any() and np.all(red[changed])
        projector=TerrainProjector(scene.recipe)
        hole=projector.project(world)
        fill=projector.project((world[0]+8,world[1]))
        np.testing.assert_array_equal(painted[int(hole[1]),int(hole[0])], draped[int(hole[1]),int(hole[0])])
        np.testing.assert_array_equal(painted[int(fill[1]),int(fill[0]),:3], [255,255,255])


@requires_gpu
@pytest.mark.parametrize("option", ("globe", "GLOBE"))
def test_globe_option_follows_actual_native_mapscene_camera(tmp_path, option):
    scene, _, world = _drape_fixture(tmp_path, mode=f"clipmap:2:32:32:10:0.3:zup:{option}", pose="overhead")
    scene.recipe.layers=[f3d.VectorOverlay(layer_id="planetary", crs="EPSG:32633", width_px=3,
        features=[{"type":"Feature", "geometry":{"type":"Point", "coordinates":world}}])]
    report = scene.validate()
    assert not report.render_blocked()
    scene.render(str(tmp_path/"globe-vector.png"))
    assert scene.last_render_metadata["vector_backend"] in {"native_oit", "python_precise_raster"}
