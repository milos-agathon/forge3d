"""SUTURA geometric integrity: 3D MapScene draws things in the right place.

SUTURA guarantees that every layer is drawn for real or blocks. These checks
add the missing half: in the 3D (``mesh:zup``) camera modes the map is
north-up and not mirrored, the sun lights the flank that faces it,
``camera.target`` moves the view, declared lighting settings are validated,
and layers that are only placed in 2D screen space block instead of being
drawn in the wrong place.
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
