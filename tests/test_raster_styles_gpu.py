"""Required D02 physical native proof. No acceptance skips or CPU substitutes."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import forge3d as f3d
import forge3d.map_scene as ms
from test_raster_styles import PALETTE, bivariate


def _physical():
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert probe["status"] == "ok", probe
    assert str(probe["device_type"]).lower() in {"discretegpu", "integratedgpu"}, probe
    assert not probe.get("software_fallback", False), probe
    assert not any(token in probe["name"].lower() for token in ("lavapipe", "llvmpipe", "swiftshader", "warp", "basic render")), probe
    return probe


def _scene(population, *, height_scale=1, layers=(), camera_mode="screen", sun=(0.8, 0.6, 0.0)):
    return f3d.MapScene(
        terrain=f3d.TerrainSource(data=population, crs="EPSG:3857", elevation_sampling_available=True,
            style=f3d.RasterHeightSurfaceStyle(height_scale, "people/km²", low_color=(128, 128, 128, 255), high_color=(128, 128, 128, 255)),
            metadata={"source_id": "physical-fixture", "width": population.shape[1], "height": population.shape[0], "resolution": [1, 1]}),
        layers=layers, camera=f3d.OrbitCamera(distance=300, elevation_deg=50, azimuth_deg=-90),
        lighting=f3d.LightingPreset(name="daylight", sun_direction=sun, settings={"hue_variation_strength": 0, "cli_params": {"camera_mode": camera_mode}}),
        output=f3d.OutputSpec(width=192, height=192, aovs=("albedo", "normal", "depth")),
    )


@pytest.fixture
def observe_native(monkeypatch):
    # Observe actual submitted native frames without changing any result.
    original = ms._render_terrain_renderer_result
    observed = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.aov_frame is not None
        observed.append(result)
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    return observed


def _render(scene, path, observed):
    scene.render(str(path))
    assert scene.last_render_backend == "gpu_terrain"
    assert scene.last_render_metadata["thematic_backend"] == "native_terrain_rgba_cells"
    frame = observed[-1].aov_frame
    return np.asarray(Image.open(path)), np.asarray(frame.albedo()), np.asarray(frame.normal()), np.asarray(frame.depth())


def _record(tmp_path, name, probe, scene, metrics):
    payload = {"adapter": probe, "metadata": scene.last_render_metadata, "measurements": metrics}
    (tmp_path / f"{name}-measurements.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
    if os.environ.get("FORGE3D_D02_ARTIFACT_DIR"):
        root = Path(os.environ["FORGE3D_D02_ARTIFACT_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}-measurements.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps({"name": name, "adapter": probe, "measurements": metrics}, sort_keys=True))


@pytest.mark.parametrize("kind", ["categorical", "bivariate"])
def test_physical_native_palette_cells_nodata_and_replay(tmp_path, observe_native, kind):
    probe = _physical()
    x = np.repeat(np.repeat(np.array([[0, 1, 2]] * 3, np.float32), 32, 0), 32, 1)
    y = np.repeat(np.repeat(np.array([[0] * 3, [10] * 3, [20] * 3], np.float32), 32, 0), 32, 1)
    style = bivariate() if kind == "bivariate" else f3d.CategoricalRasterStyle({i: PALETTE[0][i] for i in range(3)})
    layer = f3d.RasterOverlay(kind, data=x, secondary_data=y if kind == "bivariate" else None, style=style, crs="EPSG:3857")
    scene = _scene(np.zeros_like(x), layers=[layer])
    rgba, albedo, _, _ = _render(scene, tmp_path / f"{kind}.png", observe_native)
    centers = [32, 96, 160]
    actual = albedo[np.ix_(centers, centers)][..., :3]
    srgb = np.array(PALETTE if kind == "bivariate" else [PALETTE[0]] * 3)[..., :3] / 255
    expected = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    # Albedo AOV is RGBA16Float; use its measured storage precision.
    error = float(np.max(np.abs(actual - expected)))
    half_ulp = float(np.max(np.spacing(expected.astype(np.float16)).astype(np.float32)))
    assert error <= half_ulp, (actual, expected, error, half_ulp)
    scene.save_bundle(tmp_path / "style.forge3d")
    restored = f3d.MapScene.load_bundle(tmp_path / "style.forge3d")
    replay, _, _, _ = _render(restored, tmp_path / "replay.png", observe_native)
    np.testing.assert_array_equal(rgba, replay)
    # A declared nodata cell must actually discard native terrain pixels.
    x[:32, :32] = np.nan
    _, holes, _, _ = _render(scene, tmp_path / "nodata.png", observe_native)
    assert np.all(holes[32, 32] == 0), holes[32, 32]
    assert holes[96, 96, :3].any()
    _record(tmp_path, kind, probe, restored, {"palette_max_linear_error": error, "rgba16float_ulp": half_ulp,
        "cells_checked": 9, "nodata_albedo": holes[32, 32].tolist(), "replay_pixel_mismatches": 0,
        "pixel_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(), "center_albedo": actual.tolist()})


def test_physical_population_height_and_shade(tmp_path, observe_native):
    probe = _physical()
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
    data = 20 * np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / 200)
    scene = _scene(data, camera_mode="mesh:zup")
    raised, _, normals, depth = _render(scene, tmp_path / "height.png", observe_native)
    flat_scene = _scene(np.zeros_like(data), camera_mode="mesh:zup")
    flat, _, flat_normals, flat_depth = _render(flat_scene, tmp_path / "flat.png", observe_native)
    height_delta = float(np.max(np.abs(depth - flat_depth)))
    normal_delta = float(np.max(np.abs(normals - flat_normals)))
    assert height_delta > 0 and normal_delta > 0, (height_delta, normal_delta)
    # Keep the geometry and uniform palette fixed; reverse the sun to prove
    # native shading responds to the raised population surface.
    other_scene = _scene(data, camera_mode="mesh:zup", sun=(-0.8, 0.6, 0.0))
    other, _, _, _ = _render(other_scene, tmp_path / "opposite-sun.png", observe_native)
    pixel_delta = int(np.count_nonzero(np.any(raised[..., :3] != other[..., :3], axis=2)))
    assert pixel_delta > 0
    assert np.any(raised != flat)
    _record(tmp_path, "population", probe, scene, {"source_max": float(data.max()), "height_max": float(ms._load_native_heightmap(scene.recipe.terrain).max()),
        "max_depth_difference_from_flat": height_delta, "max_normal_difference_from_flat": normal_delta,
        "opposite_sun_changed_pixels": pixel_delta, "pixel_sha256": hashlib.sha256(raised.tobytes()).hexdigest()})
