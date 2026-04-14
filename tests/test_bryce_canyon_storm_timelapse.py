from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "bryce_canyon_storm_timelapse.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bryce_canyon_storm_timelapse", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    forge3d_stub = types.ModuleType("forge3d")
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = False
    previous_forge3d = sys.modules.get("forge3d")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
        added_examples_dir = True
    sys.modules["forge3d"] = forge3d_stub
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def _scene(module):
    return module.SceneConfig(
        terrain_width=160.0,
        domain=(0.0, 120.0),
        phi_deg=224.0,
        theta_deg=66.0,
        radius=180.0,
        fov_deg=30.0,
        zscale=0.24,
        target=(80.0, 14.0, 60.0),
        sun_orbit_radius=112.0,
    )


def _cache(module):
    terrain_mask = np.zeros((120, 160), dtype=np.float32)
    terrain_mask[54:, :] = 1.0
    return module._build_weather_cache((160, 120), terrain_mask), terrain_mask


def _centroid(field: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]].astype(np.float32)
    weight = np.clip(field, 0.0, 1.0).astype(np.float32)
    total = float(weight.sum())
    assert total > 1e-6
    return np.array([(weight * xx).sum() / total, (weight * yy).sum() / total], dtype=np.float32)


def _pitched_relief_mask(height: int = 120, width: int = 160) -> np.ndarray:
    terrain_mask = np.zeros((height, width), dtype=np.float32)
    skyline = np.interp(
        np.arange(width, dtype=np.float32),
        [0.0, width * 0.25, width * 0.50, width * 0.75, width - 1.0],
        [height * 0.15, height * 0.10, height * 0.05, height * 0.10, height * 0.17],
    )
    left = np.interp(
        np.arange(height, dtype=np.float32),
        [0.0, height * 0.17, height * 0.67, height - 1.0],
        [width * 0.08, 0.0, width * 0.16, width * 0.39],
    )
    right_margin = np.interp(
        np.arange(height, dtype=np.float32),
        [0.0, height * 0.17, height * 0.67, height - 1.0],
        [width * 0.06, 0.0, width * 0.15, width * 0.36],
    )
    right = (width - 1.0) - right_margin
    for x, top in enumerate(np.round(skyline).astype(np.int32)):
        for y in range(top, height):
            if int(round(left[y])) <= x <= int(round(right[y])):
                terrain_mask[y, x] = 1.0
    return terrain_mask


def test_sun_state_follows_day_arc() -> None:
    module = _load_module()

    sunrise = module._sun_state(0.0)
    noon = module._sun_state(0.5)
    sunset = module._sun_state(1.0)

    assert sunrise.azimuth_deg < noon.azimuth_deg < sunset.azimuth_deg
    assert sunrise.elevation_deg < noon.elevation_deg
    assert sunset.elevation_deg < noon.elevation_deg
    assert noon.intensity > sunrise.intensity


def test_parse_args_defaults_to_fifteen_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["bryce_canyon_storm_timelapse.py"])

    args = module._parse_args()

    assert args.duration == pytest.approx(15.0)


def test_rain_envelope_builds_into_late_downpour() -> None:
    module = _load_module()

    early = module._rain_envelope(0.35)
    build = module._rain_envelope(0.75)
    peak = module._rain_envelope(0.92)

    assert early < 0.01
    assert build > early
    assert peak > build
    assert peak > 0.85


def test_overlay_is_warm_and_opaque() -> None:
    module = _load_module()
    heightmap = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    overlay = module._build_overlay(heightmap)

    assert overlay.shape == (8, 8, 4)
    assert overlay.dtype == np.uint8
    assert np.all(overlay[..., 3] == 255)
    assert float(overlay[..., 0].mean()) > float(overlay[..., 2].mean())


def test_projected_cloud_planes_are_deterministic_and_time_variant() -> None:
    module = _load_module()
    cache, _ = _cache(module)
    scene = _scene(module)

    sun_a = module._sun_screen(scene, progress=0.35, width=160, height=120)
    sun_c = module._sun_screen(scene, progress=0.90, width=160, height=120)
    weather_a = module._projected_cloud_planes(cache, scene=scene, progress=0.35, sun_screen=sun_a, full_size=(160, 120))
    weather_b = module._projected_cloud_planes(cache, scene=scene, progress=0.35, sun_screen=sun_a, full_size=(160, 120))
    weather_c = module._projected_cloud_planes(cache, scene=scene, progress=0.90, sun_screen=sun_c, full_size=(160, 120))

    np.testing.assert_allclose(weather_a["clouds"], weather_b["clouds"])
    np.testing.assert_allclose(weather_a["shadow"], weather_b["shadow"])
    np.testing.assert_allclose(weather_a["rain_sheet"], weather_b["rain_sheet"])
    assert weather_a["clouds"].shape == (120, 160)
    assert not np.allclose(weather_a["clouds"], weather_c["clouds"])
    assert not np.allclose(weather_a["rain_sheet"], weather_c["rain_sheet"])


def test_cloud_deck_stays_broad_across_the_frame() -> None:
    module = _load_module()
    cache, _ = _cache(module)
    scene = _scene(module)

    weather = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.35,
        sun_screen=module._sun_screen(scene, progress=0.35, width=160, height=120),
        full_size=(160, 120),
    )

    cloud_grad = float(np.abs(np.diff(weather["clouds"], axis=1)).mean() + np.abs(np.diff(weather["clouds"], axis=0)).mean())

    assert float(weather["clouds"].mean()) > 0.18
    assert float(weather["clouds"][54:78, :].mean()) > 0.42
    assert cloud_grad > 0.010


def test_late_cloud_cover_wraps_the_full_relief() -> None:
    module = _load_module()
    cache, _ = _cache(module)
    scene = _scene(module)

    weather = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.92,
        sun_screen=module._sun_screen(scene, progress=0.92, width=160, height=120),
        full_size=(160, 120),
    )

    terrain = np.zeros((120, 160), dtype=bool)
    terrain[54:, :] = True
    upper_relief = weather["clouds"][54:78, :]
    upper_shadow = weather["shadow"][54:78, :]
    upper_canopy = weather["terrain_canopy"][54:78, :]
    assert float(upper_relief.mean()) > 0.68
    assert float(upper_shadow.mean()) > 0.25
    assert float(upper_canopy.mean()) > 0.44
    assert float(np.quantile(weather["terrain_canopy"][terrain], 0.10)) > 0.38

    base_rgb = np.full((120, 160, 3), 180, dtype=np.uint8)
    frame = module._composite_frame(base_rgb, terrain_mask=_cache(module)[1], weather=weather, progress=0.92)
    luminance = frame.mean(axis=2)
    assert float(np.quantile(luminance[terrain], 0.90)) < 158.0
    assert float(np.quantile(luminance[54:74, :], 0.90)) < 156.0


def test_late_cloud_cover_reaches_back_ridge_on_pitched_relief() -> None:
    module = _load_module()
    scene = module.SceneConfig(
        terrain_width=2200.0,
        domain=(2099.1435546875, 2553.602294921875),
        phi_deg=224.0,
        theta_deg=66.0,
        radius=2728.0,
        fov_deg=30.0,
        zscale=0.24,
        target=(1043.382101505656, 31.864142578125, 1049.2970292678915),
        sun_orbit_radius=1540.0,
    )
    terrain_mask = _pitched_relief_mask()
    cache = module._build_weather_cache((160, 120), terrain_mask)
    weather = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.92,
        sun_screen=module._sun_screen(scene, progress=0.92, width=160, height=120),
        full_size=(160, 120),
    )

    terrain = terrain_mask > 0.2
    skyline = np.argmax(terrain, axis=0)
    upper_relief = np.zeros_like(terrain, dtype=bool)
    for x, top in enumerate(skyline):
        bottom = min(terrain.shape[0], int(top) + 18)
        upper_relief[int(top):bottom, x] = terrain[int(top):bottom, x]

    assert float(weather["clouds"][upper_relief].mean()) > 0.62
    assert float(weather["terrain_canopy"][upper_relief].mean()) > 0.46
    frame = module._composite_frame(np.full((120, 160, 3), 180, dtype=np.uint8), terrain_mask=terrain_mask, weather=weather, progress=0.92)
    luminance = frame.mean(axis=2)
    assert float(np.quantile(luminance[upper_relief], 0.85)) < 156.0


def test_cloud_and_shadow_motion_stay_aligned() -> None:
    module = _load_module()
    cache, _ = _cache(module)
    scene = _scene(module)

    early = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.20,
        sun_screen=module._sun_screen(scene, progress=0.20, width=160, height=120),
        full_size=(160, 120),
    )
    late = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.70,
        sun_screen=module._sun_screen(scene, progress=0.70, width=160, height=120),
        full_size=(160, 120),
    )

    cloud_motion = _centroid(late["clouds"]) - _centroid(early["clouds"])
    shadow_motion = _centroid(late["shadow"]) - _centroid(early["shadow"])

    assert float(np.linalg.norm(cloud_motion)) > 0.5
    assert float(np.linalg.norm(shadow_motion)) > 0.5
    assert float(np.dot(cloud_motion, shadow_motion)) > 0.0


def test_late_rain_breaks_into_lower_shaft_clusters() -> None:
    module = _load_module()
    cache, _ = _cache(module)
    scene = _scene(module)

    late = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.92,
        sun_screen=module._sun_screen(scene, progress=0.92, width=160, height=120),
        full_size=(160, 120),
    )

    strong = late["rain_sheet"] > 0.05
    starts: list[float] = []
    extents: list[float] = []
    for column in strong.T:
        rows = np.flatnonzero(column)
        if rows.size == 0:
            continue
        starts.append(float(rows[0]) / strong.shape[0])
        extents.append(float(rows[-1] - rows[0] + 1) / strong.shape[0])

    yy, xx = np.nonzero(strong)
    assert extents
    assert yy.size
    assert float(np.quantile(late["rain_sheet"], 0.97)) > 0.20
    assert 0.03 < float(np.mean(strong)) < 0.18
    assert float(np.mean(starts)) > 0.44
    assert float(np.mean(extents)) < 0.32


def test_late_rain_advects_between_adjacent_frames() -> None:
    module = _load_module()
    cache, _ = _cache(module)
    scene = _scene(module)

    earlier = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.90,
        sun_screen=module._sun_screen(scene, progress=0.90, width=160, height=120),
        full_size=(160, 120),
    )
    later = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.93,
        sun_screen=module._sun_screen(scene, progress=0.93, width=160, height=120),
        full_size=(160, 120),
    )

    delta = np.abs(later["rain_sheet"] - earlier["rain_sheet"])
    assert float(delta.mean()) > 0.03
    assert float(np.quantile(delta, 0.95)) > 0.22


def test_sun_halo_tracks_sun_and_late_storm_darkens() -> None:
    module = _load_module()
    cache, terrain_mask = _cache(module)
    scene = _scene(module)

    mid = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.52,
        sun_screen=module._sun_screen(scene, progress=0.52, width=160, height=120),
        full_size=(160, 120),
    )
    late = module._projected_cloud_planes(
        cache,
        scene=scene,
        progress=0.92,
        sun_screen=module._sun_screen(scene, progress=0.92, width=160, height=120),
        full_size=(160, 120),
    )

    mid_halo_x = int(np.unravel_index(np.argmax(mid["sun_halo"]), mid["sun_halo"].shape)[1])
    late_halo_x = int(np.unravel_index(np.argmax(late["sun_halo"]), late["sun_halo"].shape)[1])
    assert mid_halo_x < late_halo_x
    assert float(late["dark_core"].mean()) > float(mid["dark_core"].mean()) * 1.7
    assert float(np.quantile(late["rain_sheet"], 0.97)) > 0.20
    assert float(late["streaks"].max()) > 0.08

    base_rgb = np.full((120, 160, 3), 96, dtype=np.uint8)
    frame = module._composite_frame(base_rgb, terrain_mask=terrain_mask, weather=late, progress=0.92)
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
