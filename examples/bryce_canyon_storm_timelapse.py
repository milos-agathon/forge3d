#!/usr/bin/env python3
"""Render a Bryce Canyon storm timelapse with projected cloud sheets and timed rain."""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = PROJECT_ROOT / "assets" / "tif" / "Bryce_Canyon.tif"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples" / "out" / "bryce_canyon_storm"


@dataclass(frozen=True)
class PreparedDem:
    heightmap: np.ndarray
    terrain_width: float
    domain: tuple[float, float]
    focus_uv: tuple[float, float]


@dataclass(frozen=True)
class SceneConfig:
    terrain_width: float
    domain: tuple[float, float]
    phi_deg: float
    theta_deg: float
    radius: float
    fov_deg: float
    zscale: float
    target: tuple[float, float, float]
    sun_orbit_radius: float


@dataclass(frozen=True)
class SunState:
    azimuth_deg: float
    elevation_deg: float
    intensity: float


@dataclass(frozen=True)
class WeatherCache:
    cloud_main: np.ndarray
    cloud_secondary: np.ndarray
    cloud_detail: np.ndarray
    cloud_wisp: np.ndarray
    rain_seed: np.ndarray
    rain_detail: np.ndarray
    terrain_soft: np.ndarray
    sky_mask: np.ndarray
    width: int
    height: int


def _terrain_state(scene: SceneConfig, *, background: tuple[float, float, float], sun: SunState | None = None) -> dict[str, object]:
    state_sun = sun or SunState(azimuth_deg=100.0, elevation_deg=12.0, intensity=0.62)
    return {
        "cmd": "set_terrain",
        "phi": scene.phi_deg,
        "theta": scene.theta_deg,
        "radius": scene.radius,
        "fov": scene.fov_deg,
        "sun_azimuth": state_sun.azimuth_deg,
        "sun_elevation": state_sun.elevation_deg,
        "sun_intensity": state_sun.intensity,
        "ambient": 0.08,
        "zscale": scene.zscale,
        "shadow": 0.98,
        "background": list(background),
        "water_level": -999999.0,
        "target": list(scene.target),
    }


def _terrain_pbr_state(*, sky_enabled: bool, volumetrics_enabled: bool) -> dict[str, object]:
    return {
        "cmd": "set_terrain_pbr",
        "enabled": True,
        "shadow_technique": "pcss",
        "shadow_map_res": 4096,
        "exposure": 0.64,
        "msaa": 8,
        "ibl_intensity": 0.0,
        "normal_strength": 1.4,
        "materials": {"rock_enabled": True, "rock_slope_min": 31.0, "wetness_enabled": True, "wetness_strength": 0.52},
        "tonemap": {"operator": "aces", "white_point": 5.2, "white_balance_enabled": True, "temperature": 5700.0, "tint": 0.0},
        "volumetrics": {"enabled": volumetrics_enabled, "mode": "height", "density": 0.010, "height_falloff": 0.15, "scattering": 0.50, "absorption": 0.24, "light_shafts": True, "shaft_intensity": 0.24, "steps": 32, "half_res": False},
        "sky": {"enabled": sky_enabled, "turbidity": 5.9, "ground_albedo": 0.24, "sun_intensity": 0.40, "aerial_perspective": True, "sky_exposure": 0.60},
        "lens_effects": {"enabled": False},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--size", nargs=2, type=int, default=(1600, 900), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--settle", type=float, default=0.08)
    parser.add_argument("--overlay-max-dim", type=int, default=2200)
    return parser.parse_args()


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    return np.zeros_like(values, dtype=np.float32) if hi - lo <= 1e-6 else (values - lo) / (hi - lo)


def _smoothstep(edge0: float, edge1: float, values: np.ndarray | float) -> np.ndarray:
    edge0_arr = np.asarray(edge0, dtype=np.float32)
    edge1_arr = np.asarray(edge1, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    t = np.clip((values_arr - edge0_arr) / np.maximum(edge1_arr - edge0_arr, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rain_envelope(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    build = float(_smoothstep(0.58, 0.80, p))
    downpour = float(_smoothstep(0.78, 0.95, p))
    return float(np.clip(build * (0.28 + 0.72 * downpour), 0.0, 1.0))


def _storm_darkness(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    return float(_smoothstep(0.32, 0.82, p))


def _sun_visibility(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    warm = float(np.clip(1.0 - _sun_state(p).elevation_deg / 52.0, 0.0, 1.0))
    clear = float(_smoothstep(0.74, 0.98, p))
    return float(np.clip((0.28 + 0.72 * warm) * (0.26 + 0.74 * max(clear, 1.0 - _rain_envelope(p) * 0.75)), 0.0, 1.0))


def _resize_float(field: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.round(np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def _blur_float(field: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return np.asarray(field, dtype=np.float32)
    image = Image.fromarray(np.round(np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0


def _fractal_field(width: int, height: int, *, seed: int, octaves: tuple[tuple[int, float], ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((height, width), dtype=np.float32)
    total = 0.0
    for divisor, weight in octaves:
        coarse = rng.random((max(4, height // divisor), max(4, width // divisor)), dtype=np.float32)
        image = Image.fromarray(np.round(coarse * 255.0).astype(np.uint8), mode="L")
        acc += (np.asarray(image.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0) * weight
        total += weight
    return _normalize01(acc / max(total, 1e-6))


def _motion_smear(field: np.ndarray, *, dx: int, dy: int, taps: int) -> np.ndarray:
    acc = np.zeros_like(field, dtype=np.float32)
    for index in range(max(taps, 1)):
        frac = index / max(taps - 1, 1)
        acc += np.roll(field, shift=(int(round(dy * frac)), int(round(dx * frac))), axis=(0, 1))
    return acc / float(max(taps, 1))


def _translate_no_wrap(field: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    dst = np.zeros_like(src, dtype=np.float32)
    h, w = src.shape
    sx0, sx1 = max(0, -dx), min(w, w - dx) if dx >= 0 else w
    sy0, sy1 = max(0, -dy), min(h, h - dy) if dy >= 0 else h
    dx0, dx1 = max(0, dx), min(w, w + dx) if dx < 0 else w
    dy0, dy1 = max(0, dy), min(h, h + dy) if dy < 0 else h
    if sx0 < sx1 and sy0 < sy1 and dx0 < dx1 and dy0 < dy1:
        dst[dy0:dy1, dx0:dx1] = src[sy0:sy1, sx0:sx1]
    return dst


def _affine_float(field: np.ndarray, *, x_shear: float = 0.0, x_shift: float = 0.0) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if abs(x_shear) <= 1e-6 and abs(x_shift) <= 1e-6:
        return src
    image = Image.fromarray(np.round(np.clip(src, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    transformed = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, -float(x_shear), -float(x_shift), 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(transformed, dtype=np.float32) / 255.0


def _sky_band_mask(terrain_mask: np.ndarray) -> np.ndarray:
    terrain = np.asarray(terrain_mask, dtype=np.float32) > 0.35
    height, width = terrain.shape
    if not np.any(terrain):
        return np.zeros_like(terrain_mask, dtype=np.float32)
    skyline = np.argmax(terrain, axis=0).astype(np.float32)
    yy = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, axis=1)
    band = 1.0 - _smoothstep(skyline[None, :] + max(4.0, height * 0.015), skyline[None, :] + max(42.0, height * 0.18), yy)
    band *= np.clip(1.0 - terrain_mask, 0.0, 1.0)
    return np.clip(_blur_float(band, 2.0), 0.0, 1.0)


def _load_dem_preview(path: Path, max_dim: int) -> PreparedDem:
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise SystemExit("This example requires rasterio to read Bryce_Canyon.tif.") from exc

    with rasterio.open(path) as src:
        scale = min(1.0, float(max_dim) / max(src.width, src.height, 1))
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))
        data = src.read(1, out_shape=(out_h, out_w), masked=True, resampling=Resampling.bilinear)

    filled = np.asarray(data.filled(np.nan), dtype=np.float32)
    finite = filled[np.isfinite(filled)]
    if finite.size == 0:
        raise SystemExit(f"DEM contains no finite samples: {path}")
    heightmap = np.where(np.isfinite(filled), filled, float(np.nanmin(filled))).astype(np.float32)
    gy, gx = np.gradient(heightmap)
    relief = np.hypot(gx, gy) + np.abs(np.gradient(gx, axis=1) + np.gradient(gy, axis=0)) * 0.7
    weights = np.clip(relief - float(np.percentile(relief, 72.0)), 0.0, None)
    rows, cols = np.indices(heightmap.shape, dtype=np.float32)
    if float(weights.sum()) > 1e-6:
        focus_row = float((weights * rows).sum() / weights.sum())
        focus_col = float((weights * cols).sum() / weights.sum())
    else:
        focus_row = heightmap.shape[0] * 0.5
        focus_col = heightmap.shape[1] * 0.5
    return PreparedDem(
        heightmap=heightmap,
        terrain_width=float(max(heightmap.shape[1], heightmap.shape[0])),
        domain=(float(np.min(heightmap)), float(np.max(heightmap))),
        focus_uv=(focus_col / max(heightmap.shape[1] - 1, 1), focus_row / max(heightmap.shape[0] - 1, 1)),
    )


def _build_overlay(heightmap: np.ndarray) -> np.ndarray:
    norm = _normalize01(heightmap)
    gy, gx = np.gradient(heightmap)
    slope = _normalize01(np.hypot(gx, gy))
    noise = _fractal_field(heightmap.shape[1], heightmap.shape[0], seed=17, octaves=((26, 1.0), (13, 0.6), (7, 0.3)))
    strata = 0.5 + 0.5 * np.sin(norm * 40.0 + noise * 5.0 + slope * 6.0)
    stops = np.array([0.0, 0.28, 0.58, 0.82, 1.0], dtype=np.float32)
    palette = np.array([(0.37, 0.22, 0.13), (0.58, 0.33, 0.19), (0.79, 0.55, 0.34), (0.92, 0.78, 0.58), (0.98, 0.94, 0.86)], dtype=np.float32)
    rgb = np.stack([np.interp(norm.ravel(), stops, palette[:, i]) for i in range(3)], axis=1).reshape(heightmap.shape + (3,))
    rgb *= 0.90 + 0.22 * strata[..., None]
    rgb = rgb * (1.0 - slope[..., None] * 0.14) + np.array([0.94, 0.86, 0.74], dtype=np.float32) * slope[..., None] * 0.14
    return np.dstack([np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), np.full(heightmap.shape, 255, dtype=np.uint8)])


def _build_scene(prepared: PreparedDem) -> SceneConfig:
    u, v = prepared.focus_uv
    row = int(round(v * max(prepared.heightmap.shape[0] - 1, 1)))
    col = int(round(u * max(prepared.heightmap.shape[1] - 1, 1)))
    zscale = 0.24
    target = (
        u * prepared.terrain_width,
        max(float(prepared.heightmap[row, col] - prepared.domain[0]), 0.0) * zscale * 0.55,
        v * prepared.terrain_width,
    )
    return SceneConfig(
        terrain_width=prepared.terrain_width,
        domain=prepared.domain,
        phi_deg=224.0,
        theta_deg=66.0,
        radius=prepared.terrain_width * 1.24,
        fov_deg=30.0,
        zscale=zscale,
        target=target,
        sun_orbit_radius=prepared.terrain_width * 0.70,
    )


def _project_world_to_screen(world_point: np.ndarray, *, scene: SceneConfig, width: int, height: int) -> tuple[float, float]:
    target = np.asarray(scene.target, dtype=np.float32)
    phi, theta = np.deg2rad([scene.phi_deg, scene.theta_deg])
    eye = target + np.array(
        [scene.radius * np.sin(theta) * np.cos(phi), scene.radius * np.cos(theta), scene.radius * np.sin(theta) * np.sin(phi)],
        dtype=np.float32,
    )
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    side = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    side /= max(float(np.linalg.norm(side)), 1e-6)
    up = np.cross(side, forward)
    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = side, up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    focal = 1.0 / np.tan(np.deg2rad(scene.fov_deg) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = focal / max(float(width) / max(float(height), 1.0), 1e-6)
    proj[1, 1] = focal
    proj[2, 2], proj[2, 3], proj[3, 2] = -1.0001, -0.10001, -1.0
    clip = proj @ view @ np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float32)
    if abs(float(clip[3])) <= 1e-6:
        return float(width) * 0.5, float(height) * 0.2
    ndc = clip[:3] / clip[3]
    return (float(ndc[0]) * 0.5 + 0.5) * float(width), (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(height)


def _sun_state(progress: float) -> SunState:
    p = float(np.clip(progress, 0.0, 1.0))
    day_arc = math.sin(math.pi * p)
    storm_filter = float(1.0 - 0.28 * _smoothstep(0.18, 0.56, p) - 0.28 * _storm_darkness(p) - 0.20 * _rain_envelope(p))
    return SunState(
        azimuth_deg=78.0 + p * 232.0,
        elevation_deg=5.0 + 48.0 * day_arc ** 1.04,
        intensity=float((0.30 + 0.46 * day_arc ** 1.08) * max(storm_filter, 0.30)),
    )


def _sun_screen(scene: SceneConfig, *, progress: float, width: int, height: int) -> tuple[float, float]:
    sun = _sun_state(progress)
    azimuth, elevation = np.deg2rad([sun.azimuth_deg, sun.elevation_deg])
    sun_dir = np.array([np.cos(elevation) * np.sin(azimuth), np.sin(elevation), np.cos(elevation) * np.cos(azimuth)], dtype=np.float32)
    point = np.asarray(scene.target, dtype=np.float32) + sun_dir * scene.sun_orbit_radius
    return _project_world_to_screen(point, scene=scene, width=width, height=height)


def _configure_viewer(viewer: f3d.ViewerHandle, overlay_path: Path, scene: SceneConfig) -> None:
    viewer.send_ipc(_terrain_state(scene, background=(0.10, 0.12, 0.16)))
    viewer.send_ipc(_terrain_pbr_state(sky_enabled=True, volumetrics_enabled=True))
    viewer.load_overlay("bryce_albedo", overlay_path, extent=(0.0, 0.0, 1.0, 1.0), opacity=1.0, preserve_colors=False)
    viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
    viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
    viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": False})


def _capture_terrain_mask(viewer: f3d.ViewerHandle, scene: SceneConfig, *, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    with tempfile.TemporaryDirectory(prefix="forge3d_bryce_mask_") as temp_dir_name:
        path = Path(temp_dir_name) / "mask.png"
        viewer.send_ipc(_terrain_state(scene, background=(1.0, 0.0, 1.0)))
        viewer.send_ipc(_terrain_pbr_state(sky_enabled=False, volumetrics_enabled=False))
        time.sleep(0.3)
        viewer.snapshot(path, width=width, height=height)
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        terrain = np.any(np.abs(rgb.astype(np.int16) - np.array([255, 0, 255], dtype=np.int16)) > 8, axis=2).astype(np.float32)
    viewer.send_ipc(_terrain_state(scene, background=(0.10, 0.12, 0.16)))
    viewer.send_ipc(_terrain_pbr_state(sky_enabled=True, volumetrics_enabled=True))
    return np.clip(_blur_float(terrain, 1.6), 0.0, 1.0)


def _build_weather_cache(size: tuple[int, int], terrain_mask_full: np.ndarray, *, scale: float = 0.70) -> WeatherCache:
    width = max(96, int(round(size[0] * scale)))
    height = max(96, int(round(size[1] * scale)))
    terrain_mask = _resize_float(terrain_mask_full, (width, height))
    return WeatherCache(
        cloud_main=_fractal_field(width, height, seed=23, octaves=((40, 1.0), (20, 0.72), (10, 0.34))),
        cloud_secondary=_fractal_field(width, height, seed=41, octaves=((30, 1.0), (15, 0.66), (7, 0.28))),
        cloud_detail=_fractal_field(width, height, seed=59, octaves=((18, 1.0), (9, 0.60), (4, 0.26))),
        cloud_wisp=_fractal_field(width, height, seed=77, octaves=((56, 1.0), (28, 0.44), (14, 0.18))),
        rain_seed=_fractal_field(width, height, seed=103, octaves=((24, 1.0), (12, 0.50), (6, 0.20))),
        rain_detail=_fractal_field(width, height, seed=151, octaves=((12, 1.0), (6, 0.58), (3, 0.24))),
        terrain_soft=_blur_float(terrain_mask, 2.2),
        sky_mask=_sky_band_mask(terrain_mask),
        width=width,
        height=height,
    )


def _perspective_coefficients(src_points: list[tuple[float, float]], dst_points: list[tuple[float, float]]) -> list[float]:
    matrix, rhs = [], []
    for (dx, dy), (sx, sy) in zip(dst_points, src_points, strict=True):
        matrix += [[dx, dy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sx * dy], [0.0, 0.0, 0.0, dx, dy, 1.0, -sy * dx, -sy * dy]]
        rhs += [sx, sy]
    return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64)).astype(np.float32).tolist()


def _projected_cloud_planes(cache: WeatherCache, *, scene: SceneConfig, progress: float, sun_screen: tuple[float, float], full_size: tuple[int, int]) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0 : cache.height, 0 : cache.width].astype(np.float32)
    xx /= max(cache.width - 1, 1)
    yy /= max(cache.height - 1, 1)
    coverage = float(_smoothstep(0.02, 0.88, progress))
    storm_dark = _storm_darkness(progress)
    wind_dx_a = int(round(progress * cache.width * 0.20))
    wind_dy_a = int(round(progress * cache.height * 0.06))
    wind_dx_b = int(round(cache.width * 0.10 + progress * cache.width * 0.28))
    wind_dy_b = int(round(cache.height * 0.03 + progress * cache.height * 0.07))
    main = np.roll(cache.cloud_main, shift=(wind_dy_a, wind_dx_a), axis=(0, 1))
    secondary = np.roll(cache.cloud_secondary, shift=(wind_dy_b, wind_dx_b), axis=(0, 1))
    detail = np.roll(cache.cloud_detail, shift=(int(round(progress * cache.height * 0.10)), int(round(progress * cache.width * 0.34))), axis=(0, 1))
    wisps = np.roll(cache.cloud_wisp, shift=(int(round(progress * cache.height * 0.05)), int(round(progress * cache.width * 0.42))), axis=(0, 1))
    macro_a = _normalize01(_blur_float(main, 4.8))
    macro_b = _normalize01(_blur_float(secondary, 4.2))
    overcast = _normalize01(_blur_float(macro_a * 0.56 + macro_b * 0.44, 5.8))
    billow = _normalize01(_blur_float(main * 0.44 + secondary * 0.34 + detail * 0.22, 2.4))
    feather = _normalize01(_blur_float(wisps * 0.72 + detail * 0.28, 3.0))
    billow_caps = _normalize01(_blur_float(np.clip(billow * 0.72 + detail * 0.28 - feather * 0.14, 0.0, 1.0), 1.4))
    billow_shell = _normalize01(_blur_float(np.clip(macro_a * 0.42 + macro_b * 0.34 + feather * 0.24, 0.0, 1.0), 3.8))
    tuft_noise = _normalize01(_blur_float(np.clip(detail * 0.58 + wisps * 0.42, 0.0, 1.0), 1.1))
    puff_break = _smoothstep(0.28, 0.82, _normalize01(_blur_float(np.clip(wisps * 0.64 + detail * 0.36, 0.0, 1.0), 1.8)))
    axis_a = (yy * 0.92 + xx * 0.18 + progress * 0.18) % 1.0
    axis_b = (yy * 0.86 - xx * 0.10 + progress * 0.12 + 0.24) % 1.0
    band = lambda axis, center, width: np.exp(-((np.abs(((axis - center + 0.5) % 1.0) - 0.5) / max(width, 1e-4)) ** 2)).astype(np.float32)
    swath_a = band(axis_a, 0.24 + 0.08 * coverage, 0.30)
    swath_b = band(axis_b, 0.58 - 0.05 * coverage, 0.26)
    swath_c = band((axis_a * 0.68 + axis_b * 0.32) % 1.0, 0.80, 0.20)
    front = np.clip(xx * 0.72 + yy * 0.38 + progress * 0.55, 0.0, 1.0)
    puff_low = _smoothstep(0.42 - 0.08 * coverage, 0.88, billow_caps * 0.74 + macro_a * 0.18 + tuft_noise * 0.08)
    puff_mid = _smoothstep(0.44 - 0.06 * coverage, 0.90, billow * 0.56 + macro_b * 0.26 + tuft_noise * 0.18)
    puff_high = _smoothstep(0.50 - 0.04 * coverage, 0.92, feather * 0.58 + detail * 0.26 + billow_shell * 0.16)
    deck_low = np.clip((swath_a * 0.72 + swath_b * 0.18 + front * 0.10) * (0.26 + 0.74 * puff_low) * (0.86 + 0.14 * tuft_noise) * (1.0 - puff_break * 0.22), 0.0, 1.0)
    deck_mid = np.clip((swath_b * 0.62 + swath_c * 0.28 + (1.0 - front) * 0.10) * (0.24 + 0.76 * puff_mid) * (0.82 + 0.18 * billow_shell) * (1.0 - puff_break * 0.18), 0.0, 1.0)
    deck_high = np.clip((swath_c * 0.58 + swath_a * 0.22 + 0.20) * (0.18 + 0.82 * puff_high) * (0.74 + 0.26 * tuft_noise), 0.0, 1.0)
    overcast_alpha = _blur_float(_smoothstep(0.28 - 0.08 * coverage, 0.92, overcast * 0.68 + billow_shell * 0.32) * (0.28 + 0.22 * coverage + 0.14 * storm_dark), 5.4)
    overcast_shell = _blur_float(_smoothstep(0.20, 0.86, overcast * 0.56 + feather * 0.28 + macro_b * 0.16) * (0.20 + 0.12 * coverage), 7.8)
    alpha_a = _blur_float(_smoothstep(0.24 - 0.06 * coverage, 0.82, deck_low) * (0.24 + 0.76 * billow_caps), 2.0) * (0.70 + 0.20 * coverage)
    alpha_b = _blur_float(_smoothstep(0.28 - 0.06 * coverage, 0.84, deck_mid) * (0.22 + 0.78 * puff_mid), 1.8) * (0.58 + 0.22 * coverage)
    alpha_c = _blur_float(_smoothstep(0.38 - 0.04 * coverage, 0.88, deck_high) * (0.18 + 0.82 * puff_high), 1.3) * (0.32 + 0.16 * (1.0 - storm_dark * 0.35))
    relief_height = max((scene.domain[1] - scene.domain[0]) * scene.zscale, 1.0)
    heights = (relief_height * 0.96, relief_height * 1.34, relief_height * 1.92)
    src_points = [(0.0, 0.0), (float(cache.width - 1), 0.0), (float(cache.width - 1), float(cache.height - 1)), (0.0, float(cache.height - 1))]

    def warp(alpha: np.ndarray, plane_y: float) -> np.ndarray:
        # Let elevated decks spill beyond the terrain footprint so the back ridge stays under the canopy.
        plane_factor = float(np.clip(plane_y / max(heights[-1], 1e-6), 0.0, 1.0))
        footprint_pad = scene.terrain_width * (0.12 + 0.10 * coverage + 0.12 * storm_dark + 0.08 * plane_factor)
        side_pad = footprint_pad * 0.90
        front_pad = footprint_pad * 0.56
        back_pad = footprint_pad * 1.45
        quad_world = [
            np.array([-side_pad, plane_y, -front_pad], dtype=np.float32),
            np.array([scene.terrain_width + side_pad, plane_y, -front_pad], dtype=np.float32),
            np.array([scene.terrain_width + side_pad, plane_y, scene.terrain_width + back_pad], dtype=np.float32),
            np.array([-side_pad, plane_y, scene.terrain_width + back_pad], dtype=np.float32),
        ]
        quad = [_project_world_to_screen(point, scene=scene, width=full_size[0], height=full_size[1]) for point in quad_world]
        coeffs = _perspective_coefficients(src_points, quad)
        image = Image.fromarray(np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        return np.asarray(image.transform(full_size, Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC), dtype=np.float32) / 255.0

    plane_a, plane_b, plane_c, plane_overcast, plane_shell = (
        warp(alpha_a, heights[0]),
        warp(alpha_b, heights[1]),
        warp(alpha_c, heights[2]),
        warp(overcast_alpha, heights[1]),
        warp(overcast_shell, heights[2]),
    )
    rain = _rain_envelope(progress)
    terrain_soft_full = _resize_float(cache.terrain_soft, full_size)
    terrain_cover = _smoothstep(0.03, 0.18, terrain_soft_full)
    sky_full = _resize_float(cache.sky_mask, full_size)
    frame_cover = np.clip(terrain_cover * 0.98 + sky_full * 0.44, 0.0, 1.0)
    blanket_noise = _blur_float(_resize_float(overcast * 0.62 + billow * 0.38, full_size), 10.5)
    texture_full = _blur_float(_resize_float(np.clip(billow_caps * 0.68 + tuft_noise * 0.32, 0.0, 1.0), full_size), 5.2)
    yy_full, xx_full = np.mgrid[0 : full_size[1], 0 : full_size[0]].astype(np.float32)
    terrain_columns = terrain_soft_full > 0.20
    terrain_has = np.any(terrain_columns, axis=0)
    terrain_line = np.argmax(terrain_columns, axis=0).astype(np.float32)
    terrain_line = np.where(terrain_has, terrain_line, full_size[1] * 0.72)
    clouds = np.clip((plane_overcast * 0.58 + plane_shell * 0.28 + plane_a * 1.00 + plane_b * 0.80 + plane_c * 0.38) * (0.88 + 0.26 * texture_full), 0.0, 1.0)
    cloud_fill = _blur_float(np.clip(plane_overcast * 0.92 + plane_shell * 0.40 + plane_a * 0.56 + plane_b * 0.42 + plane_c * 0.18, 0.0, 1.0), 4.4) * (0.12 + 0.08 * coverage + 0.10 * storm_dark)
    canopy_seed = np.clip(plane_overcast * 0.98 + plane_shell * 0.74 + plane_a * 0.62 + plane_b * 0.46 + plane_c * 0.22, 0.0, 1.0)
    terrain_canopy = _blur_float(canopy_seed * (0.60 + 0.40 * texture_full), 8.2) * frame_cover * (0.06 + 0.10 * coverage + 0.26 * storm_dark)
    terrain_blanket = terrain_cover * (0.04 + 0.08 * coverage + 0.32 * storm_dark) * (0.76 + 0.24 * blanket_noise)
    ridge_depth = np.maximum(yy_full - terrain_line[None, :], 0.0)
    ridge_band = (1.0 - _smoothstep(full_size[1] * 0.03, full_size[1] * (0.20 + 0.04 * storm_dark), ridge_depth)) * terrain_cover
    ridge_seed = _blur_float(np.clip(plane_overcast * 0.98 + plane_shell * 0.76 + plane_b * 0.40 + plane_a * 0.18, 0.0, 1.0), 5.8)
    ridge_blanket = ridge_seed * ridge_band * (0.10 + 0.10 * coverage + 0.28 * storm_dark)
    terrain_canopy = np.clip(terrain_canopy + terrain_blanket + ridge_blanket, 0.0, 1.0)
    clouds = np.clip(clouds + cloud_fill + terrain_canopy, 0.0, 1.0)
    cloud_mass = _blur_float(np.clip(plane_overcast * 0.54 + plane_shell * 0.28 + plane_a * 0.92 + plane_b * 0.78 + plane_c * 0.34, 0.0, 1.0), 2.4)
    dense_base = np.clip(cloud_mass * (0.78 + 0.28 * texture_full) + terrain_canopy * 0.98 - (0.20 - 0.12 * storm_dark), 0.0, 1.0)
    dark_mass = _blur_float(dense_base, 2.1)
    dark_core = np.clip(_smoothstep(0.28 - 0.04 * coverage, 0.74 - 0.04 * storm_dark, dark_mass) * (0.10 + 0.28 * coverage + 0.34 * storm_dark + 0.28 * rain), 0.0, 1.0)
    edges = np.clip(_blur_float(clouds, 0.35) - _blur_float(clouds, 1.45), 0.0, 1.0)
    gust_phase = progress * math.tau * 3.4
    rain_macro = _resize_float(cache.rain_seed, full_size)
    rain_fine = _resize_float(cache.rain_detail, full_size)
    rain_macro = np.roll(
        rain_macro,
        shift=(
            int(round(progress * full_size[1] * (0.64 + 0.86 * rain))),
            int(round(progress * full_size[0] * 0.18 + math.sin(gust_phase) * full_size[0] * 0.016)),
        ),
        axis=(0, 1),
    )
    rain_fine = np.roll(
        rain_fine,
        shift=(
            int(round(progress * full_size[1] * (1.16 + 1.72 * rain))),
            int(round(progress * full_size[0] * 0.28 + math.sin(gust_phase * 1.7 + 0.8) * full_size[0] * 0.024)),
        ),
        axis=(0, 1),
    )
    rain_anchor = np.clip(_blur_float(np.clip(plane_a * 0.82 + plane_b * 0.70 + terrain_canopy * 0.26, 0.0, 1.0), 0.8), 0.0, 1.0)
    wind_lean = max(1, int(round(full_size[0] * 0.0062)))
    shaft_noise = _normalize01(_motion_smear(rain_macro * 0.64 + rain_fine * 0.36, dx=wind_lean, dy=max(6, full_size[1] // 24), taps=6))
    filament_field = _normalize01(_motion_smear(rain_fine, dx=max(1, wind_lean // 2), dy=max(8, full_size[1] // 22), taps=8))
    shaft_cells = np.clip(_blur_float(rain_anchor * (0.34 + 0.66 * dense_base) * (0.32 + 0.68 * rain_macro), 1.1), 0.0, 1.0)
    shaft_cells = np.clip(_smoothstep(0.16 - 0.05 * rain, 0.64, shaft_cells), 0.0, 1.0)

    cloud_columns = shaft_cells > (0.22 - 0.04 * rain)
    cloud_has = np.any(cloud_columns, axis=0)
    cloud_top = np.argmax(cloud_columns, axis=0).astype(np.float32)
    cloud_bottom = (full_size[1] - 1 - np.argmax(cloud_columns[::-1], axis=0)).astype(np.float32)
    cloud_span = np.maximum(cloud_bottom - cloud_top, 0.0)
    cloud_base = cloud_top + cloud_span * (0.34 + 0.10 * rain)
    cloud_base = np.where(cloud_has, cloud_base, full_size[1] * 0.34)

    fall_budget = np.maximum(terrain_line - cloud_base, full_size[1] * 0.12)
    length_seed = _smoothstep(0.28, 0.86, np.max(rain_macro * 0.70 + rain_fine * 0.30, axis=0))
    shaft_length = np.clip(
        fall_budget * (0.54 + 0.72 * length_seed) + full_size[1] * (0.04 + 0.08 * rain),
        full_size[1] * 0.12,
        full_size[1] * 0.70,
    )
    fall = (yy_full - cloud_base[None, :]) / np.maximum(shaft_length[None, :], 1.0)
    start_fade = _smoothstep(0.01, 0.14, fall)
    tail_fade = 1.0 - _smoothstep(0.78, 1.04, fall)
    lower_growth = 0.52 + 0.48 * _smoothstep(0.10, 0.56, fall)
    breakup_source = np.maximum.accumulate(shaft_cells * (0.48 + 0.52 * shaft_noise), axis=0)
    breakup = _smoothstep(0.28, 0.76, _blur_float(breakup_source, 0.60))
    rain_mask = np.clip(start_fade * tail_fade * lower_growth * breakup * rain * (0.32 + 0.68 * shaft_noise), 0.0, 1.0)
    wind_push_px = full_size[0] * (0.050 + 0.040 * rain)
    gust_px = full_size[0] * (0.016 * math.sin(gust_phase) + 0.010 * math.sin(gust_phase * 0.63 + 0.8))
    shear_far = 0.020 + 0.010 * rain
    shear_near = 0.030 + 0.016 * rain

    sheet_dx = max(4, wind_lean * 3)
    sheet_seed = _blur_float(rain_mask, 1.1) * (0.40 + 0.60 * _smoothstep(0.34, 0.82, shaft_noise))
    sheet_far = _affine_float(sheet_seed, x_shear=shear_far, x_shift=wind_push_px * progress + gust_px)
    sheet_mid = _affine_float(np.roll(sheet_seed, shift=(0, max(1, wind_lean)), axis=(0, 1)), x_shear=shear_near, x_shift=wind_push_px * (0.78 + 0.22 * progress) + gust_px * 1.6)
    rain_sheet_raw = _motion_smear(sheet_far, dx=sheet_dx, dy=max(8, full_size[1] // 24), taps=7) * 0.56
    rain_sheet_raw += _motion_smear(sheet_mid, dx=sheet_dx + max(2, wind_lean), dy=max(10, full_size[1] // 22), taps=8) * 0.62
    rain_sheet = np.clip(_blur_float(rain_sheet_raw, 1.10) * 0.74 + _blur_float(rain_sheet_raw, 2.20) * 0.34, 0.0, 1.0)
    rain_sheet = np.clip((rain_sheet - 0.02) / 0.72, 0.0, 1.0)

    streak_source = _smoothstep(0.80 - 0.08 * rain, 0.965, filament_field * 0.76 + shaft_noise * 0.24)
    streak_source *= rain_mask * _smoothstep(0.08, 0.76, fall) * (1.0 - _smoothstep(0.84, 1.06, fall))
    streak_far_seed = _affine_float(streak_source, x_shear=shear_near * 1.10, x_shift=wind_push_px * 1.12 + gust_px * 1.4)
    streak_near_seed = _affine_float(np.roll(streak_source, shift=(0, max(1, wind_lean * 2)), axis=(0, 1)), x_shear=shear_near * 1.45, x_shift=wind_push_px * 1.42 + gust_px * 2.0)
    streaks_far = _motion_smear(streak_far_seed, dx=sheet_dx + max(3, wind_lean), dy=max(18, full_size[1] // 10), taps=16)
    streaks_near = _motion_smear(streak_near_seed, dx=sheet_dx + max(5, wind_lean * 2), dy=max(24, full_size[1] // 8), taps=18)
    streaks = np.clip(_blur_float(streaks_far * 0.40 + streaks_near * 0.72, 0.10), 0.0, 1.0)

    front_seed = rain_mask * _smoothstep(0.54, 0.96, shaft_noise) * _smoothstep(0.22, 0.74, fall)
    front_seed = _affine_float(front_seed, x_shear=shear_far * 1.18, x_shift=wind_push_px * 0.94 + gust_px * 1.2)
    rain_front = np.clip(_blur_float(_motion_smear(front_seed, dx=sheet_dx, dy=max(12, full_size[1] // 16), taps=8), 0.34), 0.0, 1.0)
    rain_front *= rain * (0.10 + 0.90 * (1.0 - terrain_soft_full * 0.42))
    mist = np.clip(_blur_float(rain_sheet * terrain_soft_full, 1.8) * (0.18 + 0.82 * rain), 0.0, 1.0)

    terrain_center = _project_world_to_screen(np.asarray(scene.target, dtype=np.float32), scene=scene, width=full_size[0], height=full_size[1])
    shadow_vec = np.array([terrain_center[0] - sun_screen[0], terrain_center[1] - sun_screen[1]], dtype=np.float32)
    shadow_vec /= max(float(np.linalg.norm(shadow_vec)), 1e-6)
    px_per_world = np.linalg.norm(np.subtract(_project_world_to_screen(np.array([scene.terrain_width, heights[0], 0.0], dtype=np.float32), scene=scene, width=full_size[0], height=full_size[1]), _project_world_to_screen(np.array([0.0, heights[0], 0.0], dtype=np.float32), scene=scene, width=full_size[0], height=full_size[1]))) / max(scene.terrain_width, 1e-6)
    sun_elev = math.radians(max(_sun_state(progress).elevation_deg, 6.0))

    def shadow_from(alpha: np.ndarray, plane_height: float, weight: float) -> np.ndarray:
        shift = float(np.clip((plane_height / math.tan(sun_elev)) * px_per_world * 0.18, 3.0, min(full_size) * 0.11))
        return _blur_float(_translate_no_wrap(alpha, dx=int(round(shadow_vec[0] * shift)), dy=int(round(shadow_vec[1] * shift))), 2.2) * weight

    shadow = np.clip(shadow_from(plane_a, heights[0], 1.04) + shadow_from(plane_b, heights[1], 0.86) + shadow_from(plane_c, heights[2], 0.42), 0.0, 1.0)
    broad_shadow = _blur_float(terrain_canopy, 6.8) * terrain_cover * (0.52 + 0.16 * storm_dark)
    shadow = np.clip(_smoothstep(0.18, 0.92, shadow) + broad_shadow, 0.0, 1.0) * terrain_cover * (0.72 + 0.10 * storm_dark)
    sun_dx, sun_dy = xx_full - float(sun_screen[0]), yy_full - float(sun_screen[1])
    sun_dist = np.sqrt(sun_dx * sun_dx + sun_dy * sun_dy) / max(float(min(full_size)), 1.0)
    vis = _sun_visibility(progress)
    halo = np.exp(-((sun_dist / 0.18) ** 2)).astype(np.float32) * vis
    core = np.exp(-((sun_dist / 0.055) ** 2)).astype(np.float32) * vis
    horizon = np.exp(-(((sun_dx / max(full_size[0] * 0.26, 1.0)) ** 2) + ((sun_dy / max(full_size[1] * 0.11, 1.0)) ** 2))).astype(np.float32) * (0.62 + 0.38 * vis)
    return {
        "sky": sky_full,
        "clouds": clouds,
        "terrain_canopy": terrain_canopy,
        "dark_core": dark_core,
        "edges": edges,
        "rain_mask": rain_mask,
        "rain_sheet": rain_sheet,
        "streaks": streaks,
        "rain_front": rain_front,
        "mist": mist,
        "shadow": shadow,
        "sun_halo": halo,
        "sun_core": core,
        "horizon": horizon,
    }


def _composite_frame(base_rgb: np.ndarray, *, terrain_mask: np.ndarray, weather: dict[str, np.ndarray], progress: float) -> np.ndarray:
    rgb = np.asarray(base_rgb, dtype=np.float32) / 255.0
    terrain = terrain_mask[..., None]
    sky = weather["sky"][..., None]
    void = np.clip(1.0 - terrain - sky, 0.0, 1.0)
    rgb = rgb * (1.0 - void) + np.array([0.11, 0.13, 0.16], dtype=np.float32) * void
    yy = np.linspace(0.0, 1.0, rgb.shape[0], dtype=np.float32)[:, None, None]
    storm_dark = _storm_darkness(progress)
    sky_top = np.array([0.22, 0.25, 0.30], dtype=np.float32) * (1.0 - 0.42 * storm_dark) + np.array([0.09, 0.10, 0.12], dtype=np.float32) * (0.42 * storm_dark)
    sky_bottom = np.array([0.38, 0.40, 0.45], dtype=np.float32) * (1.0 - 0.52 * storm_dark) + np.array([0.17, 0.18, 0.22], dtype=np.float32) * (0.52 * storm_dark)
    sky_gradient = sky_top * (1.0 - yy ** 0.85) + sky_bottom * (yy ** 0.85)
    rgb = rgb * (1.0 - sky) + sky_gradient * sky
    warm = np.clip(1.0 - _sun_state(progress).elevation_deg / 48.0, 0.0, 1.0)
    rain = _rain_envelope(progress)
    clouds = weather["clouds"][..., None]
    dark = weather["dark_core"][..., None]
    sun_break = np.clip(1.0 - clouds * (0.42 + 0.18 * storm_dark) - dark * (0.24 + 0.18 * rain), 0.0, 1.0)
    rgb = np.clip(rgb + weather["horizon"][..., None] * sky * sun_break * np.array([0.12 + 0.05 * warm, 0.08 + 0.04 * warm, 0.04], dtype=np.float32), 0.0, 1.0)
    rgb = np.clip(rgb + weather["sun_halo"][..., None] * sky * sun_break * np.array([0.08 + 0.04 * warm, 0.05 + 0.03 * warm, 0.03], dtype=np.float32), 0.0, 1.0)
    cloud_light = np.array([0.90, 0.92, 0.95], dtype=np.float32) * (1.0 - 0.54 * storm_dark) + np.array([0.44, 0.47, 0.52], dtype=np.float32) * (0.54 * storm_dark)
    cloud_dark = np.array([0.24, 0.26, 0.30], dtype=np.float32) * (1.0 - 0.40 * rain) + np.array([0.11, 0.12, 0.14], dtype=np.float32) * (0.40 * rain)
    fluff = np.clip(_smoothstep(0.34, 0.84, weather["clouds"] - weather["dark_core"] * 0.28) * (1.0 - weather["dark_core"] * 0.78), 0.0, 1.0)[..., None]
    cloud_body_alpha = clouds * (sky * (0.46 + 0.18 * storm_dark) + terrain * (0.28 + 0.14 * storm_dark) + void * (0.36 + 0.16 * storm_dark))
    cloud_body_alpha = np.clip(cloud_body_alpha + dark * (0.12 + 0.16 * rain), 0.0, 0.96)
    rgb = rgb * (1.0 - cloud_body_alpha) + cloud_light * cloud_body_alpha * 0.94
    rgb = rgb * (1.0 - dark * (0.26 + 0.22 * storm_dark + 0.18 * rain)) + cloud_dark * dark * (0.24 + 0.14 * storm_dark + 0.10 * rain)
    fluff_gain = sky * (0.14 + 0.05 * (1.0 - rain)) + terrain * (0.09 + 0.04 * (1.0 - rain)) + void * (0.12 + 0.05 * (1.0 - rain))
    rgb = np.clip(rgb + fluff * np.array([0.22 + 0.08 * warm, 0.22 + 0.06 * warm, 0.24], dtype=np.float32) * fluff_gain, 0.0, 1.0)
    edge_glow = np.array([0.10 + 0.08 * warm, 0.10 + 0.06 * warm, 0.11], dtype=np.float32) * (1.0 - 0.54 * storm_dark)
    rgb = np.clip(rgb + weather["edges"][..., None] * edge_glow, 0.0, 1.0)
    rgb = np.clip(rgb + weather["sun_core"][..., None] * sky * sun_break * np.array([0.50, 0.34, 0.14], dtype=np.float32), 0.0, 1.0)
    shadow_strength = 0.08 + 0.10 * storm_dark + 0.12 * rain
    shadow = weather["shadow"][..., None] * terrain
    rgb = rgb * (1.0 - shadow * shadow_strength)
    terrain_canopy = weather.get("terrain_canopy")
    if terrain_canopy is not None:
        canopy = terrain_canopy[..., None] * terrain
        terrain_cloud_alpha = np.clip(canopy * (0.32 + 0.24 * storm_dark) + clouds * terrain * (0.08 + 0.10 * storm_dark) + dark * terrain * 0.10, 0.0, 0.78)
        terrain_cloud_color = cloud_dark * 0.78 + np.array([0.48, 0.51, 0.56], dtype=np.float32) * 0.22
        rgb = rgb * (1.0 - terrain_cloud_alpha) + terrain_cloud_color * terrain_cloud_alpha
    storm = weather["rain_sheet"][..., None] * np.clip(weather["rain_mask"][..., None] * (0.12 + sky * 0.18 + terrain * 0.88), 0.0, 1.0)
    curtain = np.clip(storm * (0.10 + 0.82 * rain), 0.0, 1.0)
    rgb = rgb * (1.0 - curtain * (0.16 + 0.34 * rain)) + np.array([0.30, 0.34, 0.38], dtype=np.float32) * curtain * (0.06 + 0.12 * rain)
    streaks = np.clip(weather["streaks"][..., None] * storm * (0.52 + 2.30 * rain), 0.0, 1.0)
    rgb = rgb * (1.0 - streaks * (0.10 + 0.22 * rain)) + np.array([0.82, 0.87, 0.92], dtype=np.float32) * streaks * (0.10 + 0.20 * rain)
    rain_front = weather["rain_front"][..., None] * np.clip(terrain * 0.96 + sky * 0.08, 0.0, 1.0)
    rgb = rgb * (1.0 - rain_front * (0.12 + 0.20 * rain)) + np.array([0.58, 0.63, 0.68], dtype=np.float32) * rain_front * (0.04 + 0.08 * rain)
    mist = weather["mist"][..., None] * terrain
    rgb = rgb * (1.0 - mist * (0.12 + 0.20 * rain)) + np.array([0.54, 0.58, 0.62], dtype=np.float32) * mist * (0.06 + 0.12 * rain)
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr[-1200:]}")
    return True


def render_timelapse(args: argparse.Namespace) -> tuple[Path | None, Path, Path]:
    if not args.dem.exists():
        raise SystemExit(f"DEM file not found: {args.dem}")
    if not hasattr(f3d, "open_viewer_async"):
        raise SystemExit("This example requires the native forge3d viewer build.")

    width, height = int(args.size[0]), int(args.size[1])
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    preview_path = output_dir / "bryce_canyon_storm_preview.png"
    video_path = output_dir / "bryce_canyon_storm_timelapse.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for existing_frame in frames_dir.glob("frame_*.png"):
        existing_frame.unlink()
    prepared = _load_dem_preview(args.dem.resolve(), int(args.overlay_max_dim))
    scene = _build_scene(prepared)
    total_frames = 1 if args.preview_only else max(1, int(round(float(args.duration) * int(args.fps))))
    representative = max(0, int(round(0.92 * max(total_frames - 1, 0))))

    with tempfile.TemporaryDirectory(prefix="forge3d_bryce_storm_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        overlay_path = temp_dir / "overlay.png"
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_build_overlay(prepared.heightmap), mode="RGBA").save(overlay_path)
        with f3d.open_viewer_async(terrain_path=args.dem.resolve(), width=width, height=height, timeout=60.0) as viewer:
            _configure_viewer(viewer, overlay_path, scene)
            time.sleep(1.6)
            terrain_mask = _capture_terrain_mask(viewer, scene, size=(width, height))
            weather_cache = _build_weather_cache((width, height), terrain_mask)
            time.sleep(0.8)
            for _ in range(3):
                viewer.send_ipc(_terrain_state(scene, background=(0.10, 0.12, 0.16)))
                viewer.send_ipc(_terrain_pbr_state(sky_enabled=True, volumetrics_enabled=True))
                time.sleep(0.2)
            for index in range(total_frames):
                progress = 0.92 if args.preview_only else index / max(total_frames - 1, 1)
                sun = _sun_state(progress)
                sun_screen = _sun_screen(scene, progress=progress, width=width, height=height)
                viewer.send_ipc({"cmd": "set_terrain_sun", "azimuth_deg": sun.azimuth_deg, "elevation_deg": sun.elevation_deg, "intensity": sun.intensity})
                time.sleep(float(args.settle))
                raw_path = raw_dir / f"raw_{index:04d}.png"
                out_path = frames_dir / f"frame_{index:04d}.png"
                viewer.snapshot(raw_path, width=width, height=height)
                base_rgb = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
                weather = _projected_cloud_planes(weather_cache, scene=scene, progress=progress, sun_screen=sun_screen, full_size=(width, height))
                final_rgb = _composite_frame(base_rgb, terrain_mask=terrain_mask, weather=weather, progress=progress)
                Image.fromarray(final_rgb, mode="RGB").save(out_path)
                if index == representative or total_frames == 1:
                    Image.fromarray(final_rgb, mode="RGB").save(preview_path)
                print(f"\r[BryceStorm] frame {index + 1}/{total_frames}", end="", flush=True)
            print()

    if args.preview_only or args.frames_only:
        return None, preview_path, frames_dir
    return (video_path if _encode_mp4(frames_dir, video_path, int(args.fps)) else None), preview_path, frames_dir


def main() -> int:
    args = _parse_args()
    video_path, preview_path, frames_dir = render_timelapse(args)
    print(f"[BryceStorm] DEM: {args.dem.resolve()}")
    print(f"[BryceStorm] Preview: {preview_path}")
    print(f"[BryceStorm] Frames: {frames_dir}")
    if video_path is not None:
        print(f"[BryceStorm] MP4: {video_path}")
    elif not args.preview_only and not args.frames_only and shutil.which("ffmpeg") is None:
        print("[BryceStorm] ffmpeg was not found, so the example left the PNG frame sequence on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
