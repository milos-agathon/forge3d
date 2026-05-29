#!/usr/bin/env python3
"""California wildfire smoke transport as geospatial raster overlays.

The default hotspots are deterministic representative California fire-source
clusters for visualization only. They are not live or current incident data.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "examples"
    / "out"
    / "california_fire_smoke"
    / "california_fire_smoke_preview.png"
)
DEFAULT_FRAME = 72
DEFAULT_FRAMES = 120
DEFAULT_FPS = 24
DEFAULT_GRID_SIZE = 1024
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 1400
DEFAULT_SEED = 202306


@dataclass(frozen=True)
class Bounds:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class FireHotspot:
    name: str
    lon: float
    lat: float
    strength: float
    radius_px_at_1024: float
    start_frame: int
    end_frame: int


DEFAULT_BOUNDS = Bounds(
    lon_min=-125.50,
    lon_max=-107.50,
    lat_min=30.80,
    lat_max=43.80,
)

_DEFAULT_HOTSPOTS: tuple[FireHotspot, ...] = (
    FireHotspot(
        name="northern_sierra_cluster",
        lon=-121.25,
        lat=40.15,
        strength=1.00,
        radius_px_at_1024=7.0,
        start_frame=0,
        end_frame=120,
    ),
    FireHotspot(
        name="klamath_cluster",
        lon=-123.10,
        lat=41.55,
        strength=0.76,
        radius_px_at_1024=6.5,
        start_frame=6,
        end_frame=120,
    ),
    FireHotspot(
        name="central_sierra_cluster",
        lon=-119.35,
        lat=37.25,
        strength=0.64,
        radius_px_at_1024=5.8,
        start_frame=14,
        end_frame=120,
    ),
    FireHotspot(
        name="southern_california_cluster",
        lon=-118.75,
        lat=34.35,
        strength=0.54,
        radius_px_at_1024=5.2,
        start_frame=24,
        end_frame=120,
    ),
)


def default_bounds() -> Bounds:
    return DEFAULT_BOUNDS


def lonlat_to_pixel(
    lon: float,
    lat: float,
    bounds: Bounds,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Map longitude/latitude to north-up raster coordinates."""
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be greater than 1")
    lon_span = float(bounds.lon_max) - float(bounds.lon_min)
    lat_span = float(bounds.lat_max) - float(bounds.lat_min)
    if lon_span <= 0.0 or lat_span <= 0.0:
        raise ValueError("bounds must have positive longitude and latitude spans")

    x = (float(lon) - float(bounds.lon_min)) / lon_span * float(width - 1)
    y = (float(bounds.lat_max) - float(lat)) / lat_span * float(height - 1)
    return x, y


def default_hotspots() -> list[FireHotspot]:
    return list(_DEFAULT_HOTSPOTS)


def load_firms_hotspots(
    csv_path: Path,
    bounds: Bounds,
    max_hotspots: int = 160,
) -> list[FireHotspot]:
    """Load user-provided FIRMS-style hotspot rows without network access."""
    path = Path(csv_path)
    rows: list[tuple[float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        names = {name.lower().strip(): name for name in reader.fieldnames}
        lon_key = names.get("longitude") or names.get("lon")
        lat_key = names.get("latitude") or names.get("lat")
        if lon_key is None or lat_key is None:
            raise ValueError("FIRMS CSV must contain longitude/lon and latitude/lat columns")

        metric_keys = [
            key
            for key in ("frp", "brightness", "bright_ti4", "bright_t31")
            if key in names
        ]
        for row in reader:
            try:
                lon = float(row[lon_key])
                lat = float(row[lat_key])
            except (KeyError, TypeError, ValueError):
                continue
            if not (bounds.lon_min <= lon <= bounds.lon_max):
                continue
            if not (bounds.lat_min <= lat <= bounds.lat_max):
                continue

            raw_strength = 1.0
            for key in metric_keys:
                value = row.get(names[key], "")
                try:
                    raw_strength = max(float(value), 0.0)
                    break
                except (TypeError, ValueError):
                    continue
            rows.append((lon, lat, raw_strength))

    if not rows:
        raise ValueError(f"no usable FIRMS points inside bounds in {path}")

    rows.sort(key=lambda item: item[2], reverse=True)
    rows = rows[: max(1, int(max_hotspots))]
    raw = np.array([item[2] for item in rows], dtype=np.float32)
    logged = np.log1p(np.maximum(raw, 0.0))
    lo, hi = np.percentile(logged, [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        normalized = np.full_like(logged, 0.55, dtype=np.float32)
    else:
        normalized = np.clip((logged - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    hotspots: list[FireHotspot] = []
    for idx, ((lon, lat, _raw), strength_norm) in enumerate(zip(rows, normalized)):
        strength = float(0.20 + 1.05 * strength_norm)
        radius = float(4.8 + 2.6 * math.sqrt(max(strength_norm, 0.0)))
        hotspots.append(
            FireHotspot(
                name=f"firms_{idx:03d}",
                lon=lon,
                lat=lat,
                strength=strength,
                radius_px_at_1024=radius,
                start_frame=0,
                end_frame=DEFAULT_FRAMES,
            )
        )
    return hotspots


def _validate_size(size: int) -> int:
    size = int(size)
    if size <= 1:
        raise ValueError("size must be greater than 1")
    return size


def _coordinate_grids(size: int, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(bounds.lon_min, bounds.lon_max, size, dtype=np.float32)
    lat = np.linspace(bounds.lat_max, bounds.lat_min, size, dtype=np.float32)
    return np.meshgrid(lon, lat)


def _pixel_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return x, y


def _smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray:
    denom = max(float(edge1) - float(edge0), 1.0e-6)
    t = np.clip((np.asarray(value, dtype=np.float32) - float(edge0)) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box_blur_3x3(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(field, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        axial = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        diagonal = (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        out = center * 0.44 + axial * 0.115 + diagonal * 0.025
    return out.astype(np.float32, copy=False)


def _border_fade(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    distance = np.minimum.reduce((x, y, width - 1.0 - x, height - 1.0 - y))
    margin = max(4.0, min(width, height) * 0.045)
    return _smoothstep(0.0, margin, distance).astype(np.float32)


def build_base_map_rgba(
    size: int,
    bounds: Bounds,
) -> np.ndarray:
    """Build a subdued synthetic Western US basemap as an RGBA image."""
    size = _validate_size(size)
    lon_grid, lat_grid = _coordinate_grids(size, bounds)
    lat_norm = (lat_grid - bounds.lat_min) / (bounds.lat_max - bounds.lat_min)

    coast_lats = np.array([31.8, 33.2, 34.6, 36.2, 38.0, 40.0, 42.3, 43.8], dtype=np.float32)
    coast_lons = np.array([-116.9, -117.4, -119.0, -121.0, -122.6, -124.0, -124.4, -124.7], dtype=np.float32)
    coast_lon = np.interp(lat_grid, coast_lats, coast_lons).astype(np.float32)
    land_amount = _smoothstep(-0.09, 0.16, lon_grid - coast_lon)

    mountains = np.exp(-((lon_grid + 119.6) ** 2) / 2.8) * np.exp(-((lat_grid - 38.3) ** 2) / 18.0)
    basin = np.exp(-((lon_grid + 114.5) ** 2) / 9.5) * np.exp(-((lat_grid - 36.4) ** 2) / 20.0)
    terrain = (
        0.45
        + 0.20 * np.sin((lon_grid + 119.7) * 1.7)
        + 0.16 * np.cos((lat_grid - 37.4) * 2.1)
        + 0.12 * np.sin((lon_grid + lat_grid) * 2.4)
        + 0.30 * mountains
        - 0.11 * basin
        + 0.08 * lat_norm
    )
    terrain = np.clip(terrain, 0.0, 1.0)

    ocean_wave = 0.5 + 0.5 * np.sin((lon_grid + 125.2) * 7.2 + (lat_grid - 32.0) * 1.4)
    ocean = np.stack(
        (
            32.0 + 8.0 * ocean_wave,
            50.0 + 10.0 * ocean_wave,
            61.0 + 11.0 * ocean_wave,
        ),
        axis=-1,
    )
    land = np.stack(
        (
            72.0 + 54.0 * terrain,
            83.0 + 48.0 * terrain,
            66.0 + 34.0 * terrain,
        ),
        axis=-1,
    )
    rgb = ocean * (1.0 - land_amount[..., None]) + land * land_amount[..., None]

    for lon_tick in range(math.ceil(bounds.lon_min), math.floor(bounds.lon_max) + 1):
        x, _ = lonlat_to_pixel(float(lon_tick), bounds.lat_min, bounds, size, size)
        col = int(round(x))
        if 0 <= col < size:
            half = max(1, size // 900)
            rgb[:, max(0, col - half) : min(size, col + half + 1), :] *= 0.91

    for lat_tick in range(math.ceil(bounds.lat_min), math.floor(bounds.lat_max) + 1):
        _, y = lonlat_to_pixel(bounds.lon_min, float(lat_tick), bounds, size, size)
        row = int(round(y))
        if 0 <= row < size:
            half = max(1, size // 900)
            rgb[max(0, row - half) : min(size, row + half + 1), :, :] *= 0.91

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255

    image = Image.fromarray(rgba, mode="RGBA")
    lines = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(lines)
    line_width = max(1, size // 380)
    coast = [
        lonlat_to_pixel(float(lon), float(lat), bounds, size, size)
        for lon, lat in zip(coast_lons, coast_lats)
    ]
    draw.line(coast, fill=(178, 191, 175, 100), width=line_width)

    state_lines = (
        [(-124.2, 42.0), (-120.0, 42.0), (-114.62, 35.0), (-114.72, 32.72), (-117.12, 32.54)],
        [(-120.0, 42.0), (-120.0, 39.0), (-120.0, 35.0)],
        [(-114.04, 37.0), (-109.05, 37.0), (-109.05, 31.3)],
        [(-124.2, 42.0), (-111.05, 42.0), (-111.05, 44.0)],
    )
    for points in state_lines:
        pixel_points = [lonlat_to_pixel(lon, lat, bounds, size, size) for lon, lat in points]
        draw.line(pixel_points, fill=(135, 143, 123, 58), width=line_width)
    return np.asarray(Image.alpha_composite(image, lines), dtype=np.uint8)


def procedural_wind_field(
    frame: int,
    shape: tuple[int, int],
    seed: int,
    base_wind_px: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic east/northeast wind with shear and curl texture."""
    height, width = shape
    if height <= 1 or width <= 1:
        raise ValueError("wind field shape must be at least 2x2")

    x, y = _pixel_grids(shape)
    xn = x / max(float(width - 1), 1.0)
    yn = y / max(float(height - 1), 1.0)
    scale = min(width, height) / 1024.0
    t = float(frame)
    tau = 2.0 * math.pi
    phase = (int(seed) % 104729) / 104729.0 * tau

    shear = yn - 0.5
    u = np.full(shape, float(base_wind_px[0]), dtype=np.float32)
    v = np.full(shape, float(base_wind_px[1]), dtype=np.float32)
    u += (0.42 * scale * np.sin(tau * (0.72 * yn + 0.006 * t) + phase)).astype(np.float32)
    u += (0.24 * scale * shear).astype(np.float32)
    v += (0.22 * scale * np.sin(tau * (0.58 * xn - 0.34 * yn + 0.004 * t) + phase * 0.7)).astype(np.float32)

    ax, ay = 2.4 * tau, 1.2 * tau
    bx, by = 1.4 * tau, 2.8 * tau
    cx, cy = 3.3 * tau, 0.8 * tau
    p1 = phase
    p2 = phase * 1.73 + 0.41
    p3 = phase * 0.61 + 1.37
    a = ax * xn + ay * yn + 0.030 * t + p1
    b = bx * xn - by * yn + 0.021 * t + p2
    c = cx * xn + cy * yn - 0.026 * t + p3

    dpsi_dy = ay * np.cos(a) - 0.55 * by * np.cos(b) - 0.35 * cy * np.sin(c)
    dpsi_dx = ax * np.cos(a) + 0.55 * bx * np.cos(b) - 0.35 * cx * np.sin(c)
    curl_strength = 0.115 * scale
    u += (curl_strength * dpsi_dy).astype(np.float32)
    v += (-curl_strength * dpsi_dx).astype(np.float32)

    mx, my = 6.2 * tau, 4.8 * tau
    micro = mx * xn + my * yn + 0.044 * t + phase * 2.4
    micro2 = 5.1 * tau * xn - 6.7 * tau * yn - 0.037 * t + phase * 0.8
    micro_dy = my * np.cos(micro) - 0.40 * 6.7 * tau * np.cos(micro2)
    micro_dx = mx * np.cos(micro) + 0.40 * 5.1 * tau * np.cos(micro2)
    micro_strength = 0.011 * scale
    u += (micro_strength * micro_dy).astype(np.float32)
    v += (-micro_strength * micro_dx).astype(np.float32)

    return u.astype(np.float32, copy=False), v.astype(np.float32, copy=False)


def bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Sample a 2D field at floating point coordinates; outside is zero."""
    source = np.asarray(field, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("field must be a 2D array")
    height, width = source.shape
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y coordinate arrays must have matching shapes")

    valid = (x_arr >= 0.0) & (x_arr <= width - 1.0) & (y_arr >= 0.0) & (y_arr <= height - 1.0)
    x0 = np.floor(x_arr).astype(np.int32)
    y0 = np.floor(y_arr).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)

    wx = x_arr - x0.astype(np.float32)
    wy = y_arr - y0.astype(np.float32)
    top = source[y0c, x0c] * (1.0 - wx) + source[y0c, x1c] * wx
    bottom = source[y1c, x0c] * (1.0 - wx) + source[y1c, x1c] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.where(valid, sampled, 0.0).astype(np.float32, copy=False)


def advect_density(
    density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Backward semi-Lagrangian advection for one frame step."""
    source = np.asarray(density, dtype=np.float32)
    if source.shape != u.shape or source.shape != v.shape:
        raise ValueError("density and wind arrays must share a shape")
    x, y = _pixel_grids(source.shape)
    return bilinear_sample(source, x - np.asarray(u, dtype=np.float32), y - np.asarray(v, dtype=np.float32))


def inject_hotspots(
    density: np.ndarray,
    hotspots: Sequence[FireHotspot],
    frame: int,
    bounds: Bounds,
) -> np.ndarray:
    """Inject compact California smoke sources with a small downwind bias."""
    out = np.asarray(density, dtype=np.float32).copy()
    height, width = out.shape
    scale = min(width, height) / 1024.0
    wind_dir = np.array([1.0, -0.42], dtype=np.float32)
    wind_dir /= np.linalg.norm(wind_dir)
    cross_dir = np.array([-wind_dir[1], wind_dir[0]], dtype=np.float32)

    for index, hotspot in enumerate(hotspots):
        if not (hotspot.start_frame <= frame <= hotspot.end_frame):
            continue
        cx, cy = lonlat_to_pixel(hotspot.lon, hotspot.lat, bounds, width, height)
        if cx < -16.0 or cx > width + 16.0 or cy < -16.0 or cy > height + 16.0:
            continue
        radius = max(1.0, float(hotspot.radius_px_at_1024) * scale)
        tail_length = max(3.8 * radius, 5.5 * scale + 4.0)
        pad = int(math.ceil(tail_length + 4.5 * radius + 2.0))
        x0 = max(0, int(math.floor(cx - pad)))
        x1 = min(width, int(math.ceil(cx + pad + tail_length)))
        y0 = max(0, int(math.floor(cy - pad - tail_length * 0.45)))
        y1 = min(height, int(math.ceil(cy + pad)))
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - np.float32(cx)
        dy = yy - np.float32(cy)
        along = dx * wind_dir[0] + dy * wind_dir[1]
        cross = dx * cross_dir[0] + dy * cross_dir[1]
        core_sigma = max(0.65, radius * 0.62)
        core = np.exp(-(dx * dx + dy * dy) / (2.0 * core_sigma * core_sigma))
        tail_gate = _smoothstep(-0.20 * radius, 0.75 * radius, along) * (along <= tail_length)
        tail_width = radius * (0.58 + 0.40 * np.clip(along / max(tail_length, 1.0), 0.0, 1.0))
        tail = (
            np.exp(-np.maximum(along, 0.0) / max(2.4 * radius, 1.0))
            * np.exp(-(cross * cross) / (2.0 * tail_width * tail_width + 1.0e-6))
            * tail_gate
        )
        pulse = 0.88 + 0.12 * math.sin(frame * 0.31 + index * 1.71 + int(DEFAULT_SEED % 97) * 0.01)
        source = float(hotspot.strength) * pulse * (0.135 * core + 0.045 * tail)
        out[y0:y1, x0:x1] += source.astype(np.float32)

    return np.clip(out, 0.0, 4.0).astype(np.float32, copy=False)


def anisotropic_downwind_stretch(
    density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Blend shifted density copies along the mean wind direction."""
    source = np.asarray(density, dtype=np.float32)
    if source.shape != u.shape or source.shape != v.shape:
        raise ValueError("density and wind arrays must share a shape")
    if amount <= 0.0 or not np.any(source > 0.0):
        return source.copy()

    mean_u = float(np.mean(u))
    mean_v = float(np.mean(v))
    norm = math.hypot(mean_u, mean_v)
    if norm <= 1.0e-6:
        return source.copy()
    dir_x = mean_u / norm
    dir_y = mean_v / norm
    height, width = source.shape
    x, y = _pixel_grids(source.shape)
    scale = min(width, height) / 1024.0
    steps = max(4, min(12, int(round(5 + 6 * scale))))
    spacing = max(1.0, 1.8 * scale)

    out = source * (1.0 - 0.10 * float(amount))
    for step in range(1, steps + 1):
        distance = spacing * float(step)
        shifted = bilinear_sample(source, x - dir_x * distance, y - dir_y * distance)
        weight = float(amount) * 0.34 / (step ** 0.72)
        out += shifted * weight
    return np.clip(out, 0.0, 4.0).astype(np.float32, copy=False)


def diffuse_decay_density(
    density: np.ndarray,
    frame: int,
    seed: int,
) -> np.ndarray:
    """Apply light diffusion and slow persistence decay."""
    source = np.asarray(density, dtype=np.float32)
    blurred = _box_blur_3x3(source, passes=1)
    phase = (int(seed) % 997) * 0.013
    mix = 0.055 + 0.018 * (0.5 + 0.5 * math.sin(frame * 0.047 + phase))
    decay = 0.986 + 0.003 * math.sin(frame * 0.031 + phase * 0.5)
    out = (source * (1.0 - mix) + blurred * mix) * decay
    out = np.where(out >= 1.0e-7, out, 0.0)
    return np.clip(out, 0.0, 4.0).astype(np.float32, copy=False)


def apply_filament_breakup(
    density: np.ndarray,
    frame: int,
    seed: int,
) -> np.ndarray:
    """Create coherent streaks, holes, and frayed margins without random flicker."""
    source = np.asarray(density, dtype=np.float32)
    if not np.any(source > 0.0):
        return source.copy()
    height, width = source.shape
    x, y = _pixel_grids(source.shape)
    xn = x / max(float(width - 1), 1.0)
    yn = y / max(float(height - 1), 1.0)
    tau = 2.0 * math.pi
    phase = (int(seed) % 65537) / 65537.0 * tau
    t = float(frame)

    warp_x = 0.030 * np.sin(tau * (1.25 * xn + 0.76 * yn) + 0.018 * t + phase)
    warp_y = 0.026 * np.cos(tau * (-0.62 * xn + 1.18 * yn) - 0.015 * t + phase * 1.31)
    xw = xn + warp_x
    yw = yn + warp_y
    along = 0.92 * xw - 0.39 * yw + 0.0038 * t
    cross = 0.39 * xw + 0.92 * yw

    fine = (
        0.50 * np.sin(tau * (2.2 * along + 9.5 * cross) + phase * 0.4)
        + 0.32 * np.sin(tau * (5.4 * along + 18.0 * cross) - phase * 0.7)
        + 0.18 * np.cos(tau * (1.1 * along - 4.2 * cross) + phase * 1.9)
    )
    broad = (
        0.58 * np.sin(tau * (0.75 * along + 2.1 * cross) + phase * 1.2)
        + 0.42 * np.cos(tau * (1.25 * along - 1.5 * cross) - phase * 0.6)
    )

    positive = source[source > 0.0]
    scale_value = float(np.percentile(positive, 94.0)) if positive.size else 1.0
    scale_value = max(scale_value, 1.0e-5)
    normalized = np.clip(source / scale_value, 0.0, 1.8)
    filament = _smoothstep(-0.25, 0.82, fine)
    edge_weight = _smoothstep(0.006, 0.18, normalized) * (1.0 - 0.35 * _smoothstep(1.05, 1.75, normalized))
    dense_weight = _smoothstep(0.20, 1.35, normalized)
    hole_weight = _smoothstep(0.18, 0.88, broad) * edge_weight
    streak_gain = 0.98 + edge_weight * (0.28 * filament + 0.11 * fine + 0.08 * broad)
    out = source * streak_gain * (1.0 - 0.24 * hole_weight * (1.0 - 0.26 * dense_weight))
    out = out * (0.997 + 0.003 * np.sin(tau * (0.42 * along + 0.8 * cross) + phase))
    out = np.where(out >= 1.0e-7, out, 0.0)
    return np.clip(out, 0.0, 4.0).astype(np.float32, copy=False)


def _smoke_step(
    density: np.ndarray,
    hotspots: Sequence[FireHotspot],
    frame: int,
    size: int,
    bounds: Bounds,
    seed: int,
) -> np.ndarray:
    scale = size / 1024.0
    base_wind_px = (3.2 * scale, -1.35 * scale)
    u, v = procedural_wind_field(frame, density.shape, seed, base_wind_px)
    half_u = u * 0.5
    half_v = v * 0.5
    density = advect_density(density, half_u, half_v)
    density = advect_density(density, half_u, half_v)
    density = inject_hotspots(density, hotspots, frame, bounds)
    density = anisotropic_downwind_stretch(density, u, v, amount=0.15)
    density = diffuse_decay_density(density, frame, seed)
    density = apply_filament_breakup(density, frame, seed)
    density *= _border_fade(density.shape)
    density = np.where(density >= 1.0e-7, density, 0.0)
    return np.clip(density, 0.0, 4.0).astype(np.float32, copy=False)


def _iter_smoke_densities(
    hotspots: Sequence[FireHotspot],
    frames: int,
    size: int,
    bounds: Bounds,
    seed: int,
) -> Iterator[np.ndarray]:
    size = _validate_size(size)
    if frames <= 0:
        raise ValueError("frames must be positive")
    density = np.zeros((size, size), dtype=np.float32)
    for current in range(int(frames)):
        density = _smoke_step(density, hotspots, current, size, bounds, seed)
        yield density.copy()


def simulate_smoke_frame(
    hotspots: Sequence[FireHotspot],
    frame: int,
    frames: int,
    size: int,
    bounds: Bounds,
    seed: int,
) -> np.ndarray:
    """Simulate smoke from zero density through the requested frame."""
    size = _validate_size(size)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if frame < 0:
        raise ValueError("frame must be non-negative")
    target = min(int(frame), int(frames) - 1)
    density = np.zeros((size, size), dtype=np.float32)
    for current in range(target + 1):
        density = _smoke_step(density, hotspots, current, size, bounds, seed)
    return density.astype(np.float32, copy=False)


def density_to_smoke_rgba(
    density: np.ndarray,
    frame: int,
    seed: int,
) -> np.ndarray:
    """Convert smoke density to translucent gray-white RGBA."""
    source = np.clip(np.asarray(density, dtype=np.float32), 0.0, None)
    if source.ndim != 2:
        raise ValueError("density must be a 2D array")
    height, width = source.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(source > 0.0):
        return rgba

    softened = source * 0.78 + _box_blur_3x3(source, passes=1) * 0.22
    positive = softened[softened > 0.0]
    q = float(np.percentile(positive, 98.8)) if positive.size else 1.0
    q = max(q, 1.0e-4)
    norm = np.clip(softened / (q * 0.92), 0.0, 1.25)
    coverage = _smoothstep(0.0012, 0.022, softened)
    alpha_shape = _smoothstep(0.006, 0.98, norm) ** 0.72

    x, y = _pixel_grids(source.shape)
    tau = 2.0 * math.pi
    phase = (int(seed) % 7919) / 7919.0 * tau
    xn = x / max(float(width - 1), 1.0)
    yn = y / max(float(height - 1), 1.0)
    along = 0.92 * xn - 0.39 * yn + frame * 0.0038
    cross = 0.39 * xn + 0.92 * yn
    n1 = np.sin(tau * (0.85 * along + 6.8 * cross) + phase * 0.45 + frame * 0.012)
    n2 = np.sin(tau * (3.6 * along + 13.5 * cross) - phase * 0.70 - frame * 0.010)
    n3 = np.cos(tau * (7.8 * along + 23.0 * cross) + phase * 1.35 + frame * 0.006)
    texture = 0.52 * n1 + 0.30 * n2 + 0.18 * n3
    filament = _smoothstep(0.16, 0.88, 0.5 + 0.5 * texture)
    holes = 0.5 + 0.5 * np.sin(tau * (1.25 * along - 3.7 * cross) + phase * 1.6 - frame * 0.010)
    holes = np.maximum(holes, 0.5 + 0.5 * np.cos(tau * (2.4 * along + 2.2 * cross) - phase * 1.1))
    grooves = 0.5 + 0.5 * np.sin(tau * (2.1 * along + 5.6 * cross) - phase * 0.35 + frame * 0.008)
    alpha_texture = 0.22 + 1.04 * filament
    alpha_texture *= 1.0 - 0.82 * _smoothstep(0.48, 0.94, holes) * (1.0 - 0.04 * _smoothstep(1.05, 1.35, norm))
    alpha_texture *= 1.0 - 0.66 * _smoothstep(0.72, 0.98, grooves) * (1.0 - 0.10 * _smoothstep(1.10, 1.42, norm))
    wispy_floor = 14.0 * coverage * (1.0 - _smoothstep(0.18, 0.75, norm))
    alpha = np.clip((158.0 * alpha_shape + wispy_floor) * coverage * alpha_texture, 0.0, 185.0)
    alpha = np.where(alpha >= 1.5, alpha, 0.0)
    gray = 126.0 + 84.0 * _smoothstep(0.035, 1.02, norm) + 7.0 * texture * coverage
    gray = np.clip(gray, 122.0, 214.0)

    visible = alpha >= 1.0
    rgba[..., 0] = np.where(visible, np.clip(gray + 1.5, 0, 255), 0).astype(np.uint8)
    rgba[..., 1] = np.where(visible, np.clip(gray + 0.5, 0, 255), 0).astype(np.uint8)
    rgba[..., 2] = np.where(visible, np.clip(gray - 1.0, 0, 255), 0).astype(np.uint8)
    rgba[..., 3] = np.where(visible, alpha, 0).astype(np.uint8)
    return rgba


def fire_hotspots_rgba(
    hotspots: Sequence[FireHotspot],
    frame: int,
    size: int,
    bounds: Bounds,
    seed: int,
) -> np.ndarray:
    """Render small orange/yellow glows at active fire-source locations."""
    size = _validate_size(size)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    scale = size / 1024.0
    phase = (int(seed) % 97) * 0.03

    rgb_accum = np.zeros((size, size, 3), dtype=np.float32)
    alpha_accum = np.zeros((size, size), dtype=np.float32)
    for index, hotspot in enumerate(hotspots):
        if not (hotspot.start_frame <= frame <= hotspot.end_frame):
            continue
        cx, cy = lonlat_to_pixel(hotspot.lon, hotspot.lat, bounds, size, size)
        radius = max(1.15, float(hotspot.radius_px_at_1024) * scale * 0.48)
        halo_radius = max(3.0, radius * 3.1)
        pad = int(math.ceil(halo_radius * 2.5))
        x0 = max(0, int(math.floor(cx - pad)))
        x1 = min(size, int(math.ceil(cx + pad + 1)))
        y0 = max(0, int(math.floor(cy - pad)))
        y1 = min(size, int(math.ceil(cy + pad + 1)))
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist2 = (xx - np.float32(cx)) ** 2 + (yy - np.float32(cy)) ** 2
        halo = np.exp(-dist2 / (2.0 * (halo_radius * 0.46) ** 2))
        core = np.exp(-dist2 / (2.0 * max(radius * 0.43, 0.35) ** 2))
        pulse = 0.82 + 0.18 * math.sin(frame * 0.53 + index * 1.7 + phase)
        local_alpha = np.clip((72.0 * halo + 178.0 * core) * pulse * hotspot.strength, 0.0, 230.0)
        local_rgb = (
            np.array([210.0, 74.0, 18.0], dtype=np.float32) * halo[..., None]
            + np.array([255.0, 221.0, 75.0], dtype=np.float32) * core[..., None] * 1.35
        )
        divisor = np.maximum(halo[..., None] + core[..., None] * 1.35, 1.0e-5)
        local_rgb = local_rgb / divisor

        current_alpha = alpha_accum[y0:y1, x0:x1]
        replace = local_alpha > current_alpha
        alpha_accum[y0:y1, x0:x1] = np.where(replace, local_alpha, current_alpha)
        rgb_accum[y0:y1, x0:x1, :] = np.where(replace[..., None], local_rgb, rgb_accum[y0:y1, x0:x1, :])

    visible = alpha_accum >= 1.0
    rgba[..., :3] = np.where(visible[..., None], np.clip(rgb_accum, 0, 255), 0).astype(np.uint8)
    rgba[..., 3] = np.where(visible, alpha_accum, 0).astype(np.uint8)
    return rgba


def _alpha_composite_arrays(*layers: np.ndarray) -> np.ndarray:
    if not layers:
        raise ValueError("at least one layer is required")
    image = Image.fromarray(layers[0], mode="RGBA")
    for layer in layers[1:]:
        image = Image.alpha_composite(image, Image.fromarray(layer, mode="RGBA"))
    return np.asarray(image, dtype=np.uint8)


def _snapshot_needs_direct_composite(path: Path) -> bool:
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return True
    arr = np.asarray(image, dtype=np.int16)
    if arr.size == 0:
        return True
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    ocean_like = (blue > green + 8) & (green > red + 6) & (blue > 45)
    return float(np.count_nonzero(ocean_like)) / float(ocean_like.size) < 0.03


def _write_direct_composite_preview(
    output_path: Path,
    overlay_paths: dict[str, Path],
    width: int,
    height: int,
) -> None:
    base = np.asarray(Image.open(overlay_paths["base"]).convert("RGBA"), dtype=np.uint8)
    smoke = np.asarray(Image.open(overlay_paths["smoke"]).convert("RGBA"), dtype=np.uint8)
    fire = np.asarray(Image.open(overlay_paths["fire"]).convert("RGBA"), dtype=np.uint8)
    composite = Image.fromarray(_alpha_composite_arrays(base, smoke, fire), mode="RGBA")
    if composite.size != (int(width), int(height)):
        composite = composite.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    composite.save(output_path)


def write_overlay_pngs(
    directory: Path,
    frame: int,
    frames: int,
    size: int,
    bounds: Bounds,
    hotspots: Sequence[FireHotspot],
    seed: int,
) -> dict[str, Path]:
    """Write base, smoke, and fire overlay PNGs for one selected frame."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = build_base_map_rgba(size, bounds)
    density = simulate_smoke_frame(hotspots, frame, frames, size, bounds, seed)
    smoke = density_to_smoke_rgba(density, frame, seed)
    fire = fire_hotspots_rgba(hotspots, frame, size, bounds, seed)

    paths = {
        "base": out_dir / "base.png",
        "smoke": out_dir / f"smoke_{int(frame):04d}.png",
        "fire": out_dir / f"fire_{int(frame):04d}.png",
    }
    Image.fromarray(base, mode="RGBA").save(paths["base"])
    Image.fromarray(smoke, mode="RGBA").save(paths["smoke"])
    Image.fromarray(fire, mode="RGBA").save(paths["fire"])
    return paths


def _write_sequence_pngs(
    directory: Path,
    frames: int,
    size: int,
    bounds: Bounds,
    hotspots: Sequence[FireHotspot],
    seed: int,
) -> None:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = build_base_map_rgba(size, bounds)
    Image.fromarray(base, mode="RGBA").save(out_dir / "base.png")
    for frame, density in enumerate(_iter_smoke_densities(hotspots, frames, size, bounds, seed)):
        smoke = density_to_smoke_rgba(density, frame, seed)
        fire = fire_hotspots_rgba(hotspots, frame, size, bounds, seed)
        composite = _alpha_composite_arrays(base, smoke, fire)
        Image.fromarray(smoke, mode="RGBA").save(out_dir / f"smoke_{frame:04d}.png")
        Image.fromarray(fire, mode="RGBA").save(out_dir / f"fire_{frame:04d}.png")
        Image.fromarray(composite, mode="RGBA").save(out_dir / f"composite_{frame:04d}.png")


def _write_contact_sheet(
    path: Path,
    frames: int,
    size: int,
    bounds: Bounds,
    hotspots: Sequence[FireHotspot],
    seed: int,
) -> None:
    selected = sorted({0, frames // 5, (2 * frames) // 5, (3 * frames) // 5, (4 * frames) // 5, frames - 1})
    tile = min(256, size)
    sheet = Image.new("RGBA", (tile * len(selected), tile), (0, 0, 0, 255))
    for column, frame in enumerate(selected):
        density = simulate_smoke_frame(hotspots, frame, frames, size, bounds, seed)
        smoke = density_to_smoke_rgba(density, frame, seed)
        tile_image = Image.fromarray(smoke, mode="RGBA").resize((tile, tile), Image.Resampling.BILINEAR)
        sheet.alpha_composite(tile_image, (column * tile, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _ensure_repo_import() -> None:
    python_dir = PROJECT_ROOT / "python"
    if python_dir.exists() and str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))


def render_smoke_frame_with_forge3d(
    output_path: Path,
    frame: int,
    frames: int,
    size: int,
    width: int,
    height: int,
    bounds: Bounds,
    hotspots: Sequence[FireHotspot],
    seed: int,
    work_dir: Path,
) -> None:
    """Render one smoke frame through forge3d using raster overlays only."""
    _ensure_repo_import()
    import forge3d as f3d  # noqa: PLC0415

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    dem_path = work / "california_smoke_flat_terrain.npy"
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    terrain = 0.012 * np.sin(xx / max(size, 1) * 2.0 * math.pi) * np.cos(yy / max(size, 1) * 2.0 * math.pi)
    np.save(dem_path, terrain.astype(np.float32))
    overlay_paths = write_overlay_pngs(work, frame, frames, size, bounds, hotspots, seed)
    overlay_extent = (0.0, 0.0, 1.0, 1.0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_snapshot_path = work / "forge3d_raw_snapshot.png"
    with f3d.open_viewer_async(terrain_path=dem_path, width=width, height=height, timeout=75.0) as viewer:
        time.sleep(0.85)
        viewer.send_ipc(
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
                "normal_strength": 0.05,
                "height_ao": {"enabled": False},
            }
        )
        viewer.load_overlay(
            name="california-base",
            path=overlay_paths["base"],
            extent=overlay_extent,
            opacity=1.0,
            z_order=0,
            preserve_colors=True,
        )
        viewer.load_overlay(
            name="california-smoke",
            path=overlay_paths["smoke"],
            extent=overlay_extent,
            opacity=1.0,
            z_order=20,
            preserve_colors=True,
        )
        viewer.load_overlay(
            name="california-fire-hotspots",
            path=overlay_paths["fire"],
            extent=overlay_extent,
            opacity=1.0,
            z_order=30,
            preserve_colors=True,
        )
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": True})
        viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
        viewer.set_orbit_camera(phi_deg=0.0, theta_deg=7.0, radius=3.8, fov_deg=26.0)
        viewer.set_sun(azimuth_deg=295.0, elevation_deg=68.0)
        time.sleep(1.35)
        viewer.snapshot(raw_snapshot_path, width=width, height=height)
    if _snapshot_needs_direct_composite(raw_snapshot_path):
        _write_direct_composite_preview(output, overlay_paths, width, height)
    else:
        Image.open(raw_snapshot_path).save(output)


def _render_sequence_with_forge3d(
    directory: Path,
    frames: int,
    size: int,
    width: int,
    height: int,
    bounds: Bounds,
    hotspots: Sequence[FireHotspot],
    seed: int,
) -> None:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(int(frames)):
        with tempfile.TemporaryDirectory(prefix="forge3d_ca_smoke_frame_") as tmp:
            render_smoke_frame_with_forge3d(
                output_path=out_dir / f"frame_{frame:04d}.png",
                frame=frame,
                frames=frames,
                size=size,
                width=width,
                height=height,
                bounds=bounds,
                hotspots=hotspots,
                seed=seed,
                work_dir=Path(tmp),
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate California wildfire smoke raster overlays.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--firms-csv", type=Path, default=None)
    parser.add_argument("--write-overlays", type=Path, default=None)
    parser.add_argument("--write-sequence", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-sequence", action="store_true")
    parser.add_argument("--overlays-only", action="store_true")
    parser.add_argument("--contact-sheet", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bounds = DEFAULT_BOUNDS
    frame = max(0, int(args.frame))
    frames = max(1, int(args.frames))
    size = _validate_size(int(args.grid_size))
    seed = int(args.seed)
    hotspots = (
        load_firms_hotspots(args.firms_csv, bounds)
        if args.firms_csv is not None
        else default_hotspots()
    )

    wrote_any_overlay = False
    if args.write_overlays is not None:
        write_overlay_pngs(args.write_overlays, frame, frames, size, bounds, hotspots, seed)
        wrote_any_overlay = True

    if args.write_sequence is not None:
        _write_sequence_pngs(args.write_sequence, frames, size, bounds, hotspots, seed)
        wrote_any_overlay = True

    if args.contact_sheet is not None:
        _write_contact_sheet(args.contact_sheet, frames, size, bounds, hotspots, seed)
        wrote_any_overlay = True

    if args.overlays_only:
        if not wrote_any_overlay:
            write_overlay_pngs(args.output.parent / "overlays", frame, frames, size, bounds, hotspots, seed)
        return 0

    if args.render_sequence:
        sequence_dir = args.write_sequence if args.write_sequence is not None else args.output.parent / "rendered_sequence"
        _render_sequence_with_forge3d(sequence_dir, frames, size, args.width, args.height, bounds, hotspots, seed)
        return 0

    should_render_still = args.render or args.write_sequence is None
    if should_render_still:
        with tempfile.TemporaryDirectory(prefix="forge3d_ca_smoke_") as tmp:
            render_smoke_frame_with_forge3d(
                output_path=args.output,
                frame=frame,
                frames=frames,
                size=size,
                width=int(args.width),
                height=int(args.height),
                bounds=bounds,
                hotspots=hotspots,
                seed=seed,
                work_dir=Path(tmp),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
