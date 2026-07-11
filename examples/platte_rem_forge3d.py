#!/usr/bin/env python3
from __future__ import annotations

"""Render a top-down 3D-style REM plate of the Yellowstone River in Forge3D.

This example uses a broad Yellowstone River reach near Livingston, Montana. It
builds a river-relative elevation model (REM) from a tiled `1m` USGS 3DEP DEM
and NHD large-scale flowlines, then asks Forge3D for a shaded relief pass and
composites that relief back onto a stylized REM color plate.

The result intentionally aims closer to the map-like look of a rayshader REM
snapshot than to an oblique terrain-perspective render.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from PIL import Image, ImageFilter
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform as warp_transform, transform_bounds
from rasterio.windows import Window

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.terrain_params import (
    DetailSettings,
    HeightAoSettings,
    PomSettings,
    ShadowSettings,
    TonemapSettings,
    make_terrain_params_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_NAME = "Yellowstone River"
SITE_REGION = "near Livingston, Montana"
SITE_SLUG = "yellowstone_livingston"

DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "yellowstone_rem_forge3d"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "examples" / "out" / "yellowstone_rem_forge3d" / "yellowstone-river-rem-3d.png"
)
THREEDEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)
NHD_LARGE_SCALE_FLOWLINES_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query"
)
RENDER_CRS = "EPSG:3857"
USER_AGENT = "forge3d-yellowstone-rem-example/1.0"

# Yellowstone River near Livingston, Montana.
CENTER_WGS84 = (-110.50, 45.69)
SIDE_LENGTH_M = 6144.0
MAIN_CORRIDOR_AXIS_DEG = 118.0
MAIN_CORRIDOR_HALF_BAND_M = 1200.0

ROCKET_COLOR_STOPS = np.array(
    [
        [0.0000, 0.01060815, 0.01808215, 0.10018654],
        [0.0941, 0.14066867, 0.07782240, 0.19452676],
        [0.1882, 0.28144375, 0.11104133, 0.28354890],
        [0.2824, 0.43214263, 0.12179973, 0.33966284],
        [0.3765, 0.59096382, 0.10810205, 0.35847347],
        [0.5020, 0.79650140, 0.10506637, 0.31063031],
        [0.6275, 0.92676657, 0.29823282, 0.24285536],
        [0.7529, 0.95922872, 0.53307513, 0.37488950],
        [0.8745, 0.96575293, 0.73235058, 0.59242739],
        [1.0000, 0.98137749, 0.92061729, 0.86536915],
    ],
    dtype=np.float32,
)
RELIEF_STOPS = [
    (0.0, "#d9ddec"),
    (5.0, "#bec5dc"),
    (12.0, "#98a1bc"),
    (24.0, "#6f7287"),
]
HDR_FILENAME = "forge3d_rem_neutral.hdr"
REQUIRED_NATIVE = ("Session", "TerrainRenderer", "TerrainRenderParams", "MaterialSet", "IBL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Forge3D REM plate of the Yellowstone River near Livingston, Montana."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dem-resolution-m",
        type=float,
        default=1.0,
        help="Target DEM resolution in meters per pixel for the tiled USGS mosaic.",
    )
    parser.add_argument(
        "--dem-size",
        type=int,
        default=None,
        help="Legacy explicit square DEM size in pixels. Overrides --dem-resolution-m when provided.",
    )
    parser.add_argument(
        "--tile-size-px",
        type=int,
        default=2048,
        help="Maximum USGS export tile edge in pixels before mosaicking into a larger DEM.",
    )
    parser.add_argument("--render-size", type=int, default=2200, help="Square output render size in pixels.")
    parser.add_argument(
        "--aggregate-factor",
        type=int,
        default=16,
        help="REM river-surface modeling factor. Lower values preserve more DEM detail.",
    )
    parser.add_argument(
        "--river-spacing-m",
        type=float,
        default=20.0,
        help="Sampling distance along NHD flowlines in meters.",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="Inverse-distance weighting exponent for the river surface.",
    )
    parser.add_argument(
        "--zscale",
        type=float,
        default=1.9,
        help="Vertical exaggeration passed to Forge3D for the relief pass.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Stop after writing DEM and REM GeoTIFFs.")
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild cached artifacts.")
    return parser.parse_args()


def _expand_bounds_to_square(
    bounds: tuple[float, float, float, float],
    *,
    pad_fraction: float = 0.02,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = map(float, bounds)
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)
    side = max(right - left, top - bottom) * (1.0 + float(pad_fraction))
    half = 0.5 * side
    return (cx - half, cy - half, cx + half, cy + half)


def _project_bounds_wgs84_to_3857(
    bounds_wgs84: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return tuple(
        float(value)
        for value in transform_bounds("EPSG:4326", RENDER_CRS, *bounds_wgs84, densify_pts=21)
    )


def _project_bounds_3857_to_wgs84(
    bounds_3857: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return tuple(
        float(value)
        for value in transform_bounds(RENDER_CRS, "EPSG:4326", *bounds_3857, densify_pts=21)
    )


def _square_bounds_from_center_wgs84(
    center_wgs84: tuple[float, float],
    *,
    side_length_m: float,
) -> tuple[float, float, float, float]:
    cx, cy = warp_transform("EPSG:4326", RENDER_CRS, [center_wgs84[0]], [center_wgs84[1]])
    half = 0.5 * float(side_length_m)
    return (float(cx[0] - half), float(cy[0] - half), float(cx[0] + half), float(cy[0] + half))


def _densify_polyline(points_xy: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points_xy) < 2:
        return np.asarray(points_xy, dtype=np.float64)

    out = [np.asarray(points_xy[0], dtype=np.float64)]
    spacing = max(float(spacing_m), 1.0)
    for start, end in zip(points_xy[:-1], points_xy[1:]):
        delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.hypot(delta[0], delta[1]))
        steps = max(1, int(math.ceil(length / spacing)))
        for step in range(1, steps + 1):
            out.append(np.asarray(start, dtype=np.float64) + delta * (step / steps))
    return np.vstack(out)


def _filter_paths_to_main_corridor(
    paths_3857: list[np.ndarray],
    *,
    center_xy: tuple[float, float],
    axis_deg: float,
    half_band_m: float,
) -> list[np.ndarray]:
    axis = math.radians(float(axis_deg))
    direction = np.array([math.cos(axis), math.sin(axis)], dtype=np.float64)
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)

    kept: list[np.ndarray] = []
    for path in paths_3857:
        midpoint = np.mean(np.asarray(path, dtype=np.float64), axis=0)
        distance = abs(float(np.dot(midpoint - center, perp)))
        if distance <= float(half_band_m):
            kept.append(path)
    return kept if kept else paths_3857


def _grid_centers(
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    left, bottom, right, top = bounds
    rows, cols = shape
    x_step = (right - left) / float(cols)
    y_step = (top - bottom) / float(rows)
    xs = np.linspace(left + 0.5 * x_step, right - 0.5 * x_step, cols, dtype=np.float64)
    ys = np.linspace(top - 0.5 * y_step, bottom + 0.5 * y_step, rows, dtype=np.float64)
    return np.meshgrid(xs, ys)


def _sample_bilinear(
    array: np.ndarray,
    bounds: tuple[float, float, float, float],
    points_xy: np.ndarray,
) -> np.ndarray:
    rows, cols = array.shape
    left, bottom, right, top = bounds
    xs = np.asarray(points_xy[:, 0], dtype=np.float64)
    ys = np.asarray(points_xy[:, 1], dtype=np.float64)

    px = (xs - left) / max(right - left, 1e-6) * (cols - 1)
    py = (top - ys) / max(top - bottom, 1e-6) * (rows - 1)
    valid = (px >= 0.0) & (py >= 0.0) & (px <= cols - 1) & (py <= rows - 1)

    samples = np.full(len(points_xy), np.nan, dtype=np.float32)
    if not np.any(valid):
        return samples

    x0 = np.floor(px[valid]).astype(np.int32)
    y0 = np.floor(py[valid]).astype(np.int32)
    x1 = np.minimum(x0 + 1, cols - 1)
    y1 = np.minimum(y0 + 1, rows - 1)

    tx = (px[valid] - x0).astype(np.float32)
    ty = (py[valid] - y0).astype(np.float32)
    q00 = array[y0, x0]
    q10 = array[y0, x1]
    q01 = array[y1, x0]
    q11 = array[y1, x1]

    samples[valid] = (
        q00 * (1.0 - tx) * (1.0 - ty)
        + q10 * tx * (1.0 - ty)
        + q01 * (1.0 - tx) * ty
        + q11 * tx * ty
    ).astype(np.float32)
    return samples


def _aggregate_mean(array: np.ndarray, factor: int) -> np.ndarray:
    rows, cols = array.shape
    if rows % factor != 0 or cols % factor != 0:
        raise ValueError(
            f"DEM shape {array.shape} must be divisible by aggregate factor {factor}"
        )
    return (
        array.reshape(rows // factor, factor, cols // factor, factor)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    remainder = int(value) % int(multiple)
    if remainder == 0:
        return int(value)
    return int(value) + int(multiple) - remainder


def _idw_surface(
    sample_points_xy: np.ndarray,
    sample_values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    power: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    queries = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    points = np.asarray(sample_points_xy, dtype=np.float32)
    values = np.asarray(sample_values, dtype=np.float32)
    out = np.empty(len(queries), dtype=np.float32)
    power_half = np.float32(max(float(power), 0.1) * 0.5)

    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        dx = chunk[:, None, 0] - points[None, :, 0]
        dy = chunk[:, None, 1] - points[None, :, 1]
        dist2 = dx * dx + dy * dy
        exact = dist2 <= 1.0
        weights = 1.0 / np.power(np.maximum(dist2, 1.0), power_half)
        weighted = (weights * values[None, :]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)

        if np.any(exact):
            exact_rows = np.where(exact.any(axis=1))[0]
            first_hits = exact[exact_rows].argmax(axis=1)
            weighted[exact_rows] = values[first_hits]

        out[start : start + len(chunk)] = weighted.astype(np.float32)

    return out.reshape(grid_x.shape)


def _ensure_hdr(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 8, 4
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode("ascii"))
        for y in range(height):
            for x in range(width):
                r = int(140 + 18 * x)
                g = int(144 + 10 * y)
                b = int(148 + 6 * x)
                handle.write(bytes([min(r, 255), min(g, 255), min(b, 255), 128]))
    return path


def _write_geotiff(
    path: Path,
    array: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=RENDER_CRS,
        transform=from_bounds(*bounds, width, height),
        nodata=None,
        compress="lzw",
    ) as dst:
        dst.write(np.asarray(array, dtype=np.float32), 1)
    return path


def _download_three_dep_dem(
    bounds_3857: tuple[float, float, float, float],
    *,
    dem_resolution_m: float,
    dem_size: int,
    tile_size_px: int,
    align_to: int,
    cache_dir: Path,
    force: bool,
) -> tuple[Path, np.ndarray]:
    span_x = float(bounds_3857[2] - bounds_3857[0])
    span_y = float(bounds_3857[3] - bounds_3857[1])
    if dem_size is None:
        width = max(1, int(math.ceil(span_x / max(float(dem_resolution_m), 0.1))))
        height = max(1, int(math.ceil(span_y / max(float(dem_resolution_m), 0.1))))
        width = _align_up(width, int(align_to))
        height = _align_up(height, int(align_to))
    else:
        width = int(dem_size)
        height = int(dem_size)

    tile_edge = max(256, int(tile_size_px))
    res_label = f"{float(dem_resolution_m):.1f}".replace(".", "p")
    dem_path = cache_dir / f"{SITE_SLUG}_3dep_{res_label}m_{width}x{height}_3857.tif"
    if not dem_path.exists() or force:
        cache_dir.mkdir(parents=True, exist_ok=True)
        left, bottom, right, top = bounds_3857
        transform = from_bounds(*bounds_3857, width, height)
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=RENDER_CRS,
            transform=transform,
            nodata=None,
            compress="lzw",
            tiled=True,
        ) as dst:
            for row0 in range(0, height, tile_edge):
                tile_h = min(tile_edge, height - row0)
                row1 = row0 + tile_h
                tile_top = top - (row0 / height) * (top - bottom)
                tile_bottom = top - (row1 / height) * (top - bottom)
                for col0 in range(0, width, tile_edge):
                    tile_w = min(tile_edge, width - col0)
                    col1 = col0 + tile_w
                    tile_left = left + (col0 / width) * (right - left)
                    tile_right = left + (col1 / width) * (right - left)
                    params = {
                        "bbox": f"{tile_left:.3f},{tile_bottom:.3f},{tile_right:.3f},{tile_top:.3f}",
                        "bboxSR": "3857",
                        "imageSR": "3857",
                        "size": f"{tile_w},{tile_h}",
                        "format": "tiff",
                        "pixelType": "F32",
                        "f": "image",
                    }
                    req = Request(
                        THREEDEP_EXPORT_URL + "?" + urlencode(params),
                        headers={"User-Agent": USER_AGENT},
                    )
                    with urlopen(req, timeout=180) as response:
                        data = response.read()
                    with MemoryFile(data) as memfile, memfile.open() as dataset:
                        array = dataset.read(1).astype(np.float32)
                        nodata = dataset.nodata
                    if nodata is not None:
                        mask = np.isfinite(array) & (~np.isclose(array, nodata))
                    else:
                        mask = np.isfinite(array)
                    fill = float(np.nanpercentile(array[mask], 1.0)) if np.any(mask) else 0.0
                    array = np.where(mask, array, fill).astype(np.float32)
                    dst.write(array, 1, window=Window(col0, row0, tile_w, tile_h))

    with rasterio.open(dem_path) as dataset:
        return dem_path, dataset.read(1).astype(np.float32)


def _query_nhd_flowlines_3857(
    bounds_wgs84: tuple[float, float, float, float],
) -> list[np.ndarray]:
    params = {
        "f": "json",
        "where": "FTYPE = 460",
        "geometry": ",".join(str(value) for value in bounds_wgs84),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "REACHCODE",
        "returnGeometry": "true",
        "outSR": "4326",
    }
    req = Request(
        NHD_LARGE_SCALE_FLOWLINES_URL + "?" + urlencode(params),
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(req, timeout=180) as response:
        data = response.read()
    result = json.loads(data)
    features = result.get("features", [])
    paths_3857: list[np.ndarray] = []
    for feature in features:
        geometry = feature.get("geometry", {})
        for path in geometry.get("paths", []):
            if len(path) < 2:
                continue
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            proj_x, proj_y = warp_transform("EPSG:4326", RENDER_CRS, xs, ys)
            coords = np.column_stack([proj_x, proj_y]).astype(np.float64)
            paths_3857.append(coords)
    if not paths_3857:
        raise RuntimeError(f"No NHD large-scale flowlines were returned for the {SITE_NAME} bbox")
    return paths_3857


def _build_rem(
    dem: np.ndarray,
    bounds_3857: tuple[float, float, float, float],
    *,
    bounds_wgs84: tuple[float, float, float, float],
    aggregate_factor: int,
    river_spacing_m: float,
    idw_power: float,
) -> np.ndarray:
    coarse_shape = (dem.shape[0] // aggregate_factor, dem.shape[1] // aggregate_factor)
    flowline_paths = _query_nhd_flowlines_3857(bounds_wgs84)
    flowline_paths = _filter_paths_to_main_corridor(
        flowline_paths,
        center_xy=(
            0.5 * (bounds_3857[0] + bounds_3857[2]),
            0.5 * (bounds_3857[1] + bounds_3857[3]),
        ),
        axis_deg=MAIN_CORRIDOR_AXIS_DEG,
        half_band_m=MAIN_CORRIDOR_HALF_BAND_M,
    )
    river_points = np.vstack(
        [_densify_polyline(path, spacing_m=river_spacing_m) for path in flowline_paths]
    )
    river_elev = _sample_bilinear(dem, bounds_3857, river_points)
    valid = np.isfinite(river_elev)
    if valid.sum() < 8:
        raise RuntimeError("Failed to sample enough flowline elevations for the REM model")

    grid_x, grid_y = _grid_centers(bounds_3857, coarse_shape)
    river_surface_coarse = _idw_surface(
        river_points[valid],
        river_elev[valid],
        grid_x,
        grid_y,
        power=idw_power,
    )
    river_surface = np.empty_like(dem, dtype=np.float32)
    reproject(
        source=river_surface_coarse,
        destination=river_surface,
        src_transform=from_bounds(*bounds_3857, coarse_shape[1], coarse_shape[0]),
        src_crs=RENDER_CRS,
        dst_transform=from_bounds(*bounds_3857, dem.shape[1], dem.shape[0]),
        dst_crs=RENDER_CRS,
        resampling=Resampling.bilinear,
    )
    rem = dem.astype(np.float32) - river_surface
    return np.maximum(rem, 0.0).astype(np.float32)


def _build_base_plate(rem: np.ndarray, *, output_size: tuple[int, int]) -> np.ndarray:
    valid = rem[np.isfinite(rem)]
    if valid.size == 0:
        raise RuntimeError("REM grid is empty")

    p99 = float(np.quantile(valid, 0.99))
    rem_clip = np.clip(rem, 0.0, max(p99, 1e-6))
    low_scale_m = 0.35
    norm = np.log1p(rem_clip / low_scale_m) / math.log1p(max(p99, 1e-6) / low_scale_m)
    norm = np.clip(norm, 0.0, 1.0)
    palette_t = 0.03 + 0.90 * (1.0 - np.sqrt(norm))

    rgb = np.empty(rem.shape + (3,), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.interp(
            palette_t,
            ROCKET_COLOR_STOPS[:, 0],
            ROCKET_COLOR_STOPS[:, channel + 1],
        )

    rem_gray = Image.fromarray(
        (norm * 255.0).astype(np.uint8),
        mode="L",
    )
    small = np.asarray(rem_gray.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32) / 255.0
    medium = np.asarray(rem_gray.filter(ImageFilter.GaussianBlur(radius=2.8)), dtype=np.float32) / 255.0
    detail = np.clip(small - medium, -1.0, 1.0)
    rgb[..., 0] = np.clip(rgb[..., 0] + detail * 0.030, 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + detail * 0.026, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] + detail * 0.018, 0.0, 1.0)

    low_lift = np.exp(-rem / 0.10)
    channel_highlight = np.clip(low_lift * np.clip(detail * 10.0 - 0.02, 0.0, 1.0), 0.0, 1.0)
    channel_tint = np.array([0.995, 0.965, 0.900], dtype=np.float32)
    rgb = np.clip(
        rgb * (1.0 - channel_highlight[..., None] * 0.18)
        + channel_tint * (channel_highlight[..., None] * 0.18),
        0.0,
        1.0,
    )

    floodplain_cool = np.clip(low_lift * (1.0 - channel_highlight) * 0.06, 0.0, 1.0)
    cool_tint = np.array([0.46, 0.40, 0.66], dtype=np.float32)
    rgb = np.clip(
        rgb * (1.0 - floodplain_cool[..., None]) + cool_tint * floodplain_cool[..., None],
        0.0,
        1.0,
    )

    native = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    resized = native.resize(output_size, resample=Image.Resampling.LANCZOS)
    resized = resized.filter(ImageFilter.UnsharpMask(radius=1.4, percent=145, threshold=2))
    return np.asarray(resized, dtype=np.uint8)


def _require_native_render_support() -> None:
    if not f3d.has_gpu() or not all(hasattr(f3d, name) for name in REQUIRED_NATIVE):
        raise SystemExit(
            "Forge3D native offscreen terrain rendering is unavailable. "
            "The DEM and REM GeoTIFFs were still prepared successfully; "
            "rerun with --prepare-only or build the native extension first."
        )


def _render_relief_rgba(
    rem: np.ndarray,
    bounds_3857: tuple[float, float, float, float],
    *,
    render_size: int,
    zscale: float,
    cache_dir: Path,
) -> np.ndarray:
    _require_native_render_support()

    valid = rem[np.isfinite(rem)]
    if valid.size == 0:
        raise RuntimeError("REM grid is empty")
    domain_max = float(np.quantile(valid, 0.995))
    domain = (0.0, max(domain_max, 1.0))
    colormap = f3d.Colormap1D.from_stops(stops=RELIEF_STOPS, domain=domain)
    overlay = f3d.OverlayLayer.from_colormap1d(colormap, strength=1.0, domain=domain)

    terrain_span = float(bounds_3857[2] - bounds_3857[0])
    shadow_off = ShadowSettings(
        enabled=False,
        technique="NONE",
        resolution=512,
        cascades=1,
        max_distance=max(terrain_span * 2.0, 1.0),
        softness=0.0,
        intensity=0.0,
        slope_scale_bias=0.001,
        depth_bias=0.0005,
        normal_bias=0.0002,
        min_variance=1e-4,
        light_bleed_reduction=0.0,
        evsm_exponent=1.0,
        fade_start=1.0,
    )
    height_ao = HeightAoSettings(
        enabled=True,
        resolution_scale=1.0,
        directions=16,
        steps=28,
        max_distance=max(terrain_span * 0.02, 280.0),
        strength=0.65,
    )
    tonemap = TonemapSettings(
        operator="aces",
        white_point=4.0,
        white_balance_enabled=True,
        temperature=6500.0,
        tint=0.0,
    )
    pom_off = PomSettings(
        enabled=False,
        mode="Occlusion",
        scale=0.0,
        min_steps=1,
        max_steps=1,
        refine_steps=0,
        shadow=False,
        occlusion=False,
    )
    detail_off = DetailSettings(enabled=False)

    params_cfg = make_terrain_params_config(
        size_px=(render_size, render_size),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,
        z_scale=float(zscale),
        exposure=1.0,
        domain=domain,
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        light_azimuth_deg=320.0,
        light_elevation_deg=33.0,
        sun_intensity=1.45,
        ibl_intensity=0.08,
        cam_radius=terrain_span,
        cam_phi_deg=180.0,
        cam_theta_deg=0.0,
        fov_y_deg=6.0,
        camera_mode="screen",
        clip=(0.1, terrain_span * 4.0),
        overlays=[overlay],
        shadows=shadow_off,
        pom=pom_off,
        height_ao=height_ao,
        tonemap=tonemap,
        detail=detail_off,
    )

    hdr_path = _ensure_hdr(cache_dir / HDR_FILENAME)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.custom(
        base_color=(1.0, 1.0, 1.0),
        metallic=0.0,
        roughness=0.95,
        triplanar_scale=1.0,
        normal_strength=0.0,
        blend_sharpness=1.0,
    )
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    params = f3d.TerrainRenderParams(params_cfg)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=rem,
        target=None,
        water_mask=None,
    )
    return frame.to_numpy()


def _compose_final_rgb(
    base_rgb: np.ndarray,
    relief_rgba: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32) / 255.0
    height, width, _ = base.shape

    relief_image = Image.fromarray(np.asarray(relief_rgba, dtype=np.uint8), mode="RGBA")
    relief_gray = relief_image.convert("L")
    relief_gray = relief_gray.resize((width, height), resample=Image.Resampling.LANCZOS)
    relief_soft = relief_gray.filter(ImageFilter.GaussianBlur(radius=1.2))
    soft = np.asarray(relief_soft, dtype=np.float32) / 255.0
    low = float(np.quantile(soft, 0.02))
    high = float(np.quantile(soft, 0.98))
    soft_n = np.clip((soft - low) / max(high - low, 1e-6), 0.0, 1.0)

    yy, xx = np.mgrid[0:height, 0:width]
    radial = ((xx - width / 2.0) / (width / 2.0)) ** 2 + ((yy - height / 2.0) / (height / 2.0)) ** 2
    vignette = np.clip((radial - 0.42) / 0.9, 0.0, 1.0)
    base = base * (0.94 - vignette[..., None] * 0.08)

    shade = 0.82 + 0.24 * soft_n
    out = np.clip(base * shade[..., None], 0.0, 1.0)

    emboss_fine = np.asarray(
        relief_gray.filter(ImageFilter.GaussianBlur(radius=0.6)),
        dtype=np.float32,
    ) / 255.0
    emboss_coarse = np.asarray(
        relief_gray.filter(ImageFilter.GaussianBlur(radius=2.2)),
        dtype=np.float32,
    ) / 255.0
    emboss = emboss_fine - emboss_coarse
    out = np.clip(out + emboss[..., None] * 0.14, 0.0, 1.0)

    highlights = np.clip((soft_n - 0.68) / 0.32, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + highlights * 0.015, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + highlights * 0.04, 0.0, 1.0)

    midpoint = 0.5
    out = np.clip((out - midpoint) * 1.08 + midpoint, 0.0, 1.0)
    out = np.power(out, 1.02, dtype=np.float32)
    final = Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    final = final.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
    return np.asarray(final, dtype=np.uint8)


def main() -> int:
    args = parse_args()
    if args.dem_size is not None and args.dem_size <= 0:
        raise SystemExit("--dem-size must be positive when provided")
    if args.dem_resolution_m <= 0.0 or args.render_size <= 0:
        raise SystemExit("DEM resolution and render size must be positive")
    if args.tile_size_px <= 0:
        raise SystemExit("--tile-size-px must be positive")
    if args.aggregate_factor <= 0:
        raise SystemExit("--aggregate-factor must be positive")
    if args.dem_size is not None and args.dem_size % args.aggregate_factor != 0:
        raise SystemExit(
            f"--dem-size ({args.dem_size}) must be divisible by "
            f"--aggregate-factor ({args.aggregate_factor})"
        )

    cache_dir = args.cache_dir.resolve()
    output_path = args.output.resolve()
    bounds_3857 = _square_bounds_from_center_wgs84(CENTER_WGS84, side_length_m=SIDE_LENGTH_M)
    bounds_wgs84 = _project_bounds_3857_to_wgs84(bounds_3857)

    dem_path, dem = _download_three_dep_dem(
        bounds_3857,
        dem_resolution_m=float(args.dem_resolution_m),
        dem_size=None if args.dem_size is None else int(args.dem_size),
        tile_size_px=int(args.tile_size_px),
        align_to=int(args.aggregate_factor),
        cache_dir=cache_dir,
        force=bool(args.force),
    )
    dem_height, dem_width = dem.shape
    if dem_width % args.aggregate_factor != 0 or dem_height % args.aggregate_factor != 0:
        raise SystemExit(
            f"DEM shape {dem.shape} must be divisible by --aggregate-factor ({args.aggregate_factor}). "
            "Try a different aggregate factor or explicit --dem-size."
        )
    rem_path = cache_dir / (
        f"{SITE_SLUG}_rem_{dem_width}x{dem_height}_f{args.aggregate_factor}_3857.tif"
    )
    if not rem_path.exists() or args.force:
        rem = _build_rem(
            dem,
            bounds_3857,
            bounds_wgs84=bounds_wgs84,
            aggregate_factor=int(args.aggregate_factor),
            river_spacing_m=float(args.river_spacing_m),
            idw_power=float(args.idw_power),
        )
        _write_geotiff(rem_path, rem, bounds_3857)
    else:
        with rasterio.open(rem_path) as dataset:
            rem = dataset.read(1).astype(np.float32)

    print(f"[artifact] DEM: {dem_path}")
    print(f"[artifact] REM: {rem_path}")
    print(
        "[stats] REM min/max/p99 = "
        f"{float(np.min(rem)):.2f} / {float(np.max(rem)):.2f} / {float(np.quantile(rem, 0.99)):.2f} m"
    )

    if args.prepare_only:
        return 0

    base_rgb = _build_base_plate(
        rem,
        output_size=(int(args.render_size), int(args.render_size)),
    )
    relief_rgba = _render_relief_rgba(
        rem,
        bounds_3857,
        render_size=int(args.render_size),
        zscale=float(args.zscale),
        cache_dir=cache_dir,
    )
    final_rgb = _compose_final_rgb(base_rgb, relief_rgba)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_rgb, mode="RGB").save(output_path)
    print(f"[artifact] Render: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
