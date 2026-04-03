#!/usr/bin/env python3
"""Bosnia and Herzegovina land-cover terrain render in pure Python + forge3d.

Workflow:
1. Download the Bosnia and Herzegovina country polygon from Natural Earth.
2. Download the 2024 Esri 10m land-cover tiles covering BIH (33T and 34T).
3. Crop each palette GeoTIFF to the country boundary.
4. Download AWS Terrarium DEM tiles for the same country extent.
5. Reproject the DEM to a common target CRS.
6. Merge and categorical-resample the cropped land-cover tiles onto the DEM grid.
7. Convert the class raster into an RGBA overlay and render it with forge3d.

Requirements:
    pip install forge3d geopandas pillow rasterio
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples" / "out" / "bosnia_terrain_landcover"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "bosnia_terrain_landcover"
DEFAULT_HDRI_PATH = PROJECT_ROOT / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"

COUNTRY_A3 = "BIH"
COUNTRY_NAME = "Bosnia and Herzegovina"
COUNTRY_TITLE = "Land Cover in 2024: Bosnia and Herzegovina"
TARGET_CRS = "EPSG:3035"

NATURAL_EARTH_COUNTRIES_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)
TERRARIUM_TILE_URL = (
    "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
)
LANDCOVER_TILE_URLS = {
    "33T": "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2024/33T_20240101-20241231.tif",
    "34T": "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2024/34T_20240101-20241231.tif",
}

CAPTION_LINES = [
    "©2026 Milos Popovic (https://milospopovic.net)",
    "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, Microsoft, and AWS Terrarium DEM",
]

TERRARIUM_TILE_SIZE = 256
DEM_DOWNLOAD_WORKERS = 8
DEFAULT_DEM_ZOOM = 11
LANDCOVER_OPACITY = 1.00
LANDCOVER_OVERLAY_CACHE_KEY = f"display-v4-op{int(round(LANDCOVER_OPACITY * 1000.0)):04d}"
LANDCOVER_DESPECKLE_PASSES = 2
LANDCOVER_DESPECKLE_MAX_SUPPORT = 2
LANDCOVER_DESPECKLE_MIN_MAJORITY = 4
OVERLAY_PRESERVE_COLORS = True
OVERLAY_SOLID_SURFACE = False

HEIGHT_SHADE_AZIMUTH = 315.0
HEIGHT_SHADE_ELEVATION = 24.0
HEIGHT_SHADE_AMBIENT = 0.42
HEIGHT_SHADE_DIFFUSE = 0.82
HEIGHT_SHADE_GAMMA = 0.92
HEIGHT_SHADE_CONTRAST = 1.20
HEIGHT_SHADE_SATURATION = 1.26
HEIGHT_SHADE_DETAIL_ZFACTOR = 1.65
HEIGHT_SHADE_BROAD_ZFACTOR = 2.45
HEIGHT_SHADE_BROAD_BLUR_RADIUS = 2.6
HEIGHT_SHADE_BROAD_WEIGHT = 0.62
HEIGHT_SHADE_RELIEF_STRENGTH = 0.23
HEIGHT_SHADE_RELIEF_SLOPE_GAMMA = 0.85
HEIGHT_SHADE_LOCAL_CONTRAST = 0.34
HEIGHT_SHADE_LOCAL_CONTRAST_RADIUS = 1.6

REFERENCE_TERRAIN_WIDTH = 1500.0
BASE_CAMERA_RADIUS = 4800.0
BASE_ZSCALE = 0.035
RELIEF_CAMERA_PULLBACK = 0.96
RELIEF_EXAGGERATION = 0.90

COMPOSITION_BACKGROUND_RGBA = (248, 248, 245, 255)
SUBJECT_SHADOW_RGB = (195, 207, 224)
SUBJECT_SHADOW_ALPHA = 112
SUBJECT_SHADOW_BLUR_RATIO = 0.012
SUBJECT_SHADOW_OFFSET_X_RATIO = 0.018
SUBJECT_SHADOW_OFFSET_Y_RATIO = 0.026

TERRAIN_CONFIG = {
    "phi": 90.0,
    "theta": 18.0,
    "fov": 20.0,
    "sun_azimuth": 315.0,
    "sun_elevation": 25.0,
    "sun_intensity": 1.06,
    "ambient": 0.68,
    "shadow": 0.28,
    "background": [1.0, 1.0, 1.0],
}
PBR_CONFIG = {
    "enabled": True,
    "hdr_path": str(DEFAULT_HDRI_PATH.resolve()),
    "hdr_rotate_deg": 0.0,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.04,
    "msaa": 8,
    "ibl_intensity": 0.16,
    "normal_strength": 1.15,
    "height_ao": {
        "enabled": True,
        "directions": 6,
        "steps": 14,
        "max_distance": 120.0,
        "strength": 0.26,
        "resolution_scale": 0.5,
    },
    "sun_visibility": {
        "enabled": False,
        "mode": "hard",
        "samples": 1,
        "steps": 52,
        "max_distance": 2200.0,
        "softness": 0.0,
        "bias": 0.005,
        "resolution_scale": 1.0,
    },
    "tonemap": {
        "operator": "aces",
        "white_point": 6.0,
        "white_balance_enabled": True,
        "temperature": 6500.0,
        "tint": 0.0,
    },
}


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


LANDCOVER_CLASSES = [
    (1, "#419bdf", "#2f90e3", "Water"),
    (2, "#397d49", "#29924e", "Trees"),
    (4, "#7a87c6", "#6b80c8", "Flooded vegetation"),
    (5, "#e49635", "#f0c54c", "Crops"),
    (7, "#c4281b", "#ef442f", "Built area"),
    (8, "#a59b8f", "#b9ab97", "Bare ground"),
    (9, "#a8ebff", "#f8fcff", "Snow"),
    (11, "#e3e2c3", "#efe4b5", "Rangeland"),
]
LANDCOVER_CLASS_IDS = [class_id for class_id, _, _, _ in LANDCOVER_CLASSES]
LANDCOVER_SOURCE_COLORMAP = {
    0: (0, 0, 0, 0),
    1: (65, 155, 223, 255),
    2: (57, 125, 73, 255),
    4: (122, 135, 198, 255),
    5: (228, 150, 53, 255),
    7: (196, 40, 27, 255),
    8: (165, 155, 143, 255),
    9: (168, 235, 255, 255),
    11: (227, 226, 195, 255),
}
LANDCOVER_SOURCE_COLORMAP_FULL = {
    index: LANDCOVER_SOURCE_COLORMAP.get(index, (0, 0, 0, 0)) for index in range(256)
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hdr", type=Path, default=DEFAULT_HDRI_PATH)
    parser.add_argument("--hdr-intensity", type=float, default=float(PBR_CONFIG["ibl_intensity"]))
    parser.add_argument("--hdr-rotate", type=float, default=float(PBR_CONFIG["hdr_rotate_deg"]))
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--dem-zoom", type=int, default=DEFAULT_DEM_ZOOM)
    parser.add_argument("--viewer-width", type=int, default=1400)
    parser.add_argument("--viewer-height", type=int, default=1400)
    parser.add_argument("--snapshot-width", type=int, default=4200)
    parser.add_argument("--snapshot-height", type=int, default=4200)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _crs_cache_key(crs: str) -> str:
    return str(crs).lower().replace(":", "_").replace("/", "_").replace(" ", "_")


def _overlay_cache_path(cache_dir: Path) -> Path:
    return cache_dir / f"{COUNTRY_A3.lower()}_landcover_overlay_{LANDCOVER_OVERLAY_CACHE_KEY}.png"


def _download(url: str, dest: Path, *, force: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with urlopen(url, timeout=90) as response:
        with tempfile.NamedTemporaryFile(
            prefix=dest.stem,
            suffix=dest.suffix,
            dir=str(dest.parent),
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp_path.replace(dest)
    return dest


def _is_fresh(output_path: Path, input_paths: Iterable[Path]) -> bool:
    if not output_path.exists():
        return False
    output_mtime = output_path.stat().st_mtime
    for input_path in input_paths:
        if not input_path.exists() or input_path.stat().st_mtime > output_mtime:
            return False
    return True


def _load_country_gdf(boundary_zip: Path, target_crs: str) -> gpd.GeoDataFrame:
    countries = gpd.read_file(boundary_zip)
    country = countries[countries["ADM0_A3"] == COUNTRY_A3]
    if country.empty:
        raise RuntimeError(f"Could not find {COUNTRY_NAME} in Natural Earth countries data")
    return country.to_crs(target_crs)


def _load_country_geometry(boundary_zip: Path, target_crs: str):
    country = _load_country_gdf(boundary_zip, target_crs)
    if hasattr(country.geometry, "union_all"):
        return country.geometry.union_all()
    return country.geometry.unary_union


def _tile_x_from_lon(lon: float, zoom: int) -> int:
    n = 1 << zoom
    return int(np.clip(((lon + 180.0) / 360.0) * n, 0, n - 1))


def _tile_y_from_lat(lat: float, zoom: int) -> int:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 1 << zoom
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) * 0.5 * n
    return int(np.clip(y, 0, n - 1))


def _tile_bounds_mercator(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    world_span = 20037508.342789244 * 2.0
    tile_span = world_span / float(1 << zoom)
    west = -20037508.342789244 + x * tile_span
    east = west + tile_span
    north = 20037508.342789244 - y * tile_span
    south = north - tile_span
    return west, south, east, north


def _decode_terrarium(tile_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(tile_path).convert("RGB"), dtype=np.float32)
    return (rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0) - 32768.0


def _download_terrarium_tile(cache_dir: Path, zoom: int, x: int, y: int, *, force: bool) -> Path:
    tile_path = cache_dir / "terrarium" / str(zoom) / str(x) / f"{y}.png"
    url = TERRARIUM_TILE_URL.format(z=zoom, x=x, y=y)
    return _download(url, tile_path, force=force)


def _build_terrarium_source_dem(boundary_zip: Path, cache_dir: Path, zoom: int, *, force: bool) -> Path:
    if zoom < 1 or zoom > 12:
        raise ValueError("DEM zoom must be between 1 and 12")
    output_path = cache_dir / f"{COUNTRY_A3.lower()}_terrarium_dem_z{zoom}_3857.tif"
    if output_path.exists() and not force:
        return output_path

    country_wgs84 = _load_country_geometry(boundary_zip, "EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = country_wgs84.bounds
    x_min = _tile_x_from_lon(lon_min, zoom)
    x_max = _tile_x_from_lon(lon_max, zoom)
    y_min = _tile_y_from_lat(lat_max, zoom)
    y_max = _tile_y_from_lat(lat_min, zoom)
    tiles = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    if not tiles:
        raise RuntimeError(f"No DEM tiles intersected {COUNTRY_NAME}")

    def _fetch(tile: tuple[int, int]) -> tuple[tuple[int, int], Path]:
        x, y = tile
        return tile, _download_terrarium_tile(cache_dir, zoom, x, y, force=force)

    fetched: list[tuple[tuple[int, int], Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEM_DOWNLOAD_WORKERS) as executor:
        for item in executor.map(_fetch, tiles):
            fetched.append(item)

    tile_rows = y_max - y_min + 1
    tile_cols = x_max - x_min + 1
    mosaic = np.empty(
        (tile_rows * TERRARIUM_TILE_SIZE, tile_cols * TERRARIUM_TILE_SIZE),
        dtype=np.float32,
    )
    for (x, y), tile_path in fetched:
        row = y - y_min
        col = x - x_min
        r0 = row * TERRARIUM_TILE_SIZE
        c0 = col * TERRARIUM_TILE_SIZE
        mosaic[r0 : r0 + TERRARIUM_TILE_SIZE, c0 : c0 + TERRARIUM_TILE_SIZE] = _decode_terrarium(
            tile_path
        )

    west, _, _, north = _tile_bounds_mercator(x_min, y_min, zoom)
    _, south, east, _ = _tile_bounds_mercator(x_max, y_max, zoom)
    transform = from_bounds(west, south, east, north, mosaic.shape[1], mosaic.shape[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=mosaic.shape[1],
        height=mosaic.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(mosaic, 1)
    return output_path


def _build_dem(boundary_zip: Path, cache_dir: Path, target_crs: str, zoom: int, *, force: bool) -> Path:
    target_key = _crs_cache_key(target_crs)
    output_path = cache_dir / f"{COUNTRY_A3.lower()}_dem_{target_key}_z{zoom}.tif"
    source_path = _build_terrarium_source_dem(boundary_zip, cache_dir, zoom, force=force)
    if _is_fresh(output_path, [source_path]) and not force:
        return output_path

    country_target = _load_country_geometry(boundary_zip, target_crs)
    country_shapes = [country_target]
    with rasterio.open(source_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="lzw",
        )
        with MemoryFile() as memfile:
            with memfile.open(**profile) as tmp:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(tmp, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=profile["nodata"],
                    init_dest_nodata=True,
                )
            with memfile.open() as reproj:
                data, masked_transform = rasterio.mask.mask(
                    reproj,
                    country_shapes,
                    crop=True,
                    nodata=profile["nodata"],
                    filled=True,
                )
    output_profile = profile.copy()
    output_profile.update(
        transform=masked_transform,
        width=data.shape[2],
        height=data.shape[1],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(data)
    return output_path


def _download_landcover_tiles(cache_dir: Path, *, force: bool) -> dict[str, Path]:
    tile_dir = cache_dir / "landcover_tiles"
    downloads: dict[str, Path] = {}
    for tile_id, url in LANDCOVER_TILE_URLS.items():
        downloads[tile_id] = _download(url, tile_dir / url.rsplit("/", 1)[-1], force=force)
    return downloads


def _crop_landcover_tile(tile_id: str, tile_path: Path, country_wgs84, cache_dir: Path, *, force: bool) -> Path | None:
    output_path = cache_dir / "landcover_crops" / f"{COUNTRY_A3.lower()}_{tile_id}_crop.tif"
    if output_path.exists() and not force:
        return output_path

    with rasterio.open(tile_path) as src:
        country_proj = gpd.GeoSeries([country_wgs84], crs="EPSG:4326").to_crs(src.crs)
        if not country_proj.iloc[0].intersects(box(*src.bounds)):
            return None
        data, transform = rasterio.mask.mask(
            src,
            list(country_proj.geometry),
            crop=True,
            nodata=0,
            filled=True,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=data.shape[2],
            height=data.shape[1],
            transform=transform,
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            try:
                dst.write_colormap(1, src.colormap(1))
            except ValueError:
                pass
    return output_path


def _sanitize_landcover_classes(classes: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(classes, dtype=np.uint8).copy()
    valid_mask = np.isin(cleaned, np.asarray(LANDCOVER_CLASS_IDS, dtype=np.uint8))
    cleaned[~valid_mask] = 0
    return cleaned


def _despeckle_landcover_classes(classes: np.ndarray) -> np.ndarray:
    cleaned = classes.copy()
    height, width = cleaned.shape
    for _ in range(LANDCOVER_DESPECKLE_PASSES):
        same_count = np.zeros((height, width), dtype=np.uint8)
        best_count = np.zeros((height, width), dtype=np.uint8)
        best_class = np.zeros((height, width), dtype=np.uint8)
        for class_id in LANDCOVER_CLASS_IDS:
            mask = cleaned == class_id
            if not np.any(mask):
                continue
            padded = np.pad(mask, 1, mode="constant", constant_values=False).astype(np.uint8)
            count = (
                padded[0:height, 0:width]
                + padded[0:height, 1 : width + 1]
                + padded[0:height, 2 : width + 2]
                + padded[1 : height + 1, 0:width]
                + padded[1 : height + 1, 1 : width + 1]
                + padded[1 : height + 1, 2 : width + 2]
                + padded[2 : height + 2, 0:width]
                + padded[2 : height + 2, 1 : width + 1]
                + padded[2 : height + 2, 2 : width + 2]
            )
            same_count[mask] = count[mask]
            replace = count > best_count
            best_count[replace] = count[replace]
            best_class[replace] = class_id
        replace = (
            (cleaned != 0)
            & (same_count <= LANDCOVER_DESPECKLE_MAX_SUPPORT)
            & (best_count >= LANDCOVER_DESPECKLE_MIN_MAJORITY)
            & (best_class != cleaned)
            & (best_class != 0)
        )
        if not np.any(replace):
            break
        cleaned[replace] = best_class[replace]
    return cleaned


def _build_landcover_classes(
    boundary_zip: Path,
    tile_paths: dict[str, Path],
    dem_path: Path,
    output_dir: Path,
    target_crs: str,
    *,
    force: bool,
) -> Path:
    country_wgs84 = _load_country_geometry(boundary_zip, "EPSG:4326")
    crop_paths: list[Path] = []
    for tile_id, tile_path in tile_paths.items():
        crop_path = _crop_landcover_tile(tile_id, tile_path, country_wgs84, output_dir, force=force)
        if crop_path is not None:
            crop_paths.append(crop_path)
    if not crop_paths:
        raise RuntimeError(f"No land-cover tiles intersected {COUNTRY_NAME}")

    target_key = _crs_cache_key(target_crs)
    classes_path = output_dir / f"{COUNTRY_A3.lower()}_landcover_classes_{target_key}.tif"
    if _is_fresh(classes_path, [dem_path, *crop_paths]) and not force:
        return classes_path

    with rasterio.open(dem_path) as dem:
        classes = np.zeros((dem.height, dem.width), dtype=np.uint8)
        terrain_mask = dem.read(1, masked=True).mask
        for crop_path in crop_paths:
            with rasterio.open(crop_path) as src:
                warped = np.zeros_like(classes)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=warped,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dem.transform,
                    dst_crs=dem.crs,
                    resampling=Resampling.mode,
                    src_nodata=0,
                    dst_nodata=0,
                    init_dest_nodata=True,
                )
            classes = np.where(warped != 0, warped, classes)
        classes[terrain_mask] = 0
        classes = _despeckle_landcover_classes(_sanitize_landcover_classes(classes))

        profile = dem.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(classes_path, "w", **profile) as dst:
            dst.write(classes, 1)
            dst.write_colormap(1, LANDCOVER_SOURCE_COLORMAP_FULL)
    return classes_path


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _landcover_palette_rgb() -> np.ndarray:
    return np.array([_hex_to_rgb(display) for _, _, display, _ in LANDCOVER_CLASSES], dtype=np.uint8)


def _landcover_alpha(opacity: float = LANDCOVER_OPACITY) -> int:
    return int(round(float(np.clip(opacity, 0.0, 1.0)) * 255.0))


def _boost_saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    luminance = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=([2], [0]))
    return np.clip(luminance[:, :, None] + (rgb - luminance[:, :, None]) * amount, 0.0, 1.0)


def _blur_heightmap(heightmap: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return heightmap.astype(np.float32, copy=True)
    finite = np.isfinite(heightmap)
    if not np.any(finite):
        return np.zeros_like(heightmap, dtype=np.float32)
    lo = float(np.nanmin(heightmap[finite]))
    hi = float(np.nanmax(heightmap[finite]))
    span = max(hi - lo, 1e-6)
    normalized = np.clip((heightmap - lo) / span, 0.0, 1.0)
    image = Image.fromarray(np.round(normalized * 255.0).astype(np.uint8), mode="L")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32) / 255.0
    return blurred * span + lo


def _hillshade(heightmap: np.ndarray, azimuth_deg: float, elevation_deg: float, *, z_factor: float) -> np.ndarray:
    dy, dx = np.gradient(heightmap.astype(np.float32))
    slope_x = -dx * float(z_factor)
    slope_y = -dy * float(z_factor)
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    light = np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.cos(azimuth),
        ],
        dtype=np.float32,
    )
    normal = np.dstack((-slope_x, np.ones_like(heightmap, dtype=np.float32), -slope_y))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
    shade = normal @ light
    return np.clip(shade, 0.0, 1.0)


def _height_shade_from_dem(dem: np.ndarray) -> np.ndarray:
    valid = np.isfinite(dem)
    if not np.any(valid):
        return np.ones_like(dem, dtype=np.float32)
    fill_value = float(np.nanmedian(dem[valid]))
    filled = np.where(valid, dem, fill_value).astype(np.float32)
    broad_height = _blur_heightmap(filled, HEIGHT_SHADE_BROAD_BLUR_RADIUS)
    broad = _hillshade(
        broad_height,
        HEIGHT_SHADE_AZIMUTH,
        HEIGHT_SHADE_ELEVATION,
        z_factor=HEIGHT_SHADE_BROAD_ZFACTOR,
    )
    detail = _hillshade(
        filled,
        HEIGHT_SHADE_AZIMUTH,
        HEIGHT_SHADE_ELEVATION,
        z_factor=HEIGHT_SHADE_DETAIL_ZFACTOR,
    )
    shade = HEIGHT_SHADE_BROAD_WEIGHT * broad + (1.0 - HEIGHT_SHADE_BROAD_WEIGHT) * detail

    relief = filled - broad_height
    relief_scale = float(np.nanpercentile(np.abs(relief[valid]), 92.0))
    relief_scale = max(relief_scale, 1e-6)
    relief_unit = np.clip(relief / relief_scale, -1.0, 1.0)

    slope_y, slope_x = np.gradient(broad_height.astype(np.float32))
    slope = np.hypot(slope_x, slope_y)
    slope_scale = float(np.nanpercentile(slope[valid], 85.0))
    slope_scale = max(slope_scale, 1e-6)
    slope_weight = np.clip(slope / slope_scale, 0.0, 1.0)
    slope_weight = np.power(slope_weight, HEIGHT_SHADE_RELIEF_SLOPE_GAMMA, dtype=np.float32)

    shade = np.clip(
        shade + HEIGHT_SHADE_RELIEF_STRENGTH * relief_unit * slope_weight,
        0.0,
        1.0,
    )

    local_base = _blur_heightmap(shade, HEIGHT_SHADE_LOCAL_CONTRAST_RADIUS)
    shade = np.clip(
        shade + HEIGHT_SHADE_LOCAL_CONTRAST * (shade - local_base),
        0.0,
        1.0,
    )
    shade = np.clip(HEIGHT_SHADE_AMBIENT + HEIGHT_SHADE_DIFFUSE * shade, 0.0, 1.0)
    shade = np.power(shade, HEIGHT_SHADE_GAMMA, dtype=np.float32)
    shade = np.clip((shade - 0.5) * HEIGHT_SHADE_CONTRAST + 0.5, 0.0, 1.0)
    return shade.astype(np.float32)


def _classes_to_rgba(classes: np.ndarray, dem: np.ndarray, opacity: float = LANDCOVER_OPACITY) -> np.ndarray:
    palette = _landcover_palette_rgb().astype(np.float32) / 255.0
    rgb_lut = np.zeros((256, 3), dtype=np.float32)
    for index, (class_id, _, _, _) in enumerate(LANDCOVER_CLASSES):
        rgb_lut[class_id] = palette[index]
    rgb = rgb_lut[classes]
    shade = _height_shade_from_dem(dem)
    rgb = np.clip(rgb * shade[:, :, None], 0.0, 1.0)
    rgb = _boost_saturation(rgb, HEIGHT_SHADE_SATURATION)
    rgba = np.zeros((classes.shape[0], classes.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[:, :, 3] = np.where(classes != 0, _landcover_alpha(opacity), 0).astype(np.uint8)
    return rgba


def _build_overlay(
    classes_path: Path,
    dem_path: Path,
    output_path: Path,
    *,
    force: bool,
) -> tuple[Path, list[int]]:
    if _is_fresh(output_path, [classes_path, dem_path]) and not force:
        with rasterio.open(classes_path) as src:
            classes = src.read(1)
        present_classes = [class_id for class_id in LANDCOVER_CLASS_IDS if np.any(classes == class_id)]
        return output_path, present_classes

    with rasterio.open(classes_path) as src:
        classes = src.read(1)
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1, masked=True).filled(np.nan).astype(np.float32)
    present_classes = [class_id for class_id in LANDCOVER_CLASS_IDS if np.any(classes == class_id)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_classes_to_rgba(classes, dem), mode="RGBA").save(output_path)
    return output_path, present_classes


def _legend_entries(present_classes: list[int]) -> list[tuple[str, str]]:
    palette = _landcover_palette_rgb()
    entries: list[tuple[str, str]] = []
    wanted = present_classes or LANDCOVER_CLASS_IDS
    for index, (class_id, _, _, label) in enumerate(LANDCOVER_CLASSES):
        if class_id in wanted:
            color = tuple(int(channel) for channel in palette[index])
            entries.append((_rgb_to_hex(color), label))
    return entries


def _crop_subject(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha_mask = arr[:, :, 3] > 8
    alpha_coverage = float(np.count_nonzero(alpha_mask)) / float(alpha_mask.size)
    if 0.0 < alpha_coverage < 0.98:
        mask = alpha_mask
    else:
        corners = np.asarray(
            [arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]],
            dtype=np.int16,
        )
        background = np.median(corners, axis=0)
        mask = (arr[:, :, 3] > 0) & (
            np.abs(arr[:, :, :3].astype(np.int16) - background).max(axis=2) > 8
        )
    if not np.any(mask):
        return image
    arr[~mask, 3] = 0
    ys, xs = np.nonzero(mask)
    pad = max(8, round(max(image.size) * 0.01))
    return Image.fromarray(arr, mode="RGBA").crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
    )


def _trim_canvas_whitespace(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    mask = (arr[:, :, 3] > 0) & (np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8)
    if not np.any(mask):
        return image
    ys, xs = np.nonzero(mask)
    pad = max(12, round(min(image.size) * 0.012))
    return image.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
    )


def _make_subject_shadow(subject: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    alpha = subject.getchannel("A")
    shadow_alpha = alpha.point(lambda value: int(round(value * SUBJECT_SHADOW_ALPHA / 255.0)))
    shadow = Image.new("RGBA", subject.size, SUBJECT_SHADOW_RGB + (0,))
    shadow.putalpha(shadow_alpha)
    blur_radius = max(8, round(max(subject.size) * SUBJECT_SHADOW_BLUR_RATIO))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    offset = (
        max(10, round(subject.width * SUBJECT_SHADOW_OFFSET_X_RATIO)),
        max(12, round(subject.height * SUBJECT_SHADOW_OFFSET_Y_RATIO)),
    )
    return shadow, offset


def _compose_snapshot(raw_path: Path, output_path: Path, present_classes: list[int]) -> None:
    raw = Image.open(raw_path).convert("RGBA")
    subject = _crop_subject(raw)
    width, height = raw.size
    canvas = Image.new("RGBA", (width, height), COMPOSITION_BACKGROUND_RGBA)
    draw = ImageDraw.Draw(canvas)
    margin = max(20, width // 52)
    top_gap = max(16, height // 180)
    section_gap = max(10, height // 220)

    title_font = _load_font(max(28, width // 30), bold=True)
    legend_title_font = _load_font(max(18, width // 48), bold=True)
    legend_font = _load_font(max(14, width // 82))
    caption_font = _load_font(max(12, width // 105))

    title_box = draw.textbbox((0, 0), COUNTRY_TITLE, font=title_font)
    title_y = margin

    dot = max(10, legend_font.size - 2)
    row_gap = max(6, legend_font.size // 3)
    legend_entries = _legend_entries(present_classes)
    caption_gap = max(4, caption_font.size // 3)
    caption_height = len(CAPTION_LINES) * caption_font.size + max(0, len(CAPTION_LINES) - 1) * caption_gap
    footer_height = caption_height + max(14, height // 120)
    footer_top = height - margin - footer_height

    legend_label_width = 0
    for _, label in legend_entries:
        label_box = draw.textbbox((0, 0), label, font=legend_font)
        legend_label_width = max(legend_label_width, label_box[2] - label_box[0])
    legend_title_box = draw.textbbox((0, 0), "Land Cover", font=legend_title_font)
    legend_columns = 1
    legend_rows = max(1, math.ceil(len(legend_entries) / legend_columns))
    column_gap = max(18, width // 110)
    column_width = dot + 10 + legend_label_width
    legend_width = max(
        legend_title_box[2] - legend_title_box[0],
        legend_columns * column_width + max(0, legend_columns - 1) * column_gap,
    )
    legend_height = legend_title_font.size + 10 + legend_rows * dot + max(0, legend_rows - 1) * row_gap
    legend_gap_above_caption = max(12, height // 180)
    footer_lift = max(72, height // 36)
    layout_legend_y = footer_top - legend_height - legend_gap_above_caption
    legend_x = margin
    legend_y = layout_legend_y - footer_lift

    caption_top = footer_top + max(10, height // 220) - footer_lift

    map_top = title_y + (title_box[3] - title_box[1]) + top_gap
    map_width = max(1, width - margin * 2)
    map_bottom = footer_top - section_gap
    map_height = max(1, map_bottom - map_top)
    subject.thumbnail((map_width, map_height), Image.Resampling.LANCZOS)
    map_x = margin + max(0, (map_width - subject.width) // 2)
    map_y = map_top + max(0, (map_height - subject.height) // 2)
    subject_shadow, shadow_offset = _make_subject_shadow(subject)
    canvas.alpha_composite(
        subject_shadow,
        dest=(map_x + shadow_offset[0], map_y + shadow_offset[1]),
    )
    canvas.alpha_composite(subject, dest=(map_x, map_y))

    draw.text((width // 2, title_y), COUNTRY_TITLE, fill=(38, 38, 42), font=title_font, anchor="ma")
    draw.text((legend_x, legend_y), "Land Cover", fill=(54, 54, 60), font=legend_title_font)
    legend_cursor = legend_y + legend_title_font.size + 10
    dot = max(10, legend_font.size - 2)
    for index, (color, label) in enumerate(legend_entries):
        column = index // legend_rows
        row = index % legend_rows
        item_x = legend_x + column * (column_width + column_gap)
        item_y = legend_cursor + row * (dot + row_gap)
        draw.ellipse((item_x, item_y, item_x + dot, item_y + dot), fill=color, outline=(160, 160, 160))
        draw.text((item_x + dot + 10, item_y - 2), label, fill=(72, 72, 78), font=legend_font)
    for index, line in enumerate(CAPTION_LINES):
        y = caption_top + index * (caption_font.size + caption_gap)
        draw.text((width // 2, y), line, fill=(82, 82, 88), font=caption_font, anchor="ma")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _trim_canvas_whitespace(canvas).save(output_path)


def _render(
    snapshot_path: Path,
    dem_path: Path,
    overlay_path: Path,
    present_classes: list[int],
    *,
    hdr_path: Path,
    hdr_intensity: float,
    hdr_rotate: float,
    viewer_width: int,
    viewer_height: int,
    snapshot_width: int,
    snapshot_height: int,
) -> None:
    snapshot_path = snapshot_path.resolve()
    hdr_path = hdr_path.resolve()
    if not hdr_path.is_file():
        raise FileNotFoundError(f"HDRI not found: {hdr_path}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    raw_snapshot_path = snapshot_path.with_name(f"{snapshot_path.stem}_raw.png")
    raw_snapshot_path.unlink(missing_ok=True)
    snapshot_path.unlink(missing_ok=True)

    with rasterio.open(dem_path) as dem_src:
        terrain_width = float(max(dem_src.width, dem_src.height))
    terrain_xy_scale = terrain_width / REFERENCE_TERRAIN_WIDTH
    terrain_relief_scale = math.sqrt(max(terrain_xy_scale, 1e-6))
    radius = BASE_CAMERA_RADIUS * terrain_xy_scale * RELIEF_CAMERA_PULLBACK
    zscale = BASE_ZSCALE * terrain_relief_scale * RELIEF_EXAGGERATION

    terrain_cmd = dict(TERRAIN_CONFIG)
    terrain_cmd["radius"] = radius
    terrain_cmd["zscale"] = zscale
    pbr_config = copy.deepcopy(PBR_CONFIG)
    pbr_config["hdr_path"] = str(hdr_path)
    pbr_config["ibl_intensity"] = float(hdr_intensity)
    pbr_config["hdr_rotate_deg"] = float(hdr_rotate)

    with f3d.open_viewer_async(
        terrain_path=dem_path,
        width=viewer_width,
        height=viewer_height,
        timeout=45.0,
    ) as viewer:
        viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
        viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr_config})
        viewer.load_overlay(
            "bosnia_landcover",
            overlay_path,
            extent=(0.0, 0.0, 1.0, 1.0),
            opacity=1.0,
            preserve_colors=OVERLAY_PRESERVE_COLORS,
        )
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": OVERLAY_SOLID_SURFACE})
        time.sleep(2.0)
        viewer.snapshot(raw_snapshot_path, width=snapshot_width, height=snapshot_height)

    _compose_snapshot(raw_snapshot_path, snapshot_path, present_classes)


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    target_key = _crs_cache_key(TARGET_CRS)
    snapshot_path = (
        args.snapshot.resolve()
        if args.snapshot is not None
        else output_dir / "bosnia_landcover_2024.png"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    boundary_zip = _download(
        NATURAL_EARTH_COUNTRIES_URL,
        cache_dir / "ne_10m_admin_0_countries.zip",
        force=bool(args.force),
    )
    tile_paths = _download_landcover_tiles(cache_dir, force=bool(args.force))
    dem_path = _build_dem(
        boundary_zip,
        cache_dir,
        TARGET_CRS,
        int(args.dem_zoom),
        force=bool(args.force),
    )
    classes_path = _build_landcover_classes(
        boundary_zip,
        tile_paths,
        dem_path,
        cache_dir,
        TARGET_CRS,
        force=bool(args.force),
    )
    overlay_cache_path, present_classes = _build_overlay(
        classes_path,
        dem_path,
        _overlay_cache_path(cache_dir),
        force=bool(args.force),
    )
    overlay_path = output_dir / "bosnia_landcover_overlay.png"
    if not _is_fresh(overlay_path, [overlay_cache_path]) or bool(args.force):
        shutil.copy2(overlay_cache_path, overlay_path)

    dem_copy_path = output_dir / f"bosnia_dem_{target_key}.tif"
    if not _is_fresh(dem_copy_path, [dem_path]) or bool(args.force):
        shutil.copy2(dem_path, dem_copy_path)
    classes_copy_path = output_dir / f"bosnia_landcover_classes_{target_key}.tif"
    if not _is_fresh(classes_copy_path, [classes_path]) or bool(args.force):
        shutil.copy2(classes_path, classes_copy_path)

    print(f"[BIH] DEM: {dem_copy_path}")
    print(f"[BIH] classes: {classes_copy_path}")
    print(f"[BIH] overlay: {overlay_path}")
    print(
        f"[BIH] hdri: {args.hdr.resolve()} "
        f"(intensity={float(args.hdr_intensity):.2f}, rotate={float(args.hdr_rotate):.1f} deg)"
    )
    print(f"[BIH] visible classes: {', '.join(str(value) for value in present_classes) if present_classes else 'none'}")

    if args.skip_render:
        print("[BIH] skipping 3D render (--skip-render)")
        return 0

    _render(
        snapshot_path,
        dem_copy_path,
        overlay_path,
        present_classes,
        hdr_path=args.hdr,
        hdr_intensity=float(args.hdr_intensity),
        hdr_rotate=float(args.hdr_rotate),
        viewer_width=int(args.viewer_width),
        viewer_height=int(args.viewer_height),
        snapshot_width=int(args.snapshot_width),
        snapshot_height=int(args.snapshot_height),
    )
    print(f"[BIH] snapshot: {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
