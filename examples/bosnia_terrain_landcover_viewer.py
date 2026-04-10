#!/usr/bin/env python3
"""Bosnia land-cover terrain render."""

from __future__ import annotations

import argparse
import concurrent.futures
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

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "out" / "bosnia_terrain_landcover"
CACHE_DIR = ROOT / "examples" / ".cache" / "bosnia_terrain_landcover"
HDR = ROOT / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"

COUNTRY_A3 = "BIH"
COUNTRY_NAME = "Bosnia and Herzegovina"
COUNTRY_TITLE = "Land Cover in 2024: Bosnia and Herzegovina"
TARGET_CRS = "EPSG:3035"
NATURAL_EARTH = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_countries.zip"
)
TERRARIUM = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
LANDCOVER_URLS = {
    "33T": "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2024/33T_20240101-20241231.tif",
    "34T": "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2024/34T_20240101-20241231.tif",
}
CAPTION_LINES = [
    "©2026 Milos Popovic (https://milospopovic.net)",
    "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, Microsoft, and AWS Terrarium DEM",
]

VIEWER_SIZE = (1400, 1400)
SNAPSHOT_SIZE = (4200, 4200)
DEM_ZOOM = 11
TILE_SIZE = 256
DEM_WORKERS = 8
PASS_SETTLE_SECONDS = 2.0

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
LANDCOVER_CLASS_IDS = [class_id for class_id, *_ in LANDCOVER_CLASSES]
LANDCOVER_OVERLAY = {"opacity": 1.0, "bake": 0.10, "sat": 1.04, "lift": 1.02}
LANDCOVER_OVERLAY_CACHE_KEY = (
    f"display-v13-op{int(round(LANDCOVER_OVERLAY['opacity'] * 1000)):04d}"
    f"-bake{int(round(LANDCOVER_OVERLAY['bake'] * 1000)):04d}"
    f"-sat{int(round(LANDCOVER_OVERLAY['sat'] * 1000)):04d}"
    f"-lift{int(round(LANDCOVER_OVERLAY['lift'] * 1000)):04d}"
)
DISPLAY_RGB = np.array(
    [tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) for _, _, color, _ in LANDCOVER_CLASSES],
    dtype=np.uint8,
)
SOURCE_COLORMAP = {
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
SOURCE_COLORMAP_FULL = {index: SOURCE_COLORMAP.get(index, (0, 0, 0, 0)) for index in range(256)}
DESPECKLE = {"passes": 2, "max_support": 2, "min_majority": 4}
SHADE = {
    "azimuth": 314.0,
    "elevation": 22.0,
    "ambient": 0.46,
    "diffuse": 0.80,
    "gamma": 0.99,
    "contrast": 1.18,
    "detail_z": 1.85,
    "broad_z": 2.55,
    "broad_blur": 2.4,
    "broad_weight": 0.60,
    "relief_strength": 0.20,
    "slope_gamma": 0.85,
    "local_contrast": 0.24,
    "local_radius": 1.45,
}
CAMERA = {"ref": 1500.0, "radius": 4800.0, "zscale": 0.035, "pullback": 1.03, "exaggeration": 1.08}
TERRAIN = {
    "phi": 90.0,
    "theta": 32.0,
    "fov": 23.0,
    "sun_azimuth": 314.0,
    "sun_elevation": 28.0,
    "sun_intensity": 1.70,
    "ambient": 0.58,
    "shadow": 0.40,
    "background": [1.0, 1.0, 1.0],
}
PBR = {
    "enabled": True,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.01,
    "msaa": 8,
    "ibl_intensity": 0.12,
    "normal_strength": 1.00,
    "height_ao": {"enabled": True, "directions": 8, "steps": 20, "max_distance": 200.0, "strength": 0.30, "resolution_scale": 0.65},
    "sun_visibility": {"enabled": True, "mode": "soft", "samples": 2, "steps": 40, "max_distance": 2200.0, "softness": 0.55, "bias": 0.0032, "resolution_scale": 0.75},
    "tonemap": {"operator": "aces", "white_point": 6.0, "white_balance_enabled": True, "temperature": 6500.0, "tint": 0.0},
}
RELIEF_TERRAIN = {"sun_elevation": 18.0, "sun_intensity": 3.70, "ambient": 0.40, "shadow": 0.70}
RELIEF_PBR = {
    "exposure": 1.01,
    "ibl_intensity": 0.03,
    "normal_strength": 1.55,
    "height_ao": {"enabled": True, "directions": 10, "steps": 24, "max_distance": 240.0, "strength": 0.40, "resolution_scale": 0.72},
    "sun_visibility": {"enabled": True, "mode": "hard", "samples": 1, "steps": 96, "max_distance": 3200.0, "softness": 0.0, "bias": 0.0028, "resolution_scale": 0.90},
}
COMP = {
    "bg": (248, 248, 245, 255),
    "scale": 1.22,
    "shift_x": 0.055,
    "shift_y": -0.036,
    "mask_min": 4.0,
    "mask_max": 18.0,
    "relief_blur": 2.6,
    "relief_low": 6.0,
    "relief_high": 94.0,
    "relief_contrast": 1.08,
    "relief_gamma": 1.02,
    "value_floor": 0.78,
    "value_gain": 0.24,
    "highlight_start": 0.72,
    "highlight_gain": 0.03,
}
SHADOW = {"rgb": (142, 148, 157), "layers": ((48, 0.004, 0.010), (22, 0.010, 0.024))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--dem-zoom", type=int, default=DEM_ZOOM)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _overlay_cache_path(cache_dir: Path) -> Path:
    return cache_dir / f"{COUNTRY_A3.lower()}_landcover_overlay_{LANDCOVER_OVERLAY_CACHE_KEY}.png"


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _landcover_alpha(opacity: float = LANDCOVER_OVERLAY["opacity"]) -> int:
    return int(round(float(np.clip(opacity, 0.0, 1.0)) * 255.0))


def _is_fresh(output_path: Path, input_paths: Iterable[Path]) -> bool:
    return output_path.exists() and all(
        path.exists() and path.stat().st_mtime <= output_path.stat().st_mtime
        for path in input_paths
    )


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
            tmp = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp.replace(dest)
    return dest


def _country_geometry(boundary_zip: Path, target_crs: str):
    countries = gpd.read_file(boundary_zip)
    country = countries[countries["ADM0_A3"] == COUNTRY_A3]
    if country.empty:
        raise RuntimeError(f"Could not find {COUNTRY_NAME} in Natural Earth countries data")
    country = country.to_crs(target_crs)
    return country.geometry.union_all() if hasattr(country.geometry, "union_all") else country.geometry.unary_union


def _tile_x_from_lon(lon: float, zoom: int) -> int:
    return int(np.clip(((lon + 180.0) / 360.0) * (1 << zoom), 0, (1 << zoom) - 1))


def _tile_y_from_lat(lat: float, zoom: int) -> int:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 1 << zoom
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) * 0.5 * n
    return int(np.clip(y, 0, n - 1))


def _tile_bounds_mercator(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    span = 20037508.342789244 * 2.0 / float(1 << zoom)
    west = -20037508.342789244 + x * span
    east = west + span
    north = 20037508.342789244 - y * span
    return west, north - span, east, north


def _decode_terrarium(tile_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(tile_path).convert("RGB"), dtype=np.float32)
    return (rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0) - 32768.0

def _build_terrarium_source_dem(
    boundary_zip: Path,
    cache_dir: Path,
    zoom: int,
    *,
    force: bool,
) -> Path:
    if not 1 <= zoom <= 12:
        raise ValueError("DEM zoom must be between 1 and 12")
    output = cache_dir / f"{COUNTRY_A3.lower()}_terrarium_dem_z{zoom}_3857.tif"
    if output.exists() and not force:
        return output

    lon_min, lat_min, lon_max, lat_max = _country_geometry(boundary_zip, "EPSG:4326").bounds
    x_min, x_max = _tile_x_from_lon(lon_min, zoom), _tile_x_from_lon(lon_max, zoom)
    y_min, y_max = _tile_y_from_lat(lat_max, zoom), _tile_y_from_lat(lat_min, zoom)
    tiles = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    if not tiles:
        raise RuntimeError(f"No DEM tiles intersected {COUNTRY_NAME}")

    def _fetch(tile: tuple[int, int]) -> tuple[tuple[int, int], Path]:
        x, y = tile
        path = cache_dir / "terrarium" / str(zoom) / str(x) / f"{y}.png"
        return tile, _download(TERRARIUM.format(z=zoom, x=x, y=y), path, force=force)

    fetched: list[tuple[tuple[int, int], Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEM_WORKERS) as pool:
        for item in pool.map(_fetch, tiles):
            fetched.append(item)

    rows, cols = y_max - y_min + 1, x_max - x_min + 1
    mosaic = np.empty((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)
    for (x, y), path in fetched:
        r0 = (y - y_min) * TILE_SIZE
        c0 = (x - x_min) * TILE_SIZE
        mosaic[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = _decode_terrarium(path)

    west, _, _, north = _tile_bounds_mercator(x_min, y_min, zoom)
    _, south, east, _ = _tile_bounds_mercator(x_max, y_max, zoom)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=mosaic.shape[1],
        height=mosaic.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_bounds(west, south, east, north, mosaic.shape[1], mosaic.shape[0]),
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(mosaic, 1)
    return output


def _build_dem(boundary_zip: Path, cache_dir: Path, zoom: int, *, force: bool) -> Path:
    output = cache_dir / f"{COUNTRY_A3.lower()}_dem_epsg_3035_z{zoom}.tif"
    source = _build_terrarium_source_dem(boundary_zip, cache_dir, zoom, force=force)
    if _is_fresh(output, [source]) and not force:
        return output

    country = _country_geometry(boundary_zip, TARGET_CRS)
    with rasterio.open(source) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="lzw",
        )
        with MemoryFile() as mem:
            with mem.open(**profile) as tmp:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(tmp, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=profile["nodata"],
                    init_dest_nodata=True,
                )
            with mem.open() as reproj:
                data, masked_transform = rasterio.mask.mask(
                    reproj,
                    [country],
                    crop=True,
                    nodata=profile["nodata"],
                    filled=True,
                )

    profile.update(transform=masked_transform, width=data.shape[2], height=data.shape[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data)
    return output


def _download_landcover_tiles(cache_dir: Path, *, force: bool) -> dict[str, Path]:
    tile_dir = cache_dir / "landcover_tiles"
    return {
        tile_id: _download(url, tile_dir / url.rsplit("/", 1)[-1], force=force)
        for tile_id, url in LANDCOVER_URLS.items()
    }


def _crop_landcover_tile(
    tile_id: str,
    tile_path: Path,
    country_wgs84,
    cache_dir: Path,
    *,
    force: bool,
) -> Path | None:
    output = cache_dir / "landcover_crops" / f"{COUNTRY_A3.lower()}_{tile_id}_crop.tif"
    if output.exists() and not force:
        return output

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
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data)
            try:
                dst.write_colormap(1, src.colormap(1))
            except ValueError:
                pass
    return output


def _clean_landcover_classes(classes: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(classes, dtype=np.uint8).copy()
    cleaned[~np.isin(cleaned, np.asarray(LANDCOVER_CLASS_IDS, dtype=np.uint8))] = 0
    height, width = cleaned.shape
    for _ in range(DESPECKLE["passes"]):
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
            & (same_count <= DESPECKLE["max_support"])
            & (best_count >= DESPECKLE["min_majority"])
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
    cache_dir: Path,
    *,
    force: bool,
) -> Path:
    country = _country_geometry(boundary_zip, "EPSG:4326")
    crop_paths = [
        crop
        for tile_id, tile_path in tile_paths.items()
        if (crop := _crop_landcover_tile(tile_id, tile_path, country, cache_dir, force=force))
        is not None
    ]
    if not crop_paths:
        raise RuntimeError(f"No land-cover tiles intersected {COUNTRY_NAME}")
    output = cache_dir / f"{COUNTRY_A3.lower()}_landcover_classes_epsg_3035.tif"
    if _is_fresh(output, [dem_path, *crop_paths]) and not force:
        return output

    with rasterio.open(dem_path) as dem:
        classes = np.zeros((dem.height, dem.width), dtype=np.uint8)
        terrain_mask = dem.read(1, masked=True).mask
        for crop in crop_paths:
            with rasterio.open(crop) as src:
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
        classes = _clean_landcover_classes(classes)
        profile = dem.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(classes, 1)
            dst.write_colormap(1, SOURCE_COLORMAP_FULL)
    return output

def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _boost_saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    lum = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=([2], [0]))
    return np.clip(lum[:, :, None] + (rgb - lum[:, :, None]) * amount, 0.0, 1.0)


def _blur_heightmap(heightmap: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return heightmap.astype(np.float32, copy=True)
    finite = np.isfinite(heightmap)
    if not np.any(finite):
        return np.zeros_like(heightmap, dtype=np.float32)
    lo, hi = float(np.nanmin(heightmap[finite])), float(np.nanmax(heightmap[finite]))
    span = max(hi - lo, 1e-6)
    normalized = np.clip((heightmap - lo) / span, 0.0, 1.0)
    image = Image.fromarray(np.round(normalized * 255.0).astype(np.uint8), mode="L")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32) / 255.0
    return blurred * span + lo


def _hillshade(heightmap: np.ndarray, azimuth_deg: float, elevation_deg: float, *, z_factor: float) -> np.ndarray:
    dy, dx = np.gradient(heightmap.astype(np.float32))
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    light = np.array([np.cos(elevation) * np.sin(azimuth), np.sin(elevation), np.cos(elevation) * np.cos(azimuth)], dtype=np.float32)
    normal = np.dstack((-dx * float(z_factor), np.ones_like(heightmap, dtype=np.float32), dy * float(z_factor)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
    return np.clip(normal @ light, 0.0, 1.0)


def _height_shade_from_dem(dem: np.ndarray) -> np.ndarray:
    valid = np.isfinite(dem)
    if not np.any(valid):
        return np.ones_like(dem, dtype=np.float32)
    filled = np.where(valid, dem, float(np.nanmedian(dem[valid]))).astype(np.float32)
    broad_height = _blur_heightmap(filled, SHADE["broad_blur"])
    broad = _hillshade(broad_height, SHADE["azimuth"], SHADE["elevation"], z_factor=SHADE["broad_z"])
    detail = _hillshade(filled, SHADE["azimuth"], SHADE["elevation"], z_factor=SHADE["detail_z"])
    shade = SHADE["broad_weight"] * broad + (1.0 - SHADE["broad_weight"]) * detail
    relief = filled - broad_height
    relief_unit = np.clip(relief / max(float(np.nanpercentile(np.abs(relief[valid]), 92.0)), 1e-6), -1.0, 1.0)
    slope_y, slope_x = np.gradient(broad_height.astype(np.float32))
    slope = np.hypot(slope_x, slope_y)
    slope_weight = np.clip(slope / max(float(np.nanpercentile(slope[valid], 85.0)), 1e-6), 0.0, 1.0)
    slope_weight = np.power(slope_weight, SHADE["slope_gamma"], dtype=np.float32)
    shade = np.clip(shade + SHADE["relief_strength"] * relief_unit * slope_weight, 0.0, 1.0)
    shade = np.clip(shade + SHADE["local_contrast"] * (shade - _blur_heightmap(shade, SHADE["local_radius"])), 0.0, 1.0)
    shade = np.clip(SHADE["ambient"] + SHADE["diffuse"] * shade, 0.0, 1.0)
    shade = np.power(shade, SHADE["gamma"], dtype=np.float32)
    return np.clip((shade - 0.5) * SHADE["contrast"] + 0.5, 0.0, 1.0).astype(np.float32)


def _classes_to_rgba(classes: np.ndarray, dem: np.ndarray, opacity: float = LANDCOVER_OVERLAY["opacity"]) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.float32)
    for idx, class_id in enumerate(LANDCOVER_CLASS_IDS):
        lut[class_id] = DISPLAY_RGB[idx].astype(np.float32) / 255.0
    rgb = lut[classes]
    shade = 1.0 - (1.0 - _height_shade_from_dem(dem)) * LANDCOVER_OVERLAY["bake"]
    rgb = np.clip(rgb * shade[:, :, None], 0.0, 1.0)
    rgb = _boost_saturation(rgb, LANDCOVER_OVERLAY["sat"])
    rgb = np.clip(rgb * LANDCOVER_OVERLAY["lift"], 0.0, 1.0)
    rgba = np.zeros((classes.shape[0], classes.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    rgba[:, :, 3] = np.where(classes != 0, _landcover_alpha(opacity), 0).astype(np.uint8)
    return rgba


def _build_overlay(classes_path: Path, dem_path: Path, output_path: Path, *, force: bool) -> tuple[Path, list[int]]:
    with rasterio.open(classes_path) as src:
        classes = src.read(1)
    present = [class_id for class_id in LANDCOVER_CLASS_IDS if np.any(classes == class_id)]
    if _is_fresh(output_path, [classes_path, dem_path]) and not force:
        return output_path, present
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1, masked=True).filled(np.nan).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_classes_to_rgba(classes, dem), mode="RGBA").save(output_path)
    return output_path, present


def _legend_entries(present_classes: list[int]) -> list[tuple[str, str]]:
    wanted = set(present_classes or LANDCOVER_CLASS_IDS)
    return [(_rgb_to_hex(tuple(int(v) for v in DISPLAY_RGB[i])), label) for i, (class_id, _, _, label) in enumerate(LANDCOVER_CLASSES) if class_id in wanted]


def _subject_alpha(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    src_alpha = arr[:, :, 3].astype(np.float32) / 255.0
    coverage = float(np.count_nonzero(src_alpha > 0.03)) / float(src_alpha.size)
    if 0.0 < coverage < 0.98:
        return src_alpha
    corners = np.asarray([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]], dtype=np.float32)
    bg = np.median(corners, axis=0)
    dist = np.abs(arr[:, :, :3].astype(np.float32) - bg[None, None, :]).max(axis=2)
    return np.clip((dist - COMP["mask_min"]) / max(COMP["mask_max"] - COMP["mask_min"], 1e-6), 0.0, 1.0) * (arr[:, :, 3] > 0).astype(np.float32)


def _crop_subject(image: Image.Image) -> Image.Image:
    alpha = _subject_alpha(image)
    mask = alpha > 0.03
    if not np.any(mask):
        return image
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    arr[:, :, 3] = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    pad = max(8, round(max(image.size) * 0.01))
    return Image.fromarray(arr, mode="RGBA").crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(image.width, int(xs.max()) + pad + 1), min(image.height, int(ys.max()) + pad + 1)))


def _trim_background(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    mask = (arr[:, :, 3] > 0) & (np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8)
    if not np.any(mask):
        return image
    ys, xs = np.nonzero(mask)
    pad = max(12, round(min(image.size) * 0.012))
    return image.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(image.width, int(xs.max()) + pad + 1), min(image.height, int(ys.max()) + pad + 1)))


def _shadow_offset_from_sun(subject_size: tuple[int, int], *, azimuth_deg: float, elevation_deg: float, distance_ratio: float) -> tuple[int, int]:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(np.clip(elevation_deg, 0.0, 89.0)))
    light_x, light_y = math.sin(azimuth), -math.cos(azimuth)
    length = max(math.hypot(light_x, light_y), 1e-6)
    distance = max(3, round(max(subject_size) * distance_ratio * (0.82 + 0.38 * math.cos(elevation))))
    return int(round((-light_x / length) * distance)), int(round((-light_y / length) * distance))


def _make_subject_shadow(subject: Image.Image, *, azimuth_deg: float, elevation_deg: float) -> tuple[Image.Image, tuple[int, int]]:
    layers = []
    for alpha_scale, blur_ratio, distance_ratio in SHADOW["layers"]:
        alpha = subject.getchannel("A").point(lambda value, a=alpha_scale: int(round(value * a / 255.0)))
        shadow = Image.new("RGBA", subject.size, SHADOW["rgb"] + (0,))
        shadow.putalpha(alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(2, round(max(subject.size) * blur_ratio))))
        offset = _shadow_offset_from_sun(subject.size, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, distance_ratio=distance_ratio)
        layers.append((shadow, offset))
    min_x, min_y = min(offset[0] for _, offset in layers), min(offset[1] for _, offset in layers)
    max_x = max(offset[0] + shadow.width for shadow, offset in layers)
    max_y = max(offset[1] + shadow.height for shadow, offset in layers)
    composite = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
    for shadow, offset in layers:
        composite.alpha_composite(shadow, dest=(offset[0] - min_x, offset[1] - min_y))
    return composite, (min_x, min_y)


def _combine_render_passes(color_raw_path: Path, relief_raw_path: Path) -> Image.Image:
    color_image = Image.open(color_raw_path).convert("RGBA")
    relief_image = Image.open(relief_raw_path).convert("RGBA")
    if color_image.size != relief_image.size:
        raise ValueError("Color and relief passes must have identical dimensions")
    color = np.asarray(color_image, dtype=np.float32) / 255.0
    relief = np.asarray(relief_image.filter(ImageFilter.GaussianBlur(radius=COMP["relief_blur"])), dtype=np.float32) / 255.0
    alpha = _subject_alpha(color_image)
    mask = alpha > 0.03
    if not np.any(mask):
        return color_image
    luminance = relief[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low = float(np.percentile(luminance[mask], COMP["relief_low"]))
    high = float(np.percentile(luminance[mask], COMP["relief_high"]))
    shade = np.clip((luminance - low) / max(high - low, 1e-6), 0.0, 1.0)
    shade = np.clip((shade - 0.5) * COMP["relief_contrast"] + 0.5, 0.0, 1.0)
    shade = np.power(shade, COMP["relief_gamma"], dtype=np.float32)
    base_rgb = color[:, :, :3]
    base_value = np.max(base_rgb, axis=2, keepdims=True)
    base_scale = np.divide(base_rgb, np.maximum(base_value, 1e-6))
    target_value = np.clip(base_value[:, :, 0] * (COMP["value_floor"] + COMP["value_gain"] * shade) + COMP["highlight_gain"] * np.maximum(shade - COMP["highlight_start"], 0.0), 0.0, 1.0)
    combined = np.zeros_like(color)
    combined[:, :, :3] = np.clip(base_scale * target_value[:, :, None], 0.0, 1.0)
    combined[:, :, 3] = np.clip(alpha, 0.0, 1.0)
    combined[~mask, :3] = 0.0
    return Image.fromarray(np.round(combined * 255.0).astype(np.uint8), mode="RGBA")

def _compose_snapshot(raw: Image.Image, output_path: Path, present_classes: list[int], *, sun_azimuth: float, sun_elevation: float) -> None:
    subject = _crop_subject(raw)
    width, height = raw.size
    canvas = Image.new("RGBA", (width, height), COMP["bg"])
    draw = ImageDraw.Draw(canvas)
    margin = max(20, width // 52)
    top_gap, section_gap = max(16, height // 180), max(10, height // 220)
    title_font = _load_font(max(28, width // 30), bold=True)
    legend_title_font = _load_font(max(18, width // 48), bold=True)
    legend_font = _load_font(max(14, width // 82))
    caption_font = _load_font(max(12, width // 105))
    title_box = draw.textbbox((0, 0), COUNTRY_TITLE, font=title_font)
    title_y = margin
    entries = _legend_entries(present_classes)
    dot = max(10, getattr(legend_font, "size", 14) - 2)
    row_gap = max(6, getattr(legend_font, "size", 14) // 3)
    caption_gap = max(4, getattr(caption_font, "size", 12) // 3)
    caption_h = len(CAPTION_LINES) * getattr(caption_font, "size", 12) + max(0, len(CAPTION_LINES) - 1) * caption_gap
    footer_top = height - margin - (caption_h + max(14, height // 120))
    legend_h = getattr(legend_title_font, "size", 18) + 10 + len(entries) * dot + max(0, len(entries) - 1) * row_gap
    legend_x = margin
    legend_y = footer_top - legend_h - max(12, height // 180) - max(72, height // 36)
    caption_top = footer_top + max(10, height // 220) - max(72, height // 36)
    map_top = title_y + (title_box[3] - title_box[1]) + top_gap
    map_w = max(1, width - margin * 2)
    map_h = max(1, footer_top - section_gap - map_top)
    if not math.isclose(COMP["scale"], 1.0):
        subject = subject.resize((max(1, round(subject.width * COMP["scale"])), max(1, round(subject.height * COMP["scale"]))), resample=Image.Resampling.LANCZOS)
    map_x = margin + max(0, (map_w - subject.width) // 2) + round(width * COMP["shift_x"])
    map_y = map_top + max(0, (map_h - subject.height) // 2) + round(height * COMP["shift_y"])
    subject_shadow, shadow_offset = _make_subject_shadow(subject, azimuth_deg=sun_azimuth, elevation_deg=sun_elevation)
    canvas.alpha_composite(subject_shadow, dest=(map_x + shadow_offset[0], map_y + shadow_offset[1]))
    canvas.alpha_composite(subject, dest=(map_x, map_y))
    draw.text((width // 2, title_y), COUNTRY_TITLE, fill=(38, 38, 42), font=title_font, anchor="ma")
    draw.text((legend_x, legend_y), "Land Cover", fill=(54, 54, 60), font=legend_title_font)
    legend_cursor = legend_y + getattr(legend_title_font, "size", 18) + 10
    for row, (color, label) in enumerate(entries):
        item_y = legend_cursor + row * (dot + row_gap)
        draw.ellipse((legend_x, item_y, legend_x + dot, item_y + dot), fill=color, outline=(160, 160, 160))
        draw.text((legend_x + dot + 10, item_y - 2), label, fill=(72, 72, 78), font=legend_font)
    for index, line in enumerate(CAPTION_LINES):
        draw.text((width // 2, caption_top + index * (getattr(caption_font, "size", 12) + caption_gap)), line, fill=(82, 82, 88), font=caption_font, anchor="ma")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _trim_background(canvas).save(output_path)


def _render(snapshot_path: Path, dem_path: Path, overlay_path: Path, present_classes: list[int]) -> None:
    hdr = HDR.resolve()
    if not hdr.is_file():
        raise FileNotFoundError(f"HDRI not found: {hdr}")
    with rasterio.open(dem_path) as dem_src:
        terrain_width = float(max(dem_src.width, dem_src.height))
    terrain_xy = terrain_width / CAMERA["ref"]
    radius = CAMERA["radius"] * terrain_xy * CAMERA["pullback"]
    zscale = CAMERA["zscale"] * math.sqrt(max(terrain_xy, 1e-6)) * CAMERA["exaggeration"]
    terrain_cmd = {**TERRAIN, "radius": radius, "zscale": zscale}
    relief_terrain = {**terrain_cmd, **RELIEF_TERRAIN}
    color_pbr = {**PBR, "hdr_path": str(hdr), "height_ao": dict(PBR["height_ao"]), "sun_visibility": dict(PBR["sun_visibility"]), "tonemap": dict(PBR["tonemap"])}
    relief_pbr = {**color_pbr, **RELIEF_PBR, "height_ao": dict(RELIEF_PBR["height_ao"]), "sun_visibility": dict(RELIEF_PBR["sun_visibility"]), "tonemap": dict(color_pbr["tonemap"])}
    snapshot_path = snapshot_path.resolve()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge3d_bih_render_") as tmp:
        color_raw = Path(tmp) / "color_raw.png"
        relief_raw = Path(tmp) / "relief_raw.png"
        with f3d.open_viewer_async(terrain_path=dem_path, width=VIEWER_SIZE[0], height=VIEWER_SIZE[1], timeout=45.0) as viewer:
            viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **color_pbr})
            viewer.load_overlay("bosnia_landcover", overlay_path, extent=(0.0, 0.0, 1.0, 1.0), opacity=1.0, preserve_colors=True)
            viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
            viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
            viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
            time.sleep(PASS_SETTLE_SECONDS)
            viewer.snapshot(color_raw, width=SNAPSHOT_SIZE[0], height=SNAPSHOT_SIZE[1])
            viewer.send_ipc({"cmd": "set_terrain", **relief_terrain})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **relief_pbr})
            viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": False})
            time.sleep(PASS_SETTLE_SECONDS)
            viewer.snapshot(relief_raw, width=SNAPSHOT_SIZE[0], height=SNAPSHOT_SIZE[1])
        raw = _combine_render_passes(color_raw, relief_raw)
    _compose_snapshot(raw, snapshot_path, present_classes, sun_azimuth=float(terrain_cmd["sun_azimuth"]), sun_elevation=float(terrain_cmd["sun_elevation"]))


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    snapshot = args.snapshot.resolve() if args.snapshot is not None else output_dir / "bosnia_landcover_2024.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    boundary_zip = _download(NATURAL_EARTH, cache_dir / "ne_10m_admin_0_countries.zip", force=bool(args.force))
    dem_path = _build_dem(boundary_zip, cache_dir, int(args.dem_zoom), force=bool(args.force))
    tile_paths = _download_landcover_tiles(cache_dir, force=bool(args.force))
    classes_path = _build_landcover_classes(boundary_zip, tile_paths, dem_path, cache_dir, force=bool(args.force))
    overlay_path, present = _build_overlay(classes_path, dem_path, _overlay_cache_path(cache_dir), force=bool(args.force))
    _render(snapshot, dem_path, overlay_path, present)
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
