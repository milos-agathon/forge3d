#!/usr/bin/env python3
"""Iberia population terrain render.

The population layer is always written onto the DEM grid, so both rasters share
the same CRS, transform, extent, width, and height before rendering. The
display texture follows the reference rayshader workflow: build the blue relief
color texture first, paint population yellow into it, then render that texture
on the DEM surface while preserving its palette.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

import sys

_examples_dir = Path(__file__).resolve().parents[1]
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
import bosnia_terrain_landcover_viewer as base_viewer

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "examples" / "out" / "iberia_builtup_cover"
CACHE_DIR = ROOT / "examples" / ".cache" / "iberia_builtup_cover"
HDR = ROOT / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"

COUNTRY_A3 = "IBE"
CACHE_COUNTRY_A3 = "IBEM"
COUNTRY_NAME = "Iberian Peninsula"
TARGET_CRS = "EPSG:3035"
NATURAL_EARTH = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_countries.zip"
)
IBERIA_ADM0_A3 = {"ESP", "PRT", "AND", "GIB"}
IBERIA_ADMIN = {"Spain", "Portugal", "Andorra", "Gibraltar"}
IBERIA_MAINLAND_BBOX = (-10.5, 35.0, 4.5, 44.5)

GHS_POP_PATH = Path("D:/ghsl-population/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif")

CAPTION_LINES = [
    "©2024 Milos Popovic (https://milospopovic.net)",
    "Data: Global Human Settlement Layer - population (R2023) at 3 arcsec",
]

VIEWER_SIZE = (1400, 1400)
SNAPSHOT_SIZE = (4200, 4200)
TERRAIN_SUPERSAMPLE = 3
VIEWER_TIMEOUT_SECONDS = 180.0
DEM_ZOOM = 7
PASS_SETTLE_SECONDS = 2.0
POPULATION_COLOR = np.array([255, 211, 1], dtype=np.uint8)
POPULATION_SHADE_FLOOR = 0.50
POPULATION_SHADE_GAIN = 0.50
POPULATION_SHADE_GAMMA = 0.85
TERRAIN_PALETTE = np.array(
    [
        [17, 40, 54],
        [31, 71, 98],
        [119, 157, 182],
    ],
    dtype=np.float32,
)
TERRAIN_SHADE_FLOOR = 0.62
TERRAIN_SHADE_GAIN = 0.55
CAMERA = {"ref": 1500.0, "radius": 4800.0, "zscale": 0.030, "pullback": 1.00}
TERRAIN = {
    "phi": 87.0,
    "theta": 0.0,
    "fov": 24.0,
    "sun_azimuth": 314.0,
    "sun_elevation": 28.0,
    "sun_intensity": 2.05,
    "ambient": 0.78,
    "shadow": 0.30,
    "background": [1.0, 1.0, 1.0],
}
PBR = {
    "enabled": True,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.65,
    "msaa": 8,
    "ibl_intensity": 0.12,
    "normal_strength": 1.05,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--dem-zoom", type=int, default=DEM_ZOOM)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _is_fresh(output_path: Path, input_paths: list[Path]) -> bool:
    return output_path.exists() and all(
        path.exists() and path.stat().st_mtime <= output_path.stat().st_mtime
        for path in input_paths
    )


def _download(url: str, dest: Path, *, force: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with urlopen(url, timeout=120) as response:
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


def _largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda geom: geom.area)
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for geom in geometry.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                polygons.extend(geom.geoms)
        if polygons:
            return max(polygons, key=lambda geom: geom.area)
    return geometry


def _country_geometry(boundary_zip: Path, target_crs: str):
    countries = gpd.read_file(boundary_zip)
    country = countries[
        countries.get("ADM0_A3", "").isin(IBERIA_ADM0_A3)
        | countries.get("ADMIN", "").isin(IBERIA_ADMIN)
    ]
    if country.empty:
        raise RuntimeError(f"Could not find {COUNTRY_NAME} in Natural Earth countries data")

    mainland = box(*IBERIA_MAINLAND_BBOX)
    country_4326 = country.to_crs("EPSG:4326")
    clipped = country_4326.geometry.intersection(mainland)
    clipped = clipped[~clipped.is_empty]
    union = clipped.union_all() if hasattr(clipped, "union_all") else clipped.unary_union
    peninsula = _largest_polygon(union)
    return gpd.GeoSeries([peninsula], crs="EPSG:4326").to_crs(target_crs).iloc[0]


def _build_dem(boundary_zip: Path, cache_dir: Path, zoom: int, *, force: bool) -> Path:
    original_country_a3 = base_viewer.COUNTRY_A3
    original_country_name = base_viewer.COUNTRY_NAME
    original_country_geometry = base_viewer._country_geometry
    try:
        base_viewer.COUNTRY_A3 = CACHE_COUNTRY_A3
        base_viewer.COUNTRY_NAME = COUNTRY_NAME
        base_viewer._country_geometry = _country_geometry
        return base_viewer._build_dem(boundary_zip, cache_dir, zoom, force=force)
    finally:
        base_viewer.COUNTRY_A3 = original_country_a3
        base_viewer.COUNTRY_NAME = original_country_name
        base_viewer._country_geometry = original_country_geometry


def _population_tif() -> Path:
    if not GHS_POP_PATH.is_file():
        raise FileNotFoundError(f"GHSL population GeoTIFF not found: {GHS_POP_PATH}")
    return GHS_POP_PATH


def _write_population_on_dem_grid(
    source_pop_path: Path,
    boundary_wgs84,
    dem_path: Path,
    output: Path,
    *,
    force: bool,
) -> Path:
    if _is_fresh(output, [source_pop_path, dem_path]) and not force:
        return output

    with rasterio.open(dem_path) as dem, rasterio.open(source_pop_path) as src:
        dem_data = dem.read(1, masked=True)
        destination = np.zeros((dem.height, dem.width), dtype=np.float32)
        src_nodata = src.nodata if src.nodata is not None else 0.0
        dst_nodata = 0.0

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dem.transform,
            dst_crs=dem.crs,
            dst_nodata=dst_nodata,
            init_dest_nodata=True,
            resampling=Resampling.nearest,
        )

        boundary_dem = gpd.GeoSeries([boundary_wgs84], crs="EPSG:4326").to_crs(dem.crs).iloc[0]
        outside_boundary = geometry_mask(
            [boundary_dem],
            out_shape=(dem.height, dem.width),
            transform=dem.transform,
            invert=False,
        )
        destination[np.asarray(dem_data.mask) | outside_boundary] = dst_nodata

        profile = dem.profile.copy()
        profile.update(
            driver="GTiff",
            width=dem.width,
            height=dem.height,
            count=1,
            crs=dem.crs,
            transform=dem.transform,
            dtype="float32",
            nodata=dst_nodata,
            compress="lzw",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(destination.astype(np.float32), 1)
    return output


def _build_population_data(boundary_wgs84, dem_path: Path, cache_dir: Path, *, force: bool) -> Path:
    source_pop = _population_tif()
    output = cache_dir / f"{CACHE_COUNTRY_A3.lower()}_population_on_dem_grid_v3.tif"
    return _write_population_on_dem_grid(source_pop, boundary_wgs84, dem_path, output, force=force)


def _terrain_base_rgb(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((dem.shape[0], dem.shape[1], 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb

    values = dem[valid].astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    norm = np.zeros_like(dem, dtype=np.float32)
    norm[valid] = np.clip((dem[valid].astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    scaled = norm * (len(TERRAIN_PALETTE) - 1)
    idx = np.clip(np.floor(scaled).astype(np.int16), 0, len(TERRAIN_PALETTE) - 2)
    frac = scaled - idx
    base = TERRAIN_PALETTE[idx] * (1.0 - frac[:, :, None]) + TERRAIN_PALETTE[idx + 1] * frac[:, :, None]
    shade = base_viewer._height_shade_from_dem(np.where(valid, dem, np.nan).astype(np.float32))
    shaded = base * np.clip(TERRAIN_SHADE_FLOOR + TERRAIN_SHADE_GAIN * shade[:, :, None], 0.0, 1.25)
    rgb[valid] = np.round(np.clip(shaded[valid], 0.0, 255.0)).astype(np.uint8)
    return rgb


def _population_relief_multiplier(shade: np.ndarray, valid: np.ndarray) -> np.ndarray:
    multiplier = np.ones_like(shade, dtype=np.float32)
    finite = valid & np.isfinite(shade)
    if not np.any(finite):
        return multiplier

    values = shade[finite].astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    span = high - low
    if span <= 1e-6:
        return multiplier

    normalized = np.zeros_like(shade, dtype=np.float32)
    normalized[finite] = np.clip((shade[finite].astype(np.float32) - low) / span, 0.0, 1.0)
    relief = POPULATION_SHADE_FLOOR + POPULATION_SHADE_GAIN * np.power(normalized, POPULATION_SHADE_GAMMA)
    multiplier[finite] = np.clip(relief[finite], 0.0, 1.0)
    return multiplier


def _population_rgba(population: np.ndarray, dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgba = np.zeros((population.shape[0], population.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = _terrain_base_rgb(dem, valid)
    active = valid & np.isfinite(population) & (population >= 0.1)
    shade = base_viewer._height_shade_from_dem(np.where(valid, dem, np.nan).astype(np.float32))
    multiplier = _population_relief_multiplier(shade, valid)
    population_rgb = POPULATION_COLOR.astype(np.float32) * multiplier[:, :, None]
    rgba[active, :3] = np.round(np.clip(population_rgb[active], 0.0, 255.0)).astype(np.uint8)
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _build_overlay(pop_path: Path, dem_path: Path, output: Path, *, force: bool) -> Path:
    if _is_fresh(output, [pop_path, dem_path]) and not force:
        return output
    with rasterio.open(pop_path) as pop_src, rasterio.open(dem_path) as dem_src:
        if pop_src.crs != dem_src.crs:
            raise ValueError(f"Population CRS {pop_src.crs} does not match DEM CRS {dem_src.crs}")
        if pop_src.width != dem_src.width or pop_src.height != dem_src.height:
            raise ValueError("Population raster dimensions do not match DEM dimensions")
        if not pop_src.transform.almost_equals(dem_src.transform):
            raise ValueError("Population raster transform does not match DEM transform")
        population = pop_src.read(1).astype(np.float32)
        dem = dem_src.read(1, masked=True)
        dem_data = dem.filled(np.nan).astype(np.float32)
        valid = ~np.asarray(dem.mask)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_population_rgba(population, dem_data, valid), mode="RGBA").save(output)
    return output


def _load_font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.ImageFont:
    if serif:
        names = ["timesbd.ttf", "Times New Roman Bold.ttf", "Georgia Bold.ttf"] if bold else ["times.ttf", "Times New Roman.ttf", "Georgia.ttf"]
    else:
        names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _subject_alpha(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    corners = np.asarray([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]], dtype=np.float32)
    bg = np.median(corners, axis=0)
    dist = np.abs(arr[:, :, :3].astype(np.float32) - bg[None, None, :]).max(axis=2)
    return np.clip((dist - 8.0) / 18.0, 0.0, 1.0)


def _crop_subject(image: Image.Image) -> Image.Image:
    alpha = _subject_alpha(image)
    mask = alpha > 0.03
    if not np.any(mask):
        return image.convert("RGBA")
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    arr[:, :, 3] = np.round(alpha * 255.0).astype(np.uint8)
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


def _compose_snapshot(raw_path: Path, output_path: Path) -> None:
    raw = Image.open(raw_path).convert("RGBA")
    subject = _crop_subject(raw)
    canvas = Image.new("RGBA", SNAPSHOT_SIZE, (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    target_w = round(canvas.width * 0.96)
    target_h = round(canvas.height * 0.78)
    scale = min(target_w / max(subject.width, 1), target_h / max(subject.height, 1))
    subject = subject.resize(
        (max(1, round(subject.width * scale)), max(1, round(subject.height * scale))),
        resample=Image.Resampling.LANCZOS,
    )

    alpha = subject.getchannel("A")
    shadow = Image.new("RGBA", subject.size, (42, 42, 42, 0))
    shadow.putalpha(alpha.point(lambda value: int(round(value * 0.24))))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(8, canvas.width // 120)))
    map_x = round(canvas.width * 0.015)
    map_y = round(canvas.height * 0.055)
    canvas.alpha_composite(shadow, dest=(map_x + canvas.width // 95, map_y + canvas.height // 90))
    canvas.alpha_composite(subject, dest=(map_x, map_y))

    title_x = round(canvas.width * 0.765)
    title_y = round(canvas.height * 0.735)
    pop_font = _load_font(round(canvas.width * 0.044), serif=True)
    iberia_font = _load_font(round(canvas.width * 0.070), bold=True, serif=True)
    caption_font = _load_font(round(canvas.width * 0.017), serif=True)
    draw.text((title_x, title_y), "Population", fill=(0, 0, 0), font=pop_font, anchor="ma")
    draw.text((title_x, title_y + round(canvas.height * 0.052)), "IBERIA", fill=(0, 0, 0), font=iberia_font, anchor="ma")

    caption_y = round(canvas.height * 0.902)
    line_gap = round(canvas.height * 0.026)
    for index, line in enumerate(CAPTION_LINES):
        draw.text((canvas.width // 2, caption_y + index * line_gap), line, fill=(0, 0, 0), font=caption_font, anchor="ma")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def _render(snapshot_path: Path, dem_path: Path, overlay_path: Path) -> None:
    hdr = HDR.resolve()
    with rasterio.open(dem_path) as dem_src:
        terrain_width = float(max(dem_src.width, dem_src.height))
    terrain_xy = terrain_width / CAMERA["ref"]
    radius = CAMERA["radius"] * terrain_xy * CAMERA["pullback"]
    zscale = CAMERA["zscale"] * math.sqrt(max(terrain_xy, 1e-6))
    terrain_cmd = {**TERRAIN, "radius": radius, "zscale": zscale}
    pbr_cmd = {**PBR}
    if hdr.is_file():
        pbr_cmd["hdr_path"] = str(hdr)

    snapshot_path = snapshot_path.resolve()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge3d_iberia_render_") as tmp:
        raw_path = Path(tmp) / "raw.png"
        with f3d.open_viewer_async(
            terrain_path=dem_path,
            width=VIEWER_SIZE[0],
            height=VIEWER_SIZE[1],
            timeout=VIEWER_TIMEOUT_SECONDS,
        ) as viewer:
            viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr_cmd})
            viewer.load_overlay(
                "iberia_population",
                overlay_path,
                extent=(0.0, 0.0, 1.0, 1.0),
                opacity=1.0,
                preserve_colors=True,
            )
            viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
            viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
            viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
            time.sleep(PASS_SETTLE_SECONDS)
            viewer.snapshot(
                raw_path,
                width=VIEWER_SIZE[0] * TERRAIN_SUPERSAMPLE,
                height=VIEWER_SIZE[1] * TERRAIN_SUPERSAMPLE,
            )
        _compose_snapshot(raw_path, snapshot_path)


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    snapshot = args.snapshot.resolve() if args.snapshot is not None else output_dir / "iberia_builtup_cover.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    boundary_zip = _download(NATURAL_EARTH, cache_dir / "ne_10m_admin_0_countries.zip", force=bool(args.force))
    boundary_wgs84 = _country_geometry(boundary_zip, "EPSG:4326")
    dem_path = _build_dem(boundary_zip, cache_dir, int(args.dem_zoom), force=bool(args.force))
    pop_path = _build_population_data(boundary_wgs84, dem_path, cache_dir, force=bool(args.force))
    overlay_path = _build_overlay(pop_path, dem_path, cache_dir / "iberia_population_overlay_v8.png", force=bool(args.force))
    _render(snapshot, dem_path, overlay_path)

    print(f"Success! Map saved to: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
