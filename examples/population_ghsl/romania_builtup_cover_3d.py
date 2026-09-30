#!/usr/bin/env python3
"""Romania built-up terrain render.

The built-up layer is always written onto the DEM grid, so both rasters share
the same CRS, transform, extent, width, and height before rendering. The
display texture follows the reference rayshader workflow: build the blue relief
color texture first, paint built-up areas into it, then render that texture on
the DEM surface while preserving its palette.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
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
OUT_DIR = ROOT / "examples" / "out" / "romania_builtup_cover"
CACHE_DIR = ROOT / "examples" / ".cache" / "romania_builtup_cover"
HDR = ROOT / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"
HDR_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/brown_photostudio_02_4k.hdr"

COUNTRY_A3 = "ROU"
REGION_SLUG = "romania"
CACHE_COUNTRY_A3 = COUNTRY_A3
COUNTRY_NAME = "Romania"
COUNTRY_TITLE = "ROMANIA"
TITLE_LINES = ["Built-up areas", COUNTRY_TITLE]
TARGET_CRS = "EPSG:3035"
NATURAL_EARTH = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_countries.zip"
)
BUILTUP_PATH = Path("D:/ghsl-builtup/GHS_BUILT_S_NRES_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")
POPULATION_FALLBACK_PATH = (
    ROOT / "examples" / ".cache" / "romania_builtup_cover" / "rou_population_on_dem_grid_v1.tif"
)
POPULATION_FALLBACK_THRESHOLD = 5.5

CAPTION_LINES = [
    "©2026 Milos Popovic (milosgis.com)",
    "Data: Global Human Settlement Layer - built-up surface (R2023) at 30 arcsec",
]

VIEWER_SIZE = (1400, 1400)
RENDER_SNAPSHOT_SIZE = (5000, 5000)
SNAPSHOT_SIZE = (4096, 4096)
TERRAIN_SUPERSAMPLE = 3
RENDER_MAX_SIZE = 4096
VIEWER_TIMEOUT_SECONDS = 180.0
DEM_ZOOM = 11
PASS_SETTLE_SECONDS = 2.0
BUILTUP_COLOR = np.array([213, 94, 0], dtype=np.uint8)
BUILTUP_HUE_TOLERANCE = 32.0
BUILTUP_MIN_SATURATION = 0.12
BUILTUP_MIN_VALUE = 0.10
BUILTUP_SHADE_FLOOR = 0.90
BUILTUP_SHADE_GAIN = 0.30
BUILTUP_SHADE_GAMMA = 0.85
BUILTUP_FINAL_VALUE_GAIN = 1.12
TERRAIN_PALETTE = np.array(
    [
        [17, 40, 54],
        [31, 71, 98],
        [119, 157, 182],
        [216, 233, 220],
    ],
    dtype=np.uint8,
)
TERRAIN_SHADOW_TINT = np.array([67, 92, 118], dtype=np.float32)
REFERENCE_GRADE_LUMA_QUANTILES = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
REFERENCE_GRADE_LUMA_TARGETS = [11.81, 20.01, 26.53, 48.89, 80.66, 100.83, 135.0, 170.0, 210.0]
REFERENCE_GRADE_SHADOW_RGB = np.array([17.0, 26.0, 44.0], dtype=np.float32)
REFERENCE_GRADE_MID_RGB = np.array([51.0, 78.0, 119.0], dtype=np.float32)
REFERENCE_GRADE_HIGH_RGB = np.array([109.0, 147.0, 185.0], dtype=np.float32)
REFERENCE_GRADE_MIN_BLEND = 0.25
REFERENCE_GRADE_MAX_BLEND = 0.82
REFERENCE_GRADE_CHANNEL_GAIN = np.array([1.00, 0.94, 0.96], dtype=np.float32)
EDGE_TERRAIN_RGB = np.array([42.0, 78.0, 108.0], dtype=np.float32)
EDGE_SOFTEN_MAX_ALPHA = 128.0
EDGE_SOFTEN_MAX_BLEND = 0.88
EDGE_SOFTEN_FILTER_SIZE = 15
EDGE_SOFTEN_LUMA_CAP = 145.0
TERRAIN_CAST_SHADOW = {
    "enabled": True,
    "sun_azimuth": 314.0,
    "sun_elevation": 18.0,
    "max_steps": 96,
    "zscale": 20.0,
    "darkness": 0.72,
}
OVERLAY_CACHE_NAME = "romania_builtup_overlay_v21.png"
CAMERA = {"ref": 1500.0, "radius": 4800.0, "zscale": 0.145, "pullback": 1.00}
TERRAIN = {
    "phi": 87.0,
    "theta": 0.0,
    "fov": 24.0,
    "sun_azimuth": 314.0,
    "sun_elevation": 24.0,
    "sun_intensity": 2.0,
    "ambient": 0.50,
    "shadow": 0.50,
    "background": [1.0, 1.0, 1.0],
}
PBR = {
    "enabled": True,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.25,
    "msaa": 8,
    "ibl_intensity": 1.3,
    "hdr_rotate_deg": 225.0,
    "normal_strength": 1.10,
    "height_ao": {
        "enabled": True,
        "directions": 10,
        "steps": 20,
        "max_distance": 220.0,
        "strength": 0.18,
        "resolution_scale": 0.72,
    },
    "sun_visibility": {
        "enabled": True,
        "mode": "soft",
        "samples": 2,
        "steps": 64,
        "max_distance": 3000.0,
        "softness": 0.80,
        "bias": 0.0028,
        "resolution_scale": 0.90,
    },
    "tonemap": {
        "operator": "aces",
        "white_point": 6.0,
        "white_balance_enabled": True,
        "temperature": 6500.0,
        "tint": 0.0,
    },
}
RELIEF_TERRAIN = {
    "sun_elevation": 18.0,
    "sun_intensity": 3.70,
    "ambient": 0.36,
    "shadow": 0.78,
}
RELIEF_PBR = {
    "exposure": 1.02,
    "ibl_intensity": 1.3,
    "normal_strength": 1.85,
    "height_ao": {
        "enabled": True,
        "directions": 12,
        "steps": 32,
        "max_distance": 260.0,
        "strength": 0.42,
        "resolution_scale": 0.80,
    },
    "sun_visibility": {
        "enabled": True,
        "mode": "hard",
        "samples": 1,
        "steps": 64,
        "max_distance": 3400.0,
        "softness": 0.0,
        "bias": 0.0028,
        "resolution_scale": 0.90,
    },
}
COMPOSITE = {
    "final_canvas_tone_enabled": True,
    "speckle_smoothing_enabled": True,
    "reference_canvas_grade_enabled": False,
    "terrain_luma_quantiles": [1.0, 10.0, 50.0, 75.0, 90.0, 99.0],
    "terrain_luma_targets": [13.3, 26.3, 82.4, 106.0, 175.0, 252.0],
    "terrain_saturation": 1.0,
    "relief_blur": 0.4,
    "relief_low": 4.0,
    "relief_high": 96.0,
    "relief_contrast": 0.80,
    "relief_gamma": 0.95,
    "value_floor": 0.50,
    "value_gain": 0.58,
    "highlight_start": 0.78,
    "highlight_gain": 0.00,
    "canvas_luma_quantiles": [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0],
    "canvas_luma_targets": [13.3, 20.1, 26.3, 49.6, 82.4, 106.0, 175.0, 252.0, 252.0],
    "neutral_luma_cap": 155.0,
    "terrain_saturation_cap": 0.52,
    "alpha_hole_filter_size": 9,
    "terrain_gap_filter_size": 55,
    "speckle_blur": 10.0,
    "speckle_blend": 0.30,
    "highlight_speckle_blur": 4.0,
    "highlight_speckle_min_luma": 250.0,
    "highlight_speckle_delta": 50.0,
}
LAYOUT = {
    "map_target_width": 1.0,
    "map_target_height": 0.72,
    "map_scale_x": 1.0,
    "map_scale_y": 1.0,
    "map_x": 0.008,
    "map_y": 0.148,
    "title_y": 0.024,
    "title_line_gap": 0.011,
    "title_map_gap": 0.026,
    "caption_y": 0.884,
    "caption_line_gap": 0.014,
    "caption_bottom_margin": 0.038,
    "map_caption_gap": 0.024,
}
RELIEF_FINAL_BLEND = 0.25
MAP_SHADOW = {
    "enabled": True,
    "offset": (-20, 18),
    "blur": 28.0,
    "opacity": 0.18,
    "color": (118, 145, 178),
}
OVERLAY_STYLE_VERSION = (
    "romania-builtup-overlay"
    f"-populationfallback{POPULATION_FALLBACK_THRESHOLD:.3f}"
    f"-builtup{int(BUILTUP_COLOR[0])}-{int(BUILTUP_COLOR[1])}-{int(BUILTUP_COLOR[2])}"
    f"-builtupshade{BUILTUP_SHADE_FLOOR:.2f}-{BUILTUP_SHADE_GAIN:.2f}-{BUILTUP_SHADE_GAMMA:.2f}"
    f"-terrainpal{';'.join('-'.join(str(int(channel)) for channel in color) for color in TERRAIN_PALETTE)}"
    f"-castshadow{int(TERRAIN_CAST_SHADOW['enabled'])}"
    f"-{TERRAIN_CAST_SHADOW['sun_azimuth']:.1f}-{TERRAIN_CAST_SHADOW['sun_elevation']:.1f}"
    f"-{TERRAIN_CAST_SHADOW['zscale']:.2f}-{TERRAIN_CAST_SHADOW['darkness']:.2f}"
)


class RegionPreset(NamedTuple):
    slug: str
    cache_a3: str
    title: str
    name: str
    country_a3: tuple[str, ...] = ()
    admin: tuple[str, ...] = ()
    continent: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    target_crs: str = "EPSG:3035"
    dem_zoom: int = DEM_ZOOM


REGION_PRESETS: dict[str, RegionPreset] = {
    "romania": RegionPreset(
        slug="romania",
        cache_a3="ROU",
        title="ROMANIA",
        name="Romania",
        country_a3=("ROU",),
        admin=("Romania",),
        dem_zoom=10,
    ),
    "italy": RegionPreset(
        slug="italy",
        cache_a3="ITA",
        title="ITALY",
        name="Italy",
        country_a3=("ITA",),
        admin=("Italy",),
        dem_zoom=8,
    ),
    "germany": RegionPreset(
        slug="germany",
        cache_a3="DEU",
        title="GERMANY",
        name="Germany",
        country_a3=("DEU",),
        admin=("Germany",),
        dem_zoom=8,
    ),
    "france": RegionPreset(
        slug="france",
        cache_a3="FRA",
        title="FRANCE",
        name="France",
        country_a3=("FRA",),
        admin=("France",),
        bbox=(-6.0, 41.0, 10.5, 52.0),
        dem_zoom=8,
    ),
    "uk_ireland": RegionPreset(
        slug="uk_ireland",
        cache_a3="GBRIRL",
        title="UK AND IRELAND",
        name="UK and Ireland",
        country_a3=("GBR", "IRL"),
        admin=("United Kingdom", "Ireland"),
        dem_zoom=8,
    ),
    "mainland_usa": RegionPreset(
        slug="mainland_usa",
        cache_a3="USAM",
        title="MAINLAND USA",
        name="mainland USA",
        country_a3=("USA",),
        admin=("United States of America",),
        bbox=(-125.0, 24.0, -66.0, 50.0),
        target_crs="EPSG:3857",
        dem_zoom=5,
    ),
    "japan": RegionPreset(
        slug="japan",
        cache_a3="JPN",
        title="JAPAN",
        name="Japan",
        country_a3=("JPN",),
        admin=("Japan",),
        target_crs="EPSG:3857",
        dem_zoom=7,
    ),
    "switzerland": RegionPreset(
        slug="switzerland",
        cache_a3="CHE",
        title="SWITZERLAND",
        name="Switzerland",
        country_a3=("CHE",),
        admin=("Switzerland",),
        dem_zoom=9,
    ),
    "poland": RegionPreset(
        slug="poland",
        cache_a3="POL",
        title="POLAND",
        name="Poland",
        country_a3=("POL",),
        admin=("Poland",),
        dem_zoom=8,
    ),
    "turkey": RegionPreset(
        slug="turkey",
        cache_a3="TUR",
        title="TURKEY",
        name="Turkey",
        country_a3=("TUR",),
        admin=("Turkey",),
        target_crs="EPSG:3857",
        dem_zoom=7,
    ),
    "africa": RegionPreset(
        slug="africa",
        cache_a3="AFR",
        title="AFRICA",
        name="Africa",
        continent="Africa",
        target_crs="EPSG:3857",
        dem_zoom=4,
    ),
    "brazil": RegionPreset(
        slug="brazil",
        cache_a3="BRA",
        title="BRAZIL",
        name="Brazil",
        country_a3=("BRA",),
        admin=("Brazil",),
        target_crs="EPSG:3857",
        dem_zoom=5,
    ),
    "argentina": RegionPreset(
        slug="argentina",
        cache_a3="ARG",
        title="ARGENTINA",
        name="Argentina",
        country_a3=("ARG",),
        admin=("Argentina",),
        target_crs="EPSG:3857",
        dem_zoom=5,
    ),
}

ACTIVE_REGION = REGION_PRESETS[REGION_SLUG]


def _configure_region(region: RegionPreset) -> None:
    global ACTIVE_REGION, REGION_SLUG, CACHE_COUNTRY_A3, COUNTRY_A3, COUNTRY_NAME
    global COUNTRY_TITLE, TITLE_LINES, TARGET_CRS, DEM_ZOOM, OUT_DIR, CACHE_DIR
    global OVERLAY_CACHE_NAME

    ACTIVE_REGION = region
    REGION_SLUG = region.slug
    CACHE_COUNTRY_A3 = region.cache_a3
    COUNTRY_A3 = "|".join(region.country_a3)
    if region.continent is not None:
        COUNTRY_NAME = f"continent:{region.continent}"
    else:
        COUNTRY_NAME = "|".join(region.admin) if region.admin else region.name
    COUNTRY_TITLE = region.title
    TITLE_LINES = ["Built-up areas", COUNTRY_TITLE]
    TARGET_CRS = region.target_crs
    DEM_ZOOM = int(region.dem_zoom)
    if region.slug == "romania":
        OUT_DIR = ROOT / "examples" / "out" / "romania_builtup_cover"
        CACHE_DIR = ROOT / "examples" / ".cache" / "romania_builtup_cover"
        OVERLAY_CACHE_NAME = "romania_builtup_overlay_v21.png"
    else:
        OUT_DIR = ROOT / "examples" / "out" / "builtup_cover_3d" / region.slug
        CACHE_DIR = ROOT / "examples" / ".cache" / "builtup_cover_3d" / region.slug
        OVERLAY_CACHE_NAME = f"{region.slug}_builtup_overlay_v21.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument(
        "--region",
        choices=sorted(REGION_PRESETS),
        default=REGION_SLUG,
        help="Region preset to render.",
    )
    parser.add_argument(
        "--batch-requested",
        action="store_true",
        help="Render Italy, Germany, France, UK/Ireland, mainland USA, Japan, Switzerland, Poland, Africa, Brazil, and Argentina.",
    )
    parser.add_argument("--dem-zoom", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _is_fresh(output_path: Path, input_paths: list[Path]) -> bool:
    return output_path.exists() and all(
        path.exists() and path.stat().st_mtime <= output_path.stat().st_mtime
        for path in input_paths
    )


def _overlay_style_path(overlay_path: Path) -> Path:
    return overlay_path.with_suffix(overlay_path.suffix + ".style")


def _overlay_is_current(overlay_path: Path) -> bool:
    if not overlay_path.exists():
        return False
    style_path = _overlay_style_path(overlay_path)
    if not style_path.exists():
        return False
    return style_path.read_text(encoding="utf-8").strip() == OVERLAY_STYLE_VERSION


def _overlay_matches_dem(overlay_path: Path, dem_path: Path) -> bool:
    if not overlay_path.exists() or not dem_path.exists():
        return False
    try:
        with Image.open(overlay_path) as overlay, rasterio.open(dem_path) as dem:
            return overlay.size == (dem.width, dem.height)
    except Exception:
        return False


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


def _ensure_hdri(*, force: bool = False) -> Path:
    return _download(HDR_URL, HDR, force=force)


def _largest_polygonal(geometry):
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for geom in geometry.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                polygons.extend(geom.geoms)
        if polygons:
            return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    return geometry


def _valid_geometry(geometry):
    if geometry.is_empty or geometry.is_valid:
        return geometry
    try:
        from shapely.validation import make_valid

        return make_valid(geometry)
    except Exception:
        return geometry.buffer(0)


def _country_geometry(boundary_zip: Path, target_crs: str):
    countries = gpd.read_file(boundary_zip)
    mask = np.zeros(len(countries), dtype=bool)
    if COUNTRY_A3 and "ADM0_A3" in countries:
        mask |= countries["ADM0_A3"].isin(tuple(COUNTRY_A3.split("|")))
    if COUNTRY_NAME and "ADMIN" in countries:
        mask |= countries["ADMIN"].isin(tuple(COUNTRY_NAME.split("|")))
    if COUNTRY_NAME.startswith("continent:"):
        continent = COUNTRY_NAME.split(":", 1)[1]
        if "CONTINENT" in countries:
            mask |= countries["CONTINENT"] == continent
    country = countries[mask]
    if country.empty:
        raise RuntimeError(f"Could not find {COUNTRY_NAME} in Natural Earth countries data")

    country_4326 = country.to_crs("EPSG:4326")
    if ACTIVE_REGION.bbox is not None:
        clip_box = box(*ACTIVE_REGION.bbox)
        clipped = country_4326.geometry.intersection(clip_box)
        clipped = clipped[~clipped.is_empty]
        if clipped.empty:
            raise RuntimeError(f"{ACTIVE_REGION.name} does not intersect its configured bbox")
        union = clipped.union_all() if hasattr(clipped, "union_all") else clipped.unary_union
    else:
        union = country_4326.geometry.union_all() if hasattr(country_4326.geometry, "union_all") else country_4326.geometry.unary_union
    union = _valid_geometry(_largest_polygonal(union))
    return gpd.GeoSeries([union], crs="EPSG:4326").to_crs(target_crs).iloc[0]


def _build_dem(boundary_zip: Path, cache_dir: Path, zoom: int, *, force: bool) -> Path:
    original_country_a3 = base_viewer.COUNTRY_A3
    original_country_name = base_viewer.COUNTRY_NAME
    original_target_crs = base_viewer.TARGET_CRS
    original_country_geometry = base_viewer._country_geometry
    try:
        base_viewer.COUNTRY_A3 = CACHE_COUNTRY_A3
        base_viewer.COUNTRY_NAME = ACTIVE_REGION.name
        base_viewer.TARGET_CRS = TARGET_CRS
        base_viewer._country_geometry = _country_geometry
        return base_viewer._build_dem(boundary_zip, cache_dir, zoom, force=force)
    finally:
        base_viewer.COUNTRY_A3 = original_country_a3
        base_viewer.COUNTRY_NAME = original_country_name
        base_viewer.TARGET_CRS = original_target_crs
        base_viewer._country_geometry = original_country_geometry


def _prepare_render_dem(dem_path: Path, cache_dir: Path, *, force: bool) -> Path:
    with rasterio.open(dem_path) as src:
        source_width = int(src.width)
        source_height = int(src.height)
        source_max = max(source_width, source_height)
        if source_max <= RENDER_MAX_SIZE:
            return dem_path

        scale = float(RENDER_MAX_SIZE) / float(source_max)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
        output = cache_dir / f"{dem_path.stem}_render_{target_width}x{target_height}.tif"
        if _is_fresh(output, [dem_path]) and not force:
            return output

        data = src.read(
            1,
            out_shape=(target_height, target_width),
            resampling=Resampling.bilinear,
            masked=True,
        )
        transform = src.transform * Affine.scale(
            source_width / target_width,
            source_height / target_height,
        )
        profile = src.profile.copy()
        profile.update(
            width=target_width,
            height=target_height,
            transform=transform,
            dtype="float32",
            compress="lzw",
        )
        nodata = src.nodata
        if nodata is None:
            nodata = -9999.0
            profile.update(nodata=nodata)

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data.filled(nodata).astype(np.float32), 1)
    return output


def _builtup_tif() -> Path:
    if BUILTUP_PATH.is_file():
        return BUILTUP_PATH
    if REGION_SLUG == "romania" and POPULATION_FALLBACK_PATH.is_file():
        return POPULATION_FALLBACK_PATH
    raise FileNotFoundError(f"GHSL built-up GeoTIFF not found: {BUILTUP_PATH}")


def _is_population_fallback_raster(path: Path) -> bool:
    resolved = path.resolve()
    if resolved == POPULATION_FALLBACK_PATH.resolve():
        return True
    return path.name.startswith(f"{CACHE_COUNTRY_A3.lower()}_population_on_")


def _builtup_active_mask(builtup: np.ndarray, valid: np.ndarray, source_path: Path) -> np.ndarray:
    threshold = (
        POPULATION_FALLBACK_THRESHOLD
        if _is_population_fallback_raster(source_path)
        else 0.1
    )
    return valid & np.isfinite(builtup) & (builtup >= threshold)


def _write_builtup_on_dem_grid(
    source_builtup_path: Path,
    boundary_wgs84,
    dem_path: Path,
    output: Path,
    *,
    force: bool,
) -> Path:
    if _is_fresh(output, [source_builtup_path, dem_path]) and not force:
        return output

    with rasterio.open(dem_path) as dem, rasterio.open(source_builtup_path) as src:
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


def _build_builtup_data(boundary_wgs84, dem_path: Path, cache_dir: Path, *, force: bool) -> Path:
    source_builtup = _builtup_tif()
    source_kind = "population" if _is_population_fallback_raster(source_builtup) else "builtup"
    output = cache_dir / f"{CACHE_COUNTRY_A3.lower()}_{source_kind}_on_{dem_path.stem}_v1.tif"
    return _write_builtup_on_dem_grid(source_builtup, boundary_wgs84, dem_path, output, force=force)


def _terrain_base_rgb(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((dem.shape[0], dem.shape[1], 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb

    values = dem[valid].astype(np.float32)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    normalized = np.zeros_like(dem, dtype=np.float32)
    normalized[valid] = np.clip((dem[valid].astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    scaled = normalized * float(len(TERRAIN_PALETTE) - 1)
    index = np.clip(np.floor(scaled).astype(np.int16), 0, len(TERRAIN_PALETTE) - 2)
    fraction = scaled - index
    palette = TERRAIN_PALETTE.astype(np.float32)
    shaded = palette[index] * (1.0 - fraction[:, :, None]) + palette[index + 1] * fraction[:, :, None]
    shaded *= _reference_cast_shadow(dem, valid)[:, :, None]
    rgb[valid] = np.round(np.clip(shaded[valid], 0.0, 255.0)).astype(np.uint8)
    return rgb


def _dem_cast_shadow_multiplier(
    dem: np.ndarray,
    valid: np.ndarray,
    *,
    sun_azimuth: float,
    sun_elevation: float,
    max_steps: int,
    zscale: float,
    darkness: float,
) -> np.ndarray:
    multiplier = np.ones_like(dem, dtype=np.float32)
    finite = valid & np.isfinite(dem)
    if not np.any(finite) or max_steps <= 0 or sun_elevation >= 89.0:
        return multiplier

    height = dem.astype(np.float32)
    rows, cols = height.shape
    row_grid, col_grid = np.indices(height.shape, dtype=np.float32)
    azimuth = math.radians(float(sun_azimuth))
    elevation_tan = math.tan(math.radians(max(float(sun_elevation), 0.1)))
    row_step = -math.cos(azimuth)
    col_step = math.sin(azimuth)
    shadow_zscale = max(float(zscale), 1e-6)
    occluded = np.zeros_like(finite, dtype=bool)

    for step in range(1, int(max_steps) + 1):
        sample_rows = np.rint(row_grid + row_step * step).astype(np.int32)
        sample_cols = np.rint(col_grid + col_step * step).astype(np.int32)
        in_bounds = (
            (sample_rows >= 0)
            & (sample_rows < rows)
            & (sample_cols >= 0)
            & (sample_cols < cols)
            & finite
            & ~occluded
        )
        if not np.any(in_bounds):
            continue

        sampled_height = np.zeros_like(height, dtype=np.float32)
        sampled_valid = np.zeros_like(finite, dtype=bool)
        source_rows = sample_rows[in_bounds]
        source_cols = sample_cols[in_bounds]
        sampled_height[in_bounds] = height[source_rows, source_cols]
        sampled_valid[in_bounds] = finite[source_rows, source_cols]
        ray_clearance = float(step) * elevation_tan * shadow_zscale
        blocked = in_bounds & sampled_valid & ((sampled_height - height) > ray_clearance)
        occluded |= blocked

    multiplier[occluded] = 1.0 - np.clip(float(darkness), 0.0, 0.95)
    return multiplier


def _reference_cast_shadow(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not TERRAIN_CAST_SHADOW["enabled"]:
        return np.ones_like(dem, dtype=np.float32)
    return _dem_cast_shadow_multiplier(
        dem,
        valid,
        sun_azimuth=float(TERRAIN_CAST_SHADOW["sun_azimuth"]),
        sun_elevation=float(TERRAIN_CAST_SHADOW["sun_elevation"]),
        max_steps=int(TERRAIN_CAST_SHADOW["max_steps"]),
        zscale=float(TERRAIN_CAST_SHADOW["zscale"]),
        darkness=float(TERRAIN_CAST_SHADOW["darkness"]),
    )


def _builtup_relief_multiplier(shade: np.ndarray, valid: np.ndarray) -> np.ndarray:
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
    relief = BUILTUP_SHADE_FLOOR + BUILTUP_SHADE_GAIN * np.power(normalized, BUILTUP_SHADE_GAMMA)
    multiplier[finite] = np.clip(relief[finite], 0.0, 1.0)
    return multiplier


def _builtup_rgba(
    builtup: np.ndarray,
    dem: np.ndarray,
    valid: np.ndarray,
    *,
    source_path: Path,
) -> np.ndarray:
    rgba = np.zeros((builtup.shape[0], builtup.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = _terrain_base_rgb(dem, valid)
    active = _builtup_active_mask(builtup, valid, source_path)
    shade = base_viewer._height_shade_from_dem(np.where(valid, dem, np.nan).astype(np.float32))
    multiplier = _builtup_relief_multiplier(shade, valid)
    builtup_rgb = BUILTUP_COLOR.astype(np.float32) * multiplier[:, :, None]
    rgba[active, :3] = np.round(np.clip(builtup_rgb[active], 0.0, 255.0)).astype(np.uint8)
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _build_overlay(builtup_path: Path, dem_path: Path, output: Path, *, force: bool) -> Path:
    if (
        _is_fresh(output, [builtup_path, dem_path])
        and _overlay_is_current(output)
        and _overlay_matches_dem(output, dem_path)
        and not force
    ):
        return output
    with rasterio.open(builtup_path) as builtup_src, rasterio.open(dem_path) as dem_src:
        if builtup_src.crs != dem_src.crs:
            raise ValueError(f"Built-up CRS {builtup_src.crs} does not match DEM CRS {dem_src.crs}")
        if builtup_src.width != dem_src.width or builtup_src.height != dem_src.height:
            raise ValueError("Built-up raster dimensions do not match DEM dimensions")
        if not builtup_src.transform.almost_equals(dem_src.transform):
            raise ValueError("Built-up raster transform does not match DEM transform")
        builtup = builtup_src.read(1).astype(np.float32)
        dem = dem_src.read(1, masked=True)
        dem_data = dem.filled(np.nan).astype(np.float32)
        valid = ~np.asarray(dem.mask)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        _builtup_rgba(builtup, dem_data, valid, source_path=builtup_path),
        mode="RGBA",
    ).save(output)
    _overlay_style_path(output).write_text(OVERLAY_STYLE_VERSION + "\n", encoding="utf-8")
    return output


def _load_font(
    size: int,
    *,
    bold: bool = False,
    serif: bool = False,
    display: bool = False,
) -> ImageFont.ImageFont:
    if display:
        names = (
            [
                "DIN Condensed Bold.ttf",
                "Arial Black.ttf",
                "Avenir Next Condensed.ttc",
                "DejaVuSansCondensed-Bold.ttf",
            ]
            if bold
            else [
                "Optima.ttc",
                "GillSans.ttc",
                "Georgia.ttf",
                "Arial.ttf",
                "DejaVuSans.ttf",
            ]
        )
    elif serif:
        names = (
            [
                "NewYork.ttf",
                "Iowan Old Style.ttc",
                "Baskerville.ttc",
                "Georgia Bold.ttf",
                "Times New Roman Bold.ttf",
            ]
            if bold
            else [
                "NewYork.ttf",
                "Iowan Old Style.ttc",
                "Baskerville.ttc",
                "Georgia.ttf",
                "Times New Roman.ttf",
            ]
        )
    else:
        names = (
            [
                "Avenir Next.ttc",
                "Optima.ttc",
                "GillSans.ttc",
                "Arial Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ]
            if bold
            else [
                "Optima.ttc",
                "GillSans.ttc",
                "Georgia.ttf",
                "Arial.ttf",
                "DejaVuSans.ttf",
            ]
        )
    font_dirs = (
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
    )
    for name in names:
        candidates = [Path(name)] if Path(name).is_absolute() else [Path(name)]
        candidates.extend(font_dir / name for font_dir in font_dirs)
        for candidate in candidates:
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _subject_alpha(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    corners = np.asarray([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]], dtype=np.float32)
    bg = np.median(corners, axis=0)
    dist = np.abs(arr[:, :, :3].astype(np.float32) - bg[None, None, :]).max(axis=2)
    return np.clip((dist - 8.0) / 18.0, 0.0, 1.0)


def _solidify_subject_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32).copy()
    if not np.any(alpha > 0.0):
        return alpha

    filter_size = max(3, int(COMPOSITE["alpha_hole_filter_size"]))
    if filter_size % 2 == 0:
        filter_size += 1
    alpha_u8 = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    expanded = np.asarray(
        Image.fromarray(alpha_u8, mode="L").filter(ImageFilter.MaxFilter(filter_size)),
        dtype=np.float32,
    ) / 255.0
    interior = (alpha > 0.0) & (expanded > 0.75)
    alpha[interior] = np.maximum(alpha[interior], expanded[interior])
    return np.clip(alpha, 0.0, 1.0)


def _expand_terrain_gap_eligibility(eligible: np.ndarray, terrain_seed: np.ndarray) -> np.ndarray:
    eligible = np.asarray(eligible, dtype=bool)
    terrain_seed = np.asarray(terrain_seed, dtype=bool)
    if terrain_seed.shape != eligible.shape:
        raise ValueError("terrain_seed must match eligible shape")
    if not np.any(terrain_seed):
        return eligible

    filter_size = max(3, int(COMPOSITE["terrain_gap_filter_size"]))
    if filter_size % 2 == 0:
        filter_size += 1
    seed_image = Image.fromarray(terrain_seed.astype(np.uint8) * 255, mode="L")
    expanded_seed = seed_image.filter(ImageFilter.MaxFilter(filter_size))
    eligible_image = Image.fromarray(eligible.astype(np.uint8) * 255, mode="L")
    subject_limit = eligible_image.filter(ImageFilter.MaxFilter(filter_size)).filter(
        ImageFilter.MinFilter(filter_size)
    )
    filled = (
        (np.asarray(expanded_seed, dtype=np.uint8) > 0)
        & (np.asarray(subject_limit, dtype=np.uint8) > 0)
    )
    return eligible | filled


def _rgb_to_hsv_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgbf = rgb.astype(np.float32) / 255.0
    maxc = rgbf.max(axis=2)
    minc = rgbf.min(axis=2)
    value = maxc
    saturation = np.where(maxc == 0.0, 0.0, (maxc - minc) / np.maximum(maxc, 1e-6))
    span = np.maximum(maxc - minc, 1e-6)
    rc = (maxc - rgbf[:, :, 0]) / span
    gc = (maxc - rgbf[:, :, 1]) / span
    bc = (maxc - rgbf[:, :, 2]) / span
    hue = np.zeros_like(maxc)
    hue = np.where(rgbf[:, :, 0] == maxc, bc - gc, hue)
    hue = np.where(rgbf[:, :, 1] == maxc, 2.0 + rc - bc, hue)
    hue = np.where(rgbf[:, :, 2] == maxc, 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    hue = np.where(maxc == minc, 0.0, hue)
    return hue * 360.0, saturation, value


def _hue_distance_degrees(hue: np.ndarray, target: float) -> np.ndarray:
    return np.abs((hue.astype(np.float32) - target + 180.0) % 360.0 - 180.0)


def _builtup_marker_mask(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    marker_hue, _, _ = _rgb_to_hsv_channels(BUILTUP_COLOR.reshape(1, 1, 3))
    target_hue = float(marker_hue[0, 0])
    return (
        (_hue_distance_degrees(hue, target_hue) <= BUILTUP_HUE_TOLERANCE)
        & (saturation > BUILTUP_MIN_SATURATION)
        & (value > BUILTUP_MIN_VALUE)
    )


def _hsv_values_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    h = (hue.astype(np.float32) % 360.0) / 60.0
    c = value.astype(np.float32) * saturation.astype(np.float32)
    x = c * (1.0 - np.abs((h % 2.0) - 1.0))
    m = value.astype(np.float32) - c

    zeros = np.zeros_like(h, dtype=np.float32)
    rp = np.select(
        [
            (h >= 0.0) & (h < 1.0),
            (h >= 1.0) & (h < 2.0),
            (h >= 2.0) & (h < 3.0),
            (h >= 3.0) & (h < 4.0),
            (h >= 4.0) & (h < 5.0),
            (h >= 5.0) & (h < 6.0),
        ],
        [c, x, zeros, zeros, x, c],
        default=zeros,
    )
    gp = np.select(
        [
            (h >= 0.0) & (h < 1.0),
            (h >= 1.0) & (h < 2.0),
            (h >= 2.0) & (h < 3.0),
            (h >= 3.0) & (h < 4.0),
            (h >= 4.0) & (h < 5.0),
            (h >= 5.0) & (h < 6.0),
        ],
        [x, c, c, x, zeros, zeros],
        default=zeros,
    )
    bp = np.select(
        [
            (h >= 0.0) & (h < 1.0),
            (h >= 1.0) & (h < 2.0),
            (h >= 2.0) & (h < 3.0),
            (h >= 3.0) & (h < 4.0),
            (h >= 4.0) & (h < 5.0),
            (h >= 5.0) & (h < 6.0),
        ],
        [zeros, zeros, x, c, c, x],
        default=zeros,
    )
    return np.stack([rp + m, gp + m, bp + m], axis=-1)


def _apply_reference_terrain_tone(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    subject_alpha = _subject_alpha(image)
    subject = subject_alpha > 0.03
    if arr[:, :, 3].max() > 0 and float(np.count_nonzero(subject)) / float(subject.size) < 0.25:
        subject = arr[:, :, 3] > 0
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    blue_relief = (hue >= 185.0) & (hue <= 215.0) & (saturation > 0.12)
    pale_relief = saturation <= 0.28
    terrain = (
        subject
        & (arr[:, :, 3] > 0)
        & (value > 0.015)
        & (value < 1.0)
        & ~builtup_yellow
        & (blue_relief | pale_relief)
    )
    if not np.any(terrain):
        return Image.fromarray(arr, mode="RGBA")

    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)
    quantiles = np.asarray(COMPOSITE["terrain_luma_quantiles"], dtype=np.float32)
    targets = np.asarray(COMPOSITE["terrain_luma_targets"], dtype=np.float32) / 255.0
    source = np.percentile(luminance[terrain], quantiles)
    source = np.maximum.accumulate(source)
    source[1:] = np.maximum(source[1:], source[:-1] + 1e-5)
    target_luminance = np.interp(luminance, source, targets)
    scale = np.divide(target_luminance, np.maximum(luminance, 1e-5))

    toned = rgb.copy()
    toned[terrain] = np.clip(toned[terrain] * scale[terrain, None], 0.0, 1.0)
    toned_luminance = (
        toned[:, :, 0] * 0.2126
        + toned[:, :, 1] * 0.7152
        + toned[:, :, 2] * 0.0722
    ).astype(np.float32)
    sat = float(COMPOSITE["terrain_saturation"])
    toned[terrain] = np.clip(
        toned_luminance[terrain, None]
        + (toned[terrain] - toned_luminance[terrain, None]) * sat,
        0.0,
        1.0,
    )
    neutral_shadow = terrain & (saturation < 0.36) & (value < 0.95)
    if np.any(neutral_shadow):
        target = TERRAIN_SHADOW_TINT / 255.0
        target_luminance = float(target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        blue_at_luminance = target[None, :] * (
            toned_luminance[neutral_shadow, None] / max(target_luminance, 1e-6)
        )
        weight = np.clip((0.36 - saturation[neutral_shadow]) / 0.36, 0.0, 1.0)
        toned[neutral_shadow] = np.clip(
            toned[neutral_shadow] * (1.0 - weight[:, None])
            + blue_at_luminance * weight[:, None],
            0.0,
            1.0,
        )
    arr[:, :, :3] = np.round(toned * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _crop_subject(image: Image.Image) -> Image.Image:
    alpha = _solidify_subject_alpha(_subject_alpha(image))
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


def _resize_subject_to_layout(
    subject: Image.Image,
    canvas_size: tuple[int, int],
    *,
    max_height: int | None = None,
) -> Image.Image:
    canvas_width, canvas_height = canvas_size
    target_w = round(canvas_width * LAYOUT["map_target_width"])
    target_h = round(canvas_height * LAYOUT["map_target_height"])
    if max_height is not None:
        target_h = min(target_h, max(1, int(max_height)))
    scale = min(target_w / max(subject.width, 1), target_h / max(subject.height, 1))
    width = max(
        1,
        round(subject.width * scale * float(LAYOUT.get("map_scale_x", 1.0))),
    )
    height = max(
        1,
        round(subject.height * scale * float(LAYOUT.get("map_scale_y", 1.0))),
    )
    return subject.resize((width, height), resample=Image.Resampling.LANCZOS)


def _apply_luminance_transfer(
    arr: np.ndarray,
    terrain: np.ndarray,
    quantiles: list[float],
    targets_255: list[float],
) -> np.ndarray:
    if not np.any(terrain):
        return arr
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)
    source = np.percentile(luminance[terrain], np.asarray(quantiles, dtype=np.float32))
    source = np.maximum.accumulate(source)
    source[1:] = np.maximum(source[1:], source[:-1] + 1e-5)
    targets = np.asarray(targets_255, dtype=np.float32) / 255.0
    target_luminance = np.interp(luminance, source, targets)
    scale = np.divide(target_luminance, np.maximum(luminance, 1e-5))

    toned = rgb.copy()
    toned[terrain] = np.clip(toned[terrain] * scale[terrain, None], 0.0, 1.0)
    arr[:, :, :3] = np.round(toned * 255.0).astype(np.uint8)
    return arr


def _tint_neutral_terrain_blue(arr: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Force grey/neutral terrain pixels to be blue-tinted."""
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])

    # Target: any pixel with low saturation in the mid-luminance range
    # This catches grey spots that should be blue
    neutral = (
        candidate
        & (saturation <= 0.28)  # Slightly less aggressive
        & (value > 0.10)
        & (value < 0.90)
    )
    if not np.any(neutral):
        return arr

    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)

    # Blue target color from reference analysis
    target = TERRAIN_SHADOW_TINT / 255.0
    target_luminance = float(target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))

    # Scale target to match source luminance
    neutral_luminance = np.minimum(
        luminance[neutral],
        float(COMPOSITE["neutral_luma_cap"]) / 255.0,
    )
    blue_at_luminance = target[None, :] * (neutral_luminance[:, None] / max(target_luminance, 1e-6))

    # Blend strongly toward blue (92% blue)
    blend = 0.92
    rgb[neutral] = np.clip(
        rgb[neutral] * (1 - blend) + blue_at_luminance * blend,
        0.0, 1.0
    )

    arr[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return arr


def _compress_terrain_saturation(arr: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    saturation_cap = float(COMPOSITE["terrain_saturation_cap"])
    terrain = (
        candidate
        & ~builtup_yellow
        & (hue >= 175.0)
        & (hue <= 225.0)
        & (saturation > saturation_cap)
        & (value > 0.02)
    )
    if not np.any(terrain):
        return arr

    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    rgb[terrain] = _hsv_values_to_rgb(
        hue[terrain],
        np.full(np.count_nonzero(terrain), saturation_cap, dtype=np.float32),
        value[terrain],
    )
    arr[:, :, :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr


def _suppress_terrain_highlight_speckles(arr: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    if not np.any(terrain):
        return arr

    rgb = arr[:, :, :3].astype(np.float32)
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)
    blurred = np.asarray(
        Image.fromarray(arr, mode="RGBA").filter(
            ImageFilter.GaussianBlur(radius=float(COMPOSITE["highlight_speckle_blur"]))
        ),
        dtype=np.float32,
    )
    blurred_luminance = (
        blurred[:, :, 0] * 0.2126
        + blurred[:, :, 1] * 0.7152
        + blurred[:, :, 2] * 0.0722
    ).astype(np.float32)
    speckle = (
        terrain
        & (luminance >= float(COMPOSITE["highlight_speckle_min_luma"]))
        & ((luminance - blurred_luminance) >= float(COMPOSITE["highlight_speckle_delta"]))
    )
    if not np.any(speckle):
        return arr

    arr[:, :, :3][speckle] = np.round(np.clip(blurred[:, :, :3][speckle], 0.0, 255.0)).astype(np.uint8)
    return arr


def _apply_final_canvas_terrain_tone(
    image: Image.Image,
    eligible_mask: np.ndarray | None = None,
) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    if eligible_mask is None:
        eligible = np.ones_like(value, dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != value.shape:
            raise ValueError("eligible_mask must match image height and width")
    height, _ = value.shape
    y = np.indices(value.shape, dtype=np.int32)[0]
    map_band = (y > round(height * 0.15)) & (y <= round(height * 0.885))
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    terrain_color = (
        ((hue >= 175.0) & (hue <= 225.0) & (saturation > 0.10))
        | ((saturation <= 0.32) & (value > 0.74))
    )
    non_background = np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8
    blue_terrain_seed = (
        map_band
        & eligible
        & non_background
        & (arr[:, :, 3] > 0)
        & ~builtup_yellow
        & (hue >= 175.0)
        & (hue <= 225.0)
        & (saturation > 0.20)
        & (value > 0.015)
    )
    eligible = _expand_terrain_gap_eligibility(eligible, blue_terrain_seed)
    candidate = (
        map_band
        & eligible
        & non_background
        & (arr[:, :, 3] > 0)
        & (value > 0.015)
        & (value < 1.0)
        & ~builtup_yellow
    )
    terrain = candidate & terrain_color
    arr = _apply_luminance_transfer(
        arr,
        terrain,
        COMPOSITE["canvas_luma_quantiles"],
        COMPOSITE["canvas_luma_targets"],
    )
    arr = _tint_neutral_terrain_blue(arr, candidate)
    arr = _compress_terrain_saturation(arr, candidate)
    return Image.fromarray(arr, mode="RGBA")


def _smooth_terrain_speckles(
    image: Image.Image,
    eligible_mask: np.ndarray,
) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    eligible = np.asarray(eligible_mask, dtype=bool)
    if eligible.shape != arr.shape[:2]:
        raise ValueError("eligible_mask must match image height and width")

    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    non_background = np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8
    terrain = (
        eligible
        & non_background
        & (arr[:, :, 3] > 0)
        & ~builtup_yellow
    )
    if not np.any(terrain):
        return Image.fromarray(arr, mode="RGBA")

    filtered = np.asarray(
        Image.fromarray(arr, mode="RGBA").filter(
            ImageFilter.GaussianBlur(radius=float(COMPOSITE["speckle_blur"]))
        ),
        dtype=np.float32,
    )
    blend = float(COMPOSITE["speckle_blend"])
    out = arr.astype(np.float32)
    out[terrain, :3] = out[terrain, :3] * (1.0 - blend) + filtered[terrain, :3] * blend
    out[builtup_yellow, :3] = arr[builtup_yellow, :3]
    smoothed = np.round(np.clip(out, 0.0, 255.0)).astype(np.uint8)
    smoothed = _tint_neutral_terrain_blue(smoothed, terrain)
    smoothed = _compress_terrain_saturation(smoothed, terrain)
    smoothed = _suppress_terrain_highlight_speckles(smoothed, terrain)
    smoothed[builtup_yellow, :3] = arr[builtup_yellow, :3]
    return Image.fromarray(smoothed, mode="RGBA")


def _apply_subject_terrain_tone(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    terrain_color = (
        ((hue >= 175.0) & (hue <= 225.0) & (saturation > 0.10))
        | ((saturation <= 0.32) & (value > 0.74))
    )
    non_background = np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8
    candidate = (
        non_background
        & (arr[:, :, 3] > 0)
        & (value > 0.015)
        & (value < 1.0)
        & ~builtup_yellow
    )
    terrain = candidate & terrain_color
    arr = _apply_luminance_transfer(
        arr,
        terrain,
        COMPOSITE["canvas_luma_quantiles"],
        COMPOSITE["canvas_luma_targets"],
    )
    arr = _tint_neutral_terrain_blue(arr, candidate)
    arr = _compress_terrain_saturation(arr, candidate)
    return Image.fromarray(arr, mode="RGBA")


def _tint_subject_neutral_shadows(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    neutral_shadow = (
        (arr[:, :, 3] > 0)
        & ~builtup_yellow
        & (saturation <= 0.36)
        & (value < 0.95)
        & (arr[:, :, 2].astype(np.int16) >= arr[:, :, 0].astype(np.int16) + 2)
    )
    if not np.any(neutral_shadow):
        return Image.fromarray(arr, mode="RGBA")

    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)
    target = TERRAIN_SHADOW_TINT / 255.0
    target_luminance = float(target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    blue_at_luminance = target[None, :] * (luminance[neutral_shadow, None] / max(target_luminance, 1e-6))
    weight = np.clip((0.36 - saturation[neutral_shadow]) / 0.36, 0.0, 1.0)
    rgb[neutral_shadow] = np.clip(
        rgb[neutral_shadow] * (1.0 - weight[:, None]) + blue_at_luminance * weight[:, None],
        0.0,
        1.0,
    )
    arr[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _combine_render_passes(color_raw_path: Path, relief_raw_path: Path) -> Image.Image:
    color_image = Image.open(color_raw_path).convert("RGBA")
    relief_image = Image.open(relief_raw_path).convert("RGBA")
    if color_image.size != relief_image.size:
        raise ValueError("Color and relief passes must have identical dimensions")

    base_image = _apply_reference_terrain_tone(color_image)
    color = np.asarray(base_image, dtype=np.float32) / 255.0
    relief = np.asarray(
        relief_image.filter(ImageFilter.GaussianBlur(radius=COMPOSITE["relief_blur"])),
        dtype=np.float32,
    ) / 255.0
    alpha = _solidify_subject_alpha(_subject_alpha(color_image))
    mask = alpha > 0.03
    if not np.any(mask):
        return base_image

    luminance = relief[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low = float(np.percentile(luminance[mask], COMPOSITE["relief_low"]))
    high = float(np.percentile(luminance[mask], COMPOSITE["relief_high"]))
    shade = np.clip((luminance - low) / max(high - low, 1e-6), 0.0, 1.0)
    shade = np.clip((shade - 0.5) * COMPOSITE["relief_contrast"] + 0.5, 0.0, 1.0)
    shade = np.power(shade, COMPOSITE["relief_gamma"], dtype=np.float32)

    value_scale = (
        COMPOSITE["value_floor"]
        + COMPOSITE["value_gain"] * shade
        + COMPOSITE["highlight_gain"] * np.maximum(shade - COMPOSITE["highlight_start"], 0.0)
    )
    combined = np.zeros_like(color)
    combined[:, :, :3] = np.clip(color[:, :, :3] * value_scale[:, :, None], 0.0, 1.0)
    combined[:, :, 3] = np.clip(alpha, 0.0, 1.0)
    combined[~mask, :3] = 0.0
    return _apply_reference_terrain_tone(
        Image.fromarray(np.round(combined * 255.0).astype(np.uint8), mode="RGBA")
    )


def _alpha_composite_at(base: Image.Image, overlay: Image.Image, dest: tuple[int, int]) -> None:
    x, y = int(dest[0]), int(dest[1])
    src_left = max(0, -x)
    src_top = max(0, -y)
    dst_x = max(0, x)
    dst_y = max(0, y)
    width = min(overlay.width - src_left, base.width - dst_x)
    height = min(overlay.height - src_top, base.height - dst_y)
    if width <= 0 or height <= 0:
        return
    cropped = overlay.crop((src_left, src_top, src_left + width, src_top + height))
    base.alpha_composite(cropped, dest=(dst_x, dst_y))


def _map_shadow_image(alpha: Image.Image) -> Image.Image:
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(MAP_SHADOW["blur"])))
    opacity = float(MAP_SHADOW["opacity"])
    shadow_alpha = shadow_alpha.point(lambda value: int(round(value * opacity)))
    shadow = Image.new("RGBA", alpha.size, tuple(MAP_SHADOW["color"]) + (0,))
    shadow.putalpha(shadow_alpha)
    return shadow


def _reference_chroma_rgb(luminance: np.ndarray) -> np.ndarray:
    t = np.clip((luminance.astype(np.float32) - 25.0) / 150.0, 0.0, 1.0)
    target = np.empty(luminance.shape + (3,), dtype=np.float32)
    low = t < 0.5
    low_t = np.divide(t, 0.5, out=np.zeros_like(t), where=low)
    high_t = np.clip((t - 0.5) / 0.5, 0.0, 1.0)
    target[low] = (
        REFERENCE_GRADE_SHADOW_RGB
        + (REFERENCE_GRADE_MID_RGB - REFERENCE_GRADE_SHADOW_RGB) * low_t[low, None]
    )
    target[~low] = (
        REFERENCE_GRADE_MID_RGB
        + (REFERENCE_GRADE_HIGH_RGB - REFERENCE_GRADE_MID_RGB) * high_t[~low, None]
    )
    target_luminance = (
        target[:, :, 0] * 0.2126
        + target[:, :, 1] * 0.7152
        + target[:, :, 2] * 0.0722
    ).astype(np.float32)
    return target * (luminance[:, :, None] / np.maximum(target_luminance[:, :, None], 1e-3))


def _apply_reference_canvas_grade(
    image: Image.Image,
    eligible_mask: np.ndarray,
) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    eligible = np.asarray(eligible_mask, dtype=bool)
    if eligible.shape != arr.shape[:2]:
        raise ValueError("eligible_mask must match image height and width")

    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = eligible & _builtup_marker_mask(hue, saturation, value)
    non_background = np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8
    terrain = eligible & non_background & ~builtup_yellow
    if not np.any(terrain):
        return Image.fromarray(arr, mode="RGBA")

    original_rgb = arr[:, :, :3].astype(np.float32)
    arr = _apply_luminance_transfer(
        arr,
        terrain,
        REFERENCE_GRADE_LUMA_QUANTILES,
        REFERENCE_GRADE_LUMA_TARGETS,
    )

    rgb = arr[:, :, :3].astype(np.float32)
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)
    target_rgb = _reference_chroma_rgb(luminance)
    blend = np.clip(
        (125.0 - luminance) / 110.0,
        REFERENCE_GRADE_MIN_BLEND,
        REFERENCE_GRADE_MAX_BLEND,
    )
    rgb[terrain] = (
        rgb[terrain] * (1.0 - blend[terrain, None])
        + target_rgb[terrain] * blend[terrain, None]
    )
    rgb[terrain] *= REFERENCE_GRADE_CHANNEL_GAIN[None, :]
    if np.any(builtup_yellow):
        rgb[builtup_yellow] = original_rgb[builtup_yellow] * 1.08 + np.array(
            [8.0, 22.0, 35.0],
            dtype=np.float32,
        )

    arr[:, :, :3] = np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _eliminate_grey_spots(image: Image.Image, subject_alpha: np.ndarray) -> Image.Image:
    """Final pass to eliminate any remaining grey spots by forcing them to blue."""
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.asarray(subject_alpha, dtype=np.float32)
    if alpha.shape != arr.shape[:2]:
        raise ValueError("subject_alpha must match image height and width")
    subject = alpha > 24.0

    rgb = arr[:, :, :3].astype(np.float32)

    # Calculate saturation and luminance
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = np.where(maxc > 0, (maxc - minc) / maxc, 0)
    luminance = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722

    # Exclude built-up markers.
    hue, _, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)

    # Exclude white/near-white background - this catches edge areas
    is_near_white = (minc > 235) | ((maxc > 245) & (minc > 220))

    # Proper blue terrain: must have blue hue AND reasonable saturation
    is_proper_blue = (hue >= 200.0) & (hue <= 235.0) & (saturation > 0.20)

    # Grey spots: anything not proper blue, not built-up, not near-white.
    grey_spots = (
        subject
        & ~builtup_yellow
        & ~is_near_white
        & ~is_proper_blue
        & (luminance > 6)
        & (luminance < 245)
    )

    if not np.any(grey_spots):
        return Image.fromarray(arr, mode="RGBA")

    # Target blue color from reference analysis: mean RGB [58, 88, 131]
    target = np.array([58.0, 88.0, 131.0], dtype=np.float32)
    target_luma = 0.2126 * target[0] + 0.7152 * target[1] + 0.0722 * target[2]

    # Scale target to match source luminance
    grey_luma = luminance[grey_spots]
    scale = grey_luma[:, None] / max(target_luma, 1e-5)
    blue_at_luma = target[None, :] * scale

    # Replace grey spots with blue completely
    rgb[grey_spots] = np.clip(blue_at_luma, 0.0, 255.0)

    arr[:, :, :3] = np.round(rgb).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _compress_terrain_highlights(image: Image.Image, subject_alpha: np.ndarray) -> Image.Image:
    """Simple luminance adjustment to better match reference."""
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.asarray(subject_alpha, dtype=np.float32)
    if alpha.shape != arr.shape[:2]:
        raise ValueError("subject_alpha must match image height and width")

    rgb = arr[:, :, :3].astype(np.float32)

    # Exclude built-up markers and background.
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    white_bg = (arr[:, :, :3].max(axis=2) > 240) & (arr[:, :, :3].min(axis=2) > 235)
    terrain = (alpha > 0) & ~builtup_yellow & ~white_bg

    if not np.any(terrain):
        return Image.fromarray(arr, mode="RGBA")

    # Simple contrast adjustment: multiply by 0.95 to slightly darken
    rgb_out = rgb.copy()
    rgb_out[terrain] = np.clip(rgb[terrain] * 0.95, 0.0, 255.0)

    arr[:, :, :3] = np.round(rgb_out).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _brighten_builtup_markers(image: Image.Image, subject_alpha: np.ndarray) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.asarray(subject_alpha, dtype=np.float32)
    if alpha.shape != arr.shape[:2]:
        raise ValueError("subject_alpha must match image height and width")

    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    markers = (alpha > 24.0) & _builtup_marker_mask(hue, saturation, value)
    if not np.any(markers):
        return Image.fromarray(arr, mode="RGBA")

    boosted_value = value.copy()
    boosted_value[markers] = np.clip(
        boosted_value[markers] * float(BUILTUP_FINAL_VALUE_GAIN),
        0.0,
        1.0,
    )
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    rgb[markers] = _hsv_values_to_rgb(
        hue[markers],
        saturation[markers],
        boosted_value[markers],
    )
    arr[:, :, :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _soften_subject_edge(image: Image.Image, subject_alpha: np.ndarray) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = np.asarray(subject_alpha, dtype=np.float32)
    if alpha.shape != arr.shape[:2]:
        raise ValueError("subject_alpha must match image height and width")

    alpha_u8 = np.round(np.clip(alpha, 0.0, 255.0)).astype(np.uint8)
    filter_size = max(3, int(EDGE_SOFTEN_FILTER_SIZE))
    if filter_size % 2 == 0:
        filter_size += 1
    eroded = np.asarray(
        Image.fromarray(np.where(alpha_u8 > 0, 255, 0).astype(np.uint8), mode="L").filter(
            ImageFilter.MinFilter(filter_size)
        ),
        dtype=np.float32,
    )
    edge = (alpha > 0.0) & (eroded < 255.0)
    if not np.any(edge):
        return Image.fromarray(arr, mode="RGBA")

    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    builtup_yellow = _builtup_marker_mask(hue, saturation, value)
    edge &= ~builtup_yellow
    if not np.any(edge):
        return Image.fromarray(arr, mode="RGBA")

    rgb = arr[:, :, :3].astype(np.float32)
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)
    capped_luminance = np.minimum(luminance, EDGE_SOFTEN_LUMA_CAP)
    target_rgb = _reference_chroma_rgb(capped_luminance)

    alpha_fraction = np.clip(alpha / 255.0, 0.0, 1.0)
    low_alpha = alpha <= EDGE_SOFTEN_MAX_ALPHA
    composite_target = 255.0 * (1.0 - alpha_fraction[:, :, None]) + EDGE_TERRAIN_RGB * alpha_fraction[:, :, None]
    target_rgb[low_alpha] = composite_target[low_alpha]

    edge_strength = np.clip((255.0 - eroded) / 255.0, 0.0, 1.0)
    low_alpha_strength = np.clip(
        (EDGE_SOFTEN_MAX_ALPHA - alpha) / max(EDGE_SOFTEN_MAX_ALPHA, 1.0),
        0.0,
        1.0,
    )
    blend = np.maximum(edge_strength * 0.58, low_alpha_strength * EDGE_SOFTEN_MAX_BLEND)
    rgb[edge] = rgb[edge] * (1.0 - blend[edge, None]) + target_rgb[edge] * blend[edge, None]
    arr[:, :, :3] = np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _compose_snapshot(raw, output_path: Path) -> None:
    raw = raw.convert("RGBA") if isinstance(raw, Image.Image) else Image.open(raw).convert("RGBA")
    subject = _crop_subject(raw)
    canvas = Image.new("RGBA", SNAPSHOT_SIZE, (249, 253, 254, 255))
    draw = ImageDraw.Draw(canvas)

    title_x = canvas.width // 2
    title_color = (20, 27, 39)
    title_font = _load_font(round(canvas.width * 0.041), serif=True)
    country_font = _load_font(round(canvas.width * 0.057), serif=True)
    caption_font = _load_font(round(canvas.width * 0.014), serif=True)

    title_top = round(canvas.height * float(LAYOUT["title_y"]))
    title_gap = max(
        round(canvas.height * float(LAYOUT["title_line_gap"])),
        round(getattr(title_font, "size", 16) * 0.18),
    )
    title_height = _text_size(draw, TITLE_LINES[0], title_font)[1]
    country_height = _text_size(draw, TITLE_LINES[1], country_font)[1]
    title_block_bottom = title_top + title_height + title_gap + country_height

    caption_gap = max(
        round(canvas.height * float(LAYOUT["caption_line_gap"])),
        round(getattr(caption_font, "size", 16) * 0.45),
    )
    caption_heights = [_text_size(draw, line, caption_font)[1] for line in CAPTION_LINES]
    caption_block_height = (
        sum(caption_heights) + max(0, len(caption_heights) - 1) * caption_gap
    )
    preferred_caption_top = round(canvas.height * float(LAYOUT["caption_y"]))
    caption_bottom_margin = round(canvas.height * float(LAYOUT["caption_bottom_margin"]))
    caption_top = min(
        preferred_caption_top,
        canvas.height - caption_bottom_margin - caption_block_height,
    )

    title_map_gap = round(canvas.height * float(LAYOUT["title_map_gap"]))
    map_caption_gap = round(canvas.height * float(LAYOUT["map_caption_gap"]))
    map_y = max(
        round(canvas.height * LAYOUT["map_y"]),
        title_block_bottom + title_map_gap,
    )
    caption_top = max(caption_top, map_y + map_caption_gap + 1)
    available_map_height = max(1, caption_top - map_caption_gap - map_y)

    subject = _resize_subject_to_layout(
        subject,
        canvas.size,
        max_height=available_map_height,
    )
    subject = _tint_subject_neutral_shadows(subject)

    alpha = subject.getchannel("A")
    map_x = min(
        max(0, round(canvas.width * LAYOUT["map_x"])),
        max(0, canvas.width - subject.width),
    )
    subject_mask = Image.new("L", canvas.size, 0)
    subject_mask.paste(alpha, (map_x, map_y))
    if MAP_SHADOW["enabled"]:
        offset_x, offset_y = MAP_SHADOW["offset"]
        _alpha_composite_at(
            canvas,
            _map_shadow_image(alpha),
            (map_x + int(offset_x), map_y + int(offset_y)),
        )
    canvas.alpha_composite(subject, dest=(map_x, map_y))

    draw.text(
        (title_x, title_top),
        TITLE_LINES[0],
        fill=title_color,
        font=title_font,
        anchor="mt",
    )
    country_y = title_top + title_height + title_gap
    draw.text(
        (title_x, country_y),
        TITLE_LINES[1],
        fill=title_color,
        font=country_font,
        anchor="mt",
    )

    caption_y = caption_top
    for line, line_height in zip(CAPTION_LINES, caption_heights):
        draw.text(
            (canvas.width // 2, caption_y),
            line,
            fill=title_color,
            font=caption_font,
            anchor="mt",
        )
        caption_y += line_height + caption_gap

    subject_mask_arr = np.asarray(subject_mask)
    solid_subject = subject_mask_arr > 24
    if COMPOSITE["final_canvas_tone_enabled"]:
        canvas = _apply_final_canvas_terrain_tone(canvas, eligible_mask=solid_subject)
    if COMPOSITE["speckle_smoothing_enabled"]:
        canvas = _smooth_terrain_speckles(canvas, eligible_mask=solid_subject)
    if COMPOSITE["reference_canvas_grade_enabled"]:
        canvas = _apply_reference_canvas_grade(canvas, eligible_mask=solid_subject)
    canvas = _soften_subject_edge(canvas, subject_mask_arr)
    canvas = _compress_terrain_highlights(canvas, subject_mask_arr)
    canvas = _eliminate_grey_spots(canvas, subject_mask_arr)
    canvas = _brighten_builtup_markers(canvas, subject_mask_arr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def _render(
    snapshot_path: Path,
    dem_path: Path,
    overlay_path: Path,
    *,
    force_hdri: bool = False,
) -> None:
    hdr = _ensure_hdri(force=force_hdri).resolve()
    with rasterio.open(dem_path) as dem_src:
        terrain_width = float(max(dem_src.width, dem_src.height))
    terrain_xy = terrain_width / CAMERA["ref"]
    radius = CAMERA["radius"] * terrain_xy * CAMERA["pullback"]
    zscale = CAMERA["zscale"] * math.sqrt(max(terrain_xy, 1e-6))
    terrain_cmd = {**TERRAIN, "radius": radius, "zscale": zscale}
    relief_terrain = {**terrain_cmd, **RELIEF_TERRAIN}
    pbr_cmd = {
        **PBR,
        "height_ao": dict(PBR["height_ao"]),
        "sun_visibility": dict(PBR["sun_visibility"]),
        "tonemap": dict(PBR["tonemap"]),
    }
    pbr_cmd["hdr_path"] = str(hdr)
    relief_pbr = {
        **pbr_cmd,
        **RELIEF_PBR,
        "height_ao": dict(RELIEF_PBR["height_ao"]),
        "sun_visibility": dict(RELIEF_PBR["sun_visibility"]),
        "tonemap": dict(pbr_cmd["tonemap"]),
    }

    snapshot_path = snapshot_path.resolve()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge3d_romania_render_") as tmp:
        color_raw = Path(tmp) / "color_raw.png"
        relief_raw = Path(tmp) / "relief_raw.png"
        with f3d.open_viewer_async(
            terrain_path=dem_path,
            width=VIEWER_SIZE[0],
            height=VIEWER_SIZE[1],
            timeout=VIEWER_TIMEOUT_SECONDS,
            ) as viewer:
            viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr_cmd})
            viewer.load_overlay(
                f"{REGION_SLUG}_builtup",
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
                color_raw,
                width=RENDER_SNAPSHOT_SIZE[0],
                height=RENDER_SNAPSHOT_SIZE[1],
            )
            viewer.send_ipc({"cmd": "set_terrain", **relief_terrain})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **relief_pbr})
            viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": False})
            time.sleep(PASS_SETTLE_SECONDS)
            viewer.snapshot(
                relief_raw,
                width=RENDER_SNAPSHOT_SIZE[0],
                height=RENDER_SNAPSHOT_SIZE[1],
            )
        color_only = _apply_reference_terrain_tone(Image.open(color_raw).convert("RGBA"))
        relief_raw_image = _combine_render_passes(color_raw, relief_raw)
        blend = float(RELIEF_FINAL_BLEND)
        if blend <= 0.0:
            _compose_snapshot(color_only, snapshot_path)
        elif blend >= 1.0:
            _compose_snapshot(relief_raw_image, snapshot_path)
        else:
            color_final = Path(tmp) / "color_final.png"
            relief_final = Path(tmp) / "relief_final.png"
            _compose_snapshot(color_only, color_final)
            _compose_snapshot(relief_raw_image, relief_final)
            blended = Image.blend(
                Image.open(color_final).convert("RGB"),
                Image.open(relief_final).convert("RGB"),
                max(0.0, min(1.0, blend)),
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            blended.save(snapshot_path)


REQUESTED_BATCH = (
    "italy",
    "germany",
    "france",
    "uk_ireland",
    "mainland_usa",
    "japan",
    "switzerland",
    "poland",
    "africa",
    "brazil",
    "argentina",
)


def _run_region(
    region: RegionPreset,
    *,
    output_dir_arg: Path | None,
    cache_dir_arg: Path | None,
    snapshot_arg: Path | None,
    dem_zoom_arg: int | None,
    force: bool,
) -> Path:
    _configure_region(region)
    output_dir = (output_dir_arg if output_dir_arg is not None else OUT_DIR).resolve()
    cache_dir = (cache_dir_arg if cache_dir_arg is not None else CACHE_DIR).resolve()
    snapshot = (
        snapshot_arg.resolve()
        if snapshot_arg is not None
        else output_dir / f"{region.slug}_builtup_cover.png"
    )
    dem_zoom = int(dem_zoom_arg if dem_zoom_arg is not None else region.dem_zoom)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    boundary_zip = _download(NATURAL_EARTH, cache_dir / "ne_10m_admin_0_countries.zip", force=force)
    boundary_wgs84 = _country_geometry(boundary_zip, "EPSG:4326")
    dem_path = _build_dem(boundary_zip, cache_dir, dem_zoom, force=force)
    render_dem_path = _prepare_render_dem(dem_path, cache_dir, force=force)
    builtup_path = _build_builtup_data(boundary_wgs84, render_dem_path, cache_dir, force=force)
    overlay_path = _build_overlay(builtup_path, render_dem_path, cache_dir / OVERLAY_CACHE_NAME, force=force)
    _render(snapshot, render_dem_path, overlay_path, force_hdri=force)
    return snapshot


def main() -> int:
    args = _parse_args()
    if args.batch_requested:
        if args.snapshot is not None:
            raise ValueError("--snapshot cannot be used with --batch-requested")
        snapshots = []
        for region_key in REQUESTED_BATCH:
            region = REGION_PRESETS[region_key]
            print(f"[{region.slug}] Rendering {region.name}...")
            snapshots.append(
                _run_region(
                    region,
                    output_dir_arg=args.output_dir,
                    cache_dir_arg=args.cache_dir,
                    snapshot_arg=None,
                    dem_zoom_arg=args.dem_zoom,
                    force=bool(args.force),
                )
            )
            print(f"[{region.slug}] Saved: {snapshots[-1]}")
        print("Success! Maps saved to:")
        for snapshot in snapshots:
            print(f"  {snapshot}")
        return 0

    snapshot = _run_region(
        REGION_PRESETS[args.region],
        output_dir_arg=args.output_dir,
        cache_dir_arg=args.cache_dir,
        snapshot_arg=args.snapshot,
        dem_zoom_arg=args.dem_zoom,
        force=bool(args.force),
    )

    print(f"Success! Map saved to: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
