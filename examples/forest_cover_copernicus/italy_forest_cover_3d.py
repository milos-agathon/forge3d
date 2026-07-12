#!/usr/bin/env python3
"""Italy forest cover terrain render.

The tree-cover fraction layer is clipped to Italy, written to a forest-native
render grid, and used as both the height surface and color texture. Low tree
cover is warm sand, high tree cover is dark green, and the forest fraction
alone drives the surface height.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

import sys

_examples_dir = Path(__file__).resolve().parents[1]
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "examples" / "out"
CACHE_DIR = ROOT / "examples" / ".cache" / "italy_forest_cover"
HDR = ROOT / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"
HDR_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/brown_photostudio_02_4k.hdr"

COUNTRY_A3 = "ITA"
REGION_SLUG = "italy"
CACHE_COUNTRY_A3 = COUNTRY_A3
COUNTRY_NAME = "Italy"
COUNTRY_TITLE = "ITALY"
TITLE_LINES = ["Tree cover", COUNTRY_TITLE]
TARGET_CRS = "EPSG:3035"
NATURAL_EARTH = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_countries.zip"
)
SHARED_BOUNDARY_ZIP = ROOT / "examples" / ".cache" / "italy_forest_cover" / "ne_10m_admin_0_countries.zip"
GISCO_COUNTRIES_2024_NAME = "CNTR_RG_01M_2024_4326.geojson"
GISCO_COUNTRIES_2024_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/"
    f"{GISCO_COUNTRIES_2024_NAME}"
)
FOREST_DATA_DIR = Path("D:/forest2019")
FOREST_SOURCE_GLOB = "*Tree-CoverFraction*.tif"
FOREST_BOUNDS_WGS84 = (6.0, 35.0, 19.0, 48.0)

CAPTION_LINES = [
    "©2026 Milos Popovic (https://milogis.com)",
    "Data: Copernicus Land Cover 2019 Tree-CoverFraction layer",
]


class ForestLayout(NamedTuple):
    render_snapshot_size: tuple[int, int]
    compose_canvas_size: tuple[int, int]
    snapshot_size: tuple[int, int]
    map_target_land_bbox: tuple[float, float, float, float]
    title_center_x: float
    title_forest_map_bbox: tuple[float, float, float, float]
    title_country_bbox: tuple[float, float, float, float]
    subtitle_y: float
    caption_y: float
    subtitle_font_scale: float
    caption_font_scale: float
    legend_box_bbox: tuple[float, float, float, float]
    legend_orientation: str
    legend_title_font_scale: float
    legend_label_font_scale: float
    legend_title_gap: float
    legend_label_gap: float


PORTRAIT_LAYOUT = ForestLayout(
    render_snapshot_size=(5500, 6096),
    compose_canvas_size=(5500, 6096),
    snapshot_size=(5500, 6096),
    map_target_land_bbox=(0.045, 0.405, 0.820, 0.880),
    title_center_x=0.842,
    title_forest_map_bbox=(0.736, 0.214, 0.950, 0.232),
    title_country_bbox=(0.650, 0.252, 0.986, 0.312),
    subtitle_y=0.352,
    caption_y=0.374,
    subtitle_font_scale=0.0100,
    caption_font_scale=0.0118,
    legend_box_bbox=(0.870, 0.610, 0.020, 0.145),
    legend_orientation="vertical",
    legend_title_font_scale=0.0155,
    legend_label_font_scale=0.0145,
    legend_title_gap=0.026,
    legend_label_gap=0.014,
)

RUSSIA_LANDSCAPE_LAYOUT = ForestLayout(
    render_snapshot_size=(5700, 4300),
    compose_canvas_size=(5700, 4300),
    snapshot_size=(5700, 4300),
    map_target_land_bbox=(0.020, 0.300, 0.980, 0.970),
    title_center_x=0.500,
    title_forest_map_bbox=(0.441, 0.022, 0.559, 0.038),
    title_country_bbox=(0.340, 0.064, 0.660, 0.106),
    subtitle_y=0.138,
    caption_y=0.154,
    subtitle_font_scale=0.0086,
    caption_font_scale=0.0096,
    legend_box_bbox=(0.350, 0.220, 0.300, 0.013),
    legend_orientation="horizontal",
    legend_title_font_scale=0.0106,
    legend_label_font_scale=0.0096,
    legend_title_gap=0.028,
    legend_label_gap=0.020,
)

REGION_LAYOUTS = {
    "russia": RUSSIA_LANDSCAPE_LAYOUT,
}

VIEWER_SIZE = (1400, 1400)
RENDER_SNAPSHOT_SIZE = PORTRAIT_LAYOUT.render_snapshot_size
COMPOSE_CANVAS_SIZE = PORTRAIT_LAYOUT.compose_canvas_size
SNAPSHOT_SIZE = PORTRAIT_LAYOUT.snapshot_size
TERRAIN_SUPERSAMPLE = 2
VIEWER_TIMEOUT_SECONDS = 180.0
FOREST_GRID_MAX_SIZE = 8000
FOREST_RENDER_GRID_MAX_SIZE = min(
    FOREST_GRID_MAX_SIZE * max(1, int(TERRAIN_SUPERSAMPLE)),
    max(RENDER_SNAPSHOT_SIZE),
)
PASS_SETTLE_SECONDS = 2.0
COMPOSE_POSTER = False
APPLY_RELIEF_TONE_COMPOSITE = True
FOREST_NODATA = -9999.0
FOREST_HEIGHT_SMOOTH_RADIUS = 5.0
FOREST_TEXTURE_SMOOTH_RADIUS = 5.0
SURFACE_FOREST_HEIGHT = 90.0
SURFACE_FOREST_GAMMA = 0.90
FOREST_COLOR_GAMMA = 0.88
MAP_BACKGROUND_RGB = (253, 250, 253)
MAP_TITLE_RGB = (70, 68, 71)
MAP_TARGET_LAND_BBOX = PORTRAIT_LAYOUT.map_target_land_bbox
TITLE_CENTER_X = PORTRAIT_LAYOUT.title_center_x
TITLE_FOREST_MAP_BBOX = PORTRAIT_LAYOUT.title_forest_map_bbox
TITLE_COUNTRY_BBOX = PORTRAIT_LAYOUT.title_country_bbox
SUBTITLE_Y = PORTRAIT_LAYOUT.subtitle_y
CAPTION_Y = PORTRAIT_LAYOUT.caption_y
SUBTITLE_FONT_SCALE = PORTRAIT_LAYOUT.subtitle_font_scale
CAPTION_FONT_SCALE = PORTRAIT_LAYOUT.caption_font_scale
LEGEND_BOX_BBOX = PORTRAIT_LAYOUT.legend_box_bbox
LEGEND_ORIENTATION = PORTRAIT_LAYOUT.legend_orientation
LEGEND_TITLE_FONT_SCALE = PORTRAIT_LAYOUT.legend_title_font_scale
LEGEND_LABEL_FONT_SCALE = PORTRAIT_LAYOUT.legend_label_font_scale
LEGEND_TITLE_GAP = PORTRAIT_LAYOUT.legend_title_gap
LEGEND_LABEL_GAP = PORTRAIT_LAYOUT.legend_label_gap
FINAL_MAP_BRIGHTNESS = 1.60
FINAL_MAP_WHITE_MIX = 0.05
PALETTE_CHROMA_RESTORE = 1.40
MAP_BLUE_GLOW_RGBA = (190, 202, 255, 16)
MAP_LAVENDER_GLOW_RGBA = (226, 222, 250, 14)
MAP_WARM_SHADOW_RGBA = (92, 84, 46, 14)
MAP_CONTACT_BLUE_RGBA = (166, 184, 255, 22)
MAP_BLUE_GLOW_BLUR = 46
MAP_LAVENDER_GLOW_BLUR = 74
MAP_WARM_SHADOW_BLUR = 26
MAP_CONTACT_BLUE_BLUR = 28
LEGEND_TITLE = "% of cover"
LEGEND_TICKS = [("100", 1.0), ("75", 0.75), ("50", 0.50), ("25", 0.25), ("0", 0.0)]
FOREST_PALETTE_VALUE_STOPS = [
    0.0,
    2.0,
    5.0,
    8.0,
    12.0,
    20.0,
    32.0,
    49.0,
    67.0,
    77.0,
    83.0,
    91.0,
    100.0,
]
FOREST_PALETTE_HEX = [
    "#E8E6ED",
    "#E0D9D2",
    "#D6C8B4",
    "#CDB392",
    "#C39B6F",
    "#B88550",
    "#A47338",
    "#8B692F",
    "#716229",
    "#5A5D25",
    "#3F5521",
    "#214B20",
    "#024026",
]
OVERLAY_CACHE_NAME = "italy_forest_overlay_v2.png"
CAMERA = {"ref": 1500.0, "radius": 4300.0, "zscale": 1.68, "pullback": 1.00}
BASE_TERRAIN = {
    "phi": 80.0,
    "theta": 0.0,
    "fov": 24.0,
    "sun_azimuth": 225.0,
    "sun_elevation": 30.0,
    "sun_intensity": 1.55,
    "ambient": 0.74,
    "shadow": 0.28,
    "background": [1.0, 1.0, 1.0],
}
TERRAIN = dict(BASE_TERRAIN)
REGION_TERRAIN_OVERRIDES = {
    "russia": {"phi": 90.0},
}
PBR = {
    "enabled": True,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.03,
    "msaa": 8,
    "ibl_intensity": 3.0,
    "hdr_rotate_deg": 225.0,
    "normal_strength": 0.48,
    "height_ao": {
        "enabled": True,
        "directions": 10,
        "steps": 16,
        "max_distance": 180.0,
        "strength": 0.12,
        "resolution_scale": 0.72,
    },
    "sun_visibility": {
        "enabled": True,
        "mode": "soft",
        "samples": 2,
        "steps": 64,
        "max_distance": 3000.0,
        "softness": 0.95,
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
    "sun_elevation": 24.0,
    "sun_intensity": 1.65,
    "ambient": 0.72,
    "shadow": 0.25,
}
RELIEF_PBR = {
    "exposure": 1.02,
    "ibl_intensity": 3.0,
    "normal_strength": 0.55,
    "height_ao": {
        "enabled": True,
        "directions": 10,
        "steps": 20,
        "max_distance": 210.0,
        "strength": 0.14,
        "resolution_scale": 0.80,
    },
    "sun_visibility": {
        "enabled": True,
        "mode": "soft",
        "samples": 2,
        "steps": 64,
        "max_distance": 3400.0,
        "softness": 0.90,
        "bias": 0.0028,
        "resolution_scale": 0.90,
    },
}
COMPOSITE = {
    "terrain_luma_quantiles": [1.0, 10.0, 50.0, 75.0, 90.0, 99.0],
    "terrain_luma_targets": [42.0, 68.0, 112.0, 132.0, 150.0, 182.0],
    "terrain_saturation": 0.58,
    "relief_blur": 3.0,
    "relief_low": 8.0,
    "relief_high": 94.0,
    "relief_contrast": 0.68,
    "relief_gamma": 1.16,
    "value_floor": 0.88,
    "value_gain": 0.18,
    "highlight_start": 0.84,
    "highlight_gain": 0.015,
    "directional_shadow_strength": 0.50,
    "directional_highlight_strength": 0.24,
    "forest_luma_quantiles": [1.0, 5.0, 15.0, 35.0, 55.0, 75.0, 92.0, 99.0],
    "forest_luma_targets": [42.0, 52.0, 66.0, 82.0, 100.0, 122.0, 146.0, 170.0],
    "canvas_luma_quantiles": [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0],
    "canvas_luma_targets": [40.0, 52.0, 62.0, 78.0, 96.0, 116.0, 142.0, 166.0, 210.0],
}
OVERLAY_STYLE_VERSION = (
    "italy-forest-overlay-scico-fes-green"
    f"-palette{'-'.join(color.lstrip('#') for color in FOREST_PALETTE_HEX)}"
    f"-stops{'-'.join(f'{stop:g}' for stop in FOREST_PALETTE_VALUE_STOPS)}"
    f"-colorgamma{FOREST_COLOR_GAMMA:.2f}"
    f"-textureblur{FOREST_TEXTURE_SMOOTH_RADIUS:.2f}"
    f"-grid{FOREST_RENDER_GRID_MAX_SIZE}px-ss{TERRAIN_SUPERSAMPLE}"
    "-legend-from-forest-palette"
)

RUSSIA_TARGET_CRS = "+proj=laea +lat_0=60 +lon_0=100 +datum=WGS84 +units=m +no_defs"
AFRICA_TARGET_CRS = "+proj=laea +lat_0=0 +lon_0=20 +datum=WGS84 +units=m +no_defs"
TURKEY_TARGET_CRS = "+proj=laea +lat_0=39 +lon_0=35 +datum=WGS84 +units=m +no_defs"


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
    boundary_source: str = "natural_earth"


REGION_PRESETS: dict[str, RegionPreset] = {
    "italy": RegionPreset(
        slug="italy",
        cache_a3="ITA",
        title="ITALY",
        name="Italy",
        country_a3=("ITA",),
        admin=("Italy",),
        bbox=FOREST_BOUNDS_WGS84,
    ),
    "france": RegionPreset(
        slug="france",
        cache_a3="FRA",
        title="FRANCE",
        name="France",
        country_a3=("FRA",),
        admin=("France",),
        bbox=(-6.0, 41.0, 10.5, 52.0),
    ),
    "germany": RegionPreset(
        slug="germany",
        cache_a3="DEU",
        title="GERMANY",
        name="Germany",
        country_a3=("DEU",),
        admin=("Germany",),
    ),
    "poland": RegionPreset(
        slug="poland",
        cache_a3="POL",
        title="POLAND",
        name="Poland",
        country_a3=("POL",),
        admin=("Poland",),
    ),
    "turkey": RegionPreset(
        slug="turkey",
        cache_a3="TUR",
        title="TURKEY",
        name="Turkey",
        country_a3=("TUR",),
        admin=("Turkey",),
        target_crs=TURKEY_TARGET_CRS,
    ),
    "southeast_europe": RegionPreset(
        slug="southeast_europe",
        cache_a3="SEE",
        title="SOUTHEAST EUROPE",
        name="Southeast Europe",
        country_a3=(
            "ALB",
            "BIH",
            "BGR",
            "HRV",
            "GRC",
            "KOS",
            "MDA",
            "MKD",
            "MNE",
            "ROU",
            "SRB",
            "SVN",
        ),
        admin=(
            "Albania",
            "Bosnia and Herzegovina",
            "Bulgaria",
            "Croatia",
            "Greece",
            "Kosovo",
            "Moldova",
            "North Macedonia",
            "Montenegro",
            "Romania",
            "Serbia",
            "Slovenia",
        ),
        bbox=(12.0, 34.0, 31.0, 48.5),
    ),
    "mainland_usa": RegionPreset(
        slug="mainland_usa",
        cache_a3="USAM",
        title="MAINLAND USA",
        name="mainland USA",
        country_a3=("USA",),
        admin=("United States of America",),
        bbox=(-125.0, 24.0, -66.0, 50.0),
        target_crs="EPSG:5070",
    ),
    "iberia": RegionPreset(
        slug="iberia",
        cache_a3="IBR",
        title="IBERIA",
        name="Iberia",
        country_a3=("ESP", "PRT", "AND"),
        admin=("Spain", "Portugal", "Andorra"),
        bbox=(-10.5, 35.5, 4.5, 44.5),
    ),
    "russia": RegionPreset(
        slug="russia",
        cache_a3="RUS",
        title="RUSSIA",
        name="Russia",
        country_a3=("RUS",),
        admin=("Russia",),
        target_crs=RUSSIA_TARGET_CRS,
        boundary_source="gisco",
    ),
    "africa": RegionPreset(
        slug="africa",
        cache_a3="AFR",
        title="AFRICA",
        name="Africa",
        continent="Africa",
        target_crs=AFRICA_TARGET_CRS,
    ),
    "europe": RegionPreset(
        slug="europe",
        cache_a3="EUR",
        title="EUROPE",
        name="Europe",
        continent="Europe",
        bbox=(-25.0, 34.0, 45.0, 72.0),
    ),
}

ACTIVE_REGION = REGION_PRESETS[REGION_SLUG]


def _boundary_cache_suffix(region: RegionPreset | None = None) -> str:
    selected = ACTIVE_REGION if region is None else region
    return "_gisco2024" if selected.boundary_source == "gisco" else ""


def _apply_layout(layout: ForestLayout) -> None:
    global RENDER_SNAPSHOT_SIZE, COMPOSE_CANVAS_SIZE, SNAPSHOT_SIZE
    global MAP_TARGET_LAND_BBOX, TITLE_CENTER_X, TITLE_FOREST_MAP_BBOX, TITLE_COUNTRY_BBOX
    global SUBTITLE_Y, CAPTION_Y, SUBTITLE_FONT_SCALE, CAPTION_FONT_SCALE
    global LEGEND_BOX_BBOX, LEGEND_ORIENTATION
    global LEGEND_TITLE_FONT_SCALE, LEGEND_LABEL_FONT_SCALE, LEGEND_TITLE_GAP, LEGEND_LABEL_GAP

    RENDER_SNAPSHOT_SIZE = layout.render_snapshot_size
    COMPOSE_CANVAS_SIZE = layout.compose_canvas_size
    SNAPSHOT_SIZE = layout.snapshot_size
    MAP_TARGET_LAND_BBOX = layout.map_target_land_bbox
    TITLE_CENTER_X = layout.title_center_x
    TITLE_FOREST_MAP_BBOX = layout.title_forest_map_bbox
    TITLE_COUNTRY_BBOX = layout.title_country_bbox
    SUBTITLE_Y = layout.subtitle_y
    CAPTION_Y = layout.caption_y
    SUBTITLE_FONT_SCALE = layout.subtitle_font_scale
    CAPTION_FONT_SCALE = layout.caption_font_scale
    LEGEND_BOX_BBOX = layout.legend_box_bbox
    LEGEND_ORIENTATION = layout.legend_orientation
    LEGEND_TITLE_FONT_SCALE = layout.legend_title_font_scale
    LEGEND_LABEL_FONT_SCALE = layout.legend_label_font_scale
    LEGEND_TITLE_GAP = layout.legend_title_gap
    LEGEND_LABEL_GAP = layout.legend_label_gap


def _configure_region(region: RegionPreset) -> None:
    global ACTIVE_REGION, REGION_SLUG, CACHE_COUNTRY_A3, COUNTRY_A3, COUNTRY_NAME
    global COUNTRY_TITLE, TITLE_LINES, TARGET_CRS, OUT_DIR, CACHE_DIR
    global OVERLAY_CACHE_NAME, FOREST_BOUNDS_WGS84, TERRAIN

    ACTIVE_REGION = region
    REGION_SLUG = region.slug
    CACHE_COUNTRY_A3 = region.cache_a3
    COUNTRY_A3 = "|".join(region.country_a3)
    if region.continent is not None:
        COUNTRY_NAME = f"continent:{region.continent}"
    else:
        COUNTRY_NAME = "|".join(region.admin) if region.admin else region.name
    COUNTRY_TITLE = region.title
    TITLE_LINES = ["Tree cover", COUNTRY_TITLE]
    TARGET_CRS = region.target_crs
    _apply_layout(REGION_LAYOUTS.get(region.slug, PORTRAIT_LAYOUT))
    TERRAIN = {**BASE_TERRAIN, **REGION_TERRAIN_OVERRIDES.get(region.slug, {})}
    FOREST_BOUNDS_WGS84 = region.bbox if region.bbox is not None else (-180.0, -90.0, 180.0, 90.0)
    if region.slug == "italy":
        OUT_DIR = ROOT / "examples" / "out"
        CACHE_DIR = ROOT / "examples" / ".cache" / "italy_forest_cover"
        OVERLAY_CACHE_NAME = "italy_forest_overlay_v2.png"
    else:
        OUT_DIR = ROOT / "examples" / "out"
        CACHE_DIR = ROOT / "examples" / ".cache" / "forest_cover_3d" / region.slug
        OVERLAY_CACHE_NAME = f"{region.slug}_forest_overlay{_boundary_cache_suffix(region)}_v2.png"


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
        help="Render all forest-cover maps requested for this batch.",
    )
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


def _ensure_boundary_file(cache_dir: Path, *, force: bool) -> Path:
    if ACTIVE_REGION.boundary_source == "gisco":
        target = cache_dir / GISCO_COUNTRIES_2024_NAME
        return _download(GISCO_COUNTRIES_2024_URL, target, force=force)

    target = cache_dir / "ne_10m_admin_0_countries.zip"
    if target.exists() and not force:
        return target
    if SHARED_BOUNDARY_ZIP.exists() and not force:
        return SHARED_BOUNDARY_ZIP
    return _download(NATURAL_EARTH, target, force=force)


def _ensure_boundary_zip(cache_dir: Path, *, force: bool) -> Path:
    return _ensure_boundary_file(cache_dir, force=force)


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
        return _largest_polygonal(geometry)
    try:
        from shapely.validation import make_valid

        return _largest_polygonal(make_valid(geometry))
    except Exception:
        return _largest_polygonal(geometry.buffer(0))


def _project_geometry(geometry, target_crs: str):
    projected = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(target_crs).iloc[0]
    return _valid_geometry(_largest_polygonal(projected))


def _country_geometry(boundary_zip: Path, target_crs: str):
    countries = gpd.read_file(boundary_zip)
    mask = np.zeros(len(countries), dtype=bool)
    country_codes = tuple(COUNTRY_A3.split("|")) if COUNTRY_A3 else ()
    for code_column in ("ADM0_A3", "ISO3_CODE", "COUNTRY_URI"):
        if country_codes and code_column in countries:
            mask |= countries[code_column].isin(country_codes)
    if COUNTRY_NAME and not COUNTRY_NAME.startswith("continent:"):
        country_names = tuple(COUNTRY_NAME.split("|"))
        for name_column in ("ADMIN", "NAME_ENGL", "CNTR_NAME", "NAME_FREN", "NAME_GERM"):
            if name_column in countries:
                mask |= countries[name_column].isin(country_names)
    if COUNTRY_NAME.startswith("continent:") and "CONTINENT" in countries:
        continent = COUNTRY_NAME.split(":", 1)[1]
        mask |= countries["CONTINENT"] == continent
    country = countries[mask]
    if country.empty:
        raise RuntimeError(f"Could not find {COUNTRY_NAME} in {ACTIVE_REGION.boundary_source} countries data")

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
    if target_crs == "EPSG:4326":
        return union
    return _project_geometry(union, target_crs)


def _tile_bounds_from_name(path: Path) -> tuple[float, float, float, float] | None:
    match = re.match(r"([EW])(\d{3})([NS])(\d{2})_", path.name)
    if match is None:
        return None
    lon_sign = -1.0 if match.group(1) == "W" else 1.0
    lat_sign = -1.0 if match.group(3) == "S" else 1.0
    left = lon_sign * float(match.group(2))
    top = lat_sign * float(match.group(4))
    right = left + 20.0
    bottom = top - 20.0
    return (left, bottom, right, top)


def _bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _tree_cover_tifs(boundary_wgs84=None) -> list[Path]:
    candidates = [
        path
        for path in FOREST_DATA_DIR.rglob(FOREST_SOURCE_GLOB)
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    candidates.sort(key=lambda path: (len(path.relative_to(FOREST_DATA_DIR).parts), path.name, str(path)))
    selected_by_name: dict[str, Path] = {}
    boundary = _valid_geometry(_largest_polygonal(boundary_wgs84)) if boundary_wgs84 is not None else None
    for path in candidates:
        bounds = _tile_bounds_from_name(path)
        if bounds is None or not _bounds_intersect(bounds, FOREST_BOUNDS_WGS84):
            continue
        if boundary is not None and not boundary.intersects(box(*bounds)):
            continue
        selected_by_name.setdefault(path.name, path)
    selected = sorted(selected_by_name.values(), key=lambda path: path.name)
    if not selected:
        raise FileNotFoundError(
            f"No {FOREST_SOURCE_GLOB} tiles covering {ACTIVE_REGION.name} found under {FOREST_DATA_DIR}"
        )
    return selected


def _forest_grid_max_size() -> int:
    return max(1, int(FOREST_RENDER_GRID_MAX_SIZE))


def _forest_render_grid(boundary_wgs84) -> tuple[object, int, int, object]:
    boundary_target = _project_geometry(boundary_wgs84, TARGET_CRS)
    minx, miny, maxx, maxy = boundary_target.bounds
    span_x = float(maxx - minx)
    span_y = float(maxy - miny)
    if span_x <= 0.0 or span_y <= 0.0:
        raise ValueError("Italy boundary has invalid projected bounds")

    pad = max(span_x, span_y) * 0.01
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad
    span_x = float(maxx - minx)
    span_y = float(maxy - miny)
    pixel_size = max(span_x, span_y) / float(_forest_grid_max_size())
    width = max(1, int(math.ceil(span_x / pixel_size)))
    height = max(1, int(math.ceil(span_y / pixel_size)))
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    return boundary_target, width, height, transform


def _write_tree_cover_grid(
    source_forest_paths: list[Path],
    boundary_wgs84,
    output: Path,
    *,
    force: bool,
) -> Path:
    if _is_fresh(output, source_forest_paths) and not force:
        return output

    boundary_target, width, height, transform = _forest_render_grid(boundary_wgs84)
    destination = np.full((height, width), FOREST_NODATA, dtype=np.float32)

    for source_forest_path in source_forest_paths:
        with rasterio.open(source_forest_path) as src:
            if src.crs is None:
                raise ValueError(f"Forest raster has no CRS: {source_forest_path}")
            tile = np.full((height, width), FOREST_NODATA, dtype=np.float32)
            src_nodata = src.nodata if src.nodata is not None else 255.0
            reproject(
                source=rasterio.band(src, 1),
                destination=tile,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=FOREST_NODATA,
                init_dest_nodata=True,
                resampling=Resampling.bilinear,
            )
            valid_tile = (
                np.isfinite(tile)
                & (tile >= 0.0)
                & (tile <= 100.0)
                & (tile != FOREST_NODATA)
            )
            destination[valid_tile] = tile[valid_tile]

    outside_boundary = geometry_mask(
        [boundary_target],
        out_shape=(height, width),
        transform=transform,
        invert=False,
    )
    valid = np.isfinite(destination) & (destination != FOREST_NODATA)
    destination[valid] = np.clip(destination[valid], 0.0, 100.0)
    destination[~valid | outside_boundary] = FOREST_NODATA

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "crs": TARGET_CRS,
        "transform": transform,
        "dtype": "float32",
        "nodata": FOREST_NODATA,
        "compress": "lzw",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(destination.astype(np.float32), 1)
    return output


def _build_forest_data(boundary_wgs84, cache_dir: Path, *, force: bool) -> Path:
    source_forest = _tree_cover_tifs(boundary_wgs84)
    output = cache_dir / (
        f"{CACHE_COUNTRY_A3.lower()}_forest_cover_grid_"
        f"{_forest_grid_max_size()}px_ss{TERRAIN_SUPERSAMPLE}{_boundary_cache_suffix()}_v1.tif"
    )
    return _write_tree_cover_grid(source_forest, boundary_wgs84, output, force=force)


def _smooth_valid_array(data: np.ndarray, valid: np.ndarray, radius: float) -> np.ndarray:
    values = np.where(valid, data, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    if radius <= 0.0:
        return values
    blurred_values = _gaussian_blur_array(values, float(radius))
    blurred_weights = _gaussian_blur_array(weights, float(radius))
    smoothed = np.divide(
        blurred_values,
        np.maximum(blurred_weights, 1e-6),
        out=np.zeros_like(blurred_values, dtype=np.float32),
        where=blurred_weights > 1e-6,
    )
    smoothed[~valid] = data[~valid]
    return smoothed.astype(np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x.astype(np.float32) - float(edge0)) / max(float(edge1) - float(edge0), 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _reference_style_forest_cover(forest: np.ndarray, valid: np.ndarray) -> np.ndarray:
    cover = np.clip(forest.astype(np.float32), 0.0, 100.0)
    if not np.any(valid):
        return cover

    clustered = _smooth_valid_array(cover, valid, FOREST_TEXTURE_SMOOTH_RADIUS)
    broad = _smooth_valid_array(cover, valid, FOREST_TEXTURE_SMOOTH_RADIUS * 2.15)
    clustered = 0.84 * clustered + 0.16 * broad
    clustered = np.maximum(clustered, cover * 0.96)

    mid_basis = np.maximum(clustered, broad)
    mid_cover = _smoothstep(8.0, 50.0, mid_basis) * (1.0 - _smoothstep(70.0, 96.0, mid_basis))
    edge_cover = _smoothstep(0.4, 12.0, broad) * (1.0 - _smoothstep(56.0, 82.0, broad))
    clustered += 6.0 * mid_cover + 2.1 * edge_cover

    clustered = np.clip(clustered, 0.0, 100.0).astype(np.float32)
    clustered[~valid] = forest[~valid]
    return clustered


def _gaussian_blur_array(data: np.ndarray, radius: float) -> np.ndarray:
    sigma = max(float(radius), 1e-3)
    half_width = max(1, int(math.ceil(sigma * 3.0)))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)

    horizontal_pad = np.pad(data.astype(np.float32), ((0, 0), (half_width, half_width)), mode="edge")
    horizontal = np.zeros_like(data, dtype=np.float32)
    for index, weight in enumerate(kernel):
        horizontal += float(weight) * horizontal_pad[:, index:index + data.shape[1]]

    vertical_pad = np.pad(horizontal, ((half_width, half_width), (0, 0)), mode="edge")
    blurred = np.zeros_like(data, dtype=np.float32)
    for index, weight in enumerate(kernel):
        blurred += float(weight) * vertical_pad[index:index + data.shape[0], :]
    return blurred


def _normalize_valid(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(data, dtype=np.float32)
    finite = valid & np.isfinite(data)
    if not np.any(finite):
        return normalized
    values = data[finite].astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    span = max(high - low, 1e-6)
    normalized[finite] = np.clip((data[finite].astype(np.float32) - low) / span, 0.0, 1.0)
    return normalized


def _write_forest_render_surface(
    forest_path: Path,
    output: Path,
    *,
    force: bool,
) -> Path:
    if _is_fresh(output, [forest_path]) and not force:
        return output

    with rasterio.open(forest_path) as forest_src:
        forest = forest_src.read(1, masked=True)
        forest_data = forest.filled(FOREST_NODATA).astype(np.float32)
        valid = (
            ~np.asarray(forest.mask)
            & np.isfinite(forest_data)
            & (forest_data >= 0.0)
            & (forest_data <= 100.0)
        )

        forest_cover = np.clip(forest_data, 0.0, 100.0)
        broad_forest = _smooth_valid_array(
            forest_cover,
            valid,
            FOREST_HEIGHT_SMOOTH_RADIUS,
        )
        near_forest = _smooth_valid_array(
            forest_cover,
            valid,
            max(1.2, FOREST_HEIGHT_SMOOTH_RADIUS * 0.40),
        )
        smoothed_forest = 0.75 * broad_forest + 0.25 * near_forest
        forest_norm = np.power(np.clip(smoothed_forest / 100.0, 0.0, 1.0), SURFACE_FOREST_GAMMA)
        surface = SURFACE_FOREST_HEIGHT * forest_norm
        surface[~valid] = FOREST_NODATA

        profile = forest_src.profile.copy()
        profile.update(dtype="float32", nodata=FOREST_NODATA, compress="lzw")

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(surface.astype(np.float32), 1)
    return output


def _build_render_surface(forest_path: Path, cache_dir: Path, *, force: bool) -> Path:
    output = cache_dir / f"{CACHE_COUNTRY_A3.lower()}_forest_render_surface{_boundary_cache_suffix()}_v4.tif"
    return _write_forest_render_surface(forest_path, output, force=force)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _forest_palette_rgb() -> np.ndarray:
    return np.asarray([_hex_to_rgb(color) for color in FOREST_PALETTE_HEX], dtype=np.float32)


def _forest_palette_stops() -> np.ndarray:
    stops = np.asarray(FOREST_PALETTE_VALUE_STOPS, dtype=np.float32)
    if stops.shape != (len(FOREST_PALETTE_HEX),):
        raise ValueError("FOREST_PALETTE_VALUE_STOPS must match FOREST_PALETTE_HEX")
    if np.any(np.diff(stops) <= 0.0):
        raise ValueError("FOREST_PALETTE_VALUE_STOPS must be strictly increasing")
    return stops


def _moss_grain(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    hashed = np.sin(rows * 12.9898 + cols * 78.233) * 43758.5453
    grain = (hashed - np.floor(hashed)) * 2.0 - 1.0
    ridged = (
        0.58 * grain
        + 0.26 * np.sin(rows * 0.73 + cols * 1.37)
        + 0.16 * np.sin(rows * 2.11 - cols * 1.79)
    )
    ridged = _gaussian_blur_array(ridged.astype(np.float32), 0.55)
    scale = float(np.percentile(np.abs(ridged), 98.0))
    return np.clip(ridged / max(scale, 1e-6), -1.0, 1.0).astype(np.float32)


def _interpolate_hex_palette(colors: list[str], t: np.ndarray) -> np.ndarray:
    palette = np.asarray([_hex_to_rgb(color) for color in colors], dtype=np.float32)
    clipped = np.clip(t.astype(np.float32), 0.0, 1.0)
    scaled = clipped * float(len(palette) - 1)
    lower = np.clip(np.floor(scaled).astype(np.int16), 0, len(palette) - 1)
    upper = np.clip(lower + 1, 0, len(palette) - 1)
    fraction = scaled - lower.astype(np.float32)
    return palette[lower] * (1.0 - fraction[..., None]) + palette[upper] * fraction[..., None]


def _forest_rgb_for_cover(forest: np.ndarray) -> np.ndarray:
    palette = _forest_palette_rgb()
    stops = _forest_palette_stops()
    cover = np.power(
        np.clip(forest.astype(np.float32), 0.0, 100.0) / 100.0,
        FOREST_COLOR_GAMMA,
        dtype=np.float32,
    ) * 100.0
    lower = np.clip(np.searchsorted(stops, cover, side="right") - 1, 0, len(palette) - 1)
    upper = np.clip(lower + 1, 0, len(palette) - 1)
    span = np.maximum(stops[upper] - stops[lower], 1e-6)
    fraction = np.clip((cover - stops[lower]) / span, 0.0, 1.0)
    return palette[lower] * (1.0 - fraction[..., None]) + palette[upper] * fraction[..., None]


def _forest_rgba(forest: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgba = np.zeros((forest.shape[0], forest.shape[1], 4), dtype=np.uint8)
    if np.any(valid):
        rgb = _forest_rgb_for_cover(forest)
        if min(forest.shape) >= 8:
            moss = _moss_grain(forest.shape)
            strength = _smoothstep(30.0, 92.0, np.clip(forest.astype(np.float32), 0.0, 100.0))
            dark_spots = _smoothstep(0.05, 0.86, moss) * strength
            texture = 1.0 + moss[:, :, None] * (0.070 + 0.185 * strength[:, :, None])
            deep = np.asarray(_hex_to_rgb("#1B1F03"), dtype=np.float32)
            rgb = rgb * texture
            rgb = rgb * (1.0 - 0.42 * dark_spots[:, :, None]) + deep[None, None, :] * (0.42 * dark_spots[:, :, None])
        rgba[valid, :3] = np.round(np.clip(rgb[valid], 0.0, 255.0)).astype(np.uint8)
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _build_overlay(forest_path: Path, output: Path, *, force: bool) -> Path:
    if _is_fresh(output, [forest_path]) and _overlay_is_current(output) and not force:
        return output
    with rasterio.open(forest_path) as forest_src:
        forest = forest_src.read(1, masked=True)
        forest_data = forest.filled(FOREST_NODATA).astype(np.float32)
        valid = ~np.asarray(forest.mask)
        forest_data = _reference_style_forest_cover(forest_data, valid)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_forest_rgba(forest_data, valid), mode="RGBA").save(output)
    _overlay_style_path(output).write_text(OVERLAY_STYLE_VERSION + "\n", encoding="utf-8")
    return output


def _load_font(
    size: int,
    *,
    bold: bool = False,
    serif: bool = False,
    display: bool = False,
) -> ImageFont.ImageFont:
    inter_candidates = [
        Path("C:/Windows/Fonts/bahnschrift.ttf") if display else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold and not display else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts" / "Inter.ttc",
        Path("C:/Windows/Fonts/Inter.ttc"),
        Path("C:/Windows/Fonts/Inter.ttf"),
        Path("C:/Windows/Fonts/Inter-Bold.ttf"),
    ]
    if display:
        names = ["bahnschrift.ttf", "Bahnschrift", "segoeuib.ttf", "Inter.ttc", "Arial Black.ttf", "ariblk.ttf", "DejaVuSansCondensed-Bold.ttf"]
    elif serif:
        names = ["Inter.ttc", "Inter-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["Inter.ttc", "Inter.ttf", "Arial.ttf", "arial.ttf"]
    else:
        names = ["segoeuib.ttf", "Segoe UI Bold", "Inter.ttc", "Inter-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "Segoe UI", "Inter.ttc", "Inter.ttf", "Arial.ttf", "arial.ttf"]
    for path in inter_candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
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


def _blur_float_array(values: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return values.astype(np.float32, copy=True)
    arr = values.astype(np.float32, copy=True)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.nanmin(arr[finite]))
    high = float(np.nanmax(arr[finite]))
    scaled = np.zeros_like(arr, dtype=np.uint8)
    if high > low:
        scaled = np.round(np.clip((arr - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32) / 255.0
    return blurred * (high - low) + low


def _relief_shadow_fields(relief_luminance: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relief = np.asarray(relief_luminance, dtype=np.float32)
    active = np.asarray(mask, dtype=bool) & np.isfinite(relief)
    shadow = np.zeros_like(relief, dtype=np.float32)
    highlight = np.zeros_like(relief, dtype=np.float32)
    if relief.ndim != 2 or not np.any(active):
        return shadow, highlight

    low, high = np.percentile(relief[active], [2.0, 98.0])
    normalized = np.clip((relief - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    normalized[~active] = 0.0
    local = _blur_float_array(normalized, 1.15)
    broad = _blur_float_array(normalized, 9.0)
    dy, dx = np.gradient(local)

    directional = dx * 0.68 + dy * 0.74
    directional_shadow = np.clip(-directional * 8.0, 0.0, 1.0)
    directional_light = np.clip(directional * 7.2, 0.0, 1.0)
    valley = np.clip((broad - local) * 2.4, 0.0, 1.0)
    ridge = np.clip((local - broad) * 1.9, 0.0, 1.0)
    roughness = np.hypot(dx, dy)
    rough_high = float(np.percentile(roughness[active], 96.0)) if np.any(active) else 1.0
    roughness = np.clip(roughness / max(rough_high, 1e-6), 0.0, 1.0)

    shadow = np.clip(0.88 * directional_shadow + 0.42 * valley + 0.10 * roughness, 0.0, 1.0)
    highlight = np.clip(0.68 * directional_light + 0.30 * ridge, 0.0, 1.0)
    shadow[~active] = 0.0
    highlight[~active] = 0.0
    return shadow.astype(np.float32), highlight.astype(np.float32)


def _apply_relief_shadow_model(
    image: Image.Image,
    relief_luminance: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    active = np.asarray(mask, dtype=bool) & (arr[:, :, 3] > 0)
    if relief_luminance.shape != active.shape or not np.any(active):
        return Image.fromarray(arr, mode="RGBA")

    shadow, highlight = _relief_shadow_fields(relief_luminance, active)
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    shadow_color = np.asarray(_hex_to_rgb("#1c160f"), dtype=np.float32) / 255.0
    highlight_color = np.asarray(_hex_to_rgb("#F4D36F"), dtype=np.float32) / 255.0
    shadow_amount = np.clip(float(COMPOSITE["directional_shadow_strength"]) * shadow, 0.0, 0.68)
    highlight_amount = np.clip(float(COMPOSITE["directional_highlight_strength"]) * highlight, 0.0, 0.34)

    rgb = rgb * (1.0 - shadow_amount[:, :, None]) + shadow_color[None, None, :] * shadow_amount[:, :, None]
    rgb = 1.0 - (1.0 - rgb) * (1.0 - highlight_color[None, None, :] * highlight_amount[:, :, None])
    rgb[~active] = arr[:, :, :3].astype(np.float32)[~active] / 255.0
    arr[:, :, :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _restore_palette_chroma_from_source(
    source_image: Image.Image,
    relief_toned_image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    source_arr = np.asarray(source_image.convert("RGBA"), dtype=np.uint8)
    toned_arr = np.asarray(relief_toned_image.convert("RGBA"), dtype=np.uint8).copy()
    if source_arr.shape != toned_arr.shape:
        raise ValueError("Palette source and relief-toned image must have identical dimensions")

    active = (
        np.asarray(mask, dtype=bool)
        & (source_arr[:, :, 3] > 0)
        & (toned_arr[:, :, 3] > 0)
    )
    if not np.any(active):
        return Image.fromarray(toned_arr, mode="RGBA")

    _, source_saturation, source_value = _rgb_to_hsv_channels(source_arr[:, :, :3])
    active &= (source_saturation > 0.12) & (source_value > 0.06)
    if not np.any(active):
        return Image.fromarray(toned_arr, mode="RGBA")

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    source_rgb = source_arr[:, :, :3].astype(np.float32) / 255.0
    toned_rgb = toned_arr[:, :, :3].astype(np.float32) / 255.0
    source_luminance = source_rgb @ weights
    toned_luminance = toned_rgb @ weights
    palette_chroma = source_rgb - source_luminance[:, :, None]

    restored = toned_rgb.copy()
    candidate = toned_luminance[:, :, None] + palette_chroma * float(PALETTE_CHROMA_RESTORE)
    candidate = np.clip(candidate, 0.0, 1.0)
    candidate_luminance = candidate @ weights
    candidate = np.clip(candidate + (toned_luminance - candidate_luminance)[:, :, None], 0.0, 1.0)
    restored[active] = candidate[active]
    toned_arr[:, :, :3] = np.round(restored * 255.0).astype(np.uint8)
    return Image.fromarray(toned_arr, mode="RGBA")


def _apply_reference_terrain_tone(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    subject_alpha = _subject_alpha(image)
    subject = subject_alpha > 0.03
    if arr[:, :, 3].max() > 0 and float(np.count_nonzero(subject)) / float(subject.size) < 0.25:
        subject = arr[:, :, 3] > 0
    forest_green = (
        (hue >= 85.0)
        & (hue <= 155.0)
        & (saturation > 0.25)
        & (value > 0.08)
    )
    blue_relief = (hue >= 185.0) & (hue <= 215.0) & (saturation > 0.12)
    pale_relief = saturation <= 0.28
    desaturated_land = (saturation <= 0.50) & (value > 0.08)
    terrain = (
        subject
        & (arr[:, :, 3] > 0)
        & (value > 0.015)
        & (value < 1.0)
        & ~forest_green
        & (blue_relief | pale_relief | desaturated_land)
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
    zero_cover = np.asarray(_hex_to_rgb(FOREST_PALETTE_HEX[0]), dtype=np.float32) / 255.0
    warm_cover = _forest_rgb_for_cover(np.asarray(20.0, dtype=np.float32)).astype(np.float32) / 255.0
    low_cover = terrain & ~forest_green & (saturation <= 0.50)
    if np.any(low_cover):
        relief = np.clip(1.02 + 0.08 * (toned_luminance - 0.50), 0.98, 1.08)
        target = zero_cover[None, None, :] * relief[:, :, None]
        warm_weight = np.clip(0.09 + 0.04 * (toned_luminance - 0.50), 0.06, 0.13)
        target = target * (1.0 - warm_weight[:, :, None]) + warm_cover[None, None, :] * warm_weight[:, :, None]
        toned[low_cover] = np.clip(target[low_cover], 0.0, 1.0)
    arr[:, :, :3] = np.round(toned * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _apply_reference_forest_tone(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = _subject_alpha(image)
    subject = (arr[:, :, 3] > 0) & (alpha > 0.03)
    if arr[:, :, 3].max() > 0 and float(np.count_nonzero(subject)) / float(subject.size) < 0.25:
        subject = arr[:, :, 3] > 0
    if not np.any(subject):
        return Image.fromarray(arr, mode="RGBA")

    arr = _apply_luminance_transfer(
        arr,
        subject,
        COMPOSITE["forest_luma_quantiles"],
        COMPOSITE["forest_luma_targets"],
    )
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)

    forest_green = (
        subject
        & (hue >= 74.0)
        & (hue <= 155.0)
        & (saturation > 0.22)
        & (value > 0.035)
    )
    if np.any(forest_green):
        moss = _moss_grain(luminance.shape)
        moss_strength = _smoothstep(0.12, 0.62, saturation) * _smoothstep(0.10, 0.48, value)
        forest_strength = np.zeros_like(luminance, dtype=np.float32)
        forest_strength[forest_green] = moss_strength[forest_green]
        deep_moss = np.array(_hex_to_rgb("#1B1F03"), dtype=np.float32) / 255.0
        moss_shadow = _smoothstep(-0.15, 0.95, moss) * forest_strength
        moss_light = _smoothstep(-0.85, 0.20, -moss) * forest_strength
        dark_weight = np.clip(0.24 * forest_strength + 0.50 * moss_shadow, 0.0, 0.70)
        rgb[forest_green] = np.clip(
            rgb[forest_green] * (1.0 - dark_weight[forest_green, None])
            + deep_moss[None, :] * dark_weight[forest_green, None],
            0.0,
            1.0,
        )
        rgb[forest_green] = np.clip(
            rgb[forest_green] * (1.0 + 0.05 * moss_light[forest_green, None]),
            0.0,
            1.0,
        )
        rgb[forest_green] = np.clip(
            rgb[forest_green] * (1.0 + 0.28 * moss[forest_green, None] * forest_strength[forest_green, None]),
            0.0,
            1.0,
        )
        rgb[forest_green] = np.clip(
            rgb[forest_green] + 0.060 * moss[forest_green, None] * forest_strength[forest_green, None],
            0.0,
            1.0,
        )
        grain = moss[forest_green, None] * forest_strength[forest_green, None]
        grain_color = np.array([0.055, 0.090, 0.012], dtype=np.float32)
        rgb[forest_green] = np.clip(
            rgb[forest_green] + grain * grain_color[None, :],
            0.0,
            1.0,
        )

    dark_green = forest_green & (luminance < 0.26)
    if np.any(dark_green):
        target = np.array(_hex_to_rgb("#1B1F03"), dtype=np.float32) / 255.0
        target_luminance = float(target @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        target_at_luminance = target[None, :] * (luminance[dark_green, None] / max(target_luminance, 1e-6))
        weight = np.clip((0.31 - luminance[dark_green]) / 0.24, 0.0, 0.82)
        rgb[dark_green] = np.clip(
            rgb[dark_green] * (1.0 - weight[:, None]) + target_at_luminance * weight[:, None],
            0.0,
            1.0,
        )

    if np.any(forest_green):
        final_grain = moss[forest_green, None] * forest_strength[forest_green, None]
        rgb[forest_green] = np.clip(
            rgb[forest_green] * (1.0 + 0.46 * final_grain)
            + final_grain * np.array([0.098, 0.205, 0.022], dtype=np.float32)[None, :],
            0.0,
            1.0,
        )

    tan_land = subject & ~forest_green & (value > 0.12)
    if np.any(tan_land):
        warm = np.array(_hex_to_rgb("#E3CDBF"), dtype=np.float32) / 255.0
        weight = 0.12 + 0.16 * np.clip((0.38 - saturation[tan_land]) / 0.38, 0.0, 1.0)
        rgb[tan_land] = np.clip(
            rgb[tan_land] * (1.0 - weight[:, None]) + warm[None, :] * weight[:, None],
            0.0,
            1.0,
        )

    arr[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


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


def _alpha_bbox(image: Image.Image, *, threshold: int = 8) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.uint8)
    mask = alpha > int(threshold)
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _fraction_bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    return (
        round(width * bbox[0]),
        round(height * bbox[1]),
        round(width * bbox[2]),
        round(height * bbox[3]),
    )


def _fit_subject_to_reference_bbox(
    subject: Image.Image,
    size: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    source_bbox = _alpha_bbox(subject)
    if source_bbox is None:
        return subject, (0, 0)

    target_bbox = _fraction_bbox_to_pixels(MAP_TARGET_LAND_BBOX, size)
    target_w = max(1, target_bbox[2] - target_bbox[0])
    target_h = max(1, target_bbox[3] - target_bbox[1])
    source_w = max(1, source_bbox[2] - source_bbox[0])
    source_h = max(1, source_bbox[3] - source_bbox[1])
    scale = min(target_w / source_w, target_h / source_h)

    fitted = subject
    fitted_bbox = source_bbox
    for _ in range(3):
        fitted = subject.resize(
            (
                max(1, round(subject.width * scale)),
                max(1, round(subject.height * scale)),
            ),
            resample=Image.Resampling.LANCZOS,
        )
        fitted = _polish_subject_alpha(fitted)
        fitted_bbox = _alpha_bbox(fitted) or (0, 0, fitted.width, fitted.height)
        actual_w = max(1, fitted_bbox[2] - fitted_bbox[0])
        actual_h = max(1, fitted_bbox[3] - fitted_bbox[1])
        if actual_w <= target_w and actual_h <= target_h and (
            abs(actual_w - target_w) <= 1 or abs(actual_h - target_h) <= 1
        ):
            break
        scale *= min(target_w / actual_w, target_h / actual_h)

    x = round(target_bbox[0] + (target_w - (fitted_bbox[2] - fitted_bbox[0])) / 2.0 - fitted_bbox[0])
    y = round(target_bbox[1] + (target_h - (fitted_bbox[3] - fitted_bbox[1])) / 2.0 - fitted_bbox[1])
    return fitted, (x, y)


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


def _apply_final_canvas_terrain_tone(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    hue, saturation, value = _rgb_to_hsv_channels(arr[:, :, :3])
    height, _ = value.shape
    y = np.indices(value.shape, dtype=np.int32)[0]
    map_band = (y > round(height * 0.15)) & (y <= round(height * 0.885))
    forest_green = (
        (hue >= 85.0)
        & (hue <= 155.0)
        & (saturation > 0.25)
        & (value > 0.08)
    )
    terrain_color = (
        ((hue >= 175.0) & (hue <= 225.0) & (saturation > 0.10))
        | (saturation <= 0.32)
    )
    non_background = np.abs(arr[:, :, :3].astype(np.int16) - 255).max(axis=2) > 8
    terrain = (
        map_band
        & non_background
        & (arr[:, :, 3] > 0)
        & (value > 0.015)
        & (value < 1.0)
        & ~forest_green
        & terrain_color
    )
    arr = _apply_luminance_transfer(
        arr,
        terrain,
        COMPOSITE["canvas_luma_quantiles"],
        COMPOSITE["canvas_luma_targets"],
    )
    return Image.fromarray(arr, mode="RGBA")


def _apply_final_map_brightness(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    brightness = float(FINAL_MAP_BRIGHTNESS)
    if brightness == 1.0:
        return Image.fromarray(arr, mode="RGBA")

    rgb = arr[:, :, :3].astype(np.float32)
    background = np.asarray(MAP_BACKGROUND_RGB, dtype=np.int16)
    non_background = np.abs(arr[:, :, :3].astype(np.int16) - background[None, None, :]).max(axis=2) > 2
    active = (arr[:, :, 3] > 0) & non_background
    rgb[active] = _apply_final_rgb_lighting(rgb[active])
    arr[:, :, :3] = np.round(rgb).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _apply_final_rgb_lighting(rgb: tuple[int, int, int] | np.ndarray) -> tuple[int, int, int] | np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    lit = 1.0 - np.power(np.clip(1.0 - values, 0.0, 1.0), float(FINAL_MAP_BRIGHTNESS))
    lit = lit + (1.0 - lit) * float(FINAL_MAP_WHITE_MIX)
    lit_u8 = np.minimum(
        np.round(np.clip(lit * 255.0, 0.0, 255.0)).astype(np.uint8),
        np.uint8(254),
    )
    if lit_u8.ndim == 1:
        return tuple(int(value) for value in lit_u8)
    return lit_u8


def _compose_clean_palette_with_relief(
    palette_image: Image.Image,
    relief_image: Image.Image,
    mask_image: Image.Image | None = None,
) -> Image.Image:
    palette_image = palette_image.convert("RGBA")
    relief_image = relief_image.convert("RGBA")
    if palette_image.size != relief_image.size:
        raise ValueError("Palette and relief passes must have identical dimensions")
    mask_source = palette_image if mask_image is None else mask_image.convert("RGBA")
    if mask_source.size != palette_image.size:
        raise ValueError("Mask source and palette pass must have identical dimensions")

    palette = np.asarray(palette_image, dtype=np.float32) / 255.0
    relief_raw = np.asarray(relief_image, dtype=np.float32) / 255.0
    relief = np.asarray(
        relief_image.filter(ImageFilter.GaussianBlur(radius=COMPOSITE["relief_blur"])),
        dtype=np.float32,
    ) / 255.0
    alpha = _subject_alpha(mask_source)
    mask = alpha > 0.03
    if not np.any(mask):
        return palette_image

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
    combined = np.zeros_like(palette)
    combined[:, :, :3] = np.clip(palette[:, :, :3] * value_scale[:, :, None], 0.0, 1.0)
    combined[:, :, 3] = np.clip(alpha, 0.0, 1.0)
    combined[~mask, :3] = 0.0

    relief_luminance = relief_raw[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    relief_toned = _apply_relief_shadow_model(
        Image.fromarray(np.round(combined * 255.0).astype(np.uint8), mode="RGBA"),
        relief_luminance,
        mask,
    )
    return _restore_palette_chroma_from_source(palette_image, relief_toned, mask)


def _combine_render_passes(
    palette_raw_path: Path,
    relief_raw_path: Path,
    mask_raw_path: Path | None = None,
) -> Image.Image:
    palette_image = Image.open(palette_raw_path).convert("RGBA")
    if not APPLY_RELIEF_TONE_COMPOSITE:
        return palette_image

    relief_image = Image.open(relief_raw_path).convert("RGBA")
    mask_image = Image.open(mask_raw_path).convert("RGBA") if mask_raw_path is not None else None
    return _compose_clean_palette_with_relief(palette_image, relief_image, mask_image)


def _polish_subject_alpha(subject: Image.Image) -> Image.Image:
    alpha = subject.getchannel("A")
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.65))
    polished = subject.copy()
    polished.putalpha(alpha)
    return polished


def _legend_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x, y, box_width, box_height = LEGEND_BOX_BBOX
    return (
        round(width * x),
        round(height * y),
        max(10, round(width * box_width)),
        max(48, round(height * box_height)),
    )


def _draw_vertical_moss_legend(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    x, y, width, height = _legend_box(size)
    title_font = _load_font(round(size[0] * LEGEND_TITLE_FONT_SCALE), bold=True, display=True)
    label_font = _load_font(round(size[0] * LEGEND_LABEL_FONT_SCALE), bold=True)
    draw.text(
        (x + width // 2, y - round(size[1] * LEGEND_TITLE_GAP)),
        LEGEND_TITLE,
        fill=MAP_TITLE_RGB,
        font=title_font,
        anchor="mm",
    )
    for offset in range(height):
        cover = np.asarray(100.0 * (1.0 - offset / max(height - 1, 1)), dtype=np.float32)
        color = np.round(_forest_rgb_for_cover(cover)).astype(np.uint8)
        color = _apply_final_rgb_lighting(color)
        draw.line((x, y + offset, x + width, y + offset), fill=tuple(int(value) for value in color))

    for label, fraction in LEGEND_TICKS:
        ty = y + round((1.0 - fraction) * height)
        draw.text(
            (x + width + round(size[0] * 0.022), ty),
            label,
            fill=MAP_TITLE_RGB,
            font=label_font,
            anchor="lm",
        )


def _draw_horizontal_moss_legend(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    x, y, width, height = _legend_box(size)
    title_font = _load_font(round(size[0] * LEGEND_TITLE_FONT_SCALE), bold=True, display=True)
    label_font = _load_font(round(size[0] * LEGEND_LABEL_FONT_SCALE), bold=True)
    draw.text(
        (x + width // 2, y - round(size[1] * LEGEND_TITLE_GAP)),
        LEGEND_TITLE,
        fill=MAP_TITLE_RGB,
        font=title_font,
        anchor="mm",
    )
    for offset in range(width):
        cover = np.asarray(100.0 * (offset / max(width - 1, 1)), dtype=np.float32)
        color = np.round(_forest_rgb_for_cover(cover)).astype(np.uint8)
        color = _apply_final_rgb_lighting(color)
        draw.line((x + offset, y, x + offset, y + height), fill=tuple(int(value) for value in color))

    label_y = y + height + round(size[1] * LEGEND_LABEL_GAP)
    for label, fraction in reversed(LEGEND_TICKS):
        tx = x + round(fraction * (width - 1))
        draw.text(
            (tx, label_y),
            label,
            fill=MAP_TITLE_RGB,
            font=label_font,
            anchor="ma",
        )


def _draw_moss_legend(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    if LEGEND_ORIENTATION == "horizontal":
        _draw_horizontal_moss_legend(draw, size)
        return
    _draw_vertical_moss_legend(draw, size)


def _text_layer(text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> Image.Image:
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=font)
    pad = max(6, round(getattr(font, "size", 16) * 0.25))
    layer = Image.new(
        "RGBA",
        (max(1, bbox[2] - bbox[0] + pad * 2), max(1, bbox[3] - bbox[1] + pad * 2)),
        (0, 0, 0, 0),
    )
    ImageDraw.Draw(layer).text((pad - bbox[0], pad - bbox[1]), text, fill=fill, font=font)
    alpha_bbox = _alpha_bbox(layer, threshold=1)
    return layer.crop(alpha_bbox) if alpha_bbox is not None else layer


def _draw_text_fitted_to_bbox(
    canvas: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    layer = _text_layer(text, font, fill)
    target_w = max(1, bbox[2] - bbox[0])
    target_h = max(1, bbox[3] - bbox[1])
    layer = layer.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    canvas.alpha_composite(layer, dest=(bbox[0], bbox[1]))


def _draw_reference_style_text(canvas: Image.Image) -> None:
    size = canvas.size
    width, height = size
    title_center_x = round(width * TITLE_CENTER_X)

    _draw_text_fitted_to_bbox(
        canvas,
        "FOREST MAP",
        _fraction_bbox_to_pixels(TITLE_FOREST_MAP_BBOX, size),
        font=_load_font(round(width * 0.019), bold=True, display=True),
        fill=MAP_TITLE_RGB,
    )
    _draw_text_fitted_to_bbox(
        canvas,
        COUNTRY_TITLE,
        _fraction_bbox_to_pixels(TITLE_COUNTRY_BBOX, size),
        font=_load_font(round(width * 0.106), bold=True, display=True),
        fill=MAP_TITLE_RGB,
    )

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (title_center_x, round(height * SUBTITLE_Y)),
        "MILOSGIS.COM",
        fill=MAP_TITLE_RGB,
        font=_load_font(round(width * SUBTITLE_FONT_SCALE), bold=True),
        anchor="ma",
    )
    draw.text(
        (title_center_x, round(height * CAPTION_Y)),
        "COPERNICUS LAND COVER 2019",
        fill=MAP_TITLE_RGB,
        font=_load_font(round(width * CAPTION_FONT_SCALE), bold=True),
        anchor="ma",
    )


def _alpha_shadow_layer(
    size: tuple[int, int],
    alpha: Image.Image,
    origin: tuple[int, int],
    *,
    offset: tuple[int, int],
    blur: float,
    rgba: tuple[int, int, int, int],
    spread: int = 1,
) -> Image.Image:
    source = alpha
    if spread > 1:
        source = source.filter(ImageFilter.MaxFilter(spread if spread % 2 == 1 else spread + 1))
    mask = Image.new("L", size, 0)
    mask.paste(source, (origin[0] + offset[0], origin[1] + offset[1]))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    layer = Image.new("RGBA", size, rgba[:3] + (0,))
    layer.putalpha(mask.point(lambda value: int(round(value * (rgba[3] / 255.0)))))
    return layer


def _composite_reference_contact_shadows(
    canvas: Image.Image,
    alpha: Image.Image,
    origin: tuple[int, int],
) -> None:
    width, height = canvas.size
    lavender = _alpha_shadow_layer(
        canvas.size,
        alpha,
        origin,
        offset=(round(width * 0.013), round(height * 0.013)),
        blur=MAP_LAVENDER_GLOW_BLUR,
        rgba=MAP_LAVENDER_GLOW_RGBA,
        spread=3,
    )
    canvas.alpha_composite(lavender)

    broad_blue = _alpha_shadow_layer(
        canvas.size,
        alpha,
        origin,
        offset=(round(width * 0.024), round(height * 0.021)),
        blur=MAP_BLUE_GLOW_BLUR,
        rgba=MAP_BLUE_GLOW_RGBA,
        spread=5,
    )
    canvas.alpha_composite(broad_blue)

    contact_blue = _alpha_shadow_layer(
        canvas.size,
        alpha,
        origin,
        offset=(round(width * 0.030), round(height * 0.026)),
        blur=MAP_CONTACT_BLUE_BLUR,
        rgba=MAP_CONTACT_BLUE_RGBA,
        spread=9,
    )
    canvas.alpha_composite(contact_blue)

    warm_umbra = _alpha_shadow_layer(
        canvas.size,
        alpha,
        origin,
        offset=(round(width * 0.007), round(height * 0.008)),
        blur=MAP_WARM_SHADOW_BLUR,
        rgba=MAP_WARM_SHADOW_RGBA,
        spread=5,
    )
    canvas.alpha_composite(warm_umbra)


def _compose_snapshot(raw, output_path: Path) -> None:
    raw = raw.convert("RGBA") if isinstance(raw, Image.Image) else Image.open(raw).convert("RGBA")
    subject = _crop_subject(raw)
    canvas = Image.new("RGBA", COMPOSE_CANVAS_SIZE, (*MAP_BACKGROUND_RGB, 255))

    if not COMPOSE_POSTER:
        subject, (x, y) = _fit_subject_to_reference_bbox(subject, canvas.size)
        subject = _apply_final_map_brightness(subject)

        alpha = subject.getchannel("A")
        _composite_reference_contact_shadows(canvas, alpha, (x, y))

        canvas.alpha_composite(subject, dest=(x, y))
        _draw_reference_style_text(canvas)
        draw = ImageDraw.Draw(canvas)
        _draw_moss_legend(draw, canvas.size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final = canvas.convert("RGB")
        if final.size != SNAPSHOT_SIZE:
            final = final.resize(SNAPSHOT_SIZE, resample=Image.Resampling.LANCZOS)
        final.save(output_path)
        return

    draw = ImageDraw.Draw(canvas)

    target_w = round(canvas.width * 1.02)
    target_h = round(canvas.height * 0.74)
    scale = min(target_w / max(subject.width, 1), target_h / max(subject.height, 1))
    subject = subject.resize(
        (max(1, round(subject.width * scale)), max(1, round(subject.height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    subject = _polish_subject_alpha(subject)

    alpha = subject.getchannel("A")
    shadow = Image.new("RGBA", subject.size, (42, 48, 56, 0))
    shadow.putalpha(alpha.point(lambda value: int(round(value * 0.26))))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(10, canvas.width // 95)))
    map_x = round(canvas.width * 0.004)
    map_y = round(canvas.height * 0.152)
    canvas.alpha_composite(shadow, dest=(map_x - canvas.width // 75, map_y + canvas.height // 70))
    canvas.alpha_composite(subject, dest=(map_x, map_y))

    title_x = canvas.width // 2
    title_y = round(canvas.height * 0.045)
    title_color = (30, 73, 96)
    title_font = _load_font(round(canvas.width * 0.061), serif=True)
    country_font = _load_font(round(canvas.width * 0.070), bold=True, serif=True)
    caption_font = _load_font(round(canvas.width * 0.017), serif=True)
    draw.text((title_x, title_y), TITLE_LINES[0], fill=title_color, font=title_font, anchor="ma")
    draw.text((title_x, title_y + round(canvas.height * 0.072)), TITLE_LINES[1], fill=title_color, font=country_font, anchor="ma")

    caption_y = round(canvas.height * 0.900)
    line_gap = round(canvas.height * 0.026)
    for index, line in enumerate(CAPTION_LINES):
        draw.text((canvas.width // 2, caption_y + index * line_gap), line, fill=title_color, font=caption_font, anchor="ma")

    canvas = _apply_final_canvas_terrain_tone(canvas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def _render(
    snapshot_path: Path,
    surface_path: Path,
    overlay_path: Path,
    *,
    force_hdri: bool = False,
) -> None:
    hdr = _ensure_hdri(force=force_hdri).resolve()
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
    with tempfile.TemporaryDirectory(prefix=f"forge3d_{REGION_SLUG}_forest_render_") as tmp:
        with rasterio.open(surface_path) as surface_src:
            terrain_width = float(max(surface_src.width, surface_src.height))
        terrain_xy = terrain_width / CAMERA["ref"]
        radius = CAMERA["radius"] * terrain_xy * CAMERA["pullback"]
        zscale = CAMERA["zscale"] * math.sqrt(max(terrain_xy, 1e-6))
        terrain_cmd = {**TERRAIN, "radius": radius, "zscale": zscale}
        relief_terrain = {**terrain_cmd, **RELIEF_TERRAIN}
        palette_raw = Path(tmp) / "palette_raw.png"
        relief_raw = Path(tmp) / "relief_raw.png"
        with f3d.open_viewer_async(
            terrain_path=surface_path,
            width=VIEWER_SIZE[0],
            height=VIEWER_SIZE[1],
            timeout=VIEWER_TIMEOUT_SECONDS,
        ) as viewer:
            viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr_cmd})
            viewer.load_overlay(
                f"{REGION_SLUG}_forest",
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
                palette_raw,
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
        raw = _combine_render_passes(palette_raw, relief_raw, palette_raw)
        _compose_snapshot(raw, snapshot_path)


REQUESTED_BATCH = (
    "france",
    "germany",
    "poland",
    "turkey",
    "southeast_europe",
    "mainland_usa",
    "iberia",
    "russia",
    "africa",
    "europe",
)


def _run_region(
    region: RegionPreset,
    *,
    output_dir_arg: Path | None,
    cache_dir_arg: Path | None,
    snapshot_arg: Path | None,
    force: bool,
) -> Path:
    _configure_region(region)
    output_dir = (output_dir_arg if output_dir_arg is not None else OUT_DIR).resolve()
    cache_dir = (cache_dir_arg if cache_dir_arg is not None else CACHE_DIR).resolve()
    snapshot = (
        snapshot_arg.resolve()
        if snapshot_arg is not None
        else output_dir / f"{region.slug}_forest_cover.png"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    boundary_file = _ensure_boundary_file(cache_dir, force=force)
    boundary_wgs84 = _country_geometry(boundary_file, "EPSG:4326")
    forest_path = _build_forest_data(boundary_wgs84, cache_dir, force=force)
    surface_path = _build_render_surface(forest_path, cache_dir, force=force)
    overlay_path = _build_overlay(forest_path, cache_dir / OVERLAY_CACHE_NAME, force=force)
    _render(snapshot, surface_path, overlay_path, force_hdri=force)
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
        force=bool(args.force),
    )

    print(f"Success! Map saved to: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
