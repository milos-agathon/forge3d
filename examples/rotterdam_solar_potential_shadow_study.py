#!/usr/bin/env python3
"""Rotterdam Solar Potential And Shadow Study.

Sources used by default:
- 3D BAG public WFS, layer ``BAG3D:lod22``. This provides real LoD2.2 roof
  surface footprints plus AHN-derived roof height, slope, and azimuth fields.
- OpenStreetMap via Overpass for water, roads, rail, parks, bridges, and labels.
- PVGIS/JRC PVcalc for a cached annual irradiation reference when reachable.

The solar numbers are a planning and visualization estimate. They are not a
structural, electrical, legal, or grid-capacity assessment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import ssl
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely import BufferCapStyle, BufferJoinStyle, affinity
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform, triangulate, unary_union

from _import_shim import ensure_repo_import

ensure_repo_import()

import osm_city_daycycle as day  # noqa: E402
import osm_city_demo as city  # noqa: E402

try:  # noqa: E402
    import forge3d as f3d
except Exception:  # pragma: no cover - examples must still run without native ephemeris
    f3d = None

try:  # noqa: E402
    import certifi
except Exception:  # pragma: no cover
    certifi = None

try:  # noqa: E402
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "rotterdam_solar_potential"
DEFAULT_STATIC_OUTPUT = (
    PROJECT_ROOT / "examples" / "out" / "rotterdam_solar_potential_shadow_study.png"
)
DEFAULT_ANIMATION_OUTPUT = (
    PROJECT_ROOT / "examples" / "out" / "rotterdam_solar_potential_shadow_study.mp4"
)

DEFAULT_LON = 4.486
DEFAULT_LAT = 51.909
DEFAULT_RADIUS_M = 1200.0
DEFAULT_DATE = "2026-06-21"
DEFAULT_TIME = "13:00"
DEFAULT_FRAMES = 180
DEFAULT_FPS = 24
DEFAULT_MAX_3DBAG_SURFACES = 60000
DEFAULT_OSM_TIMEOUT_SECONDS = 20.0

THREEDBAG_WFS_URL = "https://data.3dbag.nl/api/BAG3D/wfs"
PVGIS_ENDPOINTS = (
    "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc",
    "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc",
)
USER_AGENT = "forge3d-rotterdam-solar-potential/1.0"
FALLBACK_IRRADIANCE_KWH_M2_YEAR = 1030.0

CONTEXT_COLORS = {
    "base": (0xA9, 0xB3, 0xAD, 255),
    "landuse": (0xB7, 0xC1, 0xB8, 255),
    "park": (0x7D, 0xA8, 0x73, 245),
    "water": (0x6E, 0x8B, 0x9E, 250),  # More neutral blue-gray, distinct from solar colors
    "paved": (0xA9, 0xB2, 0xB2, 245),
    "parking": (0x9F, 0xAA, 0xAE, 245),
    "platform": (0x96, 0xA2, 0xA7, 245),
    "road": (0x9D, 0xA7, 0xAD, 230),   # Muted gray for basemap context
    "road_hi": (0xBF, 0xC7, 0xCA, 225),
}
ROOF_COLORS = {
    "high": (0xFA, 0xD9, 0x49, 255),
    "medium": (0xD9, 0xA0, 0x3D, 255),
    "low": (0xA8, 0xB0, 0xB1, 255),
    "constrained": (0x6A, 0x74, 0x7A, 255),
    "wall": (0x88, 0x90, 0x96, 200),  # Lighter, more transparent walls
    "outline": (0x38, 0x42, 0x48, 140),  # Subtler outlines
}
SOLAR_HIGH_KWH_M2 = 840.0
SOLAR_MEDIUM_KWH_M2 = 640.0
SOLAR_LOW_KWH_M2 = 400.0
SOLAR_MIN_KWH_M2 = 0.0
SOLAR_MAX_KWH_M2 = 1000.0
COLORMAP_BINS = 20

STUDY_SHADOW_TINT_RGB = (91, 110, 136)
STUDY_SHADOW_OPACITY = 0.64
# Shadow legend color - approximate the blended appearance on a light surface
SHADOW_LEGEND_COLOR = (
    int(91 * 0.64 + 200 * 0.36),   # Blend with approximate base
    int(110 * 0.64 + 200 * 0.36),
    int(136 * 0.64 + 180 * 0.36),
    255,
)

# Vibrant solar potential palette - bright and saturated to stand out against
# the muted basemap. Light yellow for low potential, through bright orange,
# to deep red/maroon for high potential.
SOLAR_COLORS = (
    (255, 255, 180),   # Bright light yellow - low potential
    (255, 230, 90),    # Bright yellow
    (255, 190, 50),    # Bright golden yellow
    (255, 140, 30),    # Bright orange
    (240, 80, 40),     # Bright red-orange
    (190, 40, 60),     # Deep red
    (120, 20, 40),     # Dark maroon - high potential
)
STUDY_SHADOW_TINT_RGB = (91, 110, 136)
STUDY_SHADOW_OPACITY = 0.64
RECTANGULAR_AOI_ASPECT_MIN = 0.75
RECTANGULAR_AOI_ASPECT_MAX = 2.20
# Scene rotation to align horizontally with canvas (0 = no rotation from base orientation)
SCENE_ROTATION_DEG = -135.0  # Rotate scene to align river/bridge horizontally
# No labels on map
LABEL_POINTS = ()
FONT_CANDIDATES = (
    "/Users/mpopovic3/Library/Fonts/Inconsolata.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "DejaVuSans.ttf",
)

HTTPS_CONTEXT = (
    ssl.create_default_context(cafile=certifi.where()) if certifi is not None else None
)


@dataclass(frozen=True)
class MetricContext:
    epsg: int
    to_metric: Transformer
    to_wgs84: Transformer
    to_rd: Transformer
    center_xy: tuple[float, float]
    aoi_m: Polygon
    aoi_rd: Polygon
    bbox_wgs84: tuple[float, float, float, float]
    bbox_rd: tuple[float, float, float, float]


@dataclass(frozen=True)
class SolarAssumptions:
    panel_efficiency: float = 0.20
    performance_ratio: float = 0.82
    min_usable_roof_area_m2: float = 18.0
    roof_usable_fraction: float = 0.72
    orientation_penalty: float = 0.42
    slope_penalty: float = 0.22
    shadow_penalty: float = 0.25


@dataclass
class RoofSurface:
    surface_id: str
    building_id: str
    geometry: Polygon | MultiPolygon
    area_m2: float
    ground_abs_m: float
    height_ref_abs_m: float
    height_min_abs_m: float
    height_max_abs_m: float
    slope_deg: float
    aspect_deg: float
    roof_type: str
    is_glass_roof: bool
    quality_ok: bool
    gradient_xy: np.ndarray
    centroid_xy: tuple[float, float]
    source_lod: str = "lod22"
    annual_kwh_m2: float = 0.0
    annual_kwh: float = 0.0
    usable_area_m2: float = 0.0
    category: str = "low"
    constraint_reason: str | None = None
    current_incidence: float = 0.0


@dataclass(frozen=True)
class IrradianceReference:
    annual_kwh_m2: float
    source: str
    detail: str


@dataclass(frozen=True)
class MeshBuildStats:
    roof_triangles: int
    wall_triangles: int
    roof_surfaces: int
    buildings: int


@dataclass(frozen=True)
class StudySummary:
    roof_surfaces: int
    buildings: int
    usable_area_m2: float
    annual_kwh: float
    high_area_m2: float
    medium_area_m2: float
    low_area_m2: float
    constrained_area_m2: float


@dataclass(frozen=True)
class OutputLayout:
    final_width: int
    final_height: int
    map_width: int
    map_height: int
    panel_width: int
    panel_height: int
    panel_side: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Rotterdam 3D BAG solar potential and shadow study around "
            "Erasmus Bridge / Kop van Zuid."
        )
    )
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="AOI center longitude.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="AOI center latitude.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M, help="AOI radius in meters.")
    parser.add_argument("--date", type=str, default=DEFAULT_DATE, help="Study date, YYYY-MM-DD.")
    parser.add_argument("--time", type=str, default=DEFAULT_TIME, help="Local Rotterdam time, HH:MM.")
    parser.add_argument("--animate", action="store_true", help="Render an MP4 day-cycle animation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_STATIC_OUTPUT, help="Output PNG or MP4 path.")
    parser.add_argument("--size", type=int, nargs=2, default=(7680, 4320), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Animation frame count.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Animation frame rate.")
    parser.add_argument("--supersample", type=int, default=1, help="Internal render scale before downsampling.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Optional directory for rendered PNG frames.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep animation frames after encoding.")
    parser.add_argument("--refresh-osm", action="store_true", help="Ignore cached Overpass responses.")
    parser.add_argument("--refresh-buildings", action="store_true", help="Ignore cached 3D BAG WFS response.")
    parser.add_argument("--refresh-pvgis", action="store_true", help="Ignore cached PVGIS response.")
    parser.add_argument(
        "--no-lod12-fallback",
        action="store_true",
        help="Do not add 3D BAG LoD1.2 fallback geometry for buildings missing LoD2.2 surfaces.",
    )
    parser.add_argument(
        "--max-3dbag-surfaces",
        type=int,
        default=DEFAULT_MAX_3DBAG_SURFACES,
        help="Maximum 3D BAG features to request per layer.",
    )
    parser.add_argument("--panel-efficiency", type=float, default=0.20)
    parser.add_argument("--performance-ratio", type=float, default=0.82)
    parser.add_argument("--min-roof-area", type=float, default=18.0)
    parser.add_argument("--shadow-penalty", type=float, default=0.25)
    parser.add_argument("--orientation-penalty", type=float, default=0.42)
    parser.add_argument("--slope-penalty", type=float, default=0.22)
    parser.add_argument(
        "--skip-pvgis",
        action="store_true",
        help="Use the documented Rotterdam fallback irradiation instead of PVGIS.",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Render only the base map surface and 3D BAG solar layer.",
    )
    parser.add_argument(
        "--osm-timeout",
        type=float,
        default=DEFAULT_OSM_TIMEOUT_SECONDS,
        help="Per-endpoint Overpass timeout in seconds for optional context layers.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    return city.display_path(path)


def parse_study_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def parse_clock_time(value: str) -> dt.time:
    text = value.strip()
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time {value!r}; expected HH:MM or HH:MM:SS")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    return dt.time(hour=hour, minute=minute, second=second)


def load_font(size_px: int) -> ImageFont.ImageFont:
    for font_name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=max(1, int(size_px)))
        except OSError:
            continue
    return ImageFont.load_default()


def solar_to_color(kwh_m2: float) -> tuple[int, int, int, int]:
    """Map solar potential (kWh/m2/year) to colormap color."""
    t = np.clip((kwh_m2 - SOLAR_MIN_KWH_M2) / (SOLAR_MAX_KWH_M2 - SOLAR_MIN_KWH_M2), 0.0, 1.0)
    idx = min(int(t * (len(SOLAR_COLORS) - 1)), len(SOLAR_COLORS) - 2)
    frac = t * (len(SOLAR_COLORS) - 1) - idx
    c0 = SOLAR_COLORS[idx]
    c1 = SOLAR_COLORS[idx + 1]
    r = int(c0[0] + frac * (c1[0] - c0[0]))
    g = int(c0[1] + frac * (c1[1] - c0[1]))
    b = int(c0[2] + frac * (c1[2] - c0[2]))
    return (r, g, b, 255)


def solar_bin_index(kwh_m2: float) -> int:
    """Return the bin index (0 to COLORMAP_BINS-1) for a solar value."""
    t = np.clip((kwh_m2 - SOLAR_MIN_KWH_M2) / (SOLAR_MAX_KWH_M2 - SOLAR_MIN_KWH_M2), 0.0, 1.0)
    return min(int(t * COLORMAP_BINS), COLORMAP_BINS - 1)


def solar_color_for_bin(bin_idx: int) -> tuple[int, int, int, int]:
    """Return the solar color for a given bin index."""
    t = bin_idx / max(1, COLORMAP_BINS - 1)
    return solar_to_color(SOLAR_MIN_KWH_M2 + t * (SOLAR_MAX_KWH_M2 - SOLAR_MIN_KWH_M2))


def first_float(props: dict, keys: tuple[str, ...], default: float | None = None) -> float | None:
    for key in keys:
        value = props.get(key)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return default


def build_metric_context(
    lon: float,
    lat: float,
    radius_m: float,
    *,
    aoi_shape: str = "circle",
    aspect_ratio: float = 1.0,
) -> MetricContext:
    epsg = city.utm_epsg_for_lon_lat(lon, lat)
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    center = transform(to_metric.transform, Point(lon, lat))
    center_xy = tuple(center.coords[0])
    radius = max(1.0, float(radius_m))
    if aoi_shape == "circle":
        aoi_m = Point(*center_xy).buffer(radius, resolution=96)
    elif aoi_shape == "rectangle":
        aspect = float(np.clip(float(aspect_ratio), RECTANGULAR_AOI_ASPECT_MIN, RECTANGULAR_AOI_ASPECT_MAX))
        half_height = radius * math.sqrt(math.pi / (4.0 * aspect))
        half_width = half_height * aspect
        cx, cy = center_xy
        aoi_m = Polygon(
            [
                (cx - half_width, cy - half_height),
                (cx + half_width, cy - half_height),
                (cx + half_width, cy + half_height),
                (cx - half_width, cy + half_height),
                (cx - half_width, cy - half_height),
            ]
        )
    else:
        raise ValueError(f"Unsupported AOI shape {aoi_shape!r}; expected 'circle' or 'rectangle'")
    aoi_rd = transform(to_rd.transform, transform(to_wgs84.transform, aoi_m))
    bbox_wgs84 = transform(to_wgs84.transform, aoi_m).bounds
    return MetricContext(
        epsg=epsg,
        to_metric=to_metric,
        to_wgs84=to_wgs84,
        to_rd=to_rd,
        center_xy=(float(center_xy[0]), float(center_xy[1])),
        aoi_m=aoi_m,
        aoi_rd=aoi_rd,
        bbox_wgs84=tuple(float(v) for v in bbox_wgs84),
        bbox_rd=tuple(float(v) for v in aoi_rd.bounds),
    )


def request_json(url: str, cache_path: Path, *, refresh: bool, label: str) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        payload = fetch_json_url(url, label=label)
    except RuntimeError:
        if cache_path.exists():
            print(f"[{label}] warning: live request failed; using cached {cache_path.name}")
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def fetch_json_url(url: str, *, label: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(2):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=120, context=HTTPS_CONTEXT) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            sleep(1.25 * (attempt + 1))
    raise RuntimeError(f"{label} request failed: {last_error}") from last_error


def fetch_overpass_query_cached(
    query: str,
    cache_path: Path,
    refresh: bool,
    *,
    timeout_seconds: float,
) -> list[dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])

    payload = None
    last_error: Exception | None = None
    for endpoint in city.OVERPASS_URLS:
        for attempt in range(1):
            request = Request(
                endpoint,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with urlopen(request, timeout=max(3.0, float(timeout_seconds)), context=HTTPS_CONTEXT) as response:
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                sleep(1.5 * (attempt + 1))
        if payload is not None:
            break

    if payload is None:
        if cache_path.exists():
            print(f"[OSM] warning: live request failed; using cached {cache_path.name}")
            return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])
        raise RuntimeError(f"Overpass request failed for {cache_path.name}: {last_error}") from last_error

    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload.get("elements", [])


def cached_payload_complete(payload: dict, *, max_features: int) -> bool:
    matched = int(payload.get("numberMatched") or payload.get("totalFeatures") or 0)
    returned = int(payload.get("numberReturned") or len(payload.get("features", [])))
    if matched <= 0:
        return returned > 0
    return returned >= min(max(1, int(max_features)), matched)


def fetch_3dbag_layer(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    max_features: int,
    refresh: bool,
    type_name: str,
    cache_label: str,
) -> dict:
    west, south, east, north = ctx.bbox_rd
    base_params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": f"BAG3D:{type_name}",
        "OUTPUTFORMAT": "application/json",
        "SRSNAME": "EPSG:28992",
        "BBOX": f"{west:.3f},{south:.3f},{east:.3f},{north:.3f},EPSG:28992",
    }
    slug = city.cache_slug(lon, lat, radius_m)
    max_total = max(1, int(max_features))
    cache_path = CACHE_DIR / "3dbag" / f"{slug}_{type_name}_count{max_total}.geojson"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached_payload_complete(payload, max_features=max_total):
            return payload
        print(f"[3DBAG] warning: cached {type_name} subset is incomplete; refreshing {cache_path.name}")

    page_size = min(10000, max_total)
    features: list[dict] = []
    matched = 0
    first_payload: dict | None = None
    start_index = 0
    try:
        while len(features) < max_total:
            params = dict(base_params)
            params["COUNT"] = str(min(page_size, max_total - len(features)))
            params["STARTINDEX"] = str(start_index)
            page = fetch_json_url(f"{THREEDBAG_WFS_URL}?{urlencode(params)}", label="3DBAG")
            if first_payload is None:
                first_payload = dict(page)
                matched = int(page.get("numberMatched") or page.get("totalFeatures") or 0)
            page_features = list(page.get("features", []))
            if not page_features:
                break
            features.extend(page_features)
            start_index += len(page_features)
            if matched and len(features) >= matched:
                break
            if len(page_features) < int(params["COUNT"]):
                break
    except RuntimeError:
        if cache_path.exists():
            print(f"[3DBAG] warning: live request failed; using cached {cache_path.name}")
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise

    payload = first_payload or {"type": "FeatureCollection"}
    payload["features"] = features[:max_total]
    payload["numberReturned"] = len(payload["features"])
    if matched:
        payload["numberMatched"] = matched
        payload["totalFeatures"] = matched
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    returned = int(payload.get("numberReturned") or len(payload.get("features", [])))
    if matched > returned:
        print(
            f"[3DBAG] warning: returned {returned}/{matched} {cache_label}; "
            f"increase --max-3dbag-surfaces for the full AOI."
        )
    return payload


def fetch_3dbag_lod22(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    max_features: int,
    refresh: bool,
) -> dict:
    return fetch_3dbag_layer(
        ctx,
        lon=lon,
        lat=lat,
        radius_m=radius_m,
        max_features=max_features,
        refresh=refresh,
        type_name="lod22",
        cache_label="LoD2.2 surfaces",
    )


def fetch_3dbag_lod12(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    max_features: int,
    refresh: bool,
) -> dict:
    return fetch_3dbag_layer(
        ctx,
        lon=lon,
        lat=lat,
        radius_m=radius_m,
        max_features=max_features,
        refresh=refresh,
        type_name="lod12",
        cache_label="LoD1.2 fallback surfaces",
    )


def infer_payload_epsg(payload: dict) -> int | None:
    name = str(payload.get("crs", {}).get("properties", {}).get("name", ""))
    for marker in ("EPSG::", "EPSG:"):
        if marker in name:
            tail = name.rsplit(marker, 1)[-1]
            try:
                return int(tail)
            except ValueError:
                return None
    return None


def localize_metric_geometry(geom, ctx: MetricContext):
    local = affinity.translate(geom, xoff=-ctx.center_xy[0], yoff=-ctx.center_xy[1])
    if abs(float(SCENE_ROTATION_DEG)) > 1e-6:
        local = affinity.rotate(local, SCENE_ROTATION_DEG, origin=(0.0, 0.0))
    return local


def local_aoi_geometry(ctx: MetricContext):
    local = city._extract_polygonal(city._fix_geom(localize_metric_geometry(ctx.aoi_m, ctx)))
    if local is None or local.is_empty:
        return Point(0.0, 0.0).buffer(1.0, resolution=4)
    return local


def localize_feature(feature: dict, center_xy: tuple[float, float]) -> dict:
    """Localize a feature using our scene rotation instead of city module's."""
    geometry = affinity.translate(feature["geometry"], xoff=-center_xy[0], yoff=-center_xy[1])
    if abs(float(SCENE_ROTATION_DEG)) > 1e-6:
        geometry = affinity.rotate(geometry, SCENE_ROTATION_DEG, origin=(0.0, 0.0))
    return {
        "geometry": geometry,
        "tags": dict(feature["tags"]),
    }


def roof_surfaces_from_3dbag(
    payload: dict,
    ctx: MetricContext,
    *,
    source_lod: str = "lod22",
) -> list[RoofSurface]:
    source_epsg = infer_payload_epsg(payload) or 4326
    if source_epsg == ctx.epsg:
        source_to_metric = None
    else:
        source_to_metric = Transformer.from_crs(
            f"EPSG:{source_epsg}", f"EPSG:{ctx.epsg}", always_xy=True
        )

    surfaces: list[RoofSurface] = []
    for feature in payload.get("features", []):
        geom_data = feature.get("geometry")
        if not geom_data:
            continue
        try:
            geom_src = shape(geom_data)
        except Exception:
            continue
        if geom_src.is_empty:
            continue
        geom_m = (
            transform(source_to_metric.transform, geom_src)
            if source_to_metric is not None
            else geom_src
        )
        clipped = city._extract_polygonal(city._fix_geom(geom_m.intersection(ctx.aoi_m)))
        if clipped is None or clipped.is_empty:
            continue

        props = dict(feature.get("properties", {}))
        area = float(clipped.area)
        if area < 2.0:
            continue

        ground_abs = first_float(props, ("b3_h_maaiveld",), 0.0) or 0.0
        h_ref = first_float(
            props,
            ("b3_h_70p", "b3_h_50p", "b3_h_max", "b3_h_min", "b3_h_nok"),
            None,
        )
        if h_ref is None:
            continue
        h_min = first_float(props, ("b3_h_min",), h_ref) or h_ref
        h_max = first_float(props, ("b3_h_max", "b3_h_nok"), h_ref) or h_ref
        if h_min > h_max:
            h_min, h_max = h_max, h_min
        if h_max - ground_abs < 1.0:
            continue

        slope = max(0.0, min(first_float(props, ("b3_hellingshoek",), 0.0) or 0.0, 89.0))
        aspect = (first_float(props, ("b3_azimut",), 180.0) or 180.0) % 360.0
        aspect_rad = math.radians(aspect)
        gradient_xy = np.array(
            [math.sin(aspect_rad), math.cos(aspect_rad)], dtype=np.float32
        ) * math.tan(math.radians(slope))

        local = city._extract_polygonal(city._fix_geom(localize_metric_geometry(clipped, ctx)))
        if local is None or local.is_empty:
            continue
        centroid = local.centroid
        surfaces.append(
            RoofSurface(
                surface_id=str(feature.get("id") or props.get("fid") or len(surfaces)),
                building_id=str(props.get("identificatie") or props.get("fid") or feature.get("id")),
                geometry=local,
                area_m2=area,
                ground_abs_m=float(ground_abs),
                height_ref_abs_m=float(h_ref),
                height_min_abs_m=float(h_min),
                height_max_abs_m=float(h_max),
                slope_deg=float(slope),
                aspect_deg=float(aspect),
                roof_type=str(props.get("b3_dak_type") or "unknown"),
                is_glass_roof=bool(props.get("b3_is_glas_dak", False)),
                quality_ok=bool(props.get("b3_kwaliteitsindicator", True)),
                gradient_xy=gradient_xy,
                centroid_xy=(float(centroid.x), float(centroid.y)),
                source_lod=source_lod,
            )
        )
    return surfaces


def add_lod12_fallback_surfaces(
    lod22_surfaces: list[RoofSurface],
    lod12_surfaces: list[RoofSurface],
) -> list[RoofSurface]:
    represented_buildings = {surface.building_id for surface in lod22_surfaces}
    fallback = [
        surface for surface in lod12_surfaces if surface.building_id not in represented_buildings
    ]
    return lod22_surfaces + fallback


def height_at_surface(surface: RoofSurface, x: float, y: float) -> float:
    delta = np.array([float(x) - surface.centroid_xy[0], float(y) - surface.centroid_xy[1]], dtype=np.float32)
    h_abs = float(surface.height_ref_abs_m + np.dot(delta, surface.gradient_xy))
    h_abs = float(np.clip(h_abs, surface.height_min_abs_m - 0.25, surface.height_max_abs_m + 0.25))
    return max(0.15, h_abs - surface.ground_abs_m)


def unrotated_sun_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    return day.sun_direction_from_angles(float(azimuth_deg), float(elevation_deg))


def rotated_sun_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    # Apply scene rotation to sun direction
    rotated_azimuth = azimuth_deg + SCENE_ROTATION_DEG
    azimuth = math.radians(float(rotated_azimuth))
    elevation = math.radians(float(elevation_deg))
    horizontal = math.cos(elevation)
    return city.normalize(
        np.array(
            [
                -horizontal * math.sin(azimuth),
                math.sin(elevation),
                horizontal * math.cos(azimuth),
            ],
            dtype=np.float32,
        )
    )


def fallback_sun_position(lat: float, lon: float, utc_dt: dt.datetime) -> tuple[float, float]:
    day_of_year = utc_dt.timetuple().tm_yday
    hour = (
        utc_dt.hour
        + utc_dt.minute / 60.0
        + utc_dt.second / 3600.0
        + utc_dt.microsecond / 3_600_000_000.0
    )
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    true_solar_time = (hour * 60.0 + eqtime + 4.0 * float(lon)) % 1440.0
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)
    lat_rad = math.radians(float(lat))
    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    )
    zenith = math.acos(float(np.clip(cos_zenith, -1.0, 1.0)))
    elevation = 90.0 - math.degrees(zenith)
    azimuth = (
        math.degrees(
            math.atan2(
                math.sin(hour_angle),
                math.cos(hour_angle) * math.sin(lat_rad)
                - math.tan(decl) * math.cos(lat_rad),
            )
        )
        + 180.0
    ) % 360.0
    return azimuth, elevation


def local_to_utc(study_date: dt.date, study_time: dt.time) -> dt.datetime:
    local_dt = dt.datetime.combine(study_date, study_time)
    if ZoneInfo is not None:
        local_dt = local_dt.replace(tzinfo=ZoneInfo("Europe/Amsterdam"))
        return local_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return local_dt - dt.timedelta(hours=2)


def sun_angles_for_local_time(
    lat: float,
    lon: float,
    study_date: dt.date,
    study_time: dt.time,
) -> tuple[float, float]:
    utc_dt = local_to_utc(study_date, study_time)
    if f3d is not None and hasattr(f3d, "sun_position"):
        try:
            pos = f3d.sun_position(lat, lon, utc_dt.isoformat(timespec="seconds"))
            return float(pos.azimuth), float(pos.elevation)
        except Exception:
            pass
    return fallback_sun_position(lat, lon, utc_dt)


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * float(t))) for x, y in zip(a, b))


def sun_state_for_local_time(
    lat: float,
    lon: float,
    study_date: dt.date,
    study_time: dt.time,
    *,
    progress: float,
) -> day.SunState:
    azimuth, elevation = sun_angles_for_local_time(lat, lon, study_date, study_time)
    noon_weight = float(np.clip((elevation - 3.0) / 53.0, 0.0, 1.0)) ** 0.75
    dawn_weight = float(np.clip((elevation + 4.0) / 14.0, 0.0, 1.0))
    sky_top = (255, 255, 255)
    sky_bottom = (255, 255, 255)
    if elevation < -1.0:
        sky_top = (255, 255, 255)
        sky_bottom = (255, 255, 255)
    return day.SunState(
        t=float(progress),
        noon_weight=noon_weight,
        azimuth_deg=float(azimuth),
        elevation_deg=float(elevation),
        light_dir=rotated_sun_direction_from_angles(azimuth, elevation),
        shadow_strength=0.72 + 0.34 * (1.0 - noon_weight),
        sky_top_rgb=sky_top,
        sky_bottom_rgb=sky_bottom,
    )


def roof_normal_for_solar(surface: RoofSurface) -> np.ndarray:
    slope = math.radians(surface.slope_deg)
    normal_aspect = math.radians((surface.aspect_deg + 180.0) % 360.0)
    return city.normalize(
        np.array(
            [
                math.sin(normal_aspect) * math.sin(slope),
                math.cos(slope),
                -math.cos(normal_aspect) * math.sin(slope),
            ],
            dtype=np.float32,
        )
    )


def orientation_factor(surface: RoofSurface, assumptions: SolarAssumptions) -> float:
    if surface.slope_deg <= 5.0:
        aspect_factor = 0.94
    else:
        south_alignment = 0.5 + 0.5 * math.cos(math.radians(surface.aspect_deg - 180.0))
        aspect_factor = 1.0 - assumptions.orientation_penalty * (1.0 - south_alignment)
    optimal_slope = 35.0
    slope_delta = min(abs(surface.slope_deg - optimal_slope) / optimal_slope, 1.6)
    slope_factor = 1.0 - assumptions.slope_penalty * slope_delta
    return float(np.clip(aspect_factor * slope_factor, 0.30, 1.04))


def evaluate_roof_surfaces(
    surfaces: list[RoofSurface],
    *,
    assumptions: SolarAssumptions,
    irradiance_kwh_m2: float,
    sun: day.SunState,
) -> None:
    unrotated_light = unrotated_sun_direction_from_angles(sun.azimuth_deg, sun.elevation_deg)
    for surface in surfaces:
        usable_area = surface.area_m2 * assumptions.roof_usable_fraction
        normal = roof_normal_for_solar(surface)
        incidence = max(0.0, float(np.dot(normal, unrotated_light)))
        shade_risk = 1.0 - min(1.0, incidence / 0.42) if sun.elevation_deg > 0.0 else 1.0
        shadow_factor = 1.0 - assumptions.shadow_penalty * shade_risk
        factor = orientation_factor(surface, assumptions) * shadow_factor
        annual_kwh_m2 = float(irradiance_kwh_m2) * factor
        annual_kwh = (
            annual_kwh_m2
            * usable_area
            * assumptions.panel_efficiency
            * assumptions.performance_ratio
        )

        constraint_reason = None
        if usable_area < assumptions.min_usable_roof_area_m2:
            constraint_reason = "small roof surface"
        elif surface.is_glass_roof:
            constraint_reason = "glass roof"
        elif not surface.quality_ok:
            constraint_reason = "3D BAG quality flag"

        if constraint_reason is not None:
            category = "constrained"
        elif annual_kwh_m2 >= SOLAR_HIGH_KWH_M2:
            category = "high"
        elif annual_kwh_m2 >= SOLAR_MEDIUM_KWH_M2:
            category = "medium"
        else:
            category = "low"

        surface.usable_area_m2 = float(usable_area)
        surface.current_incidence = float(incidence)
        surface.annual_kwh_m2 = annual_kwh_m2
        surface.annual_kwh = annual_kwh
        surface.category = category
        surface.constraint_reason = constraint_reason


def summary_from_surfaces(surfaces: list[RoofSurface]) -> StudySummary:
    areas = defaultdict(float)
    for surface in surfaces:
        areas[surface.category] += surface.usable_area_m2
    return StudySummary(
        roof_surfaces=len(surfaces),
        buildings=len({surface.building_id for surface in surfaces}),
        usable_area_m2=sum(surface.usable_area_m2 for surface in surfaces),
        annual_kwh=sum(surface.annual_kwh for surface in surfaces if surface.category != "constrained"),
        high_area_m2=areas["high"],
        medium_area_m2=areas["medium"],
        low_area_m2=areas["low"],
        constrained_area_m2=areas["constrained"],
    )


def make_mesh_layer(
    positions: list[list[float]],
    indices: list[list[int]],
    rgba: tuple[int, int, int, int],
    *,
    shadow_alpha: int,
    specular: float = 0.02,
) -> city.MeshLayer | None:
    if not positions or not indices:
        return None
    return city.MeshLayer(
        positions=np.asarray(positions, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        rgba=rgba,
        shadow_alpha=int(shadow_alpha),
        specular=float(specular),
    )


def add_triangle_mesh(
    bucket: tuple[list[list[float]], list[list[int]]],
    coords: list[tuple[float, float]],
    surface: RoofSurface,
) -> int:
    positions, indices = bucket
    base = len(positions)
    for x, y in coords:
        positions.append([float(x), height_at_surface(surface, x, y), float(-y)])
    indices.append([base, base + 1, base + 2])
    return 1


def build_roof_mesh_layers(surfaces: list[RoofSurface]) -> tuple[list[city.MeshLayer], MeshBuildStats]:
    # Use continuous colormap bins instead of categorical buckets
    roof_buckets: dict[int, tuple[list[list[float]], list[list[int]]]] = {
        i: ([], []) for i in range(COLORMAP_BINS)
    }
    constrained_bucket: tuple[list[list[float]], list[list[int]]] = ([], [])
    wall_positions: list[list[float]] = []
    wall_indices: list[list[int]] = []
    outline_positions: list[list[float]] = []
    outline_indices: list[list[int]] = []

    roof_triangles = 0
    for surface in surfaces:
        if surface.category == "constrained":
            bucket = constrained_bucket
        else:
            bin_idx = solar_bin_index(surface.annual_kwh_m2)
            bucket = roof_buckets[bin_idx]
        for polygon in city._polygon_parts(surface.geometry):
            if polygon.area <= 0.5:
                continue
            for tri in triangulate(polygon):
                if tri.is_empty or not polygon.covers(tri.representative_point()):
                    continue
                coords = [(float(x), float(y)) for x, y in tri.exterior.coords[:-1]]
                if len(coords) != 3:
                    continue
                roof_triangles += add_triangle_mesh(bucket, coords, surface)

    by_building: dict[str, list[RoofSurface]] = defaultdict(list)
    for surface in surfaces:
        by_building[surface.building_id].append(surface)

    for building_surfaces in by_building.values():
        geom = city._extract_polygonal(
            city._fix_geom(unary_union([surface.geometry for surface in building_surfaces]))
        )
        if geom is None or geom.is_empty:
            continue
        top_height = max(
            height_at_surface(surface, *surface.centroid_xy) for surface in building_surfaces
        )
        for polygon in city._polygon_parts(geom):
            coords = list(polygon.exterior.coords)
            for p0, p1 in zip(coords[:-1], coords[1:]):
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                if math.hypot(x1 - x0, y1 - y0) <= 0.4:
                    continue
                base = len(wall_positions)
                wall_positions.extend(
                    [
                        [x0, 0.0, -y0],
                        [x1, 0.0, -y1],
                        [x1, top_height, -y1],
                        [x0, top_height, -y0],
                    ]
                )
                wall_indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

            # Add building outline edges at roof level
            outline_height = top_height + 0.15
            for p0, p1 in zip(coords[:-1], coords[1:]):
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                edge_len = math.hypot(x1 - x0, y1 - y0)
                if edge_len <= 0.5:
                    continue
                # Create thin quad for outline
                dx, dy = (x1 - x0) / edge_len, (y1 - y0) / edge_len
                nx, ny = -dy * 0.35, dx * 0.35  # perpendicular offset for line thickness
                base = len(outline_positions)
                outline_positions.extend([
                    [x0 - nx, outline_height, -(y0 - ny)],
                    [x0 + nx, outline_height, -(y0 + ny)],
                    [x1 + nx, outline_height, -(y1 + ny)],
                    [x1 - nx, outline_height, -(y1 - ny)],
                ])
                outline_indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    layers: list[city.MeshLayer] = []
    wall_layer = make_mesh_layer(
        wall_positions,
        wall_indices,
        ROOF_COLORS["wall"],
        shadow_alpha=22,
        specular=0.0,
    )
    if wall_layer is not None:
        layers.append(wall_layer)

    # Add constrained surfaces first (lowest layer)
    constrained_layer = make_mesh_layer(
        constrained_bucket[0],
        constrained_bucket[1],
        ROOF_COLORS["constrained"],
        shadow_alpha=24,
        specular=0.02,
    )
    if constrained_layer is not None:
        layers.append(constrained_layer)

    # Add continuous colormap layers (low to high solar potential)
    for bin_idx in range(COLORMAP_BINS):
        positions, indices = roof_buckets[bin_idx]
        color = solar_color_for_bin(bin_idx)
        t = bin_idx / max(1, COLORMAP_BINS - 1)
        layer = make_mesh_layer(
            positions,
            indices,
            color,
            shadow_alpha=int(24 + 6 * t),  # brighter roofs get slightly more shadow contrast
            specular=0.02 + 0.04 * t,
        )
        if layer is not None:
            layers.append(layer)

    # Add building outlines on top
    outline_layer = make_mesh_layer(
        outline_positions,
        outline_indices,
        ROOF_COLORS["outline"],
        shadow_alpha=0,
        specular=0.0,
    )
    if outline_layer is not None:
        layers.append(outline_layer)

    return layers, MeshBuildStats(
        roof_triangles=roof_triangles,
        wall_triangles=len(wall_indices),
        roof_surfaces=len(surfaces),
        buildings=len(by_building),
    )


def fetch_pvgis_irradiance(
    lat: float,
    lon: float,
    *,
    refresh: bool,
    skip: bool,
) -> IrradianceReference:
    if skip:
        return IrradianceReference(
            annual_kwh_m2=FALLBACK_IRRADIANCE_KWH_M2_YEAR,
            source="fallback",
            detail="documented Rotterdam annual global irradiation fallback",
        )
    params = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "peakpower": "1",
        "loss": "14",
        "angle": "35",
        "aspect": "0",
        "outputformat": "json",
    }
    cache_path = CACHE_DIR / "pvgis" / f"pvgis_lat{lat:.4f}_lon{lon:.4f}.json"
    last_error: Exception | None = None
    for endpoint in PVGIS_ENDPOINTS:
        try:
            payload = request_json(
                f"{endpoint}?{urlencode(params)}",
                cache_path,
                refresh=refresh,
                label="PVGIS",
            )
            fixed = payload.get("outputs", {}).get("totals", {}).get("fixed", {})
            value = fixed.get("H(i)_y") or fixed.get("H(i)_m")
            annual = float(value)
            if "H(i)_m" in fixed and "H(i)_y" not in fixed:
                annual *= 12.0
            if math.isfinite(annual) and annual > 100.0:
                return IrradianceReference(
                    annual_kwh_m2=annual,
                    source="PVGIS/JRC",
                    detail="PVcalc H(i)_y at 35 degree south-facing reference plane",
                )
        except Exception as exc:
            last_error = exc
            continue
    print(f"[PVGIS] warning: {last_error}; using fallback irradiation")
    return IrradianceReference(
        annual_kwh_m2=FALLBACK_IRRADIANCE_KWH_M2_YEAR,
        source="fallback",
        detail="PVGIS unavailable; documented Rotterdam fallback",
    )


def overpass_query_many(
    bbox: tuple[float, float, float, float],
    filter_exprs: tuple[str, ...],
    *,
    include_relations: bool,
) -> str:
    west, south, east, north = bbox
    parts: list[str] = []
    for expr in filter_exprs:
        parts.append(f"way{expr}({south},{west},{north},{east});")
        if include_relations:
            parts.append(f"relation{expr}({south},{west},{north},{east});")
    return f"[out:json][timeout:60];({''.join(parts)});out geom qt;"


def osm_way_is_closed(element: dict) -> bool:
    geom = element.get("geometry") or []
    if len(geom) < 4:
        return False
    first = geom[0]
    last = geom[-1]
    return first.get("lon") == last.get("lon") and first.get("lat") == last.get("lat")


def osm_area_elements(elements: list[dict], predicate) -> list[dict]:
    selected = []
    for element in elements:
        tags = dict(element.get("tags", {}))
        if not predicate(tags):
            continue
        if element.get("type") == "relation" or tags.get("area") == "yes" or osm_way_is_closed(element):
            selected.append(element)
    return selected


def is_paved_highway_area(tags: dict) -> bool:
    highway = tags.get("highway")
    return bool(highway) and (tags.get("area") == "yes" or highway in {"pedestrian", "platform"})


def is_parking_area(tags: dict) -> bool:
    return tags.get("amenity") in {
        "parking",
        "parking_space",
        "bicycle_parking",
        "bus_station",
        "marketplace",
        "ferry_terminal",
    }


def is_transport_platform_area(tags: dict) -> bool:
    return (
        tags.get("railway") == "platform"
        or tags.get("highway") == "platform"
        or tags.get("public_transport") in {"platform", "station"}
    )


def is_hardscape_area(tags: dict) -> bool:
    return tags.get("man_made") in {"pier", "bridge"} or tags.get("place") == "square"


def is_leisure_open_area(tags: dict) -> bool:
    return tags.get("leisure") in {
        "garden",
        "playground",
        "pitch",
        "sports_centre",
        "recreation_ground",
        "common",
    }


def build_osm_context_surfaces(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    refresh_osm: bool,
    skip_osm: bool,
    osm_timeout_seconds: float,
) -> list[city.SurfaceLayer]:
    local_aoi = local_aoi_geometry(ctx)
    surfaces = [
        city.SurfaceLayer(
            geometry=local_aoi,
            rgba=CONTEXT_COLORS["base"],
            elevation=0.02,
        )
    ]
    if skip_osm:
        return surfaces

    slug = city.cache_slug(lon, lat, radius_m)

    def cached(name: str, expr: str, *, include_relations: bool) -> list[dict]:
        query = city.overpass_query(ctx.bbox_wgs84, expr, include_relations=include_relations)
        cache_path = CACHE_DIR / "osm" / f"{slug}_{name}.json"
        try:
            return fetch_overpass_query_cached(
                query,
                cache_path,
                refresh_osm,
                timeout_seconds=osm_timeout_seconds,
            )
        except RuntimeError as exc:
            print(f"[OSM] warning: {exc}")
            return []

    def cached_many(name: str, exprs: tuple[str, ...], *, include_relations: bool) -> list[dict]:
        query = overpass_query_many(ctx.bbox_wgs84, exprs, include_relations=include_relations)
        cache_path = CACHE_DIR / "osm" / f"{slug}_{name}.json"
        try:
            return fetch_overpass_query_cached(
                query,
                cache_path,
                refresh_osm,
                timeout_seconds=osm_timeout_seconds,
            )
        except RuntimeError as exc:
            print(f"[OSM] warning: {exc}")
            return []

    def append_surface(geom, color_key: str, elevation: float, *, specular: float = 0.0, reflectivity: float = 0.0) -> None:
        polygonal = city._extract_polygonal(city._fix_geom(geom.intersection(local_aoi)))
        if polygonal is None or polygonal.is_empty:
            return
        surfaces.append(
            city.SurfaceLayer(
                geometry=polygonal,
                rgba=CONTEXT_COLORS[color_key],
                elevation=float(elevation),
                specular=float(specular),
                reflectivity=float(reflectivity),
            )
        )

    landuse_elements = cached("landuse", '["landuse"]', include_relations=True)
    park_elements = cached("parks_ways", '["leisure"="park"]', include_relations=False)
    natural_green_elements = cached(
        "natural_green_ways",
        '["natural"~"^(wood|scrub)$"]',
        include_relations=False,
    )
    water_elements = (
        cached("water_ways", '["natural"="water"]', include_relations=False)
        + cached("river_water_ways", '["water"="river"]', include_relations=False)
        + cached("riverbank_ways", '["waterway"="riverbank"]', include_relations=False)
    )
    waterway_elements = cached(
        "waterway_lines",
        '["waterway"~"^(river|canal)$"]',
        include_relations=False,
    )
    roads_elements = cached("roads", '["highway"]', include_relations=False)
    rails_elements = cached("rails", '["railway"]', include_relations=False)
    hardscape_elements = cached_many(
        "hardscape_areas",
        (
            '["amenity"~"^(parking|parking_space|bicycle_parking|bus_station|marketplace|ferry_terminal)$"]',
            '["public_transport"~"^(platform|station)$"]',
            '["railway"="platform"]',
            '["man_made"~"^(pier|bridge)$"]',
            '["place"="square"]',
        ),
        include_relations=False,
    )
    leisure_elements = cached_many(
        "leisure_open_areas",
        ('["leisure"~"^(garden|playground|pitch|sports_centre|recreation_ground|common)$"]',),
        include_relations=False,
    )

    landuse = city.parse_polygon_features(
        landuse_elements,
        ctx.to_metric,
        ctx.aoi_m,
    )
    parks = city.parse_polygon_features(
        park_elements
        + natural_green_elements
        + osm_area_elements(leisure_elements, is_leisure_open_area),
        ctx.to_metric,
        ctx.aoi_m,
    )
    water = city.parse_polygon_features(
        water_elements,
        ctx.to_metric,
        ctx.aoi_m,
    )
    water_lines = city.parse_line_features(
        waterway_elements,
        ctx.to_metric,
        ctx.aoi_m,
    )
    paved = city.parse_polygon_features(
        osm_area_elements(roads_elements, is_paved_highway_area)
        + osm_area_elements(hardscape_elements, is_hardscape_area),
        ctx.to_metric,
        ctx.aoi_m,
    )
    parking = city.parse_polygon_features(
        osm_area_elements(hardscape_elements, is_parking_area),
        ctx.to_metric,
        ctx.aoi_m,
    )
    platforms = city.parse_polygon_features(
        osm_area_elements(rails_elements, is_transport_platform_area)
        + osm_area_elements(hardscape_elements, is_transport_platform_area),
        ctx.to_metric,
        ctx.aoi_m,
    )
    roads = city.parse_line_features(
        roads_elements,
        ctx.to_metric,
        ctx.aoi_m,
    )
    rails = city.parse_line_features(
        rails_elements,
        ctx.to_metric,
        ctx.aoi_m,
    )

    landuse_local = [localize_feature(feature, ctx.center_xy) for feature in landuse]
    parks_local = [localize_feature(feature, ctx.center_xy) for feature in parks]
    water_local = [localize_feature(feature, ctx.center_xy) for feature in water]
    water_line_local = [localize_feature(feature, ctx.center_xy) for feature in water_lines]
    paved_local = [localize_feature(feature, ctx.center_xy) for feature in paved]
    parking_local = [localize_feature(feature, ctx.center_xy) for feature in parking]
    platform_local = [localize_feature(feature, ctx.center_xy) for feature in platforms]
    roads_local = [localize_feature(feature, ctx.center_xy) for feature in roads]
    rails_local = [localize_feature(feature, ctx.center_xy) for feature in rails]

    if landuse_local:
        append_surface(city.merge_surface_geometry(landuse_local, simplify_tolerance=1.4), "landuse", 0.34)
    if paved_local:
        append_surface(city.merge_surface_geometry(paved_local, simplify_tolerance=0.55), "paved", 0.46)
    if parking_local:
        append_surface(city.merge_surface_geometry(parking_local, simplify_tolerance=0.55), "parking", 0.48)
    if platform_local:
        append_surface(city.merge_surface_geometry(platform_local, simplify_tolerance=0.45), "platform", 0.50)
    if parks_local:
        append_surface(city.merge_surface_geometry(parks_local, simplify_tolerance=1.1), "park", 0.40)
    if water_line_local:
        river_lines = [
            feature
            for feature in water_line_local
            if feature["tags"].get("waterway") == "river"
        ]
        canal_lines = [
            feature
            for feature in water_line_local
            if feature["tags"].get("waterway") != "river"
        ]
        water_line_buffers = []
        for features, buffer_m, tolerance in (
            (river_lines, 125.0, 1.2),
            (canal_lines, 18.0, 0.8),
        ):
            if not features:
                continue
            line_union = unary_union([feature["geometry"] for feature in features])
            water_line_buffers.append(
                city.simplify_geom(
                    line_union.buffer(
                        buffer_m,
                        cap_style=BufferCapStyle.round,
                        join_style=BufferJoinStyle.round,
                    ),
                    tolerance,
                )
            )
        if water_line_buffers:
            append_surface(
                unary_union(water_line_buffers),
                "water",
                -0.36,
                specular=0.20,
                reflectivity=0.22,
            )
    if water_local:
        append_surface(
            city.merge_surface_geometry(water_local, simplify_tolerance=0.8),
            "water",
            -0.35,
            specular=0.20,
            reflectivity=0.22,
        )
    if roads_local:
        roads_union = unary_union([feature["geometry"] for feature in roads_local])
        road_buffer = city.simplify_geom(
            roads_union.buffer(3.0, cap_style=BufferCapStyle.round, join_style=BufferJoinStyle.round),
            0.7,
        )
        road_crown = city.simplify_geom(
            roads_union.buffer(1.25, cap_style=BufferCapStyle.round, join_style=BufferJoinStyle.round),
            0.45,
        )
        append_surface(road_buffer, "road", 0.72)
        append_surface(road_crown, "road_hi", 0.82)
    if rails_local:
        rails_union = unary_union([feature["geometry"] for feature in rails_local])
        rails_buffer = city.simplify_geom(
            rails_union.buffer(1.7, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre),
            0.55,
        )
        append_surface(rails_buffer, "road", 0.78)

    print(
        f"[OSM] context landuse={len(landuse_local)} parks={len(parks_local)} "
        f"water_polygons={len(water_local)} water_lines={len(water_line_local)} "
        f"paved={len(paved_local)} parking={len(parking_local)} platforms={len(platform_local)} "
        f"roads={len(roads_local)} rails={len(rails_local)}"
    )
    return surfaces


def project_label_point(
    prepared: day.PreparedScene,
    ctx: MetricContext,
    lon: float,
    lat: float,
    *,
    elevation: float = 12.0,
) -> tuple[float, float] | None:
    metric = transform(ctx.to_metric.transform, Point(lon, lat))
    local = localize_metric_geometry(metric, ctx)
    x, y = local.coords[0]
    projected = project_local_points(prepared, [(float(x), float(y))], elevation=elevation)
    return projected[0] if projected else None


def project_local_points(
    prepared: day.PreparedScene,
    coords_xy: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    elevation: float = 0.0,
) -> list[tuple[float, float]]:
    if not coords_xy:
        return []
    world = np.array(
        [[float(x), float(elevation), float(-y)] for x, y in coords_xy],
        dtype=np.float32,
    )
    projected = city.project_points(
        world,
        eye=prepared.eye,
        target=prepared.target,
        up=prepared.up,
        width=prepared.render_width,
        height=prepared.render_height,
        fov_deg=prepared.fov_deg,
    )
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 1.0):
        return []
    points = projected[:, :2].astype(np.float32).copy()
    points[:, 0] = points[:, 0] * prepared.scale + prepared.offset_x
    points[:, 1] = points[:, 1] * prepared.scale + prepared.offset_y
    points /= float(prepared.supersample)
    return [(float(point[0]), float(point[1])) for point in points]


def text_bbox_at(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[float, float, float, float]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox(xy, text, font=font)
        return (float(left), float(top), float(right), float(bottom))
    width = float(draw.textlength(text, font=font))
    return (float(xy[0]), float(xy[1]), float(xy[0] + width), float(xy[1] + font.size))


def boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def expand_box(box: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)


def draw_text_halo(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int],
    halo: tuple[int, int, int, int] = (20, 24, 28, 210),
) -> None:
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=halo)
    draw.text((x, y), text, font=font, fill=fill)


def add_labels(image: Image.Image, prepared: day.PreparedScene, ctx: MetricContext) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = max(0.82, canvas.width / 1100.0)
    # Smaller, subtler font for labels
    font = load_font(max(11, int(round(12 * scale))))
    occupied: list[tuple[float, float, float, float]] = [
        (0.0, canvas.height - 58.0 * scale, 230.0 * scale, float(canvas.height)),
        (canvas.width - 190.0 * scale, 0.0, float(canvas.width), 200.0 * scale),
    ]
    for text, lon, lat in LABEL_POINTS:
        point = project_label_point(prepared, ctx, lon, lat, elevation=16.0)
        if point is None:
            continue
        x, y = point
        if not (-40 <= x <= canvas.width + 40 and -40 <= y <= canvas.height + 40):
            continue
        dot_r = max(2, int(round(2.5 * scale)))
        raw_box = text_bbox_at(draw, (0.0, 0.0), text, font)
        text_w = raw_box[2] - raw_box[0]
        text_h = raw_box[3] - raw_box[1]
        offsets = (
            (10.0 * scale, -20.0 * scale),
            (10.0 * scale, 8.0 * scale),
            (-text_w - 10.0 * scale, -20.0 * scale),
            (-text_w - 10.0 * scale, 8.0 * scale),
            (-0.5 * text_w, -28.0 * scale),
            (-0.5 * text_w, 14.0 * scale),
        )
        tx, ty = x + offsets[0][0], y + offsets[0][1]
        chosen_box = expand_box(text_bbox_at(draw, (tx, ty), text, font), 3.0 * scale)
        for dx, dy in offsets:
            candidate_xy = (x + dx, y + dy)
            candidate_box = expand_box(text_bbox_at(draw, candidate_xy, text, font), 3.0 * scale)
            in_bounds = (
                candidate_box[0] >= 4.0
                and candidate_box[1] >= 4.0
                and candidate_box[2] <= canvas.width - 4.0
                and candidate_box[3] <= canvas.height - 4.0
            )
            if in_bounds and not any(boxes_intersect(candidate_box, other) for other in occupied):
                tx, ty = candidate_xy
                chosen_box = candidate_box
                break
        leader_end = (tx, ty + text_h * 0.55)
        # Subtler leader line
        draw.line((x, y, leader_end[0], leader_end[1]), fill=(40, 48, 55, 90), width=max(1, int(scale * 0.8)))
        draw.ellipse((x - dot_r, y - dot_r, x + dot_r, y + dot_r), fill=(50, 58, 65, 180))
        # Subtler text color
        draw_text_halo(draw, (tx, ty), text, font, fill=(230, 232, 225, 200), halo=(20, 28, 34, 140))
        occupied.append(chosen_box)
    return canvas


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    halo: tuple[int, int, int, int] = (15, 21, 26, 180),
) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    head_len = max(7.0, width * 3.2)
    head_angle = math.radians(30.0)
    head = [
        (ex, ey),
        (ex - head_len * math.cos(angle - head_angle), ey - head_len * math.sin(angle - head_angle)),
        (ex - head_len * math.cos(angle + head_angle), ey - head_len * math.sin(angle + head_angle)),
    ]
    draw.line((sx, sy, ex, ey), fill=halo, width=width + 3)
    draw.polygon(head, fill=halo)
    draw.line((sx, sy, ex, ey), fill=fill, width=width)
    draw.polygon(head, fill=fill)


def screen_direction_for_local_vector(
    prepared: day.PreparedScene,
    vector_xy: tuple[float, float],
    *,
    length_m: float = 280.0,
) -> np.ndarray:
    vx, vy = vector_xy
    norm = math.hypot(vx, vy)
    if norm <= 1e-6:
        return np.array([0.0, -1.0], dtype=np.float32)
    points = project_local_points(
        prepared,
        [(0.0, 0.0), (length_m * vx / norm, length_m * vy / norm)],
        elevation=0.2,
    )
    if len(points) != 2:
        return np.array([0.0, -1.0], dtype=np.float32)
    direction = np.asarray(points[1], dtype=np.float32) - np.asarray(points[0], dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return np.array([0.0, -1.0], dtype=np.float32)
    return direction / direction_norm


def add_aoi_boundary(image: Image.Image, prepared: day.PreparedScene, ctx: MetricContext) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width = max(2, int(round(canvas.width / 560.0)))
    for polygon in city._polygon_parts(local_aoi_geometry(ctx)):
        coords = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        points = project_local_points(prepared, coords, elevation=0.6)
        if len(points) < 3:
            continue
        closed = points + [points[0]]
        draw.line(closed, fill=(16, 25, 30, 145), width=width + 2, joint="curve")
        draw.line(closed, fill=(250, 252, 246, 185), width=width, joint="curve")
    return canvas


def add_scale_bar(canvas: Image.Image, prepared: day.PreparedScene) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = max(0.82, canvas.width / 1100.0)
    probe_m = 500.0
    probe = project_local_points(prepared, [(0.0, 0.0), (probe_m, 0.0)], elevation=0.2)
    if len(probe) == 2:
        px_per_m = math.dist(probe[0], probe[1]) / probe_m
    else:
        px_per_m = prepared.world_to_px * prepared.scale / float(prepared.supersample)
    px_per_m = max(px_per_m, 0.01)
    target_px = min(canvas.width * 0.20, 250.0 * scale)
    candidates = (50, 100, 200, 500, 1000, 2000)
    length_m = min(candidates, key=lambda value: abs(value * px_per_m - target_px))
    bar_px = length_m * px_per_m
    x0 = 28.0 * scale
    y0 = canvas.height - 45.0 * scale
    tick = 12.0 * scale
    line_w = max(5, int(round(5 * scale)))
    draw.line((x0, y0, x0 + bar_px, y0), fill=(250, 252, 246, 230), width=line_w + 4)
    draw.line((x0, y0, x0 + bar_px, y0), fill=(18, 27, 32, 235), width=line_w)
    for x in (x0, x0 + bar_px):
        draw.line((x, y0 - tick, x, y0 + tick), fill=(18, 27, 32, 235), width=line_w)
    label = f"{int(length_m / 1000)} km" if length_m >= 1000 else f"{int(length_m)} m"
    font = load_font(max(16, int(round(20 * scale))))
    draw_text_halo(draw, (x0, y0 - 40.0 * scale), label, font, fill=(20, 28, 32, 245), halo=(250, 252, 246, 220))


def add_direction_cues(canvas: Image.Image, prepared: day.PreparedScene, sun: day.SunState) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = max(0.82, canvas.width / 1100.0)
    font = load_font(max(20, int(round(26 * scale))))
    small_font = load_font(max(16, int(round(20 * scale))))
    north_dir = screen_direction_for_local_vector(prepared, (0.0, 1.0))
    sun_dir = screen_direction_for_local_vector(prepared, (float(sun.light_dir[0]), float(-sun.light_dir[2])))

    # North arrow - black lines without background
    north_center = np.array([canvas.width - 100.0 * scale, 90.0 * scale], dtype=np.float32)
    north_len = 42.0 * scale

    # Draw north arrow (black lines)
    arrow_start = tuple(north_center - north_dir * north_len * 0.6)
    arrow_end = tuple(north_center + north_dir * north_len * 0.6)
    draw_arrow(
        draw,
        arrow_start,
        arrow_end,
        fill=(20, 20, 20, 255),
        width=max(4, int(round(4.5 * scale))),
    )

    # Draw N label (black)
    n_pos = tuple(north_center + north_dir * north_len * 1.0 + np.array([-10.0 * scale, -12.0 * scale]))
    draw.text(n_pos, "N", font=font, fill=(20, 20, 20, 255))



def find_top_roof_clusters(
    surfaces: list[RoofSurface],
    n_clusters: int = 5,
    min_area: float = 500.0,
) -> list[tuple[float, float, float, int]]:
    """Find top N high-potential roof clusters by annual kWh. Returns (x, y, kwh, rank)."""
    # Filter to high-potential, large surfaces
    candidates = [
        s for s in surfaces
        if s.category != "constrained" and s.annual_kwh_m2 >= SOLAR_HIGH_KWH_M2 and s.usable_area_m2 >= min_area
    ]
    # Sort by annual production
    candidates.sort(key=lambda s: s.annual_kwh, reverse=True)

    # Take top N, avoiding overlapping clusters (min 50m apart)
    results = []
    for surface in candidates:
        cx, cy = surface.centroid_xy
        too_close = any(
            math.hypot(cx - rx, cy - ry) < 80.0
            for rx, ry, _, _ in results
        )
        if not too_close:
            results.append((cx, cy, surface.annual_kwh, len(results) + 1))
            if len(results) >= n_clusters:
                break
    return results


def add_best_roof_markers(
    canvas: Image.Image,
    prepared: day.PreparedScene,
    top_roofs: list[tuple[float, float, float, int]],
) -> Image.Image:
    """Add numbered markers for top roof clusters."""
    if not top_roofs:
        return canvas

    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = max(0.82, canvas.width / 1100.0)
    font = load_font(max(12, int(round(14 * scale))))

    for x, y, kwh, rank in top_roofs:
        projected = project_local_points(prepared, [(x, y)], elevation=25.0)
        if not projected:
            continue
        px, py = projected[0]
        if not (10 <= px <= canvas.width - 10 and 10 <= py <= canvas.height - 10):
            continue

        # Draw marker circle
        r = max(12, int(round(14 * scale)))
        draw.ellipse(
            (px - r, py - r, px + r, py + r),
            fill=(255, 140, 60, 230),
            outline=(255, 255, 255, 200),
            width=max(2, int(round(2 * scale))),
        )
        # Draw number
        label = str(rank)
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), label, font=font)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            lw = int(draw.textlength(label, font=font))
            lh = 12
        draw.text((px - lw // 2, py - lh // 2 - 1), label, font=font, fill=(255, 255, 255, 255))

    return canvas


def add_cartographic_overlays(
    image: Image.Image,
    prepared: day.PreparedScene,
    ctx: MetricContext,
    sun: day.SunState,
    *,
    top_roofs: list[tuple[float, float, float, int]] | None = None,
) -> Image.Image:
    canvas = image.convert("RGBA")
    canvas = add_labels(canvas, prepared, ctx)
    if top_roofs:
        canvas = add_best_roof_markers(canvas, prepared, top_roofs)
    add_scale_bar(canvas, prepared)
    add_direction_cues(canvas, prepared, sun)
    return canvas


def compute_output_layout(width: int, height: int) -> OutputLayout:
    width = max(320, int(width))
    height = max(220, int(height))
    if width >= 900:
        # Wider panel for 4K to fit larger text
        panel_width = int(np.clip(round(width * 0.24), 320, 1400))
        return OutputLayout(
            final_width=width,
            final_height=height,
            map_width=width - panel_width,
            map_height=height,
            panel_width=panel_width,
            panel_height=height,
            panel_side="left",
        )

    panel_height = int(np.clip(round(height * 0.38), 118, 170))
    return OutputLayout(
        final_width=width,
        final_height=height,
        map_width=width,
        map_height=height - panel_height,
        panel_width=width,
        panel_height=panel_height,
        panel_side="bottom",
    )


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0])
    return int(round(draw.textlength(text, font=font)))


def fit_panel_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    target_size_px: int,
    max_width_px: int,
    min_size_px: int,
) -> ImageFont.ImageFont:
    for size_px in range(max(1, int(target_size_px)), max(0, int(min_size_px)) - 1, -1):
        font = load_font(size_px)
        if text_width(draw, text, font) <= max_width_px:
            return font
    return load_font(min_size_px)


def draw_legend_swatch(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    category: str,
) -> None:
    draw.rectangle(box, fill=ROOF_COLORS[category])
    draw.rectangle(box, outline=(245, 248, 240, 130), width=1)


def draw_gradient_bar(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    """Draw a horizontal gradient bar with tick marks (high on left, low on right)."""
    # Draw gradient bar - reversed: high (left) to low (right)
    for i in range(width):
        t = 1.0 - (i / max(1, width - 1))  # Reversed
        color = solar_to_color(SOLAR_MIN_KWH_M2 + t * (SOLAR_MAX_KWH_M2 - SOLAR_MIN_KWH_M2))
        draw.line((x + i, y, x + i, y + height), fill=color, width=1)

    # Draw border
    draw.rectangle((x, y, x + width, y + height), outline=(245, 248, 240, 130), width=1)

    # Draw tick marks and labels at class breaks (reversed order)
    tick_values = [1000, int(SOLAR_HIGH_KWH_M2), int(SOLAR_MEDIUM_KWH_M2), int(SOLAR_LOW_KWH_M2), 0]
    tick_labels = ["High", "", "Med", "", "Low"]
    tick_height = 5
    for val, lbl in zip(tick_values, tick_labels):
        t = 1.0 - ((val - SOLAR_MIN_KWH_M2) / (SOLAR_MAX_KWH_M2 - SOLAR_MIN_KWH_M2))  # Reversed
        tx = x + int(t * width)
        # Draw tick mark
        draw.line((tx, y + height, tx, y + height + tick_height), fill=(200, 208, 204, 200), width=1)
        # Draw label
        if not lbl:
            continue
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), lbl, font=small_font)
            lw = bbox[2] - bbox[0]
        else:
            lw = int(draw.textlength(lbl, font=small_font))
        label_x = tx - lw // 2
        if val == 1000:
            label_x = max(x, label_x)
        elif val == 0:
            label_x = min(x + width - lw, label_x)
        draw.text((label_x, y + height + tick_height + 2), lbl, font=small_font, fill=(0, 0, 0, 255))

    return y + height + tick_height + 18  # Return y position after the legend


def make_legend_panel(
    width: int,
    height: int,
    *,
    summary: StudySummary,
    assumptions: SolarAssumptions,
    irradiance: IrradianceReference,
    sun: day.SunState,
    study_date: dt.date,
    study_time: dt.time,
) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    # Scale based on panel width for 4K support
    scale = max(1.0, width / 310.0)
    margin = max(16, int(round(22 * scale)))

    # Fonts - sized for 4K readability
    title_font = load_font(max(24, int(round(30 * scale))))
    headline_font = load_font(max(34, int(round(42 * scale))))
    body_font = load_font(max(18, int(round(22 * scale))))
    small_font = load_font(max(16, int(round(19 * scale))))
    tiny_font = load_font(max(14, int(round(16 * scale))))

    tx = margin
    y = margin

    # Title
    draw.text((tx, y), "ROTTERDAM", font=title_font, fill=(0, 0, 0, 255))
    y += max(38, int(round(42 * scale)))
    draw.text((tx, y), "Rooftop Solar Potential", font=body_font, fill=(0, 0, 0, 255))
    y += max(36, int(round(40 * scale)))

    # Key number in GWh/yr - prominent
    annual_gwh = summary.annual_kwh / 1_000_000.0
    annual_label = f"{annual_gwh:.1f} GWh/yr"
    draw.text(
        (tx, y),
        annual_label,
        font=headline_font,
        fill=(180, 100, 0, 255),
    )
    y += max(52, int(round(58 * scale)))
    draw.text((tx, y), "estimated annual generation", font=small_font, fill=(0, 0, 0, 255))
    y += max(36, int(round(40 * scale)))

    # Divider
    draw.line((tx, y, width - margin, y), fill=(0, 0, 0, 60), width=max(1, int(scale)))
    y += max(28, int(round(32 * scale)))

    # Solar Potential legend
    draw.text((tx, y), "SOLAR POTENTIAL (kWh/m\u00b2/year)", font=small_font, fill=(0, 0, 0, 255))
    y += max(32, int(round(36 * scale)))

    # Gradient bar
    bar_width = min(width - margin * 2 - 10, max(160, int(220 * scale)))
    bar_height = max(28, int(round(32 * scale)))
    y = draw_gradient_bar(draw, canvas, tx, y, bar_width, bar_height, font=body_font, small_font=tiny_font)
    y += max(20, int(round(24 * scale)))

    # Legend swatches
    sw = max(28, int(round(32 * scale)))
    row_h = max(38, int(round(42 * scale)))

    # Excluded swatch
    ly = y + max(4, int(round(5 * scale)))
    draw_legend_swatch(draw, (tx, ly, tx + sw, ly + sw), "constrained")
    draw.text((tx + sw + 18, y), "Excluded", font=body_font, fill=(0, 0, 0, 255))
    y += row_h

    # Shadow swatch - matches actual shadow appearance on map
    time_label = study_time.strftime("%H:%M")
    ly = y + max(4, int(round(5 * scale)))
    draw.rectangle((tx, ly, tx + sw, ly + sw), fill=SHADOW_LEGEND_COLOR)
    draw.rectangle((tx, ly, tx + sw, ly + sw), outline=(0, 0, 0, 150), width=1)
    draw.text((tx + sw + 18, y), f"Shadow {time_label}", font=body_font, fill=(0, 0, 0, 255))
    y += row_h + max(12, int(round(16 * scale)))

    # Footer
    if y < height - margin - 36:
        footer_y = height - margin - max(16, int(round(18 * scale)))
        draw.text(
            (tx, footer_y),
            "3D BAG LoD2.2 | Planning estimate",
            font=tiny_font,
            fill=(80, 80, 80, 200),
        )
    return canvas


def compose_study_frame(
    map_image: Image.Image,
    layout: OutputLayout,
    *,
    summary: StudySummary,
    assumptions: SolarAssumptions,
    irradiance: IrradianceReference,
    sun: day.SunState,
    study_date: dt.date,
    study_time: dt.time,
) -> Image.Image:
    panel = make_legend_panel(
        layout.panel_width,
        layout.panel_height,
        summary=summary,
        assumptions=assumptions,
        irradiance=irradiance,
        sun=sun,
        study_date=study_date,
        study_time=study_time,
    )
    canvas = Image.new(
        "RGBA",
        (layout.final_width, layout.final_height),
        (12, 20, 24, 255),
    )
    if layout.panel_side == "left":
        canvas.alpha_composite(panel, dest=(0, 0))
        canvas.alpha_composite(map_image.convert("RGBA"), dest=(layout.panel_width, 0))
    else:
        canvas.alpha_composite(map_image.convert("RGBA"), dest=(0, 0))
        canvas.alpha_composite(panel, dest=(0, layout.map_height))
    return canvas


def render_study_frame(
    prepared: day.PreparedScene,
    ctx: MetricContext,
    layout: OutputLayout,
    *,
    summary: StudySummary,
    assumptions: SolarAssumptions,
    irradiance: IrradianceReference,
    sun: day.SunState,
    study_date: dt.date,
    study_time: dt.time,
    surfaces: list[RoofSurface] | None = None,
) -> Image.Image:
    map_image = day.render_frame(
        prepared,
        sun,
        frame_index=0,
        total_frames=1,
        clock_start_hour=float(study_time.hour),
        clock_end_hour=float(study_time.hour),
        show_timer=False,
        shadow_tint_rgb=STUDY_SHADOW_TINT_RGB,
        shadow_opacity=STUDY_SHADOW_OPACITY,
    )
    # No roof markers - let the solar colors speak for themselves
    map_image = add_cartographic_overlays(map_image, prepared, ctx, sun, top_roofs=None)
    return compose_study_frame(
        map_image,
        layout,
        summary=summary,
        assumptions=assumptions,
        irradiance=irradiance,
        sun=sun,
        study_date=study_date,
        study_time=study_time,
    )


def animation_time_for_frame(study_date: dt.date, frame_index: int, total_frames: int) -> dt.time:
    t = frame_index / max(total_frames - 1, 1)
    hour = 5.0 + 16.0 * t
    total_seconds = int(round(hour * 3600.0))
    return (dt.datetime.combine(study_date, dt.time(0, 0)) + dt.timedelta(seconds=total_seconds)).time()


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to assemble the MP4 output.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ],
        check=True,
    )


def choose_output_path(args: argparse.Namespace) -> Path:
    output = args.output
    if args.animate and output == DEFAULT_STATIC_OUTPUT:
        output = DEFAULT_ANIMATION_OUTPUT
    return output.resolve()


def main() -> int:
    args = parse_args()
    study_date = parse_study_date(args.date)
    study_time = parse_clock_time(args.time)
    width, height = map(int, args.size)
    layout = compute_output_layout(width, height)
    output_path = choose_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assumptions = SolarAssumptions(
        panel_efficiency=float(args.panel_efficiency),
        performance_ratio=float(args.performance_ratio),
        min_usable_roof_area_m2=float(args.min_roof_area),
        orientation_penalty=float(args.orientation_penalty),
        slope_penalty=float(args.slope_penalty),
        shadow_penalty=float(args.shadow_penalty),
    )

    print(
        f"[Rotterdam] center=({args.lon:.5f}, {args.lat:.5f}) "
        f"radius={float(args.radius):.0f}m date={study_date.isoformat()} time={study_time.strftime('%H:%M')}"
    )
    ctx = build_metric_context(
        float(args.lon),
        float(args.lat),
        float(args.radius),
        aoi_shape="rectangle",
        aspect_ratio=layout.map_width / max(layout.map_height, 1),
    )
    building_payload = fetch_3dbag_lod22(
        ctx,
        lon=float(args.lon),
        lat=float(args.lat),
        radius_m=float(args.radius),
        max_features=int(args.max_3dbag_surfaces),
        refresh=bool(args.refresh_buildings),
    )
    surfaces = roof_surfaces_from_3dbag(building_payload, ctx, source_lod="lod22")
    if not surfaces:
        raise SystemExit(
            "No 3D BAG LoD2.2 roof surfaces were loaded for the AOI. "
            "Check network/cache availability or reduce --radius."
        )
    lod22_building_count = len({surface.building_id for surface in surfaces})
    lod12_added = 0
    if not bool(args.no_lod12_fallback):
        fallback_payload = fetch_3dbag_lod12(
            ctx,
            lon=float(args.lon),
            lat=float(args.lat),
            radius_m=float(args.radius),
            max_features=int(args.max_3dbag_surfaces),
            refresh=bool(args.refresh_buildings),
        )
        lod12_surfaces = roof_surfaces_from_3dbag(
            fallback_payload,
            ctx,
            source_lod="lod12",
        )
        surfaces_with_fallback = add_lod12_fallback_surfaces(surfaces, lod12_surfaces)
        lod12_added = len(surfaces_with_fallback) - len(surfaces)
        surfaces = surfaces_with_fallback

    irradiance = fetch_pvgis_irradiance(
        float(args.lat),
        float(args.lon),
        refresh=bool(args.refresh_pvgis),
        skip=bool(args.skip_pvgis),
    )
    study_sun = sun_state_for_local_time(
        float(args.lat),
        float(args.lon),
        study_date,
        study_time,
        progress=0.5,
    )
    evaluate_roof_surfaces(
        surfaces,
        assumptions=assumptions,
        irradiance_kwh_m2=irradiance.annual_kwh_m2,
        sun=study_sun,
    )
    summary = summary_from_surfaces(surfaces)
    roof_meshes, mesh_stats = build_roof_mesh_layers(surfaces)
    if not roof_meshes or mesh_stats.roof_triangles == 0:
        raise SystemExit("3D BAG roof surfaces loaded but produced no renderable roof mesh.")

    context_surfaces = build_osm_context_surfaces(
        ctx,
        lon=float(args.lon),
        lat=float(args.lat),
        radius_m=float(args.radius),
        refresh_osm=bool(args.refresh_osm),
        skip_osm=bool(args.skip_osm),
        osm_timeout_seconds=float(args.osm_timeout),
    )
    scene = city.SceneLayers(
        surfaces=context_surfaces,
        meshes=roof_meshes,
        roof_outlines=[],
        focus_landmarks=[],
        radius=float(args.radius),
    )
    prepared = day.prepare_scene(
        scene,
        width=layout.map_width,
        height=layout.map_height,
        supersample=int(args.supersample),
        eye_scale=(-0.62, 1.55, -0.68),  # Tilted around X - lower viewing angle
        target_height_ratio=0.012,
        fov_deg=21.0,
        margin_ratio=0.005,
    )

    print(
        f"[3DBAG] rendered 3D BAG roof surfaces={mesh_stats.roof_surfaces:,} "
        f"buildings={mesh_stats.buildings:,} roof_triangles={mesh_stats.roof_triangles:,}"
    )
    if lod12_added:
        print(
            f"[3DBAG] added {lod12_added:,} LoD1.2 fallback surfaces for buildings "
            f"missing LoD2.2 facets (LoD2.2 buildings={lod22_building_count:,})."
        )
    print(
        f"[Solar] irradiation={irradiance.annual_kwh_m2:.0f} kWh/m2/year ({irradiance.source}); "
        f"usable_area={summary.usable_area_m2:,.0f} m2 annual_pv={summary.annual_kwh / 1000.0:,.1f} MWh"
    )

    if not args.animate:
        frame = render_study_frame(
            prepared,
            ctx,
            layout,
            summary=summary,
            assumptions=assumptions,
            irradiance=irradiance,
            sun=study_sun,
            study_date=study_date,
            study_time=study_time,
            surfaces=surfaces,
        )
        frame.save(output_path, format="PNG")
        print(f"[Rotterdam] Wrote {display_path(output_path)}")
        return 0

    frame_count = max(1, int(args.frames))
    fps = max(1, int(args.fps))
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is not None:
        frames_dir = args.frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_frames:
        frames_dir = output_path.with_suffix("")
        frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="rotterdam_solar_", dir=str(output_path.parent))
        frames_dir = Path(temp_dir.name)

    try:
        for frame_index in range(frame_count):
            frame_time = animation_time_for_frame(study_date, frame_index, frame_count)
            sun = sun_state_for_local_time(
                float(args.lat),
                float(args.lon),
                study_date,
                frame_time,
                progress=frame_index / max(frame_count - 1, 1),
            )
            frame = render_study_frame(
                prepared,
                ctx,
                layout,
                summary=summary,
                assumptions=assumptions,
                irradiance=irradiance,
                sun=sun,
                study_date=study_date,
                study_time=frame_time,
                surfaces=surfaces,
            )
            frame.save(frames_dir / f"frame_{frame_index:05d}.png", format="PNG", compress_level=0)
            if frame_index == 0 or (frame_index + 1) % max(1, fps) == 0 or frame_index + 1 == frame_count:
                print(
                    f"[Rotterdam] frame {frame_index + 1}/{frame_count} "
                    f"| {frame_time.strftime('%H:%M')} sun={sun.azimuth_deg:.1f}/{sun.elevation_deg:.1f}"
                )
        encode_video(frames_dir, output_path, fps=fps)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"[Rotterdam] Wrote {display_path(output_path)}")
    if args.keep_frames or args.frames_dir is not None:
        print(f"[Rotterdam] Frames saved in {display_path(frames_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
