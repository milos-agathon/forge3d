#!/usr/bin/env python3
"""Barcelona estimated-drive-time poster rendered on 3D terrain with Forge3D.

This example mirrors the user's 2D R workflow in a Forge3D-native way:
1. Fetch a drivable OSM road graph around Barcelona.
2. Compute estimated drive times from Plaça de Catalunya with class-based speeds.
3. Color roads with a sequential travel-time ramp.
4. Build a matching DEM from Terrarium elevation tiles.
5. Drape the road texture over real 3D terrain and snapshot it with Forge3D.

Requirements:
    pip install forge3d pillow pyproj rasterio shapely
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pyproj import Transformer

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.viewer import ViewerError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "barcelona_travel_time_3d" / "barcelona_pulse_3d.png"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "barcelona_travel_time_3d"
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
TERRARIUM_URL = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
OSM_NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "forge3d-barcelona-pulse/1.0"
BARCELONA_CENTER = (2.1700471, 41.3870167)
ORIGIN_LABEL = "Plaça de Catalunya"
BARCELONA_CAPTION = (
    "Estimated drive times from a simplified bidirectional OSM road graph | "
    "Sources: OpenStreetMap contributors + AWS Terrarium terrain tiles"
)
KEEP_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
ROAD_SPEED_KMH = {
    "motorway": 90.0,
    "trunk": 70.0,
    "primary": 50.0,
    "secondary": 40.0,
    "tertiary": 35.0,
    "unclassified": 28.0,
    "residential": 22.0,
    "living_street": 16.0,
    "service": 18.0,
    "motorway_link": 45.0,
    "trunk_link": 40.0,
    "primary_link": 35.0,
    "secondary_link": 30.0,
    "tertiary_link": 26.0,
}
ROAD_WIDTH_FACTOR = {
    "motorway": 1.72,
    "trunk": 1.56,
    "primary": 1.36,
    "secondary": 1.20,
    "tertiary": 1.08,
    "unclassified": 0.98,
    "residential": 0.86,
    "living_street": 0.80,
    "service": 0.72,
    "motorway_link": 1.18,
    "trunk_link": 1.10,
    "primary_link": 1.04,
    "secondary_link": 0.98,
    "tertiary_link": 0.92,
}
ROAD_CORE_ALPHA = {
    "motorway": 255,
    "trunk": 250,
    "primary": 242,
    "secondary": 236,
    "tertiary": 226,
    "unclassified": 208,
    "residential": 186,
    "living_street": 172,
    "service": 156,
    "motorway_link": 236,
    "trunk_link": 226,
    "primary_link": 220,
    "secondary_link": 206,
    "tertiary_link": 186,
}
ROAD_GLOW_ALPHA = {
    "motorway": 138,
    "trunk": 128,
    "primary": 116,
    "secondary": 106,
    "tertiary": 92,
    "unclassified": 76,
    "residential": 62,
    "living_street": 58,
    "service": 50,
    "motorway_link": 106,
    "trunk_link": 94,
    "primary_link": 90,
    "secondary_link": 78,
    "tertiary_link": 66,
}
ROAD_DRAW_PRIORITY = {
    "service": 0,
    "living_street": 1,
    "residential": 2,
    "unclassified": 3,
    "tertiary_link": 4,
    "tertiary": 5,
    "secondary_link": 6,
    "secondary": 7,
    "primary_link": 8,
    "primary": 9,
    "trunk_link": 10,
    "trunk": 11,
    "motorway_link": 12,
    "motorway": 13,
}
ROAD_COLOR_STOPS = (
    (0.00, "#30547d"),
    (0.18, "#3f6d95"),
    (0.38, "#5488a7"),
    (0.58, "#6ea2b0"),
    (0.78, "#9ebba0"),
    (0.92, "#d9c6a2"),
    (1.00, "#f4e3c1"),
)
VIEWER_SETTLE_SECONDS = 2.2
SEA_LEVEL_M = 0.0
OSM_SPAIN_BOUNDARY_QUERY = urlencode(
    {
        "q": "Spain",
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 3,
    }
)


@dataclass(frozen=True)
class ProjectedNode:
    x: float
    y: float


@dataclass(frozen=True)
class RoadSegment:
    u: int
    v: int
    x0: float
    y0: float
    x1: float
    y1: float
    seconds: float
    highway: str


@dataclass(frozen=True)
class TimedSegment:
    x0: float
    y0: float
    x1: float
    y1: float
    tmin: float
    highway: str


@dataclass(frozen=True)
class SquareExtent:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return float(self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return float(self.max_y - self.min_y)

    @property
    def center_x(self) -> float:
        return 0.5 * (self.min_x + self.max_x)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.min_y + self.max_y)

    def to_dict(self) -> dict[str, float]:
        return {
            "min_x": float(self.min_x),
            "min_y": float(self.min_y),
            "max_x": float(self.max_x),
            "max_y": float(self.max_y),
        }


@dataclass
class TerrainPrep:
    heightmap: np.ndarray
    water_mask: np.ndarray
    lowland_mask: np.ndarray
    land_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Barcelona 3D estimated-drive-time road poster with Forge3D.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-mins", type=float, default=30.0, help="Estimated travel-time cutoff in minutes.")
    parser.add_argument("--query-radius-km", type=float, default=12.0, help="OSM road query radius around the origin.")
    parser.add_argument("--terrain-zoom", type=int, default=15, help="Terrarium tile zoom level.")
    parser.add_argument("--terrain-size", type=int, default=4096, help="DEM grid dimension in pixels.")
    parser.add_argument("--overlay-size", type=int, default=2600, help="Road overlay texture dimension in pixels.")
    parser.add_argument("--viewer-width", type=int, default=1800)
    parser.add_argument("--viewer-height", type=int, default=1800)
    parser.add_argument("--snapshot-width", type=int, default=4608)
    parser.add_argument("--snapshot-height", type=int, default=4608)
    parser.add_argument("--cam-phi", type=float, default=90.0, help="Camera azimuth in degrees.")
    parser.add_argument("--cam-theta", type=float, default=22.0, help="Camera elevation in degrees.")
    parser.add_argument("--cam-fov", type=float, default=21.5, help="Camera field of view in degrees.")
    parser.add_argument("--cam-radius-scale", type=float, default=1.88, help="Camera distance as a multiple of terrain size.")
    parser.add_argument(
        "--zscale",
        type=float,
        default=None,
        help="Terrain vertical exaggeration. Defaults to an adaptive value based on local relief.",
    )
    parser.add_argument("--origin-lon", type=float, default=BARCELONA_CENTER[0])
    parser.add_argument("--origin-lat", type=float, default=BARCELONA_CENTER[1])
    parser.add_argument(
        "--boundary-data",
        type=Path,
        default=None,
        help="Optional local OSM Spain boundary GeoJSON (.geojson/.json). Defaults to cached OSM relation geometry.",
    )
    parser.add_argument("--refresh-osm", action="store_true", help="Re-fetch the cached Overpass road graph.")
    parser.add_argument("--refresh-dem", action="store_true", help="Re-fetch cached Terrarium tiles.")
    parser.add_argument("--refresh-boundary", action="store_true", help="Re-fetch the cached OSM Spain boundary geometry.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build cached DEM + overlay assets.")
    return parser.parse_args()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resampling(name: str) -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(getattr(resampling, name))
    return int(getattr(Image, name))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> np.ndarray:
    span = max(float(edge1) - float(edge0), 1e-6)
    x = np.clip((np.asarray(value, dtype=np.float32) - float(edge0)) / span, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _resize_float_raster(values: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    resized = image.resize((int(size[0]), int(size[1])), resample=_resampling("BILINEAR"))
    return np.asarray(resized, dtype=np.float32)


def _blur_float_mask(values: np.ndarray, *, radius: float) -> np.ndarray:
    image = Image.fromarray(
        np.round(np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(float(radius), 0.0)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _terrain_masks_from_absolute_height(
    heightmap_m: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    safe_heightmap = np.where(np.isfinite(heightmap_m), np.asarray(heightmap_m, dtype=np.float32), np.float32(SEA_LEVEL_M))
    if land_mask is None:
        water_mask = 1.0 - _smoothstep(SEA_LEVEL_M + 1.5, SEA_LEVEL_M + 8.0, safe_heightmap)
        lowland_mask = 1.0 - _smoothstep(14.0, 84.0, safe_heightmap)
    else:
        land = np.clip(np.asarray(land_mask, dtype=np.float32), 0.0, 1.0)
        water_mask = np.clip(1.0 - land, 0.0, 1.0)
        lowland_mask = (1.0 - _smoothstep(14.0, 84.0, safe_heightmap)) * land
    return (
        np.clip(water_mask, 0.0, 1.0).astype(np.float32),
        np.clip(lowland_mask, 0.0, 1.0).astype(np.float32),
    )


def road_palette_rgb(value: float, *, reverse: bool = True) -> tuple[int, int, int]:
    x = float(np.clip(value, 0.0, 1.0))
    if reverse:
        x = 1.0 - x
    stops = ROAD_COLOR_STOPS
    if x <= stops[0][0]:
        return _hex_to_rgb(stops[0][1])
    if x >= stops[-1][0]:
        return _hex_to_rgb(stops[-1][1])
    for (a_t, a_hex), (b_t, b_hex) in zip(stops[:-1], stops[1:]):
        if a_t <= x <= b_t:
            t = (x - a_t) / max(b_t - a_t, 1e-6)
            a_rgb = _hex_to_rgb(a_hex)
            b_rgb = _hex_to_rgb(b_hex)
            return tuple(int(round(_lerp(c0, c1, t))) for c0, c1 in zip(a_rgb, b_rgb))
    return _hex_to_rgb(stops[-1][1])


def poster_title(max_mins: float) -> str:
    return f"Barcelona Pulse: Estimated Reach Within {int(round(max_mins))} Minutes"


def poster_subtitle() -> str:
    return (
        f"Estimated drive times on a simplified, bidirectional street graph "
        f"from {ORIGIN_LABEL}."
    )


def legend_tick_values(max_mins: float) -> list[float]:
    maximum = max(float(max_mins), 0.0)
    ticks = {0.0, maximum}
    for minutes in range(10, int(math.floor(maximum)) + 1, 10):
        ticks.add(float(minutes))
    return sorted(ticks)


def format_legend_tick(minutes: float, *, max_mins: float) -> str:
    rounded = round(float(minutes), 1)
    if math.isclose(rounded, round(rounded), abs_tol=0.05):
        value = f"{int(round(rounded))}"
    else:
        value = f"{rounded:g}"
    if math.isclose(float(minutes), float(max_mins), abs_tol=0.05):
        return f"{value} min"
    return value


def choose_font(size: int, *, italic: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf" if italic else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf" if italic else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_font_to_width(text: str, *, max_width: int, base_size: int, italic: bool = False, min_size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    size = max(int(base_size), int(min_size))
    while size >= int(min_size):
        font = choose_font(size, italic=italic)
        width = int(probe.textbbox((0, 0), text, font=font)[2])
        if width <= int(max_width):
            return font
        size -= 2
    return choose_font(int(min_size), italic=italic)


def wrap_text_to_width(text: str, *, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> list[str]:
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            width = int(probe.textbbox((0, 0), candidate, font=font)[2])
            if width <= int(max_width):
                line = candidate
            else:
                wrapped.append(line)
                line = word
        wrapped.append(line)
    return wrapped


def utm_epsg_for_lon_lat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1.0)
    return (32600 if lat >= 0.0 else 32700) + zone


def request_bytes(url: str, *, cache_path: Path, refresh: bool = False, timeout: float = 180.0) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    cache_path.write_bytes(payload)
    return payload


def request_json_payload(query: str, *, cache_path: Path, refresh: bool = False) -> dict:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload = None
    last_error: Exception | None = None
    for endpoint in OVERPASS_URLS:
        for attempt in range(3):
            request = Request(
                endpoint,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with urlopen(request, timeout=180) as response:
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(1.0 + attempt * 1.5)
        if payload is not None:
            break

    if payload is None:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise SystemExit(f"Failed to fetch OSM data: {last_error}") from last_error

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def request_json_url(url: str, *, cache_path: Path, refresh: bool = False, timeout: float = 180.0):
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload = None
    last_error: Exception | None = None
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.0 + attempt * 1.5)

    if payload is None:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise SystemExit(f"Failed to fetch JSON from {url}: {last_error}") from last_error

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _pick_osm_spain_boundary(results: object) -> dict[str, object]:
    if not isinstance(results, list):
        raise RuntimeError("Unexpected OSM boundary response; expected a list of search results")

    for result in results:
        if not isinstance(result, dict):
            continue
        geometry = result.get("geojson")
        category = str(result.get("category") or result.get("class") or "").strip().lower()
        boundary_type = str(result.get("type") or "").strip().lower()
        display_name = str(result.get("display_name") or "").strip()
        if geometry and category == "boundary" and boundary_type == "administrative":
            if "Spain" in display_name or "España" in display_name:
                return result

    for result in results:
        if isinstance(result, dict) and result.get("geojson"):
            return result

    raise RuntimeError("Could not find a Spain boundary geometry in the OSM response")


def fetch_osm_spain_boundary_geojson(*, cache_dir: Path, refresh: bool) -> Path:
    geojson_path = cache_dir / "osm" / "spain_boundary.geojson"
    if geojson_path.exists() and not refresh:
        return geojson_path

    raw_cache_path = cache_dir / "osm" / "spain_boundary_nominatim.json"
    results = request_json_url(
        f"{OSM_NOMINATIM_SEARCH_URL}?{OSM_SPAIN_BOUNDARY_QUERY}",
        cache_path=raw_cache_path,
        refresh=refresh,
    )
    matched = _pick_osm_spain_boundary(results)
    feature = {
        "type": "Feature",
        "properties": {
            "source": "OpenStreetMap contributors via Nominatim",
            "display_name": matched.get("display_name"),
            "osm_type": matched.get("osm_type"),
            "osm_id": matched.get("osm_id"),
            "class": matched.get("class"),
            "type": matched.get("type"),
        },
        "geometry": matched["geojson"],
    }
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.write_text(json.dumps(feature, ensure_ascii=False, indent=2), encoding="utf-8")
    return geojson_path


def _read_geojson_geometry(boundary_path: Path):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    geometries = []

    if isinstance(payload, list):
        geometries = [shape(item["geojson"]) for item in payload if isinstance(item, dict) and item.get("geojson")]
    elif isinstance(payload, dict):
        payload_type = str(payload.get("type") or "")
        if payload_type == "FeatureCollection":
            geometries = [
                shape(feature["geometry"])
                for feature in payload.get("features", [])
                if isinstance(feature, dict) and feature.get("geometry")
            ]
        elif payload_type == "Feature":
            geometry = payload.get("geometry")
            if geometry:
                geometries = [shape(geometry)]
        else:
            geometries = [shape(payload)]

    geometries = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not geometries:
        raise RuntimeError(f"No valid geometry found in boundary file: {display_path(boundary_path)}")
    return unary_union(geometries) if len(geometries) > 1 else geometries[0]


def load_spain_boundary(
    boundary_path: Path | None,
    *,
    cache_dir: Path,
    target_crs: str,
    refresh: bool,
):
    if boundary_path is None:
        boundary_path = fetch_osm_spain_boundary_geojson(cache_dir=cache_dir, refresh=refresh)

    if boundary_path.suffix.lower() not in {".geojson", ".json"}:
        raise RuntimeError(
            f"Spain boundary file must be GeoJSON in WGS84 when provided locally: {display_path(boundary_path)}"
        )

    from shapely.ops import transform as shapely_transform

    geometry_wgs84 = _read_geojson_geometry(boundary_path)
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected_geometry = shapely_transform(transformer.transform, geometry_wgs84)
    return projected_geometry, boundary_path


def build_coastline_query(*, west: float, south: float, east: float, north: float) -> str:
    return (
        "[out:json][timeout:180];("
        f'way["natural"="coastline"]({south:.7f},{west:.7f},{north:.7f},{east:.7f});'
        ");out body geom qt;"
    )


def fetch_osm_coastline_ways(
    *,
    extent: SquareExtent,
    to_wgs84: Transformer,
    cache_dir: Path,
    refresh: bool,
) -> tuple[list[dict], Path]:
    pad_m = max(2200.0, max(float(extent.width), float(extent.height)) * 0.16)
    padded_extent = SquareExtent(
        min_x=float(extent.min_x) - pad_m,
        min_y=float(extent.min_y) - pad_m,
        max_x=float(extent.max_x) + pad_m,
        max_y=float(extent.max_y) + pad_m,
    )
    lon, lat = to_wgs84.transform(
        [padded_extent.min_x, padded_extent.min_x, padded_extent.max_x, padded_extent.max_x],
        [padded_extent.min_y, padded_extent.max_y, padded_extent.min_y, padded_extent.max_y],
    )
    west = float(np.min(lon))
    east = float(np.max(lon))
    south = float(np.min(lat))
    north = float(np.max(lat))
    bbox_signature = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"
    cache_name = hashlib.sha1(bbox_signature.encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / "osm" / f"barcelona_coastline_{cache_name}.json"
    payload = request_json_payload(
        build_coastline_query(west=west, south=south, east=east, north=north),
        cache_path=cache_path,
        refresh=refresh,
    )
    ways = [element for element in payload.get("elements", []) if element.get("type") == "way"]
    if not ways:
        raise RuntimeError("Could not fetch OSM coastline segments for the Barcelona DEM extent")
    return ways, cache_path


def build_land_geometry_from_osm_coastline(
    coastline_ways: list[dict],
    *,
    extent: SquareExtent,
    to_projected: Transformer,
    seed_point_xy: tuple[float, float],
):
    from shapely.geometry import LineString, Point, box
    from shapely.ops import polygonize, unary_union

    extent_box = box(extent.min_x, extent.min_y, extent.max_x, extent.max_y)
    coastline_lines = []
    for way in coastline_ways:
        geometry = way.get("geometry") or []
        if len(geometry) < 2:
            continue
        lon = np.asarray([float(point["lon"]) for point in geometry], dtype=np.float64)
        lat = np.asarray([float(point["lat"]) for point in geometry], dtype=np.float64)
        xs, ys = to_projected.transform(lon, lat)
        line = LineString(
            np.column_stack(
                [
                    np.asarray(xs, dtype=np.float64),
                    np.asarray(ys, dtype=np.float64),
                ]
            )
        )
        clipped = line.intersection(extent_box)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            coastline_lines.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            coastline_lines.extend([segment for segment in clipped.geoms if not segment.is_empty])

    if not coastline_lines:
        raise RuntimeError("OSM coastline segments did not intersect the Barcelona DEM extent")

    polygon_mesh = unary_union([extent_box.boundary, *coastline_lines])
    polygons = [polygon for polygon in polygonize(polygon_mesh) if not polygon.is_empty and polygon.area > 1.0]
    if not polygons:
        raise RuntimeError("Could not polygonize an OSM coastline-based land mask for Barcelona")

    seed_point = Point(float(seed_point_xy[0]), float(seed_point_xy[1]))
    land_polygons = [polygon for polygon in polygons if polygon.buffer(1.0).contains(seed_point)]
    if not land_polygons:
        land_polygons = [min(polygons, key=lambda polygon: polygon.distance(seed_point))]
    return unary_union(land_polygons)


def rasterize_geometry_mask(
    *,
    extent: SquareExtent,
    shape: tuple[int, int],
    geometry,
) -> np.ndarray:
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    return geometry_mask(
        [mapping(geometry)],
        out_shape=shape,
        transform=from_bounds(extent.min_x, extent.min_y, extent.max_x, extent.max_y, shape[1], shape[0]),
        invert=True,
        all_touched=True,
    )


def parse_maxspeed(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        speed = float(value)
        return speed if math.isfinite(speed) and speed > 0.0 else None
    if isinstance(value, list) and value:
        return parse_maxspeed(value[0])

    text = str(value).strip().lower()
    if not text:
        return None
    for token in text.replace(",", ".").replace(";", " ").replace("|", " ").split():
        try:
            speed = float(token)
        except ValueError:
            continue
        if math.isfinite(speed) and speed > 0.0:
            if "mph" in text:
                speed *= 1.60934
            return speed
    return None


def estimate_speed_kmh(highway: str, tags: dict[str, object]) -> float:
    maxspeed = parse_maxspeed(tags.get("maxspeed"))
    if maxspeed is not None:
        return float(np.clip(maxspeed, 8.0, 120.0))
    return ROAD_SPEED_KMH.get(highway, 25.0)


def build_osm_query(*, lon: float, lat: float, radius_m: float) -> str:
    road_expr = "|".join(sorted(KEEP_HIGHWAYS))
    return (
        "[out:json][timeout:180];("
        f'way["highway"~"^({road_expr})$"]["area"!="yes"](around:{int(round(radius_m))},{lat:.7f},{lon:.7f});'
        ");out body geom qt;"
    )


def fetch_drivable_ways(*, lon: float, lat: float, radius_m: float, cache_dir: Path, refresh: bool) -> list[dict]:
    road_signature = ",".join(sorted(KEEP_HIGHWAYS))
    signature_hash = hashlib.sha1(road_signature.encode("utf-8")).hexdigest()[:10]
    cache_path = cache_dir / "osm" / f"barcelona_roads_r{int(round(radius_m))}_{signature_hash}.json"
    payload = request_json_payload(build_osm_query(lon=lon, lat=lat, radius_m=radius_m), cache_path=cache_path, refresh=refresh)
    return [element for element in payload.get("elements", []) if element.get("type") == "way"]


def build_drive_graph(
    ways: list[dict],
    *,
    transformer: Transformer,
) -> tuple[dict[int, ProjectedNode], list[RoadSegment], dict[int, list[tuple[int, float]]], dict[int, list[int]]]:
    """Build a simplified graph for estimated drive times.

    This example intentionally uses class-based speeds and treats each retained
    segment as bidirectional unless it is later extended with oneway/access logic.
    """
    nodes: dict[int, ProjectedNode] = {}
    segments: list[RoadSegment] = []
    weighted_adj: dict[int, list[tuple[int, float]]] = {}
    unweighted_adj: dict[int, list[int]] = {}

    for way in ways:
        tags = dict(way.get("tags", {}))
        highway = str(tags.get("highway", "")).strip()
        if highway not in KEEP_HIGHWAYS:
            continue

        node_ids = way.get("nodes") or []
        geometry = way.get("geometry") or []
        if len(node_ids) < 2 or len(node_ids) != len(geometry):
            continue

        lon = np.asarray([float(point["lon"]) for point in geometry], dtype=np.float64)
        lat = np.asarray([float(point["lat"]) for point in geometry], dtype=np.float64)
        xs, ys = transformer.transform(lon, lat)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        for node_id, x, y in zip(node_ids, xs, ys):
            nodes[int(node_id)] = ProjectedNode(float(x), float(y))

        speed_kmh = estimate_speed_kmh(highway, tags)
        meters_per_second = max(speed_kmh * (1000.0 / 3600.0), 0.5)

        for idx in range(len(node_ids) - 1):
            u = int(node_ids[idx])
            v = int(node_ids[idx + 1])
            x0 = float(xs[idx])
            y0 = float(ys[idx])
            x1 = float(xs[idx + 1])
            y1 = float(ys[idx + 1])
            length_m = float(math.hypot(x1 - x0, y1 - y0))
            if not math.isfinite(length_m) or length_m < 2.0:
                continue
            seconds = length_m / meters_per_second
            segments.append(RoadSegment(u=u, v=v, x0=x0, y0=y0, x1=x1, y1=y1, seconds=seconds, highway=highway))
            weighted_adj.setdefault(u, []).append((v, seconds))
            weighted_adj.setdefault(v, []).append((u, seconds))
            unweighted_adj.setdefault(u, []).append(v)
            unweighted_adj.setdefault(v, []).append(u)

    return nodes, segments, weighted_adj, unweighted_adj


def largest_component_nodes(graph: dict[int, list[int]]) -> set[int]:
    visited: set[int] = set()
    largest: set[int] = set()

    for start in graph:
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        visited.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in graph.get(node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        if len(component) > len(largest):
            largest = component

    return largest


def nearest_node(nodes: dict[int, ProjectedNode], *, x: float, y: float, candidates: set[int]) -> int:
    best_id = -1
    best_dist = math.inf
    for node_id in candidates:
        point = nodes.get(node_id)
        if point is None:
            continue
        distance = (point.x - x) ** 2 + (point.y - y) ** 2
        if distance < best_dist:
            best_dist = distance
            best_id = node_id
    if best_id < 0:
        raise RuntimeError("Could not snap the origin to any road node")
    return best_id


def dijkstra_seconds(graph: dict[int, list[tuple[int, float]]], *, start: int, allowed: set[int]) -> dict[int, float]:
    dist: dict[int, float] = {start: 0.0}
    queue: list[tuple[float, int]] = [(0.0, start)]

    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist > dist.get(node, math.inf):
            continue
        for neighbor, weight in graph.get(node, []):
            if neighbor not in allowed:
                continue
            new_dist = current_dist + float(weight)
            if new_dist < dist.get(neighbor, math.inf):
                dist[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))

    return dist


def select_timed_segments(
    segments: list[RoadSegment],
    *,
    component: set[int],
    node_times_s: dict[int, float],
    cutoff_min: float,
) -> list[TimedSegment]:
    cutoff_s = float(cutoff_min) * 60.0
    selected: list[TimedSegment] = []
    for segment in segments:
        if segment.u not in component or segment.v not in component:
            continue
        tmin_s = min(node_times_s.get(segment.u, math.inf), node_times_s.get(segment.v, math.inf))
        if not math.isfinite(tmin_s) or tmin_s > cutoff_s:
            continue
        selected.append(
            TimedSegment(
                x0=segment.x0,
                y0=segment.y0,
                x1=segment.x1,
                y1=segment.y1,
                tmin=tmin_s / 60.0,
                highway=segment.highway,
            )
        )
    return selected


def make_square_extent(
    segments: list[TimedSegment],
    *,
    pad_ratio: float = 0.12,
    min_pad_m: float = 2200.0,
) -> SquareExtent:
    if not segments:
        raise ValueError("Expected at least one road segment")
    xs = np.asarray([value for segment in segments for value in (segment.x0, segment.x1)], dtype=np.float64)
    ys = np.asarray([value for segment in segments for value in (segment.y0, segment.y1)], dtype=np.float64)
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    span = max(max_x - min_x, max_y - min_y, 1.0)
    pad = max(span * float(pad_ratio), float(min_pad_m))
    half = 0.5 * span + pad
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    return SquareExtent(min_x=cx - half, min_y=cy - half, max_x=cx + half, max_y=cy + half)


def slippy_tile_xy(lon: float | np.ndarray, lat: float | np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    lat_arr = np.clip(lat_arr, -85.05112878, 85.05112878)
    n = float(1 << int(zoom))
    x = (lon_arr + 180.0) / 360.0 * n
    lat_rad = np.deg2rad(lat_arr)
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) * 0.5 * n
    return x, y


def slippy_pixel_xy(lon: float | np.ndarray, lat: float | np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    tile_x, tile_y = slippy_tile_xy(lon, lat, zoom)
    return tile_x * 256.0, tile_y * 256.0


def decode_terrarium_tile(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return (rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0) - 32768.0


def fetch_terrarium_mosaic(
    *,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    zoom: int,
    cache_dir: Path,
    refresh: bool,
) -> tuple[np.ndarray, int, int]:
    west_tx, north_ty = slippy_tile_xy(lon_min, lat_max, zoom)
    east_tx, south_ty = slippy_tile_xy(lon_max, lat_min, zoom)
    tile_max = (1 << int(zoom)) - 1
    min_tx = int(np.clip(math.floor(min(float(west_tx), float(east_tx))), 0, tile_max))
    max_tx = int(np.clip(math.floor(max(float(west_tx), float(east_tx))), 0, tile_max))
    min_ty = int(np.clip(math.floor(min(float(north_ty), float(south_ty))), 0, tile_max))
    max_ty = int(np.clip(math.floor(max(float(north_ty), float(south_ty))), 0, tile_max))

    rows: list[np.ndarray] = []
    for ty in range(min_ty, max_ty + 1):
        tiles: list[np.ndarray] = []
        for tx in range(min_tx, max_tx + 1):
            cache_path = cache_dir / "terrarium" / str(zoom) / str(tx) / f"{ty}.png"
            request_bytes(
                TERRARIUM_URL.format(z=zoom, x=tx, y=ty),
                cache_path=cache_path,
                refresh=refresh,
            )
            with Image.open(cache_path) as tile:
                tiles.append(decode_terrarium_tile(tile))
        rows.append(np.hstack(tiles))
    return np.vstack(rows).astype(np.float32), min_tx, min_ty


def sample_terrarium_bilinear(
    mosaic: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    zoom: int,
    origin_tx: int,
    origin_ty: int,
) -> np.ndarray:
    pixel_x, pixel_y = slippy_pixel_xy(lon, lat, zoom)
    local_x = pixel_x - float(origin_tx * 256)
    local_y = pixel_y - float(origin_ty * 256)

    height, width = mosaic.shape
    out = np.full(local_x.shape, SEA_LEVEL_M, dtype=np.float32)
    valid = (
        np.isfinite(local_x)
        & np.isfinite(local_y)
        & (local_x >= 0.0)
        & (local_y >= 0.0)
        & (local_x < width - 1.0)
        & (local_y < height - 1.0)
    )
    if not np.any(valid):
        return out

    lx = local_x[valid]
    ly = local_y[valid]
    x0 = np.floor(lx).astype(np.int32)
    y0 = np.floor(ly).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (lx - x0).astype(np.float32)
    ty = (ly - y0).astype(np.float32)

    h00 = mosaic[y0, x0]
    h10 = mosaic[y0, x1]
    h01 = mosaic[y1, x0]
    h11 = mosaic[y1, x1]
    sampled = (
        h00 * (1.0 - tx) * (1.0 - ty)
        + h10 * tx * (1.0 - ty)
        + h01 * (1.0 - tx) * ty
        + h11 * tx * ty
    )
    out[valid] = sampled.astype(np.float32)
    return out


def build_heightmap(
    extent: SquareExtent,
    *,
    size: int,
    zoom: int,
    to_wgs84: Transformer,
    cache_dir: Path,
    refresh: bool,
) -> np.ndarray:
    corner_lon, corner_lat = to_wgs84.transform(
        [extent.min_x, extent.min_x, extent.max_x, extent.max_x],
        [extent.min_y, extent.max_y, extent.min_y, extent.max_y],
    )
    lon_min = float(np.min(corner_lon))
    lon_max = float(np.max(corner_lon))
    lat_min = float(np.min(corner_lat))
    lat_max = float(np.max(corner_lat))

    mosaic, origin_tx, origin_ty = fetch_terrarium_mosaic(
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        zoom=zoom,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    xs = np.linspace(extent.min_x, extent.max_x, int(size), dtype=np.float64)
    ys = np.linspace(extent.max_y, extent.min_y, int(size), dtype=np.float64)
    heightmap = np.empty((int(size), int(size)), dtype=np.float32)

    for row_start in range(0, int(size), 192):
        row_end = min(int(size), row_start + 192)
        grid_x, grid_y = np.meshgrid(xs, ys[row_start:row_end])
        lon, lat = to_wgs84.transform(grid_x, grid_y)
        heightmap[row_start:row_end] = sample_terrarium_bilinear(
            mosaic,
            np.asarray(lon, dtype=np.float64),
            np.asarray(lat, dtype=np.float64),
            zoom=zoom,
            origin_tx=origin_tx,
            origin_ty=origin_ty,
        )

    finite = heightmap[np.isfinite(heightmap)]
    floor = float(finite.min()) if finite.size else SEA_LEVEL_M
    heightmap = np.nan_to_num(heightmap, nan=floor, posinf=floor, neginf=floor).astype(np.float32)
    return np.ascontiguousarray(heightmap, dtype=np.float32)


def mask_heightmap_to_geometry(
    heightmap: np.ndarray,
    *,
    extent: SquareExtent,
    geometry,
) -> tuple[np.ndarray, np.ndarray]:
    from shapely.geometry import box

    clipped_land = geometry.intersection(box(extent.min_x, extent.min_y, extent.max_x, extent.max_y))
    if clipped_land.is_empty:
        raise RuntimeError("Spain boundary did not overlap the Barcelona DEM extent")
    land_mask = rasterize_geometry_mask(extent=extent, shape=heightmap.shape, geometry=clipped_land).astype(np.float32)
    masked_heightmap = np.where(land_mask > 0.5, np.asarray(heightmap, dtype=np.float32), np.float32(np.nan))
    return masked_heightmap.astype(np.float32), land_mask


def prepare_terrain(
    raw_heightmap: np.ndarray,
    *,
    extent: SquareExtent,
    land_geometry,
) -> TerrainPrep:
    heightmap, land_mask = mask_heightmap_to_geometry(raw_heightmap, extent=extent, geometry=land_geometry)
    water_mask, lowland_mask = _terrain_masks_from_absolute_height(heightmap, land_mask=land_mask)
    finite = heightmap[np.isfinite(heightmap)]
    floor = float(finite.min()) if finite.size else SEA_LEVEL_M
    heightmap = np.where(np.isfinite(heightmap), heightmap - floor, np.float32(np.nan)).astype(np.float32)
    return TerrainPrep(
        heightmap=np.ascontiguousarray(heightmap, dtype=np.float32),
        water_mask=np.ascontiguousarray(water_mask, dtype=np.float32),
        lowland_mask=np.ascontiguousarray(lowland_mask, dtype=np.float32),
        land_mask=np.ascontiguousarray(land_mask, dtype=np.float32),
    )


def project_to_pixels(x: float, y: float, *, extent: SquareExtent, size: int) -> tuple[float, float]:
    px = (float(x) - extent.min_x) / max(extent.width, 1e-6) * float(size - 1)
    py = (extent.max_y - float(y)) / max(extent.height, 1e-6) * float(size - 1)
    return px, py


def build_road_overlay(
    segments: list[TimedSegment],
    *,
    extent: SquareExtent,
    overlay_size: int,
    max_mins: float,
    origin_x: float,
    origin_y: float,
) -> Image.Image:
    aa = 2
    render_size = int(overlay_size) * aa
    shadow = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
    glow = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
    core = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    core_draw = ImageDraw.Draw(core, "RGBA")
    base_width = max(1.55, float(overlay_size) / 700.0)
    width_scale = 0.55

    for segment in sorted(
        segments,
        key=lambda item: (
            ROAD_DRAW_PRIORITY.get(item.highway, 0),
            -float(item.tmin),
        ),
    ):
        x0, y0 = project_to_pixels(segment.x0, segment.y0, extent=extent, size=overlay_size)
        x1, y1 = project_to_pixels(segment.x1, segment.y1, extent=extent, size=overlay_size)
        color = road_palette_rgb(segment.tmin / max(float(max_mins), 1e-6), reverse=True)
        width_factor = ROAD_WIDTH_FACTOR.get(segment.highway, 1.0)
        core_alpha = ROAD_CORE_ALPHA.get(segment.highway, 188)
        glow_alpha = ROAD_GLOW_ALPHA.get(segment.highway, 58)
        shadow_alpha = min(154, int(round(core_alpha * 0.62)))
        shadow_width = max(1, int(round(base_width * 5.3 * width_factor * aa * width_scale)))
        glow_width = max(1, int(round(base_width * 4.2 * width_factor * aa * width_scale)))
        core_width = max(1, int(round(base_width * 1.84 * width_factor * aa * width_scale)))
        shadow_draw.line(
            [(x0 * aa, y0 * aa), (x1 * aa, y1 * aa)],
            fill=(14, 23, 35, shadow_alpha),
            width=shadow_width,
        )
        glow_draw.line(
            [(x0 * aa, y0 * aa), (x1 * aa, y1 * aa)],
            fill=(*color, glow_alpha),
            width=glow_width,
        )
        core_draw.line(
            [(x0 * aa, y0 * aa), (x1 * aa, y1 * aa)],
            fill=(*color, core_alpha),
            width=core_width,
        )

    blur_radius = max(1.0, float(overlay_size) / 260.0)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius * aa * 0.75))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=blur_radius * aa))
    overlay = Image.alpha_composite(shadow, glow)
    overlay = Image.alpha_composite(overlay, core).resize(
        (int(overlay_size), int(overlay_size)),
        resample=_resampling("LANCZOS"),
    )

    origin_px, origin_py = project_to_pixels(origin_x, origin_y, extent=extent, size=overlay_size)
    origin_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    origin_draw = ImageDraw.Draw(origin_layer, "RGBA")
    hot = road_palette_rgb(0.0, reverse=True)
    rings = (
        (overlay_size * 0.034, (*hot, 34)),
        (overlay_size * 0.018, (255, 246, 225, 80)),
        (overlay_size * 0.0065, (255, 255, 255, 240)),
    )
    for radius, fill in rings:
        origin_draw.ellipse(
            (origin_px - radius, origin_py - radius, origin_px + radius, origin_py + radius),
            fill=fill,
        )
    origin_layer = origin_layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, overlay_size / 540.0)))
    overlay = Image.alpha_composite(overlay, origin_layer)
    return overlay


def crop_raw_to_road_focus(
    raw_rgba: np.ndarray,
    *,
    chroma_threshold: float = 0.085,
    min_pixels: int = 4000,
) -> np.ndarray:
    road_bounds = road_footprint_bounds(
        raw_rgba,
        chroma_threshold=chroma_threshold,
        min_pixels=min_pixels,
    )
    if road_bounds is None:
        return np.asarray(raw_rgba, dtype=np.uint8)

    min_x, min_y, max_x, max_y = road_bounds
    rgb = np.asarray(raw_rgba[:, :, :3], dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    span_x = max_x - min_x
    span_y = max_y - min_y
    pad_x = max(int(round(span_x * 0.12)), int(round(width * 0.045)))
    pad_top = max(int(round(span_y * 0.22)), int(round(height * 0.065)))
    pad_bottom = max(int(round(span_y * 0.14)), int(round(height * 0.055)))

    crop_w = span_x + pad_x * 2
    crop_h = span_y + pad_top + pad_bottom
    side = min(max(crop_w, crop_h), min(width, height))
    if side >= min(width, height) - 8:
        return np.asarray(raw_rgba, dtype=np.uint8)

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y) - side * 0.03
    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.5))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))

    crop = np.asarray(raw_rgba, dtype=np.uint8)[top : top + side, left : left + side]
    zoomed = Image.fromarray(crop, mode="RGBA").resize((width, height), resample=_resampling("LANCZOS"))
    return np.asarray(zoomed, dtype=np.uint8)


def road_footprint_bounds(
    raw_rgba: np.ndarray,
    *,
    chroma_threshold: float = 0.085,
    min_pixels: int = 4000,
) -> tuple[int, int, int, int] | None:
    rgb = np.asarray(raw_rgba[:, :, :3], dtype=np.float32) / 255.0
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    brightness = np.max(rgb, axis=2)
    road_mask = (chroma >= float(chroma_threshold)) & (brightness >= 0.16)

    if int(np.count_nonzero(road_mask)) < int(min_pixels):
        return None

    ys, xs = np.nonzero(road_mask)
    min_x = int(xs.min())
    max_x = int(xs.max()) + 1
    min_y = int(ys.min())
    max_y = int(ys.max()) + 1
    return min_x, min_y, max_x, max_y


def compose_poster(raw_rgba: np.ndarray, *, max_mins: float) -> np.ndarray:
    focused_rgba = crop_raw_to_road_focus(raw_rgba)
    image = Image.fromarray(np.asarray(focused_rgba, dtype=np.uint8), mode="RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    title_text = poster_title(max_mins)
    subtitle_text = poster_subtitle()
    road_bounds = road_footprint_bounds(focused_rgba, chroma_threshold=0.075, min_pixels=3000)
    title_left = 0
    title_top = 0
    title_right_limit = int(width * 0.50)
    if road_bounds is not None:
        road_left = int(road_bounds[0])
        title_right_limit = min(title_right_limit, road_left - int(width * 0.020))
    panel_right = max(title_left + int(width * 0.62), title_right_limit)
    panel = (title_left, title_top, panel_right, int(height * 0.214))
    panel_pad_x = int(width * 0.020)
    panel_pad_top = int(height * 0.018)
    panel_pad_bottom = int(height * 0.018)
    inner_width = panel[2] - panel[0] - panel_pad_x * 2
    title_font = fit_font_to_width(
        title_text,
        max_width=inner_width,
        base_size=max(28, int(width * 0.027)),
        italic=True,
        min_size=max(20, int(width * 0.020)),
    )
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_h = title_bbox[3] - title_bbox[1]
    subtitle_font = choose_font(max(15, int(width * 0.0130)))
    subtitle_line_gap = max(6, int(height * 0.0045))
    subtitle_box_pad_x = max(12, int(width * 0.009))
    subtitle_box_pad_y = max(8, int(height * 0.006))
    subtitle_lines = wrap_text_to_width(
        subtitle_text,
        font=subtitle_font,
        max_width=max(120, inner_width - subtitle_box_pad_x * 2),
    )
    legend_label_font = choose_font(max(24, int(width * 0.0185)))
    subtitle_text_w = max(
        draw.textbbox((0, 0), line, font=subtitle_font)[2]
        for line in subtitle_lines
    )
    subtitle_box_w = min(inner_width, subtitle_text_w + subtitle_box_pad_x * 2)
    subtitle_line_bbox = draw.textbbox((0, 0), "Ag", font=subtitle_font)
    subtitle_line_h = subtitle_line_bbox[3] - subtitle_line_bbox[1]
    subtitle_box_h = subtitle_line_h + subtitle_box_pad_y * 2
    panel_gap = int(height * 0.014)
    subtitle_block_h = len(subtitle_lines) * subtitle_box_h + max(0, len(subtitle_lines) - 1) * subtitle_line_gap
    panel_bottom = panel[1] + panel_pad_top + title_h + panel_gap + subtitle_block_h + panel_pad_bottom
    panel = (panel[0], panel[1], panel[2], panel_bottom)

    draw.rectangle(panel, fill=(4, 5, 8, 214))
    title_x = panel[0] + panel_pad_x
    title_y = panel[1] + panel_pad_top
    draw.text(
        (title_x, title_y),
        title_text,
        fill=(248, 245, 239, 255),
        font=title_font,
    )
    subtitle_x = title_x
    subtitle_y = title_y + title_h + panel_gap
    for index, line in enumerate(subtitle_lines):
        box_top = subtitle_y + index * (subtitle_box_h + subtitle_line_gap)
        box_bottom = box_top + subtitle_box_h
        draw.rounded_rectangle(
            (subtitle_x, box_top, subtitle_x + subtitle_box_w, box_bottom),
            radius=max(10, int(round(height * 0.011))),
            fill=(0, 0, 0, 228),
        )
        draw.text(
            (subtitle_x + subtitle_box_pad_x, box_top + subtitle_box_pad_y),
            line,
            fill=(230, 233, 240, 244),
            font=subtitle_font,
        )

    caption_font = fit_font_to_width(
        BARCELONA_CAPTION,
        max_width=int(width * 0.58),
        base_size=max(17, int(width * 0.014)),
        min_size=max(12, int(width * 0.0105)),
    )
    caption_bbox = draw.textbbox((0, 0), BARCELONA_CAPTION, font=caption_font)
    caption_w = caption_bbox[2] - caption_bbox[0]
    caption_h = caption_bbox[3] - caption_bbox[1]
    caption_x = width - int(width * 0.030) - caption_w
    caption_y = height - int(height * 0.032) - caption_h

    legend_w = int(width * 0.384)
    legend_h = max(56, int(height * 0.041))
    legend_right = caption_x + caption_w
    legend_x = legend_right - legend_w
    title_gap = int(height * 0.012)
    label_gap = int(height * 0.010)
    caption_gap = int(height * 0.032)
    legend_title_text = f"Estimated drive time\nfrom {ORIGIN_LABEL}"
    legend_title_font = fit_font_to_width(
        max(legend_title_text.splitlines(), key=len),
        max_width=legend_w,
        base_size=max(26, int(width * 0.021)),
        min_size=max(16, int(width * 0.0135)),
    )
    legend_title_spacing = max(4, int(height * 0.004))
    legend_title_bbox = draw.multiline_textbbox(
        (0, 0),
        legend_title_text,
        font=legend_title_font,
        spacing=legend_title_spacing,
    )
    legend_title_h = legend_title_bbox[3] - legend_title_bbox[1]
    tick_values = legend_tick_values(max_mins)
    tick_labels = [format_legend_tick(value, max_mins=max_mins) for value in tick_values]
    tick_mark_h = max(7, int(round(height * 0.008)))
    legend_label_h = max(
        draw.textbbox((0, 0), label, font=legend_label_font)[3] for label in tick_labels
    )
    legend_y = caption_y - caption_gap - legend_h - tick_mark_h - label_gap - legend_label_h - title_gap - legend_title_h
    legend = Image.new("RGBA", (legend_w, legend_h), (0, 0, 0, 0))
    legend_draw = ImageDraw.Draw(legend, "RGBA")
    for x in range(legend_w):
        t = x / max(legend_w - 1, 1)
        color = road_palette_rgb(t, reverse=True)
        legend_draw.line([(x, 0), (x, legend_h)], fill=(*color, 244))
    legend_mask = Image.new("L", (legend_w, legend_h), 0)
    ImageDraw.Draw(legend_mask).rounded_rectangle((0, 0, legend_w - 1, legend_h - 1), radius=legend_h // 2, fill=255)
    legend.putalpha(legend_mask)
    image.alpha_composite(legend, dest=(legend_x, legend_y))
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
        radius=legend_h // 2,
        outline=(246, 247, 250, 176),
        width=max(2, int(round(width * 0.0012))),
    )
    draw.multiline_text(
        (legend_x, legend_y - title_gap - legend_title_h),
        legend_title_text,
        fill=(244, 246, 250, 246),
        font=legend_title_font,
        spacing=legend_title_spacing,
    )
    tick_line_w = max(1, int(round(width * 0.0010)))
    tick_label_y = legend_y + legend_h + tick_mark_h + label_gap
    for index, (tick, label) in enumerate(zip(tick_values, tick_labels)):
        tick_x = legend_x + int(round((tick / max(float(max_mins), 1e-6)) * (legend_w - 1)))
        draw.line(
            [(tick_x, legend_y + legend_h - 1), (tick_x, legend_y + legend_h + tick_mark_h)],
            fill=(236, 240, 246, 214),
            width=tick_line_w,
        )
        label_bbox = draw.textbbox((0, 0), label, font=legend_label_font)
        label_w = label_bbox[2] - label_bbox[0]
        if index == 0:
            label_x = legend_x
        elif index == len(tick_values) - 1:
            label_x = legend_x + legend_w - label_w
        else:
            label_x = int(round(tick_x - label_w * 0.5))
            label_x = max(legend_x, min(label_x, legend_x + legend_w - label_w))
        draw.text(
            (label_x, tick_label_y),
            label,
            fill=(236, 240, 246, 236),
            font=legend_label_font,
        )
    draw.text(
        (caption_x, caption_y),
        BARCELONA_CAPTION,
        fill=(228, 232, 238, 228),
        font=caption_font,
    )
    return np.asarray(image, dtype=np.uint8)


def screen_blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - np.clip(base_rgb, 0.0, 1.0)) * (1.0 - np.clip(overlay_rgb, 0.0, 1.0))


def _bosnia_value_shade(
    base_rgb: np.ndarray,
    shade: np.ndarray,
    *,
    value_floor: float,
    value_gain: float,
    highlight_start: float,
    highlight_gain: float,
) -> np.ndarray:
    base = np.clip(np.asarray(base_rgb, dtype=np.float32), 0.0, 1.0)
    tone = np.clip(np.asarray(shade, dtype=np.float32), 0.0, 1.0)
    base_value = np.max(base, axis=2, keepdims=True)
    base_scale = np.divide(base, np.maximum(base_value, 1e-6))
    target_value = np.clip(
        base_value[:, :, 0] * (float(value_floor) + float(value_gain) * tone)
        + float(highlight_gain) * np.maximum(tone - float(highlight_start), 0.0),
        0.0,
        1.0,
    )
    return np.clip(base_scale * target_value[:, :, None], 0.0, 1.0)


def resolve_terrain_zscale(
    heightmap_m: np.ndarray,
    *,
    terrain_span_m: float,
    requested_zscale: float | None,
) -> float:
    if requested_zscale is not None:
        return float(requested_zscale)

    heights = np.asarray(heightmap_m, dtype=np.float32)
    land = heights[heights > SEA_LEVEL_M + 2.0]
    if land.size == 0:
        return 1.8

    relief_m = float(np.quantile(land, 0.999))
    relief_ratio = relief_m / max(float(terrain_span_m), 1.0)
    adaptive = 0.034 / max(relief_ratio, 1e-4)
    return float(np.clip(adaptive, 0.1, 0.85))


def combine_render_passes(
    color_rgba: np.ndarray,
    relief_rgba: np.ndarray,
    *,
    elevation_norm: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
    lowland_mask: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    color_rgb = np.asarray(color_rgba[:, :, :3], dtype=np.float32) / 255.0
    relief_blur_radius = float(np.clip(max(color_rgba.shape[0], color_rgba.shape[1]) / 1800.0, 1.8, 2.8))
    relief_rgb = np.asarray(
        Image.fromarray(relief_rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=relief_blur_radius)),
        dtype=np.float32,
    )[:, :, :3] / 255.0

    if elevation_norm is None:
        elevation_norm = np.full(color_rgb.shape[:2], 0.5, dtype=np.float32)
    else:
        elevation_norm = np.clip(np.asarray(elevation_norm, dtype=np.float32), 0.0, 1.0)

    if water_mask is None:
        water_mask = np.zeros(color_rgb.shape[:2], dtype=np.float32)
    else:
        water_mask = np.clip(np.asarray(water_mask, dtype=np.float32), 0.0, 1.0)

    if land_mask is None:
        land_mask = np.clip(1.0 - water_mask, 0.0, 1.0)
    else:
        land_mask = np.clip(np.asarray(land_mask, dtype=np.float32), 0.0, 1.0)
    sea_mask = np.clip(1.0 - land_mask, 0.0, 1.0).astype(np.float32)
    inland_water_mask = np.clip(water_mask - sea_mask, 0.0, 1.0).astype(np.float32)

    road_chroma = np.max(color_rgb, axis=2) - np.min(color_rgb, axis=2)
    bg_samples = color_rgb[(road_chroma < 0.02) & (land_mask > 0.5)]
    if bg_samples.size:
        road_bg = np.median(bg_samples, axis=0).astype(np.float32)
    else:
        road_bg = np.array([0.26, 0.27, 0.28], dtype=np.float32)
    road_strength = np.max(np.abs(color_rgb - road_bg[None, None, :]), axis=2)
    road_mask = _smoothstep(0.014, 0.112, road_strength) * _smoothstep(0.008, 0.154, road_chroma)
    road_mask = np.clip(
        np.maximum(road_mask, 0.74 * _smoothstep(0.016, 0.088, road_chroma)),
        0.0,
        1.0,
    ).astype(np.float32)
    road_mask *= land_mask
    road_glow = _blur_float_mask(
        road_mask,
        radius=max(6.0, float(max(color_rgba.shape[0], color_rgba.shape[1])) * 0.0028),
    )

    if lowland_mask is None:
        lowland_mask = 1.0 - _smoothstep(0.08, 0.24, elevation_norm)
    else:
        lowland_mask = np.clip(np.asarray(lowland_mask, dtype=np.float32), 0.0, 1.0)
    coastal_lowland = np.clip(lowland_mask - 0.45 * inland_water_mask - 0.85 * sea_mask, 0.0, 1.0).astype(np.float32)

    luminance = (
        relief_rgb[:, :, 0] * 0.2126
        + relief_rgb[:, :, 1] * 0.7152
        + relief_rgb[:, :, 2] * 0.0722
    )
    land_selector = land_mask > 0.5
    luminance_values = luminance[land_selector] if np.any(land_selector) else luminance.reshape(-1)
    lo = float(np.percentile(luminance_values, 6.0))
    hi = float(np.percentile(luminance_values, 94.0))
    tone = np.clip((luminance - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    tone = np.clip((tone - 0.5) * 1.08 + 0.5, 0.0, 1.0)
    tone = np.power(tone, 1.02, dtype=np.float32)

    highland = _smoothstep(0.18, 0.72, elevation_norm)
    relief_gray = luminance[:, :, None]
    terrain_seed = np.clip(0.24 * relief_rgb + 0.76 * relief_gray, 0.0, 1.0)
    terrain_seed = np.clip(
        terrain_seed * np.array([0.56, 0.40, 0.37], dtype=np.float32)[None, None, :]
        + np.array([0.04, 0.025, 0.025], dtype=np.float32)[None, None, :],
        0.0,
        1.0,
    )
    terrain_seed = np.clip(
        terrain_seed * (1.0 - 0.12 * highland[:, :, None])
        + np.array([0.58, 0.39, 0.32], dtype=np.float32)[None, None, :] * (0.12 * highland[:, :, None]),
        0.0,
        1.0,
    )
    terrain_rgb = np.clip(
        terrain_seed * (1.0 - 0.16 * coastal_lowland[:, :, None])
        + np.array([0.34, 0.25, 0.22], dtype=np.float32)[None, None, :] * (0.16 * coastal_lowland[:, :, None]),
        0.0,
        1.0,
    )
    terrain_rgb = np.clip(
        terrain_rgb * (1.0 - 0.42 * inland_water_mask[:, :, None])
        + np.array([0.35, 0.41, 0.49], dtype=np.float32)[None, None, :] * (0.42 * inland_water_mask[:, :, None]),
        0.0,
        1.0,
    )
    terrain_rgb = np.clip(
        terrain_rgb * (1.0 - 0.92 * sea_mask[:, :, None])
        + np.array([0.30, 0.36, 0.46], dtype=np.float32)[None, None, :] * (0.92 * sea_mask[:, :, None]),
        0.0,
        1.0,
    )
    terrain_rgb = _bosnia_value_shade(
        terrain_rgb,
        tone,
        value_floor=0.18,
        value_gain=0.34,
        highlight_start=0.72,
        highlight_gain=0.03,
    )

    road_rgb = _bosnia_value_shade(
        np.clip(np.power(np.clip(color_rgb, 0.0, 1.0), 0.96) * 1.08, 0.0, 1.0),
        tone,
        value_floor=0.94,
        value_gain=0.18,
        highlight_start=0.74,
        highlight_gain=0.05,
    )
    lit_roads = screen_blend(terrain_rgb * 0.82, road_rgb)
    road_mix = np.clip(0.94 * road_mask + 0.14 * road_glow, 0.0, 1.0).astype(np.float32)
    composite = terrain_rgb * (1.0 - road_mix[:, :, None]) + (0.88 * road_rgb + 0.12 * lit_roads) * road_mix[:, :, None]

    vignette_y, vignette_x = np.mgrid[0 : composite.shape[0], 0 : composite.shape[1]].astype(np.float32)
    vignette_x = vignette_x / max(composite.shape[1] - 1, 1)
    vignette_y = vignette_y / max(composite.shape[0] - 1, 1)
    radial = np.sqrt(((vignette_x - 0.50) / 0.84) ** 2 + ((vignette_y - 0.53) / 0.86) ** 2)
    composite *= (1.0 - 0.07 * _smoothstep(0.42, 1.04, radial))[:, :, None]
    sea_rgb = np.array([0.30, 0.38, 0.50], dtype=np.float32)
    land_alpha = np.clip(land_mask, 0.0, 1.0)[:, :, None]
    composite = composite * land_alpha + sea_rgb[None, None, :] * (1.0 - land_alpha)

    out = np.empty_like(color_rgba)
    out[:, :, :3] = np.round(np.clip(composite, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def _memory_safe_snapshot_size(snapshot_size: tuple[int, int], *, max_edge: int = 4096) -> tuple[int, int]:
    width, height = int(snapshot_size[0]), int(snapshot_size[1])
    largest = max(width, height)
    if largest <= int(max_edge):
        return width, height
    scale = float(max_edge) / float(largest)
    safe_width = max(512, int(round(width * scale)))
    safe_height = max(512, int(round(height * scale)))
    return safe_width, safe_height


def _memory_safe_pbr_state(state: dict[str, object]) -> dict[str, object]:
    safe = copy.deepcopy(state)
    safe["msaa"] = min(int(safe.get("msaa", 4)), 4)
    safe["shadow_map_res"] = min(int(safe.get("shadow_map_res", 2048)), 3072)

    height_ao = safe.get("height_ao")
    if isinstance(height_ao, dict):
        height_ao["steps"] = min(int(height_ao.get("steps", 24)), 24)
        height_ao["resolution_scale"] = min(float(height_ao.get("resolution_scale", 0.7)), 0.72)

    sun_visibility = safe.get("sun_visibility")
    if isinstance(sun_visibility, dict):
        sun_visibility["steps"] = min(int(sun_visibility.get("steps", 48)), 48)
        sun_visibility["resolution_scale"] = min(float(sun_visibility.get("resolution_scale", 0.8)), 0.72)

    volumetrics = safe.get("volumetrics")
    if isinstance(volumetrics, dict):
        volumetrics["half_res"] = True
        volumetrics["steps"] = min(int(volumetrics.get("steps", 40)), 40)
        volumetrics["shaft_intensity"] = min(float(volumetrics.get("shaft_intensity", 0.8)), 0.72)

    denoise = safe.get("denoise")
    if isinstance(denoise, dict):
        denoise["iterations"] = min(int(denoise.get("iterations", 3)), 3)

    return safe


def render_forge3d_snapshot(
    *,
    heightmap_path: Path,
    terrain_context_path: Path,
    overlay_path: Path,
    output_path: Path,
    viewer_size: tuple[int, int],
    snapshot_size: tuple[int, int],
    cam_phi: float,
    cam_theta: float,
    cam_fov: float,
    cam_radius_scale: float,
    zscale: float | None,
    terrain_span_m: float,
) -> None:
    heightmap = np.load(heightmap_path)
    with np.load(terrain_context_path) as terrain_context:
        water_mask_source = np.asarray(terrain_context["water_mask"], dtype=np.float32)
        lowland_mask_source = np.asarray(terrain_context["lowland_mask"], dtype=np.float32)
        if "land_mask" in terrain_context.files:
            land_mask_source = np.asarray(terrain_context["land_mask"], dtype=np.float32)
        else:
            land_mask_source = np.clip(1.0 - water_mask_source, 0.0, 1.0).astype(np.float32)
    finite_height = heightmap[np.isfinite(heightmap)]
    max_height = max(float(finite_height.max()) if finite_height.size else 0.0, 1e-6)
    elevation_norm_source = np.where(
        np.isfinite(heightmap),
        np.clip(heightmap / max_height, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    terrain_dim = float(max(heightmap.shape))
    resolved_zscale = resolve_terrain_zscale(
        heightmap,
        terrain_span_m=float(terrain_span_m),
        requested_zscale=zscale,
    )
    if zscale is None:
        relief_values = heightmap[np.isfinite(heightmap) & (heightmap > SEA_LEVEL_M + 2.0)]
        relief_q999 = float(np.quantile(relief_values, 0.999)) if relief_values.size else 0.0
        print(
            "[Barcelona] adaptive terrain relief "
            f"zscale={resolved_zscale:.2f} "
            f"(q99.9 relief {relief_q999:.0f}m over {float(terrain_span_m)/1000.0:.1f} km span)"
        )
    color_terrain_state = {
        "phi": float(cam_phi),
        "theta": float(cam_theta),
        "radius": terrain_dim * float(cam_radius_scale),
        "fov": float(cam_fov),
        "zscale": float(resolved_zscale),
        "sun_azimuth": 312.0,
        "sun_elevation": 25.0,
        "sun_intensity": 2.10,
        "ambient": 0.18,
        "shadow": 0.55,
        "background": [0.058, 0.061, 0.068],
    }
    color_pbr_state = {
        "enabled": True,
        "shadow_technique": "pcss",
        "shadow_map_res": 4096,
        "exposure": 0.96,
        "msaa": 8,
        "ibl_intensity": 0.06,
        "normal_strength": 1.15,
        "height_ao": {
            "enabled": True,
            "directions": 12,
            "steps": 28,
            "max_distance": 220.0,
            "strength": 0.14,
            "resolution_scale": 0.82,
        },
        "sun_visibility": {
            "enabled": True,
            "mode": "soft",
            "samples": 2,
            "steps": 40,
            "max_distance": 2000.0,
            "softness": 0.34,
            "bias": 0.0030,
            "resolution_scale": 0.88,
        },
        "materials": {
            "rock_enabled": True,
            "rock_slope_min": 32.0,
            "wetness_enabled": True,
            "wetness_strength": 0.14,
        },
        "volumetrics": {
            "enabled": True,
            "mode": "height",
            "density": 0.018,
            "height_falloff": 0.018,
            "scattering": 0.58,
            "absorption": 0.06,
            "light_shafts": True,
            "shaft_intensity": 0.72,
            "steps": 48,
            "half_res": False,
        },
        "denoise": {
            "enabled": True,
            "method": "atrous",
            "iterations": 4,
            "sigma_color": 0.18,
        },
        "lens_effects": {
            "enabled": True,
            "vignette_strength": 0.10,
            "vignette_radius": 0.82,
            "vignette_softness": 0.54,
            "distortion": 0.0,
            "chromatic_aberration": 0.0,
        },
        "sky": {
            "enabled": True,
            "turbidity": 2.0,
            "ground_albedo": 0.28,
            "sun_intensity": 1.06,
            "aerial_perspective": True,
            "sky_exposure": 0.98,
        },
        "tonemap": {
            "operator": "aces",
            "white_point": 5.2,
            "white_balance_enabled": True,
            "temperature": 6100.0,
            "tint": 0.01,
        },
    }
    relief_terrain_state = {
        **color_terrain_state,
        "sun_elevation": 20.0,
        "sun_intensity": 2.50,
        "ambient": 0.11,
        "shadow": 0.12,
    }
    relief_pbr_state = {
        **color_pbr_state,
        "exposure": 1.00,
        "ibl_intensity": 0.04,
        "normal_strength": 1.94,
        "height_ao": {
            "enabled": True,
            "directions": 12,
            "steps": 32,
            "max_distance": 280.0,
            "strength": 0.44,
            "resolution_scale": 0.84,
        },
        "sun_visibility": {
            "enabled": True,
            "mode": "soft",
            "samples": 2,
            "steps": 72,
            "max_distance": 2600.0,
            "softness": 0.36,
            "bias": 0.0030,
            "resolution_scale": 0.90,
        },
        "volumetrics": {
            "enabled": True,
            "mode": "height",
            "density": 0.024,
            "height_falloff": 0.020,
            "scattering": 0.62,
            "absorption": 0.06,
            "light_shafts": True,
            "shaft_intensity": 0.92,
            "steps": 56,
            "half_res": False,
        },
        "denoise": {
            "enabled": True,
            "method": "atrous",
            "iterations": 4,
            "sigma_color": 0.16,
        },
        "lens_effects": {
            "enabled": True,
            "vignette_strength": 0.12,
            "vignette_radius": 0.84,
            "vignette_softness": 0.58,
            "distortion": 0.0,
            "chromatic_aberration": 0.0,
        },
        "tonemap": {
            "operator": "aces",
            "white_point": 5.4,
            "white_balance_enabled": True,
            "temperature": 6000.0,
            "tint": 0.0,
        },
    }

    def _run_snapshot_attempt(
        *,
        snapshot_dims: tuple[int, int],
        color_pbr: dict[str, object],
        relief_pbr: dict[str, object],
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="forge3d_barcelona_pulse_") as temp_dir_name:
            color_path = Path(temp_dir_name) / "roads.png"
            relief_path = Path(temp_dir_name) / "relief.png"

            with f3d.open_viewer_async(
                terrain_path=heightmap_path,
                width=int(viewer_size[0]),
                height=int(viewer_size[1]),
                fov_deg=float(cam_fov),
                timeout=45.0,
            ) as viewer:
                viewer.send_ipc({"cmd": "set_taa_enabled", "enabled": True})
                viewer.send_ipc(
                    {
                        "cmd": "set_taa_params",
                        "history_weight": 0.92,
                        "jitter_scale": 1.0,
                        "enable_jitter": True,
                    }
                )
                viewer.send_ipc({"cmd": "set_terrain", **color_terrain_state})
                viewer.send_ipc({"cmd": "set_terrain_pbr", **color_pbr})
                viewer.load_overlay(
                    name="barcelona_estimated_drive_time",
                    path=overlay_path,
                    extent=(0.0, 0.0, 1.0, 1.0),
                    opacity=1.0,
                    z_order=0,
                    preserve_colors=True,
                )
                viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
                viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
                viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
                time.sleep(VIEWER_SETTLE_SECONDS)
                viewer.snapshot(color_path, width=int(snapshot_dims[0]), height=int(snapshot_dims[1]))

                viewer.send_ipc({"cmd": "set_terrain", **relief_terrain_state})
                viewer.send_ipc({"cmd": "set_terrain_pbr", **relief_pbr})
                viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": False})
                time.sleep(VIEWER_SETTLE_SECONDS)
                viewer.snapshot(relief_path, width=int(snapshot_dims[0]), height=int(snapshot_dims[1]))

            color_rgba = np.asarray(Image.open(color_path).convert("RGBA"), dtype=np.uint8)
            relief_rgba = np.asarray(Image.open(relief_path).convert("RGBA"), dtype=np.uint8)
            resized_size = (int(snapshot_dims[0]), int(snapshot_dims[1]))
            combined = combine_render_passes(
                color_rgba,
                relief_rgba,
                elevation_norm=_resize_float_raster(elevation_norm_source, size=resized_size),
                water_mask=_resize_float_raster(water_mask_source, size=resized_size),
                lowland_mask=_resize_float_raster(lowland_mask_source, size=resized_size),
                land_mask=_resize_float_raster(land_mask_source, size=resized_size),
            )
            Image.fromarray(combined, mode="RGBA").save(output_path)

    attempts = [
        {
            "name": "full-quality",
            "snapshot_dims": (int(snapshot_size[0]), int(snapshot_size[1])),
            "color_pbr": color_pbr_state,
            "relief_pbr": relief_pbr_state,
        },
        {
            "name": "memory-safe",
            "snapshot_dims": _memory_safe_snapshot_size(snapshot_size),
            "color_pbr": _memory_safe_pbr_state(color_pbr_state),
            "relief_pbr": _memory_safe_pbr_state(relief_pbr_state),
        },
    ]

    last_error: Exception | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for index, attempt in enumerate(attempts):
        try:
            if index > 0:
                dims = attempt["snapshot_dims"]
                print(
                    f"[Barcelona] retrying render with {attempt['name']} snapshot "
                    f"{int(dims[0])}x{int(dims[1])}"
                )
            _run_snapshot_attempt(
                snapshot_dims=attempt["snapshot_dims"],
                color_pbr=attempt["color_pbr"],
                relief_pbr=attempt["relief_pbr"],
            )
            return
        except Exception as exc:
            last_error = exc
            retryable = isinstance(exc, ViewerError) or "Connection closed by viewer" in str(exc)
            if not retryable or index == len(attempts) - 1:
                raise
            print(f"[Barcelona] viewer closed during snapshot ({exc}); falling back to safer snapshot settings")

    if last_error is not None:
        raise last_error


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    origin_lon = float(args.origin_lon)
    origin_lat = float(args.origin_lat)
    epsg = utm_epsg_for_lon_lat(origin_lon, origin_lat)
    to_projected = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    origin_x, origin_y = map(float, to_projected.transform(origin_lon, origin_lat))

    print(f"[Barcelona] fetching OSM roads within {args.query_radius_km:.1f} km")
    ways = fetch_drivable_ways(
        lon=origin_lon,
        lat=origin_lat,
        radius_m=float(args.query_radius_km) * 1000.0,
        cache_dir=cache_dir,
        refresh=bool(args.refresh_osm),
    )
    print(f"[Barcelona] fetched {len(ways):,} OSM ways")

    nodes, segments, weighted_adj, unweighted_adj = build_drive_graph(ways, transformer=to_projected)
    component = largest_component_nodes(unweighted_adj)
    if not component:
        raise SystemExit("No connected drivable road component was found.")
    start_node = nearest_node(nodes, x=origin_x, y=origin_y, candidates=component)
    node_times_s = dijkstra_seconds(weighted_adj, start=start_node, allowed=component)
    timed_segments = select_timed_segments(
        segments,
        component=component,
        node_times_s=node_times_s,
        cutoff_min=float(args.max_mins),
    )
    if not timed_segments:
        raise SystemExit("No roads were reachable within the requested travel-time cutoff.")

    extent = make_square_extent(timed_segments)
    print(f"[Barcelona] reachable segments within cutoff: {len(timed_segments):,}")
    print(f"[Barcelona] building {args.terrain_size}x{args.terrain_size} Terrarium DEM at z{args.terrain_zoom}")
    raw_heightmap = build_heightmap(
        extent,
        size=int(args.terrain_size),
        zoom=int(args.terrain_zoom),
        to_wgs84=to_wgs84,
        cache_dir=cache_dir,
        refresh=bool(args.refresh_dem),
    )
    print("[Barcelona] fetching OSM Spain boundary vector")
    spain_boundary_projected, boundary_path = load_spain_boundary(
        args.boundary_data,
        cache_dir=cache_dir,
        target_crs=f"EPSG:{epsg}",
        refresh=bool(args.refresh_boundary),
    )
    print(f"[Barcelona] boundary vector: {display_path(boundary_path)}")
    print("[Barcelona] fetching OSM coastline segments for the Barcelona extent")
    coastline_ways, coastline_path = fetch_osm_coastline_ways(
        extent=extent,
        to_wgs84=to_wgs84,
        cache_dir=cache_dir,
        refresh=bool(args.refresh_boundary),
    )
    print(f"[Barcelona] coastline vector: {display_path(coastline_path)}")
    print("[Barcelona] building a land-only OSM Spain mask from the coastline")
    land_geometry = build_land_geometry_from_osm_coastline(
        coastline_ways,
        extent=extent,
        to_projected=to_projected,
        seed_point_xy=(origin_x, origin_y),
    ).intersection(spain_boundary_projected)
    if land_geometry.is_empty:
        raise RuntimeError("The OSM Spain land mask was empty after intersecting the coastline with the Spain boundary")
    print("[Barcelona] masking Barcelona DEM to the OSM Spain land mask")
    terrain = prepare_terrain(
        raw_heightmap,
        extent=extent,
        land_geometry=land_geometry,
    )
    print(f"[Barcelona] rasterizing {args.overlay_size}x{args.overlay_size} road overlay")
    overlay = build_road_overlay(
        timed_segments,
        extent=extent,
        overlay_size=int(args.overlay_size),
        max_mins=float(args.max_mins),
        origin_x=origin_x,
        origin_y=origin_y,
    )

    prepared_prefix = cache_dir / "barcelona_pulse"
    heightmap_path = prepared_prefix.with_suffix(".npy")
    terrain_context_path = prepared_prefix.with_name(prepared_prefix.name + "_terrain_context.npz")
    overlay_path = prepared_prefix.with_suffix(".png")
    meta_path = prepared_prefix.with_suffix(".json")
    np.save(heightmap_path, terrain.heightmap)
    np.savez_compressed(
        terrain_context_path,
        water_mask=terrain.water_mask,
        lowland_mask=terrain.lowland_mask,
        land_mask=terrain.land_mask,
    )
    overlay.save(overlay_path)
    meta_path.write_text(
        json.dumps(
            {
                "extent": extent.to_dict(),
                "origin": {"lon": origin_lon, "lat": origin_lat, "x": origin_x, "y": origin_y},
                "max_mins": float(args.max_mins),
                "terrain_zoom": int(args.terrain_zoom),
                "terrain_size": int(args.terrain_size),
                "overlay_size": int(args.overlay_size),
                "boundary_vector": display_path(boundary_path),
                "coastline_vector": display_path(coastline_path),
                "reachable_segment_count": len(timed_segments),
                "graph_node_count": len(nodes),
                "graph_edge_count": len(segments),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Barcelona] prepared DEM: {display_path(heightmap_path)}")
    print(f"[Barcelona] prepared overlay: {display_path(overlay_path)}")

    if args.prepare_only:
        print(f"[Barcelona] wrote preparation metadata: {display_path(meta_path)}")
        return 0

    raw_output = args.output.resolve().with_name(args.output.stem + "_raw.png")
    render_forge3d_snapshot(
        heightmap_path=heightmap_path,
        terrain_context_path=terrain_context_path,
        overlay_path=overlay_path,
        output_path=raw_output,
        viewer_size=(int(args.viewer_width), int(args.viewer_height)),
        snapshot_size=(int(args.snapshot_width), int(args.snapshot_height)),
        cam_phi=float(args.cam_phi),
        cam_theta=float(args.cam_theta),
        cam_fov=float(args.cam_fov),
        cam_radius_scale=float(args.cam_radius_scale),
        zscale=None if args.zscale is None else float(args.zscale),
        terrain_span_m=float(extent.width),
    )

    raw_rgba = np.asarray(Image.open(raw_output).convert("RGBA"), dtype=np.uint8)
    final_rgba = compose_poster(raw_rgba, max_mins=float(args.max_mins))
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_rgba, mode="RGBA").save(args.output.resolve())
    print(f"[Barcelona] raw render: {display_path(raw_output)}")
    print(f"[Barcelona] final poster: {display_path(args.output.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
