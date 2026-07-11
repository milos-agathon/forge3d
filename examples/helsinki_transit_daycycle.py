#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import shutil
import tempfile
import zlib
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform, unary_union

from _import_shim import ensure_repo_import

ensure_repo_import()

import osm_city_daycycle as day
import osm_city_demo as city


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "helsinki_transit_daycycle"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "examples"
    / "out"
    / "helsinki_transit_daycycle"
    / "helsinki_transit_daycycle.mp4"
)

DEFAULT_LON = 24.94145
DEFAULT_LAT = 60.17195
DEFAULT_RADIUS_M = 2000.0
DEFAULT_CLOCK_START_HOUR = 1.0
DEFAULT_CLOCK_END_HOUR = 24.0

HELSINKI_WFS_URL = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
HELSINKI_BUILDINGS_TYPENAME = "avoindata:Rakennukset_alue_rekisteritiedot"
HELSINKI_TRAFFIC_TYPENAME = "avoindata:Ajoneuvoliikenne_liikennemaarat_viiva"
HSL_GTFS_URL = "https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip"

USER_AGENT = "forge3d-helsinki-transit-daycycle/1.0"
DEFAULT_OSM_TIMEOUT_SECONDS = 30.0
TRANSIT_ROUTE_TYPES = {
    0: (0x62, 0xF7, 0xD4),  # tram
    1: (0xFF, 0x73, 0x34),  # metro
    2: (0xB2, 0x87, 0xFF),  # rail
    3: (0x44, 0xD4, 0xFF),  # bus
    4: (0x8A, 0xE2, 0xFF),  # ferry
}
TRANSIT_DEFAULT_RGB = (0xF7, 0xFB, 0xFF)
CAR_FLOW_RGB = (0xFF, 0xC9, 0x5C)
CAR_FLOW_HEAVY_RGB = (0xFF, 0x76, 0x52)
TITLE_TEXT_RGB = (0xF6, 0xF4, 0xEA)
TITLE_BOUNDARY_RGB = (0x06, 0x08, 0x0A)
TITLE_SUBTEXT_RGB = (0xD6, 0xD2, 0xC8)
TITLE_HEADER_RGB = (0x06, 0x08, 0x0A)
HELSINKI_HEADER_HEIGHT_RATIO = 0.17
DARK_MAP_BRIGHTNESS = 0.20
DARK_MAP_BLACK_LEVEL = 5.0
CAR_FLOW_ANIMATION_SPEED_SCALE = 0.018
CAR_FLOW_LANE_OFFSET_M = 2.2
CAR_FLOW_MIN_SPACING_M = 5.5
CAR_FLOW_MAX_PARTICLES = 1600
TRAFFIC_HOURLY_WEIGHTS = np.asarray(
    [
        0.008,
        0.005,
        0.004,
        0.005,
        0.010,
        0.026,
        0.054,
        0.075,
        0.066,
        0.047,
        0.043,
        0.047,
        0.052,
        0.055,
        0.062,
        0.072,
        0.083,
        0.079,
        0.064,
        0.046,
        0.033,
        0.025,
        0.018,
        0.013,
    ],
    dtype=np.float32,
)
TRAFFIC_HOURLY_WEIGHTS = TRAFFIC_HOURLY_WEIGHTS / float(np.sum(TRAFFIC_HOURLY_WEIGHTS))
MAP_SURFACE_COLORS = {
    city.COLORS["base"],
    city.COLORS["landuse"],
    city.COLORS["park"],
    city.COLORS["water"],
    city.COLORS["road"],
    city.COLORS["road_hi"],
}


@dataclass(frozen=True)
class MetricContext:
    epsg: int
    to_metric: Transformer
    to_wgs84: Transformer
    center_xy: tuple[float, float]
    aoi_m: object
    bbox_wgs84: tuple[float, float, float, float]


@dataclass(frozen=True)
class TransitTrip:
    trip_id: str
    route_id: str
    route_type: int
    route_name: str
    route_rgb: tuple[int, int, int]
    start_seconds: int
    end_seconds: int
    points_xy: np.ndarray
    cumulative_m: np.ndarray


@dataclass(frozen=True)
class RoadTrafficSegment:
    segment_id: str
    street_name: str
    daily_vehicles: float
    heavy_share: float
    speed_mps: float
    points_xy: np.ndarray
    cumulative_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Helsinki daily transit timelapse with OSM basemap layers, "
            "official Helsinki building footprints, and HSL scheduled vehicle motion."
        )
    )
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="AOI center longitude.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="AOI center latitude.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M, help="AOI radius in meters.")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frames", type=int, default=240, help="Number of animation frames to render.")
    parser.add_argument("--fps", type=int, default=24, help="Video frame rate.")
    parser.add_argument("--supersample", type=int, default=1, help="Internal render scale before downsampling.")
    parser.add_argument("--clock-start-hour", type=float, default=DEFAULT_CLOCK_START_HOUR, help="Timelapse start hour.")
    parser.add_argument("--clock-end-hour", type=float, default=DEFAULT_CLOCK_END_HOUR, help="Timelapse end hour.")
    parser.add_argument(
        "--service-date",
        type=str,
        default=None,
        help="GTFS service date as YYYY-MM-DD. Defaults to today's local date.",
    )
    parser.add_argument(
        "--route-types",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated GTFS route_type values to animate.",
    )
    parser.add_argument(
        "--gtfs-zip",
        type=Path,
        default=None,
        help="Optional local HSL GTFS zip. Defaults to the official HSL download.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MP4 path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Optional directory for rendered PNG frames.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep the PNG frame sequence after encoding.")
    parser.add_argument("--refresh-osm", action="store_true", help="Ignore cached Overpass responses.")
    parser.add_argument("--refresh-buildings", action="store_true", help="Ignore cached Helsinki WFS buildings.")
    parser.add_argument("--refresh-traffic", action="store_true", help="Ignore cached Helsinki traffic-count lines.")
    parser.add_argument("--refresh-gtfs", action="store_true", help="Re-download the official HSL GTFS zip.")
    parser.add_argument(
        "--max-car-particles",
        type=int,
        default=CAR_FLOW_MAX_PARTICLES,
        help="Deprecated alias for --max-vehicle-particles.",
    )
    parser.add_argument(
        "--max-vehicle-particles",
        type=int,
        default=None,
        help="Maximum modeled road-vehicle particles to draw per frame.",
    )
    parser.add_argument(
        "--map-brightness",
        type=float,
        default=DARK_MAP_BRIGHTNESS,
        help="Postprocess brightness for the grayscale basemap before drawing colored dots.",
    )
    parser.add_argument(
        "--osm-timeout",
        type=float,
        default=DEFAULT_OSM_TIMEOUT_SECONDS,
        help="Per-endpoint Overpass timeout in seconds for optional OSM basemap layers.",
    )
    return parser.parse_args()


def parse_service_date(value: str | None) -> dt.date:
    if value:
        return dt.date.fromisoformat(value)
    return dt.date.today()


def parse_route_types(value: str) -> set[int]:
    route_types: set[int] = set()
    for part in value.split(","):
        text = part.strip()
        if text:
            route_types.add(int(text))
    return route_types or set(TRANSIT_ROUTE_TYPES)


def parse_gtfs_time_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hour, minute, second = (int(part) for part in parts)
    except ValueError:
        return None
    if minute < 0 or minute > 59 or second < 0 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


def build_metric_context(lon: float, lat: float, radius_m: float) -> MetricContext:
    epsg = city.utm_epsg_for_lon_lat(lon, lat)
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    center_m = transform(to_metric.transform, Point(lon, lat))
    center_xy = tuple(center_m.coords[0])
    aoi_m = Point(*center_xy).buffer(float(radius_m), resolution=96)
    bbox_wgs84 = transform(to_wgs84.transform, aoi_m).bounds
    return MetricContext(
        epsg=epsg,
        to_metric=to_metric,
        to_wgs84=to_wgs84,
        center_xy=(float(center_xy[0]), float(center_xy[1])),
        aoi_m=aoi_m,
        bbox_wgs84=tuple(float(v) for v in bbox_wgs84),
    )


def request_json(url: str, cache_path: Path, *, refresh: bool, label: str) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    last_error: Exception | None = None
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.load(response)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            sleep(1.5 * (attempt + 1))
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    raise RuntimeError(f"Could not fetch {label}: {last_error}") from last_error


def download_file(url: str, cache_path: Path, *, refresh: bool, label: str) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return cache_path

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=240) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            tmp_path.replace(cache_path)
            return cache_path
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            sleep(1.5 * (attempt + 1))
    if cache_path.exists():
        return cache_path
    raise RuntimeError(f"Could not download {label}: {last_error}") from last_error


def fetch_overpass_query_fast(query: str, cache_path: Path, refresh: bool, *, timeout_seconds: float) -> list[dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])

    payload = None
    last_error: Exception | None = None
    for endpoint in city.OVERPASS_URLS:
        request = Request(
            endpoint,
            data=query.encode("utf-8"),
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urlopen(request, timeout=max(3.0, float(timeout_seconds))) as response:
                payload = json.load(response)
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

    if payload is None:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])
        raise RuntimeError(f"Overpass request failed for {cache_path.name}: {last_error}") from last_error

    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload.get("elements", [])


def helsinki_buildings_url(ctx: MetricContext) -> str:
    west, south, east, north = ctx.bbox_wgs84
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": HELSINKI_BUILDINGS_TYPENAME,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f},EPSG:4326",
        "count": "1000000",
    }
    return HELSINKI_WFS_URL + "?" + urlencode(params)


def fetch_official_buildings(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    refresh: bool,
) -> list[dict]:
    slug = city.cache_slug(lon, lat, radius_m)
    cache_path = CACHE_DIR / "official_buildings" / f"{slug}_rakennukset.geojson"
    payload = request_json(
        helsinki_buildings_url(ctx),
        cache_path,
        refresh=refresh,
        label="official Helsinki buildings",
    )
    features: list[dict] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        geom_m = transform(ctx.to_metric.transform, geom)
        clipped = city._extract_polygonal(city._fix_geom(geom_m.intersection(ctx.aoi_m)))
        if clipped is None or clipped.is_empty:
            continue
        features.append(
            {
                "geometry": clipped,
                "tags": dict(feature.get("properties", {})),
            }
        )
    return features


def official_building_height(tags: dict[str, object], footprint_area_m2: float) -> float:
    floors = city.parse_osm_numeric(tags.get("i_kerrlkm"))
    if floors is not None and math.isfinite(floors) and floors > 0:
        return float(np.clip(floors * 3.2, 3.0, 80.0))

    floor_area = city.parse_osm_numeric(tags.get("i_kerrosala"))
    if floor_area is not None and math.isfinite(floor_area) and footprint_area_m2 > 5.0:
        estimated_floors = floor_area / max(float(footprint_area_m2), 1.0)
        if estimated_floors > 0.6:
            return float(np.clip(estimated_floors * 3.0, 3.0, 70.0))

    usage = str(tags.get("tyyppi") or tags.get("c_kayttark") or "").lower()
    if "asuin" in usage:
        return 12.0
    if "toimisto" in usage or "liike" in usage or "julkinen" in usage:
        return 15.0
    return 9.5


def helsinki_traffic_counts_url(ctx: MetricContext) -> str:
    west, south, east, north = ctx.bbox_wgs84
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": HELSINKI_TRAFFIC_TYPENAME,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f},EPSG:4326",
        "count": "1000000",
    }
    return HELSINKI_WFS_URL + "?" + urlencode(params)


def traffic_daily_vehicle_count(tags: dict[str, object]) -> float:
    for key in ("syksyn_kavl", "autot"):
        value = city.parse_osm_numeric(tags.get(key))
        if value is not None and math.isfinite(value) and value > 0:
            return float(value)

    total = 0.0
    for key in ("ha", "pa", "ka", "ra", "la", "mp", "rv"):
        value = city.parse_osm_numeric(tags.get(key))
        if value is not None and math.isfinite(value) and value > 0:
            total += float(value)
    return total


def traffic_heavy_share(tags: dict[str, object]) -> float:
    percent = city.parse_osm_numeric(tags.get("raskas_liik_pros"))
    if percent is not None and math.isfinite(percent):
        return float(np.clip(percent / 100.0, 0.0, 0.7))

    total = traffic_daily_vehicle_count(tags)
    if total <= 0:
        return 0.0
    heavy = 0.0
    for key in ("ka", "ra", "la"):
        value = city.parse_osm_numeric(tags.get(key))
        if value is not None and math.isfinite(value) and value > 0:
            heavy += float(value)
    return float(np.clip(heavy / total, 0.0, 0.7))


def traffic_segment_speed_mps(tags: dict[str, object]) -> float:
    street_class = city.parse_osm_numeric(tags.get("kl"))
    if street_class is not None and math.isfinite(street_class):
        klass = int(street_class)
        if klass <= 2:
            return 13.9
        if klass == 3:
            return 11.1
        if klass == 4:
            return 8.3
    return 7.4


def line_parts(geom) -> list[LineString]:
    linear = city._extract_linear(city._fix_geom(geom))
    if linear is None or linear.is_empty:
        return []
    if isinstance(linear, LineString):
        return [linear]
    return [part for part in getattr(linear, "geoms", []) if isinstance(part, LineString) and not part.is_empty]


def fetch_official_traffic_segments(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    refresh: bool,
) -> list[RoadTrafficSegment]:
    slug = city.cache_slug(lon, lat, radius_m)
    cache_path = CACHE_DIR / "official_traffic" / f"{slug}_vehicle_counts.geojson"
    payload = request_json(
        helsinki_traffic_counts_url(ctx),
        cache_path,
        refresh=refresh,
        label="official Helsinki vehicle traffic counts",
    )
    segments: list[RoadTrafficSegment] = []
    source_feature_count = 0
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        tags = dict(feature.get("properties", {}))
        daily_vehicles = traffic_daily_vehicle_count(tags)
        if daily_vehicles <= 0:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        geom_m = transform(ctx.to_metric.transform, geom)
        clipped = city._extract_linear(city._fix_geom(geom_m.intersection(ctx.aoi_m)))
        if clipped is None or clipped.is_empty:
            continue
        local_geom = city.localize_feature({"geometry": clipped, "tags": tags}, ctx.center_xy)["geometry"]
        part_rows: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for part_index, part in enumerate(line_parts(local_geom)):
            simplified = city.simplify_geom(part, 0.45)
            coords = np.asarray([(float(x), float(y)) for x, y, *_ in simplified.coords], dtype=np.float32)
            if coords.shape[0] < 2:
                continue
            cumulative = cumulative_distances(coords)
            length_m = float(cumulative[-1]) if cumulative.size else 0.0
            if length_m < 12.0:
                continue
            part_rows.append((part_index, coords, cumulative, length_m))
        if not part_rows:
            continue

        source_feature_count += 1
        total_length = sum(row[3] for row in part_rows)
        base_id = str(feature.get("id") or tags.get("id") or tags.get("linkki") or source_feature_count)
        street_name = str(tags.get("katu") or "")
        heavy_share = traffic_heavy_share(tags)
        speed_mps = traffic_segment_speed_mps(tags)
        for part_index, coords, cumulative, length_m in part_rows:
            count_share = length_m / max(total_length, 1.0)
            segments.append(
                RoadTrafficSegment(
                    segment_id=f"{base_id}:{part_index}",
                    street_name=street_name,
                    daily_vehicles=float(daily_vehicles) * count_share,
                    heavy_share=heavy_share,
                    speed_mps=speed_mps,
                    points_xy=coords,
                    cumulative_m=cumulative,
                )
            )

    total_daily = sum(segment.daily_vehicles for segment in segments)
    print(
        f"[Helsinki] official traffic count lines={len(segments)} "
        f"daily modeled vehicles={total_daily:,.0f}"
    )
    return segments


def build_osm_surface_layers(
    ctx: MetricContext,
    *,
    lon: float,
    lat: float,
    radius_m: float,
    refresh_osm: bool,
    osm_timeout: float,
) -> list[city.SurfaceLayer]:
    west, south, east, north = ctx.bbox_wgs84
    bbox = (west, south, east, north)
    slug = city.cache_slug(lon, lat, radius_m)

    def cached(name: str, expr: str, *, include_relations: bool) -> list[dict]:
        query = city.overpass_query(bbox, expr, include_relations=include_relations)
        cache_path = CACHE_DIR / "osm" / f"{slug}_{name}.json"
        try:
            return fetch_overpass_query_fast(query, cache_path, refresh_osm, timeout_seconds=osm_timeout)
        except RuntimeError as exc:
            print(f"[Helsinki] OSM warning: {exc}")
            return []

    landuse = city.parse_polygon_features(
        cached("landuse", '["landuse"]', include_relations=False),
        ctx.to_metric,
        ctx.aoi_m,
    )
    park_elements = cached("parks", '["leisure"="park"]', include_relations=True)
    park_elements += cached("natural_green", '["natural"~"^(wood|scrub)$"]', include_relations=True)
    parks = city.parse_polygon_features(park_elements, ctx.to_metric, ctx.aoi_m)
    water_elements = cached("water", '["natural"="water"]', include_relations=True)
    water_elements += cached("riverbanks", '["water"="river"]', include_relations=True)
    water = city.parse_polygon_features(water_elements, ctx.to_metric, ctx.aoi_m)
    roads = city.parse_line_features(
        cached("roads", '["highway"]', include_relations=False),
        ctx.to_metric,
        ctx.aoi_m,
    )
    rails = city.parse_line_features(
        cached("rails", '["railway"]', include_relations=False),
        ctx.to_metric,
        ctx.aoi_m,
    )

    landuse_local = [city.localize_feature(feature, ctx.center_xy) for feature in landuse]
    park_geoms = [
        city.localize_feature(feature, ctx.center_xy)
        for feature in landuse
        if str(feature["tags"].get("landuse", "")).lower() in city.GREEN_LANDUSE
    ]
    park_geoms.extend(city.localize_feature(feature, ctx.center_xy) for feature in parks)
    water_local = [city.localize_feature(feature, ctx.center_xy) for feature in water]
    roads_local = [city.localize_feature(feature, ctx.center_xy) for feature in roads]
    rails_local = [city.localize_feature(feature, ctx.center_xy) for feature in rails]

    print(
        f"[Helsinki] OSM landuse={len(landuse_local)} parks={len(park_geoms)} "
        f"water={len(water_local)} roads={len(roads_local)} rails={len(rails_local)}"
    )

    surfaces: list[city.SurfaceLayer] = []

    def append_surface(
        geom,
        rgba: tuple[int, int, int, int],
        *,
        elevation: float,
        specular: float = 0.0,
        reflectivity: float = 0.0,
    ) -> None:
        polygonal = city._extract_polygonal(city._fix_geom(geom))
        if polygonal is None or polygonal.is_empty:
            return
        surfaces.append(
            city.SurfaceLayer(
                geometry=polygonal,
                rgba=rgba,
                elevation=float(elevation),
                specular=float(specular),
                reflectivity=float(reflectivity),
            )
        )

    append_surface(Point(0.0, 0.0).buffer(radius_m, resolution=96), city.COLORS["base"], elevation=0.02)
    if landuse_local:
        append_surface(city.merge_surface_geometry(landuse_local, simplify_tolerance=1.4), city.COLORS["landuse"], elevation=0.5)
    if park_geoms:
        append_surface(city.merge_surface_geometry(park_geoms, simplify_tolerance=1.2), city.COLORS["park"], elevation=0.51)
    if water_local:
        append_surface(
            city.merge_surface_geometry(water_local, simplify_tolerance=0.9),
            city.COLORS["water"],
            elevation=-0.5,
            specular=city.WATER_SPECULAR_INTENSITY,
            reflectivity=0.22,
        )
    if roads_local:
        roads_union = unary_union([feature["geometry"] for feature in roads_local])
        append_surface(
            city.simplify_geom(
                roads_union.buffer(
                    3.0,
                    cap_style=city.BufferCapStyle.round,
                    join_style=city.BufferJoinStyle.round,
                ),
                0.7,
            ),
            city.COLORS["road"],
            elevation=1.0,
        )
        append_surface(
            city.simplify_geom(
                roads_union.buffer(
                    1.3,
                    cap_style=city.BufferCapStyle.round,
                    join_style=city.BufferJoinStyle.round,
                ),
                0.5,
            ),
            city.COLORS["road_hi"],
            elevation=1.12,
        )
    if rails_local:
        rails_union = unary_union([feature["geometry"] for feature in rails_local])
        append_surface(
            city.simplify_geom(
                rails_union.buffer(
                    2.0,
                    cap_style=city.BufferCapStyle.round,
                    join_style=city.BufferJoinStyle.round,
                ),
                0.6,
            ),
            city.COLORS["road"],
            elevation=1.0,
        )
    return surfaces


def build_official_building_layers(
    features_m: list[dict],
    ctx: MetricContext,
    *,
    radius_m: float,
) -> tuple[list[city.MeshLayer], list[city.RoofOutlineLayer], list[city.FocusLandmark]]:
    localized: list[dict] = []
    for feature in features_m:
        local_feature = city.localize_feature(feature, ctx.center_xy)
        geom = city.prepare_polygonal_geom(local_feature["geometry"], simplify_tolerance=0.18)
        if geom is None or geom.is_empty:
            continue
        tags = dict(local_feature["tags"])
        tags["_height_m"] = official_building_height(tags, abs(float(geom.area)))
        localized.append({"geometry": geom, "tags": tags})

    print(f"[Helsinki] official buildings={len(localized)}")

    binned: dict[str, list[dict]] = {"low": [], "mid": [], "high": [], "landmark": []}
    roof_outlines: list[city.RoofOutlineLayer] = []
    for feature in localized:
        height = float(feature["tags"].get("_height_m", 12.0))
        binned[city.building_bin(height)].append(feature)
        roof_outlines.append(city.RoofOutlineLayer(geometry=feature["geometry"], elevation=height))

    meshes: list[city.MeshLayer] = []
    for bin_name, color_key, roof_outline in (
        ("low", "building_low", True),
        ("mid", "building_mid", False),
        ("high", "building_high", False),
        ("landmark", "building_landmark", False),
    ):
        features = binned[bin_name]
        if not features:
            continue
        geojson = city.feature_collection(
            features,
            height_key="_height_m",
            default_height=12.0,
            simplify_tolerance=0.75,
        )
        layer = city.mesh_from_geojson(geojson, default_height=12.0, height_key="_height_m")
        styled = city.set_layer_style(
            layer,
            city.COLORS[color_key],
            shadow_alpha=30,
            roof_outline=roof_outline,
        )
        if styled is not None:
            meshes.append(styled)

    return meshes, roof_outlines, city.select_focus_landmarks(localized, radius=radius_m)


def build_helsinki_scene(
    lon: float,
    lat: float,
    radius_m: float,
    *,
    refresh_osm: bool,
    refresh_buildings: bool,
    osm_timeout: float,
) -> tuple[city.SceneLayers, MetricContext]:
    ctx = build_metric_context(lon, lat, radius_m)
    surfaces = build_osm_surface_layers(
        ctx,
        lon=lon,
        lat=lat,
        radius_m=radius_m,
        refresh_osm=refresh_osm,
        osm_timeout=osm_timeout,
    )
    official_features = fetch_official_buildings(
        ctx,
        lon=lon,
        lat=lat,
        radius_m=radius_m,
        refresh=refresh_buildings,
    )
    meshes, roof_outlines, focus_landmarks = build_official_building_layers(
        official_features,
        ctx,
        radius_m=radius_m,
    )
    for mesh in meshes:
        mesh.shadow_alpha = max(int(mesh.shadow_alpha), 42)
        mesh.specular = 0.0
    scene = city.SceneLayers(
        surfaces=[surface for surface in surfaces if surface.rgba in MAP_SURFACE_COLORS],
        meshes=meshes,
        roof_outlines=roof_outlines,
        focus_landmarks=focus_landmarks,
        radius=radius_m,
    )
    return scene, ctx


def iter_gtfs_rows(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as handle:
        wrapper = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
        yield from csv.DictReader(wrapper)


def gtfs_service_ids_for_date(archive: zipfile.ZipFile, service_date: dt.date) -> set[str]:
    date_key = service_date.strftime("%Y%m%d")
    weekday_key = service_date.strftime("%A").lower()
    active: set[str] = set()

    names = set(archive.namelist())
    if "calendar.txt" in names:
        for row in iter_gtfs_rows(archive, "calendar.txt"):
            start_date = str(row.get("start_date", ""))
            end_date = str(row.get("end_date", ""))
            if start_date <= date_key <= end_date and str(row.get(weekday_key, "0")) == "1":
                active.add(str(row["service_id"]))

    if "calendar_dates.txt" in names:
        for row in iter_gtfs_rows(archive, "calendar_dates.txt"):
            if str(row.get("date", "")) != date_key:
                continue
            service_id = str(row["service_id"])
            exception_type = str(row.get("exception_type", ""))
            if exception_type == "1":
                active.add(service_id)
            elif exception_type == "2":
                active.discard(service_id)
    return active


def route_rgb(row: dict[str, str]) -> tuple[int, int, int]:
    raw = str(row.get("route_color") or "").strip().lstrip("#")
    if len(raw) == 6:
        try:
            return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
        except ValueError:
            pass
    try:
        route_type = int(row.get("route_type", -1))
    except ValueError:
        route_type = -1
    return TRANSIT_ROUTE_TYPES.get(route_type, TRANSIT_DEFAULT_RGB)


def localize_lonlat_points(
    coords_lonlat: list[tuple[float, float]],
    ctx: MetricContext,
) -> np.ndarray:
    if not coords_lonlat:
        return np.empty((0, 2), dtype=np.float32)
    lons = [float(lon) for lon, _ in coords_lonlat]
    lats = [float(lat) for _, lat in coords_lonlat]
    xs, ys = ctx.to_metric.transform(lons, lats)
    arr = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    arr[:, 0] -= ctx.center_xy[0]
    arr[:, 1] -= ctx.center_xy[1]
    if abs(float(city.SCENE_ROTATION_DEG)) > 1e-6:
        theta = math.radians(float(city.SCENE_ROTATION_DEG))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x = arr[:, 0].copy()
        y = arr[:, 1].copy()
        arr[:, 0] = x * cos_t - y * sin_t
        arr[:, 1] = x * sin_t + y * cos_t
    return arr.astype(np.float32)


def cumulative_distances(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    deltas = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(deltas * deltas, axis=1, dtype=np.float32))
    return np.concatenate([[0.0], np.cumsum(distances)]).astype(np.float32)


def trip_position_at_seconds(trip: TransitTrip, seconds: float) -> np.ndarray | None:
    if seconds < trip.start_seconds or seconds > trip.end_seconds:
        return None
    total_time = max(float(trip.end_seconds - trip.start_seconds), 1.0)
    total_distance = float(trip.cumulative_m[-1]) if trip.cumulative_m.size else 0.0
    if total_distance <= 1e-3 or trip.points_xy.shape[0] < 2:
        return None
    target_distance = ((float(seconds) - float(trip.start_seconds)) / total_time) * total_distance
    idx = int(np.searchsorted(trip.cumulative_m, target_distance, side="right"))
    idx = max(1, min(idx, trip.points_xy.shape[0] - 1))
    d0 = float(trip.cumulative_m[idx - 1])
    d1 = float(trip.cumulative_m[idx])
    t = 0.0 if d1 <= d0 else (target_distance - d0) / (d1 - d0)
    return trip.points_xy[idx - 1] * (1.0 - t) + trip.points_xy[idx] * t


def stable_unit_interval(value: str) -> float:
    return float(zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF) / float(0xFFFFFFFF)


def point_and_tangent_at_distance(
    points_xy: np.ndarray,
    cumulative_m: np.ndarray,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if points_xy.shape[0] < 2 or cumulative_m.size < 2:
        return None
    total_distance = float(cumulative_m[-1])
    if total_distance <= 1e-3:
        return None
    target = float(np.clip(distance_m, 0.0, total_distance))
    idx = int(np.searchsorted(cumulative_m, target, side="right"))
    idx = max(1, min(idx, points_xy.shape[0] - 1))
    d0 = float(cumulative_m[idx - 1])
    d1 = float(cumulative_m[idx])
    t = 0.0 if d1 <= d0 else (target - d0) / (d1 - d0)
    point = points_xy[idx - 1] * (1.0 - t) + points_xy[idx] * t
    tangent = points_xy[idx] - points_xy[idx - 1]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-6:
        return point, np.asarray([1.0, 0.0], dtype=np.float32)
    return point, (tangent / norm).astype(np.float32)


def traffic_hour_share(seconds: float) -> float:
    hour = (float(seconds) / 3600.0) % 24.0
    hour0 = int(math.floor(hour)) % 24
    hour1 = (hour0 + 1) % 24
    t = hour - math.floor(hour)
    return float(TRAFFIC_HOURLY_WEIGHTS[hour0] * (1.0 - t) + TRAFFIC_HOURLY_WEIGHTS[hour1] * t)


def active_car_flow_positions(
    segments: list[RoadTrafficSegment],
    *,
    seconds: float,
    radius_m: float,
    max_particles: int,
) -> list[tuple[np.ndarray, tuple[int, int, int], float]]:
    active: list[tuple[np.ndarray, tuple[int, int, int], float]] = []
    if not segments or max_particles <= 0:
        return active

    hour_share = traffic_hour_share(seconds)
    motion_seconds = float(seconds) * CAR_FLOW_ANIMATION_SPEED_SCALE
    for segment in segments:
        length_m = float(segment.cumulative_m[-1]) if segment.cumulative_m.size else 0.0
        if length_m <= 1.0:
            continue
        vehicles_per_second = max(segment.daily_vehicles * hour_share / 3600.0, 0.0)
        if vehicles_per_second <= 1e-5:
            continue
        per_direction_flow = vehicles_per_second * 0.5
        if per_direction_flow <= 1e-6:
            continue
        spacing_m = max(segment.speed_mps / per_direction_flow, CAR_FLOW_MIN_SPACING_M)
        for direction in (1, -1):
            phase = stable_unit_interval(f"{segment.segment_id}:{direction}") * spacing_m
            first_distance = (phase + motion_seconds * segment.speed_mps) % spacing_m
            distance = first_distance
            lane_sign = 1.0 if direction > 0 else -1.0
            while distance <= length_m:
                path_distance = distance if direction > 0 else length_m - distance
                sample = point_and_tangent_at_distance(segment.points_xy, segment.cumulative_m, path_distance)
                if sample is not None:
                    point, tangent = sample
                    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
                    point = point + normal * (lane_sign * CAR_FLOW_LANE_OFFSET_M)
                    if float(np.linalg.norm(point)) <= float(radius_m) * 1.08:
                        heavy_phase = stable_unit_interval(f"{segment.segment_id}:heavy:{direction}:{int(distance)}")
                        rgb = CAR_FLOW_HEAVY_RGB if heavy_phase < segment.heavy_share else CAR_FLOW_RGB
                        active.append((point.astype(np.float32), rgb, segment.daily_vehicles))
                        if len(active) >= max_particles:
                            return active
                distance += spacing_m
    return active


def path_intersects_radius(points_xy: np.ndarray, radius_m: float) -> bool:
    if points_xy.size == 0:
        return False
    distance = np.sqrt(np.sum(points_xy * points_xy, axis=1))
    return bool(np.any(distance <= float(radius_m) * 1.18))


def load_hsl_transit_trips(
    gtfs_zip_path: Path,
    ctx: MetricContext,
    *,
    service_date: dt.date,
    route_types: set[int],
    start_hour: float,
    end_hour: float,
    radius_m: float,
) -> list[TransitTrip]:
    window_start = int(math.floor(float(start_hour) * 3600.0)) - 1800
    window_end = int(math.ceil(float(end_hour) * 3600.0)) + 1800
    west, south, east, north = ctx.bbox_wgs84

    with zipfile.ZipFile(gtfs_zip_path) as archive:
        active_services = gtfs_service_ids_for_date(archive, service_date)
        if not active_services:
            raise RuntimeError(f"No HSL GTFS services are active on {service_date.isoformat()}")

        routes: dict[str, dict[str, object]] = {}
        for row in iter_gtfs_rows(archive, "routes.txt"):
            try:
                route_type = int(row.get("route_type", -1))
            except ValueError:
                continue
            if route_type not in route_types:
                continue
            route_id = str(row["route_id"])
            routes[route_id] = {
                "route_type": route_type,
                "route_name": str(row.get("route_short_name") or row.get("route_long_name") or route_id),
                "route_rgb": route_rgb(row),
            }

        stops_in_aoi: set[str] = set()
        stop_lonlat: dict[str, tuple[float, float]] = {}
        for row in iter_gtfs_rows(archive, "stops.txt"):
            try:
                lon = float(row["stop_lon"])
                lat = float(row["stop_lat"])
            except (KeyError, ValueError):
                continue
            stop_id = str(row["stop_id"])
            stop_lonlat[stop_id] = (lon, lat)
            if west <= lon <= east and south <= lat <= north:
                stops_in_aoi.add(stop_id)

        active_trips: dict[str, dict[str, str]] = {}
        for row in iter_gtfs_rows(archive, "trips.txt"):
            if str(row.get("service_id", "")) not in active_services:
                continue
            route_id = str(row.get("route_id", ""))
            if route_id not in routes:
                continue
            trip_id = str(row.get("trip_id", ""))
            if trip_id:
                active_trips[trip_id] = {
                    "route_id": route_id,
                    "shape_id": str(row.get("shape_id") or ""),
                }

        interested: set[str] = set()
        for row in iter_gtfs_rows(archive, "stop_times.txt"):
            trip_id = str(row.get("trip_id", ""))
            if trip_id in active_trips and str(row.get("stop_id", "")) in stops_in_aoi:
                interested.add(trip_id)

        trip_stop_times: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for row in iter_gtfs_rows(archive, "stop_times.txt"):
            trip_id = str(row.get("trip_id", ""))
            if trip_id not in interested:
                continue
            seconds = parse_gtfs_time_seconds(row.get("departure_time")) or parse_gtfs_time_seconds(row.get("arrival_time"))
            if seconds is None:
                continue
            try:
                sequence = int(row.get("stop_sequence", "0"))
            except ValueError:
                sequence = len(trip_stop_times[trip_id])
            trip_stop_times[trip_id].append((sequence, seconds, str(row.get("stop_id", ""))))

        shape_ids = {active_trips[trip_id]["shape_id"] for trip_id in interested if active_trips[trip_id]["shape_id"]}
        shape_rows: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        if shape_ids and "shapes.txt" in set(archive.namelist()):
            for row in iter_gtfs_rows(archive, "shapes.txt"):
                shape_id = str(row.get("shape_id", ""))
                if shape_id not in shape_ids:
                    continue
                try:
                    seq = int(row.get("shape_pt_sequence", "0"))
                    lat = float(row["shape_pt_lat"])
                    lon = float(row["shape_pt_lon"])
                except (KeyError, ValueError):
                    continue
                shape_rows[shape_id].append((seq, lon, lat))

    trips: list[TransitTrip] = []
    for trip_id, stop_rows in trip_stop_times.items():
        if len(stop_rows) < 2:
            continue
        stop_rows = sorted(stop_rows, key=lambda item: item[0])
        start_seconds = min(row[1] for row in stop_rows)
        end_seconds = max(row[1] for row in stop_rows)
        if end_seconds <= start_seconds or end_seconds < window_start or start_seconds > window_end:
            continue

        trip_meta = active_trips[trip_id]
        route_id = trip_meta["route_id"]
        route = routes[route_id]
        shape_id = trip_meta["shape_id"]
        if shape_id and shape_id in shape_rows:
            lonlat = [(lon, lat) for _, lon, lat in sorted(shape_rows[shape_id], key=lambda item: item[0])]
        else:
            lonlat = [stop_lonlat[stop_id] for _, _, stop_id in stop_rows if stop_id in stop_lonlat]
        points_xy = localize_lonlat_points(lonlat, ctx)
        if points_xy.shape[0] < 2 or not path_intersects_radius(points_xy, radius_m):
            continue
        cumulative = cumulative_distances(points_xy)
        if cumulative.size < 2 or float(cumulative[-1]) <= 1.0:
            continue
        trips.append(
            TransitTrip(
                trip_id=trip_id,
                route_id=route_id,
                route_type=int(route["route_type"]),
                route_name=str(route["route_name"]),
                route_rgb=route["route_rgb"],  # type: ignore[arg-type]
                start_seconds=int(start_seconds),
                end_seconds=int(end_seconds),
                points_xy=points_xy,
                cumulative_m=cumulative,
            )
        )

    print(f"[Helsinki] HSL active services={len(active_services)} transit trips in AOI={len(trips)}")
    return trips


def active_transit_positions(
    trips: list[TransitTrip],
    *,
    seconds: float,
    radius_m: float,
) -> list[tuple[np.ndarray, tuple[int, int, int], int]]:
    active: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    for trip in trips:
        point = trip_position_at_seconds(trip, seconds)
        if point is None:
            continue
        if float(np.linalg.norm(point)) > float(radius_m) * 1.08:
            continue
        active.append((point, trip.route_rgb, trip.route_type))
    return active


def grayscale_rgba(image: Image.Image, *, brightness: float = DARK_MAP_BRIGHTNESS) -> Image.Image:
    arr = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
    rgb = arr[:, :, :3].astype(np.float32)
    luma = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    )
    luma = np.clip(luma * float(brightness) + DARK_MAP_BLACK_LEVEL, 0.0, 255.0).astype(np.uint8)
    arr[:, :, 0] = luma
    arr[:, :, 1] = luma
    arr[:, :, 2] = luma
    return Image.fromarray(arr, mode="RGBA")


def project_transit_points(
    prepared: day.PreparedScene,
    points_xy: np.ndarray,
    *,
    elevation_m: float = 9.0,
) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    world = np.column_stack(
        [
            points_xy[:, 0],
            np.full(points_xy.shape[0], float(elevation_m), dtype=np.float32),
            -points_xy[:, 1],
        ]
    ).astype(np.float32)
    projected = day.project_points_quiet(
        world,
        eye=prepared.eye,
        target=prepared.target,
        up=prepared.up,
        width=prepared.render_width,
        height=prepared.render_height,
        fov_deg=prepared.fov_deg,
    )
    projected[:, 0] = projected[:, 0] * prepared.scale + prepared.offset_x
    projected[:, 1] = projected[:, 1] * prepared.scale + prepared.offset_y
    if prepared.supersample > 1:
        projected[:, 0:2] /= float(prepared.supersample)
    return projected


def add_car_flow_overlay(
    image: Image.Image,
    prepared: day.PreparedScene,
    active_vehicles: list[tuple[np.ndarray, tuple[int, int, int], float]],
) -> Image.Image:
    canvas = image.convert("RGBA")
    if not active_vehicles:
        return canvas

    points_xy = np.vstack([item[0] for item in active_vehicles]).astype(np.float32)
    projected = project_transit_points(prepared, points_xy, elevation_m=2.8)
    visible = projected[:, 2] > 1.0
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    core = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core, "RGBA")
    scale = max(canvas.width / 1280.0, 0.72)
    glow_radius = max(3.8, 5.3 * scale)
    core_radius = max(1.35, 1.8 * scale)
    for (_, rgb_type, _), screen, ok in zip(active_vehicles, projected, visible):
        if not ok:
            continue
        x = float(screen[0])
        y = float(screen[1])
        if x < -16 or y < -16 or x > canvas.width + 16 or y > canvas.height + 16:
            continue
        rgb = tuple(int(v) for v in rgb_type)
        glow_draw.ellipse(
            [(x - glow_radius, y - glow_radius), (x + glow_radius, y + glow_radius)],
            fill=rgb + (58,),
        )
        core_draw.ellipse(
            [(x - core_radius, y - core_radius), (x + core_radius, y + core_radius)],
            fill=rgb + (214,),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(2.0, 3.0 * scale)))
    canvas = Image.alpha_composite(canvas, glow)
    return Image.alpha_composite(canvas, core)


def add_transit_overlay(
    image: Image.Image,
    prepared: day.PreparedScene,
    active: list[tuple[np.ndarray, tuple[int, int, int], int]],
) -> Image.Image:
    canvas = image.convert("RGBA")
    if active:
        points_xy = np.vstack([item[0] for item in active]).astype(np.float32)
        projected = project_transit_points(prepared, points_xy)
        visible = projected[:, 2] > 1.0
        glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        core = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        core_draw = ImageDraw.Draw(core, "RGBA")
        scale = max(canvas.width / 1280.0, 0.72)
        glow_radius = max(7.0, 10.0 * scale)
        core_radius = max(2.4, 3.2 * scale)
        for (point, rgb_type, route_type), screen, ok in zip(active, projected, visible):
            _ = point
            _ = route_type
            if not ok:
                continue
            x = float(screen[0])
            y = float(screen[1])
            if x < -24 or y < -24 or x > canvas.width + 24 or y > canvas.height + 24:
                continue
            rgb = tuple(int(v) for v in rgb_type)
            glow_draw.ellipse(
                [(x - glow_radius, y - glow_radius), (x + glow_radius, y + glow_radius)],
                fill=rgb + (95,),
            )
            core_draw.ellipse(
                [(x - core_radius, y - core_radius), (x + core_radius, y + core_radius)],
                fill=(255, 255, 255, 235),
            )
            inner = core_radius * 1.8
            core_draw.ellipse(
                [(x - inner, y - inner), (x + inner, y + inner)],
                outline=rgb + (190,),
                width=max(1, int(round(1.3 * scale))),
            )
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(3.0, 4.6 * scale)))
        canvas = Image.alpha_composite(canvas, glow)
        canvas = Image.alpha_composite(canvas, core)
    return canvas


def text_bbox_for(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font,
    *,
    stroke_width: int = 0,
) -> tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox(xy, text, font=font, stroke_width=stroke_width)
    x, y = xy
    width = int(round(draw.textlength(text, font=font)))
    height = int(getattr(font, "size", 14))
    return (int(x), int(y), int(x) + width, int(y) + height)


def draw_colored_text_segments(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    segments: list[tuple[str, tuple[int, int, int]]],
    font,
    *,
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int],
    alpha: int,
) -> tuple[int, int, int, int]:
    x, y = xy
    cursor = float(x)
    bounds: list[tuple[int, int, int, int]] = []
    for text, rgb in segments:
        if not text:
            continue
        fill = tuple(int(v) for v in rgb) + (int(alpha),)
        draw.text((cursor, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        bbox = text_bbox_for(draw, (cursor, y), text, font, stroke_width=stroke_width)
        bounds.append(bbox)
        cursor += float(draw.textlength(text, font=font))
    if not bounds:
        return (int(x), int(y), int(x), int(y))
    return (
        min(bbox[0] for bbox in bounds),
        min(bbox[1] for bbox in bounds),
        max(bbox[2] for bbox in bounds),
        max(bbox[3] for bbox in bounds),
    )


def color_legend_segments() -> list[tuple[str, tuple[int, int, int]]]:
    return [
        ("Road: ", TITLE_SUBTEXT_RGB),
        ("cars", CAR_FLOW_RGB),
        (" + ", TITLE_SUBTEXT_RGB),
        ("heavy", CAR_FLOW_HEAVY_RGB),
        (" | HSL: ", TITLE_SUBTEXT_RGB),
        ("tram", TRANSIT_ROUTE_TYPES[0]),
        (" ", TITLE_SUBTEXT_RGB),
        ("metro", TRANSIT_ROUTE_TYPES[1]),
        (" ", TITLE_SUBTEXT_RGB),
        ("rail", TRANSIT_ROUTE_TYPES[2]),
        (" ", TITLE_SUBTEXT_RGB),
        ("bus", TRANSIT_ROUTE_TYPES[3]),
        (" ", TITLE_SUBTEXT_RGB),
        ("ferry", TRANSIT_ROUTE_TYPES[4]),
    ]


def helsinki_header_height(width: int, height: int) -> int:
    scale = float(width) / 1280.0
    target = max(112.0 * scale, float(height) * HELSINKI_HEADER_HEIGHT_RATIO)
    max_header = max(1, int(height) - 96)
    return int(np.clip(int(round(target)), 84, max_header))


def add_helsinki_text_overlay(
    image: Image.Image,
    *,
    service_date: dt.date,
    frame_index: int,
    total_frames: int,
    start_hour: float,
    end_hour: float,
    active_hsl_count: int,
    modeled_vehicle_count: int,
    output_height: int | None = None,
    header_height: int | None = None,
) -> Image.Image:
    map_image = image.convert("RGBA")
    if output_height is None:
        output_height = map_image.height
    if header_height is None:
        header_height = helsinki_header_height(map_image.width, int(output_height))
    header_height = int(np.clip(int(header_height), 1, int(output_height)))

    if int(output_height) > map_image.height:
        canvas = Image.new("RGBA", (map_image.width, int(output_height)), TITLE_HEADER_RGB + (255,))
        canvas.alpha_composite(map_image, dest=(0, header_height))
    else:
        canvas = map_image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = canvas.width / 1280.0
    margin = max(18, int(round(28 * scale)))
    stroke = max(2, int(round(3 * scale)))

    title = "Helsinki Vehicles"
    counts = f"HSL vehicles {active_hsl_count:,} | modeled road vehicles {modeled_vehicle_count:,} | {service_date.isoformat()}"
    legend_segments = color_legend_segments()
    legend_text = "".join(text for text, _ in legend_segments)
    left_max_width = max(520, int(round(canvas.width * 0.66)))

    title_font = day.fit_timer_font(
        title,
        target_size_px=max(30, int(round(38 * scale))),
        max_width_px=max(360, int(round(canvas.width * 0.44))),
        min_size_px=max(22, int(round(28 * scale))),
    )
    legend_font = day.fit_timer_font(
        legend_text,
        target_size_px=max(12, int(round(15 * scale))),
        max_width_px=left_max_width,
        min_size_px=max(10, int(round(12 * scale))),
    )
    counts_font = day.fit_timer_font(
        counts,
        target_size_px=max(12, int(round(15 * scale))),
        max_width_px=left_max_width,
        min_size_px=max(10, int(round(12 * scale))),
    )

    x = margin
    y = margin
    gap = max(4, int(round(6 * scale)))
    title_bbox = text_bbox_for(draw, (x, y), title, title_font, stroke_width=stroke)
    legend_y = title_bbox[3] + gap
    legend_bbox = text_bbox_for(
        draw,
        (x, legend_y),
        legend_text,
        legend_font,
        stroke_width=max(1, stroke - 1),
    )
    counts_y = legend_bbox[3] + gap
    counts_bbox = text_bbox_for(
        draw,
        (x, counts_y),
        counts,
        counts_font,
        stroke_width=max(1, stroke - 1),
    )

    clock_text = day.format_clock_text(frame_index, total_frames, start_hour, end_hour)
    clock_font = day.fit_timer_font(
        clock_text,
        target_size_px=max(52, int(round(72 * scale))),
        max_width_px=max(240, int(round(canvas.width * 0.28))),
        min_size_px=max(42, int(round(54 * scale))),
    )
    clock_bbox = text_bbox_for(draw, (0, 0), clock_text, clock_font, stroke_width=stroke)
    clock_w = clock_bbox[2] - clock_bbox[0]
    clock_x = canvas.width - margin - clock_w - clock_bbox[0]
    clock_y = margin - clock_bbox[1]
    clock_draw_bbox = (
        int(clock_x + clock_bbox[0]),
        int(clock_y + clock_bbox[1]),
        int(clock_x + clock_bbox[2]),
        int(clock_y + clock_bbox[3]),
    )

    draw.rectangle([(0, 0), (canvas.width, header_height)], fill=TITLE_HEADER_RGB + (255,))

    draw.text(
        (x, y),
        title,
        font=title_font,
        fill=TITLE_TEXT_RGB + (255,),
        stroke_width=stroke,
        stroke_fill=TITLE_BOUNDARY_RGB + (230,),
    )
    draw_colored_text_segments(
        draw,
        (x, legend_y),
        legend_segments,
        legend_font,
        stroke_width=max(1, stroke - 1),
        stroke_fill=TITLE_BOUNDARY_RGB + (220,),
        alpha=250,
    )
    draw.text(
        (x, counts_y),
        counts,
        font=counts_font,
        fill=TITLE_TEXT_RGB + (248,),
        stroke_width=max(1, stroke - 1),
        stroke_fill=TITLE_BOUNDARY_RGB + (220,),
    )

    draw.text(
        (clock_x, clock_y),
        clock_text,
        font=clock_font,
        fill=TITLE_TEXT_RGB + (255,),
        stroke_width=stroke,
        stroke_fill=TITLE_BOUNDARY_RGB + (230,),
    )
    return canvas


def frame_seconds_for_index(frame_index: int, total_frames: int, start_hour: float, end_hour: float) -> float:
    t = frame_index / max(total_frames - 1, 1)
    return (float(start_hour) + (float(end_hour) - float(start_hour)) * t) * 3600.0


def resolve_gtfs_zip(args: argparse.Namespace) -> Path:
    if args.gtfs_zip is not None:
        path = args.gtfs_zip.resolve()
        if not path.exists():
            raise FileNotFoundError(f"HSL GTFS zip not found: {path}")
        return path
    return download_file(
        HSL_GTFS_URL,
        CACHE_DIR / "hsl" / "hsl.zip",
        refresh=bool(args.refresh_gtfs),
        label="HSL GTFS",
    )


def main() -> int:
    args = parse_args()
    width, height = map(int, args.size)
    frame_count = max(1, int(args.frames))
    fps = max(1, int(args.fps))
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    service_date = parse_service_date(args.service_date)
    route_types = parse_route_types(args.route_types)

    print(
        f"[Helsinki] center=({args.lon:.5f}, {args.lat:.5f}) radius={args.radius:.0f}m "
        f"date={service_date.isoformat()}",
        flush=True,
    )
    scene, ctx = build_helsinki_scene(
        args.lon,
        args.lat,
        args.radius,
        refresh_osm=bool(args.refresh_osm),
        refresh_buildings=bool(args.refresh_buildings),
        osm_timeout=float(args.osm_timeout),
    )
    if not scene.meshes:
        raise SystemExit("No official Helsinki buildings were generated for the requested AOI.")
    header_height = helsinki_header_height(width, height)
    map_height = max(96, height - header_height)
    prepared = day.prepare_scene(scene, width=width, height=map_height, supersample=int(args.supersample))
    map_brightness = float(np.clip(float(args.map_brightness), 0.12, 1.0))
    max_vehicle_particles = max(
        0,
        int(args.max_vehicle_particles if args.max_vehicle_particles is not None else args.max_car_particles),
    )

    traffic_segments = fetch_official_traffic_segments(
        ctx,
        lon=args.lon,
        lat=args.lat,
        radius_m=args.radius,
        refresh=bool(args.refresh_traffic),
    )
    if not traffic_segments:
        print("[Helsinki] warning: no official traffic-count lines intersected the requested AOI")

    gtfs_zip = resolve_gtfs_zip(args)
    trips = load_hsl_transit_trips(
        gtfs_zip,
        ctx,
        service_date=service_date,
        route_types=route_types,
        start_hour=float(args.clock_start_hour),
        end_hour=float(args.clock_end_hour),
        radius_m=float(args.radius),
    )
    if not trips:
        raise SystemExit("No HSL trips intersected the requested AOI and service window.")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is not None:
        frames_dir = args.frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_frames:
        frames_dir = output_path.with_suffix("")
        frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="helsinki_transit_", dir=str(output_path.parent))
        frames_dir = Path(temp_dir.name)

    try:
        for frame_index in range(frame_count):
            sun = day.sun_state_for_frame(frame_index, frame_count)
            frame_seconds = frame_seconds_for_index(
                frame_index,
                frame_count,
                float(args.clock_start_hour),
                float(args.clock_end_hour),
            )
            frame = day.render_frame(
                prepared,
                sun,
                frame_index=frame_index,
                total_frames=frame_count,
                clock_start_hour=float(args.clock_start_hour),
                clock_end_hour=float(args.clock_end_hour),
                show_timer=False,
            )
            frame = grayscale_rgba(frame, brightness=map_brightness)
            active_road_vehicles = active_car_flow_positions(
                traffic_segments,
                seconds=frame_seconds,
                radius_m=float(args.radius),
                max_particles=max_vehicle_particles,
            )
            frame = add_car_flow_overlay(frame, prepared, active_road_vehicles)
            active = active_transit_positions(trips, seconds=frame_seconds, radius_m=float(args.radius))
            frame = add_transit_overlay(frame, prepared, active)
            frame = add_helsinki_text_overlay(
                frame,
                service_date=service_date,
                frame_index=frame_index,
                total_frames=frame_count,
                start_hour=float(args.clock_start_hour),
                end_hour=float(args.clock_end_hour),
                active_hsl_count=len(active),
                modeled_vehicle_count=len(active_road_vehicles),
                output_height=height,
                header_height=header_height,
            )
            frame.save(frames_dir / f"frame_{frame_index:05d}.png", format="PNG", compress_level=0)
            if frame_index == 0 or (frame_index + 1) % 24 == 0 or frame_index + 1 == frame_count:
                print(
                    f"[Helsinki] frame {frame_index + 1}/{frame_count} "
                    f"| hsl={len(active):,} vehicles={len(active_road_vehicles):,} "
                    f"| sun az={sun.azimuth_deg:.1f} el={sun.elevation_deg:.1f}"
                )
        day.encode_video(frames_dir, output_path, fps=fps)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"[Helsinki] Wrote {city.display_path(output_path)}")
    if args.keep_frames or args.frames_dir is not None:
        print(f"[Helsinki] Frames saved in {city.display_path(frames_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
