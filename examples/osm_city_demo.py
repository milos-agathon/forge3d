#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageChops, ImageCms, ImageDraw, ImageFilter, ImageFont
from pyproj import Transformer
from shapely import BufferCapStyle, BufferJoinStyle, affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, mapping
from shapely.geometry.polygon import orient as orient_polygon
from shapely.ops import polygonize, transform, unary_union

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "osm_city_demo" / "copenhagen.png"
CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "osm_city_demo"
OVERPASS_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)

GREEN_LANDUSE = {"grass", "recreation_ground", "forest", "greenery"}
COLORS = {
    "building_low": (0xE2, 0xC8, 0x5E, 255),
    "building_mid": (0xD1, 0xAF, 0x48, 255),
    "building_high": (0xBF, 0x92, 0x38, 255),
    "building_landmark": (0x84, 0x5F, 0x12, 255),
    "landuse": (0xD4, 0xC5, 0x72, 255),
    "park": (0x5F, 0xBE, 0x60, 255),
    "road": (0xC5, 0xD1, 0xDE, 255),
    "road_hi": (0xF7, 0xFA, 0xFD, 255),
    "water": (0x52, 0xC2, 0xD4, 230),
    "base": (0xE6, 0xEC, 0xF1, 255),
}

AO_KERNEL_SAMPLES = 32
AO_RADIUS_WORLD_UNITS = 3.6
AO_INTENSITY = 0.40
AO_BIAS = 0.025
AO_NORMAL_INFLUENCE = 0.80
AO_BLUR_PASSES = 2
AO_BILATERAL_SIGMA_SPATIAL = 2.0
AO_BILATERAL_SIGMA_RANGE = 0.10
AO_COLOR_RGB = (0x34, 0x40, 0x4D)

SHADOW_MAP_RESOLUTION = 4096
PCSS_BLOCKER_SEARCH_TEXELS = 10.0
PCSS_PENUMBRA_SCALE = 1.5
SHADOW_LIGHT_SIZE_WORLD = 3.0
SHADOW_DEPTH_BIAS = 0.0005
SHADOW_NORMAL_BIAS = 0.02
SHADOW_COLOR_RGB = (0x3A, 0x4A, 0x5C)
SHADOW_MULTIPLY_OPACITY = 0.55

EDGE_FADE_START = 0.85
EDGE_FADE_END = 1.0
EDGE_SATURATION_MIN = 0.40
EDGE_LIGHTNESS_GAIN = 0.15
EDGE_FOG_RGB = (0xE8, 0xED, 0xF1)

ATMOS_NEAR_END = 0.50
ATMOS_MID_END = 0.80
ATMOS_SAT_NEAR = 1.00
ATMOS_SAT_MID = 0.92
ATMOS_SAT_FAR = 0.78
ATMOS_VALUE_NEAR = 0.00
ATMOS_VALUE_MID = 0.04
ATMOS_VALUE_FAR = 0.10

ROOF_OUTLINE_DARKEN = 0.85
ROOF_OUTLINE_ALPHA = 224
ROOF_OUTLINE_WIDTH_PX = 1.0

WATER_SPECULAR_INTENSITY = 0.20
WATER_ROUGHNESS = 0.68
WATER_F0 = 0.02
WATER_FRESNEL_POWER = 4.0

MICRO_CONTRAST_AMOUNT = 0.12
MICRO_CONTRAST_RADIUS_PX = 60.0
MICRO_CONTRAST_THRESHOLD = 2.0
SELECTIVE_SHARPEN_RADIUS_PX = 1.2
SELECTIVE_SHARPEN_OPACITY = 0.18
VIGNETTE_DARKEN = 0.12
VIGNETTE_FEATHER = 0.65
BLUE_NOISE_TILE_SIZE = 64
SCENE_ROTATION_DEG = -180.0
REFLECTION_PLANE_OPACITY = 0.10
REFLECTION_PLANE_BLUR_PX = 15.0
REFLECTION_PLANE_FADE_PX = 80.0
REFLECTION_PLANE_OFFSET_PX = -10.0
LANDMARK_GLOW_RADIUS_PX = 20.0
LANDMARK_GLOW_INTENSITY = 0.05
LANDMARK_GLOW_RGB = (0xFF, 0xE7, 0xBE)
FOCAL_HERO_SATURATION_BOOST = 0.08
FOCAL_SECONDARY_DESATURATION = 0.06
FOCAL_HERO_COUNT = 4
POSTER_TITLE = "Copenhagen"
POSTER_TITLE_RGB = (0x2D, 0x3A, 0x47)
POSTER_SUBTITLE_RGB = (0x5A, 0x68, 0x72)
POSTER_CAPTION_RGB = (0x7A, 0x86, 0x90)
POSTER_CAPTION = "©2026 Milos Popovic (milosgis.nl) | source: ©OpenStreetMap contributors"

BACKGROUND_STOPS = (
    (0.00, (0xB8, 0xC4, 0xCE)),
    (0.40, (0xD4, 0xDC, 0xE2)),
    (0.70, (0xE8, 0xED, 0xF1)),
    (1.00, (0xF5, 0xF7, 0xF9)),
)


@dataclass
class MeshLayer:
    positions: np.ndarray
    indices: np.ndarray
    rgba: tuple[int, int, int, int]
    shadow_alpha: int = 22
    specular: float = 0.0
    roof_outline: bool = False


@dataclass
class SurfaceLayer:
    geometry: Polygon | MultiPolygon
    rgba: tuple[int, int, int, int]
    elevation: float
    specular: float = 0.0
    reflectivity: float = 0.0


@dataclass
class RoofOutlineLayer:
    geometry: Polygon | MultiPolygon
    elevation: float


@dataclass
class FocusLandmark:
    geometry: Polygon | MultiPolygon
    elevation: float


@dataclass
class SceneLayers:
    surfaces: list[SurfaceLayer]
    meshes: list[MeshLayer]
    roof_outlines: list[RoofOutlineLayer]
    focus_landmarks: list[FocusLandmark]
    radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Copenhagen-style OSM city preview. The scene geometry is built "
            "with forge3d extrusion helpers, then previewed with a deterministic mesh renderer."
        )
    )
    parser.add_argument("--lon", type=float, default=12.56553, help="AOI center longitude")
    parser.add_argument("--lat", type=float, default=55.67594, help="AOI center latitude")
    parser.add_argument("--radius", type=float, default=1000.0, help="AOI radius in meters")
    parser.add_argument("--size", type=int, nargs=2, default=(3840, 2160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--supersample", type=int, default=2, help="Render scale factor before downsampling")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-osm", action="store_true", help="Ignore cached Overpass responses")
    return parser.parse_args()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    length = float(np.linalg.norm(vec))
    if length <= 1e-8:
        return vec
    return vec / length


def normalize_field(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(length, 1e-6)


def smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> float | np.ndarray:
    span = max(float(edge1) - float(edge0), 1e-6)
    x = np.clip((np.asarray(value, dtype=np.float32) - float(edge0)) / span, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend_saturation(rgb: np.ndarray, saturation: float | np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    sat = np.asarray(saturation, dtype=np.float32)
    luminance = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )[..., None]
    return luminance + (rgb - luminance) * sat[..., None]


def utm_epsg_for_lon_lat(lon: float, lat: float) -> int:
    if lat >= 84.0:
        return 32661
    if lat <= -80.0:
        return 32761
    zone = int(math.floor((lon + 180.0) / 6.0) + 1.0)
    return (32600 if lat >= 0.0 else 32700) + zone


def parse_osm_numeric(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    if "ft" in text or "feet" in text:
        number *= 0.3048
    return number


def infer_building_height(tags: dict[str, object]) -> float:
    height = parse_osm_numeric(tags.get("height"))
    if height is not None and math.isfinite(height):
        return min(max(height, 1.0), 80.0)
    levels = parse_osm_numeric(tags.get("building:levels"))
    if levels is not None and math.isfinite(levels):
        return min(max(levels, 1.0) * 3.2, 80.0)
    return 12.0


def building_bin(height: float) -> str:
    if height <= 15.0:
        return "low"
    if height <= 30.0:
        return "mid"
    if height <= 45.0:
        return "high"
    return "landmark"


def cache_slug(lon: float, lat: float, radius: float) -> str:
    return (
        f"lon{lon:.5f}_lat{lat:.5f}_r{int(round(radius))}"
        .replace("-", "m")
        .replace(".", "p")
    )


def overpass_query(
    bbox: tuple[float, float, float, float],
    filter_expr: str,
    *,
    include_relations: bool,
) -> str:
    west, south, east, north = bbox
    parts = [f'way{filter_expr}({south},{west},{north},{east});']
    if include_relations:
        parts.append(f'relation{filter_expr}({south},{west},{north},{east});')
    return f"[out:json][timeout:60];({''.join(parts)});out geom qt;"


def fetch_overpass_query(query: str, cache_path: Path, refresh: bool) -> list[dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])

    payload = None
    last_error: Exception | None = None
    for endpoint in OVERPASS_URLS:
        for attempt in range(2):
            request = Request(
                endpoint,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": "forge3d-osm-city-demo/1.0",
                },
            )
            try:
                with urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                sleep(1.5 * (attempt + 1))
        if payload is not None:
            break

    if payload is None:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])
        raise RuntimeError(f"Overpass request failed for {cache_path.name}: {last_error}") from last_error

    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload.get("elements", [])


def _fix_geom(geom):
    if geom.is_empty:
        return geom
    try:
        from shapely import make_valid

        geom = make_valid(geom)
    except Exception:
        geom = geom.buffer(0)
    return geom


def _extract_polygonal(geom):
    if geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(polys) if polys else None
    return None


def _extract_linear(geom):
    if geom.is_empty:
        return None
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom
    if isinstance(geom, GeometryCollection):
        lines = [part for part in geom.geoms if part.geom_type in ("LineString", "MultiLineString")]
        return unary_union(lines) if lines else None
    return None


def _coords_from_element_geometry(element: dict) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for point in element.get("geometry", []):
        if "lon" in point and "lat" in point:
            coords.append((float(point["lon"]), float(point["lat"])))
    return coords


def polygon_geometry_from_element(element: dict):
    if element.get("type") == "way":
        coords = _coords_from_element_geometry(element)
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return _fix_geom(Polygon(coords))

    if element.get("type") != "relation":
        return None

    outer_lines: list[LineString] = []
    inner_lines: list[LineString] = []
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        coords = []
        for point in member.get("geometry", []):
            if "lon" in point and "lat" in point:
                coords.append((float(point["lon"]), float(point["lat"])))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.is_empty or line.length <= 0.0:
            continue
        if member.get("role") == "inner":
            inner_lines.append(line)
        else:
            outer_lines.append(line)

    if not outer_lines:
        return None

    outer_polys = list(polygonize(outer_lines))
    if not outer_polys:
        outer_polys = list(polygonize(unary_union(outer_lines)))
    if not outer_polys:
        return None

    geom = unary_union(outer_polys)
    if inner_lines:
        inner_polys = list(polygonize(inner_lines))
        if inner_polys:
            geom = geom.difference(unary_union(inner_polys))
    return _fix_geom(geom)


def line_geometry_from_element(element: dict):
    if element.get("type") == "way":
        coords = _coords_from_element_geometry(element)
        if len(coords) < 2:
            return None
        return LineString(coords)

    if element.get("type") != "relation":
        return None

    lines = []
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        coords = []
        for point in member.get("geometry", []):
            if "lon" in point and "lat" in point:
                coords.append((float(point["lon"]), float(point["lat"])))
        if len(coords) >= 2:
            lines.append(LineString(coords))
    if not lines:
        return None
    return unary_union(lines)


def parse_polygon_features(
    elements: list[dict],
    to_metric: Transformer,
    clip_m: Polygon,
) -> list[dict]:
    features = []
    for element in elements:
        geom = polygon_geometry_from_element(element)
        if geom is None or geom.is_empty:
            continue
        geom_m = transform(to_metric.transform, geom)
        clipped = _extract_polygonal(_fix_geom(geom_m.intersection(clip_m)))
        if clipped is None or clipped.is_empty:
            continue
        features.append({"geometry": clipped, "tags": dict(element.get("tags", {}))})
    return features


def parse_line_features(
    elements: list[dict],
    to_metric: Transformer,
    clip_m: Polygon,
) -> list[dict]:
    features = []
    for element in elements:
        geom = line_geometry_from_element(element)
        if geom is None or geom.is_empty:
            continue
        geom_m = transform(to_metric.transform, geom)
        clipped = _extract_linear(_fix_geom(geom_m.intersection(clip_m)))
        if clipped is None or clipped.is_empty:
            continue
        features.append({"geometry": clipped, "tags": dict(element.get("tags", {}))})
    return features


def simplify_geom(geom, tolerance: float):
    if geom is None or geom.is_empty:
        return geom
    simplified = geom.simplify(tolerance, preserve_topology=True)
    simplified = _fix_geom(simplified)
    return simplified if not simplified.is_empty else geom


def _polygon_parts(geom) -> list[Polygon]:
    polygonal = _extract_polygonal(_fix_geom(geom))
    if polygonal is None or polygonal.is_empty:
        return []
    if isinstance(polygonal, Polygon):
        return [polygonal]
    if isinstance(polygonal, MultiPolygon):
        return [part for part in polygonal.geoms if not part.is_empty]
    if isinstance(polygonal, GeometryCollection):
        parts: list[Polygon] = []
        for part in polygonal.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def sanitize_polygonal_for_extrusion(
    geom,
    *,
    min_area: float = 0.05,
) -> Polygon | MultiPolygon | None:
    parts: list[Polygon] = []
    for part in _polygon_parts(geom):
        coords = list(part.exterior.coords)
        if len(coords) < 4:
            continue
        unique_xy = {(round(float(x), 6), round(float(y), 6)) for x, y, *_ in coords[:-1]}
        if len(unique_xy) < 3:
            continue
        area = abs(float(part.area))
        if not math.isfinite(area) or area <= min_area:
            continue
        parts.append(orient_polygon(part, sign=1.0))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def prepare_polygonal_geom(
    geom,
    *,
    simplify_tolerance: float = 0.0,
    min_area: float = 0.05,
    min_area_ratio: float = 0.2,
) -> Polygon | MultiPolygon | None:
    base = sanitize_polygonal_for_extrusion(geom, min_area=min_area)
    if base is None or simplify_tolerance <= 0.0:
        return base
    simplified = sanitize_polygonal_for_extrusion(
        simplify_geom(base, simplify_tolerance),
        min_area=min_area,
    )
    if simplified is None:
        return base
    base_area = abs(float(base.area))
    simplified_area = abs(float(simplified.area))
    if base_area > min_area and simplified_area / max(base_area, 1e-9) < min_area_ratio:
        return base
    return simplified


def localize_feature(feature: dict, center_xy: tuple[float, float]) -> dict:
    geometry = affinity.translate(feature["geometry"], xoff=-center_xy[0], yoff=-center_xy[1])
    if abs(float(SCENE_ROTATION_DEG)) > 1e-6:
        geometry = affinity.rotate(geometry, SCENE_ROTATION_DEG, origin=(0.0, 0.0))
    return {
        "geometry": geometry,
        "tags": dict(feature["tags"]),
    }


def select_focus_landmarks(
    features: list[dict],
    *,
    radius: float,
    max_count: int = FOCAL_HERO_COUNT,
) -> list[FocusLandmark]:
    candidates: list[tuple[float, FocusLandmark]] = []
    fallback: list[tuple[float, FocusLandmark]] = []
    radius = max(float(radius), 1.0)
    for feature in features:
        height = float(feature["tags"].get("_height_m", infer_building_height(feature["tags"])))
        geom = prepare_polygonal_geom(feature["geometry"], simplify_tolerance=0.18)
        if geom is None or geom.is_empty:
            continue
        area = abs(float(geom.area))
        centroid = geom.centroid
        distance = math.hypot(float(centroid.x), float(centroid.y))
        centrality = 1.14 - 0.34 * min(distance / radius, 1.0)
        size_bonus = 1.0 + 0.18 * min(area / 3200.0, 1.0)
        score = height * centrality * size_bonus
        item = FocusLandmark(geometry=geom, elevation=height)
        fallback.append((score, item))
        if height >= 34.0:
            candidates.append((score, item))
    pool = candidates if len(candidates) >= max_count else fallback
    return [item for _, item in sorted(pool, key=lambda entry: entry[0], reverse=True)[:max_count]]


def merge_surface_geometry(
    features: list[dict],
    *,
    simplify_tolerance: float = 0.0,
):
    if not features:
        return None
    geom = _extract_polygonal(_fix_geom(unary_union([feature["geometry"] for feature in features])))
    if geom is None or geom.is_empty:
        return None
    if simplify_tolerance > 0.0:
        simplified = _extract_polygonal(_fix_geom(simplify_geom(geom, simplify_tolerance)))
        if simplified is not None and not simplified.is_empty:
            geom = simplified
    return geom


def feature_collection(
    features: list[dict],
    *,
    height_key: str | None = None,
    default_height: float | None = None,
    simplify_tolerance: float = 0.0,
) -> str:
    out = {"type": "FeatureCollection", "features": []}
    for feature in features:
        geom = prepare_polygonal_geom(feature["geometry"], simplify_tolerance=simplify_tolerance)
        if geom is None or geom.is_empty:
            continue
        props = dict(feature.get("tags", {}))
        if default_height is not None and height_key is not None and height_key not in props:
            props[height_key] = float(default_height)
        out["features"].append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(geom),
            }
        )
    return json.dumps(out)


def mesh_from_geojson(
    geojson: str,
    *,
    default_height: float,
    height_key: str | None = None,
) -> MeshLayer | None:
    payload = json.loads(geojson)
    if not payload.get("features"):
        return None
    mesh = f3d.io.import_osm_buildings_from_geojson(
        geojson,
        default_height=default_height,
        height_key=height_key,
    )
    positions = np.asarray(mesh.positions, dtype=np.float32).copy()
    indices = np.asarray(mesh.indices, dtype=np.uint32).reshape(-1, 3)
    if positions.size == 0 or indices.size == 0:
        return None
    positions[:, 2] *= -1.0
    return MeshLayer(positions=positions, indices=indices, rgba=(255, 255, 255, 255))


def set_layer_style(
    layer: MeshLayer | None,
    rgba: tuple[int, int, int, int],
    *,
    y_offset: float = 0.0,
    shadow_alpha: int = 22,
    specular: float = 0.0,
    roof_outline: bool = False,
) -> MeshLayer | None:
    if layer is None:
        return None
    layer.positions[:, 1] += float(y_offset)
    layer.rgba = rgba
    layer.shadow_alpha = int(shadow_alpha)
    layer.specular = float(specular)
    layer.roof_outline = bool(roof_outline)
    return layer


def build_city_scene(
    lon: float,
    lat: float,
    radius: float,
    *,
    refresh_osm: bool,
) -> SceneLayers:
    epsg = utm_epsg_for_lon_lat(lon, lat)
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    center_m = transform(to_metric.transform, Point(lon, lat))
    center_xy = tuple(center_m.coords[0])
    aoi_m = Point(*center_xy).buffer(radius, resolution=96)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    bbox = transform(to_wgs84.transform, aoi_m).bounds
    slug = cache_slug(lon, lat, radius)

    def cached(name: str, expr: str, *, include_relations: bool) -> list[dict]:
        query = overpass_query(bbox, expr, include_relations=include_relations)
        try:
            return fetch_overpass_query(query, CACHE_DIR / f"{slug}_{name}.json", refresh_osm)
        except RuntimeError as exc:
            print(f"[OSM] warning: {exc}")
            return []

    buildings = parse_polygon_features(
        cached("buildings", '["building"]', include_relations=True), to_metric, aoi_m
    )
    landuse = parse_polygon_features(
        cached("landuse", '["landuse"]', include_relations=True), to_metric, aoi_m
    )
    park_elements = cached("parks", '["leisure"="park"]', include_relations=True)
    park_elements += cached("natural_green", '["natural"~"^(wood|scrub)$"]', include_relations=True)
    parks = parse_polygon_features(park_elements, to_metric, aoi_m)
    water_elements = cached("water", '["natural"="water"]', include_relations=True)
    water_elements += cached("riverbanks", '["water"="river"]', include_relations=True)
    water = parse_polygon_features(water_elements, to_metric, aoi_m)
    roads = parse_line_features(
        cached("roads", '["highway"]', include_relations=False), to_metric, aoi_m
    )
    rails = parse_line_features(
        cached("rails", '["railway"]', include_relations=False), to_metric, aoi_m
    )

    landuse_local = [localize_feature(feature, center_xy) for feature in landuse]
    park_geoms = [
        localize_feature(feature, center_xy)
        for feature in landuse
        if str(feature["tags"].get("landuse", "")).lower() in GREEN_LANDUSE
    ]
    park_geoms.extend(localize_feature(feature, center_xy) for feature in parks)
    water_local = [localize_feature(feature, center_xy) for feature in water]
    roads_local = [localize_feature(feature, center_xy) for feature in roads]
    rails_local = [localize_feature(feature, center_xy) for feature in rails]
    buildings_local = [localize_feature(feature, center_xy) for feature in buildings]

    print(f"[OSM] buildings: {len(buildings_local)}")
    print(f"[OSM] landuse: {len(landuse_local)} | parks: {len(park_geoms)} | water: {len(water_local)}")
    print(f"[OSM] roads: {len(roads_local)} | rails: {len(rails_local)}")

    surfaces: list[SurfaceLayer] = []
    meshes: list[MeshLayer] = []
    roof_outlines: list[RoofOutlineLayer] = []
    focus_landmarks = select_focus_landmarks(buildings_local, radius=radius)

    def append_mesh(layer: MeshLayer | None) -> None:
        if layer is not None:
            meshes.append(layer)

    def append_surface(
        geom,
        rgba: tuple[int, int, int, int],
        *,
        elevation: float,
        specular: float = 0.0,
        reflectivity: float = 0.0,
    ) -> None:
        polygonal = _extract_polygonal(_fix_geom(geom))
        if polygonal is None or polygonal.is_empty:
            return
        surfaces.append(
            SurfaceLayer(
                geometry=polygonal,
                rgba=rgba,
                elevation=float(elevation),
                specular=float(specular),
                reflectivity=float(reflectivity),
            )
        )

    append_surface(Point(0.0, 0.0).buffer(radius, resolution=96), COLORS["base"], elevation=0.02)

    if landuse_local:
        append_surface(
            merge_surface_geometry(landuse_local, simplify_tolerance=1.4),
            COLORS["landuse"],
            elevation=0.5,
        )

    if park_geoms:
        append_surface(
            merge_surface_geometry(park_geoms, simplify_tolerance=1.2),
            COLORS["park"],
            elevation=0.51,
        )

    if water_local:
        append_surface(
            merge_surface_geometry(water_local, simplify_tolerance=0.9),
            COLORS["water"],
            elevation=-0.5,
            specular=WATER_SPECULAR_INTENSITY,
            reflectivity=0.22,
        )

    if roads_local:
        roads_union = unary_union([feature["geometry"] for feature in roads_local])
        roads_buf = simplify_geom(
            roads_union.buffer(
                3.0,
                cap_style=BufferCapStyle.round,
                join_style=BufferJoinStyle.round,
            ),
            0.7,
        )
        crown_buf = simplify_geom(
            roads_union.buffer(
                1.3,
                cap_style=BufferCapStyle.round,
                join_style=BufferJoinStyle.round,
            ),
            0.5,
        )
        append_surface(roads_buf, COLORS["road"], elevation=1.0)
        append_surface(crown_buf, COLORS["road_hi"], elevation=1.12)

    if rails_local:
        rails_union = unary_union([feature["geometry"] for feature in rails_local])
        rails_buf = simplify_geom(
            rails_union.buffer(
                2.0,
                cap_style=BufferCapStyle.round,
                join_style=BufferJoinStyle.round,
            ),
            0.6,
        )
        append_surface(rails_buf, COLORS["road"], elevation=1.0)

    binned: dict[str, list[dict]] = {"low": [], "mid": [], "high": [], "landmark": []}
    for feature in buildings_local:
        height = infer_building_height(feature["tags"])
        tags = dict(feature["tags"])
        tags["_height_m"] = height
        binned[building_bin(height)].append({"geometry": feature["geometry"], "tags": tags})

    for bin_name, color_key, roof_outline in (
        ("low", "building_low", True),
        ("mid", "building_mid", False),
        ("high", "building_high", False),
        ("landmark", "building_landmark", False),
    ):
        features = binned[bin_name]
        if not features:
            continue
        geojson = feature_collection(
            features,
            height_key="_height_m",
            default_height=12.0,
            simplify_tolerance=0.75,
        )
        for feature in features:
            roof_geom = prepare_polygonal_geom(feature["geometry"], simplify_tolerance=0.18)
            if roof_geom is None or roof_geom.is_empty:
                continue
            roof_outlines.append(
                RoofOutlineLayer(
                    geometry=roof_geom,
                    elevation=float(feature["tags"].get("_height_m", 12.0)),
                )
            )
        append_mesh(
            set_layer_style(
                mesh_from_geojson(geojson, default_height=12.0, height_key="_height_m"),
                COLORS[color_key],
                shadow_alpha=28,
                roof_outline=roof_outline,
            )
        )

    return SceneLayers(
        surfaces=surfaces,
        meshes=meshes,
        roof_outlines=roof_outlines,
        focus_landmarks=focus_landmarks,
        radius=radius,
    )


def build_city_mesh_layers(
    lon: float,
    lat: float,
    radius: float,
    *,
    refresh_osm: bool,
) -> tuple[list[MeshLayer], float]:
    scene = build_city_scene(lon, lat, radius, refresh_osm=refresh_osm)
    return scene.meshes, scene.radius


def project_points(
    points: np.ndarray,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    camera_up = normalize(np.cross(right, forward))
    rel = np.asarray(points, dtype=np.float32) - eye[None, :]
    cam_x = rel @ right
    cam_y = rel @ camera_up
    cam_z = rel @ forward
    tan_half = math.tan(math.radians(fov_deg) * 0.5)
    aspect = width / max(height, 1)
    screen_x = (cam_x / np.maximum(cam_z, 1e-6)) / (tan_half * aspect)
    screen_y = (cam_y / np.maximum(cam_z, 1e-6)) / tan_half
    return np.column_stack(
        [
            (screen_x * 0.5 + 0.5) * width,
            (0.5 - screen_y * 0.5) * height,
            cam_z,
        ]
    ).astype(np.float32)


def ring_to_world_points(coords: np.ndarray, elevation: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    return np.column_stack(
        [
            coords[:, 0],
            np.full(coords.shape[0], float(elevation), dtype=np.float32),
            -coords[:, 1],
        ]
    ).astype(np.float32)


def project_elevated_polygon_parts(
    geometry,
    *,
    elevation: float,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    parts: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for polygon in _polygon_parts(geometry):
        exterior_xy = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
        if exterior_xy.shape[0] < 3:
            continue
        exterior_proj = project_points(
            ring_to_world_points(exterior_xy, elevation),
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )
        if np.any(exterior_proj[:, 2] <= 1.0):
            continue
        hole_projs: list[np.ndarray] = []
        for interior in polygon.interiors:
            hole_xy = np.asarray(interior.coords[:-1], dtype=np.float32)
            if hole_xy.shape[0] < 3:
                continue
            hole_proj = project_points(
                ring_to_world_points(hole_xy, elevation),
                eye=eye,
                target=target,
                up=up,
                width=width,
                height=height,
                fov_deg=fov_deg,
            )
            if np.any(hole_proj[:, 2] <= 1.0):
                continue
            hole_projs.append(hole_proj)
        parts.append((exterior_proj, hole_projs))
    return parts


def project_surface_parts(
    surface: SurfaceLayer,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    return project_elevated_polygon_parts(
        surface.geometry,
        elevation=surface.elevation,
        eye=eye,
        target=target,
        up=up,
        width=width,
        height=height,
        fov_deg=fov_deg,
    )


def compute_fit_transform(
    point_sets: list[np.ndarray],
    *,
    width: int,
    height: int,
    margin_ratio: float = 0.055,
) -> tuple[float, float, float]:
    if not point_sets:
        return 1.0, 0.0, 0.0
    min_x = min(float(points[:, 0].min()) for points in point_sets if points.size > 0)
    max_x = max(float(points[:, 0].max()) for points in point_sets if points.size > 0)
    min_y = min(float(points[:, 1].min()) for points in point_sets if points.size > 0)
    max_y = max(float(points[:, 1].max()) for points in point_sets if points.size > 0)
    bbox_w = max(max_x - min_x, 1.0)
    bbox_h = max(max_y - min_y, 1.0)
    margin = max(width, height) * margin_ratio
    scale = min((width - 2.0 * margin) / bbox_w, (height - 2.0 * margin) / bbox_h)
    offset_x = width * 0.5 - scale * (min_x + max_x) * 0.5
    offset_y = height * 0.5 - scale * (min_y + max_y) * 0.5
    return float(scale), float(offset_x), float(offset_y)


def fit_polygon(points_xy: np.ndarray, *, scale: float, offset_x: float, offset_y: float) -> list[tuple[float, float]]:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    return [
        (float(point[0] * scale + offset_x), float(point[1] * scale + offset_y))
        for point in points_xy
    ]


def build_surface_mask(
    parts: list[tuple[np.ndarray, list[np.ndarray]]],
    *,
    width: int,
    height: int,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    for exterior, holes in parts:
        mask_draw.polygon(
            fit_polygon(exterior[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y),
            fill=255,
        )
        for hole in holes:
            mask_draw.polygon(
                fit_polygon(hole[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y),
                fill=0,
            )
    return mask


def alpha_bounds(
    alpha: np.ndarray,
    *,
    threshold: int = 6,
    pad: int = 0,
) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(alpha, dtype=np.uint8)
    mask = alpha > int(threshold)
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    height, width = alpha.shape
    return (
        max(0, int(xs.min()) - int(pad)),
        max(0, int(ys.min()) - int(pad)),
        min(width, int(xs.max()) + int(pad) + 1),
        min(height, int(ys.max()) + int(pad) + 1),
    )


def crop_subject(
    image: Image.Image,
    *,
    threshold: int = 6,
    pad_ratio: float = 0.024,
    min_pad: int = 12,
) -> Image.Image:
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    pad = max(int(min_pad), int(round(max(image.size) * float(pad_ratio))))
    bbox = alpha_bounds(alpha, threshold=threshold, pad=pad)
    if bbox is None:
        return image
    return image.crop(bbox)


def resize_to_fit(
    image: Image.Image,
    *,
    max_width: int,
    max_height: int,
) -> Image.Image:
    scale = min(max_width / max(image.width, 1), max_height / max(image.height, 1))
    target_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    if target_size == image.size:
        return image
    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize(target_size, resample=resampling)


def _value_noise(width: int, height: int, *, scale_px: float, seed: int) -> np.ndarray:
    scale_px = max(float(scale_px), 8.0)
    grid_w = max(4, int(math.ceil(width / scale_px)) + 2)
    grid_h = max(4, int(math.ceil(height / scale_px)) + 2)
    rng = np.random.default_rng(seed)
    grid = rng.random((grid_h, grid_w), dtype=np.float32)

    xs = np.linspace(0.0, float(grid_w - 1), width, dtype=np.float32)
    ys = np.linspace(0.0, float(grid_h - 1), height, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, grid_w - 1)
    y1 = np.clip(y0 + 1, 0, grid_h - 1)
    tx = (xs - x0.astype(np.float32))[None, :]
    ty = (ys - y0.astype(np.float32))[:, None]

    g00 = grid[y0[:, None], x0[None, :]]
    g10 = grid[y0[:, None], x1[None, :]]
    g01 = grid[y1[:, None], x0[None, :]]
    g11 = grid[y1[:, None], x1[None, :]]
    return (
        (1.0 - tx) * (1.0 - ty) * g00
        + tx * (1.0 - ty) * g10
        + (1.0 - tx) * ty * g01
        + tx * ty * g11
    )


def _blue_noise_tile(size: int = BLUE_NOISE_TILE_SIZE, *, seed: int = 1177) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.random((size, size), dtype=np.float32) - 0.5
    freq_x = np.fft.fftfreq(size).reshape(1, size)
    freq_y = np.fft.fftfreq(size).reshape(size, 1)
    freq_radius = np.sqrt(freq_x * freq_x + freq_y * freq_y)
    highpass = smoothstep(0.06, 0.45, freq_radius).astype(np.float32)
    shaped = np.fft.ifft2(np.fft.fft2(white) * highpass).real
    shaped -= shaped.min()
    shaped /= max(float(shaped.max()), 1e-6)
    return shaped.astype(np.float32)


BLUE_NOISE_TILE = _blue_noise_tile()


def make_poster_background(width: int, height: int) -> Image.Image:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    gradient16 = np.zeros((height, 3), dtype=np.float32)
    for index in range(len(BACKGROUND_STOPS) - 1):
        p0, c0 = BACKGROUND_STOPS[index]
        p1, c1 = BACKGROUND_STOPS[index + 1]
        if index == len(BACKGROUND_STOPS) - 2:
            segment = (y >= float(p0)) & (y <= float(p1))
        else:
            segment = (y >= float(p0)) & (y < float(p1))
        if not np.any(segment):
            continue
        t = ((y[segment] - float(p0)) / max(float(p1) - float(p0), 1e-6)).reshape(-1, 1)
        c0_16 = np.asarray(c0, dtype=np.float32).reshape(1, 3) * (65535.0 / 255.0)
        c1_16 = np.asarray(c1, dtype=np.float32).reshape(1, 3) * (65535.0 / 255.0)
        gradient16[segment] = c0_16 + (c1_16 - c0_16) * t

    rgb = np.repeat((gradient16 / 257.0)[:, None, :], width, axis=1)
    texture = _value_noise(width, height, scale_px=280.0, seed=3071)
    texture = (texture - 0.5)[:, :, None]
    rgb = np.clip(rgb + texture * (255.0 * 0.018), 0.0, 255.0)

    # 64x64 tileable blue-noise dither (one LSB) before final 8-bit quantization.
    dither = np.tile(
        BLUE_NOISE_TILE,
        (
            int(math.ceil(height / float(BLUE_NOISE_TILE.shape[0]))),
            int(math.ceil(width / float(BLUE_NOISE_TILE.shape[1]))),
        ),
    )[:height, :width]
    rgb = np.clip(rgb + (dither[:, :, None] - 0.5) * 2.0, 0.0, 255.0).astype(np.uint8)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate([rgb, alpha], axis=2), mode="RGBA")


def add_subject_shadow(
    canvas: Image.Image,
    subject: Image.Image,
    *,
    dest: tuple[int, int],
) -> None:
    max_dim = max(subject.size)
    shadow_specs = (
        ((70, 81, 96), 0.18, max(2, int(round(max_dim * 0.018))), (0.024, 0.050)),
        ((90, 99, 112), 0.09, max(4, int(round(max_dim * 0.050))), (0.010, 0.084)),
    )
    subject_alpha = subject.getchannel("A")
    for rgb, alpha_scale, blur_radius, offset_ratio in shadow_specs:
        layer = Image.new("RGBA", subject.size, rgb + (0,))
        layer.putalpha(
            subject_alpha.point(lambda value, a=alpha_scale: int(round(value * a)))
        )
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        offset = (
            dest[0] + int(round(subject.width * offset_ratio[0])),
            dest[1] + int(round(subject.height * offset_ratio[1])),
        )
        canvas.alpha_composite(layer, dest=offset)


def add_subject_halo(
    canvas: Image.Image,
    subject: Image.Image,
    *,
    dest: tuple[int, int],
) -> None:
    halo = Image.new("RGBA", subject.size, (255, 250, 242, 0))
    halo.putalpha(subject.getchannel("A").point(lambda value: int(round(value * 0.10))))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=max(4, int(round(max(subject.size) * 0.035)))))
    canvas.alpha_composite(halo, dest=dest)


def add_reflection_plane(
    canvas: Image.Image,
    subject: Image.Image,
    *,
    dest: tuple[int, int],
) -> None:
    alpha = subject.getchannel("A")
    if alpha.getbbox() is None:
        return
    reflected = subject.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGBA")
    reflected = reflected.filter(ImageFilter.GaussianBlur(radius=REFLECTION_PLANE_BLUR_PX))
    reflected_arr = np.asarray(reflected, dtype=np.float32)
    reflected_alpha = reflected_arr[:, :, 3] / 255.0
    fade_y = np.arange(reflected.height, dtype=np.float32)[:, None]
    vertical_fade = 1.0 - smoothstep(0.0, REFLECTION_PLANE_FADE_PX, fade_y).astype(np.float32)
    reflected_arr[:, :, 3] = np.clip(
        reflected_alpha * vertical_fade * REFLECTION_PLANE_OPACITY * 255.0,
        0.0,
        255.0,
    )
    reflection_image = Image.fromarray(reflected_arr.astype(np.uint8), mode="RGBA")
    reflection_dest = (
        dest[0],
        int(round(dest[1] + subject.height + REFLECTION_PLANE_OFFSET_PX)),
    )
    canvas.alpha_composite(reflection_image, dest=reflection_dest)


def add_landmark_glow(
    image: Image.Image,
    *,
    hero_mask: Image.Image | None,
) -> Image.Image:
    if hero_mask is None or hero_mask.getbbox() is None:
        return image
    glow_mask = hero_mask.filter(ImageFilter.MaxFilter(size=9))
    glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(radius=LANDMARK_GLOW_RADIUS_PX))
    glow_alpha = np.asarray(glow_mask, dtype=np.float32) / 255.0
    glow_alpha *= LANDMARK_GLOW_INTENSITY
    glow_layer = Image.new("RGBA", image.size, LANDMARK_GLOW_RGB + (0,))
    glow_layer.putalpha(
        Image.fromarray(np.clip(glow_alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    )
    return Image.alpha_composite(image, glow_layer)


def apply_focal_hierarchy(
    image: Image.Image,
    *,
    hero_mask: Image.Image | None,
    secondary_mask: Image.Image | None,
) -> Image.Image:
    if hero_mask is None and secondary_mask is None:
        return image
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb = arr[:, :, :3]
    hero = (
        np.asarray(hero_mask.convert("L"), dtype=np.float32) / 255.0
        if hero_mask is not None
        else np.zeros(rgb.shape[:2], dtype=np.float32)
    )
    hero = np.asarray(
        Image.fromarray(np.clip(hero * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        .filter(ImageFilter.GaussianBlur(radius=4.0)),
        dtype=np.float32,
    ) / 255.0
    secondary = (
        np.asarray(secondary_mask.convert("L"), dtype=np.float32) / 255.0
        if secondary_mask is not None
        else np.zeros(rgb.shape[:2], dtype=np.float32)
    )
    secondary = np.clip(secondary * (1.0 - hero), 0.0, 1.0)

    saturation = 1.0 + hero * FOCAL_HERO_SATURATION_BOOST - secondary * FOCAL_SECONDARY_DESATURATION
    arr[:, :, :3] = np.clip(blend_saturation(rgb, saturation), 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def apply_edge_atmospheric_fade(subject: Image.Image) -> Image.Image:
    arr = np.asarray(subject.convert("RGBA"), dtype=np.float32)
    alpha = arr[:, :, 3] / 255.0
    bbox = alpha_bounds(arr[:, :, 3].astype(np.uint8), threshold=6, pad=0)
    if bbox is None:
        return subject
    left, top, right, bottom = bbox
    cx = 0.5 * (left + right - 1)
    cy = 0.5 * (top + bottom - 1)
    rx = max(1.0, 0.5 * (right - left))
    ry = max(1.0, 0.5 * (bottom - top))
    xs = np.arange(subject.width, dtype=np.float32)[None, :]
    ys = np.arange(subject.height, dtype=np.float32)[:, None]
    radial = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    edge = smoothstep(EDGE_FADE_START, EDGE_FADE_END, radial).astype(np.float32)
    edge *= (alpha > 0.01).astype(np.float32)

    sat = 1.0 - (1.0 - EDGE_SATURATION_MIN) * edge
    rgb = blend_saturation(arr[:, :, :3], sat)
    rgb = rgb + (255.0 - rgb) * (EDGE_LIGHTNESS_GAIN * edge)[:, :, None]
    fog = np.asarray(EDGE_FOG_RGB, dtype=np.float32).reshape(1, 1, 3)
    rgb = rgb * (1.0 - 0.10 * edge[:, :, None]) + fog * (0.10 * edge[:, :, None])
    arr[:, :, :3] = np.clip(rgb, 0.0, 255.0)

    feather = 1.0 - 0.26 * smoothstep(0.955, 1.0, radial).astype(np.float32)
    arr[:, :, 3] = np.clip(arr[:, :, 3] * feather, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def apply_vignette(image: Image.Image, *, strength: float = VIGNETTE_DARKEN) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, image.width, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, image.height, dtype=np.float32)[:, None]
    radial = np.sqrt((xs / 0.98) ** 2 + (ys / 0.90) ** 2)
    feather_start = max(0.0, 1.0 - float(VIGNETTE_FEATHER))
    vignette = 1.0 - float(strength) * smoothstep(feather_start, 1.0, radial).astype(np.float32)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * vignette[:, :, None], 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def apply_micro_contrast(
    image: Image.Image,
    *,
    amount: float = MICRO_CONTRAST_AMOUNT,
    mask: Image.Image | None = None,
) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb = arr[:, :, :3]
    luma = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    luma_img = Image.fromarray(np.clip(luma, 0.0, 255.0).astype(np.uint8), mode="L")
    blurred = np.asarray(luma_img.filter(ImageFilter.GaussianBlur(radius=MICRO_CONTRAST_RADIUS_PX)), dtype=np.float32)
    detail = luma - blurred
    detail *= (np.abs(detail) >= MICRO_CONTRAST_THRESHOLD).astype(np.float32)
    luma_sharp = np.clip(luma + detail * float(amount), 0.0, 255.0)
    ratio = luma_sharp / np.maximum(luma, 1e-3)
    contrasted = np.clip(rgb * ratio[:, :, None], 0.0, 255.0)
    if mask is None:
        arr[:, :, :3] = contrasted
    else:
        mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        arr[:, :, :3] = rgb * (1.0 - mask_arr[:, :, None]) + contrasted * mask_arr[:, :, None]
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def apply_chromatic_refinement(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32)
    luminance = (
        arr[:, :, 0] * 0.2126
        + arr[:, :, 1] * 0.7152
        + arr[:, :, 2] * 0.0722
    )
    shadow_weight = (1.0 - smoothstep(38.0, 124.0, luminance)).astype(np.float32)
    highlight_weight = smoothstep(186.0, 255.0, luminance).astype(np.float32)

    # Lift shadows toward blue and nudge highlights toward warm yellow.
    arr[:, :, 2] += shadow_weight * 3.0
    arr[:, :, 0] += highlight_weight * 2.0
    arr[:, :, 1] += highlight_weight * 2.0
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGBA")


def apply_masked_highpass_sharpen(
    image: Image.Image,
    *,
    mask: Image.Image | None,
    opacity: float = SELECTIVE_SHARPEN_OPACITY,
) -> Image.Image:
    if mask is None:
        return image
    base = np.asarray(image.convert("RGBA"), dtype=np.float32)
    blur = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=SELECTIVE_SHARPEN_RADIUS_PX)).convert("RGBA"),
        dtype=np.float32,
    )
    highpass = np.clip(base[:, :, :3] - blur[:, :, :3] + 128.0, 0.0, 255.0) / 255.0
    base_rgb = base[:, :, :3] / 255.0
    overlay = np.where(
        base_rgb < 0.5,
        2.0 * base_rgb * highpass,
        1.0 - 2.0 * (1.0 - base_rgb) * (1.0 - highpass),
    )
    mixed = base_rgb * (1.0 - float(opacity)) + overlay * float(opacity)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    base[:, :, :3] = np.clip(
        base_rgb * (1.0 - mask_arr[:, :, None]) * 255.0 + mixed * mask_arr[:, :, None] * 255.0,
        0.0,
        255.0,
    )
    return Image.fromarray(base.astype(np.uint8), mode="RGBA")


def compose_poster(
    subject: Image.Image,
    *,
    width: int,
    height: int,
    radius_m: float,
    detail_mask: Image.Image | None = None,
    focus_landmark_mask: Image.Image | None = None,
    roof_outline_mask: Image.Image | None = None,
    roof_outline_tint_rgb: tuple[int, int, int] | None = None,
) -> Image.Image:
    scale = width / 1920.0
    title_size = max(40, int(round(72 * scale)))
    subtitle_size = max(16, int(round(28 * scale)))
    caption_size = max(12, int(round(14 * scale)))
    title_gap = max(6.0, 12.0 * scale)
    margin_x = width * 0.08
    top_margin = height * 0.025
    bottom_margin = height * 0.035
    map_gap = height * 0.018

    title_font = load_poster_font(size_px=title_size, weight="semibold")
    subtitle_font = load_poster_font(size_px=subtitle_size, weight="regular")
    caption_font = load_poster_font(size_px=caption_size, weight="regular")
    title_tracking = -0.02 * title_size
    subtitle_tracking = 0.04 * subtitle_size
    caption_tracking = 0.02 * caption_size
    subtitle_text = f"{format_radius_label(radius_m)} radius"

    title_box = measure_tracked_text(POSTER_TITLE, title_font, tracking_px=title_tracking)
    subtitle_box = measure_tracked_text(subtitle_text, subtitle_font, tracking_px=subtitle_tracking)
    caption_box = measure_tracked_text(POSTER_CAPTION, caption_font, tracking_px=caption_tracking)
    caption_top_in_band = title_box[1] + title_gap + subtitle_box[1] - caption_box[1]
    text_band_height = max(title_box[1] + title_gap + subtitle_box[1], caption_top_in_band + caption_box[1])

    alpha = np.asarray(subject.getchannel("A"), dtype=np.uint8)
    pad = max(8, int(round(max(subject.size) * 0.018)))
    bbox = alpha_bounds(alpha, threshold=6, pad=pad)
    if bbox is None:
        cropped = subject
        detail_cropped = detail_mask.convert("L") if detail_mask is not None else None
        focus_landmark_cropped = focus_landmark_mask.convert("L") if focus_landmark_mask is not None else None
        roof_outline_cropped = roof_outline_mask.convert("L") if roof_outline_mask is not None else None
    else:
        cropped = subject.crop(bbox)
        detail_cropped = detail_mask.crop(bbox).convert("L") if detail_mask is not None else None
        focus_landmark_cropped = (
            focus_landmark_mask.crop(bbox).convert("L") if focus_landmark_mask is not None else None
        )
        roof_outline_cropped = roof_outline_mask.crop(bbox).convert("L") if roof_outline_mask is not None else None

    cropped = apply_edge_atmospheric_fade(cropped)
    south_band_top = height - bottom_margin - text_band_height
    placed = resize_to_fit(
        cropped,
        max_width=max(1, int(round(width * 0.975))),
        max_height=max(1, int(round(max(south_band_top - top_margin - map_gap, height * 0.50)))),
    )
    placed_detail: Image.Image | None = None
    if detail_cropped is not None:
        placed_detail = resize_to_fit(
            detail_cropped.convert("RGBA"),
            max_width=max(1, int(round(width * 0.975))),
            max_height=max(1, int(round(max(south_band_top - top_margin - map_gap, height * 0.50)))),
        ).convert("L")
    placed_focus_landmarks: Image.Image | None = None
    if focus_landmark_cropped is not None:
        placed_focus_landmarks = resize_to_fit(
            focus_landmark_cropped.convert("RGBA"),
            max_width=max(1, int(round(width * 0.975))),
            max_height=max(1, int(round(max(south_band_top - top_margin - map_gap, height * 0.50)))),
        ).convert("L")
    placed_roof_outline: Image.Image | None = None
    if roof_outline_cropped is not None:
        placed_roof_outline = resize_to_fit(
            roof_outline_cropped.convert("RGBA"),
            max_width=max(1, int(round(width * 0.975))),
            max_height=max(1, int(round(max(south_band_top - top_margin - map_gap, height * 0.50)))),
        ).convert("L")

    canvas = make_poster_background(width, height)
    dest = (
        (width - placed.width) // 2,
        int(round(max(top_margin, south_band_top - placed.height - map_gap))),
    )
    subject_mask_canvas = Image.new("L", (width, height), 0)
    subject_mask_canvas.paste(placed.getchannel("A"), dest)
    add_reflection_plane(canvas, placed, dest=dest)
    add_subject_shadow(canvas, placed, dest=dest)
    add_subject_halo(canvas, placed, dest=dest)
    canvas.alpha_composite(placed, dest=dest)
    detail_canvas: Image.Image | None = None
    if placed_detail is not None:
        detail_canvas = Image.new("L", (width, height), 0)
        detail_canvas.paste(placed_detail, dest)
        detail_canvas = detail_canvas.filter(ImageFilter.MaxFilter(size=5))
        detail_canvas = detail_canvas.filter(ImageFilter.GaussianBlur(radius=0.7))
    focus_canvas: Image.Image | None = None
    if placed_focus_landmarks is not None:
        focus_canvas = Image.new("L", (width, height), 0)
        focus_canvas.paste(placed_focus_landmarks, dest)
    roof_outline_canvas: Image.Image | None = None
    if placed_roof_outline is not None:
        roof_outline_canvas = Image.new("L", (width, height), 0)
        roof_outline_canvas.paste(placed_roof_outline, dest)

    canvas = apply_micro_contrast(canvas, amount=MICRO_CONTRAST_AMOUNT)
    canvas = apply_masked_highpass_sharpen(canvas, mask=detail_canvas, opacity=SELECTIVE_SHARPEN_OPACITY)
    canvas = apply_focal_hierarchy(canvas, hero_mask=focus_canvas, secondary_mask=detail_canvas)
    canvas = add_landmark_glow(canvas, hero_mask=focus_canvas)
    canvas = apply_chromatic_refinement(canvas)
    canvas = apply_vignette(canvas, strength=VIGNETTE_DARKEN)
    if roof_outline_canvas is not None and roof_outline_tint_rgb is not None:
        canvas = apply_multiply_tint(
            canvas,
            tint_rgb=roof_outline_tint_rgb,
            mask_alpha=np.asarray(roof_outline_canvas, dtype=np.float32) / 255.0,
            opacity=1.0,
        )

    title_top = south_band_top
    subtitle_top = title_top + title_box[1] + title_gap
    caption_top = south_band_top + caption_top_in_band

    draw = ImageDraw.Draw(canvas)
    draw_tracked_text(
        draw,
        text=POSTER_TITLE,
        xy=(margin_x, title_top),
        font=title_font,
        fill=POSTER_TITLE_RGB,
        tracking_px=title_tracking,
        align="left",
    )
    draw_tracked_text(
        draw,
        text=subtitle_text,
        xy=(margin_x, subtitle_top),
        font=subtitle_font,
        fill=POSTER_SUBTITLE_RGB,
        tracking_px=subtitle_tracking,
        align="left",
    )
    draw_tracked_text(
        draw,
        text=POSTER_CAPTION,
        xy=(width - margin_x, caption_top),
        font=caption_font,
        fill=POSTER_CAPTION_RGB,
        tracking_px=caption_tracking,
        align="right",
    )
    return canvas


def srgb_icc_profile_bytes() -> bytes | None:
    try:
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def format_radius_label(radius_m: float) -> str:
    radius_m = float(radius_m)
    if radius_m >= 1000.0:
        km = radius_m / 1000.0
        if abs(km - round(km)) <= 1e-6:
            return f"{int(round(km))}km"
        return f"{km:.1f}km"
    return f"{int(round(radius_m))}m"


@lru_cache(maxsize=1)
def resolve_poster_font_paths() -> tuple[Path | None, Path | None]:
    search_dirs = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
        Path("C:/Windows/Fonts"),
    ]
    semibold_names = (
        "Inter-SemiBold.ttf",
        "Inter_18pt-SemiBold.ttf",
        "Inter-SemiBold.otf",
        "Inter_18pt-SemiBold.otf",
        "seguisb.ttf",
        "arialbd.ttf",
    )
    regular_names = (
        "Inter-Regular.ttf",
        "Inter_18pt-Regular.ttf",
        "Inter-Regular.otf",
        "Inter_18pt-Regular.otf",
        "segoeui.ttf",
        "arial.ttf",
    )

    def pick(names: tuple[str, ...]) -> Path | None:
        for directory in search_dirs:
            for name in names:
                candidate = directory / name
                if candidate.exists():
                    return candidate
        return None

    return pick(semibold_names), pick(regular_names)


def load_poster_font(*, size_px: int, weight: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    semibold_path, regular_path = resolve_poster_font_paths()
    path = semibold_path if weight == "semibold" else regular_path
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size=max(1, int(size_px)))
        except OSError:
            pass
    return ImageFont.load_default()


def font_text_width(font: ImageFont.ImageFont, text: str) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    bbox = font.getbbox(text)
    return float(bbox[2] - bbox[0])


def measure_tracked_text(text: str, font: ImageFont.ImageFont, *, tracking_px: float) -> tuple[float, float]:
    if not text:
        return 0.0, 0.0
    width = 0.0
    for index, glyph in enumerate(text):
        width += font_text_width(font, glyph)
        if index < len(text) - 1:
            width += float(tracking_px)
    bbox = font.getbbox(text)
    height = float(bbox[3] - bbox[1])
    return width, max(height, 1.0)


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    xy: tuple[float, float],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    tracking_px: float = 0.0,
    align: str = "left",
) -> tuple[float, float, float, float]:
    width, height = measure_tracked_text(text, font, tracking_px=tracking_px)
    start_x = float(xy[0]) if align == "left" else float(xy[0]) - width
    cursor_x = start_x
    y = float(xy[1])
    for index, glyph in enumerate(text):
        draw.text((cursor_x, y), glyph, font=font, fill=fill)
        cursor_x += font_text_width(font, glyph)
        if index < len(text) - 1:
            cursor_x += float(tracking_px)
    return (start_x, y, start_x + width, y + height)


def box_intersects_mask(
    mask_values: np.ndarray,
    *,
    box: tuple[float, float, float, float],
    clearance_px: float,
) -> bool:
    height, width = mask_values.shape
    x0, y0, x1, y1 = box
    ix0 = max(0, int(math.floor(x0 - clearance_px)))
    iy0 = max(0, int(math.floor(y0 - clearance_px)))
    ix1 = min(width, int(math.ceil(x1 + clearance_px)))
    iy1 = min(height, int(math.ceil(y1 + clearance_px)))
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    return bool(np.any(mask_values[iy0:iy1, ix0:ix1] > 6))


def find_clear_text_top(
    mask_values: np.ndarray,
    *,
    preferred_top: float,
    min_top: float,
    max_top: float,
    box_width: float,
    box_height: float,
    align_x: float,
    clearance_px: float,
    align: str,
    step_px: float,
) -> float:
    min_top = float(min_top)
    max_top = max(float(max_top), min_top)
    preferred_top = float(np.clip(preferred_top, min_top, max_top))
    step_px = max(float(step_px), 1.0)
    if align == "left":
        x0 = float(align_x)
    else:
        x0 = float(align_x) - float(box_width)
    x1 = x0 + float(box_width)

    candidates = [preferred_top]
    span = max_top - min_top
    steps = max(1, int(math.ceil(span / step_px)))
    for index in range(1, steps + 1):
        offset = index * step_px
        down = preferred_top + offset
        up = preferred_top - offset
        if down <= max_top:
            candidates.append(down)
        if up >= min_top:
            candidates.append(up)

    fallback = preferred_top
    best_score: tuple[float, float] | None = None
    for candidate in candidates:
        box = (x0, candidate, x1, candidate + float(box_height))
        if not box_intersects_mask(mask_values, box=box, clearance_px=clearance_px):
            return candidate
        overlap_region = (
            max(0, int(math.floor(x0 - clearance_px))),
            max(0, int(math.floor(candidate - clearance_px))),
            min(mask_values.shape[1], int(math.ceil(x1 + clearance_px))),
            min(mask_values.shape[0], int(math.ceil(candidate + float(box_height) + clearance_px))),
        )
        overlap = float(mask_values[overlap_region[1]:overlap_region[3], overlap_region[0]:overlap_region[2]].sum())
        score = (overlap, abs(candidate - preferred_top))
        if best_score is None or score < best_score:
            best_score = score
            fallback = candidate
    return fallback


def shade_rgba(
    rgba: tuple[int, int, int, int],
    *,
    normal: np.ndarray,
    light_dir: np.ndarray,
    view_dir: np.ndarray,
    specular: float = 0.0,
) -> tuple[int, int, int, int]:
    normal = normalize(normal)
    view_dir = normalize(view_dir)
    if float(np.dot(normal, view_dir)) < 0.0:
        normal = -normal
    diffuse = max(0.0, float(np.dot(normal, light_dir)))
    skylight = max(0.0, float(normal[1])) * 0.17
    half_vec = normalize(light_dir + view_dir)
    highlight = max(0.0, float(np.dot(normal, half_vec))) ** 20 * float(specular)
    shade = min(1.16, 0.60 + 0.31 * diffuse + skylight + highlight)
    base = np.array(rgba[:3], dtype=np.float32)
    rgb = np.clip(base * shade, 0.0, 255.0).astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(rgba[3]))


def apply_depth_grade_rgba(
    rgba: tuple[int, int, int, int],
    *,
    radial_t: float,
) -> tuple[int, int, int, int]:
    rgb = np.asarray(rgba[:3], dtype=np.float32)

    t = float(np.clip(radial_t, 0.0, 1.0))
    if t <= ATMOS_NEAR_END:
        saturation = ATMOS_SAT_NEAR
        value_gain = ATMOS_VALUE_NEAR
    elif t <= ATMOS_MID_END:
        mix = float(smoothstep(ATMOS_NEAR_END, ATMOS_MID_END, t))
        saturation = ATMOS_SAT_NEAR + (ATMOS_SAT_MID - ATMOS_SAT_NEAR) * mix
        value_gain = ATMOS_VALUE_NEAR + (ATMOS_VALUE_MID - ATMOS_VALUE_NEAR) * mix
    else:
        mix = float(smoothstep(ATMOS_MID_END, 1.0, t))
        saturation = ATMOS_SAT_MID + (ATMOS_SAT_FAR - ATMOS_SAT_MID) * mix
        value_gain = ATMOS_VALUE_MID + (ATMOS_VALUE_FAR - ATMOS_VALUE_MID) * mix

    rgb = blend_saturation(rgb, saturation).astype(np.float32)
    rgb = np.clip(rgb * (1.0 + value_gain), 0.0, 255.0).astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(rgba[3]))


def darken_rgb(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    arr = np.asarray(rgb, dtype=np.float32)
    arr = np.clip(arr * float(factor), 0.0, 255.0).astype(np.uint8)
    return int(arr[0]), int(arr[1]), int(arr[2])


def apply_multiply_tint(
    image: Image.Image,
    *,
    tint_rgb: tuple[int, int, int],
    mask_alpha: np.ndarray,
    opacity: float,
) -> Image.Image:
    base = np.asarray(image.convert("RGBA"), dtype=np.float32)
    tinted = base.copy()
    tint = np.asarray(tint_rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    tinted[:, :, :3] = np.clip(base[:, :, :3] * tint, 0.0, 255.0)
    alpha = np.clip(mask_alpha.astype(np.float32), 0.0, 1.0) * float(opacity)
    base[:, :, :3] = (
        base[:, :, :3] * (1.0 - alpha[:, :, None])
        + tinted[:, :, :3] * alpha[:, :, None]
    )
    return Image.fromarray(np.clip(base, 0.0, 255.0).astype(np.uint8), mode="RGBA")


def bilateral_like_blur(
    mask_values: np.ndarray,
    *,
    sigma_spatial: float,
    sigma_range: float,
    passes: int,
) -> np.ndarray:
    values = np.clip(np.asarray(mask_values, dtype=np.float32), 0.0, 1.0)
    for _ in range(max(1, int(passes))):
        blur = np.asarray(
            Image.fromarray(np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=float(sigma_spatial))),
            dtype=np.float32,
        ) / 255.0
        diff = blur - values
        weight = np.exp(-0.5 * (diff * diff) / max(float(sigma_range) ** 2, 1e-6))
        values = blur * weight + values * (1.0 - weight)
    return np.clip(values, 0.0, 1.0)


def render_water_surface_layer(
    *,
    width: int,
    height: int,
    mask: Image.Image,
    shaded_rgb: tuple[int, int, int],
    alpha_value: int,
    light_dir: np.ndarray,
    specular_intensity: float,
    far_dir_screen: np.ndarray,
) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    x0, y0, x1, y1 = bbox
    local_mask_img = mask.crop(bbox).convert("L")
    local_mask = np.asarray(local_mask_img, dtype=np.float32) / 255.0
    if float(local_mask.max()) <= 0.0:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    local_h, local_w = local_mask.shape
    local_rgb = np.empty((local_h, local_w, 3), dtype=np.float32)
    local_rgb[:, :, 0] = float(shaded_rgb[0])
    local_rgb[:, :, 1] = float(shaded_rgb[1])
    local_rgb[:, :, 2] = float(shaded_rgb[2])
    ys = np.linspace(-1.0, 1.0, local_h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, local_w, dtype=np.float32)[None, :]
    view_x = np.broadcast_to(xs * 0.45, local_mask.shape)
    view_y = np.broadcast_to(ys * 0.32, local_mask.shape)
    view_z = np.ones_like(local_mask, dtype=np.float32)
    view_dir_field = normalize_field(np.stack([view_x, view_y, view_z], axis=2))

    wave_u = xs * 4.2 + ys * 1.8
    wave_v = xs * -1.6 + ys * 5.0
    normal_x = np.broadcast_to(
        0.16 * np.sin(math.pi * wave_u) + 0.08 * np.cos(math.pi * wave_v * 0.75),
        local_mask.shape,
    )
    normal_y = np.broadcast_to(
        0.09 * np.cos(math.pi * wave_u * 0.85) - 0.11 * np.sin(math.pi * wave_v),
        local_mask.shape,
    )
    normal_z = np.full(local_mask.shape, max(0.56, 1.0 - WATER_ROUGHNESS * 0.30), dtype=np.float32)
    normal_field = normalize_field(np.stack([normal_x, normal_y, normal_z], axis=2))

    light_vec = normalize(np.array([light_dir[0], -light_dir[2] * 0.72, max(light_dir[1], 0.22)], dtype=np.float32))
    half_vec = normalize_field(view_dir_field + light_vec.reshape(1, 1, 3))
    ndotv = np.clip(np.sum(normal_field * view_dir_field, axis=2), 0.0, 1.0)
    ndotl = np.clip(np.sum(normal_field * light_vec.reshape(1, 1, 3), axis=2), 0.0, 1.0)
    ndoth = np.clip(np.sum(normal_field * half_vec, axis=2), 0.0, 1.0)
    fresnel = WATER_F0 + (1.0 - WATER_F0) * (1.0 - ndotv) ** WATER_FRESNEL_POWER
    lobe_power = max(7.0, 9.0 + (1.0 - WATER_ROUGHNESS) * 22.0)

    edge = ImageChops.subtract(
        local_mask_img.filter(ImageFilter.MaxFilter(size=7)),
        local_mask_img.filter(ImageFilter.MinFilter(size=7)),
    )
    edge_band = np.asarray(edge.filter(ImageFilter.GaussianBlur(radius=2.4)), dtype=np.float32) / 255.0

    far_dir = normalize(np.asarray(far_dir_screen, dtype=np.float32))
    if float(np.linalg.norm(far_dir)) <= 1e-6:
        far_dir = np.array([0.0, -1.0], dtype=np.float32)
    far_axis = xs * float(far_dir[0]) + ys * float(far_dir[1])
    far_weight = smoothstep(-0.28, 0.82, far_axis).astype(np.float32)
    far_edge_band = edge_band * far_weight

    sun_glint = np.power(ndoth, lobe_power) * (0.30 + 0.70 * ndotl)
    sheen_alpha = specular_intensity * (
        0.40 * sun_glint
        + 1.10 * np.power(fresnel, 0.72)
        + 2.10 * far_edge_band
    )
    sheen_alpha = np.clip(sheen_alpha * local_mask, 0.0, 0.74)

    white = np.full_like(local_rgb, 255.0)
    local_rgb = local_rgb * (1.0 - sheen_alpha[:, :, None]) + white * sheen_alpha[:, :, None]
    local_rgb += far_edge_band[:, :, None] * 22.0
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[y0:y1, x0:x1, :3] = np.clip(local_rgb, 0.0, 255.0).astype(np.uint8)
    rgba[y0:y1, x0:x1, 3] = np.clip(local_mask * float(alpha_value), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def reflection_rgba(
    rgba: tuple[int, int, int, int],
    water_rgba: tuple[int, int, int, int],
    *,
    normal: np.ndarray,
    light_dir: np.ndarray,
    view_dir: np.ndarray,
    reflectivity: float,
) -> tuple[int, int, int, int]:
    shaded = shade_rgba(
        rgba,
        normal=normal,
        light_dir=light_dir,
        view_dir=view_dir,
        specular=0.0,
    )
    reflected = np.asarray(shaded[:3], dtype=np.float32)
    water = np.asarray(water_rgba[:3], dtype=np.float32)
    rgb = np.clip(0.42 * reflected + 0.58 * water, 0.0, 255.0).astype(np.uint8)
    alpha = int(np.clip(82.0 * reflectivity, 0.0, 255.0))
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)


def apply_contact_ao(
    image: Image.Image,
    *,
    footprint_mask: Image.Image,
    radius_px: float,
) -> Image.Image:
    if footprint_mask.getbbox() is None:
        return image
    radius_px = max(float(radius_px), 1.0)
    kernel_scale = math.sqrt(max(float(AO_KERNEL_SAMPLES), 1.0) / 32.0)
    spread = footprint_mask.filter(ImageFilter.GaussianBlur(radius=radius_px * kernel_scale))
    inner = footprint_mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, radius_px * 0.42)))
    ring = ImageChops.subtract(spread, inner)
    ring = ImageChops.multiply(ring, footprint_mask.filter(ImageFilter.MaxFilter(size=7)))
    ao = np.asarray(ring, dtype=np.float32) / 255.0
    ao = np.clip(ao, 0.0, 1.0) ** 2
    ao = np.clip(ao - AO_BIAS, 0.0, 1.0) / max(1.0 - AO_BIAS, 1e-6)
    ao *= AO_NORMAL_INFLUENCE
    ao = bilateral_like_blur(
        ao,
        sigma_spatial=AO_BILATERAL_SIGMA_SPATIAL,
        sigma_range=AO_BILATERAL_SIGMA_RANGE,
        passes=AO_BLUR_PASSES,
    )
    return apply_multiply_tint(
        image,
        tint_rgb=AO_COLOR_RGB,
        mask_alpha=np.clip(ao * AO_INTENSITY, 0.0, 1.0),
        opacity=1.0,
    )


def build_projected_roof_outline_mask(
    *,
    roof_outlines: list[RoofOutlineLayer],
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    scale: float,
    offset_x: float,
    offset_y: float,
    supersample: int,
 ) -> Image.Image:
    if not roof_outlines:
        return Image.new("L", (width, height), 0)
    inset_px = max(1, int(round(ROOF_OUTLINE_WIDTH_PX * max(int(supersample), 1))))
    filter_size = inset_px * 2 + 1
    alpha = Image.new("L", (width, height), 0)
    pad = filter_size + 2

    for roof in roof_outlines:
        parts = project_elevated_polygon_parts(
            roof.geometry,
            elevation=roof.elevation,
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )
        for exterior, holes in parts:
            exterior_fit = fit_polygon(exterior[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y)
            if len(exterior_fit) < 3:
                continue
            xs = [point[0] for point in exterior_fit]
            ys = [point[1] for point in exterior_fit]
            x0 = max(0, int(math.floor(min(xs))) - pad)
            y0 = max(0, int(math.floor(min(ys))) - pad)
            x1 = min(width, int(math.ceil(max(xs))) + pad + 1)
            y1 = min(height, int(math.ceil(max(ys))) + pad + 1)
            if x1 <= x0 + 1 or y1 <= y0 + 1:
                continue

            local_mask = Image.new("L", (x1 - x0, y1 - y0), 0)
            local_draw = ImageDraw.Draw(local_mask)
            local_draw.polygon([(x - x0, y - y0) for x, y in exterior_fit], fill=255)
            for hole in holes:
                hole_fit = fit_polygon(hole[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y)
                if len(hole_fit) >= 3:
                    local_draw.polygon([(x - x0, y - y0) for x, y in hole_fit], fill=0)

            edge = ImageChops.subtract(local_mask, local_mask.filter(ImageFilter.MinFilter(size=filter_size)))
            if edge.getbbox() is None:
                continue
            edge = edge.point(lambda value: int(round(value * (ROOF_OUTLINE_ALPHA / 255.0))))
            current = alpha.crop((x0, y0, x1, y1))
            alpha.paste(ImageChops.lighter(current, edge), (x0, y0))

    return alpha


def build_projected_focus_landmark_mask(
    *,
    focus_landmarks: list[FocusLandmark],
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    if not focus_landmarks:
        return mask
    draw = ImageDraw.Draw(mask)
    for landmark in focus_landmarks:
        parts = project_elevated_polygon_parts(
            landmark.geometry,
            elevation=landmark.elevation,
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )
        for exterior, holes in parts:
            draw.polygon(
                fit_polygon(exterior[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y),
                fill=255,
            )
            for hole in holes:
                draw.polygon(
                    fit_polygon(hole[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y),
                    fill=0,
                )
    mask = mask.filter(ImageFilter.MaxFilter(size=7))
    return mask


def render_preview(
    scene: SceneLayers,
    *,
    out_path: Path,
    width: int,
    height: int,
    supersample: int = 2,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    supersample = max(1, int(supersample))
    render_width = max(1, width * supersample)
    render_height = max(1, height * supersample)
    image = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))

    radius = float(scene.radius)
    eye = np.array([-radius * 1.88, radius * 1.42, -radius * 1.58], dtype=np.float32)
    target = np.array([0.0, radius * 0.10, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    light_dir = normalize(np.array([-0.38, 0.95, -0.18], dtype=np.float32))
    fov_deg = 30.8
    view_dir = normalize(eye - target)
    world_to_px = min(render_width, render_height) / max(scene.radius * 2.0, 1e-6)

    surface_batches: list[tuple[SurfaceLayer, list[tuple[np.ndarray, list[np.ndarray]]]]] = []
    fit_inputs: list[np.ndarray] = []
    for surface in scene.surfaces:
        parts = project_surface_parts(
            surface,
            eye=eye,
            target=target,
            up=up,
            width=render_width,
            height=render_height,
            fov_deg=fov_deg,
        )
        if not parts:
            continue
        surface_batches.append((surface, parts))
        fit_inputs.extend(exterior[:, :2] for exterior, _ in parts)

    mesh_projections: list[tuple[MeshLayer, np.ndarray]] = []
    for layer in scene.meshes:
        projected = project_points(
            layer.positions,
            eye=eye,
            target=target,
            up=up,
            width=render_width,
            height=render_height,
            fov_deg=fov_deg,
        )
        mesh_projections.append((layer, projected))
        fit_inputs.append(projected[:, :2])

    scale, offset_x, offset_y = compute_fit_transform(
        fit_inputs,
        width=render_width,
        height=render_height,
        margin_ratio=0.040,
    )

    ground_far_world = np.array([target[0] - eye[0], 0.0, target[2] - eye[2]], dtype=np.float32)
    if float(np.linalg.norm(ground_far_world)) <= 1e-6:
        ground_far_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        ground_far_world = normalize(ground_far_world)
    far_probe = project_points(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [
                    ground_far_world[0] * radius * 0.30,
                    0.0,
                    ground_far_world[2] * radius * 0.30,
                ],
            ],
            dtype=np.float32,
        ),
        eye=eye,
        target=target,
        up=up,
        width=render_width,
        height=render_height,
        fov_deg=fov_deg,
    )
    far_dir_screen = far_probe[1, :2] - far_probe[0, :2]
    far_dir_screen = normalize(far_dir_screen)
    if float(np.linalg.norm(far_dir_screen)) <= 1e-6:
        far_dir_screen = np.array([0.0, -1.0], dtype=np.float32)

    building_center_x = render_width * 0.5
    building_center_y = render_height * 0.5
    building_radius_x = render_width * 0.5
    building_radius_y = render_height * 0.5
    if mesh_projections:
        mesh_points = []
        for _, projected in mesh_projections:
            pts = np.asarray(projected[:, :2], dtype=np.float32).copy()
            pts[:, 0] = pts[:, 0] * scale + offset_x
            pts[:, 1] = pts[:, 1] * scale + offset_y
            mesh_points.append(pts)
        if mesh_points:
            packed = np.vstack(mesh_points)
            min_x = float(packed[:, 0].min())
            max_x = float(packed[:, 0].max())
            min_y = float(packed[:, 1].min())
            max_y = float(packed[:, 1].max())
            building_center_x = 0.5 * (min_x + max_x)
            building_center_y = 0.5 * (min_y + max_y)
            building_radius_x = max(1.0, 0.5 * (max_x - min_x))
            building_radius_y = max(1.0, 0.5 * (max_y - min_y))

    reflective_surfaces: list[tuple[SurfaceLayer, Image.Image]] = []
    for surface, parts in surface_batches:
        mask = build_surface_mask(
            parts,
            width=render_width,
            height=render_height,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        shaded = shade_rgba(
            surface.rgba,
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            light_dir=light_dir,
            view_dir=view_dir,
            specular=surface.specular,
        )
        if surface.reflectivity > 0.0:
            layer_image = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
            for part in parts:
                part_mask = build_surface_mask(
                    [part],
                    width=render_width,
                    height=render_height,
                    scale=scale,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
                layer_image = Image.alpha_composite(
                    layer_image,
                    render_water_surface_layer(
                        width=render_width,
                        height=render_height,
                        mask=part_mask,
                        shaded_rgb=shaded[:3],
                        alpha_value=surface.rgba[3],
                        light_dir=light_dir,
                        specular_intensity=surface.specular,
                        far_dir_screen=far_dir_screen,
                    ),
                )
        else:
            alpha = mask.point(lambda value, a=surface.rgba[3]: (value * a + 127) // 255)
            layer_image = Image.new("RGBA", (render_width, render_height), shaded[:3] + (0,))
            layer_image.putalpha(alpha)
        image = Image.alpha_composite(image, layer_image)
        if surface.reflectivity > 0.0:
            reflective_surfaces.append((surface, mask))

    if reflective_surfaces and scene.meshes:
        transparent = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
        blur_radius = max(2, int(round(render_width / 1600)))
        for surface, mask in reflective_surfaces:
            reflection_image = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
            reflection_draw = ImageDraw.Draw(reflection_image, "RGBA")
            for layer in scene.meshes:
                reflected_positions = np.asarray(layer.positions, dtype=np.float32).copy()
                reflected_positions[:, 1] = (2.0 * surface.elevation) - reflected_positions[:, 1] - 0.02
                reflected_projected = project_points(
                    reflected_positions,
                    eye=eye,
                    target=target,
                    up=up,
                    width=render_width,
                    height=render_height,
                    fov_deg=fov_deg,
                )
                for tri in np.asarray(layer.indices, dtype=np.uint32):
                    world = reflected_positions[tri]
                    screen = reflected_projected[tri]
                    if np.any(screen[:, 2] <= 1.0):
                        continue
                    raw_normal = np.cross(world[1] - world[0], world[2] - world[0])
                    raw_length = float(np.linalg.norm(raw_normal))
                    if raw_length <= 1e-6:
                        continue
                    rgba = reflection_rgba(
                        layer.rgba,
                        surface.rgba,
                        normal=raw_normal,
                        light_dir=light_dir,
                        view_dir=eye - world.mean(axis=0),
                        reflectivity=surface.reflectivity,
                    )
                    reflection_draw.polygon(
                        fit_polygon(
                            screen[:, :2],
                            scale=scale,
                            offset_x=offset_x,
                            offset_y=offset_y,
                        ),
                        fill=rgba,
                    )
            reflection_image = reflection_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            image = Image.alpha_composite(image, Image.composite(reflection_image, transparent, mask))

    shadow_entries: list[tuple[float, list[tuple[float, float]], float, float]] = []
    triangles: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []
    ao_footprint_mask = Image.new("L", (render_width, render_height), 0)
    ao_footprint_draw = ImageDraw.Draw(ao_footprint_mask)
    detail_mask = Image.new("L", (render_width, render_height), 0)
    detail_draw = ImageDraw.Draw(detail_mask)

    for layer, projected in mesh_projections:
        layer_min_y = float(layer.positions[:, 1].min())
        layer_max_y = float(layer.positions[:, 1].max())
        layer_height = max(layer_max_y - layer_min_y, 1e-3)
        for tri in np.asarray(layer.indices, dtype=np.uint32):
            world = layer.positions[tri]
            screen = projected[tri]
            if np.any(screen[:, 2] <= 1.0):
                continue
            raw_normal = np.cross(world[1] - world[0], world[2] - world[0])
            raw_length = float(np.linalg.norm(raw_normal))
            if raw_length <= 1e-6:
                continue
            raw_normal = raw_normal / raw_length
            mean_y = float(world[:, 1].mean())
            is_roof = abs(float(raw_normal[1])) > 0.90 and mean_y > layer_min_y + 0.22
            if mean_y <= layer_min_y + 0.05 and abs(float(raw_normal[1])) > 0.92:
                continue

            poly = fit_polygon(screen[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y)
            detail_draw.polygon(poly, fill=255)

            if is_roof:
                ground_world = world.copy()
                ground_world[:, 1] = 0.0
                ground_proj = project_points(
                    ground_world,
                    eye=eye,
                    target=target,
                    up=up,
                    width=render_width,
                    height=render_height,
                    fov_deg=fov_deg,
                )
                if np.all(ground_proj[:, 2] > 1.0):
                    ao_footprint_draw.polygon(
                        fit_polygon(
                            ground_proj[:, :2],
                            scale=scale,
                            offset_x=offset_x,
                            offset_y=offset_y,
                        ),
                        fill=255,
                    )

            if np.all(world[:, 1] > 0.05) and layer.shadow_alpha > 0 and light_dir[1] > 1e-4:
                shadow_world = world.copy()
                shadow_scale = shadow_world[:, 1:2] / light_dir[1]
                shadow_world[:, 0:1] -= light_dir[0] * shadow_scale
                shadow_world[:, 2:3] -= light_dir[2] * shadow_scale
                shadow_world[:, 1] = 0.0
                shadow_proj = project_points(
                    shadow_world,
                    eye=eye,
                    target=target,
                    up=up,
                    width=render_width,
                    height=render_height,
                    fov_deg=fov_deg,
                )
                if np.all(shadow_proj[:, 2] > 1.0):
                    caster_height = max(0.0, mean_y - layer_min_y + SHADOW_NORMAL_BIAS)
                    shadow_vec = shadow_world[:, [0, 2]].mean(axis=0) - world[:, [0, 2]].mean(axis=0)
                    receiver_distance = float(np.linalg.norm(shadow_vec))
                    blocker = max(caster_height + SHADOW_DEPTH_BIAS, 0.5)
                    penumbra_world = (
                        SHADOW_LIGHT_SIZE_WORLD
                        * (receiver_distance / blocker)
                        * PCSS_PENUMBRA_SCALE
                    )
                    penumbra_world += (
                        PCSS_BLOCKER_SEARCH_TEXELS / SHADOW_MAP_RESOLUTION
                    ) * max(scene.radius * 0.75, 1.0)
                    blur_px = max(1.0, penumbra_world * world_to_px)
                    alpha_scale = min(1.20, 0.84 + caster_height / 56.0)
                    shadow_entries.append(
                        (
                            float(shadow_proj[:, 2].mean()),
                            fit_polygon(
                                shadow_proj[:, :2],
                                scale=scale,
                                offset_x=offset_x,
                                offset_y=offset_y,
                            ),
                            float(np.clip(layer.shadow_alpha * alpha_scale, 0.0, 255.0)),
                            float(np.clip(blur_px, 1.0, render_width / 210.0)),
                        )
                    )

            rgba = shade_rgba(
                layer.rgba,
                normal=raw_normal,
                light_dir=light_dir,
                view_dir=eye - world.mean(axis=0),
                specular=layer.specular,
            )
            if (not is_roof) and abs(float(raw_normal[1])) < 0.55:
                wall_t = float(smoothstep(0.0, 1.0, np.clip((mean_y - layer_min_y) / layer_height, 0.0, 1.0)))
                wall_gain = 0.84 + 0.28 * wall_t
                wall_mix = (1.0 - wall_t) * 0.08
                wall_base = np.asarray(rgba[:3], dtype=np.float32) * wall_gain
                wall_tone = np.clip(
                    wall_base * (1.0 - wall_mix) + np.asarray(SHADOW_COLOR_RGB, dtype=np.float32) * wall_mix,
                    0.0,
                    255.0,
                ).astype(np.uint8)
                rgba = (int(wall_tone[0]), int(wall_tone[1]), int(wall_tone[2]), rgba[3])
            centroid_x = (poly[0][0] + poly[1][0] + poly[2][0]) / 3.0
            centroid_y = (poly[0][1] + poly[1][1] + poly[2][1]) / 3.0
            radial_t = math.sqrt(
                ((centroid_x - building_center_x) / building_radius_x) ** 2
                + ((centroid_y - building_center_y) / building_radius_y) ** 2
            )
            rgba = apply_depth_grade_rgba(rgba, radial_t=float(np.clip(radial_t, 0.0, 1.0)))
            triangles.append(
                (
                    float(screen[:, 2].mean()),
                    poly,
                    rgba,
                )
            )

    if shadow_entries:
        blur_bands = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
        grouped: dict[float, list[tuple[float, list[tuple[float, float]], float, float]]] = {
            band: [] for band in blur_bands
        }
        for depth_value, polygon, alpha, blur_px in shadow_entries:
            band = min(blur_bands, key=lambda candidate: abs(candidate - blur_px))
            grouped[band].append((depth_value, polygon, alpha, blur_px))

        for blur_radius in blur_bands:
            entries = grouped.get(blur_radius, [])
            if not entries:
                continue
            layer = Image.new("L", (render_width, render_height), 0)
            layer_draw = ImageDraw.Draw(layer)
            for depth_value, polygon, alpha, _ in sorted(entries, key=lambda entry: entry[0], reverse=True):
                _ = depth_value
                layer_draw.polygon(polygon, fill=int(np.clip(alpha, 0.0, 255.0)))
            layer = layer.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
            layer_alpha = np.asarray(layer, dtype=np.float32) / 255.0
            layer_alpha = 1.0 - np.power(1.0 - layer_alpha, 1.35)
            layer_alpha *= max(0.60, 1.0 - blur_radius / 28.0)
            image = apply_multiply_tint(
                image,
                tint_rgb=SHADOW_COLOR_RGB,
                mask_alpha=np.clip(layer_alpha, 0.0, 1.0),
                opacity=SHADOW_MULTIPLY_OPACITY,
            )

    draw = ImageDraw.Draw(image, "RGBA")
    for _, polygon, rgba in sorted(triangles, key=lambda entry: entry[0], reverse=True):
        draw.polygon(polygon, fill=rgba)

    ao_radius_px = max(2.0, AO_RADIUS_WORLD_UNITS * world_to_px)
    image = apply_contact_ao(image, footprint_mask=ao_footprint_mask, radius_px=ao_radius_px)
    focus_landmark_mask = build_projected_focus_landmark_mask(
        focus_landmarks=scene.focus_landmarks,
        eye=eye,
        target=target,
        up=up,
        width=render_width,
        height=render_height,
        fov_deg=fov_deg,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    roof_outline_mask = build_projected_roof_outline_mask(
        roof_outlines=scene.roof_outlines,
        eye=eye,
        target=target,
        up=up,
        width=render_width,
        height=render_height,
        fov_deg=fov_deg,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        supersample=supersample,
    )

    if supersample > 1:
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize((width, height), resample=resampling)
        detail_mask = detail_mask.resize((width, height), resample=resampling)
        focus_landmark_mask = focus_landmark_mask.resize((width, height), resample=resampling)
        roof_outline_mask = roof_outline_mask.resize((width, height), resample=resampling)
    image = compose_poster(
        image,
        width=width,
        height=height,
        radius_m=scene.radius,
        detail_mask=detail_mask,
        focus_landmark_mask=focus_landmark_mask,
        roof_outline_mask=roof_outline_mask,
        roof_outline_tint_rgb=darken_rgb((255, 255, 255), ROOF_OUTLINE_DARKEN),
    )
    save_kwargs: dict[str, object] = {"format": "PNG", "compress_level": 0}
    icc_profile = srgb_icc_profile_bytes()
    if icc_profile is not None:
        save_kwargs["icc_profile"] = icc_profile
    image.save(out_path, **save_kwargs)


def main() -> int:
    args = parse_args()
    width, height = map(int, args.size)
    print(f"[City] center=({args.lon:.5f}, {args.lat:.5f}) radius={args.radius:.0f}m")
    scene = build_city_scene(
        args.lon,
        args.lat,
        args.radius,
        refresh_osm=bool(args.refresh_osm),
    )
    if not scene.surfaces and not scene.meshes:
        raise SystemExit("No renderable geometry was produced from the requested AOI.")
    render_preview(
        scene,
        out_path=args.output.resolve(),
        width=width,
        height=height,
        supersample=int(args.supersample),
    )
    print(f"[City] Wrote {display_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
