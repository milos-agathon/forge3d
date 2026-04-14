#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
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
    "building_low": (0xC8, 0x8A, 0x1E, 255),
    "building_mid": (0xB7, 0x72, 0x0E, 255),
    "building_high": (0x98, 0x59, 0x0A, 255),
    "landuse": (0xB8, 0x9C, 0x3A, 255),
    "park": (0x66, 0xA6, 0x1E, 255),
    "road": (0x7E, 0x87, 0x92, 255),
    "road_hi": (0xE6, 0xEB, 0xF1, 255),
    "water": (0x3E, 0x8F, 0xE0, 225),
    "base": (0xD8, 0xDD, 0xE3, 255),
}


@dataclass
class MeshLayer:
    positions: np.ndarray
    indices: np.ndarray
    rgba: tuple[int, int, int, int]
    shadow_alpha: int = 50
    specular: float = 0.0


@dataclass
class SurfaceLayer:
    geometry: Polygon | MultiPolygon
    rgba: tuple[int, int, int, int]
    elevation: float
    specular: float = 0.0


@dataclass
class SceneLayers:
    surfaces: list[SurfaceLayer]
    meshes: list[MeshLayer]
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
    parser.add_argument("--size", type=int, nargs=2, default=(1800, 1800), metavar=("WIDTH", "HEIGHT"))
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
    if height <= 12.0:
        return "low"
    if height <= 24.0:
        return "mid"
    return "high"


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
    return {
        "geometry": affinity.translate(feature["geometry"], xoff=-center_xy[0], yoff=-center_xy[1]),
        "tags": dict(feature["tags"]),
    }


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
    shadow_alpha: int = 50,
    specular: float = 0.0,
) -> MeshLayer | None:
    if layer is None:
        return None
    layer.positions[:, 1] += float(y_offset)
    layer.rgba = rgba
    layer.shadow_alpha = int(shadow_alpha)
    layer.specular = float(specular)
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

    def append_mesh(layer: MeshLayer | None) -> None:
        if layer is not None:
            meshes.append(layer)

    def append_surface(
        geom,
        rgba: tuple[int, int, int, int],
        *,
        elevation: float,
        specular: float = 0.0,
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
            specular=0.25,
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

    binned: dict[str, list[dict]] = {"low": [], "mid": [], "high": []}
    for feature in buildings_local:
        height = infer_building_height(feature["tags"])
        tags = dict(feature["tags"])
        tags["_height_m"] = height
        binned[building_bin(height)].append({"geometry": feature["geometry"], "tags": tags})

    for bin_name, color_key in (("low", "building_low"), ("mid", "building_mid"), ("high", "building_high")):
        features = binned[bin_name]
        if not features:
            continue
        geojson = feature_collection(
            features,
            height_key="_height_m",
            default_height=12.0,
            simplify_tolerance=0.75,
        )
        append_mesh(
            set_layer_style(
                mesh_from_geojson(geojson, default_height=12.0, height_key="_height_m"),
                COLORS[color_key],
                shadow_alpha=58,
            )
        )

    return SceneLayers(surfaces=surfaces, meshes=meshes, radius=radius)


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
    parts: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for polygon in _polygon_parts(surface.geometry):
        exterior_xy = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
        if exterior_xy.shape[0] < 3:
            continue
        exterior_proj = project_points(
            ring_to_world_points(exterior_xy, surface.elevation),
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
                ring_to_world_points(hole_xy, surface.elevation),
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
    skylight = max(0.0, float(normal[1])) * 0.18
    half_vec = normalize(light_dir + view_dir)
    highlight = max(0.0, float(np.dot(normal, half_vec))) ** 24 * float(specular)
    shade = min(1.25, 0.28 + 0.62 * diffuse + skylight + highlight)
    base = np.array(rgba[:3], dtype=np.float32)
    rgb = np.clip(base * shade, 0.0, 255.0).astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(rgba[3]))


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
    image = Image.new("RGBA", (render_width, render_height), (255, 255, 255, 255))
    shadow_image = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_image, "RGBA")

    radius = float(scene.radius)
    eye = np.array([-radius * 2.0, radius * 1.55, -radius * 2.0], dtype=np.float32)
    target = np.array([0.0, radius * 0.08, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    light_dir = normalize(np.array([-0.55, 1.0, -0.45], dtype=np.float32))
    fov_deg = 35.0
    view_dir = normalize(eye - target)

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
    )

    for surface, parts in surface_batches:
        mask = Image.new("L", (render_width, render_height), 0)
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
        shaded = shade_rgba(
            surface.rgba,
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            light_dir=light_dir,
            view_dir=view_dir,
            specular=surface.specular,
        )
        alpha = mask.point(lambda value, a=surface.rgba[3]: (value * a + 127) // 255)
        layer_image = Image.new("RGBA", (render_width, render_height), shaded[:3] + (0,))
        layer_image.putalpha(alpha)
        image = Image.alpha_composite(image, layer_image)

    shadow_triangles: list[tuple[float, list[tuple[float, float]], int]] = []
    triangles: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []

    for layer, projected in mesh_projections:
        layer_min_y = float(layer.positions[:, 1].min())
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
            if float(world[:, 1].mean()) <= layer_min_y + 0.05 and abs(float(raw_normal[1])) > 0.92:
                continue
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
                    shadow_triangles.append(
                        (
                            float(shadow_proj[:, 2].mean()),
                            fit_polygon(
                                shadow_proj[:, :2],
                                scale=scale,
                                offset_x=offset_x,
                                offset_y=offset_y,
                            ),
                            layer.shadow_alpha,
                        )
                    )

            rgba = shade_rgba(
                layer.rgba,
                normal=raw_normal,
                light_dir=light_dir,
                view_dir=eye - world.mean(axis=0),
                specular=layer.specular,
            )
            triangles.append(
                (
                    float(screen[:, 2].mean()),
                    fit_polygon(screen[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y),
                    rgba,
                )
            )

    for _, polygon, alpha in sorted(shadow_triangles, key=lambda entry: entry[0], reverse=True):
        shadow_draw.polygon(polygon, fill=(18, 20, 24, alpha))
    if shadow_triangles:
        shadow_image = shadow_image.filter(
            ImageFilter.GaussianBlur(radius=max(2, int(round(render_width / 900))))
        )
        image = Image.alpha_composite(image, shadow_image)

    draw = ImageDraw.Draw(image, "RGBA")
    for _, polygon, rgba in sorted(triangles, key=lambda entry: entry[0], reverse=True):
        draw.polygon(polygon, fill=rgba)

    if supersample > 1:
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize((width, height), resample=resampling)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=2))

    image.save(out_path, optimize=True)


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
