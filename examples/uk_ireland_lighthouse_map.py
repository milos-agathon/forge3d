#!/usr/bin/env python3
from __future__ import annotations

"""Render a lighthouse-at-night map of the UK and Ireland with forge3d.

This example fetches:

- Terrarium elevation tiles for the British Isles
- OpenStreetMap lighthouse features from Overpass

It then:

1. Reprojects the fetched DEM into a square Lambert Azimuthal grid
2. Renders the heightfield through ``forge3d.TerrainRenderer``
3. Uploads a dim directional fill plus lighthouse spot lights
4. Composites lighthouse halos and title treatment for a poster-like final image

Examples:
    python examples/uk_ireland_lighthouse_map.py
    python examples/uk_ireland_lighthouse_map.py --size 1600 --tile-zoom 8 --refresh-all
"""

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pyproj import CRS, Transformer

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.terrain_params import BloomSettings, ShadowSettings, TonemapSettings, make_terrain_params_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "uk_ireland_lighthouse_map" / "uk_ireland_lighthouses.png"
CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "uk_ireland_lighthouse_map"
USER_AGENT = "forge3d-lighthouse-map/1.0"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OVERPASS_URLS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
NATURAL_EARTH_COUNTRIES_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
COUNTRY_CODES = ("GBR", "IRL")
TARGET_CRS = CRS.from_proj4("+proj=laea +lat_0=55 +lon_0=-4 +datum=WGS84 +units=m +no_defs")
WGS84 = CRS.from_epsg(4326)
MAX_LIGHTS = 16
LIGHTHOUSE_LIGHT_CAP = MAX_LIGHTS - 1
LIGHTHOUSE_DEDUPE_M = 2_500.0
LIGHTHOUSE_COAST_DISTANCE_M = 4_000.0
LIGHTHOUSE_NAMED_COAST_DISTANCE_M = 8_000.0
LIGHTHOUSE_EXPLICIT_SEAMARK_DISTANCE_M = 16_000.0
LIGHTHOUSE_LANDMARK_COAST_DISTANCE_M = 3_000.0
LIGHTHOUSE_GLOW_DENSITY_RADIUS_M = 24_000.0
LIGHTHOUSE_SPOT_RANGE_M = 24_000.0
LIGHTHOUSE_SPOT_INNER_ANGLE_DEG = 13.0
LIGHTHOUSE_SPOT_CONE_ANGLE_DEG = 27.0
LIGHTHOUSE_SPOT_INTENSITY = 3.8e8
SEA_LEVEL_M = 0.0
EXTENT_PADDING = 0.11
BOUNDARY_PAD_DEG = 0.18


@dataclass(frozen=True)
class GeoBBox:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class SquareExtent:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def size(self) -> float:
        return float(self.max_x - self.min_x)


@dataclass
class Lighthouse:
    lon: float
    lat: float
    name: str | None
    tags: dict[str, object]
    proj_x: float = 0.0
    proj_y: float = 0.0
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    sea_dx: float = 0.0
    sea_dy: float = -1.0
    coast_distance_m: float = 0.0
    glow_weight: float = 1.0


@dataclass(frozen=True)
class PosterMapLayout:
    crop_box: tuple[int, int, int, int]
    dest_left: int
    dest_top: int
    placed_width: int
    placed_height: int

    @property
    def scale_x(self) -> float:
        crop_width = max(self.crop_box[2] - self.crop_box[0], 1)
        return self.placed_width / crop_width

    @property
    def scale_y(self) -> float:
        crop_height = max(self.crop_box[3] - self.crop_box[1], 1)
        return self.placed_height / crop_height

    def project(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        return (
            self.dest_left + (float(pixel_x) - self.crop_box[0]) * self.scale_x,
            self.dest_top + (float(pixel_y) - self.crop_box[1]) * self.scale_y,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096, help="Final poster size in pixels (square).")
    parser.add_argument(
        "--map-size",
        type=int,
        default=3840,
        help="Intermediate terrain render size in pixels (square).",
    )
    parser.add_argument("--tile-zoom", type=int, default=7, help="Terrarium zoom level.")
    parser.add_argument("--z-scale", type=float, default=48.0, help="Terrain vertical exaggeration.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument(
        "--boundary-data",
        type=Path,
        default=None,
        help="Optional boundary dataset path (.zip/.shp/.gpkg/.geojson). Defaults to Natural Earth admin-0 countries.",
    )
    parser.add_argument("--refresh-boundary", action="store_true", help="Re-fetch the boundary dataset.")
    parser.add_argument("--refresh-terrain", action="store_true", help="Re-fetch cached elevation tiles.")
    parser.add_argument("--refresh-osm", action="store_true", help="Re-fetch cached Overpass JSON.")
    parser.add_argument("--refresh-all", action="store_true", help="Refresh both terrain tiles and OSM data.")
    return parser.parse_args()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_font(size: int, *, bold: bool = False, family: str = "serif") -> ImageFont.ImageFont:
    if family == "sans":
        names = (
            ("DejaVuSans-Bold.ttf", "Segoe UI Bold.ttf", "arialbd.ttf")
            if bold
            else ("DejaVuSans.ttf", "Segoe UI.ttf", "arial.ttf")
        )
    else:
        names = (
            ("DejaVuSerif-Bold.ttf", "Georgia Bold.ttf", "arialbd.ttf")
            if bold
            else ("DejaVuSerif.ttf", "Georgia.ttf", "arial.ttf")
        )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


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
    return image.resize(target_size, Image.Resampling.LANCZOS)


def multiline_text_size(
    text: str,
    font: ImageFont.ImageFont,
    *,
    spacing: int = 0,
) -> tuple[int, int]:
    probe = ImageDraw.Draw(Image.new("L", (1, 1), 0))
    bbox = probe.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="left")
    return (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))


def fit_multiline_font(
    text: str,
    *,
    max_width: int,
    preferred_size: int,
    min_size: int,
    bold: bool = False,
    family: str = "serif",
    spacing: int = 0,
) -> ImageFont.ImageFont:
    for size in range(int(preferred_size), int(min_size) - 1, -1):
        font = load_font(size, bold=bold, family=family)
        width, _ = multiline_text_size(text, font, spacing=spacing)
        if width <= int(max_width):
            return font
    return load_font(int(min_size), bold=bold, family=family)


def normalize_xy(dx: float, dy: float, *, fallback: tuple[float, float] = (0.0, -1.0)) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return fallback
    return (dx / length, dy / length)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    span = max(float(edge1) - float(edge0), 1e-6)
    t = np.clip((np.asarray(x, dtype=np.float32) - float(edge0)) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def slippy_tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    scale = 1 << int(zoom)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * 0.5 * scale
    return x, y


def slippy_pixel_xy(lon: np.ndarray, lat: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05112878, 85.05112878)
    lon = np.asarray(lon, dtype=np.float64)
    scale = float((1 << int(zoom)) * 256)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = np.deg2rad(lat)
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / math.pi) * 0.5 * scale
    return x, y


def terrarium_rgb_to_height(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb[..., 0] * 256.0 + rgb[..., 1] + (rgb[..., 2] / 256.0)) - 32768.0


def decode_terrarium_tile(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return terrarium_rgb_to_height(arr).astype(np.float32)


def make_square_extent(points: Iterable[tuple[float, float]], padding_ratio: float) -> SquareExtent:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_side = 0.5 * max(max_x - min_x, max_y - min_y) * (1.0 + 2.0 * float(padding_ratio))
    return SquareExtent(cx - half_side, cy - half_side, cx + half_side, cy + half_side)


def farthest_point_indices(points: np.ndarray, count: int) -> list[int]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or count <= 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if count >= len(pts):
        return list(range(len(pts)))

    centroid = pts.mean(axis=0)
    first = int(np.argmax(np.sum((pts - centroid) ** 2, axis=1)))
    selected = [first]
    min_dist_sq = np.sum((pts - pts[first]) ** 2, axis=1)

    while len(selected) < count:
        candidate = int(np.argmax(min_dist_sq))
        selected.append(candidate)
        dist_sq = np.sum((pts - pts[candidate]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return selected


def _request_bytes(url: str, *, refresh: bool, cache_path: Path) -> bytes:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    cache_path.write_bytes(payload)
    return payload


def _fetch_json_payload(query: str, *, refresh: bool, cache_path: Path) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload = None
    last_error: Exception | None = None
    for endpoint in OVERPASS_URLS:
        for attempt in range(2):
            request = Request(
                endpoint,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                sleep(1.25 * (attempt + 1))
        if payload is not None:
            break

    if payload is None:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Overpass request failed for {cache_path.name}: {last_error}") from last_error

    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise SystemExit(
            "This example requires geopandas to load UK/Ireland boundary data. Install with: pip install geopandas"
        ) from exc
    return gpd


def _union_geometry(geometry_series):
    return geometry_series.union_all() if hasattr(geometry_series, "union_all") else geometry_series.unary_union


def bbox_from_bounds(
    bounds: tuple[float, float, float, float],
    *,
    pad_deg: float = 0.0,
) -> GeoBBox:
    west, south, east, north = map(float, bounds)
    return GeoBBox(
        west=max(-180.0, west - float(pad_deg)),
        south=max(-85.05112878, south - float(pad_deg)),
        east=min(180.0, east + float(pad_deg)),
        north=min(85.05112878, north + float(pad_deg)),
    )


def load_isles_boundary(
    boundary_path: Path | None,
    *,
    refresh: bool,
) -> tuple[object, object, GeoBBox]:
    if boundary_path is None:
        boundary_path = CACHE_DIR / "boundary" / "ne_10m_admin_0_countries.zip"
        _request_bytes(
            NATURAL_EARTH_COUNTRIES_URL,
            refresh=refresh,
            cache_path=boundary_path,
        )

    gpd = _require_geopandas()
    countries = gpd.read_file(boundary_path)
    matched = None
    for field in ("ADM0_A3", "ADM0_A3_US", "ISO_A3", "ISO_A3_EH", "SOV_A3"):
        if field in countries.columns:
            candidate = countries[countries[field].isin(COUNTRY_CODES)]
            if not candidate.empty:
                matched = candidate
                break
    if matched is None or matched.empty:
        raise RuntimeError(f"Could not find {COUNTRY_CODES!r} in boundary dataset: {display_path(boundary_path)}")

    boundary_wgs84 = matched.to_crs(WGS84)
    boundary_target = matched.to_crs(TARGET_CRS)
    geom_wgs84 = _union_geometry(boundary_wgs84.geometry)
    geom_target = _union_geometry(boundary_target.geometry)
    bbox = bbox_from_bounds(geom_wgs84.bounds, pad_deg=BOUNDARY_PAD_DEG)
    return geom_wgs84, geom_target, bbox


def rasterize_geometry_mask(
    *,
    extent: SquareExtent,
    shape: tuple[int, int],
    geometry: object,
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


def mask_heightmap_to_geometry(
    heightmap: np.ndarray,
    *,
    extent: SquareExtent,
    geometry: object,
) -> np.ndarray:
    mask = rasterize_geometry_mask(extent=extent, shape=heightmap.shape, geometry=geometry)
    return np.where(mask, np.asarray(heightmap, dtype=np.float32), np.float32(SEA_LEVEL_M))


def repair_interior_heightmap_voids(heightmap: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    filled = np.asarray(heightmap, dtype=np.float32).copy()
    land = np.asarray(land_mask, dtype=bool)
    missing = land & (filled <= 0.0)
    if not np.any(missing):
        return filled

    valid = land & ~missing
    height, width = filled.shape
    for _ in range(24):
        if not np.any(missing):
            break
        pad_values = np.pad(filled, 1, mode="edge")
        pad_valid = np.pad(valid.astype(np.float32), 1, mode="constant")
        neighbor_sum = np.zeros_like(filled, dtype=np.float32)
        neighbor_count = np.zeros_like(filled, dtype=np.float32)

        for oy in range(3):
            for ox in range(3):
                if oy == 1 and ox == 1:
                    continue
                neighbor_values = pad_values[oy:oy + height, ox:ox + width]
                neighbor_valid = pad_valid[oy:oy + height, ox:ox + width]
                neighbor_sum += neighbor_values * neighbor_valid
                neighbor_count += neighbor_valid

        fillable = missing & (neighbor_count > 0.0)
        if not np.any(fillable):
            break
        filled[fillable] = neighbor_sum[fillable] / neighbor_count[fillable]
        valid[fillable] = True
        missing = land & ~valid

    if np.any(missing):
        fallback = float(np.median(filled[valid])) if np.any(valid) else 0.0
        filled[missing] = fallback
    return filled


def apply_northern_relief_boost(heightmap: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    boosted = np.asarray(heightmap, dtype=np.float32).copy()
    land = np.asarray(land_mask, dtype=np.float32)
    land_heights = boosted[land > 0.5]
    hi = float(np.quantile(land_heights, 0.995)) if land_heights.size else 1.0
    height_t = np.clip(boosted / max(hi, 1.0), 0.0, 1.0)

    y = np.linspace(0.0, 1.0, boosted.shape[0], dtype=np.float32)[:, None]
    north_emphasis = 1.0 - smoothstep(0.34, 0.72, y)
    gain = 1.0 + north_emphasis * height_t * 0.22
    return np.where(land > 0.0, boosted * gain, boosted)


def filter_lighthouses_to_geometry(
    lighthouses: list[Lighthouse],
    geometry: object,
) -> list[Lighthouse]:
    from shapely.geometry import Point

    return [lighthouse for lighthouse in lighthouses if geometry.covers(Point(lighthouse.proj_x, lighthouse.proj_y))]


def _tag_is_truthy(tags: dict[str, object], key: str) -> bool:
    value = tags.get(key)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def filter_lighthouses_for_poster(
    lighthouses: list[Lighthouse],
    geometry: object,
) -> list[Lighthouse]:
    from shapely.geometry import Point

    coastline = geometry.boundary
    curated: list[Lighthouse] = []
    for lighthouse in lighthouses:
        lighthouse.coast_distance_m = float(Point(lighthouse.proj_x, lighthouse.proj_y).distance(coastline))
        seamark_type = str(lighthouse.tags.get("seamark:type") or "").strip().lower()
        name_text = " ".join(
            part
            for part in (
                lighthouse.name,
                str(lighthouse.tags.get("name") or ""),
                str(lighthouse.tags.get("seamark:name") or ""),
            )
            if part
        ).lower()
        if _tag_is_truthy(lighthouse.tags, "disused") or "disused" in name_text:
            continue
        if seamark_type == "landmark" and "lighthouse" not in name_text:
            continue

        if seamark_type == "landmark":
            max_distance_m = LIGHTHOUSE_LANDMARK_COAST_DISTANCE_M
        elif seamark_type == "lighthouse" or seamark_type.startswith("light_"):
            max_distance_m = LIGHTHOUSE_EXPLICIT_SEAMARK_DISTANCE_M
        elif lighthouse.name:
            max_distance_m = LIGHTHOUSE_NAMED_COAST_DISTANCE_M
        else:
            max_distance_m = LIGHTHOUSE_COAST_DISTANCE_M

        if lighthouse.coast_distance_m <= max_distance_m:
            curated.append(lighthouse)
    return curated


def assign_lighthouse_glow_weights(lighthouses: list[Lighthouse]) -> None:
    if not lighthouses:
        return

    if len(lighthouses) == 1:
        lighthouses[0].glow_weight = 1.0
        return

    points = np.asarray([(light.proj_x, light.proj_y) for light in lighthouses], dtype=np.float32)
    deltas = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    np.fill_diagonal(dist_sq, np.inf)

    neighbor_counts = np.sum(dist_sq < (LIGHTHOUSE_GLOW_DENSITY_RADIUS_M ** 2), axis=1)
    nearest = np.sqrt(np.min(dist_sq, axis=1))

    weights = np.full(len(lighthouses), 1.0, dtype=np.float32)
    weights -= np.clip(neighbor_counts, 0, 4).astype(np.float32) * 0.07
    weights -= np.clip((8_000.0 - nearest) / 8_000.0, 0.0, 1.0) * 0.18
    weights += np.clip((nearest - 16_000.0) / 40_000.0, 0.0, 0.10)
    weights = np.clip(weights, 0.62, 1.14)

    for lighthouse, weight in zip(lighthouses, weights):
        lighthouse.glow_weight = float(weight)


def fetch_terrarium_mosaic(bbox: GeoBBox, zoom: int, *, refresh: bool) -> tuple[np.ndarray, int, int]:
    west_tx, north_ty = slippy_tile_xy(bbox.west, bbox.north, zoom)
    east_tx, south_ty = slippy_tile_xy(bbox.east, bbox.south, zoom)
    tile_max = (1 << int(zoom)) - 1
    min_tx = int(np.clip(math.floor(min(west_tx, east_tx)), 0, tile_max))
    max_tx = int(np.clip(math.floor(max(west_tx, east_tx)), 0, tile_max))
    min_ty = int(np.clip(math.floor(min(north_ty, south_ty)), 0, tile_max))
    max_ty = int(np.clip(math.floor(max(north_ty, south_ty)), 0, tile_max))

    rows: list[np.ndarray] = []
    for ty in range(min_ty, max_ty + 1):
        tiles: list[np.ndarray] = []
        for tx in range(min_tx, max_tx + 1):
            cache_path = CACHE_DIR / "terrarium" / str(zoom) / str(tx) / f"{ty}.png"
            _request_bytes(
                TERRARIUM_URL.format(z=zoom, x=tx, y=ty),
                refresh=refresh,
                cache_path=cache_path,
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


def build_square_heightmap(
    bbox: GeoBBox,
    *,
    zoom: int,
    size: int,
    refresh: bool,
    extent: SquareExtent | None = None,
    mask_geometry: object | None = None,
) -> tuple[np.ndarray, SquareExtent]:
    mosaic, origin_tx, origin_ty = fetch_terrarium_mosaic(bbox, zoom, refresh=refresh)
    to_projected = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)
    to_wgs84 = Transformer.from_crs(TARGET_CRS, WGS84, always_xy=True)

    if extent is None:
        corners = [
            (bbox.west, bbox.south),
            (bbox.west, bbox.north),
            (bbox.east, bbox.south),
            (bbox.east, bbox.north),
        ]
        projected = [to_projected.transform(lon, lat) for lon, lat in corners]
        extent = make_square_extent(projected, EXTENT_PADDING)

    xs = np.linspace(extent.min_x, extent.max_x, int(size), dtype=np.float64)
    ys = np.linspace(extent.max_y, extent.min_y, int(size), dtype=np.float64)
    heightmap = np.zeros((int(size), int(size)), dtype=np.float32)

    row_chunk = 256
    for row_start in range(0, int(size), row_chunk):
        row_end = min(int(size), row_start + row_chunk)
        grid_x, grid_y = np.meshgrid(xs, ys[row_start:row_end])
        lon, lat = to_wgs84.transform(grid_x, grid_y)
        heightmap[row_start:row_end] = sample_terrarium_bilinear(
            mosaic,
            lon,
            lat,
            zoom=zoom,
            origin_tx=origin_tx,
            origin_ty=origin_ty,
        )

    heightmap = np.nan_to_num(heightmap, nan=SEA_LEVEL_M, posinf=SEA_LEVEL_M, neginf=SEA_LEVEL_M)
    if mask_geometry is not None:
        heightmap = mask_heightmap_to_geometry(heightmap, extent=extent, geometry=mask_geometry)
    return np.maximum(heightmap, SEA_LEVEL_M).astype(np.float32), extent


def fetch_lighthouses(bbox: GeoBBox, *, refresh: bool) -> list[Lighthouse]:
    query = (
        "[out:json][timeout:120];("
        f'node["man_made"="lighthouse"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        f'way["man_made"="lighthouse"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        f'relation["man_made"="lighthouse"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        f'node["seamark:type"="lighthouse"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        f'way["seamark:type"="lighthouse"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        f'relation["seamark:type"="lighthouse"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
        ");out center tags qt;"
    )
    cache_path = CACHE_DIR / "osm" / "uk_ie_lighthouses.json"
    payload = _fetch_json_payload(query, refresh=refresh, cache_path=cache_path)

    to_projected = Transformer.from_crs(WGS84, TARGET_CRS, always_xy=True)
    candidates: list[tuple[float, Lighthouse]] = []
    for element in payload.get("elements", []):
        center = element.get("center", {})
        lon = element.get("lon", center.get("lon"))
        lat = element.get("lat", center.get("lat"))
        if lon is None or lat is None:
            continue
        tags = dict(element.get("tags", {}))
        x, y = to_projected.transform(float(lon), float(lat))
        score = 0.0
        if tags.get("name") or tags.get("seamark:name"):
            score += 10.0
        if tags.get("man_made") == "lighthouse":
            score += 6.0
        if tags.get("seamark:type") == "lighthouse":
            score += 3.0
        if element.get("type") == "node":
            score += 2.0
        lighthouse = Lighthouse(
            lon=float(lon),
            lat=float(lat),
            name=tags.get("name") or tags.get("seamark:name"),
            tags=tags,
            proj_x=float(x),
            proj_y=float(y),
        )
        candidates.append((score, lighthouse))

    deduped: list[Lighthouse] = []
    for _, lighthouse in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(
            math.hypot(lighthouse.proj_x - other.proj_x, lighthouse.proj_y - other.proj_y)
            < LIGHTHOUSE_DEDUPE_M
            for other in deduped
        ):
            continue
        deduped.append(lighthouse)
    return deduped


def prepare_lighthouse_positions(
    lighthouses: list[Lighthouse],
    *,
    extent: SquareExtent,
    map_size: int,
    terrain_span: float,
    heightmap: np.ndarray,
    z_scale: float,
    land_mask: np.ndarray | None = None,
) -> None:
    if land_mask is None:
        base_mask = (heightmap > 1.0).astype(np.uint8) * 255
    else:
        base_mask = np.round(np.clip(np.asarray(land_mask, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    smooth_land = np.asarray(
        Image.fromarray(base_mask, mode="L").filter(ImageFilter.GaussianBlur(radius=8.0)),
        dtype=np.float32,
    ) / 255.0
    grad_y, grad_x = np.gradient(smooth_land)

    for lighthouse in lighthouses:
        u = (lighthouse.proj_x - extent.min_x) / extent.size
        v = (extent.max_y - lighthouse.proj_y) / extent.size
        lighthouse.pixel_x = float(np.clip(u * (map_size - 1), 0.0, map_size - 1))
        lighthouse.pixel_y = float(np.clip(v * (map_size - 1), 0.0, map_size - 1))

        ix = int(round(lighthouse.pixel_x))
        iy = int(round(lighthouse.pixel_y))
        height_m = float(heightmap[iy, ix])
        lighthouse.world_x = ((lighthouse.pixel_x / max(map_size - 1, 1)) - 0.5) * terrain_span
        lighthouse.world_y = ((lighthouse.pixel_y / max(map_size - 1, 1)) - 0.5) * terrain_span
        lighthouse.world_z = height_m * z_scale + 55.0 * z_scale

        sea_dx, sea_dy = normalize_xy(-float(grad_x[iy, ix]), -float(grad_y[iy, ix]))
        lighthouse.sea_dx = sea_dx
        lighthouse.sea_dy = sea_dy


def build_lighting_override(lighthouses: list[Lighthouse]) -> list[dict[str, object]]:
    points = np.asarray([(light.proj_x, light.proj_y) for light in lighthouses], dtype=np.float64)
    active = farthest_point_indices(points, min(LIGHTHOUSE_LIGHT_CAP, len(lighthouses)))

    lights: list[dict[str, object]] = [
        {
            "type": "directional",
            "azimuth": 315.0,
            "elevation": 78.0,
            "intensity": 0.12,
            "color": [0.20, 0.28, 0.38],
        }
    ]
    for index in active:
        lighthouse = lighthouses[index]
        dir_x, dir_y = normalize_xy(lighthouse.sea_dx, lighthouse.sea_dy)
        direction = np.asarray([dir_x, dir_y, -0.18], dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        lights.append(
            {
                "type": "spot",
                "position": [lighthouse.world_x, lighthouse.world_y, lighthouse.world_z],
                "direction": [float(direction[0]), float(direction[1]), float(direction[2])],
                "range": LIGHTHOUSE_SPOT_RANGE_M,
                "inner_angle": LIGHTHOUSE_SPOT_INNER_ANGLE_DEG,
                "cone_angle": LIGHTHOUSE_SPOT_CONE_ANGLE_DEG,
                "intensity": LIGHTHOUSE_SPOT_INTENSITY,
                "color": [0.92, 0.96, 1.0],
            }
        )
    return lights


def build_relief_plate(
    heightmap: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
    z_scale: float = 10.0,
) -> np.ndarray:
    heightmap = np.asarray(heightmap, dtype=np.float32)
    rows, cols = heightmap.shape

    if land_mask is None:
        land = (heightmap > 1.0).astype(np.float32)
    else:
        land = np.clip(np.asarray(land_mask, dtype=np.float32), 0.0, 1.0)

    mask_image = Image.fromarray(np.round(land * 255.0).astype(np.uint8), mode="L")
    smooth_mask = np.asarray(
        mask_image.filter(ImageFilter.GaussianBlur(radius=max(cols / 170.0, 1.8))),
        dtype=np.float32,
    ) / 255.0
    shoreline = np.clip(smooth_mask - land * 0.72, 0.0, 1.0)

    masked_height = heightmap * land
    land_heights = masked_height[land > 0.5]
    hi = float(np.quantile(land_heights, 0.995)) if land_heights.size else 1.0
    height_t = np.clip(masked_height / max(hi, 1.0), 0.0, 1.0)
    exaggeration = float(np.clip(float(z_scale) / 10.0, 1.0, 5.0))
    relief_gain = 1.0 + (exaggeration - 1.0) * 0.42
    ridge_gain = 1.0 + (exaggeration - 1.0) * 0.36

    grad_y, grad_x = np.gradient(masked_height)
    nx = -grad_x * 0.016 * exaggeration
    ny = -grad_y * 0.016 * exaggeration
    nz = np.full_like(heightmap, 1.0, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= np.maximum(norm, 1e-6)
    ny /= np.maximum(norm, 1e-6)
    nz /= np.maximum(norm, 1e-6)

    key_light = np.asarray([-0.62, -0.42, 0.66], dtype=np.float32)
    key_light /= max(float(np.linalg.norm(key_light)), 1e-6)
    fill_light = np.asarray([0.34, 0.18, 0.92], dtype=np.float32)
    fill_light /= max(float(np.linalg.norm(fill_light)), 1e-6)
    key = np.clip(nx * key_light[0] + ny * key_light[1] + nz * key_light[2], 0.0, 1.0)
    fill = np.clip(nx * fill_light[0] + ny * fill_light[1] + nz * fill_light[2], 0.0, 1.0)

    laplacian = (
        np.roll(masked_height, 1, axis=0)
        + np.roll(masked_height, -1, axis=0)
        + np.roll(masked_height, 1, axis=1)
        + np.roll(masked_height, -1, axis=1)
        - masked_height * 4.0
    ) * land
    lap_samples = np.abs(laplacian[land > 0.5])
    lap_scale = float(np.quantile(lap_samples, 0.96)) if lap_samples.size else 1.0
    ridge = np.clip(-laplacian / max(lap_scale, 1e-3), 0.0, 1.0)
    basin = np.clip(laplacian / max(lap_scale, 1e-3), 0.0, 1.0)

    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    rgb[:] = np.asarray([0.018, 0.025, 0.029], dtype=np.float32)
    rgb += height_t[:, :, None] * np.asarray([0.040, 0.046, 0.046], dtype=np.float32) * (1.12 + 0.10 * (relief_gain - 1.0))
    rgb += key[:, :, None] * np.asarray([0.112, 0.136, 0.132], dtype=np.float32) * (1.34 * relief_gain)
    rgb += fill[:, :, None] * np.asarray([0.046, 0.068, 0.076], dtype=np.float32) * (1.04 * (1.0 + 0.55 * (relief_gain - 1.0)))
    rgb += ridge[:, :, None] * np.asarray([0.072, 0.088, 0.084], dtype=np.float32) * (1.22 * ridge_gain)
    rgb -= basin[:, :, None] * np.asarray([0.032, 0.028, 0.024], dtype=np.float32) * (0.56 * (1.0 + 0.40 * (ridge_gain - 1.0)))
    rgb += shoreline[:, :, None] * np.asarray([0.028, 0.056, 0.058], dtype=np.float32) * (0.60 + 0.04 * (relief_gain - 1.0))
    rgb *= np.clip(land[:, :, None] * 0.88 + smooth_mask[:, :, None] * 0.12, 0.0, 1.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    alpha = np.clip(land * 232.0 + smooth_mask * 14.0 + shoreline * 42.0, 0.0, 255.0)
    plate = np.zeros((rows, cols, 4), dtype=np.uint8)
    plate[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    plate[:, :, 3] = np.round(alpha).astype(np.uint8)
    return plate


def _create_neutral_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode("ascii"))
        for _ in range(width * height):
            handle.write(bytes([128, 128, 128, 128]))


def build_night_colormap(domain: tuple[float, float]) -> f3d.Colormap1D:
    lo, hi = map(float, domain)
    span = max(hi - lo, 1.0)
    stops = (
        (lo + 0.00 * span, "#05090d"),
        (lo + 0.02 * span, "#071019"),
        (lo + 0.06 * span, "#09131b"),
        (lo + 0.18 * span, "#0b1418"),
        (lo + 0.42 * span, "#10171b"),
        (lo + 0.68 * span, "#152028"),
        (lo + 0.88 * span, "#21313a"),
        (lo + 1.00 * span, "#32454d"),
    )
    return f3d.Colormap1D.from_stops(stops=stops, domain=domain)


def build_render_params(
    *,
    output_size: int,
    terrain_span: float,
    domain: tuple[float, float],
    colormap: f3d.Colormap1D,
    z_scale: float,
) -> object:
    overlay = f3d.OverlayLayer.from_colormap1d(colormap, strength=1.0, domain=domain)
    shadows = ShadowSettings(
        enabled=False,
        technique="NONE",
        resolution=512,
        cascades=1,
        max_distance=max(terrain_span, 1.0),
        softness=0.0,
        intensity=0.0,
        slope_scale_bias=0.001,
        depth_bias=0.0005,
        normal_bias=0.0002,
        min_variance=1e-4,
        light_bleed_reduction=0.0,
        evsm_exponent=1.0,
        fade_start=1.0,
    )
    bloom = BloomSettings(enabled=True, threshold=0.08, softness=0.68, intensity=0.62, radius=1.6)
    tonemap = TonemapSettings(
        operator="aces",
        white_point=4.0,
        white_balance_enabled=True,
        temperature=7800.0,
        tint=-0.08,
    )
    config = make_terrain_params_config(
        size_px=(int(output_size), int(output_size)),
        render_scale=1.0,
        terrain_span=float(terrain_span),
        msaa_samples=4,
        z_scale=float(z_scale),
        exposure=0.74,
        domain=domain,
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=False,
        light_azimuth_deg=315.0,
        light_elevation_deg=78.0,
        sun_intensity=0.0,
        cam_radius=float(terrain_span) * 0.95,
        cam_phi_deg=0.0,
        cam_theta_deg=8.0,
        fov_y_deg=22.0,
        camera_mode="screen",
        clip=(0.1, max(terrain_span * 2.0, 20_000.0)),
        shadows=shadows,
        overlays=[overlay],
        bloom=bloom,
        tonemap=tonemap,
        lambert_contrast=0.16,
    )
    return f3d.TerrainRenderParams(config)


def render_terrain_map(
    heightmap: np.ndarray,
    *,
    map_size: int,
    terrain_span: float,
    z_scale: float,
    lights: list[dict[str, object]],
) -> np.ndarray:
    if not f3d.has_gpu():
        raise SystemExit("This example requires a GPU-backed forge3d build.")
    required = ("Session", "TerrainRenderer", "TerrainRenderParams", "MaterialSet", "IBL", "OverlayLayer")
    if not all(hasattr(f3d, name) for name in required):
        raise SystemExit("This example requires `maturin develop --release` for the native forge3d build.")

    valid_land = heightmap[heightmap > 0.0]
    hi = float(np.quantile(valid_land, 0.995)) if valid_land.size else 1_200.0
    domain = (SEA_LEVEL_M, max(hi, 1.0))
    colormap = build_night_colormap(domain)
    params = build_render_params(
        output_size=map_size,
        terrain_span=terrain_span,
        domain=domain,
        colormap=colormap,
        z_scale=z_scale,
    )

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    renderer.set_lights(lights)
    material_set = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.5,
        normal_strength=1.4,
        blend_sharpness=5.0,
    )

    with tempfile.TemporaryDirectory(prefix="forge3d-lighthouse-demo-") as temp_dir:
        hdr_path = Path(temp_dir) / "neutral.hdr"
        _create_neutral_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        ibl.set_base_resolution(64)
        frame = renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=params,
            heightmap=np.ascontiguousarray(heightmap, dtype=np.float32),
            target=None,
        )
    return frame.to_numpy()


def grade_render(arr: np.ndarray) -> np.ndarray:
    rgb = np.asarray(arr[:, :, :3], dtype=np.float32) / 255.0
    alpha = np.asarray(arr[:, :, 3], dtype=np.float32) / 255.0

    rgb = np.clip(rgb * np.array([0.74, 0.88, 1.02], dtype=np.float32), 0.0, 1.0)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.10)
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    )
    rgb *= (0.74 + luminance[:, :, None] * 0.34).astype(np.float32)
    rgb += np.array([0.010, 0.018, 0.024], dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    out = np.empty_like(arr)
    out[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    out[:, :, 3] = np.round(alpha * 255.0).astype(np.uint8)
    return out


def apply_screen_glow(base_rgb: np.ndarray, glow_rgb: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - np.clip(base_rgb, 0.0, 1.0)) * (1.0 - np.clip(glow_rgb, 0.0, 1.0))


def build_poster_background(poster_size: int) -> Image.Image:
    grid_y, grid_x = np.mgrid[0:poster_size, 0:poster_size].astype(np.float32)
    x = grid_x / max(float(poster_size - 1), 1.0)
    y = grid_y / max(float(poster_size - 1), 1.0)

    rgb = np.zeros((poster_size, poster_size, 3), dtype=np.float32)
    rgb[:] = np.array([0.116, 0.188, 0.192], dtype=np.float32)
    rgb += (1.0 - y)[:, :, None] * np.array([0.020, 0.026, 0.020], dtype=np.float32) * 0.38

    left_glow = np.exp(-(((x - 0.30) / 0.32) ** 2 + ((y - 0.45) / 0.30) ** 2))
    right_glow = np.exp(-(((x - 0.82) / 0.24) ** 2 + ((y - 0.40) / 0.18) ** 2))
    lower_glow = np.exp(-(((x - 0.42) / 0.48) ** 2 + ((y - 0.86) / 0.20) ** 2))
    rgb += left_glow[:, :, None] * np.array([0.026, 0.058, 0.056], dtype=np.float32) * 0.40
    rgb += right_glow[:, :, None] * np.array([0.012, 0.024, 0.022], dtype=np.float32) * 0.22
    rgb += lower_glow[:, :, None] * np.array([0.012, 0.020, 0.018], dtype=np.float32) * 0.18

    radial = np.sqrt(((x - 0.50) / 0.88) ** 2 + ((y - 0.50) / 0.76) ** 2)
    vignette = 1.0 - 0.14 * smoothstep(0.26, 1.14, radial)
    rgb *= vignette[:, :, None]

    rng = np.random.default_rng(17)
    grain = rng.normal(0.0, 0.010, size=(poster_size, poster_size, 1)).astype(np.float32)
    rgb = np.clip(rgb + grain, 0.0, 1.0)

    rgba = np.empty((poster_size, poster_size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def crush_landmass_tones(terrain_rgba: np.ndarray, land_alpha: np.ndarray) -> np.ndarray:
    terrain_rgba = np.asarray(terrain_rgba, dtype=np.uint8)
    land_alpha = np.clip(np.asarray(land_alpha, dtype=np.float32), 0.0, 1.0)

    rgb = terrain_rgba[:, :, :3].astype(np.float32) / 255.0
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    )
    interior = smoothstep(0.28, 0.82, land_alpha)
    coastal = np.clip(land_alpha - interior * 0.92, 0.0, 1.0)
    land_fill = np.array([8.0, 15.0, 18.0], dtype=np.float32) / 255.0
    sculpt = land_fill + luminance[:, :, None] * np.array([0.054, 0.064, 0.066], dtype=np.float32) * 0.58
    detail = rgb * (0.52 + luminance[:, :, None] * 0.50)

    mix = interior[:, :, None] * 0.54
    rgb = rgb * (1.0 - mix) + (sculpt * 0.62 + detail * 0.38) * mix
    rgb *= 0.84 + interior[:, :, None] * 0.06
    rgb += coastal[:, :, None] * np.array([0.004, 0.011, 0.012], dtype=np.float32) * 0.26

    out = terrain_rgba.copy()
    out[:, :, :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def compress_relief_plate(relief_rgba: np.ndarray) -> np.ndarray:
    relief_rgba = np.asarray(relief_rgba, dtype=np.uint8)
    arr = relief_rgba.astype(np.float32) / 255.0
    alpha = arr[:, :, 3]
    interior = smoothstep(0.24, 0.84, alpha)
    coastal = np.clip(alpha - interior * 0.90, 0.0, 1.0)

    arr[:, :, :3] *= (0.48 + interior[:, :, None] * 0.24 + coastal[:, :, None] * 0.18).astype(np.float32)
    arr[:, :, 3] = alpha * (0.46 + interior * 0.11 + coastal * 0.20)

    out = np.empty_like(relief_rgba)
    out[:, :, :3] = np.round(np.clip(arr[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    out[:, :, 3] = np.round(np.clip(arr[:, :, 3], 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def apply_poster_finish(image: Image.Image) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    height, width = arr.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = grid_x / max(float(width - 1), 1.0)
    y = grid_y / max(float(height - 1), 1.0)

    radial = np.sqrt(((x - 0.48) / 0.86) ** 2 + ((y - 0.49) / 0.70) ** 2)
    vignette = 1.0 - 0.34 * smoothstep(0.16, 1.02, radial)
    text_shadow = np.exp(-(((x - 0.82) / 0.23) ** 2 + ((y - 0.73) / 0.32) ** 2))
    vignette *= 1.0 - text_shadow * 0.076

    luminance = (
        arr[:, :, 0] * 0.2126
        + arr[:, :, 1] * 0.7152
        + arr[:, :, 2] * 0.0722
    )
    rng = np.random.default_rng(31)
    fine_grain = rng.normal(0.0, 1.0, size=(height, width, 1)).astype(np.float32)
    coarse_h = max(2, height // 6)
    coarse_w = max(2, width // 6)
    coarse_seed = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
    coarse_image = Image.fromarray(
        np.round(np.clip((coarse_seed * 0.18 + 0.5), 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).resize((width, height), Image.Resampling.BILINEAR)
    coarse_grain = (np.asarray(coarse_image, dtype=np.float32)[:, :, None] / 255.0 - 0.5) * 2.0
    grain = fine_grain * 0.020 + coarse_grain * 0.024
    grain *= (0.98 + (1.0 - luminance)[:, :, None] * 0.42).astype(np.float32)

    arr[:, :, :3] *= vignette[:, :, None]
    arr[:, :, :3] = np.clip(arr[:, :, :3] + grain, 0.0, 1.0)

    out = np.empty_like(arr)
    out[:, :, :3] = arr[:, :, :3]
    out[:, :, 3] = np.asarray(image, dtype=np.float32)[:, :, 3] / 255.0
    return Image.fromarray(np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")

def plan_map_layout(alpha_mask: Image.Image, *, poster_size: int) -> PosterMapLayout:
    crop_pad = max(8, int(round(max(alpha_mask.size) * 0.010)))
    crop_box = alpha_bounds(np.asarray(alpha_mask, dtype=np.uint8), threshold=6, pad=crop_pad)
    if crop_box is None:
        crop_box = (0, 0, alpha_mask.width, alpha_mask.height)

    crop_width = max(crop_box[2] - crop_box[0], 1)
    crop_height = max(crop_box[3] - crop_box[1], 1)
    scale = min((poster_size * 0.79) / crop_width, (poster_size * 0.988) / crop_height)
    placed_width = max(1, int(round(crop_width * scale)))
    placed_height = max(1, int(round(crop_height * scale)))

    dest_left = int(round(poster_size * 0.004))
    dest_top = int(round(poster_size * 0.50 - placed_height * 0.50))
    dest_top = max(int(round(poster_size * 0.004)), dest_top)
    dest_top = min(dest_top, poster_size - placed_height - int(round(poster_size * 0.010)))
    return PosterMapLayout(
        crop_box=crop_box,
        dest_left=dest_left,
        dest_top=dest_top,
        placed_width=placed_width,
        placed_height=placed_height,
    )


def build_map_subject(
    terrain_rgba: np.ndarray,
    relief_rgba: np.ndarray,
    *,
    poster_size: int,
) -> tuple[Image.Image, Image.Image, Image.Image, PosterMapLayout]:
    raw_alpha = np.asarray(relief_rgba[:, :, 3], dtype=np.float32) / 255.0
    terrain_image = Image.fromarray(crush_landmass_tones(grade_render(terrain_rgba), raw_alpha), mode="RGBA")
    relief_image = Image.fromarray(compress_relief_plate(relief_rgba), mode="RGBA")
    terrain_image = Image.blend(terrain_image, Image.new("RGBA", terrain_image.size, (8, 16, 22, 255)), 0.08)
    terrain_image.alpha_composite(relief_image)

    subject_alpha = np.clip((raw_alpha - 0.08) / 0.78, 0.0, 1.0)
    glow_alpha = np.clip((raw_alpha - 0.02) / 0.96, 0.0, 1.0)
    subject_alpha_mask = Image.fromarray(np.round(subject_alpha * 255.0).astype(np.uint8), mode="L")
    glow_alpha_mask = Image.fromarray(np.round(glow_alpha * 255.0).astype(np.uint8), mode="L")

    layout = plan_map_layout(subject_alpha_mask, poster_size=poster_size)
    crop_box = layout.crop_box
    subject = terrain_image.crop(crop_box).resize((layout.placed_width, layout.placed_height), Image.Resampling.LANCZOS)
    placed_alpha = subject_alpha_mask.crop(crop_box).resize(
        (layout.placed_width, layout.placed_height),
        Image.Resampling.LANCZOS,
    )
    placed_glow_alpha = glow_alpha_mask.crop(crop_box).resize(
        (layout.placed_width, layout.placed_height),
        Image.Resampling.LANCZOS,
    )
    placed_alpha = placed_alpha.filter(ImageFilter.GaussianBlur(radius=max(0.35, poster_size / 3200.0)))
    subject.putalpha(placed_alpha)
    return subject, placed_alpha, placed_glow_alpha, layout


def render_coast_glow(alpha_canvas: Image.Image, *, poster_size: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(alpha_canvas, dtype=np.float32) / 255.0
    soft = np.asarray(
        alpha_canvas.filter(ImageFilter.GaussianBlur(radius=max(15.0, poster_size / 48.0))),
        dtype=np.float32,
    ) / 255.0
    near = np.asarray(
        alpha_canvas.filter(ImageFilter.GaussianBlur(radius=max(2.8, poster_size / 228.0))),
        dtype=np.float32,
    ) / 255.0
    rim = np.clip(near - base * 0.78, 0.0, 1.0)
    return soft, rim


def render_glow_layers(
    poster_size: int,
    lighthouses: list[Lighthouse],
    active_count: int,
    *,
    layout: PosterMapLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wash_mask = Image.new("L", (poster_size, poster_size), 0)
    beam_mask = Image.new("L", (poster_size, poster_size), 0)
    core_mask = Image.new("L", (poster_size, poster_size), 0)
    pin_mask = Image.new("L", (poster_size, poster_size), 0)
    star_mask = Image.new("L", (poster_size, poster_size), 0)
    wash_draw = ImageDraw.Draw(wash_mask)
    beam_draw = ImageDraw.Draw(beam_mask)
    core_draw = ImageDraw.Draw(core_mask)
    pin_draw = ImageDraw.Draw(pin_mask)
    star_draw = ImageDraw.Draw(star_mask)

    active_points = np.asarray([(light.proj_x, light.proj_y) for light in lighthouses], dtype=np.float64)
    active_indices = set(farthest_point_indices(active_points, min(active_count, len(lighthouses))))
    sparkle_order = farthest_point_indices(active_points, min(5, len(lighthouses)))
    sparkle_strengths = [1.0, 0.92, 0.84, 0.76, 0.68]
    sparkle_map = {
        index: sparkle_strengths[rank]
        for rank, index in enumerate(sparkle_order)
    }
    crop_left, crop_top, crop_right, crop_bottom = layout.crop_box
    base_radius = max(1.48, poster_size / 720.0)
    core_radius = max(1.12, poster_size / 900.0)
    wash_len = poster_size * 0.030
    beam_len = poster_size * 0.058
    beam_width = max(1, int(round(poster_size / 420.0)))

    for index, lighthouse in enumerate(lighthouses):
        if not (crop_left <= lighthouse.pixel_x <= crop_right and crop_top <= lighthouse.pixel_y <= crop_bottom):
            continue
        x, y = layout.project(lighthouse.pixel_x, lighthouse.pixel_y)
        weight = float(np.clip(lighthouse.glow_weight, 0.62, 1.14))
        cluster = float(np.clip((1.02 - weight) / 0.40, 0.0, 1.0))
        halo_scale = 1.0 - cluster * 0.34
        halo_energy = 1.0 - cluster * 0.16
        radius = base_radius * (0.98 + weight * 0.12) * (1.0 - cluster * 0.02)
        core = core_radius * (1.03 + weight * 0.10)
        pin = max(0.64, core_radius * 0.90)
        wash_width = max(4.2, poster_size / 356.0) * (1.00 + weight * 0.20) * halo_scale
        spill = wash_len * (0.92 + weight * 0.12) * halo_scale
        sea_dx, sea_dy = normalize_xy(lighthouse.sea_dx, lighthouse.sea_dy)
        wash_fill = int(round(np.clip((142.0 + weight * 40.0) * halo_energy, 112.0, 206.0)))
        core_fill = int(round(np.clip(234.0 + weight * 18.0, 232.0, 255.0)))
        pin_fill = 255

        wash_draw.ellipse(
            (x - wash_width * 0.22, y - wash_width * 0.22, x + wash_width * 0.22, y + wash_width * 0.22),
            fill=max(16, int(round(wash_fill * 0.14))),
        )
        wash_draw.line(
            (
                x + sea_dx * wash_width * 0.34,
                y + sea_dy * wash_width * 0.34,
                x + sea_dx * spill,
                y + sea_dy * spill,
            ),
            fill=wash_fill,
            width=max(2, int(round(wash_width * 2.2))),
        )
        lobe_x = x + sea_dx * (spill * 0.72 + wash_width * 0.34)
        lobe_y = y + sea_dy * (spill * 0.72 + wash_width * 0.34)
        wash_draw.ellipse(
            (
                lobe_x - wash_width * 1.00,
                lobe_y - wash_width * 1.00,
                lobe_x + wash_width * 1.00,
                lobe_y + wash_width * 1.00,
            ),
            fill=int(round(wash_fill * 0.94)),
        )
        tip_x = x + sea_dx * (spill * 1.16 + wash_width * 0.50)
        tip_y = y + sea_dy * (spill * 1.16 + wash_width * 0.50)
        wash_draw.ellipse(
            (
                tip_x - wash_width * 0.88,
                tip_y - wash_width * 0.88,
                tip_x + wash_width * 0.88,
                tip_y + wash_width * 0.88,
            ),
            fill=max(22, int(round(wash_fill * 0.48))),
        )

        core_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=core_fill)
        core_draw.ellipse((x - core, y - core, x + core, y + core), fill=255)
        pin_draw.ellipse((x - pin, y - pin, x + pin, y + pin), fill=pin_fill)

        sparkle_strength = float(sparkle_map.get(index, 0.0))
        if sparkle_strength > 0.0:
            star = radius * (1.34 + sparkle_strength * 0.44)
            pin_star = pin * (1.18 + sparkle_strength * 0.22)
            core_draw.ellipse((x - star, y - star, x + star, y + star), fill=255)
            pin_draw.ellipse((x - pin_star, y - pin_star, x + pin_star, y + pin_star), fill=255)
            star_draw.ellipse(
                (x - pin_star * 0.92, y - pin_star * 0.92, x + pin_star * 0.92, y + pin_star * 0.92),
                fill=int(round(208.0 + sparkle_strength * 47.0)),
            )
        if index in active_indices:
            beam_start_x = x + sea_dx * radius * 0.15
            beam_start_y = y + sea_dy * radius * 0.15
            x2 = x + sea_dx * beam_len
            y2 = y + sea_dy * beam_len
            beam_fill = int(round(np.clip(168.0 + weight * 44.0, 156.0, 236.0)))
            beam_draw.line((beam_start_x, beam_start_y, x2, y2), fill=beam_fill, width=beam_width)

    halo_far = np.asarray(
        wash_mask.filter(ImageFilter.GaussianBlur(radius=max(17.2, poster_size / 48.0))),
        dtype=np.float32,
    ) / 255.0
    halo_wash = np.asarray(
        wash_mask.filter(ImageFilter.GaussianBlur(radius=max(11.4, poster_size / 70.0))),
        dtype=np.float32,
    ) / 255.0
    halo_core = np.asarray(
        core_mask.filter(ImageFilter.GaussianBlur(radius=max(1.7, poster_size / 360.0))),
        dtype=np.float32,
    ) / 255.0
    beams = np.asarray(
        beam_mask.filter(ImageFilter.GaussianBlur(radius=max(3.6, poster_size / 132.0))),
        dtype=np.float32,
    ) / 255.0
    pins = np.asarray(
        pin_mask.filter(ImageFilter.GaussianBlur(radius=max(0.8, poster_size / 1200.0))),
        dtype=np.float32,
    ) / 255.0
    return halo_far, halo_wash, halo_core, beams, pins, np.asarray(star_mask, dtype=np.float32) / 255.0


def compose_poster(
    terrain_rgba: np.ndarray,
    *,
    relief_rgba: np.ndarray,
    poster_size: int,
    lighthouses: list[Lighthouse],
    title: str,
) -> Image.Image:
    subject, placed_alpha, placed_glow_alpha, layout = build_map_subject(
        terrain_rgba,
        relief_rgba,
        poster_size=poster_size,
    )
    canvas = build_poster_background(poster_size)

    coast_alpha = Image.new("L", (poster_size, poster_size), 0)
    coast_alpha.paste(placed_glow_alpha, (layout.dest_left, layout.dest_top))
    coast_soft, coast_rim = render_coast_glow(coast_alpha, poster_size=poster_size)

    canvas_np = np.asarray(canvas, dtype=np.float32) / 255.0
    coast_glow = np.zeros_like(canvas_np[:, :, :3])
    coast_glow += coast_soft[:, :, None] * np.array([0.15, 0.32, 0.31], dtype=np.float32) * 0.60
    coast_glow += coast_rim[:, :, None] * np.array([0.54, 0.82, 0.82], dtype=np.float32) * 0.40
    canvas_np[:, :, :3] = apply_screen_glow(canvas_np[:, :, :3], coast_glow)

    image = Image.fromarray(np.round(np.clip(canvas_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    image.alpha_composite(subject, dest=(layout.dest_left, layout.dest_top))

    terrain_np = np.asarray(image, dtype=np.float32) / 255.0
    halo_far, halo_wash, halo_core, beams, pins, stars = render_glow_layers(
        poster_size,
        lighthouses,
        LIGHTHOUSE_LIGHT_CAP,
        layout=layout,
    )
    outer_glow = np.zeros_like(terrain_np[:, :, :3])
    outer_glow += halo_far[:, :, None] * np.array([0.384, 0.676, 0.694], dtype=np.float32) * 0.80
    outer_glow += halo_wash[:, :, None] * np.array([0.566, 0.832, 0.838], dtype=np.float32) * 1.16
    outer_glow += beams[:, :, None] * np.array([0.726, 0.864, 0.872], dtype=np.float32) * 0.22
    terrain_np[:, :, :3] = apply_screen_glow(terrain_np[:, :, :3], np.clip(outer_glow, 0.0, 0.82))

    core_glow = np.zeros_like(terrain_np[:, :, :3])
    core_glow += halo_core[:, :, None] * np.array([1.0, 0.942, 0.812], dtype=np.float32) * 1.32
    core_glow += pins[:, :, None] * np.array([1.0, 0.934, 0.768], dtype=np.float32) * 2.18
    core_glow += stars[:, :, None] * np.array([1.0, 0.928, 0.724], dtype=np.float32) * 2.44
    terrain_np[:, :, :3] = apply_screen_glow(terrain_np[:, :, :3], np.clip(core_glow, 0.0, 1.0))
    warm_core_mask = np.clip(halo_core * 0.08 + pins * 0.18 + stars * 0.28, 0.0, 0.34)
    warm_core_color = np.array([1.0, 0.941, 0.816], dtype=np.float32)
    terrain_np[:, :, :3] = (
        terrain_np[:, :, :3] * (1.0 - warm_core_mask[:, :, None])
        + warm_core_color[None, None, :] * warm_core_mask[:, :, None]
    )
    image = Image.fromarray(np.round(np.clip(terrain_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")

    draw = ImageDraw.Draw(image)
    title_spacing = max(4, poster_size // 128)
    caption_spacing = max(4, poster_size // 170)
    text_right = int(round(poster_size * 0.92))
    text_left = max(int(round(poster_size * 0.54)), layout.dest_left + layout.placed_width + int(round(poster_size * 0.010)))
    available_width = max(132, text_right - text_left)
    overline = "LIGHTHOUSES OF THE"
    metadata = (
        f"{len(lighthouses)} coastal OSM lighthouse features\n"
        "Relief: Mapzen Terrarium\n"
        "Source: OpenStreetMap contributors"
    )

    overline_font = fit_multiline_font(
        overline,
        max_width=available_width,
        preferred_size=max(12, poster_size // 82),
        min_size=max(10, poster_size // 100),
        family="sans",
    )
    title_font = fit_multiline_font(
        title,
        max_width=available_width,
        preferred_size=max(26, poster_size // 30),
        min_size=max(18, poster_size // 42),
        bold=True,
        family="serif",
        spacing=title_spacing,
    )
    caption_font = fit_multiline_font(
        metadata,
        max_width=available_width,
        preferred_size=max(11, poster_size // 110),
        min_size=max(10, poster_size // 136),
        family="sans",
        spacing=caption_spacing,
    )

    overline_size = multiline_text_size(overline, overline_font)
    title_size = multiline_text_size(title, title_font, spacing=title_spacing)
    caption_size = multiline_text_size(metadata, caption_font, spacing=caption_spacing)
    gap_a = max(8, poster_size // 110)
    gap_b = max(14, poster_size // 58)
    block_height = overline_size[1] + gap_a + title_size[1] + gap_b + caption_size[1]
    text_top = int(round(layout.dest_top + layout.placed_height * 0.33))
    text_top = max(int(round(poster_size * 0.16)), text_top)
    text_top = min(text_top, poster_size - int(round(poster_size * 0.11)) - block_height)

    rule_y = text_top - max(8, poster_size // 96)
    draw.line(
        (text_left, rule_y, text_right, rule_y),
        fill=(62, 99, 107, 255),
        width=max(1, poster_size // 512),
    )

    overline_x = text_right - overline_size[0]
    title_x = text_right - title_size[0]
    caption_x = text_right - caption_size[0]
    draw.text(
        (overline_x, text_top),
        overline,
        fill=(150, 167, 170, 255),
        font=overline_font,
    )
    draw.multiline_text(
        (title_x, text_top + overline_size[1] + gap_a),
        title,
        fill=(232, 236, 237, 255),
        font=title_font,
        spacing=title_spacing,
        align="right",
    )
    draw.multiline_text(
        (caption_x, text_top + overline_size[1] + gap_a + title_size[1] + gap_b),
        metadata,
        fill=(126, 142, 145, 255),
        font=caption_font,
        spacing=caption_spacing,
        align="right",
    )

    return apply_poster_finish(image)


def main() -> int:
    args = parse_args()
    refresh = bool(args.refresh_all)
    _, boundary_target, bbox = load_isles_boundary(
        args.boundary_data,
        refresh=refresh or bool(args.refresh_boundary),
    )
    boundary_bounds = boundary_target.bounds
    extent = make_square_extent(
        [
            (boundary_bounds[0], boundary_bounds[1]),
            (boundary_bounds[0], boundary_bounds[3]),
            (boundary_bounds[2], boundary_bounds[1]),
            (boundary_bounds[2], boundary_bounds[3]),
        ],
        EXTENT_PADDING,
    )

    print(f"[Map] bbox=({bbox.west:.1f}, {bbox.south:.1f}, {bbox.east:.1f}, {bbox.north:.1f})")
    print(f"[Map] fetching terrain at z={args.tile_zoom} and OSM lighthouses...")
    heightmap, extent = build_square_heightmap(
        bbox,
        zoom=int(args.tile_zoom),
        size=int(args.map_size),
        refresh=refresh or bool(args.refresh_terrain),
        extent=extent,
        mask_geometry=boundary_target,
    )
    land_mask = rasterize_geometry_mask(extent=extent, shape=heightmap.shape, geometry=boundary_target)
    heightmap = repair_interior_heightmap_voids(heightmap, land_mask)
    heightmap = apply_northern_relief_boost(heightmap, land_mask)
    lighthouses = filter_lighthouses_to_geometry(
        fetch_lighthouses(bbox, refresh=refresh or bool(args.refresh_osm)),
        boundary_target,
    )
    lighthouses = filter_lighthouses_for_poster(lighthouses, boundary_target)
    assign_lighthouse_glow_weights(lighthouses)
    prepare_lighthouse_positions(
        lighthouses,
        extent=extent,
        map_size=int(args.map_size),
        terrain_span=extent.size,
        heightmap=heightmap,
        z_scale=float(args.z_scale),
        land_mask=land_mask,
    )
    lights = build_lighting_override(lighthouses)
    relief_rgba = build_relief_plate(heightmap, land_mask=land_mask, z_scale=float(args.z_scale))

    print(
        f"[Map] heightmap={heightmap.shape[1]}x{heightmap.shape[0]} "
        f"span={extent.size/1000.0:.1f}km lighthouses={len(lighthouses)} gpu_lights={len(lights)}"
    )
    render = render_terrain_map(
        heightmap,
        map_size=int(args.map_size),
        terrain_span=extent.size,
        z_scale=float(args.z_scale),
        lights=lights,
    )
    poster = compose_poster(
        render,
        relief_rgba=relief_rgba,
        poster_size=int(args.size),
        lighthouses=lighthouses,
        title="UNITED KINGDOM\n& IRELAND",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster.save(args.output, format="PNG", compress_level=0)
    print(f"[Map] wrote {display_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
