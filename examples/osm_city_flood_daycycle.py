#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import transform, unary_union
from shapely.prepared import prep

from _import_shim import ensure_repo_import

ensure_repo_import()

import osm_city_demo as city  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALENCIA_LON = -0.3763
VALENCIA_LAT = 39.4746
DEFAULT_RADIUS_M = 850.0
DEFAULT_DURATION_SECONDS = 8.0
DEFAULT_FPS = 24
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "examples"
    / "out"
    / "osm_city_flood_daycycle"
    / "valencia_turia_flood.mp4"
)

MAP_SURFACE_COLORS = {
    city.COLORS["base"],
    city.COLORS["landuse"],
    city.COLORS["park"],
    city.COLORS["water"],
    city.COLORS["road"],
    city.COLORS["road_hi"],
}
ROAD_COLORS = {city.COLORS["road"], city.COLORS["road_hi"]}
GREEN_COLORS = {city.COLORS["park"]}

FLOOD_SOURCE_RGBA = (46, 170, 210, 224)
FLOOD_WATER_RGBA = (34, 130, 205, 190)
FLOOD_SOURCE_WIDTH_M = 24.0
FLOOD_SURFACE_ELEVATION_M = 1.24
FLOOD_START_LEVEL_M = 0.0
FLOOD_END_LEVEL_M = 2.75
FLOOD_CELL_SIZE_M = 13.5
FLOOD_TRANSITION_M = 0.055
FLOOD_OVERLAY_LABEL = "VALENCIA, SPAIN"
FLOOD_OVERLAY_SUBTITLE = "CONNECTED HAND FLOOD MODEL"

STATIC_LIGHT_DIR = city.normalize(np.array([-0.42, 0.78, -0.46], dtype=np.float32))
WALL_TINT_RGB = (72, 82, 94)
BACKGROUND_TOP_RGB = (122, 157, 183)
BACKGROUND_BOTTOM_RGB = (236, 231, 210)

VALENCIA_TURIA_WGS84 = (
    (-0.3929, 39.4802),
    (-0.3883, 39.4804),
    (-0.3834, 39.4795),
    (-0.3788, 39.4778),
    (-0.3742, 39.4760),
    (-0.3693, 39.4734),
    (-0.3651, 39.4707),
    (-0.3607, 39.4679),
)
OVERLAY_FONT_CANDIDATES = (
    "/Users/mpopovic3/Library/Fonts/Inconsolata.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSans.ttf",
)


@dataclass(frozen=True)
class PreparedSurface:
    surface: city.SurfaceLayer
    mask: Image.Image


@dataclass(frozen=True)
class FloodModel:
    source_geometry: Polygon | MultiPolygon
    cell_centers_xy: np.ndarray
    activation_level_m: np.ndarray
    cell_size_m: float
    start_level_m: float
    end_level_m: float


@dataclass(frozen=True)
class PreparedFloodModel:
    cell_polygons: tuple[tuple[tuple[float, float], ...], ...]
    activation_level_m: np.ndarray
    cell_size_m: float
    start_level_m: float
    end_level_m: float


@dataclass(frozen=True)
class PreparedTriangle:
    depth: float
    polygon: list[tuple[float, float]]
    world: np.ndarray
    normal: np.ndarray
    mean_y: float
    layer_min_y: float
    layer_height: float
    rgba: tuple[int, int, int, int]
    specular: float
    radial_t: float
    is_wall: bool


@dataclass(frozen=True)
class PreparedScene:
    width: int
    height: int
    render_width: int
    render_height: int
    supersample: int
    radius: float
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    view_dir: np.ndarray
    far_dir_screen: np.ndarray
    scale: float
    offset_x: float
    offset_y: float
    surfaces: list[PreparedSurface]
    flood: PreparedFloodModel
    triangles: list[PreparedTriangle]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a flood-only OSM city timelapse. The flood extent uses a "
            "connected HAND-style model: cells flood only when the rising water "
            "surface can reach them from the river corridor."
        )
    )
    parser.add_argument(
        "--lon", type=float, default=VALENCIA_LON, help="AOI center longitude."
    )
    parser.add_argument(
        "--lat", type=float, default=VALENCIA_LAT, help="AOI center latitude."
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_RADIUS_M, help="AOI radius in meters."
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=(1280, 720), metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Output duration.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override frame count. Default is duration-seconds * fps.",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS, help="Video frame rate."
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=1,
        help="Internal render scale before downsampling.",
    )
    parser.add_argument(
        "--flood-cell-size",
        type=float,
        default=FLOOD_CELL_SIZE_M,
        help="Flood model grid cell size in meters.",
    )
    parser.add_argument(
        "--flood-start-level",
        type=float,
        default=FLOOD_START_LEVEL_M,
        help="Initial water stage in meters. Default 0 means no flood at t=0.",
    )
    parser.add_argument(
        "--flood-end-level",
        type=float,
        default=FLOOD_END_LEVEL_M,
        help="End water level label in meters.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output MP4 path."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional directory for rendered PNG frames.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the PNG frame sequence after encoding.",
    )
    parser.add_argument(
        "--refresh-osm", action="store_true", help="Ignore cached Overpass responses."
    )
    return parser.parse_args()


def resolve_frame_count(args: argparse.Namespace) -> int:
    fps = max(1, int(args.fps))
    if args.frames is not None:
        return max(1, int(args.frames))
    return max(1, int(round(max(float(args.duration_seconds), 0.1) * fps)))


def load_overlay_font(size_px: int) -> ImageFont.ImageFont:
    for font_name in OVERLAY_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=max(1, int(size_px)))
        except OSError:
            continue
    return ImageFont.load_default()


def fit_overlay_font(
    text: str, *, target_size_px: int, max_width_px: int, min_size_px: int
) -> ImageFont.ImageFont:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    for size_px in range(max(1, int(target_size_px)), max(0, int(min_size_px)) - 1, -3):
        font = load_overlay_font(size_px)
        if hasattr(probe, "textbbox"):
            bbox = probe.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
        else:
            text_w = int(round(probe.textlength(text, font=font)))
        if text_w <= max_width_px:
            return font
    return load_overlay_font(min_size_px)


def scalar_smoothstep(edge0: float, edge1: float, value: float) -> float:
    return float(city.smoothstep(float(edge0), float(edge1), float(value)))


def flood_progress_for_frame(frame_index: int, total_frames: int) -> float:
    t = frame_index / max(total_frames - 1, 1)
    raw0 = 1.0 / (1.0 + math.exp(-7.2 * (0.0 - 0.46)))
    raw1 = 1.0 / (1.0 + math.exp(-7.2 * (1.0 - 0.46)))
    raw = 1.0 / (1.0 + math.exp(-7.2 * (t - 0.46)))
    return float(np.clip((raw - raw0) / max(raw1 - raw0, 1e-6), 0.0, 1.0))


def water_level_for_progress(flood: PreparedFloodModel, progress: float) -> float:
    return flood.start_level_m + (flood.end_level_m - flood.start_level_m) * float(
        progress
    )


def add_flood_overlay(
    image: Image.Image,
    *,
    water_level_m: float,
    flood_t: float,
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = canvas.width / 1280.0
    margin = max(10, int(round(14 * scale)))
    panel_w = max(210, int(round(canvas.width * 0.23)))
    panel_h = max(58, int(round(74 * scale)))
    x0 = margin
    y0 = margin
    x1 = min(canvas.width - margin, x0 + panel_w)
    y1 = y0 + panel_h
    radius = max(7, int(round(10 * scale)))
    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)],
        radius=radius,
        fill=(12, 25, 35, 186),
        outline=(190, 230, 244, 72),
        width=max(1, int(round(1.4 * scale))),
    )

    text_w = max(120, int(round((x1 - x0) - 24 * scale)))
    title_font = fit_overlay_font(
        FLOOD_OVERLAY_LABEL,
        target_size_px=max(12, int(round(17 * scale))),
        max_width_px=text_w,
        min_size_px=max(10, int(round(12 * scale))),
    )
    elapsed_hours = 24.0 * float(np.clip(flood_t, 0.0, 1.0))
    body_text = f"+{water_level_m:.2f} m   T+{elapsed_hours:04.1f}h"
    body_font = fit_overlay_font(
        body_text,
        target_size_px=max(11, int(round(16 * scale))),
        max_width_px=text_w,
        min_size_px=max(9, int(round(11 * scale))),
    )
    subtitle_font = fit_overlay_font(
        FLOOD_OVERLAY_SUBTITLE,
        target_size_px=max(8, int(round(10 * scale))),
        max_width_px=text_w,
        min_size_px=max(7, int(round(8 * scale))),
    )

    tx = x0 + max(10, int(round(12 * scale)))
    draw.text(
        (tx, y0 + max(7, int(round(8 * scale)))),
        FLOOD_OVERLAY_LABEL,
        font=title_font,
        fill=(238, 248, 250, 255),
    )
    draw.text(
        (tx, y0 + max(24, int(round(28 * scale)))),
        body_text,
        font=body_font,
        fill=(166, 225, 248, 255),
    )
    draw.text(
        (tx, y0 + max(39, int(round(47 * scale)))),
        FLOOD_OVERLAY_SUBTITLE,
        font=subtitle_font,
        fill=(170, 189, 198, 230),
    )

    bar_x0 = tx
    bar_x1 = x1 - max(10, int(round(12 * scale)))
    bar_y0 = y1 - max(9, int(round(11 * scale)))
    bar_y1 = bar_y0 + max(4, int(round(5 * scale)))
    fill_x1 = bar_x0 + (bar_x1 - bar_x0) * float(np.clip(flood_t, 0.0, 1.0))
    draw.rounded_rectangle(
        [(bar_x0, bar_y0), (bar_x1, bar_y1)], radius=3, fill=(68, 84, 95, 220)
    )
    draw.rounded_rectangle(
        [(bar_x0, bar_y0), (fill_x1, bar_y1)], radius=3, fill=(58, 184, 230, 245)
    )
    return canvas


def project_points_quiet(
    points: np.ndarray,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return city.project_points(
            points,
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )


def project_surface_parts_quiet(
    surface: city.SurfaceLayer,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return city.project_surface_parts(
            surface,
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )


def longest_linestring(geom) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return geom
    lines: list[LineString] = []
    if isinstance(geom, MultiLineString):
        lines.extend(
            part
            for part in geom.geoms
            if isinstance(part, LineString) and not part.is_empty
        )
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            candidate = longest_linestring(part)
            if candidate is not None:
                lines.append(candidate)
    if not lines:
        return None
    return max(lines, key=lambda item: float(item.length))


def polygonal_geom(geom) -> Polygon | MultiPolygon | None:
    if geom is None or geom.is_empty:
        return None
    polygonal = city._extract_polygonal(city._fix_geom(geom))
    if polygonal is None or polygonal.is_empty:
        return None
    return polygonal


def local_turia_line(lon: float, lat: float, radius: float) -> LineString:
    epsg = city.utm_epsg_for_lon_lat(lon, lat)
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    center_m = transform(to_metric.transform, Point(lon, lat))
    center_xy = tuple(center_m.coords[0])
    river_m = transform(to_metric.transform, LineString(VALENCIA_TURIA_WGS84))
    river_local = city.localize_feature({"geometry": river_m, "tags": {}}, center_xy)[
        "geometry"
    ]
    clip = Point(0.0, 0.0).buffer(float(radius) * 1.05, resolution=96)
    clipped = longest_linestring(
        city._extract_linear(city._fix_geom(river_local.intersection(clip)))
    )
    if clipped is not None and clipped.length >= max(180.0, float(radius) * 0.25):
        return clipped

    r = float(radius)
    return LineString(
        [
            (-r * 1.02, -r * 0.22),
            (-r * 0.60, -r * 0.08),
            (-r * 0.16, -r * 0.03),
            (r * 0.26, r * 0.06),
            (r * 0.70, r * 0.16),
            (r * 1.02, r * 0.28),
        ]
    )


def line_metrics(
    line: LineString, points_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(line.coords, dtype=np.float32)
    starts = coords[:-1]
    ends = coords[1:]
    seg = ends - starts
    seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-6)
    rel = points_xy[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(rel * seg[None, :, :], axis=2) / seg_len2[None, :], 0.0, 1.0)
    nearest = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    delta = points_xy[:, None, :] - nearest
    dist2 = np.sum(delta * delta, axis=2)
    seg_idx = np.argmin(dist2, axis=1)
    row = np.arange(points_xy.shape[0])
    nearest_best = nearest[row, seg_idx]
    distance = np.sqrt(dist2[row, seg_idx])
    seg_len = np.sqrt(seg_len2)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    station_m = cumulative[seg_idx] + t[row, seg_idx] * seg_len[seg_idx]
    station = station_m / max(float(cumulative[-1]), 1e-6)
    tangent = seg[seg_idx] / np.maximum(seg_len[seg_idx, None], 1e-6)
    side = np.sign(
        tangent[:, 0] * (points_xy[:, 1] - nearest_best[:, 1])
        - tangent[:, 1] * (points_xy[:, 0] - nearest_best[:, 0])
    )
    side[side == 0.0] = 1.0
    return (
        distance.astype(np.float32),
        station.astype(np.float32),
        side.astype(np.float32),
    )


def sample_line_frame(
    line: LineString, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    length = max(float(line.length), 1e-6)
    distance = float(np.clip(fraction, 0.0, 1.0)) * length
    delta = max(3.0, length * 0.012)
    p0 = line.interpolate(max(0.0, distance - delta))
    p1 = line.interpolate(min(length, distance + delta))
    tangent = np.array([float(p1.x - p0.x), float(p1.y - p0.y)], dtype=np.float32)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
    normal = np.array([-float(tangent[1]), float(tangent[0])], dtype=np.float32)
    point = line.interpolate(distance)
    return np.array([float(point.x), float(point.y)], dtype=np.float32), normal


def mask_points_in_geom(points_xy: np.ndarray, geom) -> np.ndarray:
    if geom is None or geom.is_empty or points_xy.size == 0:
        return np.zeros(points_xy.shape[0], dtype=bool)
    prepared = prep(geom)
    return np.array(
        [prepared.contains(Point(float(x), float(y))) for x, y in points_xy], dtype=bool
    )


def priority_flood_activation(
    *,
    index_grid: np.ndarray,
    channel_indices: np.ndarray,
    threshold_m: np.ndarray,
    friction_per_m: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    activation = np.full(threshold_m.shape, np.inf, dtype=np.float32)
    queue: list[tuple[float, int]] = []
    for idx in channel_indices.tolist():
        activation[idx] = 0.0
        heapq.heappush(queue, (0.0, int(idx)))

    rows, cols = np.nonzero(index_grid >= 0)
    cell_to_rc = {
        int(index_grid[row, col]): (int(row), int(col)) for row, col in zip(rows, cols)
    }
    neighbor_steps = (
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    )

    while queue:
        current, idx = heapq.heappop(queue)
        if current > float(activation[idx]) + 1e-6:
            continue
        row, col = cell_to_rc[idx]
        for dr, dc, step_scale in neighbor_steps:
            nr = row + dr
            nc = col + dc
            if (
                nr < 0
                or nc < 0
                or nr >= index_grid.shape[0]
                or nc >= index_grid.shape[1]
            ):
                continue
            nidx = int(index_grid[nr, nc])
            if nidx < 0:
                continue
            travel_loss = cell_size_m * step_scale * float(friction_per_m[nidx])
            candidate = max(current + travel_loss, float(threshold_m[nidx]))
            if candidate + 1e-6 < float(activation[nidx]):
                activation[nidx] = candidate
                heapq.heappush(queue, (candidate, nidx))
    return activation


def build_flood_model(
    lon: float,
    lat: float,
    radius: float,
    *,
    start_level_m: float,
    end_level_m: float,
    cell_size_m: float,
    road_geometry=None,
    green_geometry=None,
) -> FloodModel:
    radius = float(radius)
    cell = max(6.0, float(cell_size_m))
    line = local_turia_line(lon, lat, radius)
    clip = Point(0.0, 0.0).buffer(radius, resolution=96)
    source = polygonal_geom(
        line.buffer(
            FLOOD_SOURCE_WIDTH_M,
            cap_style=city.BufferCapStyle.round,
            join_style=city.BufferJoinStyle.round,
        ).intersection(clip)
    )
    if source is None:
        source = Point(0.0, 0.0).buffer(FLOOD_SOURCE_WIDTH_M, resolution=32)

    xs = np.arange(-radius + cell * 0.5, radius, cell, dtype=np.float32)
    ys = np.arange(-radius + cell * 0.5, radius, cell, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    inside = (xx * xx + yy * yy) <= radius * radius
    index_grid = np.full(xx.shape, -1, dtype=np.int32)
    centers = np.column_stack([xx[inside], yy[inside]]).astype(np.float32)
    index_grid[inside] = np.arange(centers.shape[0], dtype=np.int32)

    distance, station, side = line_metrics(line, centers)
    river_dist = np.maximum(distance - FLOOD_SOURCE_WIDTH_M, 0.0)

    hand = (
        -0.14
        + 0.00315 * river_dist
        + 0.00000155 * river_dist * river_dist
        + 0.22 * (1.0 - station)
        + 0.10 * side * np.sin((station * 2.35 + 0.10) * math.pi)
    ).astype(np.float32)

    basin_specs = (
        (0.22, -1.0, 94.0, 0.34, 180.0),
        (0.36, 1.0, 126.0, 0.48, 230.0),
        (0.55, -1.0, 112.0, 0.40, 205.0),
        (0.73, 1.0, 150.0, 0.56, 265.0),
        (0.86, -1.0, 128.0, 0.32, 180.0),
    )
    for fraction, basin_side, offset_m, depth_m, sigma_m in basin_specs:
        point_xy, normal_xy = sample_line_frame(line, fraction)
        basin_center = point_xy + normal_xy * float(basin_side) * float(offset_m)
        d2 = np.sum((centers - basin_center[None, :]) ** 2, axis=1)
        hand -= float(depth_m) * np.exp(
            -0.5 * d2 / max(float(sigma_m) ** 2, 1.0)
        ).astype(np.float32)

    micro = (
        0.045 * np.sin(centers[:, 0] * 0.014 + centers[:, 1] * 0.006)
        + 0.030 * np.sin(centers[:, 0] * -0.009 + centers[:, 1] * 0.018 + 1.7)
        + 0.020 * np.cos(centers[:, 0] * 0.023 - centers[:, 1] * 0.013)
    ).astype(np.float32)
    hand += micro

    road_mask = mask_points_in_geom(centers, road_geometry)
    green_mask = mask_points_in_geom(centers, green_geometry)
    hand += road_mask.astype(np.float32) * 0.075
    hand -= green_mask.astype(np.float32) * 0.055
    hand = np.maximum(hand, -0.18).astype(np.float32)

    channel_mask = distance <= FLOOD_SOURCE_WIDTH_M * 1.25
    if not np.any(channel_mask):
        channel_mask[np.argmin(distance)] = True
    hand[channel_mask] = np.minimum(hand[channel_mask], -0.10)

    friction = (
        0.00055
        + 0.00022 * np.clip(river_dist / max(radius, 1.0), 0.0, 1.0)
        + road_mask.astype(np.float32) * 0.00045
        - green_mask.astype(np.float32) * 0.00012
    ).astype(np.float32)
    friction = np.clip(friction, 0.00035, 0.00135)
    activation = priority_flood_activation(
        index_grid=index_grid,
        channel_indices=np.nonzero(channel_mask)[0].astype(np.int32),
        threshold_m=hand,
        friction_per_m=friction,
        cell_size_m=cell,
    )

    max_level = float(end_level_m) + 0.30
    keep = np.isfinite(activation) & (activation <= max_level)
    return FloodModel(
        source_geometry=source,
        cell_centers_xy=centers[keep],
        activation_level_m=activation[keep].astype(np.float32),
        cell_size_m=cell,
        start_level_m=float(start_level_m),
        end_level_m=float(end_level_m),
    )


def merged_surface_geometry(
    scene: city.SceneLayers, colors: set[tuple[int, int, int, int]]
):
    parts = [surface.geometry for surface in scene.surfaces if surface.rgba in colors]
    if not parts:
        return None
    return polygonal_geom(unary_union(parts))


def build_lightweight_scene(
    lon: float,
    lat: float,
    radius: float,
    *,
    refresh_osm: bool,
    flood_cell_size_m: float,
    flood_start_level_m: float,
    flood_end_level_m: float,
) -> tuple[city.SceneLayers, FloodModel]:
    scene = city.build_city_scene(lon, lat, radius, refresh_osm=refresh_osm)
    for mesh in scene.meshes:
        mesh.shadow_alpha = 0
        mesh.specular = 0.0

    flood = build_flood_model(
        lon,
        lat,
        radius,
        start_level_m=flood_start_level_m,
        end_level_m=flood_end_level_m,
        cell_size_m=flood_cell_size_m,
        road_geometry=merged_surface_geometry(scene, ROAD_COLORS),
        green_geometry=merged_surface_geometry(scene, GREEN_COLORS),
    )

    surfaces = [
        surface for surface in scene.surfaces if surface.rgba in MAP_SURFACE_COLORS
    ]
    surfaces.append(
        city.SurfaceLayer(
            geometry=flood.source_geometry,
            rgba=FLOOD_SOURCE_RGBA,
            elevation=FLOOD_SURFACE_ELEVATION_M - 0.08,
            specular=0.22,
            reflectivity=0.18,
        )
    )
    lightweight = city.SceneLayers(
        surfaces=surfaces,
        meshes=scene.meshes,
        roof_outlines=[],
        focus_landmarks=[],
        radius=scene.radius,
    )
    return lightweight, flood


def project_flood_cells(
    flood: FloodModel,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> PreparedFloodModel:
    centers = np.asarray(flood.cell_centers_xy, dtype=np.float32)
    if centers.size == 0:
        return PreparedFloodModel(
            (),
            np.zeros(0, dtype=np.float32),
            flood.cell_size_m,
            flood.start_level_m,
            flood.end_level_m,
        )

    half = float(flood.cell_size_m) * 0.5
    offsets = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32
    )
    corners_xy = centers[:, None, :] + offsets[None, :, :]
    world = np.column_stack(
        [
            corners_xy.reshape(-1, 2)[:, 0],
            np.full(
                corners_xy.shape[0] * 4, FLOOD_SURFACE_ELEVATION_M, dtype=np.float32
            ),
            -corners_xy.reshape(-1, 2)[:, 1],
        ]
    ).astype(np.float32)
    projected = project_points_quiet(
        world,
        eye=eye,
        target=target,
        up=up,
        width=width,
        height=height,
        fov_deg=fov_deg,
    ).reshape(-1, 4, 3)

    polygons: list[tuple[tuple[float, float], ...]] = []
    levels: list[float] = []
    for idx, cell_proj in enumerate(projected):
        if np.any(cell_proj[:, 2] <= 1.0):
            continue
        polygon = tuple(
            city.fit_polygon(
                cell_proj[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y
            )
        )
        polygons.append(polygon)
        levels.append(float(flood.activation_level_m[idx]))
    return PreparedFloodModel(
        cell_polygons=tuple(polygons),
        activation_level_m=np.asarray(levels, dtype=np.float32),
        cell_size_m=flood.cell_size_m,
        start_level_m=flood.start_level_m,
        end_level_m=flood.end_level_m,
    )


def prepare_scene(
    scene: city.SceneLayers,
    flood: FloodModel,
    *,
    width: int,
    height: int,
    supersample: int,
) -> PreparedScene:
    supersample = max(1, int(supersample))
    render_width = width * supersample
    render_height = height * supersample
    radius = float(scene.radius)
    eye = np.array([-radius * 1.74, radius * 1.18, -radius * 1.46], dtype=np.float32)
    target = np.array([0.0, radius * 0.06, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_deg = 33.0
    view_dir = city.normalize(eye - target)

    surface_batches: list[
        tuple[city.SurfaceLayer, list[tuple[np.ndarray, list[np.ndarray]]]]
    ] = []
    fit_inputs: list[np.ndarray] = []
    for surface in scene.surfaces:
        parts = project_surface_parts_quiet(
            surface,
            eye=eye,
            target=target,
            up=up,
            width=render_width,
            height=render_height,
            fov_deg=fov_deg,
        )
        if parts:
            surface_batches.append((surface, parts))
            fit_inputs.extend(exterior[:, :2] for exterior, _ in parts)

    mesh_projections: list[tuple[city.MeshLayer, np.ndarray]] = []
    for layer in scene.meshes:
        projected = project_points_quiet(
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

    scale, offset_x, offset_y = city.compute_fit_transform(
        fit_inputs,
        width=render_width,
        height=render_height,
        margin_ratio=0.052,
    )

    far_probe = project_points_quiet(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, radius * 0.30]], dtype=np.float32),
        eye=eye,
        target=target,
        up=up,
        width=render_width,
        height=render_height,
        fov_deg=fov_deg,
    )
    far_dir_screen = city.normalize(far_probe[1, :2] - far_probe[0, :2])
    if float(np.linalg.norm(far_dir_screen)) <= 1e-6:
        far_dir_screen = np.array([0.0, -1.0], dtype=np.float32)

    prepared_surfaces = [
        PreparedSurface(
            surface=surface,
            mask=city.build_surface_mask(
                parts,
                width=render_width,
                height=render_height,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
        )
        for surface, parts in surface_batches
    ]

    prepared_flood = project_flood_cells(
        flood,
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

    mesh_points = []
    for _, projected in mesh_projections:
        pts = np.asarray(projected[:, :2], dtype=np.float32).copy()
        pts[:, 0] = pts[:, 0] * scale + offset_x
        pts[:, 1] = pts[:, 1] * scale + offset_y
        mesh_points.append(pts)
    packed = (
        np.vstack(mesh_points)
        if mesh_points
        else np.array([[render_width * 0.5, render_height * 0.5]], dtype=np.float32)
    )
    center_x = float(0.5 * (packed[:, 0].min() + packed[:, 0].max()))
    center_y = float(0.5 * (packed[:, 1].min() + packed[:, 1].max()))
    radius_x = max(1.0, float(0.5 * (packed[:, 0].max() - packed[:, 0].min())))
    radius_y = max(1.0, float(0.5 * (packed[:, 1].max() - packed[:, 1].min())))

    prepared_triangles: list[PreparedTriangle] = []
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
            normal_length = float(np.linalg.norm(raw_normal))
            if normal_length <= 1e-6:
                continue
            raw_normal = raw_normal / normal_length
            mean_y = float(world[:, 1].mean())
            if mean_y <= layer_min_y + 0.05 and abs(float(raw_normal[1])) > 0.92:
                continue
            polygon = city.fit_polygon(
                screen[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y
            )
            centroid_x = sum(point[0] for point in polygon) / 3.0
            centroid_y = sum(point[1] for point in polygon) / 3.0
            radial_t = math.sqrt(
                ((centroid_x - center_x) / radius_x) ** 2
                + ((centroid_y - center_y) / radius_y) ** 2
            )
            prepared_triangles.append(
                PreparedTriangle(
                    depth=float(screen[:, 2].mean()),
                    polygon=polygon,
                    world=world.copy(),
                    normal=raw_normal.astype(np.float32),
                    mean_y=mean_y,
                    layer_min_y=layer_min_y,
                    layer_height=layer_height,
                    rgba=layer.rgba,
                    specular=layer.specular,
                    radial_t=float(np.clip(radial_t, 0.0, 1.0)),
                    is_wall=abs(float(raw_normal[1])) < 0.55,
                )
            )
    prepared_triangles.sort(key=lambda item: item.depth, reverse=True)

    return PreparedScene(
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
        supersample=supersample,
        radius=radius,
        eye=eye,
        target=target,
        up=up,
        fov_deg=fov_deg,
        view_dir=view_dir,
        far_dir_screen=far_dir_screen,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        surfaces=prepared_surfaces,
        flood=prepared_flood,
        triangles=prepared_triangles,
    )


def make_background(width: int, height: int) -> Image.Image:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top = np.asarray(BACKGROUND_TOP_RGB, dtype=np.float32).reshape(1, 1, 3)
    bottom = np.asarray(BACKGROUND_BOTTOM_RGB, dtype=np.float32).reshape(1, 1, 3)
    rgb = top * (1.0 - y) + bottom * y
    rgb = np.repeat(rgb, width, axis=1)
    haze = city.smoothstep(0.58, 1.0, y[:, :, 0]).astype(np.float32)[:, :, None]
    rgb = rgb + haze * np.asarray([8.0, 8.0, 10.0], dtype=np.float32).reshape(1, 1, 3)
    rgba = np.concatenate(
        [
            np.clip(rgb, 0.0, 255.0).astype(np.uint8),
            np.full((height, width, 1), 255, dtype=np.uint8),
        ],
        axis=2,
    )
    return Image.fromarray(rgba)


def shade_triangle(
    triangle: PreparedTriangle, prepared: PreparedScene
) -> tuple[int, int, int, int]:
    rgba = city.shade_rgba(
        triangle.rgba,
        normal=triangle.normal,
        light_dir=STATIC_LIGHT_DIR,
        view_dir=prepared.eye - triangle.world.mean(axis=0),
        specular=triangle.specular,
    )
    if triangle.is_wall:
        wall_t = float(
            city.smoothstep(
                0.0,
                1.0,
                np.clip(
                    (triangle.mean_y - triangle.layer_min_y) / triangle.layer_height,
                    0.0,
                    1.0,
                ),
            )
        )
        wall_gain = 0.88 + 0.20 * wall_t
        wall_mix = (1.0 - wall_t) * 0.08
        base_rgb = np.asarray(rgba[:3], dtype=np.float32) * wall_gain
        wall_rgb = np.clip(
            base_rgb * (1.0 - wall_mix)
            + np.asarray(WALL_TINT_RGB, dtype=np.float32) * wall_mix,
            0.0,
            255.0,
        ).astype(np.uint8)
        rgba = (int(wall_rgb[0]), int(wall_rgb[1]), int(wall_rgb[2]), rgba[3])
    return city.apply_depth_grade_rgba(rgba, radial_t=triangle.radial_t)


def render_flood_mask(
    prepared: PreparedScene, *, water_level_m: float, flood_t: float
) -> Image.Image:
    mask = Image.new("L", (prepared.render_width, prepared.render_height), 0)
    if flood_t <= 0.0:
        return mask
    draw = ImageDraw.Draw(mask)
    levels = prepared.flood.activation_level_m
    if levels.size == 0:
        return mask
    transition = max(
        FLOOD_TRANSITION_M,
        (prepared.flood.end_level_m - prepared.flood.start_level_m) / 100.0,
    )
    weights = np.clip((float(water_level_m) - levels) / transition, 0.0, 1.0)
    active = np.nonzero(weights > 0.0)[0]
    for idx in active.tolist():
        fill = int(round(255.0 * scalar_smoothstep(0.0, 1.0, float(weights[idx]))))
        if fill <= 0:
            continue
        draw.polygon(prepared.flood.cell_polygons[idx], fill=fill)
    if mask.getbbox() is None:
        return mask
    return mask.filter(ImageFilter.MaxFilter(size=3)).filter(
        ImageFilter.GaussianBlur(radius=1.05)
    )


def render_flood_surface_layer(
    prepared: PreparedScene,
    *,
    water_level_m: float,
    flood_t: float,
) -> Image.Image:
    mask = render_flood_mask(prepared, water_level_m=water_level_m, flood_t=flood_t)
    if mask.getbbox() is None:
        return Image.new(
            "RGBA", (prepared.render_width, prepared.render_height), (0, 0, 0, 0)
        )
    shaded = city.shade_rgba(
        FLOOD_WATER_RGBA,
        normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        light_dir=STATIC_LIGHT_DIR,
        view_dir=prepared.view_dir,
        specular=0.32,
    )
    alpha = int(round(140.0 + 54.0 * scalar_smoothstep(0.0, 1.0, flood_t)))
    layer = city.render_water_surface_layer(
        width=prepared.render_width,
        height=prepared.render_height,
        mask=mask,
        shaded_rgb=shaded[:3],
        alpha_value=alpha,
        light_dir=STATIC_LIGHT_DIR,
        specular_intensity=0.32,
        far_dir_screen=prepared.far_dir_screen,
    )

    shoreline = ImageChops.subtract(
        mask.filter(ImageFilter.MaxFilter(size=9)),
        mask.filter(ImageFilter.MinFilter(size=9)),
    ).filter(ImageFilter.GaussianBlur(radius=1.35))
    shoreline_layer = Image.new(
        "RGBA", (prepared.render_width, prepared.render_height), (180, 235, 255, 0)
    )
    shoreline_layer.putalpha(shoreline.point(lambda value: int(round(value * 0.28))))
    return Image.alpha_composite(layer, shoreline_layer)


def render_frame(
    prepared: PreparedScene,
    *,
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    progress = flood_progress_for_frame(frame_index, total_frames)
    water_level = water_level_for_progress(prepared.flood, progress)
    image = make_background(prepared.render_width, prepared.render_height)

    for item in prepared.surfaces:
        shaded = city.shade_rgba(
            item.surface.rgba,
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            light_dir=STATIC_LIGHT_DIR,
            view_dir=prepared.view_dir,
            specular=item.surface.specular,
        )
        if item.surface.reflectivity > 0.0:
            layer_image = city.render_water_surface_layer(
                width=prepared.render_width,
                height=prepared.render_height,
                mask=item.mask,
                shaded_rgb=shaded[:3],
                alpha_value=item.surface.rgba[3],
                light_dir=STATIC_LIGHT_DIR,
                specular_intensity=max(0.15, item.surface.specular),
                far_dir_screen=prepared.far_dir_screen,
            )
        else:
            alpha = item.mask.point(
                lambda value, a=item.surface.rgba[3]: (value * a + 127) // 255
            )
            layer_image = Image.new(
                "RGBA",
                (prepared.render_width, prepared.render_height),
                shaded[:3] + (0,),
            )
            layer_image.putalpha(alpha)
        image = Image.alpha_composite(image, layer_image)

    image = Image.alpha_composite(
        image,
        render_flood_surface_layer(
            prepared, water_level_m=water_level, flood_t=progress
        ),
    )

    draw = ImageDraw.Draw(image, "RGBA")
    for triangle in prepared.triangles:
        draw.polygon(triangle.polygon, fill=shade_triangle(triangle, prepared))

    if prepared.supersample > 1:
        resampling = (
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        )
        image = image.resize((prepared.width, prepared.height), resample=resampling)
    return add_flood_overlay(image, water_level_m=water_level, flood_t=progress)


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


def main() -> int:
    args = parse_args()
    width, height = map(int, args.size)
    fps = max(1, int(args.fps))
    frame_count = resolve_frame_count(args)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Flood] center=({args.lon:.5f}, {args.lat:.5f}) radius={args.radius:.0f}m")
    print(
        f"[Flood] duration={frame_count / fps:.2f}s frames={frame_count} fps={fps} "
        f"levels={float(args.flood_start_level):.2f}m->{float(args.flood_end_level):.2f}m"
    )
    scene, flood = build_lightweight_scene(
        args.lon,
        args.lat,
        args.radius,
        refresh_osm=bool(args.refresh_osm),
        flood_cell_size_m=float(args.flood_cell_size),
        flood_start_level_m=float(args.flood_start_level),
        flood_end_level_m=float(args.flood_end_level),
    )
    if not scene.meshes:
        raise SystemExit("No buildings were generated for the requested AOI.")
    print(
        f"[Flood] flood cells retained={len(flood.activation_level_m)} cell={flood.cell_size_m:.1f}m"
    )
    prepared = prepare_scene(
        scene, flood, width=width, height=height, supersample=int(args.supersample)
    )

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is not None:
        frames_dir = args.frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_frames:
        frames_dir = output_path.with_suffix("")
        frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(
            prefix="osm_city_flood_", dir=str(output_path.parent)
        )
        frames_dir = Path(temp_dir.name)

    try:
        log_interval = max(1, fps)
        for frame_index in range(frame_count):
            progress = flood_progress_for_frame(frame_index, frame_count)
            water_level = water_level_for_progress(prepared.flood, progress)
            frame = render_frame(
                prepared, frame_index=frame_index, total_frames=frame_count
            )
            frame.save(
                frames_dir / f"frame_{frame_index:05d}.png",
                format="PNG",
                compress_level=0,
            )
            if (
                frame_index == 0
                or (frame_index + 1) % log_interval == 0
                or frame_index + 1 == frame_count
            ):
                print(
                    f"[Flood] frame {frame_index + 1}/{frame_count} "
                    f"| level={water_level:.2f}m extent={progress * 100.0:.1f}%"
                )
        encode_video(frames_dir, output_path, fps=fps)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"[Flood] Wrote {city.display_path(output_path)}")
    if args.keep_frames or args.frames_dir is not None:
        print(f"[Flood] Frames saved in {city.display_path(frames_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
