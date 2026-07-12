#!/usr/bin/env python3
from __future__ import annotations

"""Build a RiverREM-style Snake River REM in Python and render it in 3D.

This example follows the same overall workflow as OpenTopography's RiverREM:
1. Download a DEM for the river reach.
2. Query OpenStreetMap for river centerlines.
3. Sample river elevations at DEM pixels.
4. Estimate ``k`` from sinuosity and interpolate a water-surface elevation.
5. Subtract that water surface from the DEM to produce a raw REM.
6. Drape the REM as a Forge3D terrain overlay on the extruded DEM.

Requirements:
    pip install forge3d rasterio shapely pillow requests

Optional acceleration:
    pip install scipy
"""

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform as warp_transform, transform_bounds
from rasterio.windows import Window
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.terrain_scatter import TerrainScatterSource
from forge3d.north_arrow import NorthArrow, NorthArrowConfig
from forge3d.scale_bar import ScaleBar, ScaleBarConfig

try:
    from scipy.spatial import cKDTree as _KDTree  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional acceleration only
    _KDTree = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_NAME = "Snake River"
SITE_REGION = "Grand Teton National Park, Wyoming"
SITE_SLUG = "snake_grand_teton"

DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "snake_river_rem_forge3d"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "examples" / "out" / "snake_river_rem_forge3d" / "snake-river-rem-grand-teton-3d.png"
)

THREEDEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERPASS_RETRIES = 2
RENDER_CRS = "EPSG:3857"
USER_AGENT = "forge3d-snake-rem-example/1.0"
HDR_FILENAME = "forge3d_rem_neutral.hdr"
VIEWER_SETTLE_SECONDS = 2.0
WATERWAY_VALUES = ("river", "stream", "tidal_channel")

# Snake River between Deadmans Bar and Schwabacher Landing in Grand Teton NP.
CENTER_WGS84 = (-110.6450, 43.7360)
DEFAULT_WIDTH_M = 16384.0
DEFAULT_HEIGHT_M = 8192.0
DEFAULT_ZSCALE = 1.45
DEFAULT_CAMERA_PHI_DEG = 96.0
DEFAULT_CAMERA_THETA_DEG = 34.0
DEFAULT_CAMERA_RADIUS_SCALE = 1.58
DEFAULT_CAMERA_FOV_DEG = 19.0
DEFAULT_VIEWER_SIZE = (1700, 1100)
DEFAULT_RENDER_SIZE = (2400, 1400)
MAP_TITLE = "Snake River Relative Elevation Model"

DEFAULT_PALETTE = "mauve-terrace"
REM_PALETTES = {
    "mauve-terrace": np.array(
        [
            [0.00, 0.22, 0.17, 0.27],
            [0.18, 0.36, 0.28, 0.39],
            [0.38, 0.55, 0.38, 0.48],
            [0.58, 0.75, 0.50, 0.52],
            [0.76, 0.88, 0.66, 0.58],
            [0.91, 0.95, 0.81, 0.71],
            [1.00, 0.99, 0.92, 0.82],
        ],
        dtype=np.float32,
    ),
    "copper-silt": np.array(
        [
            [0.00, 0.16, 0.12, 0.18],
            [0.18, 0.32, 0.22, 0.24],
            [0.38, 0.52, 0.33, 0.24],
            [0.58, 0.72, 0.48, 0.28],
            [0.78, 0.86, 0.67, 0.43],
            [0.92, 0.93, 0.82, 0.63],
            [1.00, 0.98, 0.93, 0.82],
        ],
        dtype=np.float32,
    ),
    "ember-plum": np.array(
        [
            [0.00, 0.13, 0.09, 0.20],
            [0.18, 0.27, 0.16, 0.34],
            [0.38, 0.48, 0.22, 0.42],
            [0.58, 0.72, 0.34, 0.40],
            [0.78, 0.88, 0.56, 0.38],
            [0.92, 0.95, 0.77, 0.60],
            [1.00, 0.99, 0.92, 0.84],
        ],
        dtype=np.float32,
    ),
    "inferno-lite": np.array(
        [
            [0.00, 0.06, 0.04, 0.16],
            [0.18, 0.25, 0.07, 0.29],
            [0.38, 0.50, 0.12, 0.33],
            [0.58, 0.77, 0.25, 0.26],
            [0.78, 0.93, 0.53, 0.20],
            [0.92, 0.98, 0.79, 0.45],
            [1.00, 0.99, 0.95, 0.80],
        ],
        dtype=np.float32,
    ),
    "viridis": np.array(
        [
            [0.00, 0.267004, 0.004874, 0.329415],
            [0.18, 0.262290, 0.241820, 0.520629],
            [0.38, 0.171330, 0.452156, 0.557957],
            [0.58, 0.124646, 0.640096, 0.527238],
            [0.78, 0.430076, 0.808203, 0.347019],
            [0.92, 0.789582, 0.880121, 0.122165],
            [1.00, 0.993248, 0.906157, 0.143936],
        ],
        dtype=np.float32,
    ),
    "cividis": np.array(
        [
            [0.00, 0.000000, 0.135112, 0.304751],
            [0.18, 0.184958, 0.258641, 0.426839],
            [0.38, 0.383924, 0.397652, 0.436393],
            [0.58, 0.562593, 0.543444, 0.470522],
            [0.78, 0.764810, 0.703373, 0.410792],
            [0.92, 0.917060, 0.827710, 0.301715],
            [1.00, 0.995737, 0.909344, 0.217772],
        ],
        dtype=np.float32,
    ),
    "lajolla": np.array(
        [
            [0.00, 0.098791, 0.099669, 0.000088],
            [0.18, 0.288178, 0.164871, 0.102818],
            [0.38, 0.659685, 0.275771, 0.268985],
            [0.58, 0.886082, 0.477119, 0.313284],
            [0.78, 0.935304, 0.714423, 0.334318],
            [0.92, 0.983141, 0.914665, 0.572770],
            [1.00, 1.000000, 0.997796, 0.794247],
        ],
        dtype=np.float32,
    ),
    "lipari": np.array(
        [
            [0.00, 0.011370, 0.073240, 0.148284],
            [0.18, 0.193238, 0.310137, 0.455071],
            [0.38, 0.474859, 0.374113, 0.447049],
            [0.58, 0.776865, 0.401461, 0.374354],
            [0.78, 0.905538, 0.640548, 0.479357],
            [0.92, 0.929104, 0.832112, 0.682056],
            [1.00, 0.992307, 0.959017, 0.856609],
        ],
        dtype=np.float32,
    ),
    "acton": np.array(
        [
            [0.00, 0.180627, 0.129916, 0.300244],
            [0.18, 0.376151, 0.294514, 0.466264],
            [0.38, 0.627937, 0.397561, 0.575703],
            [0.58, 0.829078, 0.507293, 0.664989],
            [0.78, 0.832364, 0.680423, 0.789034],
            [0.92, 0.869879, 0.817464, 0.883652],
            [1.00, 0.900472, 0.900123, 0.940051],
        ],
        dtype=np.float32,
    ),
    "roma": np.array(
        [
            [0.00, 0.496845, 0.099626, 0.000000],
            [0.18, 0.683697, 0.486941, 0.154485],
            [0.38, 0.885300, 0.879944, 0.555133],
            [0.58, 0.597190, 0.873345, 0.843469],
            [0.78, 0.257950, 0.543770, 0.748424],
            [0.92, 0.165973, 0.321184, 0.652405],
            [1.00, 0.103699, 0.200063, 0.599992],
        ],
        dtype=np.float32,
    ),
}


@dataclass(frozen=True)
class RiverRemDiagnostics:
    river_name: str
    river_length_m: float
    sample_point_count: int
    river_pixel_count: int
    sinuosity: float
    neighbor_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a RiverREM-style Snake River relative elevation model in "
            "Python, then drape it over a Forge3D terrain render."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--center-lon", type=float, default=CENTER_WGS84[0], help="AOI center longitude.")
    parser.add_argument("--center-lat", type=float, default=CENTER_WGS84[1], help="AOI center latitude.")
    parser.add_argument("--width-m", type=float, default=DEFAULT_WIDTH_M, help="AOI width in meters.")
    parser.add_argument("--height-m", type=float, default=DEFAULT_HEIGHT_M, help="AOI height in meters.")
    parser.add_argument(
        "--side-length-m",
        type=float,
        default=None,
        help="Legacy square AOI override in meters. When set, it overrides --width-m and --height-m.",
    )
    parser.add_argument(
        "--river-name",
        type=str,
        default=SITE_NAME,
        help='Exact OSM river name to use. Pass "" to auto-select the longest named waterway.',
    )
    parser.add_argument(
        "--dem-resolution-m",
        type=float,
        default=4.0,
        help="Target DEM resolution in meters per pixel for the tiled USGS mosaic.",
    )
    parser.add_argument(
        "--dem-size",
        type=int,
        default=None,
        help="Optional explicit square DEM size in pixels. Overrides --dem-resolution-m.",
    )
    parser.add_argument(
        "--tile-size-px",
        type=int,
        default=2048,
        help="Maximum USGS export tile edge before mosaicking into a larger DEM.",
    )
    parser.add_argument(
        "--interp-pts",
        type=int,
        default=2200,
        help="Maximum number of sampled centerline points, matching RiverREM's workflow.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Nearest-neighbor count for IDW water-surface interpolation. Auto-estimated when omitted.",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=1.0,
        help="IDW exponent for the river water-surface interpolation. RiverREM uses 1.0.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="Approximate KD-tree epsilon used when SciPy acceleration is available.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="KD-tree worker count for SciPy acceleration. Ignored by the NumPy fallback.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=8000,
        help="How many DEM cells to interpolate per chunk when building the river water surface.",
    )
    parser.add_argument(
        "--zscale",
        type=float,
        default=DEFAULT_ZSCALE,
        help="Vertical exaggeration used by Forge3D for the final terrain render.",
    )
    parser.add_argument(
        "--viewer-size",
        type=int,
        nargs=2,
        default=DEFAULT_VIEWER_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Interactive viewer window size used for the live 3D scene.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=DEFAULT_RENDER_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Final snapshot size written to disk.",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=DEFAULT_PALETTE,
        choices=tuple(REM_PALETTES.keys()),
        help="REM palette preset used for the draped overlay.",
    )
    parser.add_argument(
        "--camera-phi-deg",
        type=float,
        default=DEFAULT_CAMERA_PHI_DEG,
        help="Azimuth of the terrain orbit camera in degrees.",
    )
    parser.add_argument(
        "--camera-theta-deg",
        type=float,
        default=DEFAULT_CAMERA_THETA_DEG,
        help="Elevation angle of the terrain orbit camera in degrees.",
    )
    parser.add_argument(
        "--camera-radius-scale",
        type=float,
        default=DEFAULT_CAMERA_RADIUS_SCALE,
        help="Orbit radius as a multiple of the terrain contract width.",
    )
    parser.add_argument(
        "--camera-fov-deg",
        type=float,
        default=DEFAULT_CAMERA_FOV_DEG,
        help="Field of view used for the terrain orbit camera.",
    )
    parser.add_argument(
        "--rover-gltf",
        type=Path,
        default=None,
        help="Optional rover/model GLTF or GLB to load on top of the terrain.",
    )
    parser.add_argument(
        "--rover-scale",
        type=float,
        default=1.0,
        help="Uniform scale applied to the optional rover/model.",
    )
    parser.add_argument(
        "--rover-yaw-deg",
        type=float,
        default=18.0,
        help="Yaw rotation in degrees applied to the optional rover/model.",
    )
    parser.add_argument(
        "--rover-height-offset",
        type=float,
        default=0.0,
        help="Vertical offset added after snapping the rover/model to the terrain.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Stop after writing DEM, REM, and overlay.")
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild cached artifacts.")
    return parser.parse_args()


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    remainder = int(value) % int(multiple)
    if remainder == 0:
        return int(value)
    return int(value) + int(multiple) - remainder


def _project_bounds_3857_to_wgs84(
    bounds_3857: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return tuple(
        float(value)
        for value in transform_bounds(RENDER_CRS, "EPSG:4326", *bounds_3857, densify_pts=21)
    )


def _bounds_from_center_wgs84(
    center_wgs84: tuple[float, float],
    *,
    width_m: float,
    height_m: float,
) -> tuple[float, float, float, float]:
    cx, cy = warp_transform("EPSG:4326", RENDER_CRS, [center_wgs84[0]], [center_wgs84[1]])
    half_w = 0.5 * float(width_m)
    half_h = 0.5 * float(height_m)
    return (float(cx[0] - half_w), float(cy[0] - half_h), float(cx[0] + half_w), float(cy[0] + half_h))


def _write_geotiff(
    path: Path,
    array: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=RENDER_CRS,
        transform=from_bounds(*bounds, width, height),
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(np.asarray(array, dtype=np.float32), 1)
    return path


def _ensure_hdr(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 8, 4
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode("ascii"))
        for y in range(height):
            for x in range(width):
                r = int(144 + 14 * x)
                g = int(146 + 10 * y)
                b = int(150 + 8 * x)
                handle.write(bytes([min(r, 255), min(g, 255), min(b, 255), 128]))
    return path


def _download_three_dep_dem(
    bounds_3857: tuple[float, float, float, float],
    *,
    dem_resolution_m: float,
    dem_size: int | None,
    tile_size_px: int,
    align_to: int,
    cache_dir: Path,
    force: bool,
) -> tuple[Path, np.ndarray]:
    span_x = float(bounds_3857[2] - bounds_3857[0])
    span_y = float(bounds_3857[3] - bounds_3857[1])
    if dem_size is None:
        width = max(1, int(math.ceil(span_x / max(float(dem_resolution_m), 0.1))))
        height = max(1, int(math.ceil(span_y / max(float(dem_resolution_m), 0.1))))
        width = _align_up(width, int(align_to))
        height = _align_up(height, int(align_to))
    else:
        width = int(dem_size)
        height = int(dem_size)

    tile_edge = max(256, int(tile_size_px))
    res_label = f"{float(dem_resolution_m):.1f}".replace(".", "p")
    dem_path = cache_dir / f"{SITE_SLUG}_3dep_{res_label}m_{width}x{height}_3857.tif"
    if not dem_path.exists() or force:
        cache_dir.mkdir(parents=True, exist_ok=True)
        left, bottom, right, top = bounds_3857
        transform = from_bounds(*bounds_3857, width, height)
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=RENDER_CRS,
            transform=transform,
            nodata=np.nan,
            compress="lzw",
            tiled=True,
        ) as dst:
            for row0 in range(0, height, tile_edge):
                tile_h = min(tile_edge, height - row0)
                row1 = row0 + tile_h
                tile_top = top - (row0 / height) * (top - bottom)
                tile_bottom = top - (row1 / height) * (top - bottom)
                for col0 in range(0, width, tile_edge):
                    tile_w = min(tile_edge, width - col0)
                    col1 = col0 + tile_w
                    tile_left = left + (col0 / width) * (right - left)
                    tile_right = left + (col1 / width) * (right - left)
                    params = {
                        "bbox": f"{tile_left:.3f},{tile_bottom:.3f},{tile_right:.3f},{tile_top:.3f}",
                        "bboxSR": "3857",
                        "imageSR": "3857",
                        "size": f"{tile_w},{tile_h}",
                        "format": "tiff",
                        "pixelType": "F32",
                        "f": "image",
                    }
                    req = Request(
                        THREEDEP_EXPORT_URL + "?" + urlencode(params),
                        headers={"User-Agent": USER_AGENT},
                    )
                    with urlopen(req, timeout=180) as response:
                        data = response.read()
                    with MemoryFile(data) as memfile, memfile.open() as dataset:
                        array = dataset.read(1).astype(np.float32)
                        nodata = dataset.nodata
                    if nodata is not None:
                        mask = np.isfinite(array) & (~np.isclose(array, nodata))
                    else:
                        mask = np.isfinite(array)
                    fill = float(np.nanpercentile(array[mask], 1.0)) if np.any(mask) else 0.0
                    array = np.where(mask, array, fill).astype(np.float32)
                    dst.write(array, 1, window=Window(col0, row0, tile_w, tile_h))

    with rasterio.open(dem_path) as dataset:
        return dem_path, dataset.read(1).astype(np.float32)


def _iter_line_parts(geometry: object) -> list[LineString]:
    if geometry is None:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if len(geometry.coords) >= 2 else []
    if isinstance(geometry, MultiLineString):
        out: list[LineString] = []
        for part in geometry.geoms:
            out.extend(_iter_line_parts(part))
        return out
    if isinstance(geometry, GeometryCollection):
        out = []
        for part in geometry.geoms:
            out.extend(_iter_line_parts(part))
        return out
    return []


def _query_overpass_json(query: str) -> dict[str, object]:
    payload = urlencode({"data": query}).encode("utf-8")
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        for attempt in range(OVERPASS_RETRIES):
            req = Request(url, data=payload, headers={"User-Agent": USER_AGENT})
            try:
                with urlopen(req, timeout=180) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < OVERPASS_RETRIES:
                    time.sleep(1.5 * float(attempt + 1))
                continue
    raise RuntimeError(f"Failed to query Overpass for OSM centerlines: {last_error}")


def _query_osm_centerlines_3857(
    bounds_wgs84: tuple[float, float, float, float],
    bounds_3857: tuple[float, float, float, float],
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    river_name: str | None,
) -> tuple[str, list[LineString], float]:
    west, south, east, north = bounds_wgs84
    waterway_regex = "|".join(WATERWAY_VALUES)
    if river_name and str(river_name).strip():
        river_regex = re.escape(str(river_name).strip())
        name_filter = f'[\"name\"~\"^{river_regex}$\",i]'
    else:
        name_filter = '[\"name\"]'
    query = f"""
[out:json][timeout:180];
(
  way["waterway"~"^({waterway_regex})$"]{name_filter}({south},{west},{north},{east});
);
out tags geom;
"""
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir / f"{SITE_SLUG}_overpass_{cache_key}.json"

    if cache_path is not None and cache_path.exists() and not force:
        result = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        result = _query_overpass_json(query)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    elements = result.get("elements", [])
    if not isinstance(elements, list):
        raise RuntimeError("Unexpected Overpass response while loading OSM centerlines")

    bounds_poly = box(*bounds_3857)
    grouped: dict[str, list[LineString]] = {}
    lengths: dict[str, float] = {}
    for element in elements:
        if not isinstance(element, dict):
            continue
        tags = element.get("tags", {})
        geometry = element.get("geometry", [])
        if not isinstance(tags, dict) or not isinstance(geometry, list):
            continue
        name = str(tags.get("name", "")).strip()
        if not name or len(geometry) < 2:
            continue
        lon = [float(point["lon"]) for point in geometry if isinstance(point, dict)]
        lat = [float(point["lat"]) for point in geometry if isinstance(point, dict)]
        if len(lon) < 2 or len(lat) < 2 or len(lon) != len(lat):
            continue
        xs, ys = warp_transform("EPSG:4326", RENDER_CRS, lon, lat)
        clipped = LineString(zip(xs, ys)).intersection(bounds_poly)
        for part in _iter_line_parts(clipped):
            grouped.setdefault(name, []).append(part)
            lengths[name] = lengths.get(name, 0.0) + float(part.length)

    if not lengths:
        raise RuntimeError(f"No named OSM river centerlines were returned for the {SITE_NAME} area")

    selected_name: str
    if river_name and str(river_name).strip():
        normalized = str(river_name).strip().casefold()
        matches = [name for name in lengths if name.casefold() == normalized]
        if not matches:
            available = ", ".join(sorted(lengths))
            raise RuntimeError(
                f'OSM returned no river named "{river_name}" inside the DEM extent. '
                f"Available names: {available}"
            )
        selected_name = max(matches, key=lambda value: lengths[value])
    else:
        selected_name = max(lengths, key=lengths.get)

    return selected_name, grouped[selected_name], float(lengths[selected_name])


def _sample_centerline_points(
    centerlines: list[LineString],
    *,
    interp_pts: int,
) -> tuple[list[Point], np.ndarray]:
    total_length = float(sum(max(float(line.length), 0.0) for line in centerlines))
    if total_length <= 0.0:
        raise RuntimeError("Selected centerline geometry has zero length")

    sample_points: list[Point] = []
    endpoints: list[tuple[float, float]] = []
    for line in centerlines:
        line_length = float(line.length)
        if line_length <= 0.0:
            continue
        share = line_length / total_length
        point_count = max(2, int(round(share * max(int(interp_pts), 2))))
        for distance in np.linspace(0.0, line_length, point_count, dtype=np.float64):
            sample_points.append(line.interpolate(float(distance)))
        start_xy = tuple(map(float, line.coords[0]))
        end_xy = tuple(map(float, line.coords[-1]))
        endpoints.extend((start_xy, end_xy))

    if not sample_points:
        raise RuntimeError("Failed to generate sampled centerline points for the REM model")
    return sample_points, np.asarray(endpoints, dtype=np.float64)


def _rasterize_points_mask(
    points: list[Point],
    *,
    bounds_3857: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    transform = from_bounds(*bounds_3857, shape[1], shape[0])
    mask = rasterize(
        ((point, 1) for point in points),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _indices_to_coords(
    bounds_3857: tuple[float, float, float, float],
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    left, bottom, right, top = map(float, bounds_3857)
    height, width = int(shape[0]), int(shape[1])
    x_step = (right - left) / max(width, 1)
    y_step = (top - bottom) / max(height, 1)
    xs = left + (np.asarray(cols, dtype=np.float64) + 0.5) * x_step
    ys = top - (np.asarray(rows, dtype=np.float64) + 0.5) * y_step
    return np.column_stack([xs, ys]).astype(np.float32)


def _estimate_sinuosity(river_length_m: float, endpoints_xy: np.ndarray) -> float:
    if len(endpoints_xy) < 2:
        return 1.0
    max_straight = 0.0
    for index in range(len(endpoints_xy) - 1):
        delta = endpoints_xy[index + 1 :] - endpoints_xy[index]
        if len(delta) == 0:
            continue
        distances = np.hypot(delta[:, 0], delta[:, 1])
        if len(distances):
            max_straight = max(max_straight, float(np.max(distances)))
    if max_straight <= 1e-6:
        return 1.0
    return max(1.0, float(river_length_m) / max_straight)


def _estimate_neighbor_count(
    *,
    interp_pts: int,
    river_pixel_count: int,
    sinuosity: float,
) -> int:
    scale_factor = 1.0 + 4.0 * math.tanh(max(float(sinuosity) - 1.0, 0.0))
    k_guess = int((2.0 * float(river_pixel_count) / 100.0) * scale_factor)
    return min(max(int(interp_pts / 10), 5), max(5, k_guess))


def _apply_idw_from_neighbors(
    distances: np.ndarray,
    indices: np.ndarray,
    river_values: np.ndarray,
    *,
    power: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
    if indices.ndim == 1:
        indices = indices[:, None]

    exact = distances <= 1e-6
    safe = np.maximum(distances, 1e-6)
    weights = 1.0 / np.power(safe, np.float32(max(float(power), 0.01)))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    out = (weights * river_values[indices]).sum(axis=1)

    if np.any(exact):
        exact_rows = np.where(exact.any(axis=1))[0]
        first_hits = exact[exact_rows].argmax(axis=1)
        out[exact_rows] = river_values[indices[exact_rows, first_hits]]
    return out.astype(np.float32)


def _idw_chunk_exact(
    river_coords: np.ndarray,
    river_values: np.ndarray,
    query_coords: np.ndarray,
    *,
    k: int,
    power: float,
) -> np.ndarray:
    points = np.asarray(river_coords, dtype=np.float32)
    values = np.asarray(river_values, dtype=np.float32)
    query = np.asarray(query_coords, dtype=np.float32)
    if len(points) == 0:
        raise RuntimeError("Cannot interpolate a river surface without river sample points")

    dx = query[:, None, 0] - points[None, :, 0]
    dy = query[:, None, 1] - points[None, :, 1]
    dist2 = dx * dx + dy * dy
    del dx
    del dy

    neighbor_count = max(1, min(int(k), dist2.shape[1]))
    if neighbor_count == dist2.shape[1]:
        neighbor_indices = np.broadcast_to(
            np.arange(dist2.shape[1], dtype=np.int64),
            dist2.shape,
        )
        neighbor_dist2 = dist2
    else:
        neighbor_indices = np.argpartition(dist2, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        neighbor_dist2 = np.take_along_axis(dist2, neighbor_indices, axis=1)
        order = np.argsort(neighbor_dist2, axis=1)
        neighbor_indices = np.take_along_axis(neighbor_indices, order, axis=1)
        neighbor_dist2 = np.take_along_axis(neighbor_dist2, order, axis=1)

    distances = np.sqrt(np.maximum(neighbor_dist2, 0.0)).astype(np.float32)
    return _apply_idw_from_neighbors(distances, neighbor_indices, values, power=power)


def _idw_chunk_kdtree(
    river_coords: np.ndarray,
    river_values: np.ndarray,
    query_coords: np.ndarray,
    *,
    k: int,
    power: float,
    eps: float,
    workers: int,
) -> np.ndarray:
    if _KDTree is None:  # pragma: no cover - exercised by exact fallback tests
        raise RuntimeError("KDTree acceleration is unavailable")
    tree = _KDTree(np.asarray(river_coords, dtype=np.float32))
    distances, indices = tree.query(
        np.asarray(query_coords, dtype=np.float32),
        k=max(1, min(int(k), len(river_coords))),
        eps=float(eps),
        workers=int(workers),
    )
    return _apply_idw_from_neighbors(distances, indices, np.asarray(river_values, dtype=np.float32), power=power)


def _interpolate_water_surface(
    dem: np.ndarray,
    bounds_3857: tuple[float, float, float, float],
    river_mask: np.ndarray,
    *,
    k: int,
    power: float,
    eps: float,
    workers: int,
    chunk_size: int,
) -> tuple[np.ndarray, int]:
    valid_mask = np.isfinite(dem)
    river_rows, river_cols = np.where(river_mask & valid_mask)
    if len(river_rows) < 2:
        raise RuntimeError("Too few river DEM pixels were identified for REM interpolation")

    river_coords = _indices_to_coords(bounds_3857, dem.shape, river_rows, river_cols)
    river_values = dem[river_rows, river_cols].astype(np.float32)
    water_surface = np.full(dem.shape, np.nan, dtype=np.float32)
    water_surface[river_rows, river_cols] = river_values

    query_rows, query_cols = np.where(valid_mask & (~river_mask))
    query_chunk = max(1, int(chunk_size))
    neighbor_count = max(1, min(int(k), len(river_coords)))
    use_kdtree = _KDTree is not None
    for start in range(0, len(query_rows), query_chunk):
        end = min(start + query_chunk, len(query_rows))
        coords = _indices_to_coords(bounds_3857, dem.shape, query_rows[start:end], query_cols[start:end])
        if use_kdtree:
            interpolated = _idw_chunk_kdtree(
                river_coords,
                river_values,
                coords,
                k=neighbor_count,
                power=power,
                eps=eps,
                workers=workers,
            )
        else:
            interpolated = _idw_chunk_exact(
                river_coords,
                river_values,
                coords,
                k=neighbor_count,
                power=power,
            )
        water_surface[query_rows[start:end], query_cols[start:end]] = interpolated
    return water_surface, len(river_coords)


def _build_rem(
    dem: np.ndarray,
    bounds_3857: tuple[float, float, float, float],
    *,
    bounds_wgs84: tuple[float, float, float, float],
    cache_dir: Path | None = None,
    force: bool = False,
    river_name: str | None,
    interp_pts: int,
    k: int | None,
    idw_power: float,
    eps: float,
    workers: int,
    query_chunk_size: int,
) -> tuple[np.ndarray, RiverRemDiagnostics]:
    selected_name, centerlines, river_length_m = _query_osm_centerlines_3857(
        bounds_wgs84,
        bounds_3857,
        cache_dir=cache_dir,
        force=force,
        river_name=river_name,
    )
    sample_points, endpoints_xy = _sample_centerline_points(centerlines, interp_pts=interp_pts)
    river_mask = _rasterize_points_mask(sample_points, bounds_3857=bounds_3857, shape=dem.shape)
    river_mask &= np.isfinite(dem)
    river_pixel_count = int(np.count_nonzero(river_mask))
    if river_pixel_count < 2:
        raise RuntimeError("Centerline sampling did not hit enough DEM pixels for REM generation")

    sinuosity = _estimate_sinuosity(river_length_m, endpoints_xy)
    neighbor_k = int(k) if k is not None else _estimate_neighbor_count(
        interp_pts=int(interp_pts),
        river_pixel_count=river_pixel_count,
        sinuosity=sinuosity,
    )
    water_surface, river_pixel_count = _interpolate_water_surface(
        dem,
        bounds_3857,
        river_mask,
        k=neighbor_k,
        power=idw_power,
        eps=eps,
        workers=workers,
        chunk_size=query_chunk_size,
    )
    rem = dem.astype(np.float32) - water_surface
    diagnostics = RiverRemDiagnostics(
        river_name=selected_name,
        river_length_m=float(river_length_m),
        sample_point_count=len(sample_points),
        river_pixel_count=river_pixel_count,
        sinuosity=float(sinuosity),
        neighbor_k=max(1, min(int(neighbor_k), river_pixel_count)),
    )
    return rem.astype(np.float32), diagnostics


def _build_rem_overlay(
    rem: np.ndarray,
    *,
    output_size: tuple[int, int],
    palette_name: str,
) -> np.ndarray:
    valid = rem[np.isfinite(rem)]
    if valid.size == 0:
        raise RuntimeError("REM grid is empty")
    if palette_name not in REM_PALETTES:
        available = ", ".join(REM_PALETTES)
        raise RuntimeError(f"Unknown REM palette '{palette_name}'. Available: {available}")
    color_stops = REM_PALETTES[palette_name]

    rem_clip = np.clip(rem, 0.0, None)
    p99 = float(np.quantile(rem_clip, 0.99))
    rem_clip = np.clip(rem_clip, 0.0, max(p99, 1e-6))
    low_scale_m = 0.5
    norm = np.log1p(rem_clip / low_scale_m) / math.log1p(max(p99, 1e-6) / low_scale_m)
    norm = np.clip(norm, 0.0, 1.0)
    palette_t = 0.03 + 0.90 * (1.0 - np.power(norm, 0.42))

    rgb = np.empty(rem.shape + (3,), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.interp(
            palette_t,
            color_stops[:, 0],
            color_stops[:, channel + 1],
        )

    rem_gray = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
    small = np.asarray(rem_gray.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32) / 255.0
    medium = np.asarray(rem_gray.filter(ImageFilter.GaussianBlur(radius=2.8)), dtype=np.float32) / 255.0
    detail = np.clip(small - medium, -1.0, 1.0)
    rgb[..., 0] = np.clip(rgb[..., 0] + detail * 0.026, 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + detail * 0.021, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] + detail * 0.015, 0.0, 1.0)

    low_lift = np.exp(-rem_clip / 0.10)
    channel_highlight = np.clip(low_lift * np.clip(detail * 10.0 - 0.02, 0.0, 1.0), 0.0, 1.0)
    channel_tint = np.array([0.98, 0.93, 0.84], dtype=np.float32)
    rgb = np.clip(
        rgb * (1.0 - channel_highlight[..., None] * 0.11)
        + channel_tint * (channel_highlight[..., None] * 0.11),
        0.0,
        1.0,
    )

    floodplain_cool = np.clip(low_lift * (1.0 - channel_highlight) * 0.028, 0.0, 1.0)
    cool_tint = np.array([0.50, 0.42, 0.58], dtype=np.float32)
    rgb = np.clip(
        rgb * (1.0 - floodplain_cool[..., None]) + cool_tint * floodplain_cool[..., None],
        0.0,
        1.0,
    )

    native = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    resized = native.resize(output_size, resample=Image.Resampling.LANCZOS)
    resized = resized.filter(ImageFilter.UnsharpMask(radius=1.4, percent=145, threshold=2))
    return np.asarray(resized, dtype=np.uint8)


def _save_overlay_png(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)
    return path


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _extract_subject(raw: Image.Image) -> Image.Image:
    rgba = raw.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    corners = np.asarray(
        [arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]],
        dtype=np.int16,
    )
    background = np.median(corners, axis=0)
    diff = np.abs(arr[:, :, :3].astype(np.int16) - background).max(axis=2)
    keep = diff > 8
    if not np.any(keep):
        return rgba

    out = arr.copy()
    out[:, :, 3] = np.where(keep, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(keep)
    pad = max(18, round(max(raw.size) * 0.03))
    crop = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(raw.width, int(xs.max()) + pad + 1),
        min(raw.height, int(ys.max()) + pad + 1),
    )
    return Image.fromarray(out, mode="RGBA").crop(crop)


def _compose_snapshot(
    raw_path: Path,
    output_path: Path,
    *,
    bounds_wgs84: tuple[float, float, float, float],
    camera_phi_deg: float,
    aoi_width_m: float,
    aoi_height_m: float,
) -> None:
    raw = Image.open(raw_path).convert("RGBA")
    width, height = raw.size
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    margin_x = max(42, width // 34)
    margin_top = max(36, height // 32)
    margin_bottom = max(32, height // 34)
    title_gap = max(10, height // 120)
    caption_gap = max(6, height // 220)
    footer_gap = max(24, height // 36)
    header_pad_y = max(10, height // 120)
    footer_pad_y = max(12, height // 110)
    header_edge_gap_x = max(20, width // 96)
    footer_edge_gap_x = max(20, width // 96)
    footer_item_gap = max(28, width // 70)

    title_font = _load_font(max(34, width // 30), bold=True)
    caption_font = _load_font(max(14, width // 86), bold=False)
    micro_font = _load_font(max(12, width // 120), bold=False)

    title_box = draw.textbbox((0, 0), MAP_TITLE, font=title_font)
    title_h = title_box[3] - title_box[1]
    north_rotation = float((camera_phi_deg - 90.0) % 360.0)
    north_arrow = NorthArrow(
        NorthArrowConfig(
            style="compass",
            size=max(128, width // 17),
            rotation_deg=north_rotation,
            color=(28, 28, 34, 255),
            background=(255, 255, 255, 224),
            border_width=1,
            border_color=(38, 38, 44, 255),
            font_size=max(22, width // 118),
        )
    ).render()
    north_arrow_img = Image.fromarray(north_arrow, mode="RGBA")
    header_h = max(title_h, north_arrow_img.height) + header_pad_y * 2

    scale_bar_config = ScaleBarConfig(
        units="km",
        style="alternating",
        width_px=max(420, width // 4),
        height_px=max(52, height // 22),
        divisions=4,
        font_size=max(18, width // 120),
        padding=max(12, width // 240),
        bar_height=max(13, height // 105),
        background=(255, 255, 255, 224),
        bar_color_1=(28, 28, 34, 255),
        bar_color_2=(255, 255, 255, 255),
        label_color=(32, 32, 38, 255),
        border_color=(38, 38, 44, 255),
    )
    scale_bar_reserved = Image.fromarray(ScaleBar(1.0, scale_bar_config).render(), mode="RGBA")
    caption_text = (
        f"{SITE_REGION} | AOI {aoi_width_m / 1000.0:.1f} x {aoi_height_m / 1000.0:.1f} km | "
        "USGS 3DEP DEM + OpenStreetMap waterways | Forge3D RiverREM render"
    )
    caption_lines = _wrap_text_to_width(
        draw,
        caption_text,
        font=caption_font,
        max_width=max(
            220,
            width - margin_x * 2 - footer_edge_gap_x - scale_bar_config.width_px - footer_item_gap,
        ),
    )
    micro_text = "Scale bar is approximate for this oblique 3D view"
    micro_box = draw.textbbox((0, 0), micro_text, font=micro_font)
    micro_h = micro_box[3] - micro_box[1]
    caption_box = draw.textbbox((0, 0), "Ag", font=caption_font)
    caption_line_h = caption_box[3] - caption_box[1]
    caption_block_h = len(caption_lines) * caption_line_h + max(0, len(caption_lines) - 1) * caption_gap
    footer_main_h = max(caption_block_h, scale_bar_reserved.height) + footer_pad_y * 2
    footer_h = footer_main_h + max(10, height // 140) + micro_h

    map_top = margin_top + header_h + title_gap + max(18, height // 48)
    footer_top = height - margin_bottom - footer_h
    map_bottom = footer_top - footer_gap
    map_region_w = width - margin_x * 2
    map_region_h = max(1, map_bottom - map_top)

    subject = _extract_subject(raw)
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    shadow_alpha = subject.getchannel("A").filter(ImageFilter.GaussianBlur(radius=max(8, width // 220)))
    shadow.putalpha(shadow_alpha)

    scale = min(map_region_w / max(subject.width, 1), map_region_h / max(subject.height, 1))
    new_size = (max(1, int(round(subject.width * scale))), max(1, int(round(subject.height * scale))))
    subject = subject.resize(new_size, Image.Resampling.LANCZOS)
    shadow = shadow.resize(new_size, Image.Resampling.LANCZOS)

    map_x = margin_x + (map_region_w - subject.width) // 2
    map_y = map_top + (map_region_h - subject.height) // 2
    canvas.alpha_composite(shadow, dest=(map_x + max(8, width // 180), map_y + max(12, height // 120)))
    canvas.alpha_composite(subject, dest=(map_x, map_y))

    draw = ImageDraw.Draw(canvas)
    title_y = margin_top + (header_h - title_h) // 2
    draw.text((margin_x, title_y), MAP_TITLE, fill=(38, 38, 44, 255), font=title_font)
    north_x = width - margin_x - header_edge_gap_x - north_arrow_img.width
    north_y = margin_top + (header_h - north_arrow_img.height) // 2
    canvas.alpha_composite(north_arrow_img, dest=(north_x, north_y))

    bbox_width_deg = float(bounds_wgs84[2] - bounds_wgs84[0])
    center_lat = 0.5 * float(bounds_wgs84[1] + bounds_wgs84[3])
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
    meters_per_pixel = max(1e-6, (bbox_width_deg * meters_per_deg_lon) / max(subject.width, 1))
    scale_bar = ScaleBar(meters_per_pixel, scale_bar_config).render()
    scale_bar_img = Image.fromarray(scale_bar, mode="RGBA")
    scale_bar_x = width - margin_x - footer_edge_gap_x - scale_bar_img.width
    scale_bar_y = footer_top + (footer_main_h - scale_bar_img.height) // 2
    canvas.alpha_composite(scale_bar_img, dest=(scale_bar_x, scale_bar_y))

    caption_top = footer_top + (footer_main_h - caption_block_h) // 2
    for index, line in enumerate(caption_lines):
        line_y = caption_top + index * (caption_line_h + caption_gap)
        draw.text((margin_x, line_y), line, fill=(74, 74, 82, 255), font=caption_font)
    draw.text(
        (margin_x, footer_top + footer_main_h + max(10, height // 140)),
        micro_text,
        fill=(112, 114, 122, 255),
        font=micro_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _yaw_quaternion_deg(yaw_deg: float) -> tuple[float, float, float, float]:
    half_angle = 0.5 * math.radians(float(yaw_deg))
    return (0.0, float(math.sin(half_angle)), 0.0, float(math.cos(half_angle)))


def _pick_rover_anchor(
    dem: np.ndarray,
    rem: np.ndarray,
    *,
    zscale: float,
) -> tuple[float, float, float]:
    source = TerrainScatterSource(np.asarray(dem, dtype=np.float32), z_scale=float(zscale))
    slope = source.slope_degrees
    height, width = dem.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    row_norm = rows / max(height - 1, 1)
    col_norm = cols / max(width - 1, 1)
    finite = np.isfinite(dem) & np.isfinite(rem) & np.isfinite(slope)
    margin = (row_norm > 0.16) & (row_norm < 0.88) & (col_norm > 0.14) & (col_norm < 0.84)
    candidate = finite & margin & (rem > 3.0) & (rem < 32.0) & (slope < 9.5)
    if not np.any(candidate):
        candidate = finite & margin & (rem > 1.5) & (rem < 48.0) & (slope < 14.0)
    if not np.any(candidate):
        candidate = finite & margin
    if not np.any(candidate):
        return source.pixel_to_contract(float(height // 2), float(width // 2))

    center_bias = ((col_norm - 0.58) / 0.24) ** 2 + ((row_norm - 0.70) / 0.20) ** 2
    rem_bias = np.abs(rem - 10.0) / 10.0
    slope_bias = slope / 12.0
    score = center_bias + rem_bias + slope_bias
    score = np.where(candidate, score, np.inf)
    best_index = int(np.argmin(score))
    best_row, best_col = np.unravel_index(best_index, score.shape)
    return source.pixel_to_contract(float(best_row), float(best_col))


def _render_with_forge3d(
    dem_path: Path,
    overlay_path: Path,
    output_path: Path,
    *,
    dem: np.ndarray,
    rem: np.ndarray,
    bounds_wgs84: tuple[float, float, float, float],
    aoi_width_m: float,
    aoi_height_m: float,
    dem_shape: tuple[int, int],
    cache_dir: Path,
    viewer_size: tuple[int, int],
    render_size: tuple[int, int],
    zscale: float,
    camera_phi_deg: float,
    camera_theta_deg: float,
    camera_radius_scale: float,
    camera_fov_deg: float,
    rover_gltf: Path | None,
    rover_scale: float,
    rover_yaw_deg: float,
    rover_height_offset: float,
) -> None:
    if not f3d.has_gpu():
        raise SystemExit(
            "Forge3D GPU terrain rendering is unavailable. "
            "The DEM, REM, and overlay were still prepared successfully; rerun with --prepare-only."
        )

    terrain_width = float(max(dem_shape))
    radius_scale = float(camera_radius_scale) * (1.08 if rover_gltf is not None else 1.0)
    terrain_cmd = {
        "phi": float(camera_phi_deg),
        "theta": float(camera_theta_deg),
        "radius": terrain_width * radius_scale,
        "fov": float(camera_fov_deg),
        "zscale": float(zscale),
        "sun_azimuth": 302.0,
        "sun_elevation": 34.0,
        "sun_intensity": 1.18,
        "ambient": 0.47,
        "shadow": 0.40,
        "background": [0.952, 0.958, 0.972],
    }
    hdr_path = _ensure_hdr(cache_dir / HDR_FILENAME)
    pbr_config: dict[str, object] = {
        "enabled": True,
        "shadow_technique": "pcss",
        "shadow_map_res": 4096,
        "exposure": 0.97,
        "msaa": 4,
        "ibl_intensity": 0.56,
        "normal_strength": 1.28,
        "height_ao": {
            "enabled": True,
            "directions": 10,
            "steps": 22,
            "max_distance": 260.0,
            "strength": 0.30,
            "resolution_scale": 0.72,
        },
        "sun_visibility": {
            "enabled": True,
            "mode": "soft",
            "samples": 1,
            "strength": 0.24,
            "softness": 0.42,
        },
        "tonemap": {
            "operator": "aces",
            "white_point": 2.6,
            "white_balance_enabled": True,
            "temperature": 6500.0,
            "tint": 0.0,
        },
        "hdr_path": str(hdr_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_snapshot_path = cache_dir / f"{output_path.stem}_raw.png"
    with f3d.open_viewer_async(
        terrain_path=dem_path,
        width=int(viewer_size[0]),
        height=int(viewer_size[1]),
        fov_deg=float(terrain_cmd["fov"]),
        timeout=60.0,
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
        viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
        viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr_config})
        viewer.load_overlay(
            "snake_rem",
            overlay_path,
            extent=(0.0, 0.0, 1.0, 1.0),
            opacity=1.0,
            z_order=10,
            preserve_colors=True,
        )
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
        viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
        if rover_gltf is not None:
            rover_x, rover_y, rover_z = _pick_rover_anchor(dem, rem, zscale=float(zscale))
            viewer.load_gltf(rover_gltf)
            viewer.set_transform(
                translation=(rover_x, rover_y + float(rover_height_offset), rover_z),
                rotation_quat=_yaw_quaternion_deg(float(rover_yaw_deg)),
                scale=(float(rover_scale), float(rover_scale), float(rover_scale)),
            )
        time.sleep(VIEWER_SETTLE_SECONDS)
        viewer.snapshot(raw_snapshot_path, width=int(render_size[0]), height=int(render_size[1]))
    _compose_snapshot(
        raw_snapshot_path,
        output_path,
        bounds_wgs84=bounds_wgs84,
        camera_phi_deg=float(camera_phi_deg),
        aoi_width_m=float(aoi_width_m),
        aoi_height_m=float(aoi_height_m),
    )


def main() -> int:
    args = parse_args()
    if args.side_length_m is not None and args.side_length_m <= 0.0:
        raise SystemExit("--side-length-m must be positive when provided")
    if args.width_m <= 0.0:
        raise SystemExit("--width-m must be positive")
    if args.height_m <= 0.0:
        raise SystemExit("--height-m must be positive")
    if args.dem_resolution_m <= 0.0:
        raise SystemExit("--dem-resolution-m must be positive")
    if args.dem_size is not None and args.dem_size <= 0:
        raise SystemExit("--dem-size must be positive when provided")
    if args.tile_size_px <= 0:
        raise SystemExit("--tile-size-px must be positive")
    if args.interp_pts <= 1:
        raise SystemExit("--interp-pts must be greater than 1")
    if args.k is not None and args.k <= 0:
        raise SystemExit("--k must be positive when provided")
    if args.query_chunk_size <= 0:
        raise SystemExit("--query-chunk-size must be positive")
    if args.camera_radius_scale <= 0.0:
        raise SystemExit("--camera-radius-scale must be positive")
    if args.camera_fov_deg <= 0.0:
        raise SystemExit("--camera-fov-deg must be positive")
    if args.rover_scale <= 0.0:
        raise SystemExit("--rover-scale must be positive")

    cache_dir = args.cache_dir.resolve()
    output_path = args.output.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    center = (float(args.center_lon), float(args.center_lat))
    palette_name = str(args.palette)
    if args.side_length_m is not None:
        width_m = float(args.side_length_m)
        height_m = float(args.side_length_m)
    else:
        width_m = float(args.width_m)
        height_m = float(args.height_m)
    bounds_3857 = _bounds_from_center_wgs84(center, width_m=width_m, height_m=height_m)
    bounds_wgs84 = _project_bounds_3857_to_wgs84(bounds_3857)
    river_name = str(args.river_name).strip() or None
    rover_gltf = None if args.rover_gltf is None else args.rover_gltf.resolve()
    if rover_gltf is not None and not rover_gltf.exists():
        raise SystemExit(f"--rover-gltf path does not exist: {rover_gltf}")

    dem_path, dem = _download_three_dep_dem(
        bounds_3857,
        dem_resolution_m=float(args.dem_resolution_m),
        dem_size=None if args.dem_size is None else int(args.dem_size),
        tile_size_px=int(args.tile_size_px),
        align_to=16,
        cache_dir=cache_dir,
        force=bool(args.force),
    )
    dem_height, dem_width = dem.shape

    rem_path = cache_dir / f"{SITE_SLUG}_rem_{dem_width}x{dem_height}_3857.tif"
    overlay_path = cache_dir / f"{SITE_SLUG}_rem_overlay_{palette_name}_{dem_width}x{dem_height}.png"
    diagnostics_path = cache_dir / f"{SITE_SLUG}_diagnostics_{dem_width}x{dem_height}.json"
    if not rem_path.exists() or not diagnostics_path.exists() or bool(args.force):
        rem, diagnostics = _build_rem(
            dem,
            bounds_3857,
            bounds_wgs84=bounds_wgs84,
            cache_dir=cache_dir,
            force=bool(args.force),
            river_name=river_name,
            interp_pts=int(args.interp_pts),
            k=None if args.k is None else int(args.k),
            idw_power=float(args.idw_power),
            eps=float(args.eps),
            workers=int(args.workers),
            query_chunk_size=int(args.query_chunk_size),
        )
        _write_geotiff(rem_path, rem, bounds_3857)
        diagnostics_path.write_text(
            json.dumps(
                {
                    "river_name": diagnostics.river_name,
                    "river_length_m": diagnostics.river_length_m,
                    "sample_point_count": diagnostics.sample_point_count,
                    "river_pixel_count": diagnostics.river_pixel_count,
                    "sinuosity": diagnostics.sinuosity,
                    "neighbor_k": diagnostics.neighbor_k,
                    "kdtree_acceleration": bool(_KDTree is not None),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        with rasterio.open(rem_path) as dataset:
            rem = dataset.read(1).astype(np.float32)
        diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics = RiverRemDiagnostics(
            river_name=str(diagnostics_payload["river_name"]),
            river_length_m=float(diagnostics_payload["river_length_m"]),
            sample_point_count=int(diagnostics_payload["sample_point_count"]),
            river_pixel_count=int(diagnostics_payload["river_pixel_count"]),
            sinuosity=float(diagnostics_payload["sinuosity"]),
            neighbor_k=int(diagnostics_payload["neighbor_k"]),
        )

    if not overlay_path.exists() or bool(args.force):
        overlay_rgb = _build_rem_overlay(
            rem,
            output_size=(int(dem_width), int(dem_height)),
            palette_name=palette_name,
        )
        _save_overlay_png(overlay_path, overlay_rgb)

    print(f"[site] {diagnostics.river_name} | {SITE_REGION}")
    print(f"[palette] {palette_name}")
    print(f"[artifact] DEM: {dem_path}")
    print(f"[artifact] REM: {rem_path}")
    print(f"[artifact] Overlay: {overlay_path}")
    print(
        "[river] "
        f"length={diagnostics.river_length_m:.1f} m | "
        f"sample_points={diagnostics.sample_point_count} | "
        f"river_pixels={diagnostics.river_pixel_count} | "
        f"sinuosity={diagnostics.sinuosity:.2f} | "
        f"k={diagnostics.neighbor_k}"
    )
    print(
        "[stats] REM min/max/p99 = "
        f"{float(np.nanmin(rem)):.2f} / {float(np.nanmax(rem)):.2f} / {float(np.nanquantile(rem, 0.99)):.2f} m"
    )

    if args.prepare_only:
        return 0

    _render_with_forge3d(
        dem_path,
        overlay_path,
        output_path,
        dem=dem,
        rem=rem,
        bounds_wgs84=bounds_wgs84,
        aoi_width_m=width_m,
        aoi_height_m=height_m,
        dem_shape=dem.shape,
        cache_dir=cache_dir,
        viewer_size=(int(args.viewer_size[0]), int(args.viewer_size[1])),
        render_size=(int(args.render_size[0]), int(args.render_size[1])),
        zscale=float(args.zscale),
        camera_phi_deg=float(args.camera_phi_deg),
        camera_theta_deg=float(args.camera_theta_deg),
        camera_radius_scale=float(args.camera_radius_scale),
        camera_fov_deg=float(args.camera_fov_deg),
        rover_gltf=rover_gltf,
        rover_scale=float(args.rover_scale),
        rover_yaw_deg=float(args.rover_yaw_deg),
        rover_height_offset=float(args.rover_height_offset),
    )
    print(f"[artifact] Render: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
