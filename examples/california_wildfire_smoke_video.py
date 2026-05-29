#!/usr/bin/env python3
"""California wildfire and smoke exposure video.

This example renders a NASA-GEOS/FIRMS-style wildfire map animation over a
real 3D California terrain surface. The California mask comes from the
OpenStreetMap administrative boundary relation, and the terrain elevation comes
from cached Terrarium DEM tiles.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "examples"
    / "out"
    / "california_wildfire_smoke"
    / "california_2020_wildfire_smoke_exposure.mp4"
)
DEFAULT_PREVIEW = DEFAULT_OUTPUT.with_suffix(".preview.png")
DEFAULT_DEM = PROJECT_ROOT / "assets" / "tif" / "Bryce_Canyon.tif"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "california_wildfire_smoke"

WIDTH = 1920
HEIGHT = 1080
FPS = 30
FRAMES = 240
START_DATE = dt.date(2020, 8, 16)
END_DATE = dt.date(2020, 10, 31)

OSM_RELATION_ID = 165475
OSM_BOUNDARY_URL = (
    "https://nominatim.openstreetmap.org/lookup"
    f"?osm_ids=R{OSM_RELATION_ID}"
    "&format=geojson"
    "&polygon_geojson=1"
    "&polygon_threshold=0.0"
)
USER_AGENT = "forge3d-california-wildfire-smoke/1.0"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DEFAULT_DEM_ZOOM = 8
DEFAULT_RENDER_MAX_SIZE = 1700
WEB_MERCATOR_LIMIT = 20037508.342789244

LON_MIN = -125.65
LON_MAX = -113.05
LAT_MIN = 31.10
LAT_MAX = 43.35

FONT_CANDIDATES = (
    "/Users/mpopovic3/Library/Fonts/Inconsolata.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSans.ttf",
)

WIND_U_VAR_CANDIDATES = (
    "u",
    "u10",
    "10u",
    "ugrd",
    "ugrd10m",
    "u_component_of_wind",
    "eastward_wind",
)
WIND_V_VAR_CANDIDATES = (
    "v",
    "v10",
    "10v",
    "vgrd",
    "vgrd10m",
    "v_component_of_wind",
    "northward_wind",
)
WIND_LAT_COORD_CANDIDATES = ("latitude", "lat", "y")
WIND_LON_COORD_CANDIDATES = ("longitude", "lon", "x")
WIND_TIME_COORD_CANDIDATES = ("time", "valid_time")
WIND_HEIGHT_COORD_CANDIDATES = (
    "height",
    "height_m",
    "heightAboveGround",
    "level",
    "lev",
    "altitude",
    "isobaricInhPa",
)
DEFAULT_WIND_HEIGHT_M = 500.0
DEFAULT_WIND_SPEED_SCALE = 0.36
SMOKE_PARTICLE_MAX_AGE_HOURS = 48.0
SMOKE_PARTICLE_TRAIL_HOURS = 22.0
SMOKE_PARTICLES_PER_CLUSTER = 100
SMOKE_SOURCE_MIN_RADIUS_M = 2_800.0
SMOKE_SOURCE_MAX_RADIUS_M = 24_000.0
SMOKE_WIND_SMOOTH_HOURS = (-18.0, -9.0, 0.0, 9.0, 18.0)
SMOKE_REGIONAL_ANCHOR = (-120.6, 38.1)
SMOKE_REGIONAL_BLEND = 0.58
# Veil layer parameters for broad atmospheric sheets
SMOKE_VEIL_RESOLUTION = 384
SMOKE_VEIL_BLUR_RADIUS = 4.8
SMOKE_VEIL_MAX_ALPHA = 46
SMOKE_RIBBON_COUNT = 24
SMOKE_VEIL_MAX_AGE_HOURS = 58.0
SMOKE_ADVECTION_RANGE_BOOST = 1.28
SMOKE_DENSITY_RESOLUTION = 512
SMOKE_DENSITY_MAX_ALPHA = 112
SMOKE_DENSITY_DECAY = 0.983
SMOKE_DENSITY_BORDER_FADE = 0.055


@dataclass(frozen=True)
class FireCluster:
    name: str
    lon: float
    lat: float
    start_day: float
    ramp_days: float
    final_area_ha: float
    spread_km: float
    wind_angle_deg: float
    spark_count: int
    seed: int


@dataclass(frozen=True)
class TerrainCamera:
    target: np.ndarray
    eye: np.ndarray
    forward: np.ndarray
    side: np.ndarray
    up: np.ndarray
    fov_deg: float


@dataclass(frozen=True)
class TerrainAssets:
    dem_path: Path
    overlay_path: Path
    boundary_path: Path
    bounds_mercator: tuple[float, float, float, float]
    terrain_span_m: float


@dataclass(frozen=True)
class WindField:
    """Scalar sampler for eastward/northward wind components in m/s."""

    source_label: str
    times: np.ndarray
    time_seconds: np.ndarray
    heights_m: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    u_mps: np.ndarray
    v_mps: np.ndarray

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        u_var: str | None = None,
        v_var: str | None = None,
        default_height_m: float = DEFAULT_WIND_HEIGHT_M,
    ) -> "WindField":
        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls._from_json(path)
        return cls._from_xarray(
            path,
            u_var=u_var,
            v_var=v_var,
            default_height_m=float(default_height_m),
        )

    @classmethod
    def _from_json(cls, path: Path) -> "WindField":
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = str(payload.get("source") or path.name)
        times = np.asarray(
            [np.datetime64(value, "s") for value in payload.get("times", [START_DATE.isoformat()])],
            dtype="datetime64[s]",
        )
        heights = np.asarray(
            payload.get("heights_m") or payload.get("height_m") or [DEFAULT_WIND_HEIGHT_M],
            dtype=np.float64,
        )
        latitudes = np.asarray(payload.get("latitudes") or payload.get("latitude") or payload.get("lat"), dtype=np.float64)
        longitudes = np.asarray(payload.get("longitudes") or payload.get("longitude") or payload.get("lon"), dtype=np.float64)
        u_values = np.asarray(payload.get("u_mps") or payload.get("u"), dtype=np.float32)
        v_values = np.asarray(payload.get("v_mps") or payload.get("v"), dtype=np.float32)
        return cls._from_arrays(source, times, heights, latitudes, longitudes, u_values, v_values)

    @classmethod
    def _from_xarray(
        cls,
        path: Path,
        *,
        u_var: str | None,
        v_var: str | None,
        default_height_m: float,
    ) -> "WindField":
        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Reading NetCDF/GRIB wind files requires xarray. Install forge3d[raster] "
                "or provide the example JSON wind format."
            ) from exc

        open_errors: list[Exception] = []
        for kwargs in ({}, {"engine": "cfgrib"}):
            try:
                dataset = xr.open_dataset(path, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - engine availability varies
                open_errors.append(exc)
        else:  # pragma: no cover - exercised only on unsupported files
            details = "; ".join(str(error) for error in open_errors)
            raise RuntimeError(f"Could not open wind file {path}: {details}")

        try:
            dataset = dataset.load()
            u_name = _find_wind_data_var(dataset, u_var, WIND_U_VAR_CANDIDATES)
            v_name = _find_wind_data_var(dataset, v_var, WIND_V_VAR_CANDIDATES)
            u_da = dataset[u_name]
            v_da = dataset[v_name]
            lat_name = _find_wind_coord(u_da, WIND_LAT_COORD_CANDIDATES)
            lon_name = _find_wind_coord(u_da, WIND_LON_COORD_CANDIDATES)
            time_name = _find_optional_wind_coord(u_da, WIND_TIME_COORD_CANDIDATES)
            height_name = _find_optional_wind_coord(u_da, WIND_HEIGHT_COORD_CANDIDATES)

            u_cube, times, heights, latitudes, longitudes = _wind_dataarray_to_cube(
                u_da,
                lat_name=lat_name,
                lon_name=lon_name,
                time_name=time_name,
                height_name=height_name,
                default_height_m=default_height_m,
            )
            v_cube, _, _, _, _ = _wind_dataarray_to_cube(
                v_da,
                lat_name=lat_name,
                lon_name=lon_name,
                time_name=time_name,
                height_name=height_name,
                default_height_m=default_height_m,
            )
            source = f"{path.name}:{u_name}/{v_name}"
            return cls._from_arrays(source, times, heights, latitudes, longitudes, u_cube, v_cube)
        finally:
            dataset.close()

    @classmethod
    def _from_arrays(
        cls,
        source_label: str,
        times: np.ndarray,
        heights_m: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        u_mps: np.ndarray,
        v_mps: np.ndarray,
    ) -> "WindField":
        times = np.asarray(times, dtype="datetime64[s]").reshape(-1)
        heights_m = np.asarray(heights_m, dtype=np.float64).reshape(-1)
        latitudes = np.asarray(latitudes, dtype=np.float64).reshape(-1)
        longitudes = np.asarray(longitudes, dtype=np.float64).reshape(-1)
        u_mps = _wind_values_to_cube(u_mps, len(times), len(heights_m), len(latitudes), len(longitudes))
        v_mps = _wind_values_to_cube(v_mps, len(times), len(heights_m), len(latitudes), len(longitudes))

        time_order = np.argsort(times)
        height_order = np.argsort(heights_m)
        lat_order = np.argsort(latitudes)
        lon_order = np.argsort(longitudes)

        times = times[time_order]
        heights_m = heights_m[height_order]
        latitudes = latitudes[lat_order]
        longitudes = longitudes[lon_order]
        u_mps = u_mps[time_order][:, height_order][:, :, lat_order][:, :, :, lon_order]
        v_mps = v_mps[time_order][:, height_order][:, :, lat_order][:, :, :, lon_order]
        time_seconds = times.astype("datetime64[s]").astype(np.int64)

        return cls(
            source_label=source_label,
            times=times,
            time_seconds=time_seconds,
            heights_m=heights_m,
            latitudes=latitudes,
            longitudes=longitudes,
            u_mps=u_mps,
            v_mps=v_mps,
        )

    def sample(
        self,
        lon: float,
        lat: float,
        when: dt.datetime | np.datetime64 | None,
        *,
        height_m: float = DEFAULT_WIND_HEIGHT_M,
    ) -> tuple[float, float]:
        lon_value = self._normalize_lon(float(lon))
        time_value = _datetime_to_seconds(when)
        t0, t1, tf = _axis_window(self.time_seconds, time_value)
        z0, z1, zf = _axis_window(self.heights_m, float(height_m))
        y0, y1, yf = _axis_window(self.latitudes, float(lat))
        x0, x1, xf = _axis_window(self.longitudes, lon_value)
        u_value = _sample_wind_cube(self.u_mps, t0, t1, tf, z0, z1, zf, y0, y1, yf, x0, x1, xf)
        v_value = _sample_wind_cube(self.v_mps, t0, t1, tf, z0, z1, zf, y0, y1, yf, x0, x1, xf)
        return float(u_value), float(v_value)

    def _normalize_lon(self, lon: float) -> float:
        lon_min = float(self.longitudes[0])
        lon_max = float(self.longitudes[-1])
        value = lon
        if lon_min >= 0.0 and value < 0.0:
            value += 360.0
        elif lon_max <= 180.0 and value > 180.0:
            value -= 360.0
        if lon_max - lon_min > 300.0:
            while value < lon_min:
                value += 360.0
            while value > lon_max:
                value -= 360.0
        return float(np.clip(value, lon_min, lon_max))


def _normalized_wind_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _find_wind_data_var(dataset, explicit: str | None, candidates: tuple[str, ...]) -> str:
    if explicit:
        if explicit not in dataset.data_vars:
            raise RuntimeError(f"Wind variable {explicit!r} was not found in {list(dataset.data_vars)}")
        return explicit
    candidate_keys = {_normalized_wind_name(name) for name in candidates}
    for name in dataset.data_vars:
        if _normalized_wind_name(str(name)) in candidate_keys:
            return str(name)
    raise RuntimeError(
        "Could not infer wind component variable. Provide --wind-u-var/--wind-v-var. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def _find_wind_coord(data_array, candidates: tuple[str, ...]) -> str:
    name = _find_optional_wind_coord(data_array, candidates)
    if name is None:
        raise RuntimeError(
            f"Could not infer required wind coordinate from {candidates}. "
            f"Variable dimensions are {data_array.dims}."
        )
    return name


def _find_optional_wind_coord(data_array, candidates: tuple[str, ...]) -> str | None:
    candidate_keys = {_normalized_wind_name(name) for name in candidates}
    for name in list(data_array.dims) + list(data_array.coords):
        if _normalized_wind_name(str(name)) in candidate_keys:
            return str(name)
    return None


def _wind_dataarray_to_cube(
    data_array,
    *,
    lat_name: str,
    lon_name: str,
    time_name: str | None,
    height_name: str | None,
    default_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lat_name not in data_array.dims or lon_name not in data_array.dims:
        raise RuntimeError(
            "Wind variables must expose 1-D latitude/longitude dimensions. "
            f"Got dimensions {data_array.dims}."
        )

    keep_dims = {lat_name, lon_name}
    if time_name in data_array.dims:
        keep_dims.add(time_name)
    if height_name in data_array.dims:
        keep_dims.add(height_name)
    extra_dims = {dim: 0 for dim in data_array.dims if dim not in keep_dims}
    if extra_dims:
        data_array = data_array.isel(extra_dims)

    if time_name in data_array.dims:
        time_dim = str(time_name)
        times = np.asarray(data_array[time_dim].values, dtype="datetime64[s]")
    else:
        time_dim = "_wind_time"
        times = np.asarray([np.datetime64(START_DATE.isoformat(), "s")], dtype="datetime64[s]")
        data_array = data_array.expand_dims({time_dim: times})

    if height_name in data_array.dims:
        height_dim = str(height_name)
        heights = np.asarray(data_array[height_dim].values, dtype=np.float64)
    else:
        height_dim = "_wind_height"
        heights = np.asarray([default_height_m], dtype=np.float64)
        data_array = data_array.expand_dims({height_dim: heights})

    data_array = data_array.transpose(time_dim, height_dim, lat_name, lon_name)
    values = np.asarray(data_array.values, dtype=np.float32)
    latitudes = np.asarray(data_array[lat_name].values, dtype=np.float64)
    longitudes = np.asarray(data_array[lon_name].values, dtype=np.float64)
    return values, times, heights, latitudes, longitudes


def _wind_values_to_cube(
    values: np.ndarray,
    time_count: int,
    height_count: int,
    lat_count: int,
    lon_count: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(1, 1, lat_count, lon_count)
    elif arr.ndim == 3:
        if arr.shape[0] == time_count and height_count == 1:
            arr = arr.reshape(time_count, 1, lat_count, lon_count)
        elif arr.shape[0] == height_count and time_count == 1:
            arr = arr.reshape(1, height_count, lat_count, lon_count)
        else:
            raise RuntimeError(
                "3-D wind arrays are ambiguous; expected (time, lat, lon) or "
                "(height, lat, lon)."
            )
    elif arr.ndim != 4:
        raise RuntimeError("Wind arrays must be 2-D, 3-D, or 4-D.")

    expected = (time_count, height_count, lat_count, lon_count)
    if arr.shape != expected:
        raise RuntimeError(f"Wind array shape {arr.shape} does not match expected {expected}.")
    return arr


def _datetime_to_seconds(when: dt.datetime | np.datetime64 | None) -> int:
    if when is None:
        when = dt.datetime.combine(START_DATE, dt.time(12, 0))
    if isinstance(when, np.datetime64):
        return int(when.astype("datetime64[s]").astype(np.int64))
    if isinstance(when, dt.date) and not isinstance(when, dt.datetime):
        when = dt.datetime.combine(when, dt.time(12, 0))
    if when.tzinfo is not None:
        when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return int(np.datetime64(when, "s").astype(np.int64))


def _axis_window(axis: np.ndarray, value: float | int) -> tuple[int, int, float]:
    arr = np.asarray(axis)
    if arr.size == 0:
        raise RuntimeError("Cannot sample a wind field with an empty coordinate axis.")
    if arr.size == 1:
        return 0, 0, 0.0
    clipped = float(np.clip(float(value), float(arr[0]), float(arr[-1])))
    hi = int(np.searchsorted(arr, clipped, side="right"))
    hi = min(max(hi, 1), arr.size - 1)
    lo = hi - 1
    denom = float(arr[hi] - arr[lo])
    frac = 0.0 if abs(denom) < 1e-12 else (clipped - float(arr[lo])) / denom
    return lo, hi, float(np.clip(frac, 0.0, 1.0))


def _sample_wind_cube(
    cube: np.ndarray,
    t0: int,
    t1: int,
    tf: float,
    z0: int,
    z1: int,
    zf: float,
    y0: int,
    y1: int,
    yf: float,
    x0: int,
    x1: int,
    xf: float,
) -> float:
    samples: list[float] = []
    weights: list[float] = []
    for ti, tw in ((t0, 1.0 - tf), (t1, tf)):
        for zi, zw in ((z0, 1.0 - zf), (z1, zf)):
            for yi, yw in ((y0, 1.0 - yf), (y1, yf)):
                for xi, xw in ((x0, 1.0 - xf), (x1, xf)):
                    weight = tw * zw * yw * xw
                    if weight <= 0.0:
                        continue
                    value = float(cube[ti, zi, yi, xi])
                    if np.isfinite(value):
                        samples.append(value)
                        weights.append(weight)
    if not weights:
        return float("nan")
    weight_arr = np.asarray(weights, dtype=np.float64)
    sample_arr = np.asarray(samples, dtype=np.float64)
    return float(np.sum(sample_arr * weight_arr) / max(float(np.sum(weight_arr)), 1e-12))


def move_lonlat_by_meters(
    lon: float,
    lat: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    lat_rad = math.radians(float(lat))
    meters_per_deg_lat = (
        111_132.92
        - 559.82 * math.cos(2.0 * lat_rad)
        + 1.175 * math.cos(4.0 * lat_rad)
        - 0.0023 * math.cos(6.0 * lat_rad)
    )
    meters_per_deg_lon = (
        111_412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3.0 * lat_rad)
        + 0.118 * math.cos(5.0 * lat_rad)
    )
    next_lat = lat + north_m / max(meters_per_deg_lat, 1e-6)
    next_lon = lon + east_m / max(meters_per_deg_lon, 1e-6)
    return next_lon, next_lat


@dataclass
class TerrainProjector:
    heightmap: np.ndarray
    valid_mask: np.ndarray
    bounds_mercator: tuple[float, float, float, float]
    world_scale: float
    vertical_exaggeration: float
    camera: TerrainCamera
    screen_mask: Image.Image | None = None

    def world_from_mercator(self, mx: float, my: float, elevation_m: float) -> np.ndarray:
        west, south, east, north = self.bounds_mercator
        center_x = (west + east) * 0.5
        center_y = (south + north) * 0.5
        return np.array(
            [
                (mx - center_x) * self.world_scale,
                max(float(elevation_m), 0.0) * self.world_scale * self.vertical_exaggeration,
                (my - center_y) * self.world_scale,
            ],
            dtype=np.float32,
        )

    def sample_elevation(self, lon: float, lat: float) -> float:
        mx, my = lonlat_to_web_mercator(lon, lat)
        west, south, east, north = self.bounds_mercator
        h, w = self.heightmap.shape
        col = (mx - west) / max(east - west, 1e-6) * (w - 1)
        row = (north - my) / max(north - south, 1e-6) * (h - 1)
        if col < 0.0 or row < 0.0 or col > w - 1 or row > h - 1:
            return 0.0
        c0 = int(np.clip(math.floor(col), 0, w - 1))
        r0 = int(np.clip(math.floor(row), 0, h - 1))
        c1 = min(c0 + 1, w - 1)
        r1 = min(r0 + 1, h - 1)
        tx = float(col - c0)
        ty = float(row - r0)
        vals = np.array(
            [
                self.heightmap[r0, c0],
                self.heightmap[r0, c1],
                self.heightmap[r1, c0],
                self.heightmap[r1, c1],
            ],
            dtype=np.float32,
        )
        valid = np.isfinite(vals)
        if not np.any(valid):
            return 0.0
        vals = np.where(valid, vals, float(np.nanmin(vals[valid])))
        top = vals[0] * (1.0 - tx) + vals[1] * tx
        bottom = vals[2] * (1.0 - tx) + vals[3] * tx
        return float(top * (1.0 - ty) + bottom * ty)

    def project_world(self, point: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
        cam = self.camera
        relative = point.astype(np.float32) - cam.eye
        depth = float(np.dot(relative, cam.forward))
        x_cam = float(np.dot(relative, cam.side))
        y_cam = float(np.dot(relative, cam.up))
        focal = 1.0 / math.tan(math.radians(cam.fov_deg) * 0.5)
        aspect = width / max(float(height), 1.0)
        denom = max(depth, 1e-3)
        sx = (x_cam * focal / aspect / denom) * 0.5 + 0.5
        sy = 0.5 - (y_cam * focal / denom) * 0.5
        return sx * width, sy * height, depth

    def lonlat_to_screen(self, lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
        mx, my = lonlat_to_web_mercator(lon, lat)
        elevation = self.sample_elevation(lon, lat)
        point = self.world_from_mercator(mx, my, elevation + 90.0)
        x, y, _ = self.project_world(point, width, height)
        return x, y


FIRE_CLUSTERS = (
    FireCluster("August Complex", -122.9, 39.7, 0.0, 34.0, 417_898.0, 360.0, 72.0, 96, 10),
    FireCluster("LNU Lightning", -122.3, 38.6, 1.0, 17.0, 147_000.0, 250.0, 68.0, 54, 20),
    FireCluster("SCU Lightning", -121.3, 37.4, 0.0, 18.0, 160_000.0, 270.0, 74.0, 58, 30),
    FireCluster("North Complex", -121.1, 39.8, 1.0, 29.0, 129_000.0, 260.0, 77.0, 52, 40),
    FireCluster("Creek Fire", -119.3, 37.2, 19.0, 24.0, 154_000.0, 285.0, 61.0, 58, 50),
    FireCluster("SQF Complex", -118.6, 36.2, 5.0, 29.0, 71_000.0, 215.0, 55.0, 38, 60),
    FireCluster("Bobcat Fire", -117.9, 34.3, 21.0, 21.0, 47_000.0, 185.0, 47.0, 30, 70),
    FireCluster("Glass Fire", -122.5, 38.6, 42.0, 8.0, 27_000.0, 150.0, 71.0, 28, 80),
    FireCluster("Dolan Fire", -121.6, 36.1, 2.0, 23.0, 50_000.0, 185.0, 80.0, 32, 90),
    FireCluster("Slater Fire", -123.4, 42.0, 19.0, 20.0, 64_000.0, 220.0, 72.0, 36, 100),
)

FINAL_SEASON_AREA_HA = 1_740_000.0
FINAL_EXPOSED_POP = 11_200_000.0
TERRAIN_MESH_COLS = 300
TERRAIN_MESH_ROWS = 360

CALIFORNIA_COASTLINE = (
    (-124.21, 42.00),
    (-124.36, 41.72),
    (-124.22, 41.20),
    (-124.12, 40.72),
    (-124.28, 40.43),
    (-124.12, 40.14),
    (-123.86, 39.82),
    (-123.79, 39.38),
    (-123.54, 38.93),
    (-123.26, 38.56),
    (-122.96, 38.30),
    (-122.80, 38.02),
    (-122.54, 37.82),
    (-122.36, 37.82),
    (-122.49, 37.55),
    (-122.36, 37.18),
    (-122.09, 36.95),
    (-121.93, 36.63),
    (-121.83, 36.30),
    (-121.54, 35.98),
    (-121.35, 35.66),
    (-121.10, 35.25),
    (-120.75, 34.90),
    (-120.66, 34.56),
    (-120.30, 34.46),
    (-119.86, 34.38),
    (-119.54, 34.41),
    (-119.22, 34.28),
    (-118.78, 34.03),
    (-118.48, 33.95),
    (-118.22, 33.73),
    (-117.87, 33.60),
    (-117.55, 33.33),
    (-117.25, 32.95),
    (-117.12, 32.53),
)

PACIFIC_COASTLINE = (
    (-124.55, 43.35),
    (-124.48, 42.86),
    *CALIFORNIA_COASTLINE,
    (-116.92, 32.18),
    (-116.58, 31.70),
    (-116.18, 31.10),
)

CALIFORNIA_STATE_OUTLINE = (
    *CALIFORNIA_COASTLINE,
    (-114.72, 32.72),
    (-114.66, 33.16),
    (-114.53, 33.67),
    (-114.43, 34.23),
    (-114.63, 35.00),
    (-116.09, 36.50),
    (-117.60, 38.00),
    (-119.16, 39.50),
    (-120.00, 42.00),
    (-124.21, 42.00),
)

SAN_FRANCISCO_BAY = (
    (-122.55, 38.22),
    (-122.24, 38.18),
    (-121.93, 38.02),
    (-121.86, 37.82),
    (-122.05, 37.62),
    (-122.30, 37.48),
    (-122.43, 37.64),
    (-122.39, 37.85),
)

CHANNEL_ISLANDS = (
    ((-120.46, 34.05), (-120.18, 34.02), (-119.98, 33.96), (-120.28, 33.90)),
    ((-119.95, 34.02), (-119.52, 34.01), (-119.31, 33.92), (-119.76, 33.86)),
    ((-119.02, 34.07), (-118.55, 34.05), (-118.35, 33.94), (-118.82, 33.88)),
    ((-118.62, 33.48), (-118.30, 33.43), (-118.18, 33.31), (-118.48, 33.28)),
    ((-118.56, 32.98), (-118.36, 32.95), (-118.25, 32.82), (-118.47, 32.80)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a California wildfire and smoke exposure MP4."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(WIDTH, HEIGHT),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dem-zoom", type=int, default=DEFAULT_DEM_ZOOM)
    parser.add_argument("--max-dem-size", type=int, default=DEFAULT_RENDER_MAX_SIZE)
    parser.add_argument(
        "--wind-file",
        type=Path,
        default=None,
        help=(
            "Optional real wind data file. NetCDF/GRIB files are read with xarray; "
            "JSON files use the example format with times, latitudes, longitudes, "
            "u_mps, and v_mps arrays."
        ),
    )
    parser.add_argument(
        "--wind-u-var",
        type=str,
        default=None,
        help="Eastward wind component variable name for NetCDF/GRIB input.",
    )
    parser.add_argument(
        "--wind-v-var",
        type=str,
        default=None,
        help="Northward wind component variable name for NetCDF/GRIB input.",
    )
    parser.add_argument(
        "--wind-height-m",
        type=float,
        default=DEFAULT_WIND_HEIGHT_M,
        help="Wind sampling height in meters when the input has vertical levels.",
    )
    parser.add_argument(
        "--wind-speed-scale",
        type=float,
        default=DEFAULT_WIND_SPEED_SCALE,
        help="Visual advection scale applied to physical m/s wind during smoke particle integration.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_font(size_px: int) -> ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size=max(1, int(size_px)))
        except OSError:
            continue
    return ImageFont.load_default()


def smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray | float:
    scaled = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def cluster_progress(cluster: FireCluster, season_day: float) -> float:
    return float(smoothstep(cluster.start_day, cluster.start_day + cluster.ramp_days, season_day))


def season_progress(frame_index: int, total_frames: int) -> tuple[float, float]:
    t = frame_index / max(total_frames - 1, 1)
    days = float((END_DATE - START_DATE).days)
    return t, days * t


def format_date(frame_index: int, total_frames: int) -> str:
    t = frame_index / max(total_frames - 1, 1)
    days = (END_DATE - START_DATE).days
    return (START_DATE + dt.timedelta(days=int(round(days * t)))).isoformat()


def format_area(hectares: float) -> str:
    if hectares >= 1_000_000.0:
        return f"{hectares / 1_000_000.0:.1f} M ha"
    if hectares >= 1_000.0:
        return f"{hectares / 1_000.0:.1f} k ha"
    return f"{hectares:.0f} ha"


def format_people(count: float) -> str:
    if count >= 1_000_000.0:
        return f"{count / 1_000_000.0:.1f} M people"
    if count >= 1_000.0:
        return f"{count / 1_000.0:.0f} k people"
    return f"{count:.0f} people"


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    clipped_lat = float(np.clip(lat, -85.05112878, 85.05112878))
    x = WEB_MERCATOR_LIMIT * float(lon) / 180.0
    y = WEB_MERCATOR_LIMIT * math.log(math.tan((90.0 + clipped_lat) * math.pi / 360.0)) / math.pi
    return x, y


def web_mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = x / WEB_MERCATOR_LIMIT * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp(y / WEB_MERCATOR_LIMIT * math.pi)) - math.pi / 2.0)
    return lon, lat


def _osm_boundary_path(cache_dir: Path) -> Path:
    return cache_dir / "osm" / f"california_boundary_osm_r{OSM_RELATION_ID}.geojson"


def _download_osm_boundary(cache_dir: Path, *, force: bool) -> Path:
    dest = _osm_boundary_path(cache_dir)
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(OSM_BOUNDARY_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=180) as response:
        with tempfile.NamedTemporaryFile(
            prefix=dest.stem,
            suffix=dest.suffix,
            dir=str(dest.parent),
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)

    with tmp.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("type") != "FeatureCollection" or not payload.get("features"):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("OSM boundary lookup did not return a GeoJSON feature collection")

    tmp.replace(dest)
    return dest


def _valid_geometry(geometry):
    if geometry.is_empty or geometry.is_valid:
        return geometry
    try:
        from shapely.validation import make_valid

        return make_valid(geometry)
    except Exception:
        return geometry.buffer(0)


def _load_osm_boundary(boundary_path: Path):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    features = payload.get("features") or []
    geoms = [shape(feature["geometry"]) for feature in features if feature.get("geometry")]
    if not geoms:
        raise RuntimeError(f"No geometry found in OSM boundary file: {boundary_path}")
    return _valid_geometry(unary_union(geoms))


def _boundary_to_web_mercator(boundary_wgs84):
    from shapely.ops import transform as shapely_transform

    def project(xs, ys, zs=None):
        lon = np.asarray(xs, dtype=np.float64)
        lat = np.clip(np.asarray(ys, dtype=np.float64), -85.05112878, 85.05112878)
        mx = WEB_MERCATOR_LIMIT * lon / 180.0
        my = WEB_MERCATOR_LIMIT * np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / np.pi
        if zs is None:
            return mx, my
        return mx, my, zs

    return _valid_geometry(shapely_transform(project, boundary_wgs84))


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(float(np.clip(lat, -85.05112878, 85.05112878)))
    n = 2 ** int(zoom)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return int(np.clip(x, 0, n - 1)), int(np.clip(y, 0, n - 1))


def _tile_bounds_mercator(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2 ** int(zoom)
    span = 2.0 * WEB_MERCATOR_LIMIT
    west = -WEB_MERCATOR_LIMIT + span * x / n
    east = -WEB_MERCATOR_LIMIT + span * (x + 1) / n
    north = WEB_MERCATOR_LIMIT - span * y / n
    south = WEB_MERCATOR_LIMIT - span * (y + 1) / n
    return west, south, east, north


def _download_file(url: str, dest: Path, *, force: bool) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
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


def _decode_terrarium(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0


def _resize_float_array(values: np.ndarray, size: tuple[int, int], *, resample: int = Image.Resampling.BICUBIC) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, resample=resample), dtype=np.float32)


def _normalize01(values: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    sample = arr[mask] if mask is not None else arr[np.isfinite(arr)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(sample, 1.0))
    hi = float(np.nanpercentile(sample, 99.3))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _build_terrarium_dem(
    boundary_path: Path,
    cache_dir: Path,
    zoom: int,
    max_size: int,
    *,
    force: bool,
) -> tuple[Path, tuple[float, float, float, float], float]:
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    output = cache_dir / f"california_osm_r{OSM_RELATION_ID}_terrarium_z{zoom}_max{max_size}.tif"
    meta_path = output.with_suffix(".json")
    if output.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return output, tuple(float(v) for v in meta["bounds_mercator"]), float(meta["terrain_span_m"])

    boundary_wgs84 = _load_osm_boundary(boundary_path)
    boundary_mercator = _boundary_to_web_mercator(boundary_wgs84)
    lon_min, lat_min, lon_max, lat_max = boundary_wgs84.bounds
    pad_lon = 0.28
    pad_lat = 0.22
    west, east = lon_min - pad_lon, lon_max + pad_lon
    south, north = lat_min - pad_lat, lat_max + pad_lat

    x0, y0 = _tile_xy(west, north, zoom)
    x1, y1 = _tile_xy(east, south, zoom)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    tile_dir = cache_dir / "terrarium" / str(zoom)
    tile_rows: list[np.ndarray] = []
    tile_count = (x1 - x0 + 1) * (y1 - y0 + 1)
    print(f"[Wildfire Smoke] Using OSM California boundary relation {OSM_RELATION_ID}")
    print(f"[Wildfire Smoke] Loading {tile_count} Terrarium DEM tiles at z{zoom}")
    for ty in range(y0, y1 + 1):
        row_tiles = []
        for tx in range(x0, x1 + 1):
            tile_path = tile_dir / str(tx) / f"{ty}.png"
            url = TERRARIUM_URL.format(z=zoom, x=tx, y=ty)
            _download_file(url, tile_path, force=force)
            row_tiles.append(_decode_terrarium(tile_path))
        tile_rows.append(np.concatenate(row_tiles, axis=1))
    dem = np.concatenate(tile_rows, axis=0).astype(np.float32)

    west_m, _, _, north_m = _tile_bounds_mercator(x0, y0, zoom)
    _, south_m, east_m, _ = _tile_bounds_mercator(x1, y1, zoom)
    bounds = (west_m, south_m, east_m, north_m)
    transform = from_bounds(west_m, south_m, east_m, north_m, dem.shape[1], dem.shape[0])
    mask = geometry_mask(
        [boundary_mercator],
        out_shape=dem.shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    nodata = -9999.0
    dem = np.where(mask, dem, nodata).astype(np.float32)

    if max_size > 0 and max(dem.shape) > max_size:
        scale = max(dem.shape) / float(max_size)
        out_h = max(1, round(dem.shape[0] / scale))
        out_w = max(1, round(dem.shape[1] / scale))
        valid = (dem != nodata).astype(np.float32)
        resized = _resize_float_array(np.where(dem == nodata, 0.0, dem), (out_w, out_h))
        resized_valid = _resize_float_array(valid, (out_w, out_h), resample=Image.Resampling.BILINEAR) > 0.40
        dem = np.where(resized_valid, resized, nodata).astype(np.float32)
        transform = from_bounds(west_m, south_m, east_m, north_m, out_w, out_h)

    terrain_span = max(east_m - west_m, north_m - south_m)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(dem.shape[0]),
        "width": int(dem.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(dem, 1)
    meta_path.write_text(
        json.dumps(
            {
                "bounds_mercator": bounds,
                "terrain_span_m": terrain_span,
                "osm_relation_id": OSM_RELATION_ID,
                "source": "OpenStreetMap boundary; AWS Terrarium DEM",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, bounds, terrain_span


def _build_terrain_overlay(dem_path: Path, cache_dir: Path, *, force: bool) -> Path:
    import rasterio

    output = cache_dir / f"{dem_path.stem}_dark_relief_overlay.png"
    if output.exists() and output.stat().st_mtime_ns >= dem_path.stat().st_mtime_ns and not force:
        return output
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
    valid = np.isfinite(dem) & (dem != nodata)
    fill = float(np.nanpercentile(dem[valid], 5.0)) if np.any(valid) else 0.0
    height = np.where(valid, dem, fill).astype(np.float32)
    norm = _normalize01(height, mask=valid)
    gy, gx = np.gradient(height)
    slope = _normalize01(np.hypot(gx, gy), mask=valid)
    shade = np.clip(0.66 - gx * 0.006 - gy * 0.0025 + slope * 0.18 + norm * 0.28, 0.20, 1.18)
    valley = 1.0 - smoothstep(0.18, 0.46, norm)
    ridge = smoothstep(0.56, 0.92, norm)
    rgb = np.zeros(height.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 13 + 38 * norm + 8 * valley + 10 * ridge
    rgb[..., 1] = 21 + 45 * norm + 9 * valley + 9 * ridge
    rgb[..., 2] = 24 + 48 * norm + 5 * valley + 12 * ridge
    rgb *= shade[..., None]
    rgb[..., 1] += slope * 13.0
    rgb[..., 2] += slope * 18.0
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output)
    return output


def prepare_terrain_assets(cache_dir: Path, zoom: int, max_size: int, *, force: bool) -> TerrainAssets:
    cache_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = _download_osm_boundary(cache_dir, force=force)
    dem_path, bounds, terrain_span = _build_terrarium_dem(
        boundary_path,
        cache_dir,
        int(zoom),
        int(max_size),
        force=force,
    )
    overlay_path = _build_terrain_overlay(dem_path, cache_dir, force=force)
    return TerrainAssets(
        dem_path=dem_path,
        overlay_path=overlay_path,
        boundary_path=boundary_path,
        bounds_mercator=bounds,
        terrain_span_m=terrain_span,
    )


def lonlat_to_uv(lon: float, lat: float) -> tuple[float, float]:
    u = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    v = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN)
    return u, v


def uv_to_lonlat(
    u: np.ndarray | float,
    v: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    lon = LON_MIN + u * (LON_MAX - LON_MIN)
    lat = LAT_MAX - v * (LAT_MAX - LAT_MIN)
    return lon, lat


def terrain_elevation_uv(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray | float:
    lon, lat = uv_to_lonlat(u, v)
    return california_terrain_elevation(lon, lat)


def _lat_window(
    lat: np.ndarray | float,
    south: float,
    north: float,
    feather: float = 0.35,
) -> np.ndarray | float:
    return smoothstep(south, south + feather, lat) * (1.0 - smoothstep(north - feather, north, lat))


def _coast_lon_at_lat(lat: np.ndarray | float) -> np.ndarray | float:
    coast = np.asarray(PACIFIC_COASTLINE, dtype=np.float32)
    order = np.argsort(coast[:, 1])
    return np.interp(lat, coast[order, 1], coast[order, 0])


def california_terrain_elevation(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
) -> np.ndarray | float:
    u = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    v = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN)
    relief = _fbm_like(u, v)

    coast_lon = _coast_lon_at_lat(lat)
    inland = lon - coast_lon
    land_rise = smoothstep(-0.08, 0.32, inland)

    coast_range_axis = 0.58 + 0.16 * np.sin((lat - 34.0) * 2.0)
    coast_ranges = np.exp(-((inland - coast_range_axis) ** 2) / 0.16) * _lat_window(lat, 33.0, 41.8, 0.50)
    north_coast = np.exp(-((inland - 0.46) ** 2) / 0.13) * _lat_window(lat, 38.2, 42.3, 0.45)

    sierra_center = -118.18 - 0.62 * (lat - 36.0)
    sierra = np.exp(-((lon - sierra_center) ** 2) / 0.12) * _lat_window(lat, 35.2, 41.8, 0.42)
    sierra_crest = np.exp(-((lon - (sierra_center + 0.18)) ** 2) / 0.035) * _lat_window(lat, 36.0, 40.8, 0.34)
    sierra_texture = 0.74 + 0.26 * np.sin((lat - 35.5) * 8.5 + (lon + 120.0) * 2.4) ** 2
    sierra *= sierra_texture
    sierra_crest *= 0.80 + 0.20 * np.sin((lat - 36.0) * 12.0) ** 2

    valley_center = -119.30 - 0.64 * (lat - 35.2)
    central_valley = np.exp(-((lon - valley_center) ** 2) / 0.42) * _lat_window(lat, 34.9, 40.5, 0.55)
    sacramento_delta = np.exp(-(((lon + 121.75) / 0.56) ** 2 + ((lat - 38.05) / 0.48) ** 2))

    cascades = np.exp(-((lon + 121.70) ** 2) / 0.18) * _lat_window(lat, 39.4, 42.4, 0.42)
    shasta = np.exp(-(((lon + 122.19) / 0.20) ** 2 + ((lat - 41.41) / 0.20) ** 2))
    lassen = np.exp(-(((lon + 121.51) / 0.18) ** 2 + ((lat - 40.49) / 0.18) ** 2))

    transverse_axis = 34.16 + 0.16 * np.sin((lon + 118.6) * 1.9)
    transverse = np.exp(-((lat - transverse_axis) ** 2) / 0.060) * smoothstep(-120.1, -119.1, lon)
    transverse *= 1.0 - smoothstep(-116.0, -115.3, lon)
    peninsular_axis = -116.45 - 0.36 * (lat - 33.0)
    peninsular = np.exp(-((lon - peninsular_axis) ** 2) / 0.11) * _lat_window(lat, 31.3, 34.1, 0.32)
    mojave = 0.24 * _lat_window(lat, 33.7, 36.4, 0.50) * smoothstep(-118.8, -117.2, lon)

    basin_range = smoothstep(-118.0, -116.6, lon) * _lat_window(lat, 35.0, 41.8, 0.55)
    basin_range *= 0.08 + 0.14 * (np.sin((lon + 117.0) * 3.3 + lat * 1.25) ** 2)

    ocean_shelf = smoothstep(-1.20, -0.05, inland) * (1.0 - smoothstep(-0.02, 0.25, inland))
    ocean_texture = 0.08 * (1.0 - np.clip(inland + 0.20, 0.0, 1.0)) * relief

    elevation = (
        0.12 * relief
        + 0.46 * coast_ranges
        + 0.24 * north_coast
        + 0.64 * sierra
        + 0.27 * sierra_crest
        + 0.38 * cascades
        + 0.68 * shasta
        + 0.52 * lassen
        + 0.34 * transverse
        + 0.40 * peninsular
        + mojave
        + basin_range
        + 0.14 * ocean_shelf
        + ocean_texture
    )
    elevation *= 0.32 + 0.68 * land_rise
    elevation -= 0.34 * central_valley
    elevation -= 0.20 * sacramento_delta
    return np.clip(elevation, 0.0, 1.0)


def _fbm_like(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray | float:
    tex = 0.0
    for octave, amp in enumerate((0.50, 0.25, 0.13, 0.08, 0.04)):
        freq = 2.0 ** octave
        tex += amp * np.sin((u * 7.1 + v * 3.7) * freq + octave * 1.7)
        tex += amp * np.cos((u * 2.4 - v * 8.5) * freq + octave * 0.9)
    return np.clip((tex + 1.60) / 3.20, 0.0, 1.0)


def lonlat_to_map_xy(lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
    u, v = lonlat_to_uv(lon, lat)
    return u * width, v * height


def project_uv_to_screen(
    u: np.ndarray | float,
    v: np.ndarray | float,
    width: int,
    height: int,
    elevation: np.ndarray | float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    x = width * (0.020 + 0.930 * u + 0.050 * (v - 0.5))
    y = height * (0.055 + 0.850 * v + 0.025 * u) - elevation * height * 0.075
    return x, y


def lonlat_to_screen_xy(lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
    u, v = lonlat_to_uv(lon, lat)
    elevation = float(terrain_elevation_uv(u, v))
    x, y = project_uv_to_screen(u, v, width, height, elevation)
    return float(x), float(y)


def project_path(points: tuple[tuple[float, float], ...], width: int, height: int) -> list[tuple[float, float]]:
    return [lonlat_to_map_xy(lon, lat, width, height) for lon, lat in points]


def _draw_geo_polygon(
    draw: ImageDraw.ImageDraw,
    points: tuple[tuple[float, float], ...],
    width: int,
    height: int,
    fill: int | tuple[int, int, int, int],
) -> None:
    draw.polygon(project_path(points, width, height), fill=fill)


def build_land_mask(width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    mainland = (
        *PACIFIC_COASTLINE,
        (LON_MAX + 1.0, LAT_MIN - 0.5),
        (LON_MAX + 1.0, LAT_MAX + 0.5),
        (PACIFIC_COASTLINE[0][0], LAT_MAX + 0.5),
    )
    _draw_geo_polygon(draw, mainland, width, height, 255)

    for island in CHANNEL_ISLANDS:
        _draw_geo_polygon(draw, island, width, height, 255)

    water_cutouts = (
        SAN_FRANCISCO_BAY,
        ((-122.35, 38.18), (-121.78, 38.10), (-121.50, 37.88), (-121.88, 37.76)),
        ((-122.05, 37.66), (-121.85, 37.52), (-121.96, 37.32), (-122.20, 37.43)),
    )
    for polygon in water_cutouts:
        _draw_geo_polygon(draw, polygon, width, height, 0)

    return mask.filter(ImageFilter.GaussianBlur(radius=max(0.7, width / 2600.0)))


def _fbm_texture(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    xn = x / max(width - 1, 1)
    yn = y / max(height - 1, 1)
    tex = np.zeros((height, width), dtype=np.float32)
    for octave, amp in enumerate((0.50, 0.25, 0.13, 0.08, 0.04)):
        freq = 2.0 ** octave
        tex += amp * np.sin((xn * 7.1 + yn * 3.7) * freq + octave * 1.7)
        tex += amp * np.cos((xn * 2.4 - yn * 8.5) * freq + octave * 0.9)
    tex = (tex - tex.min()) / max(float(tex.max() - tex.min()), 1e-6)
    return tex


def _draw_lakes(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    lake_color = (8, 29, 43, 236)
    lake_edge = (38, 86, 102, 145)
    lakes = (
        ((-120.22, 39.28), (-119.94, 39.26), (-119.89, 39.05), (-120.05, 38.88), (-120.25, 39.03)),
        ((-122.98, 39.08), (-122.70, 39.05), (-122.65, 38.94), (-122.92, 38.88)),
        ((-119.10, 38.18), (-118.84, 38.12), (-118.86, 37.92), (-119.08, 37.96)),
        ((-116.05, 33.60), (-115.70, 33.52), (-115.70, 33.20), (-116.03, 33.16), (-116.10, 33.38)),
        SAN_FRANCISCO_BAY,
    )
    for lake in lakes:
        pts = project_path(lake, width, height)
        draw.polygon(pts, fill=lake_color, outline=lake_edge)


def _draw_context_lines(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    state_width = max(1, width // 900)
    subtle_width = max(1, width // 1500)
    draw.line(
        project_path(CALIFORNIA_STATE_OUTLINE, width, height),
        fill=(116, 168, 178, 92),
        width=state_width,
        joint="curve",
    )
    draw.line(
        project_path(((-124.21, 42.00), (-120.00, 42.00)), width, height),
        fill=(100, 145, 158, 62),
        width=subtle_width,
    )
    draw.line(
        project_path(((-120.00, 42.00), (-114.63, 35.00)), width, height),
        fill=(100, 145, 158, 62),
        width=subtle_width,
    )
    draw.line(
        project_path(((-117.12, 32.53), (-114.72, 32.72)), width, height),
        fill=(100, 145, 158, 62),
        width=subtle_width,
    )


def _terrain_background(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    xn = x / max(width - 1, 1)
    yn = y / max(height - 1, 1)
    glow = np.clip(1.0 - np.sqrt(((xn - 0.16) / 0.55) ** 2 + ((yn - 0.40) / 0.80) ** 2), 0.0, 1.0)
    bg = np.dstack(
        (
            2 + 11 * glow,
            10 + 25 * glow,
            17 + 42 * glow,
            np.full((height, width), 255, dtype=np.float32),
        )
    )
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="RGBA")


def render_terrain_surface_3d(texture: Image.Image, elevation: np.ndarray) -> Image.Image:
    width, height = texture.size
    output = _terrain_background(width, height)
    draw = ImageDraw.Draw(output, "RGBA")

    skirt = height * 0.052

    tl = project_uv_to_screen(0.0, 0.0, width, height, 0.0)
    tr = project_uv_to_screen(1.0, 0.0, width, height, 0.0)
    br = project_uv_to_screen(1.0, 1.0, width, height, 0.0)
    bl = project_uv_to_screen(0.0, 1.0, width, height, 0.0)
    tl = (float(tl[0]), float(tl[1]))
    tr = (float(tr[0]), float(tr[1]))
    br = (float(br[0]), float(br[1]))
    bl = (float(bl[0]), float(bl[1]))

    draw.polygon((bl, br, (br[0], br[1] + skirt), (bl[0], bl[1] + skirt)), fill=(4, 13, 18, 235))
    draw.polygon((tl, bl, (bl[0] - width * 0.012, bl[1] + skirt * 0.60), (tl[0] - width * 0.010, tl[1] + skirt * 0.35)), fill=(5, 19, 27, 210))
    draw.polygon((tr, br, (br[0] + width * 0.010, br[1] + skirt * 0.55), (tr[0] + width * 0.010, tr[1] + skirt * 0.30)), fill=(2, 8, 13, 220))

    src_w, src_h = texture.size
    a = 0.930 * width / max(src_w - 1, 1)
    b = 0.050 * width / max(src_h - 1, 1)
    c = width * (0.020 - 0.025)
    d = 0.025 * height / max(src_w - 1, 1)
    e = 0.850 * height / max(src_h - 1, 1)
    f = height * 0.055
    det = a * e - b * d
    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det
    inv_c = -(inv_a * c + inv_b * f)
    inv_f = -(inv_d * c + inv_e * f)

    shade = np.gradient(elevation.astype(np.float32))
    shade_arr = np.clip(0.83 + elevation * 0.34 - shade[1] * 1.65 - shade[0] * 0.60, 0.40, 1.30)
    shade_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    shade_rgba[..., 0:3] = np.clip((shade_arr[..., None] - 1.0) * 255.0 + 128.0, 0, 255).astype(np.uint8)
    shade_rgba[..., 3] = np.clip(np.abs(shade_arr - 1.0) * 115.0, 0, 95).astype(np.uint8)
    relief_layer = Image.fromarray(shade_rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=0.35))

    surface_texture = Image.alpha_composite(texture.convert("RGBA"), relief_layer)
    warped = surface_texture.transform(
        (width, height),
        Image.Transform.AFFINE,
        (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow.putalpha(warped.getchannel("A").filter(ImageFilter.GaussianBlur(radius=max(6.0, height / 110.0))).point(lambda v: int(v * 0.22)))
    shadow_crop = shadow.crop((0, 0, max(1, width - int(width * 0.018)), max(1, height - int(height * 0.035))))
    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_layer.paste(shadow_crop, (int(width * 0.018), int(height * 0.035)), shadow_crop)
    output.alpha_composite(shadow_layer)
    output.alpha_composite(warped)

    edge = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge, "RGBA")
    edge_draw.line((tl, tr), fill=(122, 169, 178, 38), width=max(1, width // 960))
    edge_draw.line((bl, br), fill=(35, 78, 91, 62), width=max(1, width // 900))
    edge_draw.line((tl, bl), fill=(35, 78, 91, 42), width=max(1, width // 1050))
    edge_draw.line((tr, br), fill=(23, 54, 69, 38), width=max(1, width // 1050))
    output.alpha_composite(edge.filter(ImageFilter.GaussianBlur(radius=0.4)))
    return output


def build_synthetic_base_map(width: int, height: int) -> Image.Image:
    land_mask = build_land_mask(width, height)
    mask = np.asarray(land_mask, dtype=np.float32) / 255.0
    y, x = np.mgrid[0:height, 0:width]
    xn = x / max(width - 1, 1)
    yn = y / max(height - 1, 1)
    lon, lat = uv_to_lonlat(xn, yn)

    relief = _fbm_texture(width, height)
    elevation = california_terrain_elevation(lon, lat).astype(np.float32)
    coast_lon = _coast_lon_at_lat(lat)
    inland = lon - coast_lon
    shelf = np.clip(1.0 - smoothstep(-1.25, 0.12, inland), 0.0, 1.0)
    water = np.dstack(
        (
            3 + 8 * (1.0 - xn) + 9 * shelf,
            15 + 25 * (1.0 - xn) + 12 * shelf + 4 * yn,
            26 + 34 * (1.0 - xn) + 20 * shelf,
        )
    )

    grad_y, grad_x = np.gradient(elevation)
    hillshade = np.clip(0.78 - grad_x * 9.5 - grad_y * 4.4 + elevation * 0.34, 0.38, 1.36)
    valley = np.exp(-(((lon + 120.55) / 0.68) ** 2 + ((lat - 37.6) / 2.2) ** 2))
    desert = smoothstep(-118.2, -116.4, lon) * smoothstep(32.7, 35.7, lat)
    desert *= 1.0 - smoothstep(36.9, 39.5, lat)

    land_base = np.dstack(
        (
            16 + 46 * elevation + 14 * desert + 5 * relief,
            24 + 48 * elevation + 10 * desert + 6 * relief - 9 * valley,
            26 + 49 * elevation + 3 * desert + 6 * relief - 12 * valley,
        )
    )
    land_base *= hillshade[..., None]

    terrain = water * (1.0 - mask[..., None]) + land_base * mask[..., None]

    day_glow = np.clip(1.0 - smoothstep(0.02, 0.55, xn), 0.0, 1.0)
    night = smoothstep(0.52, 1.0, xn)
    vertical_shade = 0.88 + 0.18 * np.cos((yn - 0.46) * math.pi)
    terrain *= vertical_shade[..., None] * (1.02 - 0.38 * night[..., None])
    terrain += np.dstack((9 * day_glow, 33 * day_glow, 51 * day_glow))

    vignette = np.sqrt(((xn - 0.52) / 0.78) ** 2 + ((yn - 0.52) / 0.92) ** 2)
    terrain *= (1.06 - 0.42 * np.clip(vignette, 0.0, 1.0))[..., None]

    image = Image.fromarray(np.clip(terrain, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    _draw_lakes(image)

    coast = ImageChops.subtract(
        land_mask.filter(ImageFilter.MaxFilter(max(3, int(width / 420) | 1))),
        land_mask.filter(ImageFilter.MinFilter(max(3, int(width / 420) | 1))),
    )
    coast_rgba = Image.new("RGBA", (width, height), (98, 142, 154, 0))
    coast_rgba.putalpha(coast.point(lambda v: int(v * 0.36)))
    image.alpha_composite(coast_rgba)
    _draw_context_lines(image)

    terminator = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    tarr = np.zeros((height, width, 4), dtype=np.uint8)
    shadow = (smoothstep(0.62, 1.0, xn) * 120).astype(np.uint8)
    tarr[..., 0] = 2
    tarr[..., 1] = 7
    tarr[..., 2] = 12
    tarr[..., 3] = shadow
    terminator = Image.fromarray(tarr, mode="RGBA")
    image.alpha_composite(terminator)

    surface_elevation = np.clip(0.08 * relief + elevation * (0.14 + 0.86 * mask), 0.0, 1.0)
    surface_img = Image.fromarray((surface_elevation * 255.0).astype(np.uint8), mode="L")
    surface_img = surface_img.filter(ImageFilter.GaussianBlur(radius=max(1.0, width / 720.0)))
    surface_elevation = np.asarray(surface_img, dtype=np.float32) / 255.0
    return render_terrain_surface_3d(image, surface_elevation)


def _make_terrain_camera(
    target: np.ndarray,
    *,
    radius: float,
    phi_deg: float,
    theta_deg: float,
    fov_deg: float,
) -> TerrainCamera:
    phi = math.radians(phi_deg)
    theta = math.radians(theta_deg)
    eye = target + np.array(
        [
            radius * math.sin(theta) * math.cos(phi),
            radius * math.cos(theta),
            radius * math.sin(theta) * math.sin(phi),
        ],
        dtype=np.float32,
    )
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    side = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    side /= max(float(np.linalg.norm(side)), 1e-6)
    up = np.cross(side, forward)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    return TerrainCamera(target=target, eye=eye, forward=forward, side=side, up=up, fov_deg=fov_deg)


def _make_north_up_terrain_camera(
    target: np.ndarray,
    *,
    radius: float,
    pitch_deg: float,
    fov_deg: float,
) -> TerrainCamera:
    pitch = math.radians(float(pitch_deg))
    forward = np.array([0.0, -math.sin(pitch), math.cos(pitch)], dtype=np.float32)
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.cross(forward, side)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    eye = target - forward * float(radius)
    return TerrainCamera(target=target, eye=eye, forward=forward, side=side, up=up, fov_deg=fov_deg)


def _background_3d(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    xn = x / max(width - 1, 1)
    yn = y / max(height - 1, 1)
    ocean_glow = np.clip(1.0 - np.sqrt(((xn - 0.11) / 0.55) ** 2 + ((yn - 0.57) / 0.74) ** 2), 0.0, 1.0)
    top_glow = np.clip(1.0 - np.sqrt(((xn - 0.36) / 0.95) ** 2 + ((yn - 0.02) / 0.40) ** 2), 0.0, 1.0)
    bg = np.dstack(
        (
            2 + 10 * ocean_glow + 4 * top_glow,
            10 + 30 * ocean_glow + 8 * top_glow,
            17 + 48 * ocean_glow + 12 * top_glow,
            np.full((height, width), 255, dtype=np.float32),
        )
    )
    vignette = np.sqrt(((xn - 0.52) / 0.82) ** 2 + ((yn - 0.50) / 0.92) ** 2)
    bg[..., :3] *= (1.04 - 0.40 * np.clip(vignette, 0.0, 1.0))[..., None]
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="RGBA")


def _read_dem_and_overlay(assets: TerrainAssets) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    import rasterio

    with rasterio.open(assets.dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
    dem = np.where(dem == nodata, np.nan, dem).astype(np.float32)
    overlay = np.asarray(Image.open(assets.overlay_path).convert("RGBA"), dtype=np.uint8)
    if overlay.shape[:2] != dem.shape:
        overlay = np.asarray(
            Image.fromarray(overlay, mode="RGBA").resize((dem.shape[1], dem.shape[0]), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
    return dem, overlay, bounds


def _project_points(points: np.ndarray, camera: TerrainCamera, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = points - camera.eye.reshape(1, 1, 3)
    depth = (
        rel[..., 0] * camera.forward[0]
        + rel[..., 1] * camera.forward[1]
        + rel[..., 2] * camera.forward[2]
    )
    x_cam = rel[..., 0] * camera.side[0] + rel[..., 1] * camera.side[1] + rel[..., 2] * camera.side[2]
    y_cam = rel[..., 0] * camera.up[0] + rel[..., 1] * camera.up[1] + rel[..., 2] * camera.up[2]
    focal = 1.0 / math.tan(math.radians(camera.fov_deg) * 0.5)
    aspect = width / max(float(height), 1.0)
    denom = np.maximum(depth, 1e-3)
    sx = (x_cam * focal / aspect / denom) * 0.5 + 0.5
    sy = 0.5 - (y_cam * focal / denom) * 0.5
    return sx * width, sy * height, depth


def _iter_boundary_lines(geometry):
    if geometry.geom_type == "Polygon":
        yield list(geometry.exterior.coords)
        for interior in geometry.interiors:
            yield list(interior.coords)
    elif geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            yield from _iter_boundary_lines(part)
    elif geometry.geom_type in {"LineString", "LinearRing"}:
        yield list(geometry.coords)
    elif geometry.geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _iter_boundary_lines(part)


def render_cpu_terrain_base(assets: TerrainAssets, width: int, height: int) -> tuple[Image.Image, TerrainProjector]:
    dem, overlay, bounds = _read_dem_and_overlay(assets)
    valid = np.isfinite(dem)
    if not np.any(valid):
        raise RuntimeError(f"California DEM contains no finite terrain cells: {assets.dem_path}")

    west, south, east, north = bounds
    span_x = east - west
    span_y = north - south
    world_scale = 1000.0 / max(span_x, span_y, 1.0)
    vertical_exaggeration = 34.0
    center_x = (west + east) * 0.5
    center_y = (south + north) * 0.5

    target_lon, target_lat = -119.55, 37.15
    target_mx, target_my = lonlat_to_web_mercator(target_lon, target_lat)
    target_elev = float(np.nanpercentile(dem[valid], 45.0))
    target = np.array(
        [
            (target_mx - center_x) * world_scale,
            target_elev * world_scale * vertical_exaggeration * 0.40,
            (target_my - center_y) * world_scale,
        ],
        dtype=np.float32,
    )
    camera = _make_north_up_terrain_camera(
        target,
        radius=1750.0,
        pitch_deg=57.0,
        fov_deg=31.0,
    )
    projector = TerrainProjector(
        heightmap=dem,
        valid_mask=valid,
        bounds_mercator=bounds,
        world_scale=world_scale,
        vertical_exaggeration=vertical_exaggeration,
        camera=camera,
    )

    dem_h, dem_w = dem.shape
    if dem_h >= dem_w:
        mesh_rows = min(TERRAIN_MESH_ROWS, dem_h)
        mesh_cols = max(8, min(TERRAIN_MESH_COLS, round(mesh_rows * dem_w / max(dem_h, 1))))
    else:
        mesh_cols = min(TERRAIN_MESH_COLS, dem_w)
        mesh_rows = max(8, min(TERRAIN_MESH_ROWS, round(mesh_cols * dem_h / max(dem_w, 1))))

    fill_value = float(np.nanpercentile(dem[valid], 3.0))
    mesh_elev = _resize_float_array(np.where(valid, dem, fill_value), (mesh_cols, mesh_rows))
    mesh_valid = _resize_float_array(valid.astype(np.float32), (mesh_cols, mesh_rows), resample=Image.Resampling.BILINEAR) > 0.44
    mesh_rgba = np.asarray(
        Image.fromarray(overlay, mode="RGBA").resize((mesh_cols, mesh_rows), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    ).copy()
    mesh_rgba[..., 3] = np.where(mesh_valid, mesh_rgba[..., 3], 0).astype(np.uint8)

    xs = np.linspace(west, east, mesh_cols, dtype=np.float32)
    ys = np.linspace(north, south, mesh_rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.zeros((mesh_rows, mesh_cols, 3), dtype=np.float32)
    points[..., 0] = (grid_x - center_x) * world_scale
    points[..., 1] = np.maximum(mesh_elev, 0.0) * world_scale * vertical_exaggeration
    points[..., 2] = (grid_y - center_y) * world_scale
    screen_x, screen_y, depth = _project_points(points, camera, width, height)

    dy, dx = np.gradient(np.where(mesh_valid, mesh_elev, fill_value).astype(np.float32))
    normal_x = -dx * world_scale * vertical_exaggeration
    normal_y = np.ones_like(normal_x, dtype=np.float32) * 5.5
    normal_z = dy * world_scale * vertical_exaggeration
    normal_len = np.maximum(np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z), 1e-6)
    normal_x /= normal_len
    normal_y /= normal_len
    normal_z /= normal_len
    light = np.array([-0.42, 0.62, -0.66], dtype=np.float32)
    light /= max(float(np.linalg.norm(light)), 1e-6)
    shade = np.clip(normal_x * light[0] + normal_y * light[1] + normal_z * light[2], -0.45, 1.0)
    shade = np.clip(0.50 + 0.62 * shade, 0.24, 1.18)

    image = _background_3d(width, height)
    terrain_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(terrain_layer, "RGBA")
    quads: list[tuple[float, tuple[tuple[float, float], ...], tuple[int, int, int, int]]] = []
    for row in range(mesh_rows - 1):
        for col in range(mesh_cols - 1):
            cell_alpha = mesh_rgba[row : row + 2, col : col + 2, 3]
            if float(np.mean(cell_alpha)) < 70.0:
                continue
            cell_depth = depth[row : row + 2, col : col + 2]
            if float(np.min(cell_depth)) <= 0.0:
                continue
            pts = (
                (float(screen_x[row, col]), float(screen_y[row, col])),
                (float(screen_x[row, col + 1]), float(screen_y[row, col + 1])),
                (float(screen_x[row + 1, col + 1]), float(screen_y[row + 1, col + 1])),
                (float(screen_x[row + 1, col]), float(screen_y[row + 1, col])),
            )
            if max(p[0] for p in pts) < -80 or min(p[0] for p in pts) > width + 80:
                continue
            if max(p[1] for p in pts) < -80 or min(p[1] for p in pts) > height + 80:
                continue
            base = np.mean(mesh_rgba[row : row + 2, col : col + 2, :3], axis=(0, 1)).astype(np.float32)
            cell_shade = float(np.mean(shade[row : row + 2, col : col + 2]))
            rgb = np.clip(base * cell_shade, 0, 255).astype(np.uint8)
            alpha = int(np.clip(np.mean(cell_alpha) * 1.04, 0, 255))
            quads.append((float(np.mean(cell_depth)), pts, (int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)))

    for _, pts, fill in sorted(quads, key=lambda item: item[0], reverse=True):
        draw.polygon(pts, fill=fill)

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_alpha = terrain_layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius=max(7, height // 100)))
    shadow.putalpha(shadow_alpha.point(lambda value: int(value * 0.22)))
    shadow_crop = shadow.crop((0, 0, max(1, width - width // 70), max(1, height - height // 36)))
    image.alpha_composite(shadow_crop, dest=(width // 70, height // 36))
    image.alpha_composite(terrain_layer)
    projector.screen_mask = terrain_layer.getchannel("A")

    boundary = _load_osm_boundary(assets.boundary_path)
    boundary_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    boundary_draw = ImageDraw.Draw(boundary_layer, "RGBA")
    for line in _iter_boundary_lines(boundary):
        if len(line) < 2:
            continue
        pts = [projector.lonlat_to_screen(float(lon), float(lat), width, height) for lon, lat, *_ in line[:: max(1, len(line) // 900)]]
        if len(pts) >= 2:
            boundary_draw.line(pts, fill=(102, 163, 178, 145), width=max(1, width // 760), joint="curve")
    image.alpha_composite(boundary_layer.filter(ImageFilter.GaussianBlur(radius=0.25)))

    return image, projector


def build_base_map(
    width: int,
    height: int,
    *,
    cache_dir: Path,
    dem_zoom: int,
    max_dem_size: int,
    force: bool,
) -> tuple[Image.Image, TerrainProjector]:
    assets = prepare_terrain_assets(cache_dir, int(dem_zoom), int(max_dem_size), force=force)
    return render_cpu_terrain_base(assets, width, height)


def frame_datetime(frame_index: int, total_frames: int) -> dt.datetime:
    _, season_day = season_progress(frame_index, total_frames)
    return dt.datetime.combine(START_DATE, dt.time(18, 0)) + dt.timedelta(days=season_day)


def _fallback_wind_from_angle(
    angle_deg: float,
    seed: int,
    t: float,
    lon: float | None = None,
    lat: float | None = None,
) -> tuple[float, float]:
    """Generate a deterministic synoptic-scale fallback wind field."""
    base_angle = math.radians(angle_deg)
    lon_term = 0.0 if lon is None else float(lon) + 121.0
    lat_term = 0.0 if lat is None else float(lat) - 37.4
    synoptic_sweep = 24.0 * math.sin(t * math.tau * 0.22 + seed * 0.17 + lat_term * 0.34)
    regional_shear = 13.0 * math.sin(lon_term * 0.55 - t * math.tau * 0.15 + seed * 0.09)
    ridge_bend = 7.0 * math.sin((lon_term - lat_term) * 0.48 + t * math.tau * 0.42)
    angle = base_angle + math.radians(synoptic_sweep + regional_shear + ridge_bend)
    speed_mps = 10.5 + 3.8 * math.sin(t * math.tau * 0.34 + seed * 0.13 + lon_term * 0.25)
    speed_mps += 1.4 * math.cos(lat_term * 0.8 - t * math.tau * 0.18)
    speed_mps = float(np.clip(speed_mps, 7.0, 16.0))
    return math.cos(angle) * speed_mps, math.sin(angle) * speed_mps


def _sample_smoke_wind(
    wind_field: WindField | None,
    lon: float,
    lat: float,
    when: dt.datetime,
    *,
    height_m: float,
    fallback_angle_deg: float,
    seed: int,
    t: float,
) -> tuple[float, float]:
    if wind_field is not None:
        try:
            u_mps, v_mps = wind_field.sample(lon, lat, when, height_m=height_m)
            if np.isfinite(u_mps) and np.isfinite(v_mps) and math.hypot(u_mps, v_mps) > 0.05:
                return float(u_mps), float(v_mps)
        except Exception:
            pass
    return _fallback_wind_from_angle(fallback_angle_deg, seed, t, lon, lat)


def _sample_smoothed_smoke_wind(
    wind_field: WindField | None,
    lon: float,
    lat: float,
    when: dt.datetime,
    *,
    height_m: float,
    fallback_angle_deg: float,
    seed: int,
    t: float,
) -> tuple[float, float]:
    if wind_field is None:
        return _fallback_wind_from_angle(fallback_angle_deg, seed, t, lon, lat)

    def average_at(sample_lon: float, sample_lat: float) -> tuple[float, float] | None:
        values: list[tuple[float, float]] = []
        for hour_offset in SMOKE_WIND_SMOOTH_HOURS:
            sample_time = when + dt.timedelta(hours=hour_offset)
            try:
                u_mps, v_mps = wind_field.sample(sample_lon, sample_lat, sample_time, height_m=height_m)
            except Exception:
                continue
            if np.isfinite(u_mps) and np.isfinite(v_mps):
                values.append((float(u_mps), float(v_mps)))
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float32)
        return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

    local = average_at(lon, lat)
    regional = average_at(SMOKE_REGIONAL_ANCHOR[0], SMOKE_REGIONAL_ANCHOR[1])
    if local is None and regional is None:
        return _fallback_wind_from_angle(fallback_angle_deg, seed, t, lon, lat)
    if local is None:
        u_mps, v_mps = regional
    elif regional is None:
        u_mps, v_mps = local
    else:
        blend = SMOKE_REGIONAL_BLEND
        u_mps = local[0] * (1.0 - blend) + regional[0] * blend
        v_mps = local[1] * (1.0 - blend) + regional[1] * blend
    if math.hypot(u_mps, v_mps) <= 0.05:
        return _fallback_wind_from_angle(fallback_angle_deg, seed, t, lon, lat)
    return float(u_mps), float(v_mps)


def _screen_direction_from_wind(
    projector: TerrainProjector,
    lon: float,
    lat: float,
    width: int,
    height: int,
    u_mps: float,
    v_mps: float,
) -> np.ndarray:
    lon2, lat2 = move_lonlat_by_meters(lon, lat, u_mps * 3600.0, v_mps * 3600.0)
    x1, y1 = projector.lonlat_to_screen(lon, lat, width, height)
    x2, y2 = projector.lonlat_to_screen(lon2, lat2, width, height)
    direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length <= 1e-5:
        direction = np.array([u_mps, -v_mps], dtype=np.float32)
        length = float(np.linalg.norm(direction))
    return direction / max(length, 1e-6)


def _advect_smoke_lonlat(
    lon: float,
    lat: float,
    age_hours: float,
    when: dt.datetime,
    wind_field: WindField | None,
    *,
    height_m: float,
    speed_scale: float,
    fallback_angle_deg: float,
    seed: int,
    t: float,
) -> tuple[float, float]:
    """Advect smoke with large-scale curling arcs (hook/comma/S-shapes)."""
    if age_hours <= 0.0:
        return lon, lat

    steps = max(6, min(28, int(math.ceil(age_hours / 1.35))))
    lon_p = float(lon)
    lat_p = float(lat)
    step_hours = age_hours / steps

    rng = np.random.default_rng(seed)
    curl_amplitude = rng.uniform(2.4, 6.4)
    curl_frequency = rng.uniform(0.55, 1.25)
    curl_phase = rng.uniform(0.0, math.tau)
    lateral_bias = rng.uniform(-1.1, 1.1)
    secondary_phase = rng.uniform(0.0, math.tau)
    shear_bias = rng.uniform(-0.75, 0.75)

    for step in range(steps):
        frac = (step + 0.5) / steps
        sample_time = when - dt.timedelta(hours=age_hours * (1.0 - frac))
        u_mps, v_mps = _sample_smoke_wind(
            wind_field,
            lon_p,
            lat_p,
            sample_time,
            height_m=height_m,
            fallback_angle_deg=fallback_angle_deg,
            seed=seed,
            t=t + frac * 0.02,
        )

        speed = max(math.hypot(u_mps, v_mps), 1e-6)
        cross_u = -v_mps / speed
        cross_v = u_mps / speed

        maturity = float(smoothstep(0.10, 0.88, frac))
        curl_strength = curl_amplitude * math.sin(curl_phase + frac * math.pi * curl_frequency)
        comma_hook = 3.0 * math.sin(secondary_phase + frac * math.pi * 1.65) * smoothstep(0.35, 1.0, frac)
        s_bend = lateral_bias * math.sin((frac - 0.18) * math.pi * 1.18) * (0.55 + 0.95 * frac)
        shear = shear_bias * (frac ** 1.35)
        lateral_mps = (curl_strength * maturity + comma_hook + s_bend + shear) * 1.55

        u_mps += cross_u * lateral_mps
        v_mps += cross_v * lateral_mps

        range_boost = SMOKE_ADVECTION_RANGE_BOOST * (1.0 + 0.32 * smoothstep(0.18, 0.82, frac))
        dt_seconds = step_hours * 3600.0 * max(float(speed_scale), 0.0) * range_boost
        lon_p, lat_p = move_lonlat_by_meters(lon_p, lat_p, u_mps * dt_seconds, v_mps * dt_seconds)

    return lon_p, lat_p


def _smoke_particle_age_hours(
    fire_age_hours: float,
    phase_hours: float,
    max_age_hours: float = SMOKE_PARTICLE_MAX_AGE_HOURS,
) -> float | None:
    if fire_age_hours <= 0.0:
        return None
    max_age = max(float(max_age_hours), 1e-6)
    phase = float(phase_hours) % max_age
    if fire_age_hours < max_age and phase > fire_age_hours:
        return None
    return float((fire_age_hours - phase) % max_age)


def _smoke_lifecycle_alpha(age_hours: float, max_age_hours: float = SMOKE_PARTICLE_MAX_AGE_HOURS) -> float:
    """Lifecycle alpha: dense bloom near source, gradual fade to diffuse haze."""
    age_frac = float(np.clip(age_hours / max(float(max_age_hours), 1e-6), 0.0, 1.0))
    # Quick emergence
    birth = float(smoothstep(0.0, 0.06, age_frac))
    # Peak density in early-mid travel
    peak = 1.0 - 0.3 * float(smoothstep(0.0, 0.3, age_frac))
    # Long gradual dissipation into haze
    dissipation = 1.0 - float(smoothstep(0.15, 1.0, age_frac)) ** 0.6
    return float(np.clip(birth * peak * dissipation, 0.0, 1.0))


def _smoke_noise_field(width: int, height: int, t: float, seed: int) -> np.ndarray:
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    xn = x_coords / max(width - 1, 1)
    yn = y_coords / max(height - 1, 1)
    phase = seed * 0.011
    field = np.zeros((height, width), dtype=np.float32)
    for octave, amp in enumerate((0.55, 0.28, 0.13, 0.07)):
        freq = 2.0 ** octave
        field += amp * np.sin((xn * 2.1 + yn * 1.3) * math.tau * freq + phase + t * math.tau * (0.05 + 0.03 * octave))
        field += amp * np.cos((xn * 1.1 - yn * 2.4) * math.tau * freq + phase * 1.7 - t * math.tau * (0.04 + 0.02 * octave))
    lo = float(field.min())
    hi = float(field.max())
    return ((field - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def _roll_distort_smoke_field(field: np.ndarray, t: float, seed: int) -> np.ndarray:
    if field.size == 0:
        return field
    height, width = field.shape
    phase = seed * 0.013
    row_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)
    col_axis = np.linspace(0.0, 1.0, width, dtype=np.float32)
    max_x = max(1, int(round(width * 0.055)))
    max_y = max(1, int(round(height * 0.050)))
    row_offsets = np.round(
        (
            0.68 * np.sin(row_axis * math.tau * 1.25 + t * math.tau * 0.17 + phase)
            + 0.32 * np.sin(row_axis * math.tau * 3.10 - t * math.tau * 0.11 + phase * 1.8)
        )
        * max_x
    ).astype(np.int32)
    col_offsets = np.round(
        (
            0.62 * np.cos(col_axis * math.tau * 1.05 - t * math.tau * 0.13 + phase * 0.7)
            + 0.38 * np.sin(col_axis * math.tau * 2.65 + t * math.tau * 0.19 + phase * 1.4)
        )
        * max_y
    ).astype(np.int32)

    warped = np.empty_like(field)
    for row in range(height):
        warped[row, :] = np.roll(field[row, :], int(row_offsets[row]))
    curled = np.empty_like(field)
    for col in range(width):
        curled[:, col] = np.roll(warped[:, col], int(col_offsets[col]))
    return curled


def _build_smoke_veil_layer(
    width: int,
    height: int,
    frame_index: int,
    total_frames: int,
    projector: TerrainProjector,
    wind_field: WindField | None,
    *,
    wind_height_m: float,
    wind_speed_scale: float,
) -> Image.Image:
    """Build a low-resolution atmospheric veil layer with broad smoke sheets."""
    vw = SMOKE_VEIL_RESOLUTION
    vh = max(1, int(round(vw * height / max(width, 1))))
    veil = np.zeros((vh, vw), dtype=np.float32)

    _, season_day = season_progress(frame_index, total_frames)
    t = frame_index / max(total_frames - 1, 1)
    frame_when = frame_datetime(frame_index, total_frames)

    for cluster in FIRE_CLUSTERS:
        progress = cluster_progress(cluster, season_day)
        if progress <= 0.02:
            continue

        fire_age_hours = max(0.0, (season_day - cluster.start_day) * 24.0)
        if fire_age_hours <= 0.0:
            continue

        u_mps, v_mps = _sample_smoothed_smoke_wind(
            wind_field,
            cluster.lon,
            cluster.lat,
            frame_when,
            height_m=wind_height_m,
            fallback_angle_deg=cluster.wind_angle_deg,
            seed=cluster.seed,
            t=t,
        )
        wind_speed = max(math.hypot(u_mps, v_mps), 1e-6)
        wind_unit = np.array([u_mps / wind_speed, v_mps / wind_speed], dtype=np.float32)
        cross_unit = np.array([-wind_unit[1], wind_unit[0]], dtype=np.float32)
        source_radius_m = float(
            np.clip(
                cluster.spread_km * 76.0 * math.sqrt(progress),
                SMOKE_SOURCE_MIN_RADIUS_M,
                SMOKE_SOURCE_MAX_RADIUS_M,
            )
        )

        source_layer = Image.new("L", (vw, vh), 0)
        source_draw = ImageDraw.Draw(source_layer)
        source_x, source_y = projector.lonlat_to_screen(cluster.lon, cluster.lat, vw, vh)
        bloom_radius = max(1.8, (2.2 + cluster.spread_km / 120.0) * math.sqrt(progress) * vw / 192.0)
        for ring in range(3):
            radius = bloom_radius * (1.0 + ring * 0.85)
            alpha = int((22 - ring * 5) * progress)
            if alpha > 0:
                source_draw.ellipse(
                    (source_x - radius, source_y - radius, source_x + radius, source_y + radius),
                    fill=alpha,
                )
        veil = np.clip(veil + np.asarray(source_layer.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0, 0.0, 1.0)

        for ribbon_idx in range(SMOKE_RIBBON_COUNT):
            ribbon_seed = cluster.seed * 1000 + ribbon_idx
            rng = np.random.default_rng(cluster.seed * 5009 + ribbon_idx * 131)
            phase_jitter = rng.uniform(-0.35, 0.35)
            phase_hours = ((ribbon_idx + phase_jitter) / max(SMOKE_RIBBON_COUNT, 1)) * SMOKE_VEIL_MAX_AGE_HOURS
            ribbon_age = _smoke_particle_age_hours(
                fire_age_hours,
                phase_hours,
                max_age_hours=SMOKE_VEIL_MAX_AGE_HOURS,
            )
            if ribbon_age is None or ribbon_age < 1.0:
                continue

            age_frac = ribbon_age / SMOKE_VEIL_MAX_AGE_HOURS
            source_cross_m = float(np.clip(rng.normal(0.0, source_radius_m * 0.45), -source_radius_m, source_radius_m))
            source_along_m = float(np.clip(rng.normal(0.0, source_radius_m * 0.22), -source_radius_m * 0.55, source_radius_m * 0.55))
            source_lon, source_lat = move_lonlat_by_meters(
                cluster.lon,
                cluster.lat,
                wind_unit[0] * source_along_m + cross_unit[0] * source_cross_m,
                wind_unit[1] * source_along_m + cross_unit[1] * source_cross_m,
            )
            lane_offset_m = rng.uniform(-0.95, 0.95) * (34_000.0 + cluster.spread_km * 85.0)
            lane_phase = rng.uniform(0.0, math.tau)
            lane_wave_m = 13_000.0 + cluster.spread_km * 54.0
            ribbon_points = max(10, min(26, int(10 + age_frac * 17)))
            points: list[tuple[float, float]] = []

            for pi in range(ribbon_points):
                point_frac = pi / max(ribbon_points - 1, 1)
                sample_age = ribbon_age * point_frac
                lon_p, lat_p = _advect_smoke_lonlat(
                    source_lon,
                    source_lat,
                    sample_age,
                    frame_when,
                    wind_field,
                    height_m=wind_height_m,
                    speed_scale=wind_speed_scale,
                    fallback_angle_deg=cluster.wind_angle_deg,
                    seed=ribbon_seed,
                    t=t,
                )

                spread_m = lane_offset_m * (point_frac ** 1.10)
                spread_m += math.sin(lane_phase + point_frac * math.pi * 1.55) * lane_wave_m * point_frac
                lon_p, lat_p = move_lonlat_by_meters(
                    lon_p,
                    lat_p,
                    cross_unit[0] * spread_m,
                    cross_unit[1] * spread_m,
                )
                sx, sy = projector.lonlat_to_screen(lon_p, lat_p, vw, vh)
                points.append((float(sx), float(sy)))

            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            margin = max(vw, vh) * 0.18
            if max(xs) < -margin or min(xs) > vw + margin or max(ys) < -margin or min(ys) > vh + margin:
                continue

            lane_img = Image.new("L", (vw, vh), 0)
            lane_draw = ImageDraw.Draw(lane_img)
            lane_lifecycle = _smoke_lifecycle_alpha(ribbon_age, max_age_hours=SMOKE_VEIL_MAX_AGE_HOURS)
            lane_width = (5.4 + 9.8 * math.sqrt(progress)) * rng.uniform(0.78, 1.28)
            lane_width *= 1.0 - 0.22 * age_frac
            lane_alpha = (52.0 + 42.0 * progress) * lane_lifecycle * rng.uniform(0.80, 1.16)

            for point_index in range(len(points) - 1):
                seg_frac = point_index / max(len(points) - 1, 1)
                width_mult = 0.98 - 0.78 * seg_frac + 0.08 * math.sin(lane_phase + seg_frac * math.tau)
                alpha_mult = 1.0 - 0.70 * seg_frac
                alpha_mult *= 1.0 - 0.58 * smoothstep(0.58, 1.0, seg_frac)
                seg_width = max(1, int(round(lane_width * max(width_mult, 0.20))))
                seg_alpha = int(np.clip(lane_alpha * alpha_mult, 0, 120))
                if seg_alpha > 0:
                    lane_draw.line(
                        [points[point_index], points[point_index + 1]],
                        fill=seg_alpha,
                        width=seg_width,
                    )

            blur_radius = max(0.8, lane_width * 0.22)
            lane_arr = np.asarray(lane_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32) / 255.0
            veil = np.clip(veil + lane_arr * 0.95, 0.0, 1.0)

    if float(np.max(veil)) <= 1e-5:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    veil = _roll_distort_smoke_field(veil, t, 1301)
    noise = _smoke_noise_field(vw, vh, t, 7919)
    veil = np.clip(veil * (0.70 + 0.43 * noise), 0.0, 1.0)

    veil_img = Image.fromarray((veil * 255).astype(np.uint8), mode="L")
    fine = np.asarray(veil_img.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32) / 255.0
    medium = np.asarray(veil_img.filter(ImageFilter.GaussianBlur(radius=SMOKE_VEIL_BLUR_RADIUS * 0.62)), dtype=np.float32) / 255.0
    broad = np.asarray(veil_img.filter(ImageFilter.GaussianBlur(radius=SMOKE_VEIL_BLUR_RADIUS * 1.35)), dtype=np.float32) / 255.0
    veil_arr = np.clip(fine * 0.72 + medium * 0.36 + broad * 0.05, 0.0, 1.0)
    veil_arr = np.clip(veil_arr ** 1.22, 0.0, 1.0)

    veil_img = Image.fromarray((veil_arr * 255).astype(np.uint8), mode="L")
    veil_img = veil_img.resize((width, height), Image.Resampling.BICUBIC)
    veil_img = veil_img.filter(ImageFilter.GaussianBlur(radius=max(0.7, width / 2800.0)))
    veil_arr = np.asarray(veil_img, dtype=np.float32) / 255.0
    detail = _smoke_noise_field(width, height, t, 17713)
    veil_arr *= 0.52 + 0.72 * detail
    veil_arr = np.clip(veil_arr, 0.0, 1.0)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 208
    rgba[..., 1] = 214
    rgba[..., 2] = 210
    rgba[..., 3] = np.clip(veil_arr * SMOKE_VEIL_MAX_ALPHA, 0, 255).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def _draw_smoke_ribbon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    base_width: float,
    base_alpha: int,
    age_frac: float,
) -> None:
    """Draw a broad, tapered smoke ribbon that thins into wispy filaments."""
    if len(points) < 2 or base_alpha <= 1:
        return

    n = len(points)
    for i in range(n - 1):
        seg_frac = i / max(n - 1, 1)

        if seg_frac < 0.3:
            width_mult = 1.0 - seg_frac * 0.28
        else:
            width_mult = 0.14 + 0.64 * (1.0 - (seg_frac - 0.3) / 0.7) ** 0.58

        seg_width = max(1, int(base_width * width_mult * (0.72 + 0.28 * (1.0 - age_frac))))

        alpha_mult = (0.94 - seg_frac * 0.72) * (1.0 - age_frac * 0.35)
        seg_alpha = int(max(0.0, base_alpha * alpha_mult))
        if seg_alpha <= 1:
            continue

        gray_shift = int(seg_frac * 15)
        color = (210 + gray_shift, 215 + gray_shift, 208 + gray_shift, seg_alpha)

        p1 = points[i]
        p2 = points[i + 1]
        draw.line([p1, p2], fill=color, width=seg_width)


def _draw_advected_cluster_particles(
    draw: ImageDraw.ImageDraw,
    cluster: FireCluster,
    progress: float,
    season_day: float,
    frame_when: dt.datetime,
    t: float,
    projector: TerrainProjector,
    width: int,
    height: int,
    wind_field: WindField | None,
    *,
    height_m: float,
    speed_scale: float,
) -> None:
    """Draw broad smoke ribbons and streamers that extend far downwind."""
    if progress <= 0.03:
        return

    rng = np.random.default_rng(cluster.seed * 991)
    scale = width / 1920.0
    fire_age_hours = max(0.0, (season_day - cluster.start_day) * 24.0)
    if fire_age_hours <= 0.0:
        return

    # Source region
    source_radius_m = np.clip(
        cluster.spread_km * 55.0 * math.sqrt(progress),
        SMOKE_SOURCE_MIN_RADIUS_M,
        SMOKE_SOURCE_MAX_RADIUS_M,
    )

    # Get wind
    source_u, source_v = _sample_smoothed_smoke_wind(
        wind_field,
        cluster.lon,
        cluster.lat,
        frame_when,
        height_m=height_m,
        fallback_angle_deg=cluster.wind_angle_deg,
        seed=cluster.seed,
        t=t,
    )
    source_speed = max(math.hypot(source_u, source_v), 1e-6)
    wind_unit = np.array([source_u / source_speed, source_v / source_speed], dtype=np.float32)
    cross_unit = np.array([-wind_unit[1], wind_unit[0]], dtype=np.float32)

    ribbon_count = max(18, int(SMOKE_PARTICLES_PER_CLUSTER * 0.58 * (0.3 + 0.7 * progress)))

    for idx in range(ribbon_count):
        phase_hours = rng.uniform(0.0, SMOKE_PARTICLE_MAX_AGE_HOURS)
        age_hours = _smoke_particle_age_hours(fire_age_hours, phase_hours)
        if age_hours is None or age_hours < 0.2:
            continue

        # Spawn position with lateral spread
        source_along_m = rng.normal(0.0, source_radius_m * 0.2)
        source_cross_m = rng.normal(0.0, source_radius_m * 0.5)
        source_along_m = float(np.clip(source_along_m, -source_radius_m * 0.6, source_radius_m * 0.6))
        source_cross_m = float(np.clip(source_cross_m, -source_radius_m, source_radius_m))

        source_lon, source_lat = move_lonlat_by_meters(
            cluster.lon,
            cluster.lat,
            wind_unit[0] * source_along_m + cross_unit[0] * source_cross_m,
            wind_unit[1] * source_along_m + cross_unit[1] * source_cross_m,
        )

        particle_seed = cluster.seed * 10_000 + idx
        age_frac = age_hours / SMOKE_PARTICLE_MAX_AGE_HOURS

        trail_points = max(8, min(22, int(8 + age_frac * 15)))
        trail_hours = min(age_hours, SMOKE_PARTICLE_TRAIL_HOURS * rng.uniform(0.75, 1.35))

        screen_points: list[tuple[float, float]] = []
        for ti in range(trail_points):
            trail_frac = ti / max(trail_points - 1, 1)
            sample_age = age_hours - trail_hours * (1.0 - trail_frac)
            if sample_age < 0:
                sample_age = 0.0

            lon_p, lat_p = _advect_smoke_lonlat(
                source_lon,
                source_lat,
                sample_age,
                frame_when,
                wind_field,
                height_m=height_m,
                speed_scale=speed_scale,
                fallback_angle_deg=cluster.wind_angle_deg,
                seed=particle_seed,
                t=t + trail_frac * 0.03,
            )
            x_p, y_p = projector.lonlat_to_screen(lon_p, lat_p, width, height)
            screen_points.append((float(x_p), float(y_p)))

        # Skip if entirely off-screen
        xs = [p[0] for p in screen_points]
        ys = [p[1] for p in screen_points]
        if max(xs) < -350 or min(xs) > width + 350 or max(ys) < -350 or min(ys) > height + 350:
            continue

        lifecycle_alpha = _smoke_lifecycle_alpha(age_hours)
        base_alpha = 62.0 * progress * lifecycle_alpha * (1.10 - age_frac * 0.22) * rng.uniform(0.72, 1.22)
        alpha = int(np.clip(base_alpha, 0, 170))
        if alpha <= 2:
            continue

        ribbon_width = (9.0 + 34.0 * (0.45 + 0.55 * progress)) * scale
        ribbon_width *= rng.uniform(0.7, 1.4)
        ribbon_width *= 1.0 - age_frac * 0.32

        _draw_smoke_ribbon(draw, screen_points, ribbon_width, alpha, age_frac)

        # Add bloom glow at source for young ribbons
        if age_frac < 0.25 and len(screen_points) >= 1:
            source_pt = screen_points[0]
            bloom_r = ribbon_width * 2.5 * (1.0 - age_frac * 3.0)
            bloom_alpha = int(alpha * 0.35 * (1.0 - age_frac * 3.0))
            if bloom_r > 2 and bloom_alpha > 2:
                draw.ellipse(
                    (source_pt[0] - bloom_r, source_pt[1] - bloom_r,
                     source_pt[0] + bloom_r, source_pt[1] + bloom_r),
                    fill=(220, 225, 218, bloom_alpha),
                )

    wisp_count = max(10, int(SMOKE_PARTICLES_PER_CLUSTER * 0.12 * progress))
    for idx in range(wisp_count):
        phase_hours = rng.uniform(0.0, SMOKE_PARTICLE_MAX_AGE_HOURS)
        age_hours = _smoke_particle_age_hours(fire_age_hours, phase_hours)
        if age_hours is None or age_hours < 0.5:
            continue

        age_frac = age_hours / SMOKE_PARTICLE_MAX_AGE_HOURS
        if age_frac < 0.3:  # Wisps appear in mid-to-late lifecycle
            continue

        source_lon, source_lat = move_lonlat_by_meters(
            cluster.lon, cluster.lat,
            rng.normal(0.0, source_radius_m * 0.7),
            rng.normal(0.0, source_radius_m * 0.7),
        )

        lon_p, lat_p = _advect_smoke_lonlat(
            source_lon, source_lat, age_hours, frame_when, wind_field,
            height_m=height_m, speed_scale=speed_scale,
            fallback_angle_deg=cluster.wind_angle_deg,
            seed=cluster.seed * 20_000 + idx, t=t,
        )
        x_p, y_p = projector.lonlat_to_screen(lon_p, lat_p, width, height)

        if -100 < x_p < width + 100 and -100 < y_p < height + 100:
            wisp_alpha = int(34 * progress * (1.0 - age_frac) * rng.uniform(0.45, 0.95))
            wisp_size = (5 + 12 * age_frac) * scale * rng.uniform(0.6, 1.4)
            if wisp_alpha > 1 and wisp_size > 1:
                draw.ellipse(
                    (x_p - wisp_size, y_p - wisp_size, x_p + wisp_size, y_p + wisp_size),
                    fill=(225, 228, 222, wisp_alpha),
                )


def _smoke_density_resolution(width: int, height: int) -> int:
    target = max(96, int(round(max(width, height) / 3.7)))
    return int(np.clip(target, 96, SMOKE_DENSITY_RESOLUTION))


def _lonlat_to_density_pixel(lon: float, lat: float, resolution: int) -> tuple[float, float]:
    x = (float(lon) - LON_MIN) / max(LON_MAX - LON_MIN, 1e-6) * float(resolution - 1)
    y = (LAT_MAX - float(lat)) / max(LAT_MAX - LAT_MIN, 1e-6) * float(resolution - 1)
    return x, y


def _density_pixel_to_lonlat(
    x: np.ndarray | float,
    y: np.ndarray | float,
    resolution: int,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    lon = LON_MIN + np.asarray(x) / max(float(resolution - 1), 1.0) * (LON_MAX - LON_MIN)
    lat = LAT_MAX - np.asarray(y) / max(float(resolution - 1), 1.0) * (LAT_MAX - LAT_MIN)
    return lon, lat


def _density_blur(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(field, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        axial = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        diagonal = (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        out = center * 0.38 + axial * 0.125 + diagonal * 0.03
    return out.astype(np.float32, copy=False)


def _density_bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    height, width = src.shape
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    x1c = np.clip(x1, 0, width - 1)
    y1c = np.clip(y1, 0, height - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    top = src[y0c, x0c] * (1.0 - wx) + src[y0c, x1c] * wx
    bottom = src[y1c, x0c] * (1.0 - wx) + src[y1c, x1c] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.where(valid, sampled, 0.0).astype(np.float32)


def _density_mean_wind(
    frame_index: int,
    total_frames: int,
    wind_field: WindField | None,
    *,
    wind_height_m: float,
) -> tuple[float, float]:
    _, season_day = season_progress(frame_index, total_frames)
    t = frame_index / max(total_frames - 1, 1)
    when = frame_datetime(frame_index, total_frames)
    weighted_u = 0.0
    weighted_v = 0.0
    total_weight = 0.0
    for cluster in FIRE_CLUSTERS:
        progress = cluster_progress(cluster, season_day)
        if progress <= 0.01:
            continue
        u_mps, v_mps = _sample_smoothed_smoke_wind(
            wind_field,
            cluster.lon,
            cluster.lat,
            when,
            height_m=wind_height_m,
            fallback_angle_deg=cluster.wind_angle_deg,
            seed=cluster.seed,
            t=t,
        )
        weight = (0.35 + 0.65 * progress) * math.sqrt(max(cluster.final_area_ha, 1.0))
        weighted_u += u_mps * weight
        weighted_v += v_mps * weight
        total_weight += weight
    if total_weight <= 0.0:
        return _fallback_wind_from_angle(64.0, 123, t, -120.8, 38.2)
    return weighted_u / total_weight, weighted_v / total_weight


def _smoke_density_wind_field(
    frame_index: int,
    total_frames: int,
    resolution: int,
    wind_field: WindField | None,
    *,
    wind_height_m: float,
    wind_speed_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_mps, v_mps = _density_mean_wind(
        frame_index,
        total_frames,
        wind_field,
        wind_height_m=wind_height_m,
    )
    speed = max(math.hypot(u_mps, v_mps), 1e-6)
    unit_east = u_mps / speed
    unit_north = v_mps / speed
    scale = resolution / 512.0
    base_speed_px = (2.25 + 2.15 * float(wind_speed_scale)) * scale * np.clip(speed / 10.5, 0.75, 1.55)
    base_u = unit_east * base_speed_px
    base_v = -unit_north * base_speed_px

    yy, xx = np.mgrid[0:resolution, 0:resolution].astype(np.float32)
    xn = xx / max(resolution - 1, 1)
    yn = yy / max(resolution - 1, 1)
    phase = frame_index * 0.045
    broad = np.sin(math.tau * (yn * 1.20 + 0.20 * xn + phase))
    shear = np.cos(math.tau * (xn * 0.92 - yn * 0.58 - phase * 0.72))
    curl_stream = (
        np.sin(math.tau * (xn * 1.55 + yn * 0.35 + phase))
        * np.sin(math.tau * (yn * 1.10 - xn * 0.28 - phase * 0.63))
    )
    dpsi_dy, dpsi_dx = np.gradient(curl_stream.astype(np.float32))

    u = base_u + scale * (0.30 * broad + 0.18 * shear + 1.15 * dpsi_dy)
    v = base_v + scale * (0.18 * broad - 0.12 * shear - 1.05 * dpsi_dx)
    if base_u >= 0.0:
        u = np.maximum(u, base_u * 0.18)
    if base_v <= 0.0:
        v = np.minimum(v, abs(base_v) * 0.22)
    return u.astype(np.float32), v.astype(np.float32)


def _advect_smoke_density(density: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0 : density.shape[0], 0 : density.shape[1]].astype(np.float32)
    return _density_bilinear_sample(density, xx - u, yy - v)


def _inject_smoke_density(
    density: np.ndarray,
    frame_index: int,
    total_frames: int,
    resolution: int,
    wind_field: WindField | None,
    *,
    wind_height_m: float,
) -> np.ndarray:
    out = np.asarray(density, dtype=np.float32).copy()
    _, season_day = season_progress(frame_index, total_frames)
    t = frame_index / max(total_frames - 1, 1)
    when = frame_datetime(frame_index, total_frames)

    for cluster in FIRE_CLUSTERS:
        progress = cluster_progress(cluster, season_day)
        if progress <= 0.006:
            continue
        x0, y0 = _lonlat_to_density_pixel(cluster.lon, cluster.lat, resolution)
        if x0 < -8.0 or x0 > resolution + 8.0 or y0 < -8.0 or y0 > resolution + 8.0:
            continue

        u_mps, v_mps = _sample_smoothed_smoke_wind(
            wind_field,
            cluster.lon,
            cluster.lat,
            when,
            height_m=wind_height_m,
            fallback_angle_deg=cluster.wind_angle_deg,
            seed=cluster.seed,
            t=t,
        )
        speed = max(math.hypot(u_mps, v_mps), 1e-6)
        wind_x = u_mps / speed
        wind_y = -v_mps / speed
        wind_len = max(math.hypot(wind_x, wind_y), 1e-6)
        wind_x /= wind_len
        wind_y /= wind_len

        source_radius = (7.0 + 0.055 * cluster.spread_km) * math.sqrt(progress) * resolution / 512.0
        source_radius = float(np.clip(source_radius, 2.0, 19.0 * resolution / 512.0))
        roi = int(math.ceil(source_radius * 5.2))
        x_min = max(0, int(math.floor(x0)) - roi)
        x_max = min(resolution, int(math.floor(x0)) + roi + 1)
        y_min = max(0, int(math.floor(y0)) - roi)
        y_max = min(resolution, int(math.floor(y0)) + roi + 1)
        if x_min >= x_max or y_min >= y_max:
            continue

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float32)
        dx = xx - float(x0)
        dy = yy - float(y0)
        along = dx * wind_x + dy * wind_y
        cross = -dx * wind_y + dy * wind_x
        core = np.exp(-(dx * dx + dy * dy) / (2.0 * source_radius * source_radius))
        seed_tail = np.exp(
            -(
                ((along - source_radius * 1.35) ** 2) / (2.0 * (source_radius * 2.35) ** 2)
                + (cross * cross) / (2.0 * (source_radius * 0.82) ** 2)
            )
        )
        area_weight = np.clip(math.sqrt(cluster.final_area_ha / 155_000.0), 0.35, 1.65)
        pulse = 0.94 + 0.06 * math.sin(frame_index * 0.31 + cluster.seed * 0.19)
        source = (0.72 * core + 0.64 * seed_tail) * progress * area_weight * pulse
        out[y_min:y_max, x_min:x_max] += source.astype(np.float32) * 0.155

    return out


def _diffuse_decay_smoke_density(density: np.ndarray, frame_index: int) -> np.ndarray:
    out = np.nan_to_num(np.asarray(density, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = _density_blur(np.clip(out, 0.0, None), passes=1)
    out *= SMOKE_DENSITY_DECAY + 0.004 * math.sin(frame_index * 0.041)

    resolution = out.shape[0]
    yy, xx = np.mgrid[0:resolution, 0:resolution].astype(np.float32)
    phase = frame_index * 0.052
    modulation = (
        1.0
        + 0.035 * np.sin(0.043 * xx + 0.019 * yy + phase)
        + 0.028 * np.sin(0.017 * xx - 0.035 * yy - phase * 0.8)
    )
    out *= np.clip(modulation, 0.90, 1.08).astype(np.float32)
    edge_distance = np.minimum.reduce((xx, yy, resolution - 1 - xx, resolution - 1 - yy))
    fade_px = max(4.0, resolution * SMOKE_DENSITY_BORDER_FADE)
    edge_fade = smoothstep(0.0, fade_px, edge_distance)
    out *= edge_fade.astype(np.float32)
    return np.clip(out, 0.0, 4.0).astype(np.float32)


def simulate_smoke_density_frame(
    frame_index: int,
    total_frames: int,
    resolution: int,
    wind_field: WindField | None = None,
    *,
    wind_height_m: float = DEFAULT_WIND_HEIGHT_M,
    wind_speed_scale: float = DEFAULT_WIND_SPEED_SCALE,
) -> np.ndarray:
    """Simulate 2.5D smoke density from frame zero through ``frame_index``."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if resolution <= 1:
        raise ValueError("resolution must be greater than 1")

    density = np.zeros((int(resolution), int(resolution)), dtype=np.float32)
    for step in range(int(frame_index) + 1):
        u, v = _smoke_density_wind_field(
            step,
            total_frames,
            resolution,
            wind_field,
            wind_height_m=wind_height_m,
            wind_speed_scale=wind_speed_scale,
        )
        density = _advect_smoke_density(density, u, v)
        density = _inject_smoke_density(
            density,
            step,
            total_frames,
            resolution,
            wind_field,
            wind_height_m=wind_height_m,
        )
        density = _diffuse_decay_smoke_density(density, step)
    return density


def _smoke_density_sequence(
    total_frames: int,
    resolution: int,
    wind_field: WindField | None,
    *,
    wind_height_m: float,
    wind_speed_scale: float,
) -> list[np.ndarray]:
    density = np.zeros((int(resolution), int(resolution)), dtype=np.float32)
    frames: list[np.ndarray] = []
    for frame_index in range(int(total_frames)):
        u, v = _smoke_density_wind_field(
            frame_index,
            total_frames,
            resolution,
            wind_field,
            wind_height_m=wind_height_m,
            wind_speed_scale=wind_speed_scale,
        )
        density = _advect_smoke_density(density, u, v)
        density = _inject_smoke_density(
            density,
            frame_index,
            total_frames,
            resolution,
            wind_field,
            wind_height_m=wind_height_m,
        )
        density = _diffuse_decay_smoke_density(density, frame_index)
        frames.append(density.astype(np.float32, copy=True))
    return frames


_SMOKE_DENSITY_CACHE: dict[tuple[object, ...], list[np.ndarray]] = {}


def _cached_smoke_density_frame(
    frame_index: int,
    total_frames: int,
    resolution: int,
    wind_field: WindField | None,
    *,
    wind_height_m: float,
    wind_speed_scale: float,
) -> np.ndarray:
    cluster_key = tuple((cluster.name, cluster.seed, cluster.start_day, cluster.ramp_days) for cluster in FIRE_CLUSTERS)
    wind_key = None if wind_field is None else (wind_field.source_label, id(wind_field))
    key = (
        int(total_frames),
        int(resolution),
        wind_key,
        round(float(wind_height_m), 3),
        round(float(wind_speed_scale), 5),
        cluster_key,
    )
    sequence = _SMOKE_DENSITY_CACHE.get(key)
    if sequence is None:
        sequence = _smoke_density_sequence(
            total_frames,
            resolution,
            wind_field,
            wind_height_m=wind_height_m,
            wind_speed_scale=wind_speed_scale,
        )
        _SMOKE_DENSITY_CACHE.clear()
        _SMOKE_DENSITY_CACHE[key] = sequence
    return sequence[int(np.clip(frame_index, 0, len(sequence) - 1))]


def _pil_blur_float(field: np.ndarray, radius: float) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    peak = max(float(np.max(arr)), 1e-6)
    image = Image.fromarray(np.clip(arr / peak * 255.0, 0, 255).astype(np.uint8))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(blurred, dtype=np.float32) / 255.0 * peak


def _smoke_density_to_rgba(density: np.ndarray, frame_index: int) -> np.ndarray:
    src = np.nan_to_num(np.asarray(density, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fine = _pil_blur_float(src, 1.6)
    medium = _pil_blur_float(src, 5.5)
    broad = _pil_blur_float(src, 14.0)
    sheet = np.clip(fine * 0.54 + medium * 0.58 + broad * 0.25, 0.0, None)

    height, width = sheet.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    phase = frame_index * 0.047
    texture = (
        1.0
        + 0.10 * np.sin(0.028 * xx + 0.015 * yy + phase)
        + 0.06 * np.sin(0.059 * xx - 0.026 * yy - phase * 0.72)
        + 0.035 * np.cos(0.013 * xx + 0.039 * yy + phase * 1.4)
    )
    sheet *= np.clip(texture, 0.75, 1.18).astype(np.float32)

    optical_depth = 1.0 - np.exp(-0.30 * sheet)
    alpha_norm = smoothstep(0.035, 0.72, optical_depth)
    alpha = np.clip(SMOKE_DENSITY_MAX_ALPHA * (alpha_norm ** 1.35), 0.0, SMOKE_DENSITY_MAX_ALPHA)
    alpha[sheet < 0.012] = 0.0
    gray = np.clip(126.0 + 82.0 * smoothstep(0.035, 0.70, optical_depth), 122.0, 212.0)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    positive = alpha > 0.5
    rgba[..., 0] = np.where(positive, gray, 0.0).astype(np.uint8)
    rgba[..., 1] = np.where(positive, gray, 0.0).astype(np.uint8)
    rgba[..., 2] = np.where(positive, gray, 0.0).astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


def _warp_density_overlay_to_screen(
    smoke_rgba: np.ndarray,
    width: int,
    height: int,
    projector: TerrainProjector,
) -> Image.Image:
    source = Image.fromarray(np.asarray(smoke_rgba, dtype=np.uint8))
    src_w, src_h = source.size
    src = np.array(
        [
            [0.0, 0.0, 1.0],
            [src_w - 1.0, 0.0, 1.0],
            [src_w - 1.0, src_h - 1.0, 1.0],
            [0.0, src_h - 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    corners_lonlat = (
        (LON_MIN, LAT_MAX),
        (LON_MAX, LAT_MAX),
        (LON_MAX, LAT_MIN),
        (LON_MIN, LAT_MIN),
    )
    dst = np.asarray(
        [projector.lonlat_to_screen(lon, lat, width, height) for lon, lat in corners_lonlat],
        dtype=np.float64,
    )
    ax, bx, cx = np.linalg.lstsq(src, dst[:, 0], rcond=None)[0]
    ay, by, cy = np.linalg.lstsq(src, dst[:, 1], rcond=None)[0]
    matrix = np.array([[ax, bx, cx], [ay, by, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    coeffs = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )
    warped = source.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    return warped.filter(ImageFilter.GaussianBlur(radius=max(0.45, width / 3600.0)))


def draw_smoke_layer(
    width: int,
    height: int,
    frame_index: int,
    total_frames: int,
    projector: TerrainProjector,
    wind_field: WindField | None = None,
    *,
    wind_height_m: float = DEFAULT_WIND_HEIGHT_M,
    wind_speed_scale: float = DEFAULT_WIND_SPEED_SCALE,
) -> Image.Image:
    """Render smoke through a 2.5D geospatial density raster overlay."""
    resolution = _smoke_density_resolution(width, height)
    density = _cached_smoke_density_frame(
        frame_index,
        total_frames,
        resolution,
        wind_field,
        wind_height_m=wind_height_m,
        wind_speed_scale=wind_speed_scale,
    )
    smoke_rgba = _smoke_density_to_rgba(density, frame_index)
    return _warp_density_overlay_to_screen(smoke_rgba, width, height, projector)


def _draw_fire_layer(
    width: int,
    height: int,
    frame_index: int,
    total_frames: int,
    projector: TerrainProjector,
    *,
    include_glow: bool,
    include_core: bool,
) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    _, season_day = season_progress(frame_index, total_frames)
    t = frame_index / max(total_frames - 1, 1)

    for cluster in FIRE_CLUSTERS:
        progress = cluster_progress(cluster, season_day)
        if progress <= 0.0:
            continue

        cx, cy = projector.lonlat_to_screen(cluster.lon, cluster.lat, width, height)
        radius = max(2.5, cluster.spread_km * width / 45_000.0) * math.sqrt(progress)
        dot_count = max(1, min(7, int(round(1 + cluster.spark_count * 0.018 * progress))))
        pulse = 0.72 + 0.28 * math.sin(t * math.tau * 5.0 + cluster.seed)
        rng = np.random.default_rng(cluster.seed)

        for _ in range(dot_count):
            angle = rng.uniform(0.0, math.tau)
            distance = radius * math.sqrt(rng.uniform(0.0, 1.0))
            sx = cx + math.cos(angle) * distance
            sy = cy + math.sin(angle) * distance * 0.72
            size = rng.uniform(1.2, 2.7) * (width / 1920.0)
            alpha = int((130 + rng.integers(0, 78)) * pulse)
            glow_radius = size * rng.uniform(5.0, 9.0)
            if include_glow:
                glow_draw.ellipse(
                    (sx - glow_radius, sy - glow_radius, sx + glow_radius, sy + glow_radius),
                    fill=(255, 102, 24, int(alpha * 0.13)),
                )
            if include_core:
                draw.ellipse(
                    (sx - size, sy - size, sx + size, sy + size),
                    fill=(255, 126, 32, alpha),
                )
                core = max(0.8, size * 0.42)
                draw.ellipse(
                    (sx - core, sy - core, sx + core, sy + core),
                    fill=(255, 232, 132, min(255, alpha + 35)),
                )

    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if include_glow:
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(2.0, width / 700.0)))
        result.alpha_composite(glow)
    if include_core:
        result.alpha_composite(layer)
    if projector.screen_mask is not None:
        mask = projector.screen_mask
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.BICUBIC)
        mask = mask.filter(ImageFilter.MaxFilter(max(3, int(width / 360) | 1)))
        clipped_alpha = ImageChops.multiply(result.getchannel("A"), mask)
        result.putalpha(clipped_alpha)
    return result


def draw_fire_layer(
    width: int,
    height: int,
    frame_index: int,
    total_frames: int,
    projector: TerrainProjector,
) -> Image.Image:
    return _draw_fire_layer(
        width,
        height,
        frame_index,
        total_frames,
        projector,
        include_glow=True,
        include_core=True,
    )


def draw_fire_core_layer(
    width: int,
    height: int,
    frame_index: int,
    total_frames: int,
    projector: TerrainProjector,
) -> Image.Image:
    return _draw_fire_layer(
        width,
        height,
        frame_index,
        total_frames,
        projector,
        include_glow=False,
        include_core=True,
    )


def calc_stats(frame_index: int, frame_count: int) -> tuple[float, float]:
    _, season_day = season_progress(frame_index, frame_count)
    area = 0.0
    plume_driver = 0.0
    for cluster in FIRE_CLUSTERS:
        progress = cluster_progress(cluster, season_day)
        if progress > 0:
            area += cluster.final_area_ha * (progress ** 1.18)
            plume_driver += cluster.final_area_ha * min(1.0, progress * 1.4)

    season_extra = smoothstep(5.0, 55.0, season_day)
    area = min(FINAL_SEASON_AREA_HA, area + 495_000.0 * (season_extra ** 1.35))
    plume_driver = min(FINAL_SEASON_AREA_HA, plume_driver + 355_000.0 * season_extra)

    exposure = FINAL_EXPOSED_POP * smoothstep(8.0, 47.0, season_day)
    exposure *= 0.84 + 0.16 * math.sin(season_day * 0.22) ** 2
    exposure *= min(1.0, plume_driver / max(FINAL_SEASON_AREA_HA * 0.42, 1.0))
    return area, exposure


def draw_text_overlays(
    image: Image.Image,
    *,
    frame_index: int,
    total_frames: int,
    area_ha: float,
    exposed_pop: float,
    wind_source: str | None = None,
) -> Image.Image:
    width, height = image.size
    scale = width / 1920.0
    draw = ImageDraw.Draw(image, "RGBA")

    small_font = load_font(max(20, int(round(30 * scale))))
    date_font = load_font(max(34, int(round(48 * scale))))
    label_font = load_font(max(26, int(round(38 * scale))))
    value_font = load_font(max(48, int(round(72 * scale))))
    exposure_font = load_font(max(20, int(round(30 * scale))))

    margin_x = int(round(100 * scale))
    margin_y = int(round(54 * scale))
    text_color = (245, 248, 246, 238)
    shadow = (0, 0, 0, 140)

    def text_with_shadow(pos: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int, int]) -> None:
        x, y = pos
        off = max(1, int(round(2 * scale)))
        draw.text((x + off, y + off), text, font=font, fill=shadow)
        draw.text(pos, text, font=font, fill=fill)

    data_text = "Data: NASA GEOS-FP/FIRMS; OSM boundary; Terrarium DEM"
    if wind_source:
        short_wind = wind_source if len(wind_source) <= 42 else f"{wind_source[:39]}..."
        data_text = f"Data: NASA GEOS-FP/FIRMS; wind {short_wind}; OSM boundary; Terrarium DEM"
    text_with_shadow((margin_x, margin_y), data_text, small_font, text_color)
    text_with_shadow(
        (margin_x, margin_y + int(round(36 * scale))),
        "California wildfire season, 2020",
        small_font,
        text_color,
    )

    date_text = format_date(frame_index, total_frames)
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_w = bbox[2] - bbox[0]
    else:
        date_w = int(round(draw.textlength(date_text, font=date_font)))
    text_with_shadow(
        (width - margin_x - date_w, margin_y + int(round(12 * scale))),
        date_text,
        date_font,
        (248, 250, 250, 245),
    )

    base_y = height - int(round(210 * scale))
    text_with_shadow((margin_x, base_y), "Area Burned:", label_font, (248, 248, 248, 245))
    text_with_shadow(
        (margin_x, base_y + int(round(54 * scale))),
        format_area(area_ha),
        value_font,
        (252, 254, 252, 250),
    )
    text_with_shadow(
        (margin_x, height - int(round(72 * scale))),
        f"Smoke Exposure: {format_people(exposed_pop)}",
        exposure_font,
        (237, 240, 238, 220),
    )

    return image


def render_frame(
    base_map: Image.Image,
    projector: TerrainProjector,
    frame_index: int,
    frame_count: int,
    wind_field: WindField | None = None,
    *,
    wind_height_m: float = DEFAULT_WIND_HEIGHT_M,
    wind_speed_scale: float = DEFAULT_WIND_SPEED_SCALE,
) -> Image.Image:
    frame = base_map.copy()
    width, height = frame.size
    smoke = draw_smoke_layer(
        width,
        height,
        frame_index,
        frame_count,
        projector,
        wind_field,
        wind_height_m=wind_height_m,
        wind_speed_scale=wind_speed_scale,
    )
    frame.alpha_composite(
        _draw_fire_layer(
            width,
            height,
            frame_index,
            frame_count,
            projector,
            include_glow=True,
            include_core=False,
        )
    )
    frame.alpha_composite(smoke)
    frame.alpha_composite(draw_fire_core_layer(width, height, frame_index, frame_count, projector))
    area_ha, exposed_pop = calc_stats(frame_index, frame_count)
    return draw_text_overlays(
        frame,
        frame_index=frame_index,
        total_frames=frame_count,
        area_ha=area_ha,
        exposed_pop=exposed_pop,
        wind_source=wind_field.source_label if wind_field is not None else None,
    )


def encode_video(frames_dir: Path, fps: int, output: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the MP4 output.")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def render_video(args: argparse.Namespace) -> None:
    width, height = args.size
    frame_count = max(1, int(args.frames))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)

    wind_field = None
    if args.wind_file is not None:
        wind_path = args.wind_file.resolve()
        print(f"[Wildfire Smoke] Loading wind field from {wind_path}")
        wind_field = WindField.from_file(
            wind_path,
            u_var=args.wind_u_var,
            v_var=args.wind_v_var,
            default_height_m=float(args.wind_height_m),
        )
        print(f"[Wildfire Smoke] Using real wind data: {wind_field.source_label}")
    else:
        print("[Wildfire Smoke] No --wind-file supplied; using deterministic fallback wind angles")

    print(f"[Wildfire Smoke] Building OSM-clipped 3D terrain base at {width}x{height}")
    base_map, projector = build_base_map(
        width,
        height,
        cache_dir=args.cache_dir.resolve(),
        dem_zoom=int(args.dem_zoom),
        max_dem_size=int(args.max_dem_size),
        force=bool(args.force),
    )

    with tempfile.TemporaryDirectory(prefix="forge3d-wildfire-") as temp_dir:
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        render_total = 1 if args.preview_only else frame_count
        start = time.perf_counter()
        for output_index in range(render_total):
            frame_index = min(frame_count - 1, frame_count // 2) if args.preview_only else output_index
            frame = render_frame(
                base_map,
                projector,
                frame_index,
                frame_count,
                wind_field,
                wind_height_m=float(args.wind_height_m),
                wind_speed_scale=float(args.wind_speed_scale),
            ).convert("RGB")
            frame_path = frames_dir / f"frame_{output_index:05d}.png"
            frame.save(frame_path, optimize=False)

            if output_index == 0 or args.preview_only:
                frame.save(args.preview, optimize=False)

            if (output_index + 1) % 30 == 0 or output_index + 1 == render_total:
                elapsed = time.perf_counter() - start
                print(
                    f"[Wildfire Smoke] Rendered {output_index + 1}/{render_total} "
                    f"frames in {elapsed:.1f}s"
                )

        if args.preview_only:
            print(f"[Wildfire Smoke] Preview saved to {args.preview}")
            return

        encode_video(frames_dir, int(args.fps), args.output)

    print(f"[Wildfire Smoke] Video saved to {args.output}")
    print(f"[Wildfire Smoke] Preview saved to {args.preview}")


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
