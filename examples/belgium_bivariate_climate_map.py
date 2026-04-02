#!/usr/bin/env python3
"""Belgium bivariate climate map in pure Python + forge3d.

Workflow:
1. TerraClimate yearly NetCDF data loaded with xarray over OPeNDAP.
2. Clipped to Belgium with geopandas + rioxarray.
3. Smoothed with a masked Gaussian filter.
4. Classified into a balanced 4x4 bivariate grid with asymmetric quantiles.
5. Colored with the exact biscale "BlueGold" 4x4 matrix.
6. Exported as a 2D map and a forge3d terrain render.

Requirements:
    pip install forge3d geopandas pillow rasterio xarray rioxarray scipy netCDF4 "dask[array]"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import netCDF4  # noqa: F401
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.windows import Window, transform as window_transform
from scipy.ndimage import distance_transform_edt, gaussian_filter

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples" / "out" / "belgium_bivariate_climate"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "belgium_bivariate_climate"
DEFAULT_TARGET_CRS = "EPSG:3035"

NATURAL_EARTH_COUNTRIES_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)
TERRARIUM_TILE_URL = (
    "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
)
TERRACLIMATE_OPENDAP_URL = (
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
    "TERRACLIMATE_ALL/data/TerraClimate_{var}_{year}.nc"
)

DEFAULT_YEAR_START = 1971
DEFAULT_YEAR_END = 2020
DEFAULT_SMOOTH_SIGMA = 1.35
DEFAULT_PAD_DEGREES = 0.25
DEFAULT_DEM_ZOOM = 10

TERRARIUM_TILE_SIZE = 256
DEM_DOWNLOAD_WORKERS = 8
TERRACLIMATE_WORKERS = 3
TERRACLIMATE_RETRIES = 3
REFERENCE_TERRAIN_WIDTH = 365.0
BASE_TERRAIN_ZSCALE = 0.12
MAP_SCALE_2D = 1.12

MAP_TITLE = "BELGIUM: Temperature and Precipitation"
MAP_CAPTION = (
    "Sources: TerraClimate + AWS Terrain Tiles | ©2026 Milos Popovic (milospopovic.net)"
)

BLUEGOLD_4X4 = {
    (1, 1): "#d3d3d3",
    (2, 1): "#a6bcc7",
    (3, 1): "#77a6bb",
    (4, 1): "#488fb0",
    (1, 2): "#d6c597",
    (2, 2): "#a8b08f",
    (3, 2): "#799c86",
    (4, 2): "#49867e",
    (1, 3): "#d9b653",
    (2, 3): "#aba24e",
    (3, 3): "#7b904a",
    (4, 3): "#4a7b45",
    (1, 4): "#dea301",
    (2, 4): "#ae9101",
    (3, 4): "#7d8001",
    (4, 4): "#4c6e01",
}


def _bivariate_palette_color(temp_class: int, precip_class: int) -> str:
    # Swap the palette axes so warmer reads as the gold ramp and rainier as the blue ramp.
    return BLUEGOLD_4X4[(precip_class, temp_class)]

ASYMMETRIC_QUANTILE_CANDIDATES = (
    (0.12, 0.34, 0.62),
    (0.14, 0.36, 0.64),
    (0.16, 0.40, 0.68),
    (0.18, 0.44, 0.72),
    (0.20, 0.46, 0.74),
    (0.22, 0.48, 0.78),
)


@dataclass(frozen=True)
class RasterBundle:
    temperature: np.ndarray
    precipitation: np.ndarray
    elevation: np.ndarray
    mask: np.ndarray
    transform: rasterio.Affine
    crs: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--target-crs", default=DEFAULT_TARGET_CRS)
    parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
    parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--terraclimate-pad-deg", type=float, default=DEFAULT_PAD_DEGREES)
    parser.add_argument("--quantile-sample", type=int, default=180000)
    parser.add_argument("--temp-quantiles", default="")
    parser.add_argument("--precip-quantiles", default="")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dem-zoom", type=int, default=DEFAULT_DEM_ZOOM)
    parser.add_argument("--snapshot-width", type=int, default=4000)
    parser.add_argument("--snapshot-height", type=int, default=4000)
    parser.add_argument("--viewer-width", type=int, default=1200)
    parser.add_argument("--viewer-height", type=int, default=1200)
    parser.add_argument("--cam-phi", type=float, default=90.0)
    parser.add_argument("--cam-theta", type=float, default=20.0)
    parser.add_argument("--cam-radius", type=float, default=1040.0)
    parser.add_argument("--cam-fov", type=float, default=24.0)
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def _parse_quantiles(text: str) -> tuple[float, float, float] | None:
    if not text.strip():
        return None
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3 or not (parts[0] < parts[1] < parts[2]):
        raise ValueError("Quantiles must be three strictly increasing values")
    if any(value <= 0.0 or value >= 1.0 for value in parts):
        raise ValueError("Quantiles must be strictly between 0 and 1")
    return parts[0], parts[1], parts[2]


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with urlopen(url, timeout=60) as response:
        with tempfile.NamedTemporaryFile(
            prefix=dest.stem,
            suffix=dest.suffix,
            dir=str(dest.parent),
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp_path.replace(dest)
    return dest


def _load_belgium_gdf(boundary_zip: Path, target_crs: str) -> gpd.GeoDataFrame:
    countries = gpd.read_file(boundary_zip)
    belgium = countries[countries["ADM0_A3"] == "BEL"]
    if belgium.empty:
        raise RuntimeError("Could not find Belgium in Natural Earth countries data")
    return belgium.to_crs(target_crs)


def _load_belgium_geometry(boundary_zip: Path, target_crs: str):
    belgium = _load_belgium_gdf(boundary_zip, target_crs)
    if hasattr(belgium.geometry, "union_all"):
        return belgium.geometry.union_all()
    return belgium.geometry.unary_union


def _tile_x_from_lon(lon: float, zoom: int) -> int:
    n = 1 << zoom
    return int(np.clip(((lon + 180.0) / 360.0) * n, 0, n - 1))


def _tile_y_from_lat(lat: float, zoom: int) -> int:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 1 << zoom
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) * 0.5 * n
    return int(np.clip(y, 0, n - 1))


def _tile_bounds_mercator(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    world_span = 20037508.342789244 * 2.0
    tile_span = world_span / float(1 << zoom)
    west = -20037508.342789244 + x * tile_span
    east = west + tile_span
    north = 20037508.342789244 - y * tile_span
    south = north - tile_span
    return west, south, east, north


def _decode_terrarium(tile_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(tile_path).convert("RGB"), dtype=np.float32)
    return (rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0) - 32768.0


def _download_terrarium_tile(
    cache_dir: Path,
    zoom: int,
    x: int,
    y: int,
    *,
    force: bool,
) -> Path:
    tile_path = cache_dir / "terrarium" / str(zoom) / str(x) / f"{y}.png"
    url = TERRARIUM_TILE_URL.format(z=zoom, x=x, y=y)
    return _download(url, tile_path, force=force)


def _build_terrarium_dem(
    boundary_zip: Path,
    cache_dir: Path,
    zoom: int,
    *,
    force: bool,
) -> Path:
    if zoom < 1 or zoom > 12:
        raise ValueError("DEM zoom must be between 1 and 12")
    dem_path = cache_dir / f"belgium_terrarium_dem_z{zoom}_3857.tif"
    if dem_path.exists() and not force:
        return dem_path

    belgium_wgs84 = _load_belgium_geometry(boundary_zip, "EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = belgium_wgs84.bounds
    x_min = _tile_x_from_lon(lon_min, zoom)
    x_max = _tile_x_from_lon(lon_max, zoom)
    y_min = _tile_y_from_lat(lat_max, zoom)
    y_max = _tile_y_from_lat(lat_min, zoom)
    tiles = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    if not tiles:
        raise RuntimeError("No DEM tiles intersected Belgium bounds")

    def _fetch(tile: tuple[int, int]) -> tuple[tuple[int, int], Path]:
        x, y = tile
        return tile, _download_terrarium_tile(cache_dir, zoom, x, y, force=force)

    fetched: list[tuple[tuple[int, int], Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEM_DOWNLOAD_WORKERS) as executor:
        for item in executor.map(_fetch, tiles):
            fetched.append(item)

    tile_rows = y_max - y_min + 1
    tile_cols = x_max - x_min + 1
    mosaic = np.empty(
        (tile_rows * TERRARIUM_TILE_SIZE, tile_cols * TERRARIUM_TILE_SIZE),
        dtype=np.float32,
    )
    for (x, y), tile_path in fetched:
        row = y - y_min
        col = x - x_min
        r0 = row * TERRARIUM_TILE_SIZE
        c0 = col * TERRARIUM_TILE_SIZE
        mosaic[r0 : r0 + TERRARIUM_TILE_SIZE, c0 : c0 + TERRARIUM_TILE_SIZE] = _decode_terrarium(
            tile_path
        )

    west, _, _, north = _tile_bounds_mercator(x_min, y_min, zoom)
    _, south, east, _ = _tile_bounds_mercator(x_max, y_max, zoom)
    transform = from_bounds(west, south, east, north, mosaic.shape[1], mosaic.shape[0])
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=mosaic.shape[1],
        height=mosaic.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(mosaic, 1)
    return dem_path


def _subtitle(year_start: int, year_end: int) -> str:
    return f"TerraClimate mean annual temperature and precipitation ({year_start}-{year_end})"


def _terraclimate_cache_path(cache_dir: Path, year_start: int, year_end: int) -> Path:
    return cache_dir / f"belgium_terraclimate_{year_start}_{year_end}_4326.nc"


def _prepare_geo_da(da: xr.DataArray) -> xr.DataArray:
    return da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")


def _annual_terraclimate_field(
    var_name: str,
    year: int,
    belgium_wgs84: gpd.GeoDataFrame,
    *,
    pad_deg: float,
) -> xr.DataArray:
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in belgium_wgs84.total_bounds)
    url = TERRACLIMATE_OPENDAP_URL.format(var=var_name, year=year)

    last_error: Exception | None = None
    for attempt in range(1, TERRACLIMATE_RETRIES + 1):
        dataset: xr.Dataset | None = None
        try:
            dataset = xr.open_dataset(url, engine="netcdf4")
            da = dataset[var_name].sel(
                lon=slice(lon_min - pad_deg, lon_max + pad_deg),
                lat=slice(lat_max + pad_deg, lat_min - pad_deg),
            )
            da = _prepare_geo_da(da)
            annual = da.sum("time", skipna=True) if var_name == "ppt" else da.mean("time", skipna=True)
            clipped = _prepare_geo_da(annual).rio.clip(
                belgium_wgs84.geometry, belgium_wgs84.crs, drop=True
            )
            clipped = clipped.astype(np.float32)
            clipped.load()
            clipped.name = var_name
            return clipped
        except Exception as exc:
            last_error = exc
            if attempt == TERRACLIMATE_RETRIES:
                raise RuntimeError(
                    f"Failed to load TerraClimate {var_name} for {year} after {attempt} attempts"
                ) from exc
        finally:
            if dataset is not None:
                dataset.close()
    raise RuntimeError(f"Failed to load TerraClimate {var_name} for {year}") from last_error


def _build_terraclimate_climatology(
    boundary_zip: Path,
    cache_dir: Path,
    *,
    year_start: int,
    year_end: int,
    pad_deg: float,
    force: bool,
) -> Path:
    cache_path = _terraclimate_cache_path(cache_dir, year_start, year_end)
    if cache_path.exists() and not force:
        return cache_path

    belgium_wgs84 = _load_belgium_gdf(boundary_zip, "EPSG:4326")
    tmean_years: list[xr.DataArray] = []
    ppt_years: list[xr.DataArray] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=TERRACLIMATE_WORKERS) as executor:
        for year in range(year_start, year_end + 1):
            future_ppt = executor.submit(
                _annual_terraclimate_field, "ppt", year, belgium_wgs84, pad_deg=pad_deg
            )
            future_tmin = executor.submit(
                _annual_terraclimate_field, "tmin", year, belgium_wgs84, pad_deg=pad_deg
            )
            future_tmax = executor.submit(
                _annual_terraclimate_field, "tmax", year, belgium_wgs84, pad_deg=pad_deg
            )
            ppt = future_ppt.result().rename("precipitation")
            tmin = future_tmin.result()
            tmax = future_tmax.result()
            tmean = ((tmin + tmax) / 2.0).astype(np.float32)
            tmean.name = "temperature"
            tmean_years.append(tmean.expand_dims(year=[year]))
            ppt_years.append(ppt.expand_dims(year=[year]))
            print(f"[Belgium] TerraClimate {year} loaded")

    temperature = xr.concat(tmean_years, dim="year").mean("year", skipna=True).astype(np.float32)
    precipitation = xr.concat(ppt_years, dim="year").mean("year", skipna=True).astype(np.float32)
    out = xr.Dataset({"temperature": temperature, "precipitation": precipitation})
    out.attrs["year_start"] = int(year_start)
    out.attrs["year_end"] = int(year_end)
    out.attrs["source"] = "TerraClimate"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(cache_path)
    out.close()
    return cache_path


def _masked_gaussian(data: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        out = np.asarray(data, dtype=np.float32).copy()
        out[~mask] = np.nan
        return out
    weights = mask.astype(np.float32)
    filled = np.where(mask, data, 0.0).astype(np.float32)
    smooth_data = gaussian_filter(filled, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.full(data.shape, np.nan, dtype=np.float32)
    valid = smooth_weights > 1e-6
    out[valid] = smooth_data[valid] / smooth_weights[valid]
    out[~mask] = np.nan
    return out


def _project_climatology_to_dem(
    climate_cache: Path,
    dem_path: Path,
    boundary_zip: Path,
    target_crs: str,
    *,
    smooth_sigma: float,
) -> RasterBundle:
    belgium_target = _load_belgium_geometry(boundary_zip, target_crs)
    climate = xr.open_dataset(climate_cache, engine="netcdf4")
    try:
        temperature_da = _prepare_geo_da(climate["temperature"])
        precipitation_da = _prepare_geo_da(climate["precipitation"])
        dem_da = rioxarray.open_rasterio(dem_path).squeeze(drop=True).rio.reproject(
            target_crs,
            resampling=Resampling.bilinear,
        )
        temperature_proj = temperature_da.rio.reproject_match(dem_da, resampling=Resampling.bilinear)
        precipitation_proj = precipitation_da.rio.reproject_match(
            dem_da, resampling=Resampling.bilinear
        )
        temperature = np.asarray(temperature_proj.values, dtype=np.float32)
        precipitation = np.asarray(precipitation_proj.values, dtype=np.float32)
        elevation = np.asarray(dem_da.values, dtype=np.float32)
        transform = dem_da.rio.transform()

        border_mask = geometry_mask(
            [belgium_target],
            out_shape=elevation.shape,
            transform=transform,
            invert=True,
        )
        valid = (
            np.isfinite(temperature)
            & np.isfinite(precipitation)
            & np.isfinite(elevation)
            & border_mask
        )
        if not np.any(valid):
            raise RuntimeError("Belgium polygon mask produced no valid raster cells")

        temperature = _masked_gaussian(temperature, valid, smooth_sigma)
        precipitation = _masked_gaussian(precipitation, valid, smooth_sigma)
        temperature[~valid] = np.nan
        precipitation[~valid] = np.nan
        elevation[~valid] = np.nan

        rows, cols = np.where(valid)
        row_min = max(int(rows.min()) - 1, 0)
        row_max = min(int(rows.max()) + 2, valid.shape[0])
        col_min = max(int(cols.min()) - 1, 0)
        col_max = min(int(cols.max()) + 2, valid.shape[1])
        crop_window = Window(
            col_off=col_min,
            row_off=row_min,
            width=col_max - col_min,
            height=row_max - row_min,
        )
        return RasterBundle(
            temperature=temperature[row_min:row_max, col_min:col_max],
            precipitation=precipitation[row_min:row_max, col_min:col_max],
            elevation=elevation[row_min:row_max, col_min:col_max],
            mask=valid[row_min:row_max, col_min:col_max],
            transform=window_transform(crop_window, transform),
            crs=target_crs,
        )
    finally:
        climate.close()


def _quantile_breaks(values: np.ndarray, quantiles: tuple[float, float, float]) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    edges = np.quantile(finite, quantiles)
    return [float(finite.min()), *[float(edge) for edge in edges], float(finite.max())]


def _choose_balanced_quantiles(
    temperature: np.ndarray,
    precipitation: np.ndarray,
    *,
    temp_override: tuple[float, float, float] | None,
    precip_override: tuple[float, float, float] | None,
    sample_size: int,
    seed: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if temp_override is not None and precip_override is not None:
        return temp_override, precip_override

    temp = np.asarray(temperature, dtype=np.float64)
    precip = np.asarray(precipitation, dtype=np.float64)
    valid = np.isfinite(temp) & np.isfinite(precip)
    temp = temp[valid]
    precip = precip[valid]
    if temp.size > sample_size:
        rng = np.random.default_rng(seed)
        indices = rng.choice(temp.size, size=sample_size, replace=False)
        temp = temp[indices]
        precip = precip[indices]

    quartiles = np.array([0.25, 0.50, 0.75], dtype=np.float64)
    temp_candidates = (temp_override,) if temp_override is not None else ASYMMETRIC_QUANTILE_CANDIDATES
    precip_candidates = (
        (precip_override,) if precip_override is not None else ASYMMETRIC_QUANTILE_CANDIDATES
    )
    best_score = -np.inf
    best = (ASYMMETRIC_QUANTILE_CANDIDATES[0], ASYMMETRIC_QUANTILE_CANDIDATES[0])

    for temp_q in temp_candidates:
        temp_edges = np.quantile(temp, temp_q)
        temp_classes = np.searchsorted(temp_edges, temp, side="right")
        for precip_q in precip_candidates:
            precip_edges = np.quantile(precip, precip_q)
            precip_classes = np.searchsorted(precip_edges, precip, side="right")
            counts = np.bincount(precip_classes * 4 + temp_classes, minlength=16).astype(np.float64)
            probs = counts / counts.sum()
            nz = probs > 0.0
            entropy = float(-np.sum(probs[nz] * np.log(probs[nz])))
            zero_bins = int(np.count_nonzero(counts == 0.0))
            nonzero = counts[counts > 0.0]
            balance = float(nonzero.min() / nonzero.max()) if nonzero.size else 0.0
            asym_bonus = float(
                np.abs(np.asarray(temp_q) - quartiles).sum()
                + np.abs(np.asarray(precip_q) - quartiles).sum()
            )
            score = entropy + (2.2 * balance) - (0.45 * zero_bins) + (0.05 * asym_bonus)
            if score > best_score:
                best_score = score
                best = (temp_q, precip_q)
    return best


def _classify(values: np.ndarray, breaks: list[float], valid_mask: np.ndarray) -> np.ndarray:
    edges = np.asarray(breaks[1:-1], dtype=np.float32)
    classes = np.zeros(values.shape, dtype=np.uint8)
    classes[valid_mask] = np.searchsorted(edges, values[valid_mask], side="right") + 1
    return classes


def _compute_border(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _build_texture(
    temperature_classes: np.ndarray,
    precipitation_classes: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    rgba = np.zeros((*valid_mask.shape, 4), dtype=np.uint8)
    for x in range(1, 5):
        for y in range(1, 5):
            rgb = _hex_to_rgb(_bivariate_palette_color(x, y))
            class_mask = valid_mask & (temperature_classes == x) & (precipitation_classes == y)
            rgba[class_mask, 0] = rgb[0]
            rgba[class_mask, 1] = rgb[1]
            rgba[class_mask, 2] = rgb[2]
            rgba[class_mask, 3] = 255
    border = _compute_border(valid_mask)
    rgba[border] = np.array([36, 36, 44, 255], dtype=np.uint8)
    return rgba


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    head: int,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head * math.cos(angle - math.pi / 6.0),
        end[1] - head * math.sin(angle - math.pi / 6.0),
    )
    right = (
        end[0] - head * math.cos(angle + math.pi / 6.0),
        end[1] - head * math.sin(angle + math.pi / 6.0),
    )
    draw.polygon((end, left, right), fill=fill)


def _make_legend(*, panel_alpha: int = 255) -> Image.Image:
    panel = Image.new("RGBA", (660, 500), (255, 255, 255, panel_alpha))
    draw = ImageDraw.Draw(panel)
    label_font = _load_font(34, bold=True)
    cell = 86
    x0 = 214
    y0 = 54
    grid_width = cell * 4
    axis_fill = (55, 55, 62, 255)

    for x in range(1, 5):
        for y in range(1, 5):
            top = y0 + (4 - y) * cell
            left = x0 + (x - 1) * cell
            draw.rectangle(
                (left, top, left + cell, top + cell),
                fill=_hex_to_rgb(_bivariate_palette_color(x, y)),
                outline=(70, 70, 80, 255),
                width=1,
            )

    x_arrow_y = y0 + grid_width + 44
    _draw_arrow(draw, (x0 + 6, x_arrow_y), (x0 + grid_width - 6, x_arrow_y), fill=axis_fill, width=6, head=18)
    x_label = "warmer"
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text((x0 + (grid_width - (x_box[2] - x_box[0])) / 2, x_arrow_y + 14), x_label, fill=axis_fill, font=label_font)

    y_arrow_x = x0 - 46
    _draw_arrow(draw, (y_arrow_x, y0 + grid_width - 6), (y_arrow_x, y0 + 6), fill=axis_fill, width=6, head=18)
    y_text = Image.new("RGBA", (240, 54), (0, 0, 0, 0))
    ImageDraw.Draw(y_text).text((0, 0), "rainier", fill=axis_fill, font=label_font)
    rotated = y_text.rotate(90, expand=True)
    panel.alpha_composite(rotated, dest=(58, y0 + (grid_width - rotated.height) // 2))
    return panel


def _measure_centered(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _compose_2d(texture_rgba: np.ndarray, output_path: Path, *, subtitle: str) -> None:
    texture = Image.fromarray(texture_rgba, mode="RGBA")
    scaled = texture.resize(
        (int(texture.width * MAP_SCALE_2D), int(texture.height * MAP_SCALE_2D)),
        Image.Resampling.NEAREST,
    )
    legend = _make_legend(panel_alpha=255).resize((1600, 1210), Image.Resampling.LANCZOS)
    width = max(scaled.width + 70, 2200)
    footer_height = 92
    top_margin = 18
    title_gap = 20
    subtitle_gap = 26
    title_font = _load_font(66, bold=True)
    subtitle_font = _load_font(36)
    caption_font = _load_font(28)
    metrics_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (255, 255, 255, 0)))

    title_w, title_h = _measure_text(metrics_draw, MAP_TITLE, title_font)
    subtitle_w, subtitle_h = _measure_text(metrics_draw, subtitle, subtitle_font)
    caption_w, caption_h = _measure_text(metrics_draw, MAP_CAPTION, caption_font)
    title_y = top_margin
    subtitle_y = title_y + title_h + title_gap
    map_y = subtitle_y + subtitle_h + subtitle_gap
    height = map_y + scaled.height + 22 + footer_height
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    map_x = (width - scaled.width) // 2
    canvas.alpha_composite(scaled, dest=(map_x, map_y))

    draw.text(((width - title_w) / 2, title_y), MAP_TITLE, fill=(35, 35, 40, 255), font=title_font)
    draw.text(((width - subtitle_w) / 2, subtitle_y), subtitle, fill=(70, 70, 78, 255), font=subtitle_font)
    canvas.alpha_composite(legend, dest=(34, height - footer_height - legend.height - 18))
    footer_y = height - footer_height
    draw.rectangle((0, footer_y, width, height), fill=(247, 247, 249, 255))
    draw.line((0, footer_y, width, footer_y), fill=(222, 222, 228, 255), width=2)
    draw.text(
        ((width - caption_w) / 2, footer_y + (footer_height - caption_h) / 2 - 1),
        MAP_CAPTION,
        fill=(58, 58, 66, 255),
        font=caption_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_dem(path: Path, bundle: RasterBundle) -> Path:
    data = np.asarray(bundle.elevation, dtype=np.float32).copy()
    if not np.all(bundle.mask):
        indices = distance_transform_edt(~bundle.mask, return_distances=False, return_indices=True)
        data[~bundle.mask] = data[indices[0][~bundle.mask], indices[1][~bundle.mask]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs=bundle.crs,
        transform=bundle.transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)
    return path


def _save_texture(path: Path, texture_rgba: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture_rgba, mode="RGBA").save(path)
    return path


def _annotate_3d_snapshot(raw_path: Path, final_path: Path, *, subtitle: str) -> None:
    image = Image.open(raw_path).convert("RGBA")
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(1.18)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
    image = Image.merge("RGBA", (*rgb.split(), image.getchannel("A")))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = _load_font(max(44, image.width // 36), bold=True)
    subtitle_font = _load_font(max(26, image.width // 62))
    caption_font = _load_font(max(24, image.width // 82))
    legend = _make_legend(panel_alpha=238).resize((800, 606), Image.Resampling.LANCZOS)

    title_w, title_h = _measure_text(draw, MAP_TITLE, title_font)
    subtitle_w, subtitle_h = _measure_text(draw, subtitle, subtitle_font)
    caption_w, caption_h = _measure_text(draw, MAP_CAPTION, caption_font)
    panel_margin = max(18, image.width // 70)
    top_panel_height = max(152, title_h + subtitle_h + 72, image.height // 8)
    top_panel = (
        panel_margin,
        panel_margin,
        image.width - panel_margin,
        panel_margin + top_panel_height,
    )
    title_y = top_panel[1] + 16
    subtitle_y = title_y + title_h + 18
    draw.rounded_rectangle(top_panel, radius=24, fill=(255, 255, 255, 205))
    draw.text(((image.width - title_w) / 2, title_y), MAP_TITLE, fill=(25, 25, 30, 255), font=title_font)
    draw.text(((image.width - subtitle_w) / 2, subtitle_y), subtitle, fill=(58, 58, 66, 255), font=subtitle_font)
    overlay.alpha_composite(legend, dest=(panel_margin, image.height - legend.height - panel_margin))
    caption_pad_x = max(18, image.width // 110)
    caption_pad_y = max(12, image.height // 170)
    caption_panel = (
        image.width - panel_margin - caption_w - (caption_pad_x * 2),
        image.height - panel_margin - caption_h - (caption_pad_y * 2),
        image.width - panel_margin,
        image.height - panel_margin,
    )
    draw.rounded_rectangle(
        caption_panel,
        radius=22,
        fill=(255, 255, 255, 218),
        outline=(255, 255, 255, 240),
        width=2,
    )
    draw.text(
        (caption_panel[0] + caption_pad_x, caption_panel[1] + caption_pad_y - 1),
        MAP_CAPTION,
        fill=(30, 30, 36, 255),
        font=caption_font,
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).save(final_path)


def _render_3d(
    dem_path: Path,
    texture_path: Path,
    raw_output_path: Path,
    final_output_path: Path,
    *,
    subtitle: str,
    viewer_size: tuple[int, int],
    snapshot_size: tuple[int, int],
    cam_phi: float,
    cam_theta: float,
    cam_radius: float,
    cam_fov: float,
) -> None:
    with rasterio.open(dem_path) as dem_src:
        terrain_width = float(max(dem_src.width, dem_src.height))
    terrain_xy_scale = terrain_width / REFERENCE_TERRAIN_WIDTH
    terrain_relief_scale = math.sqrt(terrain_xy_scale)
    camera_radius = cam_radius * terrain_xy_scale
    zscale = BASE_TERRAIN_ZSCALE * terrain_relief_scale
    ao_distance = 220.0 * terrain_relief_scale

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    with f3d.open_viewer_async(
        terrain_path=dem_path,
        width=viewer_size[0],
        height=viewer_size[1],
        fov_deg=32.0,
        timeout=45.0,
    ) as viewer:
        viewer.send_ipc(
            {
                "cmd": "set_terrain",
                "phi": cam_phi,
                "theta": cam_theta,
                "radius": camera_radius,
                "fov": cam_fov,
                "zscale": zscale,
                "sun_azimuth": 145.0,
                "sun_elevation": 34.0,
                "sun_intensity": 1.10,
                "ambient": 0.20,
                "shadow": 0.34,
                "background": [0.955, 0.962, 0.978],
            }
        )
        viewer.send_ipc(
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "shadow_technique": "pcss",
                "shadow_map_res": 4096,
                "exposure": 1.00,
                "msaa": 4,
                "normal_strength": 1.10,
                "height_ao": {
                    "enabled": True,
                    "directions": 8,
                    "steps": 18,
                    "max_distance": ao_distance,
                    "strength": 0.34,
                    "resolution_scale": 0.5,
                },
            }
        )
        viewer.load_overlay(
            name="belgium_climate",
            path=texture_path,
            extent=(0.0, 0.0, 1.0, 1.0),
            opacity=0.92,
            z_order=0,
        )
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
        viewer.snapshot(raw_output_path, width=snapshot_size[0], height=snapshot_size[1])
    _annotate_3d_snapshot(raw_output_path, final_output_path, subtitle=subtitle)


def _print_summary(
    bundle: RasterBundle,
    *,
    temp_quantiles: tuple[float, float, float],
    precip_quantiles: tuple[float, float, float],
    temp_breaks: list[float],
    precip_breaks: list[float],
    output_dir: Path,
) -> None:
    temp_valid = bundle.temperature[bundle.mask]
    precip_valid = bundle.precipitation[bundle.mask]
    elev_valid = bundle.elevation[bundle.mask]
    print(f"[Belgium] grid={bundle.temperature.shape[1]}x{bundle.temperature.shape[0]} crs={bundle.crs}")
    print(f"[Belgium] temperature range={float(np.nanmin(temp_valid)):.2f}..{float(np.nanmax(temp_valid)):.2f} C")
    print(f"[Belgium] precipitation range={float(np.nanmin(precip_valid)):.1f}..{float(np.nanmax(precip_valid)):.1f} mm")
    print(f"[Belgium] elevation range={float(np.nanmin(elev_valid)):.1f}..{float(np.nanmax(elev_valid)):.1f} m")
    print(f"[Belgium] temperature quantiles={', '.join(f'{value:.2f}' for value in temp_quantiles)}")
    print(f"[Belgium] precipitation quantiles={', '.join(f'{value:.2f}' for value in precip_quantiles)}")
    print(f"[Belgium] temperature breaks={', '.join(f'{value:.2f}' for value in temp_breaks)}")
    print(f"[Belgium] precipitation breaks={', '.join(f'{value:.1f}' for value in precip_breaks)}")
    print(f"[Belgium] outputs={output_dir}")


def main() -> int:
    args = _parse_args()
    if int(args.year_end) < int(args.year_start):
        raise ValueError("--year-end must be greater than or equal to --year-start")

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    subtitle = _subtitle(int(args.year_start), int(args.year_end))
    temp_override = _parse_quantiles(str(args.temp_quantiles))
    precip_override = _parse_quantiles(str(args.precip_quantiles))

    boundary_zip = _download(
        NATURAL_EARTH_COUNTRIES_URL,
        cache_dir / "ne_10m_admin_0_countries.zip",
        force=bool(args.force_download),
    )
    climate_cache = _build_terraclimate_climatology(
        boundary_zip,
        cache_dir,
        year_start=int(args.year_start),
        year_end=int(args.year_end),
        pad_deg=float(args.terraclimate_pad_deg),
        force=bool(args.force_download),
    )
    source_dem_path = _build_terrarium_dem(
        boundary_zip,
        cache_dir,
        int(args.dem_zoom),
        force=bool(args.force_download),
    )
    bundle = _project_climatology_to_dem(
        climate_cache,
        source_dem_path,
        boundary_zip,
        str(args.target_crs),
        smooth_sigma=float(args.smooth_sigma),
    )

    temp_quantiles, precip_quantiles = _choose_balanced_quantiles(
        bundle.temperature[bundle.mask],
        bundle.precipitation[bundle.mask],
        temp_override=temp_override,
        precip_override=precip_override,
        sample_size=int(args.quantile_sample),
        seed=int(args.seed),
    )
    temp_breaks = _quantile_breaks(bundle.temperature[bundle.mask], temp_quantiles)
    precip_breaks = _quantile_breaks(bundle.precipitation[bundle.mask], precip_quantiles)

    temperature_classes = _classify(bundle.temperature, temp_breaks, bundle.mask)
    precipitation_classes = _classify(bundle.precipitation, precip_breaks, bundle.mask)
    texture_rgba = _build_texture(temperature_classes, precipitation_classes, bundle.mask)
    map_texture_path = _save_texture(output_dir / "belgium_bivariate_texture.png", texture_rgba)
    dem_path = _write_dem(output_dir / "belgium_dem_3035.tif", bundle)
    map_2d_path = output_dir / "belgium_bivariate_2d.png"
    _compose_2d(texture_rgba, map_2d_path, subtitle=subtitle)

    raw_3d_path = output_dir / "belgium_bivariate_3d_raw.png"
    final_3d_path = output_dir / "belgium_bivariate_3d.png"
    _print_summary(
        bundle,
        temp_quantiles=temp_quantiles,
        precip_quantiles=precip_quantiles,
        temp_breaks=temp_breaks,
        precip_breaks=precip_breaks,
        output_dir=output_dir,
    )
    print(f"[Belgium] wrote {climate_cache.name}")
    print(f"[Belgium] wrote {map_texture_path.name}")
    print(f"[Belgium] wrote {dem_path.name}")
    print(f"[Belgium] wrote {map_2d_path.name}")

    if args.skip_3d:
        print("[Belgium] skipping 3D render (--skip-3d)")
        return 0

    _render_3d(
        dem_path=dem_path,
        texture_path=map_texture_path,
        raw_output_path=raw_3d_path,
        final_output_path=final_3d_path,
        subtitle=subtitle,
        viewer_size=(int(args.viewer_width), int(args.viewer_height)),
        snapshot_size=(int(args.snapshot_width), int(args.snapshot_height)),
        cam_phi=float(args.cam_phi),
        cam_theta=float(args.cam_theta),
        cam_radius=float(args.cam_radius),
        cam_fov=float(args.cam_fov),
    )
    print(f"[Belgium] wrote {final_3d_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
