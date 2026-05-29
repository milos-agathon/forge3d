#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from osgeo import ogr, osr
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import transform as rio_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if PYTHON_DIR.is_dir():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d


ogr.UseExceptions()
osr.UseExceptions()

COUNTRY_URL = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_01M_2024_4326.geojson"
HYDRO_RIVERS_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip"
HYDRO_BASINS_URL = "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_eu_lev04_v1c.zip"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HDR_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/limpopo_golf_course_4k.hdr"
CRS_LAEA = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +datum=WGS84 +units=m +no_defs"
POLAND_BBOX_WGS84 = (14.12290, 49.00285, 24.14544, 54.83568)
TILE_SIZE = 256
DEFAULT_CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "poland_river_basins"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "poland_river_basins" / "poland-3d-river-basins.png"
MAP_TITLE = "3D River Basin Map of Poland"
MAP_AUTHOR = "Milos Popovic | milosgis.com"
MAP_CAPTION = (
    "Sources: GISCO country boundary, HydroSHEDS HydroBASINS level 04 + HydroRIVERS, "
    "AWS Terrain Tiles, Polyhaven HDRI | Copyright 2026 Milos Popovic"
)

PALETTE = [
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#1F78B4",
    "#B2DF8A",
    "#B15928",
]
OVERLAY_STYLE_VERSION = 8
OVERLAY_TEXTURE_SCALE = 2
BASIN_FILL_ALPHA = 170
BASIN_FILL_LIGHTEN_AMOUNT = 0.30
RIVER_LIGHTEN_AMOUNT = 0.58


@dataclass
class Basin:
    hybas_id: int
    geom_wgs84: ogr.Geometry


@dataclass
class RiverSegment:
    hybas_id: int
    ord_flow: int
    width_px: int
    geom_wgs84: ogr.Geometry


@dataclass
class RasterFrame:
    heightmap: np.ndarray
    mask: np.ndarray
    transform: object
    bounds: tuple[float, float, float, float]
    crs: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Poland 3D river basin map with HydroSHEDS data and forge3d."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dem-zoom", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=1700, help="Largest DEM grid dimension.")
    parser.add_argument("--render-width", type=int, default=4096)
    parser.add_argument("--render-height", type=int, default=3724)
    parser.add_argument("--viewer-width", type=int, default=1400)
    parser.add_argument("--viewer-height", type=int, default=1100)
    parser.add_argument("--zscale", type=float, default=0.30)
    parser.add_argument("--no-hdri", action="store_true", help="Do not use the Polyhaven HDRI environment.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached derived products.")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after DEM and overlay generation.")
    return parser.parse_args()


def _srs(value: str | int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    if isinstance(value, int):
        srs.ImportFromEPSG(value)
    else:
        srs.SetFromUserInput(value)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


WGS84_SRS = _srs(4326)
LAEA_SRS = _srs(CRS_LAEA)
WGS84_TO_LAEA = osr.CoordinateTransformation(WGS84_SRS, LAEA_SRS)


def _download(url: str, path: Path, *, force: bool = False) -> Path:
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "forge3d-poland-river-basins/1.0"})
    print(f"[download] {url}")
    with urlopen(request, timeout=180) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(path)
    return path


def _unzip(zip_path: Path, dest_dir: Path, *, force: bool = False) -> Path:
    if dest_dir.exists() and any(dest_dir.rglob("*.shp")) and not force:
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    return dest_dir


def _find_shp(root: Path, token: str) -> Path:
    matches = [p for p in root.rglob("*.shp") if token.lower() in p.name.lower()]
    if not matches:
        matches = list(root.rglob("*.shp"))
    if not matches:
        raise FileNotFoundError(f"No shapefile found in {root}")
    return matches[0]


def _drop_spatial_index_sidecars(shp: Path) -> None:
    for suffix in (".sbn", ".sbx", ".qix"):
        shp.with_suffix(suffix).unlink(missing_ok=True)


def _field_names(layer: ogr.Layer) -> list[str]:
    definition = layer.GetLayerDefn()
    return [definition.GetFieldDefn(i).GetName() for i in range(definition.GetFieldCount())]


def _feature_field(feature: ogr.Feature, names: list[str], candidates: tuple[str, ...]) -> object | None:
    present = {name.upper(): name for name in names}
    for candidate in candidates:
        name = present.get(candidate.upper())
        if name is not None:
            return feature.GetField(name)
    return None


def load_poland_geometry(cache_dir: Path, *, force: bool) -> ogr.Geometry:
    country_path = _download(COUNTRY_URL, cache_dir / "CNTR_RG_01M_2024_4326.geojson", force=force)
    dataset = ogr.Open(str(country_path))
    if dataset is None:
        raise RuntimeError(f"Could not open {country_path}")
    layer = dataset.GetLayer(0)
    names = _field_names(layer)
    for feature in layer:
        iso2 = _feature_field(feature, names, ("CNTR_ID", "ISO2_CODE", "ISO_A2"))
        iso3 = _feature_field(feature, names, ("ISO3_CODE", "ISO_A3"))
        name = str(_feature_field(feature, names, ("NAME_ENGL", "NAME_EN", "NAME")) or "")
        if iso2 == "PL" or iso3 == "POL" or name.lower() == "poland":
            geom = feature.GetGeometryRef().Clone()
            geom.AssignSpatialReference(WGS84_SRS)
            return geom.MakeValid()
    raise RuntimeError("Poland geometry not found in GISCO countries layer")


def _safe_intersection(a: ogr.Geometry, b: ogr.Geometry) -> ogr.Geometry | None:
    try:
        result = a.Intersection(b)
    except RuntimeError:
        result = a.MakeValid().Intersection(b.MakeValid())
    if result is None or result.IsEmpty():
        return None
    return result.MakeValid()


def load_basins(cache_dir: Path, country: ogr.Geometry, *, force: bool) -> list[Basin]:
    zip_path = _download(HYDRO_BASINS_URL, cache_dir / "hybas_eu_lev04_v1c.zip", force=False)
    shp = _find_shp(_unzip(zip_path, cache_dir / "hybas_eu_lev04_v1c", force=force), "hybas")
    _drop_spatial_index_sidecars(shp)
    dataset = ogr.Open(str(shp))
    if dataset is None:
        raise RuntimeError(f"Could not open {shp}")
    layer = dataset.GetLayer(0)
    layer.SetSpatialFilter(country)
    basins: list[Basin] = []
    for feature in layer:
        geom_ref = feature.GetGeometryRef()
        if geom_ref is None:
            continue
        clipped = _safe_intersection(geom_ref, country)
        if clipped is None:
            continue
        basins.append(Basin(int(feature.GetField("HYBAS_ID")), clipped))
    basins.sort(key=lambda item: item.hybas_id)
    return basins


def _river_width(ord_flow: int) -> int:
    return {3: 10, 4: 7, 5: 4, 6: 2}.get(int(ord_flow), 0)


def load_rivers(cache_dir: Path, country: ogr.Geometry, basins: list[Basin], *, force: bool) -> list[RiverSegment]:
    zip_path = _download(HYDRO_RIVERS_URL, cache_dir / "HydroRIVERS_v10_eu_shp.zip", force=False)
    shp = _find_shp(_unzip(zip_path, cache_dir / "HydroRIVERS_v10_eu_shp", force=force), "rivers")
    _drop_spatial_index_sidecars(shp)
    dataset = ogr.Open(str(shp))
    if dataset is None:
        raise RuntimeError(f"Could not open {shp}")
    layer = dataset.GetLayer(0)
    min_lon, max_lon, min_lat, max_lat = country.GetEnvelope()
    layer.SetSpatialFilterRect(min_lon, min_lat, max_lon, max_lat)

    segments: list[RiverSegment] = []
    for feature in layer:
        ord_flow = int(feature.GetField("ORD_FLOW") or 99)
        width = _river_width(ord_flow)
        if width <= 0:
            continue
        geom_ref = feature.GetGeometryRef()
        if geom_ref is None:
            continue
        try:
            if not geom_ref.Intersects(country):
                continue
        except RuntimeError:
            pass
        for basin in basins:
            try:
                if not geom_ref.Intersects(basin.geom_wgs84):
                    continue
            except RuntimeError:
                pass
            clipped = _safe_intersection(geom_ref, basin.geom_wgs84)
            if clipped is not None:
                segments.append(RiverSegment(basin.hybas_id, ord_flow, width, clipped))
    segments.sort(key=lambda item: (item.width_px, item.ord_flow))
    return segments


def _tile_x_from_lon(lon: float, zoom: int) -> int:
    n = 1 << int(zoom)
    return int(np.clip(((float(lon) + 180.0) / 360.0) * n, 0, n - 1))


def _tile_y_from_lat(lat: float, zoom: int) -> int:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 1 << int(zoom)
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) * 0.5 * n
    return int(np.clip(y, 0, n - 1))


def _decode_terrarium(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return (rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0) - 32768.0


def _fetch_terrarium_mosaic(
    cache_dir: Path,
    bbox: tuple[float, float, float, float],
    zoom: int,
) -> tuple[np.ndarray, int, int]:
    lon_min, lat_min, lon_max, lat_max = bbox
    x_min = _tile_x_from_lon(lon_min, zoom)
    x_max = _tile_x_from_lon(lon_max, zoom)
    y_min = _tile_y_from_lat(lat_max, zoom)
    y_max = _tile_y_from_lat(lat_min, zoom)
    tiles = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    print(f"[dem] fetching {len(tiles)} Terrarium tiles at z={zoom}")

    def fetch(tile: tuple[int, int]) -> tuple[tuple[int, int], Path]:
        x, y = tile
        path = cache_dir / "terrarium" / str(zoom) / str(x) / f"{y}.png"
        return tile, _download(TERRARIUM_URL.format(z=zoom, x=x, y=y), path)

    fetched: list[tuple[tuple[int, int], Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for item in executor.map(fetch, tiles):
            fetched.append(item)

    rows = y_max - y_min + 1
    cols = x_max - x_min + 1
    mosaic = np.empty((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)
    for (x, y), path in fetched:
        r0 = (y - y_min) * TILE_SIZE
        c0 = (x - x_min) * TILE_SIZE
        mosaic[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = _decode_terrarium(path)
    return mosaic, x_min, y_min


def _slippy_pixel_xy(lon: np.ndarray, lat: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05112878, 85.05112878)
    lon = np.asarray(lon, dtype=np.float64)
    scale = float((1 << int(zoom)) * TILE_SIZE)
    x = (lon + 180.0) / 360.0 * scale
    y = (1.0 - np.arcsinh(np.tan(np.deg2rad(lat))) / math.pi) * 0.5 * scale
    return x, y


def _sample_terrarium(
    mosaic: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    zoom: int,
    origin_x: int,
    origin_y: int,
) -> np.ndarray:
    px, py = _slippy_pixel_xy(lon, lat, zoom)
    lx = px - float(origin_x * TILE_SIZE)
    ly = py - float(origin_y * TILE_SIZE)
    height, width = mosaic.shape
    out = np.full(lx.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(lx)
        & np.isfinite(ly)
        & (lx >= 0.0)
        & (ly >= 0.0)
        & (lx < width - 1.0)
        & (ly < height - 1.0)
    )
    if not np.any(valid):
        return out
    x0 = np.floor(lx[valid]).astype(np.int32)
    y0 = np.floor(ly[valid]).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (lx[valid] - x0).astype(np.float32)
    ty = (ly[valid] - y0).astype(np.float32)
    sampled = (
        mosaic[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + mosaic[y0, x1] * tx * (1.0 - ty)
        + mosaic[y1, x0] * (1.0 - tx) * ty
        + mosaic[y1, x1] * tx * ty
    )
    out[valid] = sampled.astype(np.float32)
    return out


def _laea_bounds(country: ogr.Geometry) -> tuple[float, float, float, float]:
    geom = country.Clone()
    geom.Transform(WGS84_TO_LAEA)
    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    pad = max(max_x - min_x, max_y - min_y) * 0.025
    return min_x - pad, min_y - pad, max_x + pad, max_y + pad


def _grid_shape(bounds: tuple[float, float, float, float], max_size: int) -> tuple[int, int]:
    left, bottom, right, top = bounds
    width_m = right - left
    height_m = top - bottom
    if width_m >= height_m:
        cols = int(max_size)
        rows = max(1, int(round(cols * height_m / width_m)))
    else:
        rows = int(max_size)
        cols = max(1, int(round(rows * width_m / height_m)))
    return rows, cols


def _country_mask(country: ogr.Geometry, shape: tuple[int, int], transform: object) -> np.ndarray:
    geom = country.Clone()
    geom.Transform(WGS84_TO_LAEA)
    return geometry_mask(
        [json.loads(geom.ExportToJson())],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )


def build_dem(
    cache_dir: Path,
    country: ogr.Geometry,
    *,
    zoom: int,
    max_size: int,
    force: bool,
) -> tuple[Path, RasterFrame]:
    bounds = _laea_bounds(country)
    rows, cols = _grid_shape(bounds, max_size)
    dem_path = cache_dir / f"poland_dem_laea_z{zoom}_{cols}x{rows}.tif"
    if dem_path.exists() and not force:
        with rasterio.open(dem_path) as src:
            heightmap = src.read(1).astype(np.float32)
            mask = _country_mask(country, heightmap.shape, src.transform)
            return dem_path, RasterFrame(heightmap, mask, src.transform, src.bounds, str(src.crs))

    mosaic, origin_x, origin_y = _fetch_terrarium_mosaic(cache_dir, POLAND_BBOX_WGS84, zoom)
    left, bottom, right, top = bounds
    xs = left + (np.arange(cols, dtype=np.float64) + 0.5) * ((right - left) / cols)
    ys = top - (np.arange(rows, dtype=np.float64) + 0.5) * ((top - bottom) / rows)
    heightmap = np.empty((rows, cols), dtype=np.float32)
    chunk_rows = 96
    print(f"[dem] sampling {cols}x{rows} LAEA grid")
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        grid_x, grid_y = np.meshgrid(xs, ys[start:end])
        lon, lat = rio_transform(CRS_LAEA, "EPSG:4326", grid_x.ravel(), grid_y.ravel())
        sampled = _sample_terrarium(
            mosaic,
            np.asarray(lon).reshape(end - start, cols),
            np.asarray(lat).reshape(end - start, cols),
            zoom=zoom,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        heightmap[start:end] = sampled

    transform = from_bounds(left, bottom, right, top, cols, rows)
    mask = _country_mask(country, heightmap.shape, transform)
    finite = np.isfinite(heightmap)
    valid = finite & mask
    if not np.any(valid):
        raise RuntimeError("DEM sampling produced no valid pixels inside Poland")
    fill = float(np.nanpercentile(heightmap[valid], 1.0))
    heightmap = np.where(finite, heightmap, fill).astype(np.float32)
    heightmap[~mask] = fill
    heightmap -= float(np.min(heightmap[mask]))
    heightmap = np.maximum(heightmap, 0.0).astype(np.float32)

    dem_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=cols,
        height=rows,
        count=1,
        dtype="float32",
        crs=CRS_LAEA,
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(heightmap, 1)
    return dem_path, RasterFrame(heightmap, mask, transform, bounds, CRS_LAEA)


def _hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), int(alpha)


def _lightened_hex_to_rgba(color: str, alpha: int, amount: float = 0.52) -> tuple[int, int, int, int]:
    r, g, b, _ = _hex_to_rgba(color, alpha)
    mix = float(np.clip(amount, 0.0, 1.0))
    return (
        int(round(r * (1.0 - mix) + 255.0 * mix)),
        int(round(g * (1.0 - mix) + 255.0 * mix)),
        int(round(b * (1.0 - mix) + 255.0 * mix)),
        int(alpha),
    )


def _palette_for(ids: list[int]) -> dict[int, str]:
    colors = PALETTE.copy()
    random.Random(20231121).shuffle(colors)
    return {hybas_id: colors[i % len(colors)] for i, hybas_id in enumerate(sorted(ids))}


def _xy_to_pixel(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    scale: int,
) -> tuple[float, float]:
    rows, cols = shape
    left, bottom, right, top = bounds
    px = (float(x) - left) / max(right - left, 1e-6) * cols * scale
    py = (top - float(y)) / max(top - bottom, 1e-6) * rows * scale
    return px, py


def _draw_polygon_geom(
    draw: ImageDraw.ImageDraw,
    geom: ogr.Geometry,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    scale: int,
    fill: tuple[int, int, int, int],
) -> None:
    name = geom.GetGeometryName().upper()
    if name == "POLYGON":
        ring = geom.GetGeometryRef(0)
        if ring is None:
            return
        points = [_xy_to_pixel(x, y, bounds, shape, scale) for x, y, *_ in ring.GetPoints()]
        if len(points) >= 3:
            draw.polygon(points, fill=fill)
        return
    if "MULTI" in name or name == "GEOMETRYCOLLECTION":
        for index in range(geom.GetGeometryCount()):
            _draw_polygon_geom(draw, geom.GetGeometryRef(index), bounds, shape, scale, fill)


def _draw_line_geom(
    draw: ImageDraw.ImageDraw,
    geom: ogr.Geometry,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    scale: int,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    name = geom.GetGeometryName().upper()
    if name in {"LINESTRING", "LINEARRING"}:
        points = [_xy_to_pixel(x, y, bounds, shape, scale) for x, y, *_ in geom.GetPoints()]
        if len(points) >= 2:
            draw.line(points, fill=fill, width=max(1, int(round(width * scale))), joint="curve")
        return
    if "MULTI" in name or name == "GEOMETRYCOLLECTION":
        for index in range(geom.GetGeometryCount()):
            _draw_line_geom(draw, geom.GetGeometryRef(index), bounds, shape, scale, fill, width)


def build_overlay(
    cache_dir: Path,
    basins: list[Basin],
    rivers: list[RiverSegment],
    frame: RasterFrame,
    *,
    force: bool,
) -> tuple[Path, dict[int, str]]:
    rows, cols = frame.heightmap.shape
    scale = OVERLAY_TEXTURE_SCALE
    overlay_path = cache_dir / (
        f"poland_basins_rivers_overlay_v{OVERLAY_STYLE_VERSION}_{cols * scale}x{rows * scale}.png"
    )
    ids = sorted({basin.hybas_id for basin in basins})
    palette = _palette_for(ids)
    if overlay_path.exists() and not force:
        return overlay_path, palette

    image = Image.new("RGBA", (cols * scale, rows * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    print(f"[overlay] drawing {len(basins)} basin polygons")
    for basin in basins:
        geom = basin.geom_wgs84.Clone()
        geom.Transform(WGS84_TO_LAEA)
        _draw_polygon_geom(
            draw,
            geom,
            frame.bounds,
            frame.heightmap.shape,
            scale,
            _lightened_hex_to_rgba(
                palette[basin.hybas_id],
                BASIN_FILL_ALPHA,
                amount=BASIN_FILL_LIGHTEN_AMOUNT,
            ),
        )

    print(f"[overlay] drawing {len(rivers)} clipped river segments")
    for river in rivers:
        geom = river.geom_wgs84.Clone()
        geom.Transform(WGS84_TO_LAEA)
        _draw_line_geom(
            draw,
            geom,
            frame.bounds,
            frame.heightmap.shape,
            scale,
            _lightened_hex_to_rgba(palette[river.hybas_id], 215, amount=RIVER_LIGHTEN_AMOUNT),
            river.width_px,
        )

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(overlay_path)
    return overlay_path, palette


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"]
        if bold
        else ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_font(text: str, max_width: int, preferred_size: int, min_size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    for size in range(int(preferred_size), int(min_size) - 1, -1):
        font = _load_font(size, bold=bold)
        left, _, right, _ = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
        if right - left <= max_width:
            return font
    return _load_font(min_size, bold=bold)


def compose_map_text(path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    margin_x = max(64, int(width * 0.045))
    title_font = _fit_font(MAP_TITLE, int(width * 0.86), max(44, int(width * 0.040)), 30, bold=True)
    author_font = _fit_font(MAP_AUTHOR, int(width * 0.78), max(24, int(width * 0.019)), 18, bold=False)
    caption_font = _fit_font(MAP_CAPTION, int(width * 0.92), max(19, int(width * 0.014)), 13, bold=False)

    title_y = max(52, int(height * 0.042))
    draw.text((margin_x, title_y), MAP_TITLE, fill=(24, 24, 24, 255), font=title_font)
    author_y = title_y + int(getattr(title_font, "size", 44) * 1.12)
    draw.text((margin_x, author_y), MAP_AUTHOR, fill=(68, 68, 68, 255), font=author_font)
    caption_y = height - max(86, int(height * 0.052))
    draw.text((margin_x, caption_y), MAP_CAPTION, fill=(74, 74, 74, 245), font=caption_font)
    image.save(path)


def render_with_forge3d(
    dem_path: Path,
    overlay_path: Path,
    output_path: Path,
    frame: RasterFrame,
    *,
    viewer_size: tuple[int, int],
    render_size: tuple[int, int],
    zscale: float,
    hdr_path: Path | None,
) -> None:
    if not f3d.has_gpu():
        raise RuntimeError("forge3d GPU runtime is required for this map")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    terrain_width = float(max(frame.heightmap.shape))
    terrain_cmd = {
        "phi": 89.0,
        "theta": 0.0,
        "radius": terrain_width * 3.35,
        "fov": 18.5,
        "zscale": float(zscale),
        "sun_azimuth": 315.0,
        "sun_elevation": 30.0,
        "sun_intensity": 1.35,
        "ambient": 0.52,
        "shadow": 0.34,
        "background": [1.0, 1.0, 1.0],
    }
    pbr_config = {
        "enabled": True,
        "shadow_technique": "pcss",
        "shadow_map_res": 4096,
        "exposure": 1.03,
        "msaa": 4,
        "ibl_intensity": 0.72 if hdr_path is not None else 0.22,
        "normal_strength": 1.18,
        "height_ao": {
            "enabled": True,
            "directions": 8,
            "steps": 18,
            "max_distance": 260.0,
            "strength": 0.28,
            "resolution_scale": 0.65,
        },
        "sun_visibility": {
            "enabled": True,
            "mode": "soft",
            "samples": 1,
            "strength": 0.22,
            "softness": 0.42,
        },
        "tonemap": {
            "operator": "aces",
            "white_point": 3.0,
            "white_balance_enabled": True,
            "temperature": 6500.0,
            "tint": 0.0,
        },
    }
    if hdr_path is not None:
        pbr_config["hdr_path"] = str(hdr_path)

    print("[render] launching forge3d viewer")
    with f3d.open_viewer_async(
        terrain_path=dem_path,
        width=int(viewer_size[0]),
        height=int(viewer_size[1]),
        timeout=60.0,
    ) as viewer:
        viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
        viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr_config})
        viewer.load_overlay(
            "poland_river_basins",
            overlay_path,
            extent=(0.0, 0.0, 1.0, 1.0),
            opacity=1.0,
            z_order=10,
            preserve_colors=True,
        )
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
        viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
        time.sleep(2.0)
        viewer.snapshot(output_path, width=int(render_size[0]), height=int(render_size[1]))
    compose_map_text(output_path)
    print(f"[render] wrote {output_path}")


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    output_path = args.output.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[data] loading Poland boundary")
    country = load_poland_geometry(cache_dir, force=bool(args.force))
    print("[data] loading HydroBASINS level 04")
    basins = load_basins(cache_dir, country, force=bool(args.force))
    if not basins:
        raise SystemExit("No HydroBASINS polygons intersected Poland")
    print(f"[data] basins: {len(basins)}")

    print("[data] loading HydroRIVERS and clipping to basins")
    rivers = load_rivers(cache_dir, country, basins, force=bool(args.force))
    if not rivers:
        raise SystemExit("No HydroRIVERS segments intersected Poland basins")
    print(f"[data] river segments: {len(rivers)}")

    dem_path, frame = build_dem(
        cache_dir,
        country,
        zoom=int(args.dem_zoom),
        max_size=int(args.max_size),
        force=bool(args.force),
    )
    overlay_path, _ = build_overlay(cache_dir, basins, rivers, frame, force=bool(args.force))
    print(f"[artifact] DEM: {dem_path}")
    print(f"[artifact] overlay: {overlay_path}")

    if args.prepare_only:
        return 0

    hdr_path = None
    if not args.no_hdri:
        hdr_path = _download(HDR_URL, cache_dir / Path(HDR_URL).name, force=False)

    render_with_forge3d(
        dem_path,
        overlay_path,
        output_path,
        frame,
        viewer_size=(int(args.viewer_width), int(args.viewer_height)),
        render_size=(int(args.render_width), int(args.render_height)),
        zscale=float(args.zscale),
        hdr_path=hdr_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
