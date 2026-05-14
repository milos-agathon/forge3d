#!/usr/bin/env python3
"""Germany land-cover terrain render."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import rasterio
from PIL import Image
from rasterio.enums import Resampling

import bosnia_terrain_landcover_viewer as viewer

ROOT = Path(__file__).resolve().parents[1]

LANDCOVER_BASE_URL = (
    "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/"
    "lc2024/{tile}_20240101-20241231.tif"
)
GERMANY_LANDCOVER_TILES = ("31U", "32T", "32U", "33T", "33U")
OSM_GERMANY_BOUNDARY = "https://polygons.openstreetmap.fr/get_geojson.py?id=51477&params=0"
MAX_RENDER_DIMENSION = 8192

viewer.__doc__ = __doc__
viewer.OUT_DIR = ROOT / "examples" / "out" / "germany_terrain_landcover"
viewer.CACHE_DIR = ROOT / "examples" / ".cache" / "germany_terrain_landcover"
viewer.COUNTRY_A3 = "DEU"
viewer.COUNTRY_NAME = "Germany"
viewer.COUNTRY_TITLE = "Land Cover in 2024: Germany"
viewer.TARGET_CRS = "EPSG:3035"
viewer.DEM_ZOOM = 10
viewer.SNAPSHOT_SIZE = (4096, 4096)
viewer.LANDCOVER_URLS = {
    tile: LANDCOVER_BASE_URL.format(tile=tile) for tile in GERMANY_LANDCOVER_TILES
}
viewer.CAPTION_LINES = [
    "(c)2026 Milos Popovic (https://milospopovic.net)",
    "Data: Sentinel-2 10m Land Use/Land Cover - Esri, Impact Observatory, Microsoft, and AWS Terrarium DEM",
]
viewer.COMP = {
    **viewer.COMP,
    "scale": 1.28,
    "shift_x": 0.0,
    "shift_y": -0.040,
}
viewer.CAMERA = {
    **viewer.CAMERA,
    "exaggeration": 1.50,
}


def _download_osm_boundary(cache_dir: Path, *, force: bool) -> Path:
    output = cache_dir / "germany_osm_boundary_relation_51477.geojson"
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(OSM_GERMANY_BOUNDARY, headers={"User-Agent": "forge3d-germany-example/1.0"})
    with urlopen(request, timeout=90) as response:
        with tempfile.NamedTemporaryFile(
            prefix=output.stem,
            suffix=output.suffix,
            dir=str(output.parent),
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp.replace(output)
    return output


def _country_geometry(boundary_path: Path, target_crs: str):
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise RuntimeError("Could not read Germany boundary from OpenStreetMap GeoJSON")
    boundary = boundary.set_crs("EPSG:4326", allow_override=True).to_crs(target_crs)
    return boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.geometry.unary_union


def _render_cache_key(dem_path: Path) -> str:
    return f"{viewer.COUNTRY_A3.lower()}_render_max{MAX_RENDER_DIMENSION}_{dem_path.stem}"


def _prepare_render_dem(dem_path: Path, cache_dir: Path) -> Path:
    with rasterio.open(dem_path) as src:
        max_dim = max(src.width, src.height)
        if max_dim <= MAX_RENDER_DIMENSION:
            return dem_path
        scale = MAX_RENDER_DIMENSION / float(max_dim)
        width = max(1, round(src.width * scale))
        height = max(1, round(src.height * scale))
        output = cache_dir / f"{_render_cache_key(dem_path)}.tif"
        if viewer._is_fresh(output, [dem_path]):
            return output
        data = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
            masked=True,
        )
        profile = src.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=src.transform * src.transform.scale(src.width / width, src.height / height),
            count=1,
            dtype="float32",
            nodata=src.nodata,
            compress="lzw",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data.filled(src.nodata).astype("float32"), 1)
        return output


def _prepare_render_overlay(overlay_path: Path, render_dem_path: Path, cache_dir: Path) -> Path:
    with rasterio.open(render_dem_path) as dem:
        size = (dem.width, dem.height)
    with Image.open(overlay_path) as src:
        if src.size == size:
            return overlay_path
        output = cache_dir / f"{_render_cache_key(render_dem_path)}_overlay.png"
        if viewer._is_fresh(output, [overlay_path, render_dem_path]):
            return output
        src.convert("RGBA").resize(size, resample=Image.Resampling.NEAREST).save(output)
        return output


def main() -> int:
    args = viewer._parse_args()
    viewer._country_geometry = _country_geometry
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    snapshot = (
        args.snapshot.resolve()
        if args.snapshot is not None
        else output_dir / "germany_landcover_2024.png"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = _download_osm_boundary(cache_dir, force=bool(args.force))
    work_cache_dir = cache_dir / "osm_boundary"
    work_cache_dir.mkdir(parents=True, exist_ok=True)
    dem_path = viewer._build_dem(
        boundary_path,
        work_cache_dir,
        int(args.dem_zoom),
        force=bool(args.force),
    )
    tile_paths = viewer._download_landcover_tiles(work_cache_dir, force=bool(args.force))
    classes_path = viewer._build_landcover_classes(
        boundary_path,
        tile_paths,
        dem_path,
        work_cache_dir,
        force=bool(args.force),
    )
    overlay_path, present = viewer._build_overlay(
        classes_path,
        dem_path,
        viewer._overlay_cache_path(work_cache_dir),
        force=bool(args.force),
    )
    render_dem_path = _prepare_render_dem(dem_path, work_cache_dir)
    render_overlay_path = _prepare_render_overlay(overlay_path, render_dem_path, work_cache_dir)
    viewer._render(snapshot, render_dem_path, render_overlay_path, present)
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
