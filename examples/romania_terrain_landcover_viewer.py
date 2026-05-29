#!/usr/bin/env python3
"""Romania land-cover terrain render clipped to the OSM country boundary.

The rendering pipeline is shared with ``bosnia_terrain_landcover_viewer.py``.
This wrapper only changes the country configuration and boundary source.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

import bosnia_terrain_landcover_viewer as base

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "out" / "romania_terrain_landcover"
CACHE_DIR = ROOT / "examples" / ".cache" / "romania_terrain_landcover"

COUNTRY_A3 = "ROU"
COUNTRY_NAME = "Romania"
COUNTRY_TITLE = "Land Cover in 2024: Romania"
OSM_RELATION_ID = 90689
OSM_BOUNDARY_URL = (
    "https://nominatim.openstreetmap.org/lookup"
    f"?osm_ids=R{OSM_RELATION_ID}"
    "&format=geojson"
    "&polygon_geojson=1"
    "&polygon_threshold=0.0"
)
USER_AGENT = "forge3d-romania-terrain-landcover/1.0"
DEFAULT_DEM_ZOOM = 10
DEFAULT_RENDER_MAX_SIZE = 4096
DEFAULT_SNAPSHOT_SIZE = (2700, 2700)
DEFAULT_SUPERSAMPLE = 3
DEFAULT_VIEWER_TIMEOUT = 180.0
DEFAULT_OPEN_WATER_TRIM_PX = 8
DEFAULT_LANDCOVER_TILES = ("34T", "34U", "35T", "35U")
LANDCOVER_TILE_URL = (
    "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/"
    "lc2024/{tile}_20240101-20241231.tif"
)
CAPTION_LINES = [
    "©2026 Milos Popovic (https://milospopovic.net)",
    (
        "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, "
        "Microsoft; AWS Terrarium DEM; OSM Romania boundary"
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--dem-zoom", type=int, default=DEFAULT_DEM_ZOOM)
    parser.add_argument(
        "--render-max-size",
        type=int,
        default=DEFAULT_RENDER_MAX_SIZE,
        help="Maximum DEM/overlay edge sent to the viewer. Use 0 for native size.",
    )
    parser.add_argument(
        "--snapshot-size",
        type=int,
        nargs=2,
        default=DEFAULT_SNAPSHOT_SIZE,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=DEFAULT_SUPERSAMPLE,
        help="Render scale before final downsampling.",
    )
    parser.add_argument(
        "--viewer-timeout",
        type=float,
        default=DEFAULT_VIEWER_TIMEOUT,
        help="Timeout in seconds for viewer startup and high-resolution snapshots.",
    )
    parser.add_argument(
        "--open-water-trim-px",
        type=int,
        default=DEFAULT_OPEN_WATER_TRIM_PX,
        help=(
            "Trim large water bodies connected to the raster border. This removes "
            "OSM maritime/open-sea water without using Natural Earth data. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--landcover-tiles",
        default=",".join(DEFAULT_LANDCOVER_TILES),
        help=(
            "Comma-separated Esri LULC MGRS tiles. Romania defaults to "
            "34T,34U,35T,35U so the small area north of 48N is included."
        ),
    )
    parser.add_argument(
        "--boundary-geojson",
        type=Path,
        default=None,
        help="Use a local OSM boundary GeoJSON instead of downloading relation 90689.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_tile_ids(value: str) -> tuple[str, ...]:
    tile_ids = tuple(
        token.strip().upper() for token in value.split(",") if token.strip()
    )
    if not tile_ids:
        raise ValueError("At least one land-cover tile id is required")
    for tile_id in tile_ids:
        if len(tile_id) != 3 or not tile_id[:2].isdigit() or not tile_id[2].isalpha():
            raise ValueError(f"Invalid land-cover tile id: {tile_id!r}")
    return tile_ids


def _landcover_urls(tile_ids: tuple[str, ...]) -> dict[str, str]:
    return {tile_id: LANDCOVER_TILE_URL.format(tile=tile_id) for tile_id in tile_ids}


def _resampled_transform(src, width: int, height: int):
    return src.transform * src.transform.scale(
        src.width / float(width),
        src.height / float(height),
    )


def _render_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    if max_size <= 0 or max(width, height) <= max_size:
        return width, height
    scale = max(width, height) / float(max_size)
    return max(1, round(width / scale)), max(1, round(height / scale))


def _build_render_dem(dem_path: Path, cache_dir: Path, max_size: int, *, force: bool) -> Path:
    with rasterio.open(dem_path) as src:
        width, height = _render_size(src.width, src.height, int(max_size))
        if (width, height) == (src.width, src.height):
            return dem_path
        output = cache_dir / f"{dem_path.stem}_render_{width}x{height}.tif"
        if base._is_fresh(output, [dem_path]) and not force:
            return output

        profile = src.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=_resampled_transform(src, width, height),
            dtype="float32",
            nodata=src.nodata if src.nodata is not None else -9999.0,
            compress="lzw",
        )
        data = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=profile["nodata"],
            init_dest_nodata=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data, 1)
    return output


def _build_render_landcover_classes(
    classes_path: Path,
    render_dem_path: Path,
    cache_dir: Path,
    max_size: int,
    *,
    force: bool,
) -> Path:
    output = cache_dir / f"{classes_path.stem}_render_{int(max_size)}.tif"
    with rasterio.open(classes_path) as src, rasterio.open(render_dem_path) as dem:
        same_grid = (
            src.width == dem.width
            and src.height == dem.height
            and src.crs == dem.crs
            and src.transform.almost_equals(dem.transform)
        )
        if same_grid:
            return classes_path
        if base._is_fresh(output, [classes_path, render_dem_path]) and not force:
            return output

        data = np.zeros((dem.height, dem.width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dem.transform,
            dst_crs=dem.crs,
            resampling=Resampling.mode,
            src_nodata=0,
            dst_nodata=0,
            init_dest_nodata=True,
        )
        profile = dem.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, base.SOURCE_COLORMAP_FULL)
    return output


def _render_overlay_path(cache_dir: Path, render_dem_path: Path) -> Path:
    return cache_dir / (
        f"{COUNTRY_A3.lower()}_landcover_overlay_"
        f"{base.LANDCOVER_OVERLAY_CACHE_KEY}_{render_dem_path.stem}.png"
    )


def _trim_border_open_water(
    dem_path: Path,
    classes_path: Path,
    cache_dir: Path,
    trim_px: int,
    *,
    force: bool,
) -> tuple[Path, Path]:
    trim_px = int(trim_px)
    if trim_px <= 0:
        return dem_path, classes_path

    output_dem = cache_dir / f"{dem_path.stem}_osm_land_px{trim_px}.tif"
    output_classes = cache_dir / f"{classes_path.stem}_osm_land_px{trim_px}.tif"
    if (
        base._is_fresh(output_dem, [dem_path, classes_path])
        and base._is_fresh(output_classes, [dem_path, classes_path])
        and not force
    ):
        return output_dem, output_classes

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "Open-water trimming requires scipy. Run through the project environment, "
            "for example: uv run --no-sync python examples/romania_terrain_landcover_viewer.py"
        ) from exc

    with rasterio.open(classes_path) as classes_src:
        classes = classes_src.read(1)
        classes_profile = classes_src.profile.copy()

    water = classes == 1
    outside_osm_mask = classes == 0
    border_seed = water & ndimage.binary_dilation(
        outside_osm_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    border_water = ndimage.binary_propagation(
        border_seed,
        structure=np.ones((3, 3), dtype=bool),
        mask=water,
    )
    open_water_core = border_water & (ndimage.distance_transform_edt(water) >= float(trim_px))
    remove_water = ndimage.binary_dilation(
        open_water_core,
        structure=np.ones((3, 3), dtype=bool),
        iterations=trim_px,
        mask=border_water,
    )
    if not np.any(remove_water):
        return dem_path, classes_path

    classes = classes.copy()
    classes[remove_water] = 0
    classes_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
    output_classes.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_classes, "w", **classes_profile) as dst:
        dst.write(classes, 1)
        dst.write_colormap(1, base.SOURCE_COLORMAP_FULL)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_profile = dem_src.profile.copy()
        nodata = dem_src.nodata if dem_src.nodata is not None else -9999.0
    dem = dem.copy()
    dem[remove_water] = nodata
    dem_profile.update(nodata=nodata, compress="lzw")
    with rasterio.open(output_dem, "w", **dem_profile) as dst:
        dst.write(dem, 1)

    return output_dem, output_classes


def _raise_if_mostly_black(image_path: Path) -> None:
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    black_fraction = float((rgb.max(axis=2) < 4).mean())
    if black_fraction > 0.75:
        raise RuntimeError(
            "Viewer render is mostly black. Retry with a smaller "
            "--render-max-size or --snapshot-size."
        )


def _downsample_snapshot(image_path: Path, final_size: tuple[int, int], supersample: int) -> None:
    if supersample <= 1:
        return

    from PIL import Image

    image = Image.open(image_path).convert("RGBA")
    if image.size == final_size:
        return
    image.resize(final_size, Image.Resampling.LANCZOS).save(image_path)


def _render(
    snapshot_path: Path,
    dem_path: Path,
    overlay_path: Path,
    present_classes: list[int],
    *,
    timeout: float,
) -> None:
    hdr = base.HDR.resolve()
    if not hdr.is_file():
        raise FileNotFoundError(f"HDRI not found: {hdr}")
    with rasterio.open(dem_path) as dem_src:
        terrain_width = float(max(dem_src.width, dem_src.height))
    terrain_xy = terrain_width / base.CAMERA["ref"]
    radius = base.CAMERA["radius"] * terrain_xy * base.CAMERA["pullback"]
    zscale = base.CAMERA["zscale"] * math.sqrt(max(terrain_xy, 1e-6)) * base.CAMERA["exaggeration"]
    terrain_cmd = {**base.TERRAIN, "radius": radius, "zscale": zscale}
    relief_terrain = {**terrain_cmd, **base.RELIEF_TERRAIN}
    color_pbr = {
        **base.PBR,
        "hdr_path": str(hdr),
        "height_ao": dict(base.PBR["height_ao"]),
        "sun_visibility": dict(base.PBR["sun_visibility"]),
        "tonemap": dict(base.PBR["tonemap"]),
    }
    relief_pbr = {
        **color_pbr,
        **base.RELIEF_PBR,
        "height_ao": dict(base.RELIEF_PBR["height_ao"]),
        "sun_visibility": dict(base.RELIEF_PBR["sun_visibility"]),
        "tonemap": dict(color_pbr["tonemap"]),
    }

    snapshot_path = snapshot_path.resolve()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge3d_rou_render_") as tmp:
        color_raw = Path(tmp) / "color_raw.png"
        relief_raw = Path(tmp) / "relief_raw.png"
        with base.f3d.open_viewer_async(
            terrain_path=dem_path,
            width=base.VIEWER_SIZE[0],
            height=base.VIEWER_SIZE[1],
            timeout=max(float(timeout), 45.0),
        ) as viewer:
            viewer.send_ipc({"cmd": "set_terrain", **terrain_cmd})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **color_pbr})
            viewer.load_overlay(
                "romania_landcover",
                overlay_path,
                extent=(0.0, 0.0, 1.0, 1.0),
                opacity=1.0,
                preserve_colors=True,
            )
            viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
            viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
            viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
            time.sleep(base.PASS_SETTLE_SECONDS)
            viewer.snapshot(color_raw, width=base.SNAPSHOT_SIZE[0], height=base.SNAPSHOT_SIZE[1])
            viewer.send_ipc({"cmd": "set_terrain", **relief_terrain})
            viewer.send_ipc({"cmd": "set_terrain_pbr", **relief_pbr})
            viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": False})
            time.sleep(base.PASS_SETTLE_SECONDS)
            viewer.snapshot(relief_raw, width=base.SNAPSHOT_SIZE[0], height=base.SNAPSHOT_SIZE[1])
        raw = base._combine_render_passes(color_raw, relief_raw)
    base._compose_snapshot(
        raw,
        snapshot_path,
        present_classes,
        sun_azimuth=float(terrain_cmd["sun_azimuth"]),
        sun_elevation=float(terrain_cmd["sun_elevation"]),
    )


def _download_osm_boundary(cache_dir: Path, *, force: bool) -> Path:
    dest = cache_dir / "osm" / f"romania_boundary_osm_r{OSM_RELATION_ID}.geojson"
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
        raise RuntimeError("OSM boundary lookup did not return a GeoJSON feature")

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


def _country_geometry(boundary_geojson: Path, target_crs: str):
    boundary = gpd.read_file(boundary_geojson)
    boundary = boundary[boundary.geometry.notna()]
    if boundary.empty:
        raise RuntimeError(f"Could not read an OSM boundary for {COUNTRY_NAME}")
    boundary = boundary.to_crs(target_crs)
    geometry = (
        boundary.geometry.union_all()
        if hasattr(boundary.geometry, "union_all")
        else boundary.geometry.unary_union
    )
    geometry = _valid_geometry(geometry)
    if geometry.is_empty:
        raise RuntimeError(f"OSM boundary for {COUNTRY_NAME} is empty")
    return geometry


def _configure_base(tile_ids: tuple[str, ...]) -> None:
    base.OUT_DIR = OUT_DIR
    base.CACHE_DIR = CACHE_DIR
    base.COUNTRY_A3 = COUNTRY_A3
    base.COUNTRY_NAME = COUNTRY_NAME
    base.COUNTRY_TITLE = COUNTRY_TITLE
    base.DEM_ZOOM = DEFAULT_DEM_ZOOM
    base.SNAPSHOT_SIZE = DEFAULT_SNAPSHOT_SIZE
    base.LANDCOVER_URLS = _landcover_urls(tile_ids)
    base.CAPTION_LINES = CAPTION_LINES
    base._country_geometry = _country_geometry

    base.PBR = {
        **base.PBR,
        "shadow_map_res": 2048,
        "height_ao": {
            **base.PBR["height_ao"],
            "directions": 6,
            "steps": 14,
            "strength": 0.22,
            "resolution_scale": 0.45,
        },
        "sun_visibility": {
            **base.PBR["sun_visibility"],
            "steps": 24,
            "max_distance": 1800.0,
            "resolution_scale": 0.55,
        },
    }
    base.CAMERA = {
        **base.CAMERA,
        "radius": 5200.0,
        "zscale": 0.032,
        "pullback": 1.08,
        "exaggeration": 1.00,
    }
    base.TERRAIN = {
        **base.TERRAIN,
        "theta": 18.0,
        "fov": 22.0,
        "sun_azimuth": 308.0,
        "sun_elevation": 27.0,
    }
    base.COMP = {
        **base.COMP,
        "scale": 1.30,
        "shift_x": 0.030,
        "shift_y": -0.018,
    }


def main() -> int:
    args = _parse_args()
    tile_ids = _parse_tile_ids(args.landcover_tiles)
    _configure_base(tile_ids)
    final_size = (int(args.snapshot_size[0]), int(args.snapshot_size[1]))
    supersample = max(1, int(args.supersample))
    base.SNAPSHOT_SIZE = (final_size[0] * supersample, final_size[1] * supersample)

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    snapshot = (
        args.snapshot.resolve()
        if args.snapshot is not None
        else output_dir / "romania_landcover_2024_osm_boundary.png"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    boundary = (
        args.boundary_geojson.resolve()
        if args.boundary_geojson is not None
        else _download_osm_boundary(cache_dir, force=bool(args.force))
    )
    dem_path = base._build_dem(boundary, cache_dir, int(args.dem_zoom), force=bool(args.force))
    tile_paths = base._download_landcover_tiles(cache_dir, force=bool(args.force))
    classes_path = base._build_landcover_classes(
        boundary, tile_paths, dem_path, cache_dir, force=bool(args.force)
    )
    render_dem_path = _build_render_dem(
        dem_path, cache_dir, int(args.render_max_size), force=bool(args.force)
    )
    render_classes_path = _build_render_landcover_classes(
        classes_path,
        render_dem_path,
        cache_dir,
        int(args.render_max_size),
        force=bool(args.force),
    )
    render_dem_path, render_classes_path = _trim_border_open_water(
        render_dem_path,
        render_classes_path,
        cache_dir,
        int(args.open_water_trim_px),
        force=bool(args.force),
    )
    render_overlay_path, present = base._build_overlay(
        render_classes_path,
        render_dem_path,
        _render_overlay_path(cache_dir, render_dem_path),
        force=bool(args.force),
    )
    print(f"Render DEM: {render_dem_path}")
    print(f"Render overlay: {render_overlay_path}")
    _render(
        snapshot,
        render_dem_path,
        render_overlay_path,
        present,
        timeout=float(args.viewer_timeout),
    )
    _downsample_snapshot(snapshot, final_size, supersample)
    _raise_if_mostly_black(snapshot)
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
