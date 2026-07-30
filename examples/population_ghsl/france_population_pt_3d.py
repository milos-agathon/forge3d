#!/usr/bin/env python3
"""France population lit by the PROMETHEUS terrain path tracer — with towers.

Same register as ``egypt_population_pt_3d`` (steel-blue relief, orange GHS_POP
population, serif plate) applied to metropolitan France + Corsica, plus one
signature effect: population is fused into the traced heightfield as real 3D
towers, so the Paris/Lyon/Marseille cores cast genuine long soft shadows
across the relief inside the converged path trace. Differences from Egypt,
all forced by the subject:

* **Projection.** EPSG:2154 (RGF93 / Lambert-93) — France's official national
  conformal grid (true parallels 44/49). No custom LCC needed.
* **Extent.** Natural Earth's FRA multipolygon includes French Guiana and the
  DOM-TOM; the ``RegionPreset.bbox`` clip keeps metropolitan France + Corsica
  only (the Iberia offshore-territory lesson).
* **Towers.** ``height = dem01 + tower01`` where ``tower01`` is population
  mapped through a display floor + gamma and expressed in WORLD units
  (``TOWER_WORLD / relief``), so changing relief never rescales towers and
  towers never rescale the terrain normalization (the LoD2 lesson).
* **Budget.** France's near-square grid is bigger than Egypt's: at 4096² the
  19 B/texel + 366 B/frame-px law caps the per-cell render at 722 px, so the
  default frame is 596 (Egypt's 672 would be refused at the gate).
* **Storage.** Every generated file lives under ``D:/france_population_pt/``
  (user directive; C: is full).

The DEM chain, overlay texture builder and plate composer are reused from
``romania_builtup_cover_3d``; the mosaic tracer follows ``egypt_population_pt_3d``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from scipy import ndimage

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import romania_builtup_cover_3d as rom  # noqa: E402  (sibling example module)

from forge3d.path_tracing import hybrid_render_terrain_reference  # noqa: E402

# --- storage (user directive: all generated files on D:) ----------------------
DATA_ROOT = Path("D:/france_population_pt")
DEFAULT_CACHE_DIR = DATA_ROOT / "cache"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "out"

# --- region ------------------------------------------------------------------
# EPSG:2154 = RGF93 / Lambert-93, the official French national grid
# (conformal, standard parallels 44/49, origin 46.5N 3E).
FRANCE_CRS = "EPSG:2154"
# bbox clip (lon/lat) keeps metropolitan France + Corsica and drops French
# Guiana, the Antilles, Réunion, Mayotte and the Pacific territories that
# Natural Earth folds into the same FRA multipolygon.
FRANCE_REGION = rom.RegionPreset(
    slug="france_population",
    cache_a3="FRA",
    title="FRANCE",
    name="France",
    country_a3=("FRA",),
    admin=("France",),
    bbox=(-5.9, 41.2, 9.9, 51.4),
    target_crs=FRANCE_CRS,
    dem_zoom=10,
)
POPULATION_SOURCE = Path("D:/ghsl-population/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif")
TITLE_LINES = ["Population", "FRANCE"]
CAPTION_LINES = [
    "©2026 Milos Popovic (milosgis.com)",
    "Data: Global Human Settlement Layer - population (R2023A, epoch 2020) at 3 arcsec",
]

# --- population marker (orange overlay) --------------------------------------
# Same rule family as Egypt: a cell is MARKED populated at >= 1 resident per
# 3 arcsec source cell — above GHS_POP's disaggregation dust, below a hamlet.
POPULATION_MIN_PERSONS = 1.0

# --- towers ------------------------------------------------------------------
# Towers are the signature effect: population ADDED to the heightfield in
# world units so cores cast real PT shadows. Three knobs, all swept in Task 4:
#   TOWER_MIN_PERSONS — cells below this get NO height (they stay flat orange
#     markers). Without it every dust cell grows a needle and France turns
#     into the WorldPop "hairy grey carpet" (MIN_DISPLAY~25 lesson).
#   TOWER_GAMMA — exponent on normalized population. >1 suppresses villages'
#     height (they keep their orange marker) so Paris reads as a spire
#     cluster, not the whole country as fuzz.
#   TOWER_WORLD — world-unit height of the p99.9 population cell. SPAN_X=100,
#     so 2.5 world units ≈ 2.5% of the map width. Terrain world height is
#     RELIEF_WORLD (dem01 max = 1.0), so towers stay a fraction of Mont Blanc.
TOWER_MIN_PERSONS = 25.0
TOWER_GAMMA = 1.6
TOWER_WORLD = 2.5
TOWER_REF_PCT = 99.9  # normalization anchor percentile of tower-active cells

# --- DEM ---------------------------------------------------------------------
# France's lowest genuine land is around -10 m (Camargue, Nord polders);
# anything below -15 m inside the generalized coastline is leaked bathymetry
# and is clamped to sea level.
SEA_FLOOR_M = -15.0

# --- PT scene ----------------------------------------------------------------
SPAN_X = 100.0  # world units across the DEM grid's x extent
# Settled by --measure-relief probes in Task 4 (map-maker p90 band, Alps kept
# sculpted without crushing the Paris basin). 7.0 is the starting default.
RELIEF_WORLD = 7.0
CAMERA_FOV_Y = 8.0  # quasi-orthographic nadir (relief parallax < 0.5%)
CAMERA_MARGIN = 1.06
# Measured PT convention (quadrant probe, 2026-07-12): light compass =
# sun_azimuth + 90, so 225 = NW light with shadows falling SE. MUST be
# re-verified on THIS terrain in Task 4 before the production trace.
SUN_AZIMUTH = 225.0
SUN_ELEVATION = 16.0
SUN_INTENSITY = 3.2
ENV_INTENSITY = 0.62
PT_ALBEDO = (0.62, 0.62, 0.62)

# --- light field -> overlay modulation (Egypt values, re-judged in Task 5) ---
SHADE_LOW_PCT = 2.0
SHADE_HIGH_PCT = 99.5
SHADE_GAMMA = 1.0
TERRAIN_FLOOR = 0.28
TERRAIN_GAIN = 0.72
MARKER_FLOOR = 0.58
MARKER_GAIN = 0.42
MODERN_SHADE_HIGH_PCT = 96.0
MODERN_SHADE_GAMMA = 0.90
MODERN_TERRAIN_FLOOR = 0.34
MODERN_TERRAIN_GAIN = 0.66
SHADE_UNSHARP_SIGMA = 3.0
SHADE_UNSHARP_AMOUNT = 0.30
FINAL_CHROMA = 1.15
FINAL_BRIGHTNESS = 1.06
CAVITY_SIGMA = 2.0
CAVITY_DARKEN = 0.08
CAVITY_LIGHTEN = 0.03

# --- memory gate -------------------------------------------------------------
# Measured law (2026-07-17): tracked total ~ 19 B/grid-texel + 366 B/frame-px
# against the enforced 512 MiB budget. Solve BEFORE launching. At 4096² the
# frame+2*margin ceiling is 722 px -> default frame 596 with margin 48.
BYTES_PER_GRID_TEXEL = 19.0
BYTES_PER_FRAME_PIXEL = 366.0
BUDGET_BYTES = 512 * 2**20
BUDGET_HEADROOM = 0.95

DEVICE_LOST_EXIT = 75  # relaunch the PROCESS; the per-cell npz cache resumes

# --- plate -------------------------------------------------------------------
SNAPSHOT_SIZE = (6144, 6144)  # Iberia-proven 8K-class compose canvas


# --- pure helpers (covered by --self-test) -----------------------------------


def _max_downsample(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """MAX-reduce ``arr`` to ``shape`` (any integer ratio, non-integer OK).

    The honest reducer for threshold/tower rasters: a target cell is as
    populated as its MOST populated source cell. BILINEAR (what
    ``_load_dem_grid`` uses for terrain) would smear Paris and delete hamlets.
    """
    if arr.shape == tuple(shape):
        return arr
    ys = np.arange(arr.shape[0]) * shape[0] // arr.shape[0]
    xs = np.arange(arr.shape[1]) * shape[1] // arr.shape[1]
    out = np.zeros(shape, dtype=arr.dtype)
    np.maximum.at(out, (ys[:, None], xs[None, :]), arr)
    return out


def _tower_heights(
    pop: np.ndarray,
    *,
    floor_persons: float = TOWER_MIN_PERSONS,
    gamma: float = TOWER_GAMMA,
    tower_world: float = TOWER_WORLD,
) -> np.ndarray:
    """Residents/cell -> additive tower height in WORLD units.

    Cells below ``floor_persons`` get exactly 0 (they remain flat orange
    markers). Active cells are normalized by the p99.9 active cell (robust to
    the Paris outlier), clipped to 1, raised to ``gamma`` and scaled to
    ``tower_world``. Monotone in population by construction.
    """
    heights = np.zeros(pop.shape, dtype=np.float32)
    active = np.isfinite(pop) & (pop >= float(floor_persons))
    if not active.any():
        return heights
    ref = float(np.percentile(pop[active], TOWER_REF_PCT))
    norm = np.clip(pop[active] / max(ref, 1e-6), 0.0, 1.0)
    heights[active] = np.power(norm, float(gamma), dtype=np.float32) * float(tower_world)
    return heights


def _budget_estimate(dem_grid: np.ndarray, render: int) -> float:
    texels = float(dem_grid.shape[0]) * float(dem_grid.shape[1])
    return BYTES_PER_GRID_TEXEL * texels + BYTES_PER_FRAME_PIXEL * float(render) ** 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--dem-zoom", type=int, default=None)
    parser.add_argument("--grid-max", type=int, default=4096, help="Max DEM edge fed to the tracer")
    parser.add_argument(
        "--frame", type=int, default=596,
        help="PT cell edge in pixels (596: the 4096² budget ceiling is render<=722)",
    )
    parser.add_argument("--cell-margin", type=int, default=48, help="Feather overlap per cell edge")
    parser.add_argument("--tiles", type=int, default=8, help="Camera cells per axis")
    parser.add_argument("--spp", type=int, default=1, help="spp>=4 on big grids trips Windows TDR")
    parser.add_argument("--max-frames", type=int, default=3072)
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--variance-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--relief", type=float, default=None, help="Override RELIEF_WORLD")
    parser.add_argument("--sun-elevation", type=float, default=None)
    parser.add_argument("--sun-azimuth", type=float, default=None)
    parser.add_argument("--tower-world", type=float, default=None, help="Override TOWER_WORLD")
    parser.add_argument("--tower-gamma", type=float, default=None, help="Override TOWER_GAMMA")
    parser.add_argument("--tower-min", type=float, default=None, help="Override TOWER_MIN_PERSONS")
    parser.add_argument("--no-towers", action="store_true", help="Trace terrain only (A/B probe)")
    parser.add_argument("--tag", type=str, default="", help="Suffix for cache/output names")
    parser.add_argument("--population-min", type=float, default=POPULATION_MIN_PERSONS)
    parser.add_argument("--modern-grade", action="store_true")
    parser.add_argument("--cavity", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prep-only", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--measure-relief", action="store_true",
                        help="Land-only p90 world-slope table on the TERRAIN-ONLY grid, then stop")
    parser.add_argument("--reuse-shade", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="Fast low-res pass (grid 768, one cell at 640) for look checks")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _neutral_sky_env(height: int = 32, width: int = 64) -> np.ndarray:
    """Small lat-long environment: pale blue zenith -> warm-white horizon -> grey ground."""
    zenith = np.array([0.62, 0.72, 0.92], dtype=np.float32) * 1.1
    horizon = np.array([1.00, 0.99, 0.96], dtype=np.float32) * 0.85
    ground = np.array([0.28, 0.28, 0.29], dtype=np.float32)
    rows = np.linspace(1.0, -1.0, height, dtype=np.float32)
    env_rows = np.empty((height, 3), dtype=np.float32)
    up = np.clip(rows, 0.0, 1.0)[:, None] ** 0.65
    env_rows[:] = horizon[None, :] * (1.0 - up) + zenith[None, :] * up
    below = rows < 0.0
    down = np.clip(-rows[below], 0.0, 1.0)[:, None] ** 0.5
    env_rows[below] = horizon[None, :] * (1.0 - down) + ground[None, :] * down
    return np.repeat(env_rows[:, None, :], width, axis=1)


def _clamp_sea_leak(render_dem_path: Path, cache_dir: Path, *, force: bool) -> Path:
    """Replace leaked bathymetry with sea level; keep genuine sub-sea land.

    France's lowest land is ~-10 m (Camargue, Nord polders). The DEM
    normalization runs on min/max, so a fringe of Atlantic/Mediterranean
    bathymetry through the generalized coastline would silently flatten every
    slope on the plate.
    """
    output = cache_dir / f"{render_dem_path.stem}_sealeak_clamped.tif"
    if rom._is_fresh(output, [render_dem_path]) and not force:
        return output
    with rasterio.open(render_dem_path) as src:
        profile = src.profile.copy()
        dem = src.read(1, masked=True)
        nodata = src.nodata if src.nodata is not None else -9999.0
    data = dem.filled(np.nan).astype(np.float32)
    valid = np.isfinite(data)
    leak = valid & (data < SEA_FLOOR_M)
    lowland = valid & (data < 0.0) & (data >= SEA_FLOOR_M)
    print(
        f"[dem] sea leak clamped: {int(leak.sum())} px below {SEA_FLOOR_M:.0f} m "
        f"(min {float(np.nanmin(data)):.1f} m), genuine sub-sea px kept: {int(lowland.sum())}"
    )
    data = np.where(leak, 0.0, data)
    assert float(np.nanmin(np.where(valid, data, np.nan))) >= SEA_FLOOR_M, "clamp failed"
    profile.update(dtype="float32", nodata=nodata, compress="lzw")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(np.where(valid, data, nodata).astype(np.float32), 1)
    return output


def _write_population_on_dem_grid(
    boundary_wgs84, dem_path: Path, output: Path, *, force: bool
) -> Path:
    """Reproject GHS_POP onto the DEM grid with MAX (the Egypt lesson).

    The 3 arcsec source (~90 m) is finer than the ~250 m Lambert-93 render
    cell; NEAREST keeps 1 sample in ~8 and silently deletes hamlets. MAX is
    the honest reducer for a threshold marker + tower map.

    Prints the INSEE sanity gate: the masked SUM should land near
    metropolitan France 2020 (~65 M). MAX-reduced cells can't be summed for
    that check, so the gate sums the SOURCE window under the mask.
    """
    if rom._is_fresh(output, [POPULATION_SOURCE, dem_path]) and not force:
        return output

    with rasterio.open(dem_path) as dem, rasterio.open(POPULATION_SOURCE) as src:
        dem_data = dem.read(1, masked=True)
        destination = np.zeros((dem.height, dem.width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else 0.0,
            dst_transform=dem.transform,
            dst_crs=dem.crs,
            dst_nodata=0.0,
            init_dest_nodata=True,
            resampling=Resampling.max,
        )
        # SUM pass for the population-total gate (same window, honest reducer
        # for totals). Only used for the printed sanity check.
        totals = np.zeros((dem.height, dem.width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=totals,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else 0.0,
            dst_transform=dem.transform,
            dst_crs=dem.crs,
            dst_nodata=0.0,
            init_dest_nodata=True,
            resampling=Resampling.sum,
        )
        boundary_dem = gpd.GeoSeries([boundary_wgs84], crs="EPSG:4326").to_crs(dem.crs).iloc[0]
        outside = geometry_mask(
            [boundary_dem],
            out_shape=(dem.height, dem.width),
            transform=dem.transform,
            invert=False,
        )
        dead = np.asarray(dem_data.mask) | outside
        destination[dead] = 0.0
        totals[dead] = 0.0
        profile = dem.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=0.0, compress="lzw")

    total_m = float(totals.sum()) / 1e6
    inhabited = destination >= POPULATION_MIN_PERSONS
    print(
        f"[pop] reprojected (max) -> {int(inhabited.sum())} cells >= "
        f"{POPULATION_MIN_PERSONS:g} residents, peak cell {float(destination.max()):.0f}, "
        f"masked total {total_m:.1f} M (INSEE metro 2020 ~65 M)"
    )
    if not (55.0 <= total_m <= 75.0):
        raise RuntimeError(
            f"masked population total {total_m:.1f} M is outside the 55-75 M sanity "
            "band — wrong mask, wrong window or wrong source"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(destination, 1)
    return output


def _prepare_inputs(args: argparse.Namespace, cache_dir: Path) -> tuple[Path, Path, Path]:
    """Return (render DEM, population raster, light-free overlay), building caches."""
    rom._configure_region(FRANCE_REGION)
    rom.CAPTION_LINES = list(CAPTION_LINES)
    rom.TITLE_LINES = list(TITLE_LINES)  # _configure_region says "Built-up areas"
    rom.SNAPSHOT_SIZE = SNAPSHOT_SIZE
    # PROMETHEUS owns ALL shadowing (design decision: pure PT, no viewer pass).
    rom.TERRAIN_CAST_SHADOW = {**rom.TERRAIN_CAST_SHADOW, "enabled": False}

    threshold = float(args.population_min)

    def _france_active_mask(population, valid, source_path):  # noqa: ANN001,ARG001
        return valid & np.isfinite(population) & (population >= threshold)

    rom._builtup_active_mask = _france_active_mask
    rom.OVERLAY_STYLE_VERSION = (
        "france-population-overlay-pt-lightfree"
        f"-src{POPULATION_SOURCE.stem}"
        f"-minpop{threshold:.2f}"
        f"-marker{int(rom.BUILTUP_COLOR[0])}-{int(rom.BUILTUP_COLOR[1])}-{int(rom.BUILTUP_COLOR[2])}"
        f"-terrainpal{';'.join('-'.join(str(int(c)) for c in color) for color in rom.TERRAIN_PALETTE)}"
        "-castshadow0-sealeakclamp-bboxmetro"
    )

    dem_zoom = int(args.dem_zoom if args.dem_zoom is not None else FRANCE_REGION.dem_zoom)
    print("== Preparing France inputs (cached) ==")
    boundary_zip = rom._download(
        rom.NATURAL_EARTH, cache_dir / "ne_10m_admin_0_countries.zip", force=args.force
    )
    boundary_wgs84 = rom._country_geometry(boundary_zip, "EPSG:4326")
    dem_path = rom._build_dem(boundary_zip, cache_dir, dem_zoom, force=args.force)
    render_dem_path = rom._prepare_render_dem(dem_path, cache_dir, force=args.force)
    render_dem_path = _clamp_sea_leak(render_dem_path, cache_dir, force=args.force)
    population_path = _write_population_on_dem_grid(
        boundary_wgs84,
        render_dem_path,
        cache_dir / f"fra_population_on_{render_dem_path.stem}_v1.tif",
        force=args.force,
    )
    overlay_path = rom._build_overlay(
        population_path,
        render_dem_path,
        cache_dir / f"france_population_overlay_pt_lightfree_p{threshold:g}_v1.png",
        force=args.force,
    )
    with rasterio.open(render_dem_path) as src:
        print(f"[dem] render grid {src.width}x{src.height} in {FRANCE_CRS}")
    return render_dem_path, population_path, overlay_path


def _self_test() -> int:
    """Pure-function gates; no GPU, no network, no D: writes."""
    # _max_downsample: block maxima survive, shape is exact.
    src = np.array(
        [[1, 2, 0, 0], [3, 4, 0, 9], [0, 0, 5, 0], [7, 0, 0, 6]], dtype=np.float32
    )
    out = _max_downsample(src, (2, 2))
    assert out.shape == (2, 2), out.shape
    assert np.array_equal(out, np.array([[4, 9], [7, 6]], dtype=np.float32)), out
    # identity when shapes match
    assert _max_downsample(src, (4, 4)) is src
    # _tower_heights: floor zeroes, monotone, max == tower_world at the anchor.
    pop = np.array([[0.0, 10.0], [100.0, 1000.0]], dtype=np.float32)
    h = _tower_heights(pop, floor_persons=25.0, gamma=1.6, tower_world=2.5)
    assert h[0, 0] == 0.0 and h[0, 1] == 0.0, h  # below floor -> flat
    assert 0.0 < h[1, 0] < h[1, 1], h  # monotone
    assert abs(float(h[1, 1]) - 2.5) < 1e-5, h  # anchor cell reaches TOWER_WORLD
    # all-below-floor input stays flat without div-by-zero
    assert not _tower_heights(np.full((3, 3), 1.0, np.float32)).any()
    # budget solver: the documented default frame fits the gate at 4096².
    grid = np.zeros((4096, 4096), dtype=np.float32)
    assert _budget_estimate(grid, 596 + 2 * 48) <= BUDGET_BYTES * BUDGET_HEADROOM
    assert _budget_estimate(grid, 672 + 2 * 48) > BUDGET_BYTES * BUDGET_HEADROOM
    print("self-test: OK")
    return 0


def main() -> int:
    global RELIEF_WORLD, SUN_ELEVATION, SUN_AZIMUTH, TOWER_WORLD, TOWER_GAMMA, TOWER_MIN_PERSONS
    args = _parse_args()
    if args.self_test:
        return _self_test()
    if args.relief is not None:
        RELIEF_WORLD = float(args.relief)
    if args.sun_elevation is not None:
        SUN_ELEVATION = float(args.sun_elevation)
    if args.sun_azimuth is not None:
        SUN_AZIMUTH = float(args.sun_azimuth)
    if args.tower_world is not None:
        TOWER_WORLD = float(args.tower_world)
    if args.tower_gamma is not None:
        TOWER_GAMMA = float(args.tower_gamma)
    if args.tower_min is not None:
        TOWER_MIN_PERSONS = float(args.tower_min)
    if args.no_towers:
        TOWER_WORLD = 0.0
    if args.test:
        args.grid_max = min(args.grid_max, 768)
        args.tiles = 1
        args.frame = min(args.frame, 640)
        args.cell_margin = 0
        args.max_frames = min(args.max_frames, 1024)
        args.variance_threshold = max(args.variance_threshold, 4e-3)

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    tag = f"_{args.tag}" if args.tag else ""
    snapshot = (
        args.snapshot.resolve()
        if args.snapshot is not None
        else output_dir / f"france_population_pt{tag}.png"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    render_dem_path, population_path, overlay_path = _prepare_inputs(args, cache_dir)
    if args.prep_only:
        print(f"Prepared: {render_dem_path}\nPrepared: {population_path}\nPrepared: {overlay_path}")
        return 0

    print("later tasks: towers, trace, modulate, compose")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
