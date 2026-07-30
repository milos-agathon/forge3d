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
# Same rule family as Egypt, but France is not an empty desert: at Egypt's cut
# (>= 1 resident/cell) the dispersed rural settlement marks ~60% of the land
# and drowns the steel-blue relief in orange. Swept 1/5/10/25 on the traced
# probe (2026-07-30): 25 kills the countryside story, 10 reduces towns to
# isolated dots, 5 keeps the urban network AND the settlement geography (the
# dense north-west vs the diagonale du vide) with the relief still dominant.
POPULATION_MIN_PERSONS = 5.0

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
# Swept 3x3 (world 1.5/2.5/4.0 x gamma 1.0/1.6/2.4) + floor 10/25/50 at
# RELIEF 5 (2026-07-30, judged on Paris/Lyon/Marseille crops + full field):
# 1.5/g1.0 speckles Paris into undifferentiated texture, 4.0/g2.4 collapses
# it to one monolithic shadow; 2.5/g1.6 gives the spire cluster + coherent SE
# streak with suburbs subordinate. Floor 10 -> 7.4% tower cells (needle
# carpet over northern France), floor 50 -> 0.5% (Brittany/Normandy bald);
# floor 25 -> 2.2%, cities present, no carpet.
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
# Settled by --measure-relief + eye probes (2026-07-30). Land-only p90
# world-slope at the production grid (4096x3801):
#     RELIEF   2.0   4.0   5.0   6.0   7.0   8.0
#     p90 deg 40.55 59.70 64.95 68.72 71.54 73.71
# The calibrated register anchor (Switzerland v6) is ~62 deg -> between 4 and
# 5. Probes at 4/5/6: 4 leaves the Paris basin featureless, 6 drowns the
# southern Alps in shadow mass; 5 keeps Massif Central/Vosges/Jura sculpted
# with the Alps legible.
RELIEF_WORLD = 5.0
CAMERA_FOV_Y = 8.0  # quasi-orthographic nadir (relief parallax < 0.5%)
CAMERA_MARGIN = 1.06
# Measured PT convention (quadrant probe, 2026-07-12): light compass =
# sun_azimuth + 90, so 225 = NW light with shadows falling SE. Re-verified on
# THIS terrain (2026-07-30, 10 tallest well-separated peaks at RELIEF 5):
# mean flank luminance NW 0.400 / NE 0.364 / SW 0.327 / SE 0.253 ->
# brightest NW, darkest SE. PASS.
SUN_AZIMUTH = 225.0
# Probed 12/16/20 at the winning tower combo: 12 darkens the whole terrain,
# 20 shortens the signature tower streaks; 16 keeps both.
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
    clamped_min = float(np.nanmin(np.where(valid, data, np.nan)))
    if clamped_min < SEA_FLOOR_M:  # not assert: must survive python -O
        raise RuntimeError(f"sea-leak clamp failed: min {clamped_min:.1f} m below {SEA_FLOOR_M:.0f} m")
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


def _load_dem_grid(render_dem_path: Path, grid_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the render DEM, fill nodata, downsample, normalize 0..1 (TERRAIN ONLY).

    Returns (dem_grid, land_grid). Towers are NEVER part of this normalization:
    the terrain's 0..1 range is anchored to real elevation so the tallest tower
    cannot rescale the relief (the LoD2 lesson).
    """
    with rasterio.open(render_dem_path) as src:
        dem = src.read(1, masked=True)
    data = dem.filled(np.nan).astype(np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        raise RuntimeError("Render DEM has no finite samples")
    fill = float(np.nanpercentile(data[finite], 1.0))
    data = np.where(finite, data, fill)

    land = finite.astype(np.float32)
    scale = min(1.0, grid_max / float(max(data.shape)))
    if scale < 1.0:
        new_size = (max(2, round(data.shape[1] * scale)), max(2, round(data.shape[0] * scale)))
        data = np.asarray(
            Image.fromarray(data, mode="F").resize(new_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        land = np.asarray(
            Image.fromarray(land, mode="F").resize(new_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    data = data - data.min()
    data = data / max(float(data.max()), 1e-6)
    return np.ascontiguousarray(data, dtype=np.float32), land > 0.5


def _load_tower_grid(population_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Tower heights (world units) on the PT grid.

    Population is MAX-downsampled to the PT grid FIRST (bilinear would smear
    Paris into a plateau and erase hamlets), then mapped through the tower
    transform. Returns zeros everywhere when TOWER_WORLD == 0 (--no-towers).
    """
    if TOWER_WORLD <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    with rasterio.open(population_path) as src:
        pop = src.read(1).astype(np.float32)
    pop = _max_downsample(pop, shape)
    towers = _tower_heights(
        pop,
        floor_persons=TOWER_MIN_PERSONS,
        gamma=TOWER_GAMMA,
        tower_world=TOWER_WORLD,
    )
    n = int((towers > 0).sum())
    print(
        f"[towers] {n} tower cells ({100.0 * n / towers.size:.2f}% of grid), "
        f"tallest {float(towers.max()):.2f} world units "
        f"(terrain range = {RELIEF_WORLD:g}), floor {TOWER_MIN_PERSONS:g} persons, "
        f"gamma {TOWER_GAMMA:g}"
    )
    return towers


def _traced_grid(dem_grid: np.ndarray, towers_world: np.ndarray) -> np.ndarray:
    """Fuse towers into the traced heightfield.

    The tracer multiplies the grid by ``exaggeration=RELIEF_WORLD`` to get
    world height, so world-unit towers are divided by RELIEF_WORLD here. This
    keeps TOWER_WORLD physically meaningful and relief-invariant.
    """
    if not towers_world.any():
        return dem_grid
    fused = dem_grid + towers_world / float(RELIEF_WORLD)
    return np.ascontiguousarray(fused, dtype=np.float32)


def _p90_land_slope(dem_grid: np.ndarray, land: np.ndarray, relief: float) -> float:
    """p90 of the world-unit slope the tracer sees, over country land only."""
    spacing = SPAN_X / (dem_grid.shape[1] - 1)
    gy, gx = np.gradient(dem_grid * relief, spacing)
    slope = np.hypot(gx, gy)[land]
    return float(np.degrees(np.arctan(np.percentile(slope, 90.0))))


# --- path tracing ------------------------------------------------------------


def _camera_for_grid(
    rows: int, cols: int, *, tiles: int = 1, tx: int = 0, ty: int = 0, expand: float = 1.0
) -> dict:
    """Nadir camera over one cell of an NxN mosaic.

    Every cell keeps the full-frame camera distance (identical relief parallax)
    and narrows the FOV; ``expand`` > 1 widens the framing so the render covers
    the cell plus a margin for feathered stitching.
    """
    span_z = SPAN_X * rows / cols
    half_extent = 0.5 * max(SPAN_X, span_z) * CAMERA_MARGIN
    distance = half_extent / math.tan(math.radians(CAMERA_FOV_Y / 2.0))
    tile_half = half_extent / tiles
    # Image up is -Z (north at the top): cell row 0 is the most-negative-Z strip.
    center_x = -half_extent + (tx + 0.5) * 2.0 * tile_half
    center_z = -half_extent + (ty + 0.5) * 2.0 * tile_half
    fov_y = math.degrees(
        2.0 * math.atan(math.tan(math.radians(CAMERA_FOV_Y / 2.0)) / tiles * expand)
    )
    return {
        "origin": (center_x, distance, center_z),
        "look_at": (center_x, 0.0, center_z),
        "up": (0.0, 0.0, -1.0),
        "fov_y": fov_y,
        "exposure": 1.0,
    }


def _pt_pass(
    dem_grid: np.ndarray, render: int, camera: dict, args: argparse.Namespace, *, label: str = "full"
) -> tuple[np.ndarray, np.ndarray]:
    spacing = SPAN_X / (dem_grid.shape[1] - 1)
    out = hybrid_render_terrain_reference(
        dem_grid,
        render,
        render,
        camera,
        spacing=(spacing, spacing),
        exaggeration=RELIEF_WORLD,
        albedo=PT_ALBEDO,
        sun_azimuth_deg=SUN_AZIMUTH,
        sun_elevation_deg=SUN_ELEVATION,
        sun_intensity=SUN_INTENSITY,
        env_map=_neutral_sky_env(),
        env_intensity=ENV_INTENSITY,
        spp=int(args.spp),
        max_frames=int(args.max_frames),
        min_frames=int(args.min_frames),
        variance_threshold=float(args.variance_threshold),
        seed=int(args.seed),
    )
    print(
        f"[PT:{label}] converged={out['converged']} frames={out['frames']} "
        f"variance={out['variance']:.3e} "
        f"peak_host_visible={out.get('peak_host_visible_bytes', 0) / 2**20:.1f} MiB",
        flush=True,
    )
    rgba = out["rgba"].astype(np.float32) / 255.0
    hit = np.isfinite(out["depth"])
    rgb = np.where(hit[:, :, None], rgba[:, :, :3], np.nan).astype(np.float32)
    return rgb, hit


def _cell_weights(size: int, margin: int) -> np.ndarray:
    """1D trapezoid: 1.0 in the cell interior, linear 0..1 across the margin."""
    if margin <= 0:
        return np.ones(size, dtype=np.float32)
    i = np.arange(size, dtype=np.float32) + 0.5
    return np.minimum(1.0, np.minimum(i / margin, (size - i) / margin)).astype(np.float32)


def _pt_light_field(
    dem_grid: np.ndarray, args: argparse.Namespace, cell_cache_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Light field as a mosaic of feather-blended overlapping camera cells.

    Butt-joined cells leave luminance steps at the stitch lines (each cell views
    boundary terrain from a slightly different angle). Rendering each cell with
    a margin and accumulating under a trapezoid weight drops the worst seam to
    ~1.5x the interior gradient — which matters far more on Egypt's smooth
    desert than it did on Romania's mountains.
    """
    rows, cols = dem_grid.shape
    tiles = max(1, int(args.tiles))
    frame = int(args.frame)
    margin = max(0, int(args.cell_margin)) if tiles > 1 else 0
    render = frame + 2 * margin
    estimate = _budget_estimate(dem_grid, render)
    print(
        f"[PT] grid={cols}x{rows} cells={tiles}x{tiles} frame={frame} margin={margin} "
        f"render={render} field={tiles * frame} "
        f"budget~{estimate / 2**20:.0f} MiB of {BUDGET_BYTES / 2**20:.0f} MiB"
    )
    if estimate > BUDGET_BYTES * BUDGET_HEADROOM:
        raise RuntimeError(
            f"PT pass would need ~{estimate / 2**20:.0f} MiB against the enforced "
            f"{BUDGET_BYTES / 2**20:.0f} MiB gate. Lower --frame or --grid-max "
            "(refusing rather than letting the run fail after the first cells)."
        )

    cell_cache_dir.mkdir(parents=True, exist_ok=True)
    if tiles == 1:
        rgb, hit = _pt_pass(dem_grid, render, _camera_for_grid(rows, cols), args)
        if not hit.any():
            raise RuntimeError("Path tracer produced no terrain hits — check camera framing")
        return rgb, hit

    expand = render / float(frame)
    field = tiles * frame
    rgb_acc = np.zeros((field, field, 3), dtype=np.float64)
    w_acc = np.zeros((field, field), dtype=np.float64)
    w1d = _cell_weights(render, margin)
    w2d = np.minimum(w1d[:, None], w1d[None, :]).astype(np.float32)
    for ty in range(tiles):
        for tx in range(tiles):
            # Per-cell cache: a GPU hiccup on a long sequential run becomes a
            # resume instead of a loss (the device has no in-process recreate
            # path, so recovery means relaunching this script).
            cache = cell_cache_dir / f"cell_{tx}_{ty}.npz"
            if cache.is_file():
                cached = np.load(cache)
                cell_rgb, cell_hit = cached["rgb"], cached["hit"]
                print(f"[PT:cell {tx},{ty}] cached: {cache.name}", flush=True)
            else:
                camera = _camera_for_grid(rows, cols, tiles=tiles, tx=tx, ty=ty, expand=expand)
                cell_rgb, cell_hit = _pt_pass(
                    dem_grid, render, camera, args, label=f"cell {tx},{ty}"
                )
                np.savez_compressed(cache, rgb=cell_rgb, hit=cell_hit)
            w_cell = np.where(cell_hit, w2d, 0.0).astype(np.float32)
            y0 = ty * frame - margin
            x0 = tx * frame - margin
            sy0, sx0 = max(0, -y0), max(0, -x0)
            dy0, dx0 = max(0, y0), max(0, x0)
            sy1 = render - max(0, y0 + render - field)
            sx1 = render - max(0, x0 + render - field)
            src_rgb = np.nan_to_num(cell_rgb[sy0:sy1, sx0:sx1], nan=0.0)
            src_w = w_cell[sy0:sy1, sx0:sx1]
            rgb_acc[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] += src_rgb * src_w[:, :, None]
            w_acc[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] += src_w
    hit = w_acc > 1e-6
    if not hit.any():
        raise RuntimeError("Path tracer produced no terrain hits — check camera framing")
    rgb = np.full((field, field, 3), np.nan, dtype=np.float32)
    rgb[hit] = (rgb_acc[hit] / w_acc[hit][:, None]).astype(np.float32)
    return rgb, hit


# --- modulation and compose --------------------------------------------------


def _center_map_horizontally() -> None:
    """Make the composer center the subject instead of pinning it left.

    ``LAYOUT["map_x"]`` is a fraction of canvas width, and 0.008 was tuned for
    Romania, whose 1.37:1 silhouette fills the canvas to within ~11 px. France's
    render grid is 1.08:1, so the height limit binds first and the constant
    leaves ALL the slack on one side. Centering needs the resized width, which
    only exists after the composer's own resize -- so hook that call rather
    than re-deriving its sizing math here, where it would silently drift out
    of sync.
    """
    original = rom._resize_subject_to_layout

    def centered(subject, canvas_size, *, max_height=None):
        resized = original(subject, canvas_size, max_height=max_height)
        canvas_width = max(1, int(canvas_size[0]))
        slack = max(0, canvas_width - resized.width)
        rom.LAYOUT = {**rom.LAYOUT, "map_x": (slack / 2.0) / canvas_width}
        return resized

    rom._resize_subject_to_layout = centered


def _shade_on_overlay_grid(
    rgb_field: np.ndarray,
    hit: np.ndarray,
    overlay_size: tuple[int, int],
    *,
    modern_grade: bool = False,
) -> np.ndarray:
    """Crop the PT field to the terrain hit bbox and upsample to the overlay grid."""
    luminance = rgb_field @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = np.where(hit, luminance, np.nan).astype(np.float32)
    ys, xs = np.nonzero(hit)
    crop = luminance[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    finite = np.isfinite(crop)
    crop = np.where(finite, crop, float(np.nanmedian(crop)))

    high_pct = MODERN_SHADE_HIGH_PCT if modern_grade else SHADE_HIGH_PCT
    gamma = MODERN_SHADE_GAMMA if modern_grade else SHADE_GAMMA
    low = float(np.percentile(crop[finite], SHADE_LOW_PCT))
    high = float(np.percentile(crop[finite], high_pct))
    shade = np.clip((crop - low) / max(high - low, 1e-6), 0.0, 1.0)
    shade = np.power(shade, gamma, dtype=np.float32)

    shade_img = Image.fromarray(shade, mode="F").resize(overlay_size, Image.Resampling.BICUBIC)
    shade_hi = np.clip(np.asarray(shade_img, dtype=np.float32), 0.0, 1.0)
    if modern_grade:
        blur = ndimage.gaussian_filter(shade_hi, sigma=SHADE_UNSHARP_SIGMA)
        shade_hi = np.clip(shade_hi + SHADE_UNSHARP_AMOUNT * (shade_hi - blur), 0.0, 1.0)
    return shade_hi


def _cavity_scale(dem_grid: np.ndarray, overlay_size: tuple[int, int]) -> np.ndarray:
    curv = ndimage.gaussian_laplace(dem_grid, sigma=CAVITY_SIGMA)
    norm = float(np.percentile(np.abs(curv), 95.0))
    c = np.clip(curv / max(norm, 1e-9), -1.0, 1.0)
    scale = 1.0 - CAVITY_DARKEN * np.maximum(c, 0.0) + CAVITY_LIGHTEN * np.maximum(-c, 0.0)
    img = Image.fromarray(scale.astype(np.float32), mode="F").resize(
        overlay_size, Image.Resampling.BICUBIC
    )
    return np.asarray(img, dtype=np.float32)


def _modulate_overlay(
    overlay_path: Path,
    shade: np.ndarray,
    *,
    modern_grade: bool = False,
    cavity: np.ndarray | None = None,
) -> Image.Image:
    overlay = np.asarray(Image.open(overlay_path).convert("RGBA"), dtype=np.uint8).copy()
    rgb = overlay[:, :, :3].astype(np.float32) / 255.0

    hue, saturation, value = rom._rgb_to_hsv_channels(overlay[:, :, :3])
    markers = rom._builtup_marker_mask(hue, saturation, value)

    floor = MODERN_TERRAIN_FLOOR if modern_grade else TERRAIN_FLOOR
    gain = MODERN_TERRAIN_GAIN if modern_grade else TERRAIN_GAIN
    scale = floor + gain * shade
    marker_scale = MARKER_FLOOR + MARKER_GAIN * shade
    scale = np.where(markers, marker_scale, scale)
    if cavity is not None:
        # cavity is terrain emphasis only; hard-color markers stay exempt
        scale = scale * np.where(markers, 1.0, cavity)
    rgb = np.clip(rgb * scale[:, :, None], 0.0, 1.0)

    if modern_grade:
        lum = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[:, :, None]
        graded = np.clip(lum + (rgb - lum) * FINAL_CHROMA, 0.0, 1.0)
        graded = np.clip(graded * FINAL_BRIGHTNESS, 0.0, 1.0)
        rgb = np.where(markers[:, :, None], rgb, graded)

    overlay[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGBA")


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

    if args.measure_relief:
        dem_grid, land = _load_dem_grid(render_dem_path, int(args.grid_max))
        print(f"[relief] grid {dem_grid.shape[1]}x{dem_grid.shape[0]}, land px {int(land.sum())}")
        for relief in (2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, RELIEF_WORLD):
            print(
                f"  RELIEF {relief:5.1f} -> p90 land slope "
                f"{_p90_land_slope(dem_grid, land, relief):5.2f} deg"
            )
        return 0

    field_key = (
        f"g{args.grid_max}_c{args.tiles}_f{args.frame}_m{args.cell_margin}"
        f"_r{RELIEF_WORLD:g}_s{SUN_AZIMUTH:g}-{SUN_ELEVATION:g}"
        f"_tw{TOWER_WORLD:g}-{TOWER_GAMMA:g}-{TOWER_MIN_PERSONS:g}"
    )
    shade_cache = output_dir / f"pt_light_field_{field_key}.npz"
    reusable = sorted(output_dir.glob(f"pt_light_field_{field_key}*.npz"))
    if args.reuse_shade and reusable:
        shade_cache = reusable[-1]
        print(f"== Reusing cached PT light field: {shade_cache} ==")
        cached = np.load(shade_cache)
        rgb_field, hit = cached["rgb"], cached["hit"]
    else:
        print("== Path tracing the light field (PROMETHEUS) ==")
        dem_grid, land = _load_dem_grid(render_dem_path, int(args.grid_max))
        towers = _load_tower_grid(population_path, dem_grid.shape)
        traced = _traced_grid(dem_grid, towers)
        import hashlib

        grid_hash = hashlib.sha1(np.ascontiguousarray(traced).tobytes()).hexdigest()[:10]
        print(
            f"[PT] traced hash {grid_hash} p90 land slope (terrain only) "
            f"{_p90_land_slope(dem_grid, land, RELIEF_WORLD):.2f} deg at RELIEF {RELIEF_WORLD:g}"
        )
        shade_cache = output_dir / f"pt_light_field_{field_key}_{grid_hash}.npz"
        try:
            rgb_field, hit = _pt_light_field(
                traced, args, cache_dir / f"pt_cells_{field_key}_{grid_hash}"
            )
        except BaseException as exc:  # noqa: BLE001 - device loss surfaces as a panic
            if "device" in str(exc).lower() or "Queue::submit" in str(exc):
                print(f"[PT] device lost: {exc}\n[PT] relaunch to resume from the cell cache")
                return DEVICE_LOST_EXIT
            raise
        np.savez_compressed(shade_cache, rgb=rgb_field, hit=hit)
        print(f"[PT] light field cached: {shade_cache}")

    print("== Modulating class overlay by the PT light field ==")
    with Image.open(overlay_path) as overlay_probe:
        overlay_size = overlay_probe.size
    shade = _shade_on_overlay_grid(
        rgb_field, hit, overlay_size, modern_grade=bool(args.modern_grade)
    )
    cavity = None
    if args.cavity:
        print("== Cavity shading from the DEM grid ==")
        cavity_grid, _land = _load_dem_grid(render_dem_path, int(args.grid_max))
        cavity = _cavity_scale(cavity_grid, overlay_size)
    raw = _modulate_overlay(
        overlay_path, shade, modern_grade=bool(args.modern_grade), cavity=cavity
    )

    if args.raw_only:
        raw_path = snapshot.with_name(snapshot.stem + "_raw.png")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep RGBA: the alpha channel IS the country matte, and the composer's
        # behaviour depends on it. Saving RGB here would make this debug artifact
        # take a different path through the composer than the real render.
        raw.save(raw_path)
        print(f"Raw modulated subject saved to: {raw_path}")
        return 0

    print("== Composing final plate ==")
    _center_map_horizontally()
    rom._compose_snapshot(raw, snapshot)
    print(f"Success! Map saved to: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
