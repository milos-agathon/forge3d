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


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print("prep/trace stages arrive in later tasks; only --self-test is wired")
    raise SystemExit(2)
