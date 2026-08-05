# tools/europe_smoke/population.py
"""GHSL 30ss -> CAMS 0.4 deg population, by exact block-48 summation (§1.3).

Population is an extensive COUNT, so the aggregation is a SUM, not a mean.
0.4 * 120 = 48 exactly, but CAMS cell CENTRES are multiples of 0.4, so cell
EDGES land on 0.4k +/- 0.2 -- a half-cell, 24-pixel offset from any
round-degree tiling. Anchoring on the edge lattice is what makes the reduce
exactly mass-conserving.

Do NOT use rasterio warp Resampling.sum: measured 37,736 people max per-cell
error and 60x slower.
"""
from __future__ import annotations

import numpy as np

from . import config

BLOCK = 48  # 0.4 deg / (1/120 deg)


def block_anchor(lon: np.ndarray, lat: np.ndarray, transform) -> tuple[int, int, int, int]:
    """Pixel window covering the cell EDGES of the given cell-centre axes."""
    west = float(lon[0]) - config.LATTICE / 2.0
    north = float(lat[0]) + config.LATTICE / 2.0
    col0 = round((west - transform.c) / transform.a)
    row0 = round((north - transform.f) / transform.e)
    return col0, row0, lon.size * BLOCK, lat.size * BLOCK


def aggregate(lon: np.ndarray, lat: np.ndarray, return_raw_sum: bool = False):
    """Sum GHSL population into each CAMS cell. Returns (nlat, nlon) float64."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(config.GHSL_TIF) as src:
        col0, row0, ncols, nrows = block_anchor(lon, lat, src.transform)
        if col0 < 0 or row0 < 0 or col0 + ncols > src.width or row0 + nrows > src.height:
            raise RuntimeError(
                f"window ({col0},{row0},{ncols},{nrows}) leaves the raster "
                f"({src.width}x{src.height})"
            )
        arr = src.read(1, window=Window(col0, row0, ncols, nrows)).astype("float64")

    raw_sum = float(arr.sum())
    grid = arr.reshape(lat.size, BLOCK, lon.size, BLOCK).sum(axis=(1, 3))
    if return_raw_sum:
        return grid, raw_sum
    return grid


def display_block(grid: np.ndarray) -> np.ndarray:
    """The 106x175 cell-centre block every printed figure uses (§1.3).

    The full domain carries ~219.74 M people of Atlantic, Saharan, Russian and
    Arctic margin that must never surface in the UI.
    """
    lon, lat = config.grid_axes()
    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    ci = np.flatnonzero((lon >= lon_min - 1e-9) & (lon <= lon_max + 1e-9))
    ri = np.flatnonzero((lat >= lat_min - 1e-9) & (lat <= lat_max + 1e-9))
    return grid[np.ix_(ri, ci)]




def largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    """Round ``values`` to integers summing exactly to ``total``.

    Deterministic: ties break by index, so two runs on equal input agree.
    """
    v = np.asarray(values, dtype="float64")
    floor = np.floor(v).astype("int64")
    remainder = int(total - floor.sum())
    if remainder > 0:
        order = np.lexsort((np.arange(v.size), -(v - floor)))
        floor[order[:remainder]] += 1
    elif remainder < 0:
        order = np.lexsort((np.arange(v.size), v - floor))
        floor[order[:-remainder]] -= 1
    return floor


def masked_sum(values: np.ndarray, mask: np.ndarray, rows: int = 512) -> float:
    """``values[mask].sum()`` without a 300 MB boolean-index temporary."""
    total = 0.0
    for r0 in range(0, values.shape[0], rows):
        sl = slice(r0, r0 + rows)
        total += float(values[sl][mask[sl]].sum())
    return total


def rasterise_countries(cache_dir, lon: np.ndarray, lat: np.ndarray,
                        weights: np.ndarray | None = None,
                        return_stats: bool = False):
    """Country code per GHSL pixel, with a nearest-country fill.

    ~1% of domain population lands on no polygon (a coastline-resolution
    artifact); ~96% of that mass is within 3 px. The fill is a Euclidean
    transform in INDEX space -- a <=3 px snap heuristic, NOT a geodesic
    nearest neighbour, because a pixel is 0.9266 km tall but 0.2863 km wide
    at 72 N.

    ``weights`` is the population window; when given, the stats carry how many
    people the fill had to rescue. ``return_stats`` appends a third return
    value and is what makes gate 8's "0 pixels unassigned" a measurement
    rather than an assumption.
    """
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    from .boundary import load_countries

    gdf = load_countries(cache_dir).reset_index(drop=True)
    names = list(gdf["ADM0_A3"].astype(str))

    with rasterio.open(config.GHSL_TIF) as src:
        col0, row0, ncols, nrows = block_anchor(lon, lat, src.transform)
        transform = src.window_transform(
            rasterio.windows.Window(col0, row0, ncols, nrows))

    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    codes = rasterize(shapes, out_shape=(nrows, ncols), transform=transform,
                      fill=0, dtype="int32")

    unassigned = codes == 0
    stats = {
        "countries": names,
        "n_countries": len(names),
        "total_pixels": int(codes.size),
        "unassigned_pixels_before_fill": int(unassigned.sum()),
        "unassigned_people_before_fill": (
            masked_sum(weights, unassigned) if weights is not None else None),
    }
    if return_stats and weights is None:
        # Hand the caller the mask instead of demanding the population window
        # up front. Passing ``weights`` forces the caller to hold a 604 MB
        # float64 array across the distance transform, whose own index arrays
        # already cost ~1.8 GB: measured peak working set 3,636 MB against
        # 3,029 MB when the window is read afterwards. The mask is 75 MB.
        # ``country_cell_table`` pops this; it must never reach a JSON report.
        stats["_unassigned_mask"] = unassigned
    if unassigned.any():
        _, (ri, ci) = distance_transform_edt(unassigned, return_indices=True)
        codes = codes[ri, ci]
    codes = codes - 1
    stats["unassigned_pixels_after_fill"] = int((codes < 0).sum())
    return (codes, names, stats) if return_stats else (codes, names)


def country_cell_table(cache_dir, return_stats: bool = False):
    """(ADM0_A3, cams_row, cams_col, people) with exact closure."""
    import rasterio
    from rasterio.windows import Window

    lon, lat = config.grid_axes()

    # Rasterise FIRST, then read the window. The reverse order pins 604 MB of
    # float64 population across the distance transform for no benefit.
    codes, names, stats = rasterise_countries(cache_dir, lon, lat, return_stats=True)
    mask = stats.pop("_unassigned_mask", None)

    with rasterio.open(config.GHSL_TIF) as src:
        col0, row0, ncols, nrows = block_anchor(lon, lat, src.transform)
        pop = src.read(1, window=Window(col0, row0, ncols, nrows)).astype("float64")

    if mask is not None:
        stats["unassigned_people_before_fill"] = masked_sum(pop, mask)
        del mask

    nlat, nlon = lat.size, lon.size
    ncountry = len(names)

    # One pass with bincount, not a mask per country: the window is
    # 6528 x 11568 = 75.5 M pixels, so 45 boolean masks would allocate ~3.4 GB
    # and take minutes. Encode (cell, country) as a single integer key instead.
    # int32 keys: the largest is nlat*nlon*ncountry ~ 1.5 M, so int64 would
    # double the 75.5 M-element temporary for nothing.
    rows = np.repeat(np.arange(nlat, dtype="int32"), BLOCK)[:, None]
    cols = np.repeat(np.arange(nlon, dtype="int32"), BLOCK)[None, :]
    key = (rows * nlon + cols) * ncountry + codes.astype("int32")
    counts = np.bincount(key.ravel(), weights=pop.ravel(),
                         minlength=nlat * nlon * ncountry)

    nz = np.flatnonzero(counts > 0)
    vals = counts[nz]
    keys = [(int(k % ncountry), int((k // ncountry) // nlon), int((k // ncountry) % nlon))
            for k in nz]
    # round ALL positive pairs first; dropping sub-1 rows before rounding
    # loses 76,300 people and closure fails
    rounded = largest_remainder(vals, total=int(round(vals.sum())))
    out = [(names[k[0]], k[1], k[2], int(n))
           for k, n in zip(keys, rounded) if n > 0]
    if return_stats:
        stats.update(
            attributed_people_float=float(vals.sum()),
            table_people=int(rounded.sum()),
            pairs_before_dropping_zeros=int(nz.size),
            table_rows=len(out),
        )
        return out, stats
    return out
