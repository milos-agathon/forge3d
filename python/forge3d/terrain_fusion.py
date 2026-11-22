"""Resolution-priority terrain fusion on one explicit preparation grid."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np

from .alignment import _require_rasterio, _resampling_method
from .preparation import (
    PreparationError, PreparationGrid, PreparationRecipe, PreparationSource,
    _FUSION_RESAMPLING, _fail, _height_conversion, _json, _local, _read_raster, _sha,
)

__all__ = ["TerrainFusion", "fuse_terrain"]

# Default GDAL warped VRT block dimensions, also used to batch point transforms.
_WARP_BLOCK_WIDTH, _WARP_BLOCK_HEIGHT = 512, 128
_SURFACE_KINDS = {"bare_earth": "bare_earth", "dsm": "DSM"}
_UNRESOLVED_REFERENCE = "unresolved"


def _reference_declared(value: str | None) -> bool:
    return isinstance(value, str) and value == value.strip() and value.casefold() not in {
        "", "unknown", "unspecified", "undefined", _UNRESOLVED_REFERENCE,
    }


def _surface_kind(value: Any, source_id: str) -> str:
    key = value.strip().casefold() if isinstance(value, str) else None
    kind = _SURFACE_KINDS.get(key)
    if kind is None:
        _fail("invalid_surface_kind", f"Unknown input surface kind {value!r}.",
              "Declare each input as bare_earth or DSM.", source_id)
    return kind


def _check_source_declarations(source: PreparationSource, metadata: Mapping[str, Any], kind: str) -> None:
    surface_kinds = [source.metadata.get("surface_kind")]
    reference = str(source.vertical_reference).casefold()
    for tags in (metadata.get("tags", {}), metadata.get("band_tags", {})):
        for key, value in tags.items():
            key = key.casefold()
            if key == "surface_kind":
                surface_kinds.append(value)
            if (key in {"vertical_reference", "height_system"} and value
                    and value.casefold() != reference):
                _fail("conflicting_vertical_reference",
                      f"File height reference {value!r} differs from {source.vertical_reference!r}.",
                      "Correct the declaration or supply a documented external datum conversion.", source.source_id)
    for value in surface_kinds:
        if value is not None and _surface_kind(value, source.source_id) != kind:
            _fail("invalid_surface_kind", f"Source metadata declares {value!r}, but the input declares {kind!r}.",
                  "Correct the source declaration to match the verified surface.", source.source_id)


def _check_grid(grid: PreparationGrid, source_id: str | None = None) -> None:
    _, a, b, _, d, e = grid.geotransform
    determinant = a * e - b * d
    if not math.isfinite(determinant) or determinant == 0:
        _fail("invalid_grid", "The affine determinant must be finite and nonzero.",
              "Use a finite invertible transform.", source_id)
    if not all(math.isfinite(value) for point in ((grid.width, 0), (0, grid.height), (grid.width, grid.height))
               for value in grid.transform * point):
        _fail("invalid_grid", "The affine produces nonfinite coordinates across the raster extent.",
              "Use finite grid bounds in the declared CRS.", source_id)


def _ground_steps(grid: PreparationGrid) -> tuple[float, float]:
    _check_grid(grid)
    crs = _require_rasterio().crs.CRS.from_user_input(grid.crs)
    if not crs.is_projected:
        _fail("unsupported_fusion_crs", "Distance-based fusion requires a projected target CRS.",
              "Choose a projected CRS with linear coordinates appropriate for the area.")
    _, a, b, _, d, e = grid.geotransform
    if b != 0 or d != 0:
        _fail("unsupported_fusion_grid", "The fusion distance transform uses axis-aligned target cells.",
              "Use an unrotated, unsheared target grid. Source grids may be rotated or sheared.")
    factor = crs.linear_units_factor[1]
    dx, dy = abs(a) * factor, abs(e) * factor
    if not (math.isfinite(dx) and math.isfinite(dy) and min(dx, dy) > 0
            and (0.5 * (min(dx, dy) / max(dx, dy))) ** 2 > 0):
        _fail("invalid_grid", "Target cell dimensions cannot be represented by the distance metric.",
              "Choose finite, positive cell dimensions in the target CRS.")
    return dx, dy


def _read_fusion_elevation(
    source: PreparationSource, path: Path, base: Path, reference: str, *,
    surface_kind: str = "bare_earth", allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, PreparationGrid, dict[str, Any]]:
    source_reference = source.vertical_reference
    if reference == _UNRESOLVED_REFERENCE:
        if not isinstance(source_reference, str) or not source_reference or source_reference != source_reference.strip():
            _fail("missing_vertical_reference", "The source needs an explicit height reference declaration.",
                  "Use the verified datum or explicitly declare unknown.", source.source_id)
        # Retain the source reference during conversion so only the units change.
        conversion_reference = source_reference
    else:
        if not _reference_declared(source_reference) or not _reference_declared(reference):
            _fail("missing_vertical_reference", "Fusion sampling requires verified source and target height references.",
                  "Declare the measurement datums.", source.source_id)
        conversion_reference = reference
    data, valid, grid, metadata = _read_raster(source, path, base, allow_empty=allow_empty, for_fusion=True)
    _check_grid(grid, source.source_id)
    _check_source_declarations(source, metadata, _surface_kind(surface_kind, source.source_id))
    with np.errstate(over="ignore", invalid="ignore"):
        converted = _height_conversion(data, valid, grid, source, conversion_reference)
    if not np.all(np.isfinite(converted[valid])):
        _fail("fusion_height_precision", "Converted heights must be finite.",
              "Check the physical height range and vertical reference.", source.source_id)
    return converted, valid, grid, metadata


def _column_distances(costs: np.ndarray, rows: np.ndarray, step_squared: float) -> np.ndarray:
    """Evaluate the lower envelope of parabolas centred at finite-cost rows."""
    sites: list[int] = []
    starts: list[float] = []
    for site in np.flatnonzero(np.isfinite(costs)).tolist():
        start = -math.inf
        while sites:
            previous = sites[-1]
            start = ((site + previous) / 2
                     + (float(costs[site]) - float(costs[previous]))
                     / (2 * step_squared * (site - previous)))
            if len(sites) == 1 or start > starts[-1]:
                break
            starts.pop()
            sites.pop()
        sites.append(site)
        starts.append(start)
    nearest = np.asarray(sites)[np.searchsorted(starts, rows, side="right") - 1]
    return costs[nearest] + step_squared * np.square(rows - nearest, dtype=np.float64)


def _transition_weights(valid: np.ndarray, overlap: np.ndarray, dx: float, dy: float,
                        width_m: float) -> np.ndarray:
    """Euclidean distance from sample centres to edges of the valid cell union."""
    height, width = valid.shape
    above = np.pad(valid, ((1, 1), (0, 0)), constant_values=False)
    beside = np.pad(valid, ((0, 0), (1, 1)), constant_values=False)
    horizontal = above[:-1] != above[1:]
    vertical = beside[:, :-1] != beside[:, 1:]

    # In half-cell coordinates, the closest point on a boundary edge to any
    # sample centre is either its midpoint or an endpoint (an orthogonal grid).
    normalization = max(dx, dy)
    x_step = dx / normalization / 2
    y_step = dy / normalization / 2
    x = np.arange(2 * width + 1)
    centres_x = x[1::2]
    row_costs = np.full((2 * height + 1, width), np.inf)
    points = np.empty(x.shape, dtype=bool)
    for row in range(2 * height + 1):
        if row % 2:
            points[::2] = vertical[row // 2]
            points[1::2] = False
        else:
            segments = horizontal[row // 2]
            points[1::2] = segments
            points[0] = False
            points[2::2] = segments
            points[:-1:2] |= segments
            # Corners of vertical edges are also sites, including when the
            # vertical boundary runs straight through an interior row.
            if row:
                points[::2] |= vertical[row // 2 - 1]
            if row < 2 * height:
                points[::2] |= vertical[row // 2]
        if points.any():
            previous = np.maximum.accumulate(np.where(points, x, -x.size))[1::2]
            following = np.minimum.accumulate(np.where(points, x, 2 * x.size)[::-1])[::-1][1::2]
            horizontal_steps = np.minimum(centres_x - previous, following - centres_x)
            row_costs[row] = (horizontal_steps * x_step) ** 2

    weights = np.zeros(valid.shape, dtype=np.float64)
    for col in np.flatnonzero(overlap.any(axis=0)):
        rows = np.flatnonzero(overlap[:, col])
        squared = _column_distances(row_costs[:, col], 2 * rows + 1, y_step ** 2)
        with np.errstate(over="ignore"):
            t = np.minimum(np.sqrt(squared) * normalization / width_m, 1.0)
        weights[rows, col] = t * t * (3 - 2 * t)
    return weights


def _copy_aligned_window(data: np.ndarray, valid: np.ndarray, source: PreparationGrid,
                         target: PreparationGrid, window: tuple[int, int, int, int]) -> np.ndarray | None:
    source_affine = source._lattice_transform()
    target_affine = target._lattice_transform()
    if source_affine.b != 0 or source_affine.d != 0 or (source_affine.a, source_affine.e) != (target_affine.a, target_affine.e):
        return None
    x = (target_affine.c - source_affine.c) / source_affine.a
    y = (target_affine.f - source_affine.f) / source_affine.e
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    col, row = round(x), round(y)
    if source_affine * (col, row) != (target_affine.c, target_affine.f):
        return None
    # The source and target have an exact whole-cell relationship. Copy the
    # intersecting data, including signed zero and any source mask holes.
    left, top, width, height = window
    source_col, source_row = source._lattice()[2:]
    target_col, target_row = target._lattice()[2:]
    col += target_col - source_col + left
    row += target_row - source_row + top
    x0, y0 = max(col, 0), max(row, 0)
    x1, y1 = min(col + width, source.width), min(row + height, source.height)
    destination = np.full((1, height, width), np.nan, dtype=np.float64)
    if x0 < x1 and y0 < y1:
        intersection = np.s_[:, y0:y1, x0:x1]
        destination[:, y0-row:y1-row, x0-col:x1-col] = np.where(valid[intersection], data[intersection], np.nan)
    return destination


def _point_transform(source: PreparationGrid, target: PreparationGrid, same_crs: bool) -> Any:
    if same_crs:
        return None
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("Cross-CRS fusion sampling uses pyproj from forge3d[raster].") from exc
    return Transformer.from_crs(target.crs, source.crs, always_xy=True)


def _source_cells(
    source: PreparationGrid, project: Any, x: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, tuple[int, np.ndarray, np.ndarray]]:
    # PROJ returns nonfinite coordinates for points outside its domain.
    if project is None:
        sx, sy = x, y
    elif x.size == 1:
        sx, sy = project.transform(x.item(), y.item(), errcheck=False)
    else:
        sx, sy = project.transform(x, y, errcheck=False)
    x0, a, b, y0, d, e = source.geotransform
    dx = np.asarray(sx).reshape(x.shape) - x0
    dy = np.asarray(sy).reshape(y.shape) - y0
    # Subtract the origin before applying the inverse linear transform.
    with np.errstate(invalid="ignore", over="ignore"):
        if b == d == 0:
            source_col, source_row = dx / a, dy / e
        else:
            determinant = a * e - b * d
            source_col = (e * dx - b * dy) / determinant
            source_row = (a * dy - d * dx) / determinant
    covered = ((source_col >= 0) & (source_col < source.width)
               & (source_row >= 0) & (source_row < source.height))
    return covered, (0, source_row[covered].astype(np.intp), source_col[covered].astype(np.intp))


def _warp_nearest(data: np.ndarray, valid: np.ndarray, source: PreparationGrid,
                  target: PreparationGrid, window: tuple[int, int, int, int], *, same_crs: bool) -> np.ndarray:
    project = _point_transform(source, target, same_crs)
    left, top, width, height = window
    destination = np.full((1, height, width), np.nan, dtype=np.float64)
    for row in range(0, height, _WARP_BLOCK_HEIGHT):
        for col in range(0, width, _WARP_BLOCK_WIDTH):
            row_end = min(row + _WARP_BLOCK_HEIGHT, height)
            col_end = min(col + _WARP_BLOCK_WIDTH, width)
            rows, cols = np.ogrid[top + row:top + row_end, left + col:left + col_end]
            covered, ids = _source_cells(source, project, *target.xy(cols, rows))
            tile = destination[0, row:row_end, col:col_end]
            tile[covered] = np.where(valid[ids], data[ids], np.nan)
    return destination


def _read_warp_window(
    dataset: Any, transform: Any, width: int, height: int, window: Any, *,
    exact: bool, warp_options: Mapping[str, Any],
) -> np.ndarray:
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy as copy_raster
    from rasterio.vrt import WarpedVRT

    with WarpedVRT(dataset, transform=transform, width=width, height=height, **warp_options) as warped:
        if not exact:
            return warped.read(1, window=window)
        # Rasterio 1.4.3 cannot build an explicit-grid WarpedVRT with
        # tolerance=0. GDAL's serialized VRT contains the exact transformer
        # under the approximator. Keep its kernel, grid and source options.
        with MemoryFile(ext=".vrt") as serialized:
            copy_raster(warped, serialized.name, driver="VRT")
            serialized.seek(0)
            vrt = ElementTree.fromstring(serialized.read())
        transformer = vrt.find("GDALWarpOptions/Transformer")
        if transformer is not None and transformer.find("GenImgProjTransformer") is not None:
            return warped.read(1, window=window)
        precise = vrt.find("GDALWarpOptions/Transformer/ApproxTransformer/BaseTransformer/GenImgProjTransformer")
        source = vrt.find("GDALWarpOptions/SourceDataset")
        if transformer is None or precise is None or source is None:
            _fail("fusion_exact_warp_unavailable", "GDAL did not provide an exact raster transformer.",
                  "Use a Rasterio/GDAL build with warped VRT support.")
        transformer[:] = [precise]
        source.text = dataset.name
        source.set("relativeToVRT", "0")
        with MemoryFile(ElementTree.tostring(vrt), ext=".vrt") as memory, memory.open() as reference:
            return reference.read(1, window=window)


def _read_parent_samples(dataset: Any, target: PreparationGrid, tile: np.ndarray,
                         block: Any, region: Any, missing: np.ndarray, warp_options: Mapping[str, Any]) -> None:
    from rasterio.transform import Affine
    from rasterio.windows import Window

    # Retry eligible cells on the unshifted parent affine. Reflect the
    # appropriate axis for negative indices so the origin itself never moves.
    tile_rows, tile_cols = np.nonzero(missing)
    parent_rows, parent_cols = tile_rows + region.row_off, tile_cols + region.col_off
    parent = target._lattice_transform()
    max_dimension = np.iinfo(np.intc).max
    for row_sign in (1, -1):
        for col_sign in (1, -1):
            selected = ((parent_rows >= 0) == (row_sign > 0)) & ((parent_cols >= 0) == (col_sign > 0))
            if not np.any(selected):
                continue
            rows = row_sign * parent_rows[selected] - (row_sign < 0)
            cols = col_sign * parent_cols[selected] - (col_sign < 0)
            top, left = int(rows.min()), int(cols.min())
            bottom, right = int(rows.max()) + 1, int(cols.max()) + 1
            if max(bottom, right) > max_dimension:
                _fail("unsupported_fusion_grid", "Declared parent indices exceed GDAL's raster dimensions.",
                      "Supply a parent grid closer to the requested target window.")
            # Use complete GDAL blocks independently of which crop needed a retry.
            row_end = block.row_off + block.height if row_sign > 0 else -block.row_off
            col_end = block.col_off + block.width if col_sign > 0 else -block.col_off
            height = min(row_end + (-row_end % _WARP_BLOCK_HEIGHT), max_dimension)
            width = min(col_end + (-col_end % _WARP_BLOCK_WIDTH), max_dimension)
            samples = _read_warp_window(
                dataset, parent * Affine.scale(col_sign, row_sign), width, height,
                Window(left, top, right - left, bottom - top), exact=True, warp_options=warp_options,
            )
            tile[tile_rows[selected], tile_cols[selected]] = samples[rows - top, cols - left]


def _warp_interpolated(
    dataset: Any, valid: np.ndarray, source: PreparationGrid, target: PreparationGrid, method: str,
    window: tuple[int, int, int, int], *, same_crs: bool, source_id: str,
) -> np.ndarray:
    from rasterio.transform import Affine
    from rasterio.windows import Window

    window_left, window_top, width, height = window
    project = _point_transform(source, target, same_crs)
    # Use a stable block index for separately reconstructed decimal grids.
    block_width, block_height = _WARP_BLOCK_WIDTH, _WARP_BLOCK_HEIGHT
    transform = target.transform
    origin_x, origin_y, origin_col, origin_row = target._lattice()
    x, dx = Fraction(str(origin_x)), Fraction(str(transform.a))
    y, dy = Fraction(str(origin_y)), Fraction(str(transform.e))
    left = window_left + origin_col
    top = window_top + origin_row
    # Start the block lattice at the first nonnegative corner on each world
    # axis, then count tiles in the direction of increasing pixel indices.
    col_index = (x // abs(dx)) * (1 if dx > 0 else -1)
    row_index = (y // abs(dy)) * (1 if dy > 0 else -1)
    first_col = left - ((col_index + left) % block_width)
    first_row = top - ((row_index + top) % block_height)
    if same_crs:
        # Translate the exact binary64 inputs in the original coordinate system.
        x, dx = Fraction(origin_x), Fraction(transform.a)
        y, dy = Fraction(origin_y), Fraction(transform.e)
    requested = Window(left, top, width, height)
    warp_options = {"crs": target.crs, "resampling": _resampling_method(method)}
    src_affine = dataset.transform
    if same_crs and src_affine.b == src_affine.d == 0:
        sx, sy = abs(src_affine.a / transform.a), abs(src_affine.e / transform.e)
        if all(math.isfinite(scale) and scale > 0 for scale in (sx, sy)):
            warp_options.update(XSCALE=sx, YSCALE=sy)
    destination = np.empty((1, height, width), dtype=np.float64)
    for row in range(first_row, top + height, block_height):
        for col in range(first_col, left + width, block_width):
            block = Window(col, row, block_width, block_height)
            region = requested.intersection(block)
            rows, cols = Window(region.col_off - left, region.row_off - top,
                                region.width, region.height).toslices()
            read_window = Window(region.col_off - col, region.row_off - row, region.width, region.height)
            affine = Affine(transform.a, 0, float(x + col * dx), 0, transform.e, float(y + row * dy))
            tile = _read_warp_window(dataset, affine, block_width, block_height, read_window,
                                     exact=not same_crs, warp_options=warp_options)
            # GDAL interpolates on the block affine. The declared centre and
            # decoded mask, rather than the rebased affine, decide eligibility.
            rr, cc = np.ogrid[window_top + rows.start:window_top + rows.stop,
                              window_left + cols.start:window_left + cols.stop]
            covered, ids = _source_cells(source, project, *target.xy(cc, rr))
            eligible = np.zeros_like(covered)
            eligible[covered] = valid[ids]
            tile[~eligible] = np.nan
            missing = eligible & ~np.isfinite(tile)
            if np.any(missing):
                _read_parent_samples(dataset, target, tile, block, region, missing, warp_options)
                unresolved = np.argwhere(missing & ~np.isfinite(tile))
                if unresolved.size:
                    sample_row, sample_col = unresolved[0]
                    _fail(
                        "fusion_interpolation_failed",
                        f"{method.capitalize()} interpolation produced no finite height for the valid sample "
                        f"at parent row {region.row_off + sample_row}, column {region.col_off + sample_col}.",
                        "Check the source mask and coordinate reference systems, or explicitly select nearest sampling.",
                        source_id,
                    )
            destination[0, rows, cols] = tile
    return destination


def _warp_on_target(data: np.ndarray, valid: np.ndarray, source: PreparationGrid,
                    target: PreparationGrid, method: str, *, source_id: str,
                    window: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    rasterio = _require_rasterio()
    crs = rasterio.crs.CRS
    same_crs = crs.from_user_input(source.crs) == crs.from_user_input(target.crs)
    region = window or (0, 0, target.width, target.height)
    if same_crs:
        copied = _copy_aligned_window(data, valid, source, target, region)
        if copied is not None:
            return copied, np.isfinite(copied)
    if method == "nearest":
        destination = _warp_nearest(data, valid, source, target, region, same_crs=same_crs)
        return destination, np.isfinite(destination)

    from rasterio.io import MemoryFile

    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=source.width, height=source.height, count=1,
                         dtype="float64", nodata=np.nan, crs=source.crs, transform=source.transform) as scratch:
            scratch.write(np.where(valid, data, np.nan))
        with memory.open() as scratch:
            destination = _warp_interpolated(scratch, valid, source, target, method, region,
                                             same_crs=same_crs, source_id=source_id)
    return destination, np.isfinite(destination)


def _transition_on_target(data: np.ndarray, valid: np.ndarray, source: PreparationGrid,
                          target: PreparationGrid, method: str, target_valid: np.ndarray,
                          overlap: np.ndarray, dx: float, dy: float, width_m: float, *,
                          source_id: str) -> np.ndarray:
    # A cropped target is not the edge of a source. Extend only the sides
    # from which a boundary could influence an overlapping target sample.
    if source.same_as(target):
        return _transition_weights(target_valid, overlap, dx, dy, width_m)
    rows, cols = np.nonzero(overlap)
    distances = (cols.min() + 0.5, rows.min() + 0.5,
                 target.width - cols.max() - 0.5, target.height - rows.max() - 0.5)
    maximum = (math.inf,) * 4
    crs = _require_rasterio().crs.CRS
    if crs.from_user_input(source.crs) == crs.from_user_input(target.crs):
        inverse = ~target.transform
        x, y = zip(*(inverse * (source.transform * point)
                     for point in ((0, 0), (0, source.height), (source.width, 0), (source.width, source.height))))
        maximum = (-min(x), -min(y), max(x) - target.width, max(y) - target.height)

    cells = [max(0.0, min(width_m / step - distance, bound))
             for step, distance, bound in zip((dx, dy, dx, dy), distances, maximum)]
    gdal_max = np.iinfo(np.intc).max
    if any(cell >= gdal_max for cell in cells):
        _fail("unsupported_fusion_grid", "Transition context exceeds GDAL's raster dimensions.",
              "Use a coarser target grid or a smaller transition width.")
    left, top, right, bottom = (math.ceil(cell) for cell in cells)
    if left == top == right == bottom == 0:
        return _transition_weights(target_valid, overlap, dx, dy, width_m)

    width, height = target.width + left + right, target.height + top + bottom
    if max(width, height) > gdal_max:
        _fail("unsupported_fusion_grid", "Transition context exceeds GDAL's raster dimensions.",
              "Use a coarser target grid or a smaller transition width.")
    _, context_valid = _warp_on_target(data, valid, source, target, method, source_id=source_id,
                                       window=(-left, -top, width, height))
    crop = (slice(top, top + target.height), slice(left, left + target.width))
    # Preserve the exact target warp and use the expanded warp for exterior
    # validity only. Resampling heights always uses the caller's target grid.
    context_valid[0][crop] = target_valid
    context_overlap = np.pad(overlap, ((top, bottom), (left, right)), constant_values=False)
    return _transition_weights(context_valid[0], context_overlap, dx, dy, width_m)[crop]


def _overlap_statistics(high: np.ndarray, low: np.ndarray, samples: np.ndarray,
                        source_id: str) -> dict[str, int | float | None]:
    count = int(np.count_nonzero(samples))
    if not count:
        return {"sample_count": 0, "median_offset_m": None, "median_absolute_deviation_m": None}
    with np.errstate(over="ignore", invalid="ignore"):
        delta = high[samples] - low[samples]
        median = float(np.median(delta))
        mad = float(np.median(np.abs(delta - median)))
    if not (math.isfinite(median) and math.isfinite(mad) and np.all(np.isfinite(delta))):
        _fail("fusion_height_precision", "Overlap statistics exceed float64 range.",
              "Check the physical height range of the inputs.", source_id)
    return {"sample_count": count, "median_offset_m": median, "median_absolute_deviation_m": mad}


def _low_slope_cells(height: np.ndarray, valid: np.ndarray, dx: float, dy: float, limit: float) -> np.ndarray:
    """Qualify cells with four valid cardinal neighbours using rise over run."""
    center = height[1:-1, 1:-1]
    supported = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
                 & valid[:-2, 1:-1] & valid[2:, 1:-1])
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.maximum(np.abs(center - height[1:-1, :-2]), np.abs(height[1:-1, 2:] - center)) / dx
        y = np.maximum(np.abs(center - height[:-2, 1:-1]), np.abs(height[2:, 1:-1] - center)) / dy
        low = supported & (np.hypot(x, y) <= limit)
    result = np.zeros_like(valid)
    result[1:-1, 1:-1] = low
    return result


def _blend_step_statistics(high: np.ndarray, low: np.ndarray, fused: np.ndarray,
                           weight: np.ndarray, overlap: np.ndarray,
                           source_id: str) -> dict[str, int | float] | None:
    statistics = {"pair_count": 0, "max_total_step_m": 0.0,
                  "max_blend_step_m": 0.0, "max_blend_step_bound_m": 0.0}
    valid = np.isfinite(fused)
    band = overlap & (weight > 0) & (weight < 1)
    # Compare horizontal and vertical neighbours once each. A pair can reach
    # outside the overlap if it has a valid endpoint with both source heights.
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        pairs = (valid[first] & valid[second] & (overlap[first] | overlap[second])
                 & (band[first] | band[second] | (weight[first] != weight[second])))
        count = int(np.count_nonzero(pairs))
        if not count:
            continue

        # End in the overlap so the end height can be evaluated at either
        # weight. This also covers the pairs entering or leaving a source.
        forward = overlap[second][pairs]
        start_weight = np.where(forward, weight[first][pairs], weight[second][pairs])
        end_weight = np.where(forward, weight[second][pairs], weight[first][pairs])
        start_height = np.where(forward, fused[first][pairs], fused[second][pairs])
        end_height = np.where(forward, fused[second][pairs], fused[first][pairs])
        end_high = np.where(forward, high[second][pairs], high[first][pairs])
        end_low = np.where(forward, low[second][pairs], low[first][pairs])

        info = np.finfo(np.float64)
        half_limit = info.max / 2
        # Halve the inputs if a difference could overflow even though all the
        # heights are finite. The scale is exact except at the subnormal end.
        scaled = any(np.max(np.abs(values)) > half_limit
                     for values in (start_height, end_height, end_high, end_low))
        if scaled:
            for values in (start_height, end_height, end_high, end_low):
                values *= 0.5

        # Holding the start weight at the end cell accounts for terrain rise.
        # The remaining height change is due to changing the blend weight.
        fixed_weight_height = end_high * start_weight + end_low * (1 - start_weight)
        total_step = np.abs(end_height - start_height)
        blend_step = np.abs(end_height - fixed_weight_height)
        bound = np.abs((end_high - end_low) * (end_weight - start_weight))

        # Each of the four end-height terms in the residual has at most five
        # rounded operations on its path. The absolute subnormal allowance
        # covers the two four-operation mixtures, the two differences, the
        # three operations in the bound, and three relevant scaling operations.
        u = info.eps / 2
        gamma = 5 * u / (1 - 5 * u)
        scale = np.maximum(np.abs(end_high), np.abs(end_low))
        roundoff = (4 * gamma) * scale + (13 + 3 * scaled) * info.smallest_subnormal
        if not np.all(blend_step - bound <= roundoff):
            _fail("fusion_blend_step_bound", "An adjacent blend-induced step exceeds the local bound.",
                  "Check the source heights and calculated transition weights.", source_id)

        statistics["pair_count"] += count
        for key, values in (("max_total_step_m", total_step), ("max_blend_step_m", blend_step),
                            ("max_blend_step_bound_m", bound)):
            peak = float(np.max(values))
            if scaled:
                # Beyond Float64's range the scaled value is already an exact
                # integer. Keep the reported magnitude finite in the JSON receipt.
                peak = 2 * int(peak) if peak > half_limit else 2 * peak
            statistics[key] = max(statistics[key], peak)
    return statistics if statistics["pair_count"] else None


@dataclass(frozen=True)
class TerrainFusion:
    """Float64 terrain and attribution. Weight planes follow priority order."""

    height: np.ndarray
    valid_mask: np.ndarray
    weights: np.ndarray
    grid: PreparationGrid
    vertical_reference: str
    source_ids: tuple[str, ...]
    overlap_statistics: dict[str, dict[str, Any]]
    record: dict[str, Any]

    @property
    def source_index(self) -> np.ndarray:
        """Index into source_ids, with priority breaking ties and -1 marking voids."""
        index = self.weights.argmax(axis=0).astype(np.int64, copy=False)
        index[~self.valid_mask] = -1
        return index

    @property
    def dominant_source_id(self) -> np.ndarray:
        """Per-cell ID, with priority breaking ties and None marking voids."""
        dominant = np.full(self.valid_mask.shape, None, dtype=object)
        dominant[self.valid_mask] = np.asarray(self.source_ids, dtype=object)[self.source_index[self.valid_mask]]
        return dominant

    def _check_arrays(self) -> None:
        shape = (self.grid.height, self.grid.width)
        if (self.height.shape != shape or self.height.dtype != np.float64
                or self.valid_mask.shape != shape or self.valid_mask.dtype != np.bool_
                or self.weights.shape != (len(self.source_ids), *shape)
                or self.weights.dtype != np.float64
                or not np.array_equal(np.isfinite(self.height), self.valid_mask)
                or not np.all(np.isnan(self.height[~self.valid_mask]))
                or not np.all((self.weights >= 0) & (self.weights <= 1))
                or np.any(self.weights[:, ~self.valid_mask])):
            _fail("invalid_fusion_assets", "Height, validity and weights disagree.",
                  "Regenerate the terrain and its attribution together.")
        # Unit roundoff and a conservative gamma bound for one complement,
        # one multiplication and the summation per source.
        nu = 3 * len(self.source_ids) * (np.finfo(np.float64).eps / 2)
        error_bound = nu / (1 - nu)
        if np.any(np.abs(self.weights.sum(axis=0)[self.valid_mask] - 1) > error_bound):
            _fail("invalid_fusion_assets", "Valid-cell contribution weights do not sum to one.",
                  "Regenerate the terrain and its attribution together.")

    def save(self, directory: str | Path) -> Path:
        """Publish a float64 GeoTIFF, attribution rasters, receipt and preparation recipe."""
        self._check_arrays()
        root = Path(directory).resolve()
        if root.exists():
            _fail("output_exists", str(root), "Choose a new output directory.")
        rasterio = _require_rasterio()
        surface_kind = self.record.get("surface_kind", "bare_earth")
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
                staging = Path(temporary)
                height_path = staging / "terrain.tif"
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
                    height_path, "w", driver="GTiff", width=self.grid.width, height=self.grid.height,
                    count=1, dtype="float64", crs=self.grid.crs, transform=self.grid.transform,
                    nodata=np.nan,
                ) as dataset:
                    dataset.write(self.height, 1)
                    dataset.write_mask(self.valid_mask)
                    dataset.update_tags(AREA_OR_POINT="Area", vertical_reference=self.vertical_reference,
                                        surface_kind=surface_kind)
                    dataset.set_band_unit(1, "m")
                weights_path = staging / "weights.npy"
                np.save(weights_path, self.weights, allow_pickle=False)
                index_path = staging / "source_index.npy"
                np.save(index_path, self.source_index.astype("<i8", copy=False), allow_pickle=False)
                receipt = staging / "fusion.json"
                assets = {
                    "terrain.tif": _sha(height_path),
                    "weights.npy": _sha(weights_path),
                    "source_index.npy": _sha(index_path),
                }
                receipt.write_text(_json({**self.record, "assets": assets}) + "\n", encoding="utf-8")
                source = PreparationSource(
                    "fused_terrain", "terrain.tif", "elevation", grid=self.grid, units="m",
                    vertical_reference=self.vertical_reference,
                    metadata={"surface_kind": surface_kind, "fusion_receipt_sha256": _sha(receipt)},
                )
                PreparationRecipe(self.grid, (source,), self.vertical_reference).save(staging / "recipe.json")
                staging.rename(root)
        except (OSError, ValueError, TypeError, rasterio.errors.RasterioError) as exc:
            if isinstance(exc, PreparationError):
                raise
            raise PreparationError("fusion_write_failed", str(exc),
                                   "Use a writable, new output directory.") from exc
        return root

    @classmethod
    def load(cls, directory: str | Path) -> TerrainFusion:
        """Reopen a saved fusion without access to its original DEMs."""
        root = Path(directory).resolve()
        try:
            recipe = PreparationRecipe.load(root / "recipe.json")
            receipt = (root / "fusion.json").read_bytes()
            if (len(recipe.sources) != 1 or recipe.sources[0].metadata.get("fusion_receipt_sha256")
                    != hashlib.sha256(receipt).hexdigest()):
                raise ValueError("Checksum mismatch for fusion.json")
            record = json.loads(receipt)
            _json(record)
            if record["schema_version"] != 1:
                raise ValueError("Unsupported fusion receipt version")
            grid = PreparationGrid(**record["target_grid"])
            _ground_steps(grid)
            reference = record["vertical_reference"]
            policy = record.get("vertical_reference_policy", "verified")
            verified = policy == "verified" and _reference_declared(reference)
            unresolved = policy == "unresolved" and reference == _UNRESOLVED_REFERENCE
            surface_kind = record.get("surface_kind", "bare_earth")
            if (surface_kind not in {"bare_earth", "DSM", "mixed"}
                    or (surface_kind == "mixed" and record.get("allow_mixed_surfaces") is not True)
                    or recipe.sources[0].metadata.get("surface_kind") != surface_kind):
                raise ValueError("Invalid or conflicting output surface kind")
            ids = tuple(record["source_ids"])
            if (not (verified or unresolved) or not ids
                    or not all(isinstance(i, str) and i.strip() for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Invalid vertical reference or source order")
            assets = record["assets"]
            names = ["terrain.tif", "weights.npy"]
            if "source_index.npy" in assets:
                names.append("source_index.npy")
            for name in names:
                if _sha(root / name) != assets[name]:
                    raise ValueError(f"Checksum mismatch for {name}")
            source = PreparationSource("fused_terrain", "terrain.tif", "elevation", grid=grid,
                                       units="m", vertical_reference=reference)
            data, valid, _, metadata = _read_raster(source, root / "terrain.tif", root, allow_empty=True)
            if (metadata["ancillary_identities"] or metadata["file_units"] != ["m"]
                    or metadata["tags"].get("vertical_reference") != reference
                    or metadata["tags"].get("surface_kind") != surface_kind):
                raise ValueError("Unexpected fused raster metadata or sidecar")
            weights = np.load(root / "weights.npy", allow_pickle=False)
            result = cls(data[0], valid[0], weights, grid, reference, ids,
                         record["overlap_statistics"], record)
            result._check_arrays()
            if "source_index.npy" in assets:
                index = np.load(root / "source_index.npy", allow_pickle=False)
                if index.dtype != np.dtype("<i8") or not np.array_equal(index, result.source_index):
                    raise ValueError("Source index disagrees with the contribution weights")
            return result
        except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise PreparationError("invalid_fusion_assets", str(exc),
                                   "Restore the original fusion directory or regenerate it.") from exc


def fuse_terrain(sources: Sequence[PreparationSource], target_grid: PreparationGrid, *,
                 vertical_reference: str, surface_kinds: Mapping[str, str],
                 transition_width_m: float, calibration_masks: Mapping[str, np.ndarray] | None = None,
                 bias_estimation_masks: Mapping[str, np.ndarray] | None = None, max_bias_slope: float | None = None,
                 allow_mixed_surfaces: bool = False, allow_unresolved_vertical_reference: bool = False,
                 base_dir: str | Path = ".") -> TerrainFusion:
    """Fuse local elevation models in priority order, highest first.

    Estimation masks select stable, low-slope cells for median bias estimates.
    A calibration mask selects cells and opts its source into correction.
    A slope limit additionally filters both surfaces. Masks live on the target grid.
    """
    sources = tuple(sources)
    if not sources or any(not isinstance(s, PreparationSource) or s.kind != "elevation"
                          or not isinstance(s.source_id, str) or not s.source_id.strip() for s in sources):
        _fail("invalid_fusion_sources", "Fusion requires at least one elevation source with a nonempty ID.",
              "Supply ordered elevation PreparationSource inputs.")
    ids = tuple(source.source_id for source in sources)
    if len(set(ids)) != len(ids):
        _fail("invalid_fusion_sources", "Source IDs must be unique.", "Assign each DEM a distinct ID.")
    if not isinstance(surface_kinds, Mapping) or set(surface_kinds) != set(ids):
        _fail("invalid_surface_kind", "Every DEM needs an explicit surface-kind declaration.",
              "Supply surface_kinds keyed by source ID, with values bare_earth or DSM.")
    kinds = {source_id: _surface_kind(surface_kinds[source_id], source_id) for source_id in ids}
    if not isinstance(allow_mixed_surfaces, bool):
        _fail("invalid_surface_kind", "The mixed-surface permission must be boolean.",
              "Set allow_mixed_surfaces=True only for a deliberately mixed output.")
    mixed = len(set(kinds.values())) > 1
    if mixed and not allow_mixed_surfaces:
        _fail("invalid_surface_kind", "The sources have different surface types.",
              "Use one surface type or explicitly set allow_mixed_surfaces=True.")
    unresolved = vertical_reference == _UNRESOLVED_REFERENCE
    if (not isinstance(allow_unresolved_vertical_reference, bool)
            or (allow_unresolved_vertical_reference and not unresolved)):
        _fail("invalid_vertical_reference_policy", "Allowing unresolved references requires an unresolved output label.",
              "Set vertical_reference='unresolved' with allow_unresolved_vertical_reference=True.")
    if not (_reference_declared(vertical_reference) or (unresolved and allow_unresolved_vertical_reference)):
        _fail("missing_vertical_reference", "Target height reference is unknown.",
              "Declare the verified measurement datum.")
    try:
        if isinstance(transition_width_m, (bool, np.bool_)):
            raise ValueError
        width_m = float(transition_width_m)
    except (ValueError, TypeError, OverflowError):
        width_m = math.nan
    if not math.isfinite(width_m) or width_m <= 0:
        _fail("invalid_transition_width", "Transition width must be finite and positive.",
              "Supply the transition width in ground metres.")
    dx, dy = _ground_steps(target_grid)
    shape = (target_grid.height, target_grid.width)
    corrections = {} if calibration_masks is None else calibration_masks
    estimates = {} if bias_estimation_masks is None else bias_estimation_masks
    for masks, code in ((corrections, "invalid_calibration_mask"), (estimates, "invalid_bias_estimation_mask")):
        if not isinstance(masks, Mapping) or not set(masks) <= set(ids[:-1]):
            _fail(code, "A stable-cell mask requires a source with lower priorities.",
                  "Key masks by a higher-priority input source ID.")
        for source_id, mask in masks.items():
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != shape:
                _fail(code, "Stable-cell masks must be boolean arrays on the target grid.",
                      "Supply True for eligible target cells.", source_id)
    if set(corrections) & set(estimates):
        _fail("invalid_bias_estimation_mask", "A source has two stable-cell masks.",
              "Use calibration_masks when applying the estimate and bias_estimation_masks when reporting it alone.")
    selections = {**estimates, **corrections}
    slope_limit = None
    if max_bias_slope is not None:
        try:
            if isinstance(max_bias_slope, (bool, np.bool_)):
                raise ValueError
            slope_limit = float(max_bias_slope)
        except (ValueError, TypeError, OverflowError):
            slope_limit = math.nan
        if not math.isfinite(slope_limit) or slope_limit < 0:
            _fail("invalid_bias_slope", "The slope limit must be finite and nonnegative.",
                  "Supply a maximum rise per metre of horizontal run.")

    base = Path(base_dir).resolve()
    height = np.full(shape, np.nan, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    weights = np.zeros((len(sources), *shape), dtype=np.float64)
    overlap_statistics: dict[str, dict[str, Any]] = {}
    source_records = []
    for index in reversed(range(len(sources))):
        source = sources[index]
        sid = source.source_id
        path = _local(source.path, base)
        identity = {"path": str(path), "sha256": _sha(path)}
        converted, source_valid, grid, metadata = _read_fusion_elevation(
            source, path, base, vertical_reference, surface_kind=kinds[sid], allow_empty=True,
        )
        method = (source.resampling or "bilinear").lower()
        method = _FUSION_RESAMPLING.get(method, method)
        if method not in {"nearest", "bilinear", "cubic"}:
            _fail("invalid_elevation_resampling", f"Unsupported terrain sampling {method!r}.",
                  "Use nearest, bilinear or cubic.", sid)
        same_grid = grid.same_as(target_grid)
        warped, warped_valid = ((converted, source_valid) if same_grid else
                                _warp_on_target(converted, source_valid, grid, target_grid, method, source_id=sid))
        high, usable = warped[0], warped_valid[0]

        overlap = usable & valid
        stats: dict[str, Any] = _overlap_statistics(high, height, overlap, sid)
        stats["applied_offset_m"] = 0.0
        stats["bias_estimate"] = None
        if sid in selections:
            mask = selections[sid]
            eligible = overlap & mask
            if slope_limit is not None:
                eligible &= _low_slope_cells(high, usable, dx, dy, slope_limit)
                eligible &= _low_slope_cells(height, valid, dx, dy, slope_limit)
            estimate = {
                **_overlap_statistics(high, height, eligible, sid),
                "selection": "stable_low_slope_target_cells",
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "max_slope_m_per_m": slope_limit,
            }
            stats["bias_estimate"] = estimate
        if sid in corrections:
            estimate = stats["bias_estimate"]
            median = estimate["median_offset_m"]
            if median is None:
                _fail("empty_calibration_overlap", "Calibration selects no valid comparable samples.",
                      "Choose stable target cells valid in both surfaces and satisfying the slope limit.", sid)
            stats["calibration"] = {**estimate, "selection": "explicit_target_boolean_mask"}
            stats["applied_offset_m"] = -median
            if median != 0:
                with np.errstate(over="ignore"):
                    high[usable] -= median
                if not np.all(np.isfinite(high[usable])):
                    _fail("fusion_height_precision", "Bias correction exceeds float64 range.",
                          "Check the calibration selection and declared height range.", sid)
        overlap_statistics[sid] = stats

        only_high = usable & ~valid
        height[only_high] = high[only_high]
        weights[index, only_high] = 1
        if overlap.any():
            lower = height.copy()
            transition = _transition_on_target(converted, source_valid, grid, target_grid, method,
                                               usable, overlap, dx, dy, width_m, source_id=sid)
            weights[index, overlap] = transition[overlap]
            weights[index + 1:] *= 1 - transition
            inside = overlap & (transition == 1)
            height[inside] = high[inside]
            blend = overlap & (transition > 0) & ~inside & (high != height)
            t = transition[blend]
            with np.errstate(over="ignore"):
                height[blend] = high[blend] * t + height[blend] * (1 - t)
            if not np.all(np.isfinite(height[blend])):
                _fail("fusion_height_precision", "Blended heights exceed float64 range.",
                      "Check the physical height range of the inputs.", sid)
            steps = _blend_step_statistics(high, lower, height, weights[index], overlap, sid)
            if steps is not None:
                stats["blend_steps"] = steps
        valid |= usable
        mask_identity = None
        if source.mask_path is not None:
            mask_path = _local(source.mask_path, base)
            mask_identity = {"path": str(mask_path), "sha256": _sha(mask_path)}
        source_records.append({
            "source": source.to_dict(), "identity": identity, "mask_identity": mask_identity,
            "reader_metadata": metadata, "source_grid": grid.to_dict(),
            "surface_kind": kinds[sid], "warped_valid_cells": int(np.count_nonzero(usable)),
            "operations": [
                {"operation": "masked_decode", "validity": "intersection of file mask, raw file and explicit nodata, finite values, and explicit mask"},
                {"operation": "height_units" if unresolved else "height_reference",
                 "from": source.vertical_reference, "to": vertical_reference,
                 "input_units": source.units, "output_units": "m"},
                {"operation": "retain_grid" if same_grid else "geographic_warp",
                 "resampling": "none" if same_grid else method},
                {"operation": "vertical_bias", "offset_m": stats["applied_offset_m"]},
            ],
        })

    record = {
        "schema_version": 1, "target_grid": target_grid.to_dict(),
        "vertical_reference": vertical_reference, "height_units": "m", "source_ids": list(ids),
        "surface_kind": "mixed" if mixed else kinds[ids[0]], "allow_mixed_surfaces": allow_mixed_surfaces,
        "vertical_reference_policy": "unresolved" if unresolved else "verified",
        "max_bias_slope_m_per_m": slope_limit,
        "sources": list(reversed(source_records)), "target_cell_size_m": [dx, dy],
        "pixel_registration": "GDAL corner-normalized; samples at half-pixel centres",
        "transition_width_m": width_m, "transition": "smoothstep_3t_squared_minus_2t_cubed",
        "distance": "Euclidean from sample centre to warped valid-cell edges, including holes",
        "transition_context": "extend beyond target wherever exterior validity can influence the blend",
        "overlap_statistics": overlap_statistics,
        "offset_sign": "current higher-priority height minus accumulated lower-priority height before correction",
        "valid_cells": int(np.count_nonzero(valid)),
    }
    try:
        _json(record)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreparationError("invalid_fusion_metadata", str(exc), "Use finite JSON-compatible source metadata.") from exc
    return TerrainFusion(height, valid, weights, target_grid, vertical_reference, ids, overlap_statistics, record)
