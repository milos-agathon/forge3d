"""Declared source-grid repairs. Every entry point returns the change record."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .dem_audit import _surface


def _selection(shape, region):
    """Select a finding's cell, half-open bounds, or polygon with holes."""
    if not isinstance(region, Mapping):
        raise ValueError("supply a confirmed source-grid repair region")
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if set(region) == {"cells"}:
        for cell in region["cells"]:
            c, r = cell
            if not (
                isinstance(c, int) and isinstance(r, int) and 0 <= r < h and 0 <= c < w
            ):
                raise ValueError("repair cell outside source grid")
            mask[r, c] = True
    elif set(region) == {"bounds"}:
        c, r, end_c, end_r = region["bounds"]
        if not (0 <= c < end_c <= w and 0 <= r < end_r <= h):
            raise ValueError("repair bounds outside source grid")
        mask[r:end_r, c:end_c] = True
    elif set(region) == {"rings"}:
        from rasterio.features import geometry_mask
        from affine import Affine

        mask = geometry_mask(
            [{"type": "Polygon", "coordinates": region["rings"]}],
            shape,
            Affine.identity(),
            invert=True,
        )
    else:
        raise ValueError("region requires cells, bounds, or rings")
    return mask


def _notch(z, selected, frequencies, axis):
    """Remove the requested harmonic components on the selected valid samples."""
    frequencies = np.asarray(frequencies, dtype=float)
    if (
        axis not in ("row", "column")
        or frequencies.ndim != 1
        or not frequencies.size
        or not np.all((frequencies > 0) & (frequencies <= 0.5))
        or np.unique(frequencies).size != frequencies.size
    ):
        raise ValueError(
            "supply an axis and distinct frequencies in (0, 0.5] cycles/cell"
        )
    a = np.where(selected, z, 0.0)
    across = 1 if axis == "row" else 0
    count = selected.sum(axis=across)
    keep = count > 0
    t = np.arange(count.size, dtype=float)
    columns = [np.ones_like(t), t - t.mean()]
    for frequency in frequencies:
        phase = 2 * np.pi * frequency * t
        columns.append(np.cos(phase))
        if frequency != 0.5:
            columns.append(np.sin(phase))
    design = np.column_stack(columns)
    weight = np.sqrt(count[keep])
    coeff, _, rank, _ = np.linalg.lstsq(
        design[keep] * weight[:, None], a.sum(axis=across)[keep] / weight, rcond=None
    )
    if rank < design.shape[1]:
        raise ValueError(
            "insufficient valid lines to resolve the requested frequencies"
        )
    correction = design[:, 2:] @ coeff[2:]
    return z - (correction[:, None] if axis == "row" else correction)


def apply_repair(dem, operation, *, region, **parameters):
    a = np.ma.asarray(dem)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("repairs require a two-dimensional real raster")
    original = a.data
    valid = ~np.ma.getmaskarray(a) & np.isfinite(original)
    region, parameters = deepcopy((region, parameters))
    target = _selection(original.shape, region)
    out = original.copy()
    after = valid.copy()
    z = np.where(valid, original, np.nan).astype(np.float64)
    if operation == "fill_missing":
        parameters.setdefault("method", "nearest")
        if parameters != {"method": "nearest"}:
            raise ValueError("fill_missing supports method='nearest'")
        target &= ~valid
        if target.any():
            if not valid.any():
                raise ValueError("fill_missing requires a valid donor")
            try:
                from scipy.ndimage import distance_transform_edt
            except ImportError as exc:
                raise ImportError("nearest repair requires SciPy") from exc
            indices = distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            out[target] = original[tuple(indices[:, target])]
            after[target] = True
    else:
        target &= valid
        if operation == "remove_spikes" and not parameters:
            candidate, _ = _surface(z)
        elif operation == "notch_filter" and set(parameters) == {"frequencies", "axis"}:
            candidate = _notch(z, target, **parameters)
        elif operation == "dequantize" and set(parameters) == {"step"}:
            step = parameters["step"]
            if not np.isfinite(step) or step <= 0 or original.dtype.kind != "f":
                raise ValueError(
                    "dequantize requires a floating DEM and a positive height step"
                )
            surface, _ = _surface(z)
            candidate = z + np.clip(surface - z, -step / 2, step / 2)
        else:
            raise ValueError(f"unknown operation or parameters for {operation!r}")
        target &= np.isfinite(candidate)
        if original.dtype.kind in "iu":
            candidate = np.clip(
                np.rint(candidate),
                np.iinfo(original.dtype).min,
                np.iinfo(original.dtype).max,
            )
        out[target] = candidate[target]
        if not np.all(np.isfinite(out[target])):
            raise ValueError("repair exceeds representable elevation range")
    changed = target & ((after != valid) | (out != original))
    out[target & ~changed] = original[target & ~changed]
    paired = changed & valid & after
    delta = abs(out[paired].astype(float) - original[paired].astype(float))
    # A missing observation has no physical before/after height difference.
    runs = []
    for r in np.flatnonzero(changed.any(axis=1)):
        edges = np.flatnonzero(np.diff(np.r_[False, changed[r], False]))
        runs.extend(
            [int(r), int(c), int(end)] for c, end in zip(edges[::2], edges[1::2])
        )
    count = int(changed.sum())
    record = {
        "operation": operation,
        "parameters": {"region": region, **parameters},
        "cells_changed": count,
        "changed_runs": runs,
        "measured_changes": int(delta.size),
        "newly_valid": int((after & ~valid).sum()),
        "maximum_change": float(delta.max())
        if delta.size
        else (None if count else 0.0),
        "mean_change": float(delta.mean()) if delta.size else (None if count else 0.0),
    }
    if record["newly_valid"]:
        filled = out[after & ~valid]
        record["filled_elevations"] = {
            "min": float(filled.min()),
            "max": float(filled.max()),
            "mean": float(np.mean(filled, dtype=np.float64)),
        }
    mask = np.ma.getmaskarray(a).copy()
    mask[after & ~valid] = False
    return np.ma.array(out, mask=mask), record
