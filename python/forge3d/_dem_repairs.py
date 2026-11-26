"""Declared source-grid repairs. Every entry point returns the change record."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .dem_audit import _cell_runs, _surface
from .preparation import PreparationError


def _invalid(message):
    return PreparationError(
        "invalid_repair",
        message,
        "Check the named repair, numeric parameters and source-grid region.",
    )


def _json_value(value):
    """Detach the declaration, normalizing NumPy and Python values identically."""
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise _invalid("Repair field names must be strings.")
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return float(value)
    raise _invalid(
        "Supply finite numeric parameters and a confirmed source-grid region."
    )


def normalize_repair(step):
    step = _json_value(step)
    if not isinstance(step, dict) or not isinstance(step.get("region"), dict):
        raise _invalid("Supply a confirmed source-grid repair region.")
    if set(step["region"]) not in ({"cells"}, {"runs"}, {"bounds"}, {"rings"}):
        raise _invalid("A repair region requires cells, runs, bounds or rings.")
    op = step.get("operation")
    fields = {
        "fill_missing": {"method"},
        "remove_spikes": set(),
        "notch_filter": {"axis", "frequencies"},
        "dequantize": {"step"},
    }
    if op == "fill_missing":
        step.setdefault("method", "nearest")
    if (
        not isinstance(op, str)
        or op not in fields
        or set(step) != fields[op] | {"operation", "region"}
    ):
        raise _invalid("Unknown repair operation or incorrect parameter fields.")
    if op == "fill_missing" and step["method"] != "nearest":
        raise _invalid("fill_missing supports method='nearest'.")
    if op == "dequantize":
        if not isinstance(step["step"], (int, float)) or step["step"] <= 0:
            raise _invalid("dequantize requires a positive height step.")
        try:
            step["step"] = float(step["step"])
        except OverflowError as exc:
            raise _invalid(
                "dequantize requires a finite binary64 height step."
            ) from exc
    if op == "notch_filter":
        fs = step["frequencies"]
        if (
            step["axis"] not in ("row", "column")
            or not isinstance(fs, list)
            or not fs
            or not all(isinstance(f, (int, float)) and 0 < f <= 0.5 for f in fs)
            or len(fs) != len(set(fs))
        ):
            raise _invalid(
                "Supply an axis and distinct frequencies in (0, 0.5] cycles/cell."
            )
        step["frequencies"] = list(map(float, fs))
    return step


# Binary64 represents every consecutive integer through 2**53.
_EXACT_INTEGER = 2 ** (np.finfo(np.float64).nmant + 1)


def require_exact_integers(data, valid):
    if data.dtype.kind in "iu" and data.dtype.itemsize >= 8 and valid.any():
        heights = data[valid]
        if int(heights.min()) < -_EXACT_INTEGER or int(heights.max()) > _EXACT_INTEGER:
            raise _invalid(
                "Integer elevations exceed the consecutive integer range of float64 repair arithmetic."
            )


def _selection(shape, region):
    """Select a finding's cell, half-open bounds, or polygon with holes."""
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
    elif set(region) == {"runs"}:
        for r, c, end in region["runs"]:
            if not all(isinstance(v, int) for v in (r, c, end)) or not (
                0 <= r < h and 0 <= c < end <= w
            ):
                raise ValueError("repair run outside source grid")
            mask[r, c:end] = True
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
        raise ValueError("region requires cells, runs, bounds, or rings")
    return mask


def _notch(z, selected, frequencies, axis):
    """Remove the requested harmonic components on the selected valid samples."""
    frequencies = np.asarray(frequencies, dtype=float)
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
    try:
        step = normalize_repair(
            {"operation": operation, "region": region, **parameters}
        )
        region = step.pop("region")
        step.pop("operation")
        return _apply(dem, operation, region, step)
    except PreparationError:
        raise
    except (TypeError, ValueError, OverflowError, IndexError, KeyError) as exc:
        raise _invalid(str(exc)) from exc


def _apply(dem, operation, region, parameters):
    a = np.ma.asarray(dem)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("repairs require a two-dimensional real raster")
    original = a.data
    valid = ~np.ma.getmaskarray(a) & np.isfinite(original)
    require_exact_integers(original, valid)
    target = _selection(original.shape, region)
    out = original.copy()
    after = valid.copy()
    z = np.where(valid, original, np.nan).astype(np.float64)
    if operation == "fill_missing":
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
        if operation == "remove_spikes":
            candidate, _ = _surface(z)
        elif operation == "notch_filter":
            candidate = _notch(z, target, **parameters)
        elif operation == "dequantize":
            step = parameters["step"]
            if original.dtype.kind != "f":
                raise ValueError(
                    "dequantize requires a floating DEM and a positive height step"
                )
            surface, _ = _surface(z)
            candidate = z + np.clip(surface - z, -step / 2, step / 2)
        else:
            raise ValueError(f"unknown operation or parameters for {operation!r}")
        target &= np.isfinite(candidate)
        if original.dtype.kind in "iu":
            if np.any(np.abs(candidate[target]) > _EXACT_INTEGER):
                raise _invalid(
                    "Replacement exceeds the consecutive integer range of float64."
                )
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
    if original.dtype.kind in "iu":
        # Subtract before conversion to float. Both endpoints are within ±2**53.
        delta = abs(out[paired].astype(np.int64) - original[paired].astype(np.int64))
        maximum = int(delta.max()) if delta.size else None
        mean = None
        if delta.size:
            total = sum(map(int, delta))
            quotient, remainder = divmod(total, delta.size)
            mean = quotient if remainder == 0 else total / delta.size
    else:
        delta = abs(out[paired].astype(float) - original[paired].astype(float))
        maximum = float(delta.max()) if delta.size else None
        mean = float(delta.mean()) if delta.size else None
    # A missing observation has no physical before/after height difference.
    count = int(changed.sum())
    record = {
        "operation": operation,
        "parameters": {"region": region, **parameters},
        "cells_changed": count,
        "changed_runs": _cell_runs(changed),
        "measured_changes": int(delta.size),
        "newly_valid": int((after & ~valid).sum()),
        "maximum_change": maximum if count else 0.0,
        "mean_change": mean if count else 0.0,
    }
    if original.dtype.kind in "iu" and delta.size:
        record["total_change"] = total
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
