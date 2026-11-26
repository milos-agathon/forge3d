"""Inspect original DEM samples. Pixel coordinates are (column, row), zero based."""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path
import warnings

import numpy as np

from .diagnostics import Diagnostic

__all__ = [
    "audit",
    "AuditReport",
    "AuditParameterError",
    "fill_missing",
    "remove_spikes",
    "notch_filter",
    "dequantize",
]


class AuditParameterError(ValueError):
    """Invalid detector setting with a machine-readable correction."""

    def __init__(self, parameter, requirement):
        self.diagnostic = Diagnostic(
            "invalid_audit_parameter",
            "error",
            f"Invalid {parameter}.",
            requirement,
            details={"parameter": parameter},
        )
        super().__init__(f"{parameter}: {requirement}")


def _detector_value(name, value, *, count=False, fraction=False):
    if count:
        if isinstance(value, Integral) and not isinstance(value, bool) and value >= 2:
            return int(value)
        raise AuditParameterError(name, "Supply an integer of at least two.")
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            number = float("inf")
        if np.isfinite(number) and number > 0 and (not fraction or number <= 1):
            return number
    requirement = (
        "Supply a fraction in (0, 1]."
        if fraction
        else "Supply a finite positive multiple."
    )
    raise AuditParameterError(name, requirement)


def _median(values, axis=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(values, axis=axis)


def _surface(z):
    """Median opposing-pair interpolation reproduces a plane, even on a slope."""
    p = np.pad(z, 1, constant_values=np.nan)
    surface = np.empty_like(z)
    spread = np.empty_like(z)
    for row in range(z.shape[0]):
        above, center, below = p[row : row + 3]
        neighbors = np.stack(
            [
                above[:-2],
                above[1:-1],
                above[2:],
                center[:-2],
                center[2:],
                below[:-2],
                below[1:-1],
                below[2:],
            ]
        )
        surface[row] = _median((neighbors[:4] + neighbors[:3:-1]) / 2, axis=0)
        spread[row] = _median(np.abs(neighbors - surface[row]), axis=0)
    # A corner has no opposite pair. Three inward samples determine a plane.
    if min(z.shape) >= 2:
        for r, dr in ((0, 1), (-1, -1)):
            for c, dc in ((0, 1), (-1, -1)):
                inward = np.array([z[r + dr, c], z[r, c + dc], z[r + dr, c + dc]])
                surface[r, c] = inward[0] + inward[1] - inward[2]
                spread[r, c] = np.median(abs(inward - surface[r, c]))
    return surface, spread


def _polygons(mask, labels=None):
    from rasterio.features import shapes

    if not mask.any():
        return
    raster = mask.astype("uint8") if labels is None else labels
    for geom, label in shapes(raster, mask=mask, connectivity=4):
        rings = [[[int(x), int(y)] for x, y in ring] for ring in geom["coordinates"]]
        yield rings, int(label)


def _area(rings):
    def twice(ring):
        return abs(sum(x * v - y * u for (x, y), (u, v) in zip(ring, ring[1:])))

    return (twice(rings[0]) - sum(map(twice, rings[1:]))) // 2


def _regions(z, cell_area, flat_cells):
    h, w = z.shape
    for rings, _ in _polygons(~np.isfinite(z)):
        count = _area(rings)
        yield {
            "class": "void",
            "rings": rings,
            "cells": count,
            "area": count * cell_area,
            "enclosed": all(0 < x < w and 0 < y < h for x, y in rings[0]),
        }

    # Mark connected equal-valued plateaus. Encode their actual value before
    # polygonizing so adjacent plateaus at different heights stay distinct.
    same_x = z[:, 1:] == z[:, :-1]
    same_y = z[1:] == z[:-1]
    flat = np.zeros(z.shape, bool)
    flat[:, 1:] |= same_x
    flat[:, :-1] |= same_x
    flat[1:] |= same_y
    flat[:-1] |= same_y
    if flat.any():
        levels, inverse = np.unique(z[flat], return_inverse=True)
        labels = np.zeros(z.shape, np.int32)
        labels[flat] = inverse.ravel()
        for rings, code in _polygons(flat, labels):
            count = _area(rings)
            if count >= flat_cells:
                yield {
                    "class": "flat_fill",
                    "rings": rings,
                    "cells": count,
                    "area": count * cell_area,
                    "elevation": float(levels[code]),
                }


def _stripes(z, axis, stripe_fraction):
    a = z if axis == "row" else z.T
    n = a.shape[0]
    correction = np.zeros(n)
    peaks = []
    if n < 4:
        return peaks, correction
    profile = _median(np.diff(a, axis=0), axis=1)
    trend = _median(profile)
    if not np.isfinite(trend):
        return peaks, correction
    signal = np.append(np.nan_to_num(profile - trend, nan=0.0), 0.0)
    spectrum = np.fft.rfft(signal)
    power = abs(spectrum) ** 2
    power[0] = 0
    total = power.sum()
    if total > 0:
        heights = np.zeros_like(spectrum)
        for k in np.flatnonzero(power >= stripe_fraction * total):
            if k >= 2:
                peaks.append(
                    {
                        "class": "striping",
                        "axis": axis,
                        "period_cells": n / int(k),
                        "frequency": int(k) / n,
                        "power_fraction": float(power[k] / total),
                        "analysis_bounds": [0, 0, z.shape[1], z.shape[0]],
                        "repair_region": None,
                    }
                )
                heights[k] = spectrum[k] / np.expm1(2j * np.pi * k / n)
        correction = np.fft.irfft(heights, n=n)
    return peaks, correction


def _seams(z, axis, seam_multiple, seam_length):
    a = z if axis == "row" else z.T
    if a.shape[0] < 4:
        return
    d = np.diff(a, axis=0)
    # A seam has a coherent excess cross-line jump relative to the slopes
    # directly on either side. Retain contiguous support, including partial seams.
    excess = d[1:-1] - (d[:-2] + d[2:]) / 2
    scale = _median(abs(excess))
    if not np.isfinite(scale):
        return
    threshold = max(seam_multiple * scale, _roundoff(z))
    for i, row in enumerate(excess):
        for sign in (1, -1):
            hit = np.flatnonzero(np.diff(np.r_[False, sign * row > threshold, False]))
            for start, end in zip(hit[::2], hit[1::2]):
                if end - start < seam_length:
                    continue
                # A single step creates weaker, opposite side-lobes. Only
                # the locally strongest discontinuity owns the line.
                line = i + 2
                jump = np.median(row[start:end])
                adjacent = [
                    abs(_median(excess[j, start:end]))
                    for j in (i - 1, i + 1)
                    if 0 <= j < len(excess)
                ]
                if any(abs(jump) <= neighbor for neighbor in adjacent):
                    continue
                bounds = (
                    [start, line, end, line + 1]
                    if axis == "row"
                    else [line, start, line + 1, end]
                )
                yield {
                    "class": "seam",
                    "axis": axis,
                    "bounds": list(map(int, bounds)),
                    "step": float(jump),
                    "length_cells": int(end - start),
                }


def _roundoff(z):
    finite = z[np.isfinite(z)]
    return (
        float(np.spacing(max(float(np.max(abs(finite))), 1.0))) * 8
        if finite.size
        else 0.0
    )


def _quantization(z, fraction, sample_resolution):
    values = z[np.isfinite(z)]
    levels, counts = np.unique(values, return_counts=True)
    repeated = levels[counts > 1]
    if len(levels) < 3:
        return
    occupied = repeated if len(repeated) >= 3 else levels
    # The median adjacent occupied-bin interval is robust to a minority
    # of extra off-lattice levels and to floating point encoding of the step.
    gaps = np.diff(occupied)
    step = float(np.median(gaps))
    # Each gap has two rounded endpoints. Average equivalent gaps to avoid
    # accumulating the float32 error of one interval across the height range.
    tolerance = max(_roundoff(z), 2 * sample_resolution)
    equal_gaps = gaps[abs(gaps - step) <= tolerance]
    if equal_gaps.size:
        step = float(equal_gaps.mean())
    anchor = occupied[len(occupied) // 2]
    # Long baselines estimate the step more precisely than one rounded gap.
    indices = np.rint((occupied - anchor) / step)
    off_origin = indices != 0
    if off_origin.any():
        step = float(np.median((occupied[off_origin] - anchor) / indices[off_origin]))
    errors = abs((levels - anchor) - np.rint((levels - anchor) / step) * step)
    if step <= 2 * tolerance:
        return
    aligned = float(counts[errors <= tolerance].sum() / values.size)
    if aligned >= fraction:
        return {
            "class": "quantization",
            "analysis_bounds": [0, 0, z.shape[1], z.shape[0]],
            "repair_region": None,
            "step": step,
            "origin": float(anchor),
            "aligned_fraction": aligned,
            "tolerance": float(tolerance),
        }


@dataclass
class AuditReport:
    """JSON-compatible evidence plus the annotated inspection sheet as a PIL image."""

    findings: list
    metadata: dict
    hillshade: object

    def save(self, stem):
        """Write <stem>.json and <stem>.png."""
        path = Path(stem)
        path.with_suffix(".json").write_text(
            json.dumps({**self.metadata, "findings": self.findings}, allow_nan=False),
            encoding="utf-8",
        )
        self.hillshade.save(path.with_suffix(".png"))


def audit(
    dem,
    transform=None,
    *,
    valid_mask=None,
    nodata=None,
    band=1,
    spike_multiple=5.0,
    seam_multiple=12.0,
    seam_length=9,
    stripe_fraction=0.84,
    quantized_fraction=0.9,
    flat_cells=16,
):
    """Inspect a 2D array, masked array, or original GeoTIFF (one-based band).

    Supply an Affine or six GDAL-order coefficients. Bounds are half-open,
    polygon rings follow pixel edges. Area uses squared CRS units.
    """
    from affine import Affine
    from ._dem_audit_sheet import sheet

    if not isinstance(band, Integral) or isinstance(band, bool) or band < 1:
        raise AuditParameterError("band", "Supply a positive one-based band index.")
    band = int(band)
    source, crs, unit = None, None, None
    if isinstance(dem, (str, Path)):
        import rasterio

        source = str(dem)
        with rasterio.open(dem) as ds:
            if transform is not None:
                raise AuditParameterError(
                    "transform", "Use the transform supplied by the GeoTIFF."
                )
            if band > ds.count:
                raise AuditParameterError(
                    "band", f"Select a source band between 1 and {ds.count}."
                )
            transform, crs, unit = (
                ds.transform,
                str(ds.crs) if ds.crs else None,
                ds.units[band - 1],
            )
            dem = ds.read(band, masked=True)
            if ds.nodatavals[band - 1] is not None:
                dem = np.ma.masked_where(
                    dem.data == ds.nodatavals[band - 1], dem, copy=False
                )
            scale, offset = ds.scales[band - 1], ds.offsets[band - 1]
    else:
        if band != 1:
            raise AuditParameterError("band", "Use band 1 for a two-dimensional array.")
        scale, offset = 1.0, 0.0
    a = np.ma.asarray(dem)
    if a.ndim != 2 or not a.size or a.dtype.kind not in "iuf":
        raise AuditParameterError("dem", "Supply a nonempty real-valued 2D raster.")
    valid = np.isfinite(a.data) & ~np.ma.getmaskarray(a)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != a.shape or mask.dtype != bool:
            raise AuditParameterError(
                "valid_mask", "Supply a boolean mask matching the raster shape."
            )
        valid &= mask
    if nodata is not None:
        valid &= a.data != nodata
    if transform is None:
        transform = Affine.identity()
    elif not isinstance(transform, Affine):
        try:
            transform = Affine.from_gdal(*(float(v) for v in transform))
        except (TypeError, ValueError, OverflowError) as exc:
            raise AuditParameterError(
                "transform", "Supply six GDAL-order coefficients."
            ) from exc
    if not np.all(np.isfinite(transform)) or transform.determinant == 0:
        raise AuditParameterError("transform", "Supply a finite invertible transform.")
    spike_multiple = _detector_value("spike_multiple", spike_multiple)
    seam_multiple = _detector_value("seam_multiple", seam_multiple)
    seam_length = _detector_value("seam_length", seam_length, count=True)
    stripe_fraction = _detector_value("stripe_fraction", stripe_fraction, fraction=True)
    quantized_fraction = _detector_value(
        "quantized_fraction", quantized_fraction, fraction=True
    )
    flat_cells = _detector_value("flat_cells", flat_cells, count=True)
    z = np.where(valid, a.data, np.nan).astype(np.float64) * scale + offset
    findings = list(_regions(z, float(abs(transform.determinant)), flat_cells))
    if valid.any():
        epsilon = _roundoff(z)
        # Discount coherent hatch when estimating isolated relief and cross-line steps.
        corrected = z.copy()
        for axis in ("row", "column"):
            peaks, harmonic = _stripes(z, axis, stripe_fraction)
            findings.extend(peaks)
            corrected -= harmonic[:, None] if axis == "row" else harmonic
        surface, spread = _surface(corrected)
        # Use terrain-wide typical spread as the floor on nearly flat neighbors.
        floor = max(epsilon, float(np.nan_to_num(_median(spread))))
        residual = corrected - surface
        rows, cols = np.nonzero(
            abs(residual) > spike_multiple * np.maximum(spread, floor)
        )
        for r, c in zip(rows, cols):
            delta = float(residual[r, c])
            findings.append(
                {
                    "class": "spike" if delta > 0 else "pit",
                    "cell": [int(c), int(r)],
                    "xy": list(transform * (float(c) + 0.5, float(r) + 0.5)),
                    "height": delta,
                }
            )
        for axis in ("row", "column"):
            findings.extend(_seams(corrected, axis, seam_multiple, seam_length))
        sample_resolution = 0.0
        if a.dtype.kind == "f":
            sample_resolution = abs(scale) * float(
                np.spacing(np.max(np.abs(a.data[valid])))
            )
        quant = _quantization(z, quantized_fraction, sample_resolution)
        if quant:
            findings.append(quant)
    metadata = {
        "source": source,
        "band": band,
        "shape": list(z.shape),
        "crs": crs,
        "height_unit": unit,
        "geotransform": list(map(float, transform.to_gdal())),
        "status": "candidates for inspection",
        "valid_cells": int(valid.sum()),
        "parameters": dict(
            spike_multiple=spike_multiple,
            seam_multiple=seam_multiple,
            seam_length=seam_length,
            stripe_fraction=stripe_fraction,
            quantized_fraction=quantized_fraction,
            flat_cells=flat_cells,
        ),
    }
    return AuditReport(findings, metadata, sheet(z, transform, findings, source))


def fill_missing(dem, *, region, method="nearest"):
    """Fill selected invalid cells from their nearest valid donor."""
    from ._dem_repairs import apply_repair

    return apply_repair(dem, "fill_missing", region=region, method=method)


def remove_spikes(dem, *, region):
    """Replace selected cells with a robust opposing-neighbor interpolation."""
    from ._dem_repairs import apply_repair

    return apply_repair(dem, "remove_spikes", region=region)


def notch_filter(dem, *, region, frequencies, axis):
    """Remove the specified row or column harmonics within a declared region."""
    from ._dem_repairs import apply_repair

    return apply_repair(
        dem, "notch_filter", region=region, frequencies=frequencies, axis=axis
    )


def dequantize(dem, *, region, step):
    """Interpolate selected floating heights within their original rounding bins."""
    from ._dem_repairs import apply_repair

    return apply_repair(dem, "dequantize", region=region, step=step)
