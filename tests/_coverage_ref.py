"""Independent NumPy box-filter reference for LIMES primitive records.

The native test entry returns the exact materialized records consumed by the
GPU. Feeding those records here keeps the reference independent of Rust
coverage math without allowing Python and Rust ingest to describe different
geometry. The committed gate always uses 64x64 samples per pixel (4096 total).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

REFERENCE_SAMPLES_PER_AXIS = 64
LINE = 0
ARC = 1


def supersample_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    fill_rules: Sequence[str],
    samples_per_axis: int = REFERENCE_SAMPLES_PER_AXIS,
) -> np.ndarray:
    """Return `[layer, y, x]` coverage from directed line/y-monotone arcs.

    This is deliberately a sample-counting oracle, not a port of the analytic
    scan-cell formula. At the committed 64x64 setting each pixel is the mean of
    4096 sample classifications over the identical GPU primitive records.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if samples_per_axis <= 0:
        raise ValueError("samples_per_axis must be positive")
    if not fill_rules:
        raise ValueError("at least one fill rule is required")
    rules = tuple(_parse_fill_rule(rule) for rule in fill_rules)
    normalized = tuple(_normalize_record(record, len(rules)) for record in records)

    sample_x = (np.arange(width * samples_per_axis, dtype=np.float64) + 0.5) / samples_per_axis
    sample_y = (np.arange(height * samples_per_axis, dtype=np.float64) + 0.5) / samples_per_axis
    result = np.empty((len(rules), height, width), dtype=np.float64)

    for layer, rule in enumerate(rules):
        state = np.zeros((sample_y.size, sample_x.size), dtype=np.int32)
        for record in normalized:
            if record["layer"] != layer or record["winding"] == 0:
                continue
            min_y = record["bounds"][1]
            max_y = record["bounds"][3]
            rows = np.flatnonzero((sample_y >= min_y) & (sample_y < max_y))
            if rows.size == 0:
                continue
            crossings = _crossing_x(record, sample_y[rows])
            contribution = record["winding"] if rule == "nonzero" else 1
            state[rows] += contribution * (sample_x[None, :] >= crossings[:, None])

        inside = state != 0 if rule == "nonzero" else (state & 1) != 0
        result[layer] = inside.reshape(
            height,
            samples_per_axis,
            width,
            samples_per_axis,
        ).mean(axis=(1, 3), dtype=np.float64)
    return result


def composite_linear(
    coverage: np.ndarray,
    colors: Sequence[Sequence[float]],
) -> np.ndarray:
    """Fixed-order premultiplied-linear source-over mirror of the GPU resolve."""

    values = np.asarray(coverage, dtype=np.float64)
    rgba = np.asarray(colors, dtype=np.float64)
    if values.ndim != 3 or rgba.shape != (values.shape[0], 4):
        raise ValueError("coverage must be [layer,y,x] and colors [layer,4]")
    output = np.zeros((*values.shape[1:], 4), dtype=np.float64)
    for layer in range(values.shape[0]):
        alpha = np.clip(rgba[layer, 3] * values[layer], 0.0, 1.0)
        remaining = 1.0 - alpha
        output[..., :3] = (
            rgba[layer, :3] * alpha[..., None]
            + output[..., :3] * remaining[..., None]
        )
        output[..., 3] = alpha + output[..., 3] * remaining
    return output


def error_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return the gate's mean and maximum absolute coverage errors."""

    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.asarray(reference, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: actual={lhs.shape}, reference={rhs.shape}")
    difference = np.abs(lhs - rhs)
    return {
        "mean_abs_error": float(difference.mean()),
        "max_abs_error": float(difference.max(initial=0.0)),
    }


def deterministic_coverage_hash(coverage: np.ndarray) -> str:
    """Hash canonical little-endian f32 coverage bytes."""

    canonical = np.asarray(coverage, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def line_records_for_rings(
    rings: Iterable[Sequence[Sequence[float]]],
    *,
    layer: int = 0,
    first_stable_id: int = 0,
) -> list[dict[str, Any]]:
    """Materialize simple directed line rings for reference self-tests."""

    records: list[dict[str, Any]] = []
    stable_id = first_stable_id
    for source in rings:
        ring = [tuple(map(float, point)) for point in source]
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring.pop()
        if len(ring) < 3:
            raise ValueError("a ring requires at least three vertices")
        for start, end in zip(ring, ring[1:] + ring[:1], strict=True):
            winding = 1 if end[1] > start[1] else -1 if end[1] < start[1] else 0
            records.append(
                {
                    "kind": LINE,
                    "geometry": [start[0], start[1], end[0], end[1]],
                    "bounds": [
                        min(start[0], end[0]),
                        min(start[1], end[1]),
                        max(start[0], end[0]),
                        max(start[1], end[1]),
                    ],
                    "winding": winding,
                    "layer": layer,
                    "stable_id": stable_id,
                }
            )
            stable_id += 1
    return records


def _crossing_x(record: Mapping[str, Any], y: np.ndarray) -> np.ndarray:
    geometry = record["geometry"]
    if record["kind"] == LINE:
        x0, y0, x1, y1 = geometry
        return x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    center_x, center_y, radius, branch = geometry
    half_width = np.sqrt(np.maximum(radius * radius - (y - center_y) ** 2, 0.0))
    return center_x + branch * half_width


def _parse_fill_rule(value: str) -> str:
    if value not in {"nonzero", "evenodd"}:
        raise ValueError(f"unknown fill rule: {value!r}")
    return value


def _normalize_record(record: Mapping[str, Any], layer_count: int) -> dict[str, Any]:
    kind = int(record["kind"])
    geometry = tuple(float(value) for value in record["geometry"])
    bounds = tuple(float(value) for value in record["bounds"])
    layer = int(record["layer"])
    winding = int(record["winding"])
    stable_id = int(record["stable_id"])
    if kind not in {LINE, ARC}:
        raise ValueError(f"unknown primitive kind: {kind}")
    if len(geometry) != 4 or len(bounds) != 4:
        raise ValueError("geometry and bounds must contain four values")
    if not 0 <= layer < layer_count:
        raise ValueError(f"record layer {layer} is out of range")
    if winding not in {-1, 0, 1}:
        raise ValueError(f"invalid winding: {winding}")
    if not np.isfinite((*geometry, *bounds)).all():
        raise ValueError("record contains non-finite values")
    return {
        "kind": kind,
        "geometry": geometry,
        "bounds": bounds,
        "layer": layer,
        "winding": winding,
        "stable_id": stable_id,
    }
