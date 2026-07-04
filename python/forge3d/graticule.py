"""Latitude/longitude graticule generation for MapScene furniture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class GraticuleSpec:
    """Configuration for generated latitude/longitude graticules."""

    bounds: Sequence[float] | None = None
    interval_deg: float = 1.0
    target_crs: str = "EPSG:4326"
    include_labels: bool = True
    precision: int = 3
    line_steps: int = 32

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "interval_deg": float(self.interval_deg),
            "target_crs": str(self.target_crs),
            "include_labels": bool(self.include_labels),
            "precision": int(self.precision),
            "line_steps": int(self.line_steps),
        }


def _bounds_tuple(bounds: Sequence[float]) -> Bounds:
    if len(bounds) != 4:
        raise ValueError("bounds must be (west, south, east, north)")
    west, south, east, north = (float(value) for value in bounds)
    if not west < east:
        raise ValueError("bounds west must be less than east")
    if not south < north:
        raise ValueError("bounds south must be less than north")
    if south < -90.0 or north > 90.0:
        raise ValueError("graticule latitude bounds must be within [-90, 90]")
    return west, south, east, north


def _grid_values(start: float, end: float, interval: float) -> list[float]:
    first = math.ceil((start - 1.0e-9) / interval) * interval
    values: list[float] = []
    value = first
    while value <= end + 1.0e-9:
        values.append(0.0 if abs(value) < 1.0e-9 else value)
        value += interval
    return values


def _linspace(start: float, end: float, steps: int) -> list[float]:
    count = max(2, int(steps))
    return [start + (end - start) * (index / (count - 1)) for index in range(count)]


def _format_coord(value: float, axis: str, precision: int) -> str:
    suffix = ""
    if axis == "lon":
        suffix = "E" if value > 0.0 else "W" if value < 0.0 else ""
    else:
        suffix = "N" if value > 0.0 else "S" if value < 0.0 else ""
    rounded = round(abs(float(value)), max(0, int(precision)))
    if rounded.is_integer():
        text = str(int(rounded))
    else:
        text = f"{rounded:.{max(0, int(precision))}f}".rstrip("0").rstrip(".")
    return f"{text} deg{suffix}"


def _transform_lines(lines: list[list[tuple[float, float]]], target_crs: str) -> list[list[tuple[float, float]]]:
    if str(target_crs).upper() in {"EPSG:4326", "WGS84", "WGS 84"}:
        return lines
    import numpy as np

    from .crs import transform_coords

    flat = [point for line in lines for point in line]
    transformed = transform_coords(np.asarray(flat, dtype=np.float64), "EPSG:4326", str(target_crs))
    out: list[list[tuple[float, float]]] = []
    cursor = 0
    for line in lines:
        count = len(line)
        out.append([(float(x), float(y)) for x, y in transformed[cursor : cursor + count]])
        cursor += count
    return out


def _transform_points(points: list[tuple[float, float]], target_crs: str) -> list[tuple[float, float]]:
    if not points or str(target_crs).upper() in {"EPSG:4326", "WGS84", "WGS 84"}:
        return points
    import numpy as np

    from .crs import transform_coords

    transformed = transform_coords(np.asarray(points, dtype=np.float64), "EPSG:4326", str(target_crs))
    return [(float(x), float(y)) for x, y in transformed]


def generate_graticule(
    bounds: Sequence[float] | GraticuleSpec,
    *,
    interval_deg: float | None = None,
    target_crs: str | None = None,
    include_labels: bool | None = None,
    precision: int | None = None,
    line_steps: int | None = None,
) -> dict[str, Any]:
    """Generate a GeoJSON-like graticule FeatureCollection.

    Input bounds are always geographic WGS84 coordinates in
    ``(west, south, east, north)`` order. Use ``target_crs`` to explicitly
    transform output coordinates.
    """
    if isinstance(bounds, GraticuleSpec):
        spec = bounds
        if spec.bounds is None:
            raise ValueError("GraticuleSpec.bounds is required")
    else:
        spec = GraticuleSpec(bounds=bounds)
    interval = float(interval_deg if interval_deg is not None else spec.interval_deg)
    if interval <= 0.0:
        raise ValueError("interval_deg must be positive")
    target = str(target_crs if target_crs is not None else spec.target_crs)
    labels_enabled = bool(include_labels if include_labels is not None else spec.include_labels)
    label_precision = int(precision if precision is not None else spec.precision)
    steps = int(line_steps if line_steps is not None else spec.line_steps)
    west, south, east, north = _bounds_tuple(spec.bounds or ())

    raw_lines: list[list[tuple[float, float]]] = []
    descriptors: list[tuple[str, float]] = []
    for lon in _grid_values(west, east, interval):
        raw_lines.append([(lon, lat) for lat in _linspace(south, north, steps)])
        descriptors.append(("meridian", lon))
    for lat in _grid_values(south, north, interval):
        raw_lines.append([(lon, lat) for lon in _linspace(west, east, steps)])
        descriptors.append(("parallel", lat))

    transformed_lines = _transform_lines(raw_lines, target)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in line]},
            "properties": {"kind": kind, "value": value},
        }
        for (kind, value), line in zip(descriptors, transformed_lines)
    ]

    labels: list[Mapping[str, Any]] = []
    if labels_enabled:
        raw_points: list[tuple[float, float]] = []
        label_payloads: list[dict[str, Any]] = []
        for kind, value in descriptors:
            if kind == "meridian":
                raw_points.append((value, south))
                label_payloads.append({"kind": kind, "value": value, "text": _format_coord(value, "lon", label_precision)})
            else:
                raw_points.append((west, value))
                label_payloads.append({"kind": kind, "value": value, "text": _format_coord(value, "lat", label_precision)})
        transformed_points = _transform_points(raw_points, target)
        labels = [
            {**payload, "coordinate": [point[0], point[1]]}
            for payload, point in zip(label_payloads, transformed_points)
        ]

    return {
        "type": "FeatureCollection",
        "source_crs": "EPSG:4326",
        "target_crs": target,
        "bounds": [west, south, east, north],
        "interval_deg": interval,
        "features": features,
        "labels": labels,
    }


__all__ = ["GraticuleSpec", "generate_graticule"]
