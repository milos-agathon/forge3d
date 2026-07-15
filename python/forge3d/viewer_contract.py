"""Typed, synchronous validation for viewer world-coordinate IPC payloads."""

from __future__ import annotations

import math
from typing import Sequence, TypeAlias

WorldPosition: TypeAlias = tuple[float, float, float]
VectorOverlayVertex: TypeAlias = tuple[float, float, float, float, float, float, float, int]
NormalizedExtent: TypeAlias = tuple[float, float, float, float]

_U32_MAX = 4_294_967_295


def world_position(value: Sequence[float], *, name: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 finite f64 coordinates")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain exactly 3 finite f64 coordinates")
    return result


def normalized_extent(value: Sequence[float], *, name: str = "extent") -> list[float]:
    if len(value) != 4:
        raise ValueError(f"{name} must be normalized UV (u0, v0, u1, v1)")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) and 0.0 <= component <= 1.0 for component in result):
        raise ValueError(f"{name} must be normalized UV values in [0, 1]")
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"{name} normalized UV ranges must have positive area")
    return result


def vector_overlay_vertices(vertices: Sequence[Sequence[float]]) -> list[list[float | int]]:
    validated: list[list[float | int]] = []
    for row_index, row in enumerate(vertices):
        if len(row) != 8:
            raise ValueError(
                f"vector vertex {row_index} must contain exactly 8 lanes: XYZ, RGBA, feature ID"
            )
        xyz = [float(component) for component in row[:3]]
        rgba = [float(component) for component in row[3:7]]
        if not all(math.isfinite(component) for component in xyz):
            raise ValueError(f"vector vertex {row_index} XYZ must be finite f64")
        if not all(math.isfinite(component) and 0.0 <= component <= 1.0 for component in rgba):
            raise ValueError(f"vector vertex {row_index} RGBA must be finite values in [0, 1]")
        raw_id = row[7]
        if isinstance(raw_id, bool):
            raise ValueError(f"vector vertex {row_index} feature ID must be an integer u32")
        try:
            feature_id_float = float(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"vector vertex {row_index} feature ID must be an integer u32"
            ) from exc
        if (
            not math.isfinite(feature_id_float)
            or not feature_id_float.is_integer()
            or not 0.0 <= feature_id_float <= _U32_MAX
        ):
            raise ValueError(f"vector vertex {row_index} feature ID must be an integer u32")
        validated.append([*xyz, *rgba, int(feature_id_float)])
    return validated
