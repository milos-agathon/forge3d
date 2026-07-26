"""Curvature-aware terrain analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._native import get_native_module, native_import_error


def viewshed(
    dem: np.ndarray,
    observer: tuple[float, float] | tuple[float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    height_system: str,
    observer_height: float = 1.7,
    target_height: float = 0.0,
    max_distance: float | None = None,
    earth_model: str = "ellipsoid",
    sphere_radius_m: float = 6_371_008.8,
    refraction_model: str = "bennett",
    refraction_k: float = 0.13,
    pressure_mbar: float = 1013.25,
    temperature_c: float = 15.0,
    return_diagnostics: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Compute a GPU visibility raster for an EPSG:4326 north-up DEM."""
    native = get_native_module()
    if native is None:
        cause = native_import_error()
        detail = f": {cause}" if cause is not None else ""
        raise RuntimeError(f"forge3d native extension is unavailable{detail}")
    if not hasattr(native, "terrain_viewshed"):
        raise RuntimeError(
            "forge3d native extension does not provide terrain_viewshed; rebuild the extension"
        )
    if len(observer) == 3:
        observer_height = float(observer[2])
    elif len(observer) != 2:
        raise ValueError("observer must be (lat, lon) or (lat, lon, h_agl)")
    result: dict[str, Any] = native.terrain_viewshed(
        np.ascontiguousarray(dem, dtype=np.float32),
        (float(observer[0]), float(observer[1])),
        bounds,
        height_system,
        observer_height=observer_height,
        target_height=target_height,
        max_distance=max_distance,
        earth_model=earth_model,
        sphere_radius_m=sphere_radius_m,
        refraction_model=refraction_model,
        refraction_k=refraction_k,
        pressure_mbar=pressure_mbar,
        temperature_c=temperature_c,
    )
    arrays = {
        key: np.asarray(value, dtype=np.bool_ if key == "visibility" else np.float32)
        for key, value in result.items()
    }
    return arrays if return_diagnostics else arrays["visibility"]


__all__ = ["viewshed"]
