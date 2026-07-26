from __future__ import annotations

from typing import Literal, overload
import numpy as np
from numpy.typing import NDArray

@overload
def viewshed(
    dem: NDArray[np.floating],
    observer: tuple[float, float] | tuple[float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    height_system: Literal["ellipsoidal", "orthometric_egm96"],
    observer_height: float = ...,
    target_height: float = ...,
    max_distance: float | None = ...,
    earth_model: Literal["flat", "sphere", "ellipsoid", "wgs84"] = ...,
    sphere_radius_m: float = ...,
    refraction_model: Literal["none", "bennett", "saemundsson", "effective_radius"] = ...,
    refraction_k: float = ...,
    pressure_mbar: float = ...,
    temperature_c: float = ...,
    return_diagnostics: Literal[False] = ...,
) -> NDArray[np.bool_]: ...

@overload
def viewshed(
    dem: NDArray[np.floating],
    observer: tuple[float, float] | tuple[float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    height_system: Literal["ellipsoidal", "orthometric_egm96"],
    observer_height: float = ...,
    target_height: float = ...,
    max_distance: float | None = ...,
    earth_model: Literal["flat", "sphere", "ellipsoid", "wgs84"] = ...,
    sphere_radius_m: float = ...,
    refraction_model: Literal["none", "bennett", "saemundsson", "effective_radius"] = ...,
    refraction_k: float = ...,
    pressure_mbar: float = ...,
    temperature_c: float = ...,
    return_diagnostics: Literal[True],
) -> dict[str, NDArray[np.bool_] | NDArray[np.float32]]: ...
