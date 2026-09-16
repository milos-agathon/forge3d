# python/forge3d/inverse.pyi
# DIFFERENTIA: typed surface for the differentiable inverse path tracer.
# RELEVANT FILES: python/forge3d/inverse.py, src/py_functions/inverse.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

class InverseSolveUnavailable(RuntimeError): ...

@dataclass
class RecoveredScene:
    albedo: np.ndarray
    sun_dir: np.ndarray
    sun_intensity: float
    turbidity: float
    loss_history: List[float]
    rgba: np.ndarray
    iterations_run: int
    peak_host_visible_bytes: int
    albedo_delta_e2000_median: Optional[float] = ...

    @property
    def sun_angles_deg(self) -> Tuple[float, float]: ...

def recover_scene(
    target: np.ndarray,
    dem: np.ndarray,
    *,
    cam: Optional[Dict[str, Any]] = ...,
    width: Optional[int] = ...,
    height: Optional[int] = ...,
    spacing: Tuple[float, float] = ...,
    exaggeration: float = ...,
    init_albedo: Optional[Any] = ...,
    sun_azimuth_deg: float = ...,
    sun_elevation_deg: float = ...,
    sun_intensity: float = ...,
    turbidity: float = ...,
    sun_color: Tuple[float, float, float] = ...,
    env_map: Optional[np.ndarray] = ...,
    env_intensity: float = ...,
    iters: int = ...,
    spp: int = ...,
    frames: int = ...,
    tile_size: int = ...,
    seed: int = ...,
    lr_albedo: float = ...,
    lr_sun: float = ...,
    lr_turbidity: float = ...,
    early_stop_tol: float = ...,
    early_stop_patience: int = ...,
    spatial_reuse: bool = ...,
    edge_term: bool = ...,
    score_correction: bool = ...,
    reference_albedo: Optional[Any] = ...,
) -> RecoveredScene: ...

def render_primal(
    dem: np.ndarray,
    albedo: Any,
    *,
    cam: Optional[Dict[str, Any]] = ...,
    width: int = ...,
    height: int = ...,
    spacing: Tuple[float, float] = ...,
    exaggeration: float = ...,
    sun_azimuth_deg: float = ...,
    sun_elevation_deg: float = ...,
    sun_intensity: float = ...,
    turbidity: float = ...,
    sun_color: Tuple[float, float, float] = ...,
    env_map: Optional[np.ndarray] = ...,
    env_intensity: float = ...,
    spp: int = ...,
    frames: int = ...,
    seed: int = ...,
) -> np.ndarray: ...
