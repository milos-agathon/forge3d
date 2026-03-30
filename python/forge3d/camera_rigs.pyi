from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .animation import CameraAnimation
from .terrain_scatter import TerrainScatterSource


class TerrainClearance:
    minimum_height: float
    max_refine_passes: int
    def __init__(self, minimum_height: float = ..., max_refine_passes: int = ...) -> None: ...


class TerrainOrbitRig:
    target_xz: Tuple[float, float]
    duration: float
    radius: float
    phi_start_deg: float
    phi_end_deg: float
    theta_start_deg: float
    theta_end_deg: Optional[float]
    radius_end: Optional[float]
    fov_start_deg: float
    fov_end_deg: Optional[float]
    target_height_offset: float
    clearance: TerrainClearance
    def __init__(
        self,
        target_xz: Sequence[float],
        duration: float,
        radius: float,
        phi_start_deg: float,
        phi_end_deg: float,
        theta_start_deg: float = ...,
        theta_end_deg: Optional[float] = ...,
        radius_end: Optional[float] = ...,
        fov_start_deg: float = ...,
        fov_end_deg: Optional[float] = ...,
        target_height_offset: float = ...,
        clearance: TerrainClearance = ...,
    ) -> None: ...
    def bake(self, source: TerrainScatterSource, *, samples_per_second: int = ...) -> CameraAnimation: ...


class TerrainRailRig:
    path_xz: Tuple[Tuple[float, float], ...]
    duration: float
    camera_height_offset: float
    look_ahead_distance: float
    lateral_offset: float
    target_height_offset: float
    fov_deg: float
    clearance: TerrainClearance
    def __init__(
        self,
        path_xz: Sequence[Sequence[float]],
        duration: float,
        camera_height_offset: float,
        look_ahead_distance: float,
        lateral_offset: float = ...,
        target_height_offset: float = ...,
        fov_deg: float = ...,
        clearance: TerrainClearance = ...,
    ) -> None: ...
    def bake(self, source: TerrainScatterSource, *, samples_per_second: int = ...) -> CameraAnimation: ...


class TerrainTargetFollowRig:
    target_path_xz: Tuple[Tuple[float, float], ...]
    duration: float
    radius: float
    theta_deg: float
    heading_offset_deg: float
    target_height_offset: float
    fov_deg: float
    clearance: TerrainClearance
    def __init__(
        self,
        target_path_xz: Sequence[Sequence[float]],
        duration: float,
        radius: float,
        theta_deg: float = ...,
        heading_offset_deg: float = ...,
        target_height_offset: float = ...,
        fov_deg: float = ...,
        clearance: TerrainClearance = ...,
    ) -> None: ...
    def bake(self, source: TerrainScatterSource, *, samples_per_second: int = ...) -> CameraAnimation: ...


__all__ = [
    "TerrainClearance",
    "TerrainOrbitRig",
    "TerrainRailRig",
    "TerrainTargetFollowRig",
]
