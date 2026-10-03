# python/forge3d/splat.pyi
# SPLAT-FUSED: typed surface for Gaussian splats and the fused
# splat + LiDAR + terrain ReSTIR render.
# RELEVANT FILES: python/forge3d/splat.py, src/py_functions/splat.rs

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, overload

import numpy as np

HIT_MISS: int
HIT_TERRAIN: int
HIT_SPLAT: int
HIT_LIDAR: int

PathLike = Union[str, "os.PathLike[str]"]

class SplatFusionUnavailable(RuntimeError): ...

def splat_fusion_available() -> bool: ...

class GaussianSplatCloud:
    @staticmethod
    def from_arrays(
        positions: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacities: np.ndarray,
        sh0: np.ndarray,
        sh_rest: Optional[np.ndarray] = ...,
    ) -> "GaussianSplatCloud": ...
    @property
    def count(self) -> int: ...
    @property
    def sh_degree(self) -> int: ...
    @property
    def byte_size(self) -> int: ...
    @property
    def bounds(self) -> Optional[Tuple[List[float], List[float]]]: ...
    @property
    def positions(self) -> np.ndarray: ...
    @property
    def scales(self) -> np.ndarray: ...
    @property
    def rotations(self) -> np.ndarray: ...
    @property
    def opacities(self) -> np.ndarray: ...
    @property
    def sh0(self) -> np.ndarray: ...
    @property
    def inverse_covariance(self) -> np.ndarray: ...
    def color(self, index: int, direction: Sequence[float]) -> List[float]: ...
    def save(self, path: str) -> None: ...
    def write_page_store(self, path: str, page_capacity: int = ...) -> Dict[str, int]: ...
    def __len__(self) -> int: ...

def load_gaussian_splats(path: PathLike) -> GaussianSplatCloud: ...
def build_splat_page_store(
    ply_path: PathLike,
    out_path: PathLike,
    *,
    page_capacity: int = ...,
    memory_budget_bytes: int = ...,
) -> Dict[str, int]: ...
def write_synthetic_point_field(
    path: PathLike,
    *,
    page_capacity: int = ...,
    template_pages: int = ...,
    grid: Tuple[int, int] = ...,
    cell_size: float = ...,
    origin: Tuple[float, float, float] = ...,
    thickness: float = ...,
    hole: Tuple[float, float, float, float] = ...,
    seed: int = ...,
) -> Dict[str, int]: ...
def ray_gaussian(
    origin: Sequence[float],
    direction: Sequence[float],
    center: Sequence[float],
    scale: Sequence[float],
    rotation: Sequence[float] = ...,
    opacity: float = ...,
    *,
    tmin: float = ...,
    tmax: float = ...,
    kappa: float = ...,
) -> Dict[str, float]: ...

@dataclass(frozen=True)
class FusedCamera:
    origin: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    up: Tuple[float, float, float] = ...
    fov_y_deg: float = ...

@dataclass(frozen=True)
class FusedTerrain:
    heights: np.ndarray
    spacing: Tuple[float, float] = ...
    exaggeration: float = ...
    albedo: Tuple[float, float, float] = ...

@dataclass(frozen=True)
class FusedPointCloud:
    path: PathLike
    origin: Tuple[float, float, float] = ...
    z_up: bool = ...

@dataclass
class FusedRenderResult:
    rgba: np.ndarray
    radiance: np.ndarray
    albedo: np.ndarray
    normal: np.ndarray
    position: np.ndarray
    direct: np.ndarray
    depth: np.ndarray
    transmittance: np.ndarray
    hit_kind: np.ndarray
    sun_cosine: np.ndarray
    reservoir_visibility: np.ndarray
    stats: Dict[str, Any] = ...

    @property
    def shadow_mask(self) -> np.ndarray: ...

SplatsInput = Union[GaussianSplatCloud, PathLike, Sequence[Union[GaussianSplatCloud, PathLike]], None]
PointCloudInput = Union[
    PathLike,
    FusedPointCloud,
    Mapping[str, Any],
    Sequence[Union[PathLike, FusedPointCloud, Mapping[str, Any]]],
    None,
]
TerrainInput = Union[np.ndarray, FusedTerrain, Mapping[str, Any], None]
CameraInput = Union[FusedCamera, Mapping[str, Any]]

@overload
def render_fused(
    *,
    splats: SplatsInput = ...,
    pointcloud: PointCloudInput = ...,
    terrain: TerrainInput = ...,
    camera: CameraInput,
    samples: int = ...,
    samples_per_frame: Optional[int] = ...,
    width: int = ...,
    height: int = ...,
    seed: int = ...,
    exposure: float = ...,
    sun_azimuth_deg: float = ...,
    sun_elevation_deg: float = ...,
    sun_intensity: float = ...,
    sun_color: Tuple[float, float, float] = ...,
    sky_turbidity: float = ...,
    sky_ground_albedo: float = ...,
    sky_intensity: float = ...,
    kappa: float = ...,
    lidar_radius: float = ...,
    lidar_opacity: float = ...,
    ibl_occlusion_distance: float = ...,
    sun_angular_radius_deg: float = ...,
    brdf: str = ...,
    roughness: float = ...,
    metallic: float = ...,
    fog_density: float = ...,
    fog_height_falloff: float = ...,
    page_capacity: int = ...,
    splat_page_size: Optional[int] = ...,
    splat_slots: int = ...,
    point_slots: int = ...,
    policy: Literal["exact", "progressive"] = ...,
    return_aovs: Literal[False] = ...,
    certificate: Union[bool, PathLike] = ...,
    cache: Optional[str] = ...,
) -> np.ndarray: ...
@overload
def render_fused(
    *,
    splats: SplatsInput = ...,
    pointcloud: PointCloudInput = ...,
    terrain: TerrainInput = ...,
    camera: CameraInput,
    samples: int = ...,
    samples_per_frame: Optional[int] = ...,
    width: int = ...,
    height: int = ...,
    seed: int = ...,
    exposure: float = ...,
    sun_azimuth_deg: float = ...,
    sun_elevation_deg: float = ...,
    sun_intensity: float = ...,
    sun_color: Tuple[float, float, float] = ...,
    sky_turbidity: float = ...,
    sky_ground_albedo: float = ...,
    sky_intensity: float = ...,
    kappa: float = ...,
    lidar_radius: float = ...,
    lidar_opacity: float = ...,
    ibl_occlusion_distance: float = ...,
    sun_angular_radius_deg: float = ...,
    brdf: str = ...,
    roughness: float = ...,
    metallic: float = ...,
    fog_density: float = ...,
    fog_height_falloff: float = ...,
    page_capacity: int = ...,
    splat_page_size: Optional[int] = ...,
    splat_slots: int = ...,
    point_slots: int = ...,
    policy: Literal["exact", "progressive"] = ...,
    return_aovs: Literal[True],
    certificate: Union[bool, PathLike] = ...,
    cache: Optional[str] = ...,
) -> FusedRenderResult: ...
def render_fused_reference(
    *,
    splats: SplatsInput = ...,
    pointcloud: PointCloudInput = ...,
    terrain: TerrainInput = ...,
    camera: CameraInput,
    region: Tuple[Sequence[float], Sequence[float]],
    samples: int = ...,
    width: int = ...,
    height: int = ...,
    seed: int = ...,
    sun_azimuth_deg: float = ...,
    sun_elevation_deg: float = ...,
    sun_intensity: float = ...,
    sun_color: Tuple[float, float, float] = ...,
    kappa: float = ...,
    lidar_radius: float = ...,
    lidar_opacity: float = ...,
    page_capacity: int = ...,
    splat_page_size: Optional[int] = ...,
    min_sun_cosine: float = ...,
    certificate: Union[bool, PathLike] = ...,
    cache: Optional[str] = ...,
) -> Dict[str, Any]: ...
def shadow_iou(
    fused: np.ndarray, reference: np.ndarray, valid: Optional[np.ndarray] = ...
) -> float: ...
