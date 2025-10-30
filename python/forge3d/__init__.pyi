# python/forge3d/__init__.pyi
# Type stubs for forge3d public Python API and fallbacks
# Exists to mirror runtime signatures for IDEs and keep tooling aligned
# RELEVANT FILES: python/forge3d/__init__.py, python/forge3d/config.py, src/render/params.rs, examples/terrain_demo.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Sequence, Any, overload, Union, Dict, Literal, Mapping
import os
import numpy as np
from . import geometry
from .terrain_params import (
    LightSettings,
    IblSettings,
    ShadowSettings,
    TriplanarSettings,
    PomSettings,
    LodSettings,
    SamplingSettings,
    ClampSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
)

PathLikeStr = os.PathLike[str] | str

__version__: str

class RendererConfig:
    lighting: Dict[str, Any]
    shading: Dict[str, Any]
    shadows: Dict[str, Any]
    gi: Dict[str, Any]
    atmosphere: Dict[str, Any]
    brdf_override: Optional[str]
    def __init__(
        self,
        *,
        lighting: Dict[str, Any] | None = ...,
        shading: Dict[str, Any] | None = ...,
        shadows: Dict[str, Any] | None = ...,
        gi: Dict[str, Any] | None = ...,
        atmosphere: Dict[str, Any] | None = ...,
        brdf_override: Optional[str] = ...,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def copy(self) -> RendererConfig: ...
    def validate(self) -> None: ...

class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        config: RendererConfig | Mapping[str, Any] | PathLikeStr | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def info(self) -> str: ...
    def render_triangle_rgba(self) -> np.ndarray: ...  # (H,W,4) uint8, C-contiguous
    def render_triangle_png(self, path: PathLikeStr) -> None: ...
    def get_config(self) -> Dict[str, Any]: ...
    def set_lights(self, lights: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> None: ...
    def set_msaa_samples(self, samples: int) -> int: ...
    # Terrain helpers (subset)
    def add_terrain(self, heightmap: np.ndarray, spacing: Tuple[float, float], exaggeration: float, colormap: str) -> None: ...
    def terrain_stats(self) -> Tuple[float, float, float, float]: ...
    def set_height_range(self, min: float, max: float) -> None: ...
    def upload_height_r32f(self) -> None: ...
    def read_full_height_texture(self) -> np.ndarray: ...

class Scene:
    def __init__(self, width: int, height: int, grid: int = ..., colormap: str = ...) -> None: ...
    def set_camera_look_at(self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float],
        fovy_deg: float, znear: float, zfar: float) -> None: ...
    def set_height_from_r32f(self, height_r32f: np.ndarray) -> None: ...
    def render_png(self, path: PathLikeStr) -> None: ...
    def render_rgba(self) -> np.ndarray: ...  # (H,W,4) uint8, C-contiguous
    def set_msaa_samples(self, samples: int) -> int: ...
    def ssao_enabled(self) -> bool: ...
    def set_ssao_enabled(self, enabled: bool) -> bool: ...
    def set_ssao_parameters(self, radius: float, intensity: float, bias: float = ...) -> None: ...
    def get_ssao_parameters(self) -> Tuple[float, float, float]: ...
    def debug_uniforms_f32(self) -> np.ndarray: ...
    def enable_dof(self, quality: str | None = ...) -> None: ...
    def disable_dof(self) -> None: ...
    def dof_enabled(self) -> bool: ...
    def set_dof_camera_params(self, aperture: float, focus_distance: float, focal_length: float) -> None: ...
    def set_dof_f_stop(self, f_stop: float) -> None: ...
    def set_dof_focus_distance(self, distance: float) -> None: ...
    def set_dof_focal_length(self, focal_length: float) -> None: ...
    def set_dof_bokeh_rotation(self, rotation: float) -> None: ...
    def set_dof_transition_ranges(self, near_range: float, far_range: float) -> None: ...
    def set_dof_coc_bias(self, bias: float) -> None: ...
    def set_dof_method(self, method: Literal["gather", "separable"]) -> None: ...
    def set_dof_debug_mode(self, mode: int) -> None: ...
    def set_dof_show_coc(self, show: bool) -> None: ...
    def get_dof_params(self) -> Tuple[float, float, float]: ...
    def enable_cloud_shadows(self, quality: Literal["low", "medium", "high", "ultra"] | None = ...) -> None: ...
    def disable_cloud_shadows(self) -> None: ...
    def is_cloud_shadows_enabled(self) -> bool: ...
    def set_cloud_speed(self, speed_x: float, speed_y: float) -> None: ...
    def set_cloud_scale(self, scale: float) -> None: ...
    def set_cloud_density(self, density: float) -> None: ...
    def set_cloud_coverage(self, coverage: float) -> None: ...
    def set_cloud_shadow_intensity(self, intensity: float) -> None: ...
    def set_cloud_shadow_softness(self, softness: float) -> None: ...
    def set_cloud_wind(self, direction: float, strength: float) -> None: ...
    def set_cloud_wind_vector(self, x: float, y: float, strength: float) -> None: ...
    def set_cloud_noise_params(self, frequency: float, amplitude: float) -> None: ...
    def enable_clouds(self, quality: Literal["low", "medium", "high", "ultra"] = "medium") -> None: ...
    def disable_clouds(self) -> None: ...
    def is_clouds_enabled(self) -> bool: ...
    def set_cloud_render_mode(self, mode: Literal["billboard", "volumetric", "hybrid"]) -> None: ...
    def update_cloud_animation(self, delta_time: float) -> None: ...
    def get_clouds_params(self) -> Tuple[float, float, float, float]: ...
    def set_cloud_animation_preset(self, preset_name: Literal["static", "gentle", "moderate", "stormy", "calm", "windy"] | str) -> None: ...
    def update_cloud_animation(self, delta_time: float) -> None: ...
    def set_cloud_debug_mode(self, mode: int) -> None: ...
    def set_cloud_show_clouds_only(self, show: bool) -> None: ...
    def get_cloud_params(self) -> Tuple[float, float, float, float]: ...
    def debug_lut_format(self) -> str: ...

class Session:
    window: bool
    def __init__(self, window: bool = ..., backend: Optional[str] = ...) -> None: ...
    def info(self) -> str: ...
    @property
    def adapter_name(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def device_type(self) -> str: ...

class IBL:
    intensity: float
    rotation_deg: float
    @staticmethod
    def from_hdr(
        path: PathLikeStr,
        intensity: float = ...,
        rotate_deg: float = ...,
        quality: str = ...,
    ) -> "IBL": ...
    def set_intensity(self, value: float) -> None: ...
    def set_rotation_deg(self, value: float) -> None: ...
    def quality(self) -> str: ...
    def dimensions(self) -> Optional[Tuple[int, int]]: ...
    def path(self) -> str: ...
    def base_resolution(self) -> int: ...
    def set_base_resolution(self, resolution: int) -> None: ...
    def set_cache_dir(self, path: PathLikeStr | None = ...) -> None: ...
    def cache_dir(self) -> Optional[str]: ...

class Colormap1D:
    domain: Tuple[float, float]
    @staticmethod
    def from_stops(
        stops: Sequence[Tuple[float, str]],
        domain: Tuple[float, float],
    ) -> Colormap1D: ...

class OverlayLayer:
    strength: float
    offset: float
    blend_mode: str
    domain: Tuple[float, float]
    @staticmethod
    def from_colormap1d(
        colormap: Colormap1D,
        strength: float = ...,
        offset: float = ...,
        blend_mode: str = ...,
        domain: Tuple[float, float] = ...,
    ) -> OverlayLayer: ...
    @property
    def colormap(self) -> Optional[Colormap1D]: ...
    @property
    def kind(self) -> str: ...

class TerrainRenderParams:
    size_px: Tuple[int, int]
    render_scale: float
    msaa_samples: int
    z_scale: float
    cam_target: Tuple[float, float, float]
    cam_radius: float
    cam_phi_deg: float
    cam_theta_deg: float
    cam_gamma_deg: float
    fov_y_deg: float
    clip: Tuple[float, float]
    exposure: float
    gamma: float
    albedo_mode: str
    colormap_strength: float
    overlays: Sequence[OverlayLayer]
    def __init__(self, params: Any) -> None: ...
    @property
    def light(self) -> Any: ...
    @property
    def ibl(self) -> Any: ...
    @property
    def shadows(self) -> Any: ...
    @property
    def triplanar(self) -> Any: ...
    @property
    def pom(self) -> Any: ...
    @property
    def lod(self) -> Any: ...
    @property
    def sampling(self) -> Any: ...
    @property
    def clamp(self) -> Any: ...
    @property
    def python_object(self) -> Any: ...

class Frame:
    width: int
    height: int
    format: str
    def save(self, path: PathLikeStr) -> None: ...
    def to_numpy(self) -> np.ndarray: ...
    def size(self) -> Tuple[int, int]: ...

class TerrainRenderer:
    def __init__(self, session: "Session") -> None: ...
    def render_terrain_pbr_pom(
        self,
        material_set: "MaterialSet",
        env_maps: "IBL",
        params: TerrainRenderParams,
        heightmap: np.ndarray,
        target: Optional[Any] = ...,
    ) -> Frame: ...

# Vector Picking & OIT helpers
def set_point_shape_mode(mode: int) -> None: ...
def set_point_lod_threshold(threshold: float) -> None: ...

def is_weighted_oit_available() -> bool: ...

def vector_oit_and_pick_demo(width: int = ..., height: int = ...) -> Tuple[np.ndarray, int]: ...

def vector_render_oit_py(
    width: int,
    height: int,
    *,
    points_xy: Optional[Sequence[Tuple[float, float]]] = ...,
    point_rgba: Optional[Sequence[Tuple[float, float, float, float]]] = ...,
    point_size: Optional[Sequence[float]] = ...,
    polylines: Optional[Sequence[Sequence[Tuple[float, float]]]] = ...,
    polyline_rgba: Optional[Sequence[Tuple[float, float, float, float]]] = ...,
    stroke_width: Optional[Sequence[float]] = ...,
) -> np.ndarray: ...  # (H,W,4) uint8

def vector_render_pick_map_py(
    width: int,
    height: int,
    *,
    points_xy: Optional[Sequence[Tuple[float, float]]] = ...,
    polylines: Optional[Sequence[Sequence[Tuple[float, float]]]] = ...,
    base_pick_id: Optional[int] = ...,
) -> np.ndarray: ...  # (H,W) uint32

def vector_render_oit_and_pick_py(
    width: int,
    height: int,
    *,
    points_xy: Optional[Sequence[Tuple[float, float]]] = ...,
    point_rgba: Optional[Sequence[Tuple[float, float, float, float]]] = ...,
    point_size: Optional[Sequence[float]] = ...,
    polylines: Optional[Sequence[Sequence[Tuple[float, float]]]] = ...,
    polyline_rgba: Optional[Sequence[Tuple[float, float, float, float]]] = ...,
    stroke_width: Optional[Sequence[float]] = ...,
    base_pick_id: Optional[int] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...  # (H,W,4) uint8, (H,W) uint32

def composite_rgba_over(bottom: np.ndarray, top: np.ndarray, *, premultiplied: bool = ...) -> np.ndarray: ...  # (H,W,4) uint8

# Optional export if compiled with --features terrain_spike
class TerrainSpike: ...

def png_to_numpy(path: PathLikeStr) -> np.ndarray: ...          # (H,W,4) uint8
def numpy_to_png(path: PathLikeStr, array: np.ndarray) -> None: ...

def render_triangle_rgba(width: int, height: int) -> np.ndarray: ...
def render_triangle_png(path: PathLikeStr, width: int, height: int) -> None: ...

def dem_stats(heightmap: np.ndarray) -> Tuple[float, float, float, float]: ...
def dem_normalize(heightmap: np.ndarray, *, mode: str = ..., out_range: Tuple[float, float] = ..., eps: float = ..., return_stats: bool = ...) -> np.ndarray | Tuple[np.ndarray, Tuple[float, float, float, float]]: ...

def render_debug_pattern_frame(width: int, height: int) -> Any: ...

def enumerate_adapters() -> list[dict[str, Any]]: ...
def device_probe(backend: Optional[str] = ...) -> dict[str, Any]: ...

def memory_metrics() -> Dict[str, Any]: ...
def budget_remaining() -> int: ...
def utilization_ratio() -> float: ...
def override_memory_limit(limit_bytes: int) -> None: ...

# Add the timing API stub at top-level
from typing import Dict, TypedDict

class _Stats(TypedDict):
    min_ms: float
    p50_ms: float
    mean_ms: float
    p95_ms: float
    max_ms: float
    std_ms: float

class _Throughput(TypedDict):
    fps: float
    mpix_per_s: float

class _Env(TypedDict, total=False):
    status: str
    adapter_name: str
    backend: str
    device_type: str

class BenchmarkResult(TypedDict):
    op: str
    width: int
    height: int
    pixels: int
    iterations: int
    warmup: int
    stats: _Stats
    throughput: _Throughput
    env: _Env

def run_benchmark(
    op: Literal["renderer_rgba","renderer_png","scene_rgba","numpy_to_png","png_to_numpy"],
    width: int,
    height: int,
    *,
    iterations: int = ...,
    warmup: int = ...,
    grid: int = ...,
    colormap: str = ...,
    seed: int = ...,
) -> Dict[str, Any]: ...

# Convenience vector scene wrapper (re-exported from forge3d.vector)
class VectorScene:
    def __init__(self) -> None: ...
    def clear(self) -> None: ...
    def add_point(self, x: float, y: float, rgba: Tuple[float, float, float, float] | None = ..., size: float | None = ...) -> None: ...
    def add_polyline(self, path: Sequence[Tuple[float, float]], rgba: Tuple[float, float, float, float] | None = ..., width: float | None = ...) -> None: ...
    def render_oit(self, width: int, height: int) -> np.ndarray: ...  # (H,W,4) uint8
    def render_pick_map(self, width: int, height: int, base_pick_id: int = ...) -> np.ndarray: ...  # (H,W) uint32
    def render_oit_and_pick(self, width: int, height: int, base_pick_id: int = ...) -> Tuple[np.ndarray, np.ndarray]: ...  # (H,W,4) uint8, (H,W) uint32

# P5: Screen-space effects classes
class SSAOSettings:
    radius: float
    intensity: float
    bias: float
    sample_count: int
    spiral_turns: float
    technique: str  # "SSAO" or "GTAO"
    blur_radius: int
    temporal_alpha: float
    def __init__(
        self,
        radius: float = ...,
        intensity: float = ...,
        bias: float = ...,
        sample_count: int = ...,
        spiral_turns: float = ...,
        technique: str = ...,
        blur_radius: int = ...,
        temporal_alpha: float = ...
    ) -> None: ...
    @staticmethod
    def ssao(radius: float, intensity: float) -> SSAOSettings: ...
    @staticmethod
    def gtao(radius: float, intensity: float) -> SSAOSettings: ...

class SSGISettings:
    ray_steps: int
    ray_radius: float
    ray_thickness: float
    intensity: float
    temporal_alpha: float
    use_half_res: bool
    ibl_fallback: float
    def __init__(
        self,
        ray_steps: int = ...,
        ray_radius: float = ...,
        ray_thickness: float = ...,
        intensity: float = ...,
        temporal_alpha: float = ...,
        use_half_res: bool = ...,
        ibl_fallback: float = ...
    ) -> None: ...

class SSRSettings:
    max_steps: int
    max_distance: float
    thickness: float
    stride: float
    intensity: float
    roughness_fade: float
    edge_fade: float
    temporal_alpha: float
    def __init__(
        self,
        max_steps: int = ...,
        max_distance: float = ...,
        thickness: float = ...,
        stride: float = ...,
        intensity: float = ...,
        roughness_fade: float = ...,
        edge_fade: float = ...,
        temporal_alpha: float = ...
    ) -> None: ...

# P6: Atmospherics & sky classes
class SkySettings:
    sun_direction: Tuple[float, float, float]
    turbidity: float
    ground_albedo: float
    model: str  # "off", "preetham", or "hosek-wilkie"
    sun_intensity: float
    exposure: float
    def __init__(
        self,
        sun_direction: Tuple[float, float, float] = ...,
        turbidity: float = ...,
        ground_albedo: float = ...,
        model: str = ...,
        sun_intensity: float = ...,
        exposure: float = ...
    ) -> None: ...
    @staticmethod
    def preetham(turbidity: float, ground_albedo: float) -> SkySettings: ...
    @staticmethod
    def hosek_wilkie(turbidity: float, ground_albedo: float) -> SkySettings: ...
    def with_sun_angles(self, azimuth_deg: float, elevation_deg: float) -> None: ...

class VolumetricSettings:
    density: float
    height_falloff: float
    phase_g: float
    max_steps: int
    start_distance: float
    max_distance: float
    absorption: float
    sun_intensity: float
    scattering_color: Tuple[float, float, float]
    temporal_alpha: float
    ambient_color: Tuple[float, float, float]
    use_shadows: bool
    jitter_strength: float
    phase_function: str  # "isotropic" or "hg"
    def __init__(
        self,
        density: float = ...,
        height_falloff: float = ...,
        phase_g: float = ...,
        max_steps: int = ...,
        start_distance: float = ...,
        max_distance: float = ...,
        absorption: float = ...,
        sun_intensity: float = ...,
        scattering_color: Tuple[float, float, float] = ...,
        temporal_alpha: float = ...,
        ambient_color: Tuple[float, float, float] = ...,
        use_shadows: bool = ...,
        jitter_strength: float = ...,
        phase_function: str = ...
    ) -> None: ...
    @staticmethod
    def with_god_rays(density: float, phase_g: float) -> VolumetricSettings: ...
    @staticmethod
    def uniform_fog(density: float) -> VolumetricSettings: ...

# A13: Path guiding (Python utility)
class OnlineGuidingGrid:
    width: int
    height: int
    bins_per_cell: int
    def __init__(self, width: int, height: int, bins_per_cell: int = ...) -> None: ...
    def update(self, x: int, y: int, bin_index: int, weight: float = ...) -> None: ...
    def pdf(self, x: int, y: int) -> np.ndarray: ...  # (B,) float32, sum=1
    def dims(self) -> Tuple[int, int, int]: ...

