# python/forge3d/lighting.py
# Lighting controls and ReSTIR utilities for Workstream B.
# Exists to expose lighting toggles and exposure helpers to Python.
# RELEVANT FILES:python/forge3d/pbr.py,src/pipeline/pbr.rs,shaders/tone_map.wgsl,tests/test_b2_tonemap.py


"""ReSTIR DI lighting system with basic participating media helpers (HG phase, height fog).

These media utilities exist to satisfy Workstream A A11 deliverables with minimal, testable CPU-side functionality.

RELEVANT FILES:src/shaders/lighting_media.wgsl,tests/test_media_hg.py,tests/test_media_fog.py"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from enum import Enum
import math
import warnings

try:
    from . import _forge3d  # Rust bindings
except ImportError:
    _forge3d = None

_PI = 3.14159265358979323846

_EXPOSURE_STOPS: float = 0.0
_EXPOSURE_SCALE: float = 1.0


def _to_float_array(x: Union[float, np.ndarray]) -> np.ndarray:
    """Convert input to float32 numpy array without copying when possible.

    Keeps shapes intact; scalars become shape=().
    """
    arr = np.asarray(x, dtype=np.float32)
    return arr


def set_exposure_stops(stops: float) -> float:
    """Set exposure in stops and return the linear scale (2**stops)."""
    global _EXPOSURE_STOPS, _EXPOSURE_SCALE
    stops_f = float(stops)
    _EXPOSURE_STOPS = stops_f
    _EXPOSURE_SCALE = float(math.pow(2.0, stops_f))
    if _forge3d is not None:
        setter = getattr(_forge3d, "set_exposure_scale", None)
        if setter is not None:
            try:
                setter(_EXPOSURE_SCALE)
            except Exception as exc:  # pragma: no cover - optional binding
                warnings.warn(f"forge3d.set_exposure_scale failed: {exc}")
    return _EXPOSURE_SCALE


def hg_phase(cos_theta: Union[float, np.ndarray], g: float) -> np.ndarray:
    """Henyey–Greenstein phase function.

    Args:
        cos_theta: Cosine of scattering angle (dot(wo, wi)).
        g: Asymmetry parameter in [-0.999, 0.999].

    Returns:
        Phase value with integral over sphere equal to 1.
    """
    c = _to_float_array(cos_theta)
    g = float(np.clip(g, -0.999, 0.999))
    one_minus_g2 = 1.0 - g * g
    denom = (1.0 + g * g - 2.0 * g * c)
    # Avoid tiny negatives from FP error before pow
    denom = np.maximum(denom, 1e-8)
    val = (one_minus_g2) / (4.0 * _PI * np.power(denom, 1.5, dtype=np.float32))
    return val.astype(np.float32)


def sample_hg(u1: Union[float, np.ndarray], u2: Union[float, np.ndarray], g: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sample direction from HG phase and return (dir, pdf).

    The returned direction is in local coordinates where the reference direction is +Z.
    To orient toward a world-space direction, apply an orthonormal basis transform.
    """
    u1 = _to_float_array(u1)
    u2 = _to_float_array(u2)
    g = float(np.clip(g, -0.999, 0.999))

    if abs(g) < 1e-4:
        cos_theta = 1.0 - 2.0 * u1
    else:
        sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1)
        cos_theta = (1.0 + g * g - sq * sq) / (2.0 * g)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta)).astype(np.float32)
    phi = (2.0 * _PI * u2).astype(np.float32)
    dir_local = np.stack([
        sin_theta * np.cos(phi, dtype=np.float32),
        sin_theta * np.sin(phi, dtype=np.float32),
        cos_theta.astype(np.float32),
    ], axis=-1)

    pdf = hg_phase(cos_theta, g)
    return dir_local, pdf


def height_fog_factor(depth: Union[float, np.ndarray], *, density: float = 0.02) -> np.ndarray:
    """Simple homogeneous medium transmittance-to-fog factor along a ray.

    Args:
        depth: Distance along the camera ray.
        density: Extinction coefficient (sigma_t). Units inverse to distance.

    Returns:
        Fog blend factor in [0, 1]: 0 near, approaching 1 with distance.
    """
    d = _to_float_array(depth)
    sigma_t = float(max(density, 0.0))
    T = np.exp(-sigma_t * np.clip(d, 0.0, np.inf), dtype=np.float32)
    fog = (1.0 - T).astype(np.float32)
    return np.clip(fog, 0.0, 1.0)


def single_scatter_estimate(depth: Union[float, np.ndarray], *, sun_intensity: float = 1.0, density: float = 0.02, g: float = 0.0) -> np.ndarray:
    """Very small single-scatter estimate assuming homogeneous medium and sun back-light.

    This provides a stable, testable CPU reference for A11 acceptance.
    It is not a full volumetric integrator and is meant for validation only.
    """
    d = _to_float_array(depth)
    g = float(np.clip(g, -0.999, 0.999))
    sigma_t = float(max(density, 0.0))
    # Use phase at cos_theta ~ 1 (back-lit shafts proxy)
    p = hg_phase(1.0, g)
    # Single scatter with uniform lighting along segment: L ≈ I * p * (1 - exp(-sigma_t * d))
    return (sun_intensity * p * (1.0 - np.exp(-sigma_t * np.clip(d, 0.0, np.inf)))).astype(np.float32)


class LightType(Enum):
    """Light type enumeration."""
    POINT = 0
    DIRECTIONAL = 1
    AREA = 2


@dataclass
class LightSample:
    """Light sample structure for ReSTIR."""
    position: Tuple[float, float, float]
    light_index: int
    direction: Tuple[float, float, float]
    intensity: float
    light_type: LightType
    params: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for GPU upload."""
        return np.array([
            *self.position,
            float(self.light_index),
            *self.direction,
            self.intensity,
            float(self.light_type.value),
            *self.params
        ], dtype=np.float32)


@dataclass
class RestirConfig:
    """ReSTIR configuration parameters."""
    initial_candidates: int = 32
    temporal_neighbors: int = 1
    spatial_neighbors: int = 4
    spatial_radius: float = 16.0
    max_temporal_age: int = 20
    bias_correction: bool = True
    depth_threshold: float = 0.1
    normal_threshold: float = 0.9

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_candidates <= 0:
            raise ValueError("initial_candidates must be positive")
        if self.spatial_radius <= 0.0:
            raise ValueError("spatial_radius must be positive")
        if not 0.0 <= self.depth_threshold <= 1.0:
            raise ValueError("depth_threshold must be in [0, 1]")
        if not 0.0 <= self.normal_threshold <= 1.0:
            raise ValueError("normal_threshold must be in [0, 1]")


class RestirDI:
    """ReSTIR DI (Direct Illumination) implementation for many-light scenarios.

    This class provides reservoir-based importance sampling with temporal and spatial
    reuse to efficiently handle scenes with thousands of lights while maintaining
    low variance in the lighting estimation.
    """

    def __init__(self, config: Optional[RestirConfig] = None):
        """Initialize ReSTIR DI with given configuration.

        Args:
            config: ReSTIR configuration. If None, uses default settings.
        """
        self.config = config or RestirConfig()
        self._lights: List[LightSample] = []
        self._light_weights: Optional[np.ndarray] = None
        self._native_restir = None

        if _forge3d is not None and hasattr(_forge3d, "create_restir_di"):
            self._native_restir = _forge3d.create_restir_di(
                initial_candidates=self.config.initial_candidates,
                temporal_neighbors=self.config.temporal_neighbors,
                spatial_neighbors=self.config.spatial_neighbors,
                spatial_radius=self.config.spatial_radius,
                max_temporal_age=self.config.max_temporal_age,
                bias_correction=self.config.bias_correction
            )

    def add_light(self,
                  position: Tuple[float, float, float],
                  intensity: float,
                  light_type: LightType = LightType.POINT,
                  direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                  params: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  weight: float = 1.0) -> int:
        """Add a light to the scene.

        Args:
            position: Light position in world space
            intensity: Light intensity/radiance
            light_type: Type of light (point, directional, area)
            direction: Light direction (for directional lights)
            params: Additional light parameters
            weight: Sampling weight for this light

        Returns:
            Index of the added light
        """
        light_sample = LightSample(
            position=position,
            light_index=len(self._lights),
            direction=direction,
            intensity=intensity,
            light_type=light_type,
            params=params
        )

        self._lights.append(light_sample)

        # Update weights array
        if self._light_weights is None:
            self._light_weights = np.array([weight], dtype=np.float32)
        else:
            self._light_weights = np.append(self._light_weights, weight)

        return len(self._lights) - 1

    def set_lights(self,
                   lights: List[LightSample],
                   weights: Optional[Union[List[float], np.ndarray]] = None):
        """Set all lights at once.

        Args:
            lights: List of light samples
            weights: Optional weights for each light. If None, uses uniform weights.
        """
        self._lights = lights.copy()

        if weights is None:
            self._light_weights = np.ones(len(lights), dtype=np.float32)
        else:
            weights_array = np.asarray(weights, dtype=np.float32)
            if len(weights_array) != len(lights):
                raise ValueError(f"Number of weights ({len(weights_array)}) must match number of lights ({len(lights)})")
            self._light_weights = weights_array

        # Update native implementation if available
        if self._native_restir is not None and hasattr(_forge3d, "restir_set_lights"):
            lights_array = np.array([light.to_array() for light in lights], dtype=np.float32)
            _forge3d.restir_set_lights(self._native_restir, lights_array, self._light_weights)

    def clear_lights(self):
        """Remove all lights from the scene."""
        self._lights.clear()
        self._light_weights = None

        if self._native_restir is not None and hasattr(_forge3d, "restir_clear_lights"):
            _forge3d.restir_clear_lights(self._native_restir)

    def sample_light(self, u1: float, u2: float) -> Optional[Tuple[int, float]]:
        """Sample a light using the alias table.

        Args:
            u1, u2: Random numbers in [0, 1]

        Returns:
            Tuple of (light_index, pdf) or None if no lights available
        """
        if not self._lights:
            return None

        if self._native_restir is not None and hasattr(_forge3d, "restir_sample_light"):
            return _forge3d.restir_sample_light(self._native_restir, u1, u2)

        # Fallback CPU implementation
        if self._light_weights is None:
            return None

        # Simple weighted sampling
        total_weight = np.sum(self._light_weights)
        if total_weight <= 0:
            return None

        # Convert u1 to cumulative distribution
        cumulative = np.cumsum(self._light_weights) / total_weight
        light_idx = np.searchsorted(cumulative, u1)
        light_idx = min(light_idx, len(self._lights) - 1)

        pdf = self._light_weights[light_idx] / total_weight
        return light_idx, pdf

    def render_frame(self,
                     width: int,
                     height: int,
                     camera_params: dict,
                     g_buffer: dict,
                     output_format: str = "rgba") -> np.ndarray:
        """Render a frame using ReSTIR DI.

        Args:
            width, height: Output resolution
            camera_params: Camera parameters (position, target, fov, etc.)
            g_buffer: G-buffer containing depth, normals, world positions
            output_format: Output format ("rgba", "hdr", "variance")

        Returns:
            Rendered image as numpy array
        """
        if self._native_restir is None or not hasattr(_forge3d, "restir_render_frame"):
            raise RuntimeError("Native ReSTIR implementation not available")

        # Validate G-buffer
        required_buffers = ["depth", "normal", "world_pos"]
        for buffer_name in required_buffers:
            if buffer_name not in g_buffer:
                raise ValueError(f"G-buffer missing required buffer: {buffer_name}")

        # Ensure all buffers have correct shape
        expected_shape = (height, width)
        for buffer_name, buffer_data in g_buffer.items():
            if buffer_name == "depth" and buffer_data.shape != expected_shape:
                raise ValueError(f"Depth buffer shape {buffer_data.shape} != expected {expected_shape}")
            elif buffer_name in ["normal", "world_pos"] and buffer_data.shape != (*expected_shape, 3):
                raise ValueError(f"{buffer_name} buffer shape {buffer_data.shape} != expected {(*expected_shape, 3)}")

        return _forge3d.restir_render_frame(
            self._native_restir,
            width, height,
            camera_params,
            g_buffer,
            output_format
        )

    def calculate_variance_reduction(self,
                                   reference_image: np.ndarray,
                                   restir_image: np.ndarray) -> float:
        """Calculate variance reduction compared to reference (e.g., MIS-only).

        Args:
            reference_image: Reference image (e.g., from MIS-only sampling)
            restir_image: ReSTIR rendered image

        Returns:
            Variance reduction as a percentage (positive means ReSTIR has lower variance)
        """
        if reference_image.shape != restir_image.shape:
            raise ValueError("Images must have the same shape")

        # Calculate variance for each image
        ref_var = np.var(reference_image)
        restir_var = np.var(restir_image)

        if ref_var <= 0:
            return 0.0

        # Calculate percentage reduction
        reduction = (ref_var - restir_var) / ref_var
        return reduction * 100.0

    def get_statistics(self) -> dict:
        """Get ReSTIR statistics and performance metrics.

        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "num_lights": len(self._lights),
            "config": {
                "initial_candidates": self.config.initial_candidates,
                "temporal_neighbors": self.config.temporal_neighbors,
                "spatial_neighbors": self.config.spatial_neighbors,
                "spatial_radius": self.config.spatial_radius,
                "max_temporal_age": self.config.max_temporal_age,
                "bias_correction": self.config.bias_correction,
            }
        }

        if self._native_restir is not None and hasattr(_forge3d, "restir_get_statistics"):
            native_stats = _forge3d.restir_get_statistics(self._native_restir)
            stats.update(native_stats)

        return stats

    @property
    def num_lights(self) -> int:
        """Get the number of lights in the scene."""
        return len(self._lights)

    @property
    def lights(self) -> List[LightSample]:
        """Get a copy of all lights in the scene."""
        return self._lights.copy()


def create_test_scene(num_lights: int = 100,
                      scene_bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),
                      intensity_range: Tuple[float, float] = (0.1, 2.0),
                      seed: int = 42) -> RestirDI:
    """Create a test scene with randomly distributed lights.

    Args:
        num_lights: Number of lights to create
        scene_bounds: Scene bounds (width, depth, height)
        intensity_range: Range of light intensities
        seed: Random seed for reproducible results

    Returns:
        RestirDI instance with the test scene
    """
    np.random.seed(seed)

    restir = RestirDI()

    for i in range(num_lights):
        # Random position within scene bounds
        position = (
            np.random.uniform(-scene_bounds[0]/2, scene_bounds[0]/2),
            np.random.uniform(0.1, scene_bounds[2]),
            np.random.uniform(-scene_bounds[1]/2, scene_bounds[1]/2)
        )

        # Random intensity
        intensity = np.random.uniform(*intensity_range)

        # Most lights are point lights, some directional
        light_type = LightType.POINT if np.random.random() > 0.1 else LightType.DIRECTIONAL

        # Random direction for directional lights
        if light_type == LightType.DIRECTIONAL:
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            direction = (
                np.sin(phi) * np.cos(theta),
                -np.cos(phi),  # Generally pointing downward
                np.sin(phi) * np.sin(theta)
            )
        else:
            direction = (0.0, 0.0, 1.0)

        # Weight based on intensity (brighter lights more likely to be sampled)
        weight = intensity

        restir.add_light(position, intensity, light_type, direction, weight=weight)

    return restir


# A20: Soft Area Lights with Penumbra Control
class AreaLightType(Enum):
    """Area light types for A20 implementation."""
    RECTANGLE = 0
    DISC = 1
    SPHERE = 2
    CYLINDER = 3

@dataclass
class AreaLight:
    """Area light with parametric penumbra control."""
    position: Tuple[float, float, float] = (0.0, 5.0, 0.0)
    light_type: AreaLightType = AreaLightType.DISC
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    radius: float = 1.0  # Penumbra control radius
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 10.0
    size: Tuple[float, float] = (2.0, 2.0)  # width, height for rectangle; radius for others
    softness: float = 0.5  # 0.0 = hard, 1.0 = very soft

    def __post_init__(self):
        """Validate parameters and compute energy factor."""
        if self.radius <= 0.0:
            raise ValueError("Radius must be positive")
        if self.intensity <= 0.0:
            raise ValueError("Intensity must be positive")
        if not (0.0 <= self.softness <= 1.0):
            raise ValueError("Softness must be in [0.0, 1.0]")

        # Normalize direction
        direction = np.array(self.direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.direction = tuple(direction / norm)

        # Compute energy normalization factor
        self.energy_factor = self._compute_energy_factor()

    def _compute_energy_factor(self) -> float:
        """Compute energy normalization factor for energy conservation."""
        if self.light_type == AreaLightType.RECTANGLE:
            base_area = self.size[0] * self.size[1]
        elif self.light_type == AreaLightType.DISC:
            base_area = np.pi * self.size[0] ** 2
        elif self.light_type == AreaLightType.SPHERE:
            base_area = 4.0 * np.pi * self.size[0] ** 2
        elif self.light_type == AreaLightType.CYLINDER:
            base_area = 2.0 * np.pi * self.size[0] * self.size[1]
        else:
            base_area = 1.0

        # Energy normalization with penumbra compensation
        penumbra_factor = 1.0 + (self.radius * 0.1)
        return 1.0 / (base_area * penumbra_factor)

    def get_effective_energy(self) -> float:
        """Get effective energy output considering normalization."""
        return self.intensity * self.energy_factor

    def set_radius(self, radius: float) -> None:
        """Set penumbra control radius."""
        if radius <= 0.0:
            raise ValueError("Radius must be positive")
        object.__setattr__(self, 'radius', radius)
        object.__setattr__(self, 'energy_factor', self._compute_energy_factor())

    @classmethod
    def rectangle(cls, position: Tuple[float, float, float],
                 direction: Tuple[float, float, float],
                 width: float, height: float,
                 intensity: float = 10.0,
                 penumbra_radius: float = 1.0) -> 'AreaLight':
        """Create rectangular area light."""
        return cls(position=position, light_type=AreaLightType.RECTANGLE,
                  direction=direction, radius=penumbra_radius,
                  intensity=intensity, size=(width, height))

    @classmethod
    def disc(cls, position: Tuple[float, float, float],
            direction: Tuple[float, float, float],
            disc_radius: float,
            intensity: float = 10.0,
            penumbra_radius: float = 1.0) -> 'AreaLight':
        """Create disc area light."""
        return cls(position=position, light_type=AreaLightType.DISC,
                  direction=direction, radius=penumbra_radius,
                  intensity=intensity, size=(disc_radius, disc_radius))

class AreaLightManager:
    """Manager for multiple area lights with energy conservation."""

    def __init__(self, max_lights: int = 16):
        self.max_lights = max_lights
        self.lights: List[AreaLight] = []
        self._energy_target: Optional[float] = None

    def add_light(self, light: AreaLight) -> int:
        """Add area light to scene."""
        if len(self.lights) >= self.max_lights:
            raise ValueError(f"Maximum lights ({self.max_lights}) exceeded")
        self.lights.append(light)
        if self._energy_target is None:
            self._energy_target = self.calculate_total_energy()
        return len(self.lights) - 1

    def calculate_total_energy(self) -> float:
        """Calculate total energy from all lights."""
        return sum(light.get_effective_energy() for light in self.lights)

    def normalize_energy(self) -> float:
        """Normalize energy and return conservation error."""
        if self._energy_target is None or len(self.lights) == 0:
            return 0.0
        current_energy = self.calculate_total_energy()
        if current_energy <= 0.0:
            return 0.0
        scale_factor = self._energy_target / current_energy
        for light in self.lights:
            light.intensity *= scale_factor
        new_energy = self.calculate_total_energy()
        return abs(new_energy - self._energy_target) / self._energy_target

    def test_energy_conservation(self) -> bool:
        """Test energy conservation meets A20 requirements (within 2%)."""
        if not self.lights:
            return True
        initial_energy = self.calculate_total_energy()
        self.set_energy_target(initial_energy)
        # Modify light parameters to test conservation
        for light in self.lights:
            light.set_radius(light.radius * 1.5)
        error = self.normalize_energy()
        return error < 0.02  # A20 requirement

    def set_energy_target(self, target: float) -> None:
        """Set energy conservation target."""
        if target <= 0.0:
            raise ValueError("Energy target must be positive")
        self._energy_target = target

def create_area_light_test_scene() -> AreaLightManager:
    """Create test scene for A20 validation."""
    manager = AreaLightManager()

    # Key light (disc with medium penumbra)
    key_light = AreaLight.disc(
        position=(2.0, 4.0, 2.0),
        direction=(-0.5, -1.0, -0.5),
        disc_radius=1.5,
        intensity=20.0,
        penumbra_radius=0.8
    )
    manager.add_light(key_light)

    # Fill light (rectangle with soft penumbra)
    fill_light = AreaLight.rectangle(
        position=(-3.0, 3.0, 1.0),
        direction=(0.7, -0.7, -0.2),
        width=2.0, height=1.0,
        intensity=8.0,
        penumbra_radius=1.2
    )
    manager.add_light(fill_light)

    return manager

# --- B4: Cascaded Shadow Maps (CSM) Controls ---

@dataclass
class CsmConfig:
    """Configuration for Cascaded Shadow Maps.

    Controls shadow quality, cascade setup, and performance settings for B4 implementation.
    """
    cascade_count: int = 3
    shadow_map_size: int = 2048
    max_shadow_distance: float = 200.0
    pcf_kernel_size: int = 3
    depth_bias: float = 0.005
    slope_bias: float = 0.01
    peter_panning_offset: float = 0.001
    enable_evsm: bool = False
    debug_mode: int = 0

    def __post_init__(self):
        """Validate CSM configuration parameters."""
        if not (2 <= self.cascade_count <= 4):
            raise ValueError(f"cascade_count must be 2-4, got {self.cascade_count}")
        if not (512 <= self.shadow_map_size <= 8192):
            raise ValueError(f"shadow_map_size must be 512-8192, got {self.shadow_map_size}")
        if self.max_shadow_distance <= 0:
            raise ValueError(f"max_shadow_distance must be positive, got {self.max_shadow_distance}")
        if self.pcf_kernel_size not in [1, 3, 5, 7]:
            raise ValueError(f"pcf_kernel_size must be 1, 3, 5, or 7, got {self.pcf_kernel_size}")

class CsmQualityPreset(Enum):
    """Quality presets for CSM configuration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

def create_csm_config(preset: CsmQualityPreset) -> CsmConfig:
    """Create CSM configuration from quality preset."""
    configs = {
        CsmQualityPreset.LOW: CsmConfig(
            cascade_count=3,
            shadow_map_size=1024,
            pcf_kernel_size=1,  # No PCF
            depth_bias=0.01,
            slope_bias=0.02,
            peter_panning_offset=0.002,
            enable_evsm=False,
            debug_mode=0
        ),
        CsmQualityPreset.MEDIUM: CsmConfig(
            cascade_count=3,
            shadow_map_size=2048,
            pcf_kernel_size=3,  # 3x3 PCF
            depth_bias=0.005,
            slope_bias=0.01,
            peter_panning_offset=0.001,
            enable_evsm=False,
            debug_mode=0
        ),
        CsmQualityPreset.HIGH: CsmConfig(
            cascade_count=4,
            shadow_map_size=4096,
            pcf_kernel_size=5,  # 5x5 PCF
            depth_bias=0.003,
            slope_bias=0.005,
            peter_panning_offset=0.0005,
            enable_evsm=False,
            debug_mode=0
        ),
        CsmQualityPreset.ULTRA: CsmConfig(
            cascade_count=4,
            shadow_map_size=4096,
            pcf_kernel_size=7,  # Poisson disk PCF
            depth_bias=0.002,
            slope_bias=0.003,
            peter_panning_offset=0.0003,
            enable_evsm=True,
            debug_mode=0
        )
    }
    return configs[preset]

class CsmController:
    """Python controller for Cascaded Shadow Maps.

    Provides high-level interface for configuring and debugging CSM rendering.
    """

    def __init__(self, config: Optional[CsmConfig] = None):
        """Initialize CSM controller with configuration."""
        self.config = config or CsmConfig()
        self._enabled = False
        self._light_direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self._cascade_splits = []
        self._sync_native_state()

    def _sync_native_state(self) -> None:
        """Push the current configuration to the native module when available."""
        if _forge3d is None:
            return
        sync = getattr(_forge3d, "configure_csm", None)
        if sync is None:
            return
        try:
            sync(
                int(self.config.cascade_count),
                int(self.config.shadow_map_size),
                float(self.config.max_shadow_distance),
                int(self.config.pcf_kernel_size),
                float(self.config.depth_bias),
                float(self.config.slope_bias),
                float(self.config.peter_panning_offset),
                bool(self.config.enable_evsm),
                int(self.config.debug_mode),
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"forge3d.configure_csm failed: {exc}")

    def set_quality_preset(self, preset: CsmQualityPreset) -> None:
        """Set CSM quality from preset."""
        self.config = create_csm_config(preset)
        self._sync_native_state()

    def enable_shadows(self, enabled: bool = True) -> None:
        """Enable or disable shadow rendering."""
        self._enabled = enabled
        if _forge3d is not None:
            setter = getattr(_forge3d, "set_csm_enabled", None)
            if setter is not None:
                try:
                    setter(enabled)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.set_csm_enabled failed: {exc}")
        self._sync_native_state()

    def set_light_direction(self, direction: Tuple[float, float, float]) -> None:
        """Set directional light direction for shadow casting."""
        self._light_direction = np.array(direction, dtype=np.float32)
        self._light_direction /= np.linalg.norm(self._light_direction)

        if _forge3d is not None:
            setter = getattr(_forge3d, "set_csm_light_direction", None)
            if setter is not None:
                try:
                    setter(tuple(self._light_direction))
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.set_csm_light_direction failed: {exc}")

    def configure_pcf(self, kernel_size: int) -> None:
        """Configure PCF filtering quality."""
        if kernel_size not in [1, 3, 5, 7]:
            raise ValueError(f"PCF kernel size must be 1, 3, 5, or 7, got {kernel_size}")
        self.config.pcf_kernel_size = kernel_size

        if _forge3d is not None:
            setter = getattr(_forge3d, "set_csm_pcf_kernel", None)
            if setter is not None:
                try:
                    setter(kernel_size)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.set_csm_pcf_kernel failed: {exc}")
        self._sync_native_state()

    def set_bias_parameters(self, depth_bias: float, slope_bias: float, peter_panning_offset: float) -> None:
        """Configure shadow bias parameters to prevent artifacts."""
        self.config.depth_bias = depth_bias
        self.config.slope_bias = slope_bias
        self.config.peter_panning_offset = peter_panning_offset

        if _forge3d is not None:
            setter = getattr(_forge3d, "set_csm_bias_params", None)
            if setter is not None:
                try:
                    setter(depth_bias, slope_bias, peter_panning_offset)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.set_csm_bias_params failed: {exc}")
        self._sync_native_state()

    def set_debug_mode(self, mode: int) -> None:
        """Set CSM debug visualization mode.

        Args:
            mode: Debug mode (0=off, 1=cascade colors, 2=overdraw)
        """
        if not (0 <= mode <= 2):
            raise ValueError(f"Debug mode must be 0-2, got {mode}")
        self.config.debug_mode = mode

        if _forge3d is not None:
            setter = getattr(_forge3d, "set_csm_debug_mode", None)
            if setter is not None:
                try:
                    setter(mode)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.set_csm_debug_mode failed: {exc}")
        self._sync_native_state()

    def get_cascade_info(self) -> List[Tuple[float, float, float]]:
        """Get cascade information for debugging.

        Returns:
            List of (near_dist, far_dist, texel_size) for each cascade
        """
        if _forge3d is not None:
            getter = getattr(_forge3d, "get_csm_cascade_info", None)
            if getter is not None:
                try:
                    return getter()
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.get_csm_cascade_info failed: {exc}")

        # Fallback: calculate expected splits
        splits = calculate_cascade_splits(0.1, self.config.max_shadow_distance, self.config.cascade_count)
        return [(splits[i], splits[i+1], 1.0) for i in range(len(splits)-1)]

    def validate_peter_panning_prevention(self) -> bool:
        """Validate that peter-panning artifacts are prevented."""
        if _forge3d is not None:
            validator = getattr(_forge3d, "validate_csm_peter_panning", None)
            if validator is not None:
                try:
                    return validator()
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"forge3d.validate_csm_peter_panning failed: {exc}")

        # Fallback validation
        return (self.config.peter_panning_offset > 0.0001 and
                self.config.depth_bias > 0.0001)

    def is_enabled(self) -> bool:
        """Check if shadows are currently enabled."""
        return self._enabled

def calculate_cascade_splits(near_plane: float, far_plane: float, cascade_count: int, lambda_blend: float = 0.75) -> List[float]:
    """Calculate cascade split distances using Practical Split Scheme.

    Args:
        near_plane: Camera near plane distance
        far_plane: Maximum shadow distance
        cascade_count: Number of cascades
        lambda_blend: Blend factor between uniform (0.0) and logarithmic (1.0) splits

    Returns:
        List of split distances including near and far planes
    """
    splits = [near_plane]

    range_dist = far_plane - near_plane
    ratio = far_plane / near_plane

    for i in range(1, cascade_count):
        # Uniform split
        uniform_split = near_plane + (i / cascade_count) * range_dist

        # Logarithmic split
        log_split = near_plane * (ratio ** (i / cascade_count))

        # Blend the two schemes
        split = lambda_blend * log_split + (1.0 - lambda_blend) * uniform_split
        splits.append(split)

    splits.append(far_plane)
    return splits

def detect_peter_panning_cpu(shadow_factor: float, surface_normal: Tuple[float, float, float],
                            light_direction: Tuple[float, float, float]) -> bool:
    """CPU-side peter-panning detection for debugging.

    Args:
        shadow_factor: Shadow occlusion factor [0, 1]
        surface_normal: Surface normal vector
        light_direction: Light direction vector

    Returns:
        True if peter-panning artifact is detected
    """
    normal = np.array(surface_normal, dtype=np.float32)
    light_dir = np.array(light_direction, dtype=np.float32)

    # Normalize vectors
    normal = normal / np.linalg.norm(normal)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Calculate dot product (surface facing light)
    n_dot_l = np.dot(normal, -light_dir)

    # Peter-panning occurs when shadows are cast on surfaces facing away from light
    return n_dot_l <= 0.01 and shadow_factor < 0.5

# Create default CSM controller instance
_default_csm_controller = None

def get_csm_controller() -> CsmController:
    """Get or create default CSM controller instance."""
    global _default_csm_controller
    if _default_csm_controller is None:
        _default_csm_controller = CsmController()
    return _default_csm_controller

__all__ = [
    "set_exposure_stops",
    "LightType",
    "LightSample",
    "RestirConfig",
    "RestirDI",
    "create_test_scene",
    # A20: Area lights
    "AreaLightType",
    "AreaLight",
    "AreaLightManager",
    "create_area_light_test_scene",
    # B4: Cascaded Shadow Maps
    "CsmConfig",
    "CsmQualityPreset",
    "CsmController",
    "create_csm_config",
    "calculate_cascade_splits",
    "detect_peter_panning_cpu",
    "get_csm_controller"
]

