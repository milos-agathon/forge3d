# python/forge3d/lighting.py
# ReSTIR DI (Reservoir-based Spatio-Temporal Importance Resampling) Python bindings and basic media utilities

"""ReSTIR DI lighting system with basic participating media helpers (HG phase, height fog).

These media utilities exist to satisfy Workstream A A11 deliverables with minimal, testable CPU-side functionality.

RELEVANT FILES:src/shaders/lighting_media.wgsl,tests/test_media_hg.py,tests/test_media_fog.py"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
from enum import Enum

try:
    from . import _forge3d  # Rust bindings
except ImportError:
    _forge3d = None

_PI = 3.14159265358979323846


def _to_float_array(x: Union[float, np.ndarray]) -> np.ndarray:
    """Convert input to float32 numpy array without copying when possible.

    Keeps shapes intact; scalars become shape=().
    """
    arr = np.asarray(x, dtype=np.float32)
    return arr


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


__all__ = [
    "LightType",
    "LightSample",
    "RestirConfig",
    "RestirDI",
    "create_test_scene"
]
