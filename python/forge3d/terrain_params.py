# python/forge3d/terrain_params.py
# Typed dataclasses describing terrain renderer configuration
# Exists to gather all tunable terrain parameters in one validated place
# RELEVANT FILES: python/forge3d/__init__.py, tests/test_terrain_params.py, src/session.rs, src/colormap1d.rs
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class LightSettings:
    """Directional, point, or spot light configuration."""

    light_type: str  # "Directional", "Point", "Spot"
    azimuth_deg: float
    elevation_deg: float
    intensity: float
    color: List[float]  # [R, G, B]

    def __post_init__(self) -> None:
        valid_types = {"Directional", "Point", "Spot"}
        if self.light_type not in valid_types:
            raise ValueError(f"Invalid light_type: {self.light_type}")

        if len(self.color) != 3:
            raise ValueError("color must be [R, G, B]")

        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")


@dataclass
class IblSettings:
    """Image based lighting configuration."""

    enabled: bool
    intensity: float
    rotation_deg: float

    def __post_init__(self) -> None:
        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")


@dataclass
class ShadowSettings:
    """Shadow mapping configuration."""

    enabled: bool
    technique: str  # "PCSS", "ESM", "EVSM", "PCF"
    resolution: int
    cascades: int
    max_distance: float
    softness: float
    intensity: float
    slope_scale_bias: float
    depth_bias: float
    normal_bias: float
    min_variance: float
    light_bleed_reduction: float
    evsm_exponent: float
    fade_start: float

    def __post_init__(self) -> None:
        valid = {"PCSS", "ESM", "EVSM", "PCF"}
        if self.technique not in valid:
            raise ValueError(f"Invalid technique: {self.technique}")

        valid_resolutions = {512, 1024, 2048, 4096, 8192}
        if self.resolution not in valid_resolutions:
            raise ValueError("resolution must be power of 2 between 512-8192")

        if not 1 <= self.cascades <= 4:
            raise ValueError("cascades must be 1-4")

        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")

        if self.softness < 0.0:
            raise ValueError("softness must be >= 0")

        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")

        if self.min_variance < 0.0:
            raise ValueError("min_variance must be >= 0")

        if self.light_bleed_reduction < 0.0:
            raise ValueError("light_bleed_reduction must be >= 0")

        if self.evsm_exponent <= 0.0:
            raise ValueError("evsm_exponent must be > 0")


@dataclass
class TriplanarSettings:
    """Triplanar texture mapping configuration."""

    scale: float
    blend_sharpness: float
    normal_strength: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be > 0")

        if self.blend_sharpness <= 0.0:
            raise ValueError("blend_sharpness must be > 0")

        if self.normal_strength < 0.0:
            raise ValueError("normal_strength must be >= 0")


@dataclass
class PomSettings:
    """Parallax occlusion mapping configuration."""

    enabled: bool
    mode: str  # "Occlusion", "Relief", "Parallax"
    scale: float
    min_steps: int
    max_steps: int
    refine_steps: int
    shadow: bool
    occlusion: bool

    def __post_init__(self) -> None:
        valid_modes = {"Occlusion", "Relief", "Parallax"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.scale < 0.0:
            raise ValueError("scale must be >= 0")

        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")

        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")

        if self.max_steps > 100:
            raise ValueError("max_steps must be <= 100")

        if self.refine_steps < 0:
            raise ValueError("refine_steps must be >= 0")


@dataclass
class LodSettings:
    """Level of detail configuration."""

    level: int
    bias: float
    lod0_bias: float

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("level must be >= 0")


@dataclass
class SamplingSettings:
    """Texture sampling configuration."""

    mag_filter: str  # "Linear", "Nearest"
    min_filter: str
    mip_filter: str
    anisotropy: int
    address_u: str  # "Repeat", "ClampToEdge", "MirrorRepeat"
    address_v: str
    address_w: str

    def __post_init__(self) -> None:
        valid_filters = {"Linear", "Nearest"}
        if self.mag_filter not in valid_filters:
            raise ValueError(f"Invalid mag_filter: {self.mag_filter}")

        if self.min_filter not in valid_filters:
            raise ValueError(f"Invalid min_filter: {self.min_filter}")

        if self.mip_filter not in valid_filters:
            raise ValueError(f"Invalid mip_filter: {self.mip_filter}")

        valid_address = {"Repeat", "ClampToEdge", "MirrorRepeat"}
        for name, value in [
            ("address_u", self.address_u),
            ("address_v", self.address_v),
            ("address_w", self.address_w),
        ]:
            if value not in valid_address:
                raise ValueError(f"Invalid {name}: {value}")

        if not 1 <= self.anisotropy <= 16:
            raise ValueError("anisotropy must be 1-16")


@dataclass
class ClampSettings:
    """Value clamping configuration."""

    height_range: Tuple[float, float]
    slope_range: Tuple[float, float]
    ambient_range: Tuple[float, float]
    shadow_range: Tuple[float, float]
    occlusion_range: Tuple[float, float]

    def __post_init__(self) -> None:
        for name, (min_val, max_val) in [
            ("height_range", self.height_range),
            ("slope_range", self.slope_range),
            ("ambient_range", self.ambient_range),
            ("shadow_range", self.shadow_range),
            ("occlusion_range", self.occlusion_range),
        ]:
            if min_val >= max_val:
                raise ValueError(f"{name}: min must be < max")


@dataclass
class TerrainRenderParams:
    """Master terrain rendering parameter container."""

    size_px: Tuple[int, int]
    render_scale: float
    msaa_samples: int
    z_scale: float
    cam_target: List[float]
    cam_radius: float
    cam_phi_deg: float
    cam_theta_deg: float
    cam_gamma_deg: float
    fov_y_deg: float
    clip: Tuple[float, float]
    light: LightSettings
    ibl: IblSettings
    shadows: ShadowSettings
    triplanar: TriplanarSettings
    pom: PomSettings
    lod: LodSettings
    sampling: SamplingSettings
    clamp: ClampSettings
    overlays: List  # forward reference to overlay types
    exposure: float
    gamma: float
    albedo_mode: str
    colormap_strength: float
    height_curve_mode: str = "linear"
    height_curve_strength: float = 0.0
    height_curve_power: float = 1.0
    height_curve_lut: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        width, height = self.size_px
        if width < 64 or height < 64:
            raise ValueError("size_px must be >= 64x64")

        if width > 8192 or height > 8192:
            raise ValueError("size_px must be <= 8192x8192")

        if not 0.25 <= self.render_scale <= 4.0:
            raise ValueError("render_scale must be 0.25-4.0")

        if self.msaa_samples not in {1, 2, 4, 8}:
            raise ValueError("msaa_samples must be 1, 2, 4, or 8")

        if not 0.1 <= self.z_scale <= 10.0:
            raise ValueError("z_scale must be 0.1-10.0")

        if len(self.cam_target) != 3:
            raise ValueError("cam_target must be [x, y, z]")

        if self.cam_radius <= 0.0:
            raise ValueError("cam_radius must be > 0")

        if not 0.0 <= self.fov_y_deg <= 180.0:
            raise ValueError("fov_y_deg must be 0-180")

        near, far = self.clip
        if near <= 0.0 or near >= far:
            raise ValueError("Invalid clip planes")

        if self.albedo_mode not in {"colormap", "mix", "material"}:
            raise ValueError(f"Invalid albedo_mode: {self.albedo_mode}")

        if not 0.0 <= self.colormap_strength <= 1.0:
            raise ValueError("colormap_strength must be 0-1")

        valid_curve_modes = {"linear", "pow", "smoothstep", "lut"}
        if self.height_curve_mode not in valid_curve_modes:
            raise ValueError(
                f"height_curve_mode must be one of {sorted(valid_curve_modes)}, "
                f"got {self.height_curve_mode}"
            )

        if not 0.0 <= self.height_curve_strength <= 1.0:
            raise ValueError("height_curve_strength must be in [0, 1]")

        if self.height_curve_power <= 0.0:
            raise ValueError("height_curve_power must be > 0")

        if self.height_curve_mode == "lut":
            if self.height_curve_lut is None:
                raise ValueError("height_curve_lut is required when height_curve_mode='lut'")

            lut = np.asarray(self.height_curve_lut, dtype=np.float32)
            if lut.ndim != 1 or lut.shape[0] != 256:
                raise ValueError("height_curve_lut must be a 1D float32 array of length 256")
            if not np.isfinite(lut).all():
                raise ValueError("height_curve_lut must contain finite values")
            if np.any(lut < 0.0) or np.any(lut > 1.0):
                raise ValueError("height_curve_lut values must be within [0, 1]")

            # Store normalized LUT back on the instance for downstream consumption
            self.height_curve_lut = lut


__all__ = [
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "TriplanarSettings",
    "PomSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "TerrainRenderParams",
]
