# python/forge3d/terrain_params.py
# Typed dataclasses describing terrain renderer configuration
# Exists to gather all tunable terrain parameters in one validated place
# RELEVANT FILES: python/forge3d/__init__.py, tests/test_terrain_params.py, src/session.rs, src/colormap1d.rs
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

import numpy as np
from pathlib import Path


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
    technique: str  # "HARD", "PCF", "PCSS", "CSM" (terrain-supported); VSM/EVSM/MSM not implemented
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
    # Optional PCSS light radius (world units). Defaults to hard shadows when zero.
    pcss_light_radius: float = 0.0

    # Shadow technique constants matching Rust ShadowTechnique enum
    # ALL_TECHNIQUES: Full set recognized by config layer (for forward compatibility)
    ALL_TECHNIQUES = {"NONE", "HARD", "PCF", "PCSS", "VSM", "EVSM", "MSM"}
    # TERRAIN_SUPPORTED_TECHNIQUES: Actually implemented in terrain_pbr_pom.wgsl
    # VSM/EVSM/MSM require moment-map sampling which is NOT implemented in terrain shader
    # Note: CSM is the pipeline, not a filter - use HARD/PCF/PCSS as the technique
    TERRAIN_SUPPORTED_TECHNIQUES = {"NONE", "HARD", "PCF", "PCSS"}
    # Alias for backwards compatibility
    SUPPORTED_TECHNIQUES = ALL_TECHNIQUES
    # Memory budget: 512 MiB host-visible heap (AGENTS.md constraint)
    MAX_SHADOW_MEMORY_BYTES = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        # Normalize technique to uppercase for consistent validation
        self.technique = self.technique.upper()
        
        # Validate technique against full set first (catch typos)
        if self.technique not in self.ALL_TECHNIQUES:
            supported_list = ", ".join(sorted(self.ALL_TECHNIQUES - {"NONE"}))
            raise ValueError(
                f"Unsupported shadow technique: {self.technique!r}. "
                f"Supported techniques: {supported_list}. "
                f"Use 'NONE' to disable shadows."
            )
        
        # When technique is NONE, shadows are disabled
        if self.technique == "NONE":
            self.enabled = False

        valid_resolutions = {512, 1024, 2048, 4096, 8192}
        if self.resolution not in valid_resolutions:
            raise ValueError("resolution must be power of 2 between 512-8192")

        if not 1 <= self.cascades <= 4:
            raise ValueError("cascades must be 1-4")

        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")

        if self.softness < 0.0:
            raise ValueError("softness must be >= 0")

        if self.pcss_light_radius < 0.0:
            raise ValueError("pcss_light_radius must be >= 0")

        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")

        if self.min_variance < 0.0:
            raise ValueError("min_variance must be >= 0")

        if self.light_bleed_reduction < 0.0:
            raise ValueError("light_bleed_reduction must be >= 0")

        if self.evsm_exponent <= 0.0:
            raise ValueError("evsm_exponent must be > 0")

        # Memory budget check (AGENTS.md: ≤512 MiB host-visible heap)
        mem_bytes = self._estimate_memory_bytes()
        if mem_bytes > self.MAX_SHADOW_MEMORY_BYTES:
            mem_mib = mem_bytes / (1024 * 1024)
            max_mib = self.MAX_SHADOW_MEMORY_BYTES / (1024 * 1024)
            raise ValueError(
                f"Shadow resources exceed memory budget: {mem_mib:.1f} MiB > {max_mib:.0f} MiB. "
                f"Reduce resolution ({self.resolution}) or cascades ({self.cascades})."
            )

    def _estimate_memory_bytes(self) -> int:
        """Estimate GPU memory for shadow resources."""
        pixels = self.resolution * self.resolution * self.cascades
        depth_bytes = pixels * 4  # Depth32Float
        # Moment maps for VSM/EVSM/MSM techniques
        if self.technique == "VSM":
            moment_bytes = pixels * 8  # 2 channels * 4 bytes
        elif self.technique in {"EVSM", "MSM"}:
            moment_bytes = pixels * 16  # 4 channels * 4 bytes
        else:
            moment_bytes = 0
        return depth_bytes + moment_bytes

    def validate_for_terrain(self) -> None:
        """Validate that this shadow technique is implemented in the terrain pipeline.
        
        Raises ValueError with a clear message if the technique is not supported.
        VSM/EVSM/MSM are recognized by the config layer but NOT implemented in
        terrain_pbr_pom.wgsl (moment_maps binding exists but is never sampled).
        """
        if self.technique not in self.TERRAIN_SUPPORTED_TECHNIQUES:
            unsupported = {"VSM", "EVSM", "MSM"}
            terrain_list = ", ".join(sorted(self.TERRAIN_SUPPORTED_TECHNIQUES - {"NONE"}))
            raise ValueError(
                f"Shadow technique {self.technique!r} is not implemented for terrain rendering. "
                f"The terrain shader does not support variance/moment-based shadows (VSM/EVSM/MSM). "
                f"Terrain-supported techniques: {terrain_list}. "
                f"Use 'NONE' to disable shadows."
            )


@dataclass
class FogSettings:
    """P2: Atmospheric fog configuration.
    
    Height-based exponential fog applied after PBR, before tonemap.
    When density = 0.0, fog is disabled (no-op for P1 compatibility).
    
    base_height: World-space Z coordinate below which fog is at full density.
                 Should be set to the minimum terrain elevation (in world units).
                 If None, will be auto-computed from terrain bounds.
    
    aerial_perspective: M3 feature - distance-based desaturation and blue shift
                       simulating Rayleigh scattering. 0.0 = disabled, 1.0 = full effect.
    """

    density: float = 0.0  # 0.0 = disabled
    height_falloff: float = 0.0
    base_height: Optional[float] = None  # None = auto from terrain min height
    inscatter: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    aerial_perspective: float = 0.0  # M3: 0.0 = disabled, 1.0 = full effect

    def __post_init__(self) -> None:
        if self.density < 0.0:
            raise ValueError("density must be >= 0")
        if self.height_falloff < 0.0:
            raise ValueError("height_falloff must be >= 0")
        if len(self.inscatter) != 3:
            raise ValueError("inscatter must be (R, G, B)")
        for c in self.inscatter:
            if not 0.0 <= c <= 1.0:
                raise ValueError("inscatter components must be in [0, 1]")
        if not 0.0 <= self.aerial_perspective <= 1.0:
            raise ValueError("aerial_perspective must be in [0, 1]")


@dataclass
class ReflectionSettings:
    """P4: Water planar reflection configuration.
    
    When enabled=False, reflections are disabled (no-op for P3 compatibility).
    Reflections sample a half-resolution render of the scene mirrored across
    the water plane, with wave-based UV distortion and Fresnel mixing.
    """

    enabled: bool = False  # Disabled by default (P3 compatibility)
    intensity: float = 0.8  # Reflection intensity (0.0-1.0)
    fresnel_power: float = 5.0  # Fresnel falloff exponent
    wave_strength: float = 0.02  # Wave-based UV distortion strength
    shore_atten_width: float = 0.3  # Reduce reflections near land
    water_plane_height: float = 0.0  # Water plane height in world space

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be in [0, 1]")
        if self.fresnel_power < 0.0:
            raise ValueError("fresnel_power must be >= 0")
        if self.wave_strength < 0.0:
            raise ValueError("wave_strength must be >= 0")
        if self.shore_atten_width < 0.0:
            raise ValueError("shore_atten_width must be >= 0")


@dataclass
class BloomSettings:
    """M2: Bloom post-processing configuration.
    
    When enabled=False, bloom is disabled (identical output for backward compatibility).
    Bloom extracts bright pixels above threshold and applies Gaussian blur,
    then composites the result back onto the original image.
    """

    enabled: bool = False  # Disabled by default for backward compatibility
    threshold: float = 1.5  # Brightness threshold (1.5 = HDR only)
    softness: float = 0.5  # Threshold transition softness (0.0-1.0)
    intensity: float = 0.3  # Bloom intensity when compositing
    radius: float = 1.0  # Blur radius multiplier

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        if not 0.0 <= self.softness <= 1.0:
            raise ValueError("softness must be in [0, 1]")
        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")


@dataclass
class HeightAoSettings:
    """Heightfield ray-traced ambient occlusion configuration.
    
    Computes AO by ray-marching the heightfield in multiple directions.
    When enabled=False, AO is disabled (default for backward compatibility).
    """

    enabled: bool = False
    resolution_scale: float = 0.5  # Render at half resolution for performance
    directions: int = 6  # Number of horizon directions to sample
    steps: int = 16  # Steps per direction
    max_distance: float = 200.0  # Max ray distance in world units
    strength: float = 1.0  # AO intensity multiplier
    blur: bool = False  # Optional bilateral blur

    def __post_init__(self) -> None:
        if not 0.1 <= self.resolution_scale <= 1.0:
            raise ValueError("resolution_scale must be in [0.1, 1.0]")
        if not 1 <= self.directions <= 16:
            raise ValueError("directions must be in [1, 16]")
        if not 1 <= self.steps <= 64:
            raise ValueError("steps must be in [1, 64]")
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("strength must be in [0.0, 2.0]")


@dataclass
class SunVisibilitySettings:
    """Heightfield ray-traced sun visibility / soft shadows configuration.
    
    Computes sun visibility by ray-marching toward the sun direction.
    When enabled=False, sun visibility is disabled (default for backward compatibility).
    """

    enabled: bool = False
    mode: str = "hard"  # "hard" or "soft"
    resolution_scale: float = 0.5  # Render at half resolution for performance
    samples: int = 4  # Number of jittered samples for soft shadows
    steps: int = 24  # Steps per ray
    max_distance: float = 400.0  # Max ray distance in world units
    softness: float = 1.0  # Penumbra softness multiplier
    bias: float = 0.01  # Self-shadowing bias

    def __post_init__(self) -> None:
        valid_modes = {"hard", "soft"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode!r}")
        if not 0.1 <= self.resolution_scale <= 1.0:
            raise ValueError("resolution_scale must be in [0.1, 1.0]")
        if not 1 <= self.samples <= 16:
            raise ValueError("samples must be in [1, 16]")
        if not 1 <= self.steps <= 64:
            raise ValueError("steps must be in [1, 64]")
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        if self.softness < 0.0:
            raise ValueError("softness must be >= 0")
        if self.bias < 0.0:
            raise ValueError("bias must be >= 0")


@dataclass
class DetailSettings:
    """P6: Micro-detail configuration for close-range surface enhancement.
    
    When enabled=False, micro-detail is disabled (no-op for P5 compatibility).
    Adds triplanar detail normals and procedural albedo noise that fade
    with distance to prevent LOD popping and shimmer.
    
    P6 Gradient Match extension:
    - detail_normal_path: Path to DEM-derived detail normal texture (optional).
      When provided, samples from texture instead of procedural normals.
    - detail_sigma_px: Gaussian sigma used to generate detail normals (for metadata).
    - detail_strength: Blending strength for DEM-derived detail normals (0.0-1.0).
    
    detail_scale: World-space repeat interval for detail normals (default 2.0 meters).
    normal_strength: Strength of detail normal perturbation (0.0-1.0).
    albedo_noise: Brightness variation amplitude (±percentage, e.g., 0.1 = ±10%).
    fade_start: Distance at which detail begins fading (world units).
    fade_end: Distance at which detail is fully faded out (world units).
    """

    enabled: bool = False  # Disabled by default (P5 compatibility)
    detail_scale: float = 2.0  # 2 meter repeat interval
    normal_strength: float = 0.3  # Detail normal blending strength
    albedo_noise: float = 0.1  # ±10% brightness variation
    fade_start: float = 50.0  # Start fading at 50 units
    fade_end: float = 200.0  # Fully faded at 200 units
    # P6 Gradient Match: DEM-derived detail normal map
    detail_normal_path: Optional[str] = None  # Path to detail normal texture
    detail_sigma_px: float = 3.0  # Gaussian sigma used to generate detail normals
    detail_strength: float = 0.0  # DEM-derived detail normal strength (0=off)

    def __post_init__(self) -> None:
        if self.detail_scale <= 0.0:
            raise ValueError("detail_scale must be > 0")
        if not 0.0 <= self.normal_strength <= 1.0:
            raise ValueError("normal_strength must be in [0, 1]")
        if not 0.0 <= self.albedo_noise <= 0.5:
            raise ValueError("albedo_noise must be in [0, 0.5]")
        if self.fade_start < 0.0:
            raise ValueError("fade_start must be >= 0")
        if self.fade_end <= self.fade_start:
            raise ValueError("fade_end must be > fade_start")
        if self.detail_sigma_px <= 0.0:
            raise ValueError("detail_sigma_px must be > 0")
        if not 0.0 <= self.detail_strength <= 1.0:
            raise ValueError("detail_strength must be in [0, 1]")


@dataclass
class MaterialLayerSettings:
    """M4: Terrain material layering configuration.
    
    Provides slope/aspect/altitude-driven material blending for realistic terrain:
    - Snow: deposits on high-altitude, low-slope areas (south-facing receives less)
    - Rock: exposed on steep slopes (>45°)
    - Wetness: darkening in concave areas (placeholder: based on slope curvature)
    
    When all layers are disabled (default), output is identical to baseline.
    """
    
    # Snow layer settings
    snow_enabled: bool = False
    snow_altitude_min: float = 2000.0  # Minimum altitude for snow (world units)
    snow_altitude_blend: float = 500.0  # Altitude blend range
    snow_slope_max: float = 45.0  # Maximum slope angle (degrees) for snow
    snow_slope_blend: float = 15.0  # Slope blend range (degrees)
    snow_aspect_influence: float = 0.3  # 0=no aspect effect, 1=full (south-facing less snow)
    snow_color: Tuple[float, float, float] = (0.95, 0.95, 0.98)  # Snow albedo
    snow_roughness: float = 0.4  # Snow surface roughness
    
    # Rock layer settings
    rock_enabled: bool = False
    rock_slope_min: float = 45.0  # Minimum slope angle (degrees) for rock exposure
    rock_slope_blend: float = 10.0  # Slope blend range (degrees)
    rock_color: Tuple[float, float, float] = (0.35, 0.32, 0.28)  # Rock albedo
    rock_roughness: float = 0.8  # Rock surface roughness
    
    # Wetness layer settings (darkening in concave areas)
    wetness_enabled: bool = False
    wetness_strength: float = 0.3  # Darkening strength (0-1)
    wetness_slope_influence: float = 0.5  # How much slope affects wetness

    def __post_init__(self) -> None:
        if self.snow_altitude_blend <= 0.0:
            raise ValueError("snow_altitude_blend must be > 0")
        if not 0.0 <= self.snow_slope_max <= 90.0:
            raise ValueError("snow_slope_max must be in [0, 90]")
        if self.snow_slope_blend <= 0.0:
            raise ValueError("snow_slope_blend must be > 0")
        if not 0.0 <= self.snow_aspect_influence <= 1.0:
            raise ValueError("snow_aspect_influence must be in [0, 1]")
        if len(self.snow_color) != 3:
            raise ValueError("snow_color must be (R, G, B)")
        if not 0.0 <= self.snow_roughness <= 1.0:
            raise ValueError("snow_roughness must be in [0, 1]")
        
        if not 0.0 <= self.rock_slope_min <= 90.0:
            raise ValueError("rock_slope_min must be in [0, 90]")
        if self.rock_slope_blend <= 0.0:
            raise ValueError("rock_slope_blend must be > 0")
        if len(self.rock_color) != 3:
            raise ValueError("rock_color must be (R, G, B)")
        if not 0.0 <= self.rock_roughness <= 1.0:
            raise ValueError("rock_roughness must be in [0, 1]")
        
        if not 0.0 <= self.wetness_strength <= 1.0:
            raise ValueError("wetness_strength must be in [0, 1]")
        if not 0.0 <= self.wetness_slope_influence <= 1.0:
            raise ValueError("wetness_slope_influence must be in [0, 1]")


@dataclass
class VectorOverlaySettings:
    """M5: Vector overlay configuration for depth-correct rendering and halos.
    
    Controls how vector overlays (lines, polygons) interact with terrain:
    - depth_test: When True, vectors are occluded by terrain ridges
    - halo: Adds outline/shadow for improved readability over terrain
    
    When depth_test=False (default), output is identical to baseline.
    """
    
    # Depth testing
    depth_test: bool = False  # When True, vectors hidden behind terrain
    depth_bias: float = 0.001  # Depth offset to prevent z-fighting (smaller = closer)
    depth_bias_slope: float = 1.0  # Slope-scaled bias for grazing angles
    
    # Halo/outline for readability
    halo_enabled: bool = False
    halo_width: float = 2.0  # Halo width in pixels
    halo_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)  # RGBA
    halo_blur: float = 1.0  # Blur/softness of halo edge
    
    # Contour rendering (ink-like effect)
    contour_enabled: bool = False
    contour_width: float = 1.0  # Contour line width in pixels
    contour_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.8)

    def __post_init__(self) -> None:
        if self.depth_bias < 0.0:
            raise ValueError("depth_bias must be >= 0")
        if self.depth_bias_slope < 0.0:
            raise ValueError("depth_bias_slope must be >= 0")
        if self.halo_width < 0.0:
            raise ValueError("halo_width must be >= 0")
        if len(self.halo_color) != 4:
            raise ValueError("halo_color must be (R, G, B, A)")
        if self.halo_blur < 0.0:
            raise ValueError("halo_blur must be >= 0")
        if self.contour_width < 0.0:
            raise ValueError("contour_width must be >= 0")
        if len(self.contour_color) != 4:
            raise ValueError("contour_color must be (R, G, B, A)")


@dataclass
class TonemapSettings:
    """M6: Tonemap configuration for HDR to SDR conversion.
    
    Controls tone mapping operator selection, 3D LUT application, and white balance.
    
    Operators:
    - 'reinhard': Simple Reinhard (default)
    - 'reinhard_extended': Extended Reinhard with white point
    - 'aces': ACES filmic (cinematic look)
    - 'uncharted2': Uncharted 2 filmic
    - 'exposure': Simple exposure mapping
    
    White balance uses temperature (Kelvin) and tint (green-magenta).
    """
    
    # Tonemap operator selection
    operator: str = "aces"  # reinhard, reinhard_extended, aces, uncharted2, exposure
    
    # White point for extended operators
    white_point: float = 4.0
    
    # 3D LUT support (cube format)
    lut_enabled: bool = False
    lut_path: Optional[str] = None  # Path to .cube LUT file
    lut_strength: float = 1.0  # Blend strength 0-1
    
    # White balance (temperature/tint)
    white_balance_enabled: bool = False
    temperature: float = 6500.0  # Color temperature in Kelvin (2000-12000)
    tint: float = 0.0  # Green-magenta tint (-1.0 to 1.0)

    def __post_init__(self) -> None:
        valid_operators = {"reinhard", "reinhard_extended", "aces", "uncharted2", "exposure"}
        if self.operator not in valid_operators:
            raise ValueError(f"operator must be one of {valid_operators}, got '{self.operator}'")
        if self.white_point <= 0.0:
            raise ValueError("white_point must be > 0")
        if self.lut_strength < 0.0 or self.lut_strength > 1.0:
            raise ValueError("lut_strength must be in range [0, 1]")
        if self.temperature < 2000.0 or self.temperature > 12000.0:
            raise ValueError("temperature must be in range [2000, 12000] Kelvin")
        if self.tint < -1.0 or self.tint > 1.0:
            raise ValueError("tint must be in range [-1, 1]")


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
    # Physical span of the terrain in world units. Used to scale UVs to world XY.
    terrain_span: float
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
    # P5-L: Lambert contrast parameter [0,1] for gradient enhancement
    lambert_contrast: float = 0.0
    # P2: Atmospheric fog (defaults to disabled for P1 compatibility)
    fog: Optional[FogSettings] = None
    # P4: Water planar reflections (defaults to disabled for P3 compatibility)
    reflection: Optional[ReflectionSettings] = None
    # P5: AO weight/multiplier (0.0 = no AO effect, 1.0 = full AO). Default 0.0 for P4 compatibility.
    ao_weight: float = 0.0
    # P6: Micro-detail (defaults to disabled for P5 compatibility)
    detail: Optional[DetailSettings] = None
    # Heightfield ray-traced AO (defaults to disabled for backward compatibility)
    height_ao: Optional[HeightAoSettings] = None
    # Heightfield ray-traced sun visibility (defaults to disabled for backward compatibility)
    sun_visibility: Optional[SunVisibilitySettings] = None
    # P6.1: Color space correctness toggles (defaults to False for P5 compatibility)
    colormap_srgb: bool = False  # Use Rgba8UnormSrgb for colormap texture (correct sampling)
    output_srgb_eotf: bool = False  # Use exact linear_to_srgb() instead of pow-gamma
    # P7: Camera projection mode ("screen" = fullscreen triangle, "mesh" = perspective grid)
    camera_mode: str = "screen"
    # P7: Debug mode for projection probes (0=normal, 40=view-depth, 41=NDC depth, 42=view-pos XYZ)
    debug_mode: int = 0
    # M1: Accumulation AA sample count (1 = no AA, 16/64/256 typical for offline)
    aa_samples: int = 1
    # M1: Accumulation AA seed for deterministic jitter (None = default sequence)
    aa_seed: Optional[int] = None
    # M2: Bloom post-processing (defaults to disabled for backward compatibility)
    bloom: Optional[BloomSettings] = None
    # M4: Material layering (snow/rock/wetness, defaults to disabled for backward compatibility)
    materials: Optional[MaterialLayerSettings] = None
    # M5: Vector overlay settings (depth test, halos)
    vector_overlay: Optional[VectorOverlaySettings] = None
    # M6: Tonemap settings (operator, LUT, white balance)
    tonemap: Optional[TonemapSettings] = None

    def __post_init__(self) -> None:
        # Default fog to disabled if not provided
        if self.fog is None:
            self.fog = FogSettings()
        # Default reflection to disabled if not provided
        if self.reflection is None:
            self.reflection = ReflectionSettings()
        # Default detail to disabled if not provided
        if self.detail is None:
            self.detail = DetailSettings()
        # Default height_ao to disabled if not provided
        if self.height_ao is None:
            self.height_ao = HeightAoSettings()
        # Default sun_visibility to disabled if not provided
        if self.sun_visibility is None:
            self.sun_visibility = SunVisibilitySettings()
        # M2: Default bloom to disabled if not provided
        if self.bloom is None:
            self.bloom = BloomSettings()
        # M4: Default materials to disabled if not provided
        if self.materials is None:
            self.materials = MaterialLayerSettings()
        # M5: Default vector overlay to disabled if not provided
        if self.vector_overlay is None:
            self.vector_overlay = VectorOverlaySettings()
        # M6: Default tonemap to ACES if not provided
        if self.tonemap is None:
            self.tonemap = TonemapSettings()
        width, height = self.size_px
        if width < 64 or height < 64:
            raise ValueError("size_px must be >= 64x64")

        if width > 8192 or height > 8192:
            raise ValueError("size_px must be <= 8192x8192")

        if self.terrain_span <= 0.0:
            raise ValueError("terrain_span must be > 0")

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

        # P5: Validate ao_weight
        if not 0.0 <= self.ao_weight <= 1.0:
            raise ValueError("ao_weight must be 0.0-1.0")

        # M1: Validate aa_samples (must be >= 1)
        if self.aa_samples < 1:
            raise ValueError("aa_samples must be >= 1")
        if self.aa_samples > 4096:
            raise ValueError("aa_samples must be <= 4096 (practical limit for offline rendering)")


def load_height_curve_lut(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Height curve LUT not found: {p}")

    try:
        data = np.load(p)
    except Exception:
        try:
            data = np.loadtxt(p)
        except Exception as exc:
            raise ValueError(f"Failed to load height curve LUT from {p}: {exc}")

    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if data.shape[0] != 256:
        raise ValueError(f"height_curve_lut must contain 256 entries, found {data.shape[0]} in {p}")
    if not np.isfinite(data).all():
        raise ValueError("height_curve_lut must contain finite values")
    if np.any(data < 0.0) or np.any(data > 1.0):
        raise ValueError("height_curve_lut values must lie within [0, 1]")
    return data


def make_terrain_params_config(
    *,
    size_px: Tuple[int, int],
    render_scale: float,
    terrain_span: float,
    msaa_samples: int,
    z_scale: float,
    exposure: float,
    domain: Tuple[float, float],
    albedo_mode: str = "mix",
    colormap_strength: float = 0.5,
    ibl_enabled: bool = True,
    light_azimuth_deg: float = 135.0,
    light_elevation_deg: float = 35.0,
    sun_intensity: float = 3.0,
    sun_color: Optional[Sequence[float]] = None,
    ibl_intensity: float = 1.0,
    cam_radius: float = 1200.0,
    cam_phi_deg: float = 135.0,
    cam_theta_deg: float = 45.0,
    fov_y_deg: float = 55.0,
    camera_mode: str = "screen",  # "screen" (fullscreen triangle) or "mesh" (perspective grid)
    debug_mode: int = 0,  # 0=normal, 40=view-depth probe, 41=NDC depth, 42=view-pos XYZ
    clip: Optional[Tuple[float, float]] = None,
    height_curve_mode: str = "linear",
    height_curve_strength: float = 0.0,
    height_curve_power: float = 1.0,
    height_curve_lut: Optional[np.ndarray] = None,
    lambert_contrast: float = 0.0,  # P5-L: Lambert contrast [0,1]
    shadows: Optional[ShadowSettings] = None,
    triplanar: Optional[TriplanarSettings] = None,
    pom: Optional[PomSettings] = None,
    lod: Optional[LodSettings] = None,
    sampling: Optional[SamplingSettings] = None,
    clamp: Optional[ClampSettings] = None,
    overlays: Optional[list] = None,
    fog: Optional[FogSettings] = None,
    reflection: Optional[ReflectionSettings] = None,
    ao_weight: float = 0.0,
    detail: Optional[DetailSettings] = None,
    height_ao: Optional[HeightAoSettings] = None,
    sun_visibility: Optional[SunVisibilitySettings] = None,
    aa_samples: int = 1,  # M1: Accumulation AA sample count (1 = no AA)
    aa_seed: Optional[int] = None,  # M1: Accumulation AA seed for determinism
    bloom: Optional[BloomSettings] = None,  # M2: Bloom post-processing
    materials: Optional[MaterialLayerSettings] = None,  # M4: Material layering
    vector_overlay: Optional[VectorOverlaySettings] = None,  # M5: Vector overlay settings
    tonemap: Optional[TonemapSettings] = None,  # M6: Tonemap settings
) -> TerrainRenderParams:
    light_color = [1.0, 1.0, 1.0]
    if sun_color is not None:
        try:
            light_color = [
                float(sun_color[0]),
                float(sun_color[1]),
                float(sun_color[2]),
            ]
        except (TypeError, IndexError, ValueError):
            light_color = [1.0, 1.0, 1.0]
    light_intensity = float(sun_intensity)

    if shadows is None:
        # Default shadow settings with small bias values for proper depth comparison.
        # Large bias values (e.g., 0.5) cause all fragments to appear lit (no shadows).
        shadows = ShadowSettings(
            enabled=True,
            technique="PCSS",
            resolution=4096,
            cascades=3,
            max_distance=4000.0,
            softness=1.5,
            pcss_light_radius=0.0,
            intensity=0.8,
            slope_scale_bias=0.001,   # Slope-scaled bias for grazing angles
            depth_bias=0.0005,        # Base depth bias
            normal_bias=0.0002,       # Peter-panning offset (bias along normal)
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        )

    if triplanar is None:
        triplanar = TriplanarSettings(
            scale=6.0,
            blend_sharpness=4.0,
            normal_strength=1.0,
        )

    if pom is None:
        pom = PomSettings(
            enabled=True,
            mode="Occlusion",
            scale=0.04,
            min_steps=12,
            max_steps=40,
            refine_steps=4,
            shadow=True,
            occlusion=True,
        )

    if lod is None:
        lod = LodSettings(level=0, bias=0.0, lod0_bias=-0.5)

    if sampling is None:
        sampling = SamplingSettings(
            mag_filter="Linear",
            min_filter="Linear",
            mip_filter="Linear",
            anisotropy=8,
            address_u="Repeat",
            address_v="Repeat",
            address_w="Repeat",
        )

    if clamp is None:
        clamp = ClampSettings(
            height_range=(float(domain[0]), float(domain[1])),
            slope_range=(0.04, 1.0),
            ambient_range=(0.22, 0.38),  # P2-S1: ambient floor in [0.22, 0.38]
            shadow_range=(0.30, 1.0),    # P2-S3: shadow factor in [0.30, 1.0]
            occlusion_range=(0.65, 1.0), # P2-S2: AO capped at 35% darkening (min 0.65)
        )

    if overlays is None:
        overlays = []

    if clip is None:
        clip = (0.1, 6000.0)

    return TerrainRenderParams(
        size_px=size_px,
        render_scale=render_scale,
        terrain_span=float(terrain_span),
        msaa_samples=msaa_samples,
        z_scale=z_scale,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=float(cam_radius),
        cam_phi_deg=float(cam_phi_deg),
        cam_theta_deg=float(cam_theta_deg),
        cam_gamma_deg=0.0,
        fov_y_deg=float(fov_y_deg),
        clip=clip,
        light=LightSettings(
            light_type="Directional",
            azimuth_deg=float(light_azimuth_deg),
            elevation_deg=float(light_elevation_deg),
            intensity=light_intensity,
            color=light_color,
        ),
        ibl=IblSettings(
            enabled=ibl_enabled,
            intensity=float(ibl_intensity),
            rotation_deg=0.0,
        ),
        shadows=shadows,
        triplanar=triplanar,
        pom=pom,
        lod=lod,
        sampling=sampling,
        clamp=clamp,
        overlays=overlays,
        exposure=exposure,
        gamma=2.2,
        albedo_mode=albedo_mode,
        colormap_strength=colormap_strength,
        height_curve_mode=height_curve_mode,
        height_curve_strength=height_curve_strength,
        height_curve_power=height_curve_power,
        height_curve_lut=height_curve_lut,
        lambert_contrast=lambert_contrast,
        fog=fog,
        reflection=reflection,
        ao_weight=ao_weight,
        detail=detail,
        height_ao=height_ao,
        sun_visibility=sun_visibility,
        camera_mode=str(camera_mode),
        debug_mode=int(debug_mode),
        aa_samples=int(aa_samples),
        aa_seed=aa_seed,
        bloom=bloom,
        materials=materials,
        vector_overlay=vector_overlay,
        tonemap=tonemap,
    )


__all__ = [
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "FogSettings",
    "ReflectionSettings",
    "BloomSettings",
    "HeightAoSettings",
    "SunVisibilitySettings",
    "DetailSettings",
    "MaterialLayerSettings",
    "VectorOverlaySettings",
    "TonemapSettings",
    "TriplanarSettings",
    "PomSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "TerrainRenderParams",
]
