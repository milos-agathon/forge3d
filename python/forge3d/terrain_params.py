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
    # Optional PCSS light radius (world units). Defaults to hard shadows when zero.
    pcss_light_radius: float = 0.0

    def __post_init__(self) -> None:
        # P0: "NONE" disables shadows entirely
        valid = {"NONE", "PCSS", "ESM", "EVSM", "PCF", "CSM"}
        if self.technique not in valid:
            raise ValueError(f"Invalid technique: {self.technique}")
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


@dataclass
class FogSettings:
    """P2: Atmospheric fog configuration.
    
    Height-based exponential fog applied after PBR, before tonemap.
    When density = 0.0, fog is disabled (no-op for P1 compatibility).
    
    base_height: World-space Z coordinate below which fog is at full density.
                 Should be set to the minimum terrain elevation (in world units).
                 If None, will be auto-computed from terrain bounds.
    """

    density: float = 0.0  # 0.0 = disabled
    height_falloff: float = 0.0
    base_height: Optional[float] = None  # None = auto from terrain min height
    inscatter: Tuple[float, float, float] = (1.0, 1.0, 1.0)

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
    # P2: Atmospheric fog (defaults to disabled for P1 compatibility)
    fog: Optional[FogSettings] = None
    # P4: Water planar reflections (defaults to disabled for P3 compatibility)
    reflection: Optional[ReflectionSettings] = None
    # P5: AO weight/multiplier (0.0 = no AO effect, 1.0 = full AO). Default 0.0 for P4 compatibility.
    ao_weight: float = 0.0

    def __post_init__(self) -> None:
        # Default fog to disabled if not provided
        if self.fog is None:
            self.fog = FogSettings()
        # Default reflection to disabled if not provided
        if self.reflection is None:
            self.reflection = ReflectionSettings()
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

        # P5: Validate ao_weight
        if not 0.0 <= self.ao_weight <= 1.0:
            raise ValueError("ao_weight must be 0.0-1.0")


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
    height_curve_mode: str = "linear",
    height_curve_strength: float = 0.0,
    height_curve_power: float = 1.0,
    height_curve_lut: Optional[np.ndarray] = None,
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
            ambient_range=(0.0, 1.0),
            shadow_range=(0.0, 1.0),
            occlusion_range=(0.0, 1.0),
        )

    if overlays is None:
        overlays = []

    return TerrainRenderParams(
        size_px=size_px,
        render_scale=render_scale,
        msaa_samples=msaa_samples,
        z_scale=z_scale,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=float(cam_radius),
        cam_phi_deg=float(cam_phi_deg),
        cam_theta_deg=float(cam_theta_deg),
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
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
        fog=fog,
        reflection=reflection,
        ao_weight=ao_weight,
    )


__all__ = [
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "FogSettings",
    "ReflectionSettings",
    "TriplanarSettings",
    "PomSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "TerrainRenderParams",
]
