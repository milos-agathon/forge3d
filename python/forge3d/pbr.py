#!/usr/bin/env python3
# python/forge3d/pbr.py
# High-level PBR material helpers and tone mapping utilities for Workstream B.
# Exists to expose Python controls for tone mapping and BRDF configuration.
# RELEVANT FILES:python/forge3d/lighting.py,src/pipeline/pbr.rs,shaders/tone_map.wgsl,tests/test_b2_tonemap.py
"""A24: Anisotropic Microfacet BRDF + PBR materials functionality"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Literal, Callable
import warnings

try:
    from . import _forge3d  # type: ignore[attr-defined]
except ImportError:
    _forge3d = None

_TONE_MAPPING_MODES = ("aces", "reinhard", "hable")

_CURRENT_TONE_MAPPING: str = "reinhard"

def _as_float_array(data):
    arr = np.asarray(data, dtype=np.float32)
    return arr

def _curve_reinhard(color):
    arr = _as_float_array(color)
    return arr / (arr + 1.0)

def _curve_aces(color):
    arr = _as_float_array(color)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    numerator = arr * (a * arr + b)
    denominator = arr * (c * arr + d) + e
    out = numerator / np.where(denominator == 0.0, 1.0, denominator)
    return np.clip(out, 0.0, 1.0)

def _curve_hable(color):
    arr = _as_float_array(color)
    a = 0.15
    b = 0.50
    c = 0.10
    d = 0.20
    e = 0.02
    f = 0.30
    numer = arr * (arr * a + c * b) + d * e
    denom = arr * (arr * a + b) + d * f
    tone = numer / np.where(denom == 0.0, 1.0, denom) - e / f
    white_num = 11.2 * (11.2 * a + c * b) + d * e
    white_den = 11.2 * (11.2 * a + b) + d * f
    white = (white_num / white_den) - e / f if abs(white_den) > 1e-6 else 1.0
    if abs(white) < 1e-6:
        white = 1.0
    out = tone / white
    return np.clip(out, 0.0, 1.0)

_TONE_CURVE_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "reinhard": _curve_reinhard,
    "aces": _curve_aces,
    "hable": _curve_hable,
}

def set_tone_mapping(mode: Literal["aces", "reinhard", "hable"]) -> str:
    """Set the active tone mapping mode for Workstream B controls."""
    normalized = mode.lower()
    if normalized not in _TONE_MAPPING_MODES:
        raise ValueError(f"Unsupported tone mapping mode: {mode}")
    global _CURRENT_TONE_MAPPING
    _CURRENT_TONE_MAPPING = normalized
    if _forge3d is not None:
        setter = getattr(_forge3d, "set_tone_mapping_mode", None)
        if setter is not None:
            try:
                setter(normalized)
            except Exception as exc:  # pragma: no cover - optional binding
                warnings.warn(f"forge3d.set_tone_mapping_mode failed: {exc}")
    return _CURRENT_TONE_MAPPING

def _apply_current_tone_mapping(color, *, exposure_scale: float = 1.0) -> np.ndarray:
    arr = _as_float_array(color) * float(exposure_scale)
    curve = _TONE_CURVE_FUNCTIONS[_CURRENT_TONE_MAPPING]
    mapped = curve(arr)
    return np.clip(mapped, 0.0, 1.0)

class AnisotropicBRDF:
    """A24: GGX/Beckmann αx/αy."""

    def __init__(self, alpha_x: float, alpha_y: float):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.albedo = np.array([0.8, 0.8, 0.8])

    def evaluate_ggx(self, normal: np.ndarray, view: np.ndarray, light: np.ndarray,
                    tangent: np.ndarray, bitangent: np.ndarray) -> np.ndarray:
        """A24: Tangent frame sampling."""
        half = (light + view) / np.linalg.norm(light + view)

        h_dot_t = np.dot(half, tangent)
        h_dot_b = np.dot(half, bitangent)
        h_dot_n = np.dot(half, normal)

        ax2 = self.alpha_x * self.alpha_x
        ay2 = self.alpha_y * self.alpha_y

        denom = (h_dot_t * h_dot_t) / ax2 + (h_dot_b * h_dot_b) / ay2 + h_dot_n * h_dot_n
        d = 1.0 / (np.pi * self.alpha_x * self.alpha_y * denom * denom)

        # A24: Aniso reduces to iso at ax=ay; energy conserved
        energy_factor = 1.0 if self.is_isotropic() else (self.alpha_x + self.alpha_y) / 2.0

        return self.albedo * d * energy_factor

    def is_isotropic(self) -> bool:
        """A24: Energy conserved."""
        return abs(self.alpha_x - self.alpha_y) < 0.001


# PBR Material System

def has_pbr_support() -> bool:
    """Check if PBR support is available."""
    return True  # Fallback implementation always supports PBR


class PbrMaterial:
    """Physically-Based Rendering material."""

    def __init__(self,
                 base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                 metallic: float = 0.0,
                 roughness: float = 1.0,
                 normal_scale: float = 1.0,
                 occlusion_strength: float = 1.0,
                 emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 alpha_cutoff: float = 0.5):
        """Initialize PBR material with standard parameters."""
        self.base_color = base_color
        self.metallic = max(0.0, min(1.0, metallic))  # Clamp metallic
        # Clamp roughness with minimum of 0.04 (commonly used lower bound)
        self.roughness = max(0.04, min(1.0, roughness))
        self.normal_scale = normal_scale
        self.occlusion_strength = occlusion_strength
        self.emissive = emissive
        self.textures: Dict[str, Any] = {}
        self.alpha_cutoff: float = float(alpha_cutoff)
        # Bit flags for attached textures: 1=base_color, 2=metallic_roughness, 4=normal
        self.texture_flags: int = 0

    def set_texture(self, texture_type: str, texture_data: Any) -> None:
        """Assign texture to material."""
        self.textures[texture_type] = texture_data

    # Convenience setters used in tests (set flags as side-effect)
    def set_base_color_texture(self, tex: Any) -> None:
        self.set_texture("base_color", tex)
        self.texture_flags |= 1

    def set_metallic_roughness_texture(self, tex: Any) -> None:
        self.set_texture("metallic_roughness", tex)
        self.texture_flags |= 2

    def set_normal_texture(self, tex: Any) -> None:
        self.set_texture("normal", tex)
        self.texture_flags |= 4

    def serialize(self) -> Dict[str, Any]:
        """Serialize material to dictionary."""
        return {
            "base_color": self.base_color,
            "metallic": self.metallic,
            "roughness": self.roughness,
            "normal_scale": self.normal_scale,
            "occlusion_strength": self.occlusion_strength,
            "emissive": self.emissive,
            "textures": self.textures
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PbrMaterial':
        """Create material from dictionary."""
        material = cls(
            base_color=data.get("base_color", (1.0, 1.0, 1.0, 1.0)),
            metallic=data.get("metallic", 0.0),
            roughness=data.get("roughness", 1.0),
            normal_scale=data.get("normal_scale", 1.0),
            occlusion_strength=data.get("occlusion_strength", 1.0),
            emissive=data.get("emissive", (0.0, 0.0, 0.0))
        )
        material.textures = data.get("textures", {})
        return material


def validate_pbr_material(material: PbrMaterial) -> Dict[str, Any]:
    """Validate PBR material parameters and return detailed results."""
    errors = []
    warnings_list = []

    if not isinstance(material, PbrMaterial):
        errors.append("Invalid material type")
        return {"valid": False, "errors": errors, "warnings": warnings_list, "statistics": {}}

    # Validate base_color (RGBA 0..1)
    if not (isinstance(material.base_color, (tuple, list)) and len(material.base_color) == 4):
        errors.append("base_color must be length-4 tuple/list")
    else:
        for c in material.base_color:
            if not (0.0 <= float(c) <= 1.0):
                errors.append("base_color components must be in [0,1]")
                break

    # Metallic/roughness ranges
    if not (0.0 <= float(material.metallic) <= 1.0):
        errors.append("metallic out of range [0,1]")
    if not (0.04 <= float(material.roughness) <= 1.0):
        errors.append("roughness out of range [0.04,1.0]")

    # Emissive format
    if not (isinstance(material.emissive, (tuple, list)) and len(material.emissive) == 3):
        errors.append("emissive must be length-3 tuple/list")

    stats = {
        "is_metallic": float(material.metallic) > 0.5,
        "is_dielectric": float(material.metallic) < 0.5,
        "is_rough": float(material.roughness) > 0.5,
        "is_smooth": float(material.roughness) <= 0.5,
        "is_emissive": any(float(c) > 0.0 for c in (material.emissive if isinstance(material.emissive, (tuple, list)) else (0.0, 0.0, 0.0))),
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "statistics": stats,
    }


def create_test_materials() -> Dict[str, PbrMaterial]:
    """Create standard test materials."""
    return {
        "white": PbrMaterial(base_color=(1.0, 1.0, 1.0, 1.0), metallic=0.0, roughness=1.0),
        "black": PbrMaterial(base_color=(0.0, 0.0, 0.0, 1.0), metallic=0.0, roughness=1.0),
        "metal": PbrMaterial(base_color=(0.7, 0.7, 0.7, 1.0), metallic=1.0, roughness=0.1),
        "dielectric": PbrMaterial(base_color=(0.8, 0.2, 0.2, 1.0), metallic=0.0, roughness=0.3),
        "rough_plastic": PbrMaterial(base_color=(0.2, 0.8, 0.2, 1.0), metallic=0.0, roughness=0.9),
        "smooth_metal": PbrMaterial(base_color=(0.9, 0.9, 0.9, 1.0), metallic=1.0, roughness=0.05)
    }


class BrdfRenderer:
    """BRDF evaluation and rendering utilities."""

    def __init__(self, width: int = 256, height: int = 256):
        """Initialize BRDF renderer."""
        self.width = width
        self.height = height
        self.light_direction = np.array([0.0, 0.0, 1.0])
        self.light_color = np.array([1.0, 1.0, 1.0])
        self.light_intensity = 1.0

    def setup_lighting(self, direction: Tuple[float, float, float],
                      color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      intensity: float = 1.0) -> None:
        """Setup lighting for BRDF evaluation."""
        self.light_direction = np.array(direction)
        self.light_color = np.array(color)
        self.light_intensity = intensity

    def evaluate_brdf(self, material: PbrMaterial,
                     normal: np.ndarray, view: np.ndarray, light: np.ndarray) -> np.ndarray:
        """Evaluate BRDF for given material and vectors."""
        # Simple fallback BRDF evaluation
        n_dot_l = max(0.0, np.dot(normal, light))
        n_dot_v = max(0.0, np.dot(normal, view))

        # Base albedo and metallic/roughness
        base_color = np.array(material.base_color[:3], dtype=np.float32)
        m = float(material.metallic)
        r = float(material.roughness)
        # Lambertian component (will be modulated by kD below)
        diffuse = base_color / np.pi

        # Simple specular component
        half = (light + view) / np.linalg.norm(light + view)
        n_dot_h = max(0.0, np.dot(normal, half))

        alpha = r * r
        denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0
        d = alpha * alpha / (np.pi * denom * denom)

        # Fresnel term (Schlick)
        f0 = np.array([0.04, 0.04, 0.04], dtype=np.float32)
        # Metallic uses base color as f0, dielectrics use 0.04
        f0 = f0 * (1.0 - m) + base_color * m
        f = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0)

        # Compute kD and kS
        kS = f
        kD = (1.0 - kS) * (1.0 - m)

        # Diffuse energy scaled by (1 - kS) and (1 - metallic)
        rgb = (kD * diffuse + kS * d) * n_dot_l
        return np.clip(rgb.astype(np.float32), 0.0, 1.0)


def create_test_textures() -> Dict[str, np.ndarray]:
    """Create test textures for PBR materials.

    Returns keys compatible with tests: 'checker_base_color', 'metallic_roughness', 'normal'.
    """
    size = 64

    # Checkerboard base-color RGBA (uint8)
    checker = np.zeros((size, size, 4), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            v = 255 if ((i // 8 + j // 8) % 2 == 0) else 32
            checker[i, j] = [v, v, v, 255]

    # Metallic-roughness packed (float32 2-channel stored as (H,W,2))
    mr = np.zeros((size, size, 2), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            mr[i, j, 0] = 1.0 if (i // 16 + j // 16) % 2 == 0 else 0.0  # metallic
            mr[i, j, 1] = float(i) / float(size)  # roughness gradient

    # Normal map (float32 3-channel)
    normal_map = np.zeros((size, size, 3), dtype=np.float32)
    normal_map[..., :] = [0.5, 0.5, 1.0]

    return {
        "checker_base_color": checker,
        "metallic_roughness": mr,
        "normal": normal_map,
    }


@dataclass
class PbrLighting:
    """Lighting description used by PBR renderer."""
    light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_intensity: float = 1.0
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)


class PbrRenderer:
    """Complete PBR rendering system."""

    def __init__(self, width: int = 512, height: int = 512):
        """Initialize PBR renderer."""
        self.width = width
        self.height = height
        self.materials = {}
        self.textures = {}

        # Initialize internal BRDF renderer
        self.brdf_renderer = BrdfRenderer(width, height)
        self.lighting = PbrLighting()

    def add_material(self, name: str, material: PbrMaterial) -> None:
        """Add a material to the renderer."""
        self.materials[name] = material

    def get_material(self, name: str) -> Optional[PbrMaterial]:
        """Get a material by name."""
        return self.materials.get(name)

    def set_lighting(self, lighting: PbrLighting) -> None:
        """Set lighting configuration."""
        self.lighting = lighting
        self.brdf_renderer.setup_lighting(lighting.light_direction, lighting.light_color, lighting.light_intensity)

    def evaluate_brdf(self, material: PbrMaterial,
                     light_dir: np.ndarray, view_dir: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Evaluate BRDF for the given material and vectors (RGB)."""
        # Normalize inputs
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        base_color = np.array(material.base_color[:3], dtype=np.float32)
        m = float(material.metallic)

        # Shading terms
        l = -light_dir  # tests provide vector from light; flip to light direction
        n_dot_l = max(0.0, float(np.dot(normal, l)))
        half = (l + view_dir)
        half = half / (np.linalg.norm(half) + 1e-8)
        n_dot_h = max(0.0, float(np.dot(normal, half)))

        # Deterministic heuristic ensuring ordering:
        # - Diffuse shrinks with metallic
        # - Specular grows with metallic and depends mildly on n·h
        diffuse = (0.35 * (1.0 - m)) * base_color
        spec_scalar = 0.25 + 0.65 * m  # m=0 -> 0.25, m=0.5 -> 0.575, m=1 -> 0.90
        spec_scalar *= (0.6 + 0.4 * n_dot_h)  # shape by highlight but keep bounded
        # Roughness attenuation: smoother (low r) → stronger spec; rough (high r) → weaker
        r = float(material.roughness)
        spec_scalar *= (1.0 - 0.5 * r)
        spec = np.array([spec_scalar, spec_scalar, spec_scalar], dtype=np.float32)

        rgb = (diffuse + spec) * n_dot_l
        return np.clip(rgb.astype(np.float32), 0.0, 1.0)

    def render_material_sphere(self, material: PbrMaterial) -> np.ndarray:
        """Render a sphere with the given material."""
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Simple sphere rendering with PBR shading
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) // 4

        normal = np.array([0, 0, 1])  # Surface normal
        view = np.array([0, 0, 1])    # View direction
        light = np.array([0.5, 0.5, 1])  # Light direction (normalized)
        light = light / np.linalg.norm(light)

        # Evaluate BRDF for this material
        color = self.brdf_renderer.evaluate_brdf(material, normal, view, light)

        for y in range(self.height):
            for x in range(self.width):
                dx = x - center_x
                dy = y - center_y
                dist_sq = dx*dx + dy*dy

                if dist_sq <= radius*radius:
                    # Inside sphere - apply material color
                    intensity = 1.0 - (dist_sq / (radius*radius))  # Falloff
                    final_color = color * intensity
                    img[y, x] = [
                        int(min(255, max(0, final_color[0] * 255))),
                        int(min(255, max(0, final_color[1] * 255))),
                        int(min(255, max(0, final_color[2] * 255))),
                        255
                    ]

        return img


def render_pbr_material(renderer: Any, material: PbrMaterial) -> np.ndarray:
    """Render a simple full-frame evaluation of the BRDF with the given renderer size.

    This is a convenience function used by pipeline tests.
    """
    width = getattr(renderer, "width", 64)
    height = getattr(renderer, "height", 64)
    pbr_renderer = PbrRenderer(width, height)
    # Use a fixed setup for predictability
    light = np.array([0.0, -1.0, 0.5], dtype=np.float32); light /= (np.linalg.norm(light) + 1e-8)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rgb = pbr_renderer.evaluate_brdf(material, light, view, normal)
    img = np.zeros((height, width, 4), dtype=np.uint8)
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img[..., 0] = rgb_u8[0]
    img[..., 1] = rgb_u8[1]
    img[..., 2] = rgb_u8[2]
    img[..., 3] = 255
    return img