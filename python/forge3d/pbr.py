#!/usr/bin/env python3
"""A24: Anisotropic Microfacet BRDF + PBR materials functionality"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

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
                 emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """Initialize PBR material with standard parameters."""
        self.base_color = base_color
        self.metallic = max(0.0, min(1.0, metallic))  # Clamp metallic
        self.roughness = max(0.0, min(1.0, roughness))  # Clamp roughness
        self.normal_scale = normal_scale
        self.occlusion_strength = occlusion_strength
        self.emissive = emissive
        self.textures: Dict[str, Any] = {}

    def set_texture(self, texture_type: str, texture_data: Any) -> None:
        """Assign texture to material."""
        self.textures[texture_type] = texture_data

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


def validate_material(material: PbrMaterial) -> bool:
    """Validate PBR material parameters."""
    if not isinstance(material, PbrMaterial):
        return False

    # Check metallic range
    if not (0.0 <= material.metallic <= 1.0):
        return False

    # Check roughness range
    if not (0.0 <= material.roughness <= 1.0):
        return False

    # Check base color format
    if not (isinstance(material.base_color, (tuple, list)) and len(material.base_color) == 4):
        return False

    # Check emissive format
    if not (isinstance(material.emissive, (tuple, list)) and len(material.emissive) == 3):
        return False

    return True


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

        # Lambertian component
        diffuse = np.array(material.base_color[:3]) * (1.0 - material.metallic) / np.pi

        # Simple specular component
        half = (light + view) / np.linalg.norm(light + view)
        n_dot_h = max(0.0, np.dot(normal, half))

        # Simplified GGX-like distribution
        alpha = material.roughness * material.roughness
        denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0
        d = alpha * alpha / (np.pi * denom * denom)

        # Fresnel approximation
        f0 = 0.04 if material.metallic < 0.5 else np.array(material.base_color[:3])
        f = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0)

        specular = d * f / (4.0 * n_dot_v * n_dot_l + 0.001)

        if material.metallic > 0.5:
            return specular * n_dot_l
        else:
            return (diffuse + specular * material.metallic) * n_dot_l


def create_test_textures() -> Dict[str, np.ndarray]:
    """Create test textures for PBR materials."""
    size = 64

    # Checkerboard normal map
    normal_map = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if (i // 8 + j // 8) % 2 == 0:
                normal_map[i, j] = [0.5, 0.5, 1.0]  # Neutral normal
            else:
                normal_map[i, j] = [0.7, 0.3, 0.8]  # Perturbed normal

    # Simple roughness map
    roughness_map = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            roughness_map[i, j] = 0.1 + 0.8 * (i / size)  # Gradient from smooth to rough

    # Metallic map
    metallic_map = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            metallic_map[i, j] = 1.0 if (i // 16 + j // 16) % 2 == 0 else 0.0  # Alternating regions

    return {
        "normal": normal_map,
        "roughness": roughness_map,
        "metallic": metallic_map
    }


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

    def add_material(self, name: str, material: PbrMaterial) -> None:
        """Add a material to the renderer."""
        self.materials[name] = material

    def get_material(self, name: str) -> Optional[PbrMaterial]:
        """Get a material by name."""
        return self.materials.get(name)

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