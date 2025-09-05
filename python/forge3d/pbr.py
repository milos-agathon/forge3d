"""
Physically-Based Rendering (PBR) materials system.

Provides PBR material definitions, texture management, and BRDF calculations
following the metallic-roughness workflow for realistic material rendering.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum
import struct

logger = logging.getLogger(__name__)


class PbrWorkflow(Enum):
    """PBR workflow types."""
    METALLIC_ROUGHNESS = "metallic_roughness"
    SPECULAR_GLOSSINESS = "specular_glossiness"  # Future support


class PbrMaterial:
    """PBR material with metallic-roughness workflow."""
    
    def __init__(
        self,
        base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        metallic: float = 0.0,
        roughness: float = 1.0,
        normal_scale: float = 1.0,
        occlusion_strength: float = 1.0,
        emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        alpha_cutoff: float = 0.5,
    ):
        """
        Create PBR material with metallic-roughness workflow.
        
        Args:
            base_color: Base color (albedo) RGBA [0,1]
            metallic: Metallic factor [0,1] - 0=dielectric, 1=metallic
            roughness: Roughness factor [0,1] - 0=mirror, 1=rough
            normal_scale: Normal map intensity multiplier
            occlusion_strength: Ambient occlusion strength [0,1]
            emissive: Emissive color RGB
            alpha_cutoff: Alpha testing threshold
        """
        self.base_color = base_color
        self.metallic = np.clip(metallic, 0.0, 1.0)
        self.roughness = np.clip(roughness, 0.04, 1.0)  # Min roughness to avoid singularities
        self.normal_scale = normal_scale
        self.occlusion_strength = np.clip(occlusion_strength, 0.0, 1.0)
        self.emissive = emissive
        self.alpha_cutoff = alpha_cutoff
        
        # Texture references
        self.textures = {
            'base_color': None,
            'metallic_roughness': None,
            'normal': None,
            'occlusion': None,
            'emissive': None,
        }
        
        # Texture flags
        self.texture_flags = 0
        
        logger.debug(f"Created PBR material: metallic={metallic:.2f}, roughness={roughness:.2f}")
    
    def set_base_color_texture(self, texture_data: np.ndarray) -> None:
        """Set base color texture data."""
        if not isinstance(texture_data, np.ndarray):
            raise TypeError("texture_data must be numpy ndarray")
        
        if texture_data.dtype != np.uint8:
            texture_data = (np.clip(texture_data, 0, 1) * 255).astype(np.uint8)
        
        self.textures['base_color'] = texture_data
        self.texture_flags |= 1  # BASE_COLOR flag
        logger.debug(f"Set base color texture: {texture_data.shape}")
    
    def set_metallic_roughness_texture(self, texture_data: np.ndarray) -> None:
        """Set metallic-roughness texture data (B=metallic, G=roughness)."""
        if not isinstance(texture_data, np.ndarray):
            raise TypeError("texture_data must be numpy ndarray")
        
        if texture_data.dtype != np.uint8:
            texture_data = (np.clip(texture_data, 0, 1) * 255).astype(np.uint8)
        
        self.textures['metallic_roughness'] = texture_data
        self.texture_flags |= 2  # METALLIC_ROUGHNESS flag
        logger.debug(f"Set metallic-roughness texture: {texture_data.shape}")
    
    def set_normal_texture(self, texture_data: np.ndarray) -> None:
        """Set normal map texture data (tangent space)."""
        if not isinstance(texture_data, np.ndarray):
            raise TypeError("texture_data must be numpy ndarray")
        
        if texture_data.dtype != np.uint8:
            texture_data = (np.clip(texture_data, 0, 1) * 255).astype(np.uint8)
        
        self.textures['normal'] = texture_data
        self.texture_flags |= 4  # NORMAL flag
        logger.debug(f"Set normal texture: {texture_data.shape}")
    
    def set_occlusion_texture(self, texture_data: np.ndarray) -> None:
        """Set ambient occlusion texture data."""
        if not isinstance(texture_data, np.ndarray):
            raise TypeError("texture_data must be numpy ndarray")
        
        if texture_data.dtype != np.uint8:
            texture_data = (np.clip(texture_data, 0, 1) * 255).astype(np.uint8)
        
        self.textures['occlusion'] = texture_data
        self.texture_flags |= 8  # OCCLUSION flag
        logger.debug(f"Set occlusion texture: {texture_data.shape}")
    
    def set_emissive_texture(self, texture_data: np.ndarray) -> None:
        """Set emissive texture data."""
        if not isinstance(texture_data, np.ndarray):
            raise TypeError("texture_data must be numpy ndarray")
        
        if texture_data.dtype != np.uint8:
            texture_data = (np.clip(texture_data, 0, 1) * 255).astype(np.uint8)
        
        self.textures['emissive'] = texture_data
        self.texture_flags |= 16  # EMISSIVE flag
        logger.debug(f"Set emissive texture: {texture_data.shape}")
    
    def get_material_data(self) -> np.ndarray:
        """Get material data as structured array for GPU upload."""
        # Pack material data according to GPU layout
        material_data = np.array([
            # base_color (vec4)
            self.base_color[0], self.base_color[1], self.base_color[2], self.base_color[3],
            # metallic, roughness, normal_scale, occlusion_strength
            self.metallic, self.roughness, self.normal_scale, self.occlusion_strength,
            # emissive (vec3) + alpha_cutoff
            self.emissive[0], self.emissive[1], self.emissive[2], self.alpha_cutoff,
            # texture_flags + padding
            float(self.texture_flags), 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        
        return material_data


class PbrLighting:
    """PBR lighting configuration."""
    
    def __init__(
        self,
        light_direction: Tuple[float, float, float] = (0.0, -1.0, 0.3),
        light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        light_intensity: float = 3.0,
        camera_position: Tuple[float, float, float] = (0.0, 0.0, 5.0),
        ibl_intensity: float = 1.0,
        ibl_rotation: float = 0.0,
        exposure: float = 1.0,
        gamma: float = 2.2,
    ):
        """
        Create PBR lighting configuration.
        
        Args:
            light_direction: Primary directional light direction
            light_color: Light color RGB [0,1]
            light_intensity: Light intensity multiplier
            camera_position: Camera position for view-dependent effects
            ibl_intensity: Image-based lighting intensity
            ibl_rotation: IBL environment rotation in radians
            exposure: Exposure adjustment
            gamma: Gamma correction value
        """
        self.light_direction = np.array(light_direction, dtype=np.float32)
        self.light_color = np.array(light_color, dtype=np.float32)
        self.light_intensity = float(light_intensity)
        self.camera_position = np.array(camera_position, dtype=np.float32)
        self.ibl_intensity = float(ibl_intensity)
        self.ibl_rotation = float(ibl_rotation)
        self.exposure = float(exposure)
        self.gamma = float(gamma)
    
    def get_lighting_data(self) -> np.ndarray:
        """Get lighting data as structured array for GPU upload."""
        lighting_data = np.array([
            # light_direction (vec3) + padding
            self.light_direction[0], self.light_direction[1], self.light_direction[2], 0.0,
            # light_color (vec3) + light_intensity
            self.light_color[0], self.light_color[1], self.light_color[2], self.light_intensity,
            # camera_position (vec3) + padding
            self.camera_position[0], self.camera_position[1], self.camera_position[2], 0.0,
            # ibl_intensity, ibl_rotation, exposure, gamma
            self.ibl_intensity, self.ibl_rotation, self.exposure, self.gamma,
        ], dtype=np.float32)
        
        return lighting_data


class PbrRenderer:
    """PBR material renderer with CPU-side BRDF evaluation."""
    
    def __init__(self):
        """Initialize PBR renderer."""
        self.materials = {}
        self.lighting = PbrLighting()
        
    def add_material(self, name: str, material: PbrMaterial) -> None:
        """Add material to renderer."""
        self.materials[name] = material
        logger.debug(f"Added PBR material: {name}")
    
    def set_lighting(self, lighting: PbrLighting) -> None:
        """Set lighting configuration."""
        self.lighting = lighting
    
    def evaluate_brdf(
        self,
        material: PbrMaterial,
        light_dir: np.ndarray,
        view_dir: np.ndarray,
        normal: np.ndarray,
        uv: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Evaluate PBR BRDF for given material and lighting.
        
        Args:
            material: PBR material
            light_dir: Light direction (normalized)
            view_dir: View direction (normalized)
            normal: Surface normal (normalized)
            uv: UV coordinates for texture sampling
            
        Returns:
            RGB color as (3,) array
        """
        # Sample material properties
        base_color = np.array(material.base_color[:3], dtype=np.float32)
        metallic = material.metallic
        roughness = material.roughness
        
        if uv is not None and material.textures['base_color'] is not None:
            # Sample base color texture (simplified)
            base_color = self._sample_texture(material.textures['base_color'], uv) / 255.0
        
        if uv is not None and material.textures['metallic_roughness'] is not None:
            # Sample metallic-roughness texture
            mr_sample = self._sample_texture(material.textures['metallic_roughness'], uv)
            metallic = metallic * (mr_sample[2] / 255.0) if mr_sample.size > 2 else metallic  # Blue channel
            roughness = roughness * (mr_sample[1] / 255.0) if mr_sample.size > 1 else roughness  # Green channel
        
        # Ensure minimum roughness
        roughness = max(roughness, 0.04)
        
        # Calculate BRDF
        half_dir = (light_dir + view_dir)
        half_dir = half_dir / np.linalg.norm(half_dir)
        
        n_dot_l = max(np.dot(normal, light_dir), 0.0)
        n_dot_v = max(np.dot(normal, view_dir), 0.0)
        n_dot_h = max(np.dot(normal, half_dir), 0.0)
        v_dot_h = max(np.dot(view_dir, half_dir), 0.0)
        
        if n_dot_l <= 0.0 or n_dot_v <= 0.0:
            return np.zeros(3, dtype=np.float32)
        
        # Calculate F0 (surface reflection at zero incidence)
        dielectric_f0 = np.array([0.04, 0.04, 0.04])
        f0 = base_color * metallic + dielectric_f0 * (1.0 - metallic)
        
        # BRDF components
        D = self._distribution_ggx(n_dot_h, roughness)
        G = self._geometry_smith(n_dot_v, n_dot_l, roughness)
        F = self._fresnel_schlick(v_dot_h, f0)
        
        # Cook-Torrance specular BRDF
        specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-6)
        
        # Lambertian diffuse BRDF
        kS = F
        kD = (np.ones(3) - kS) * (1.0 - metallic)
        diffuse = kD * base_color / np.pi
        
        return (diffuse + specular) * n_dot_l
    
    def _sample_texture(self, texture: np.ndarray, uv: Tuple[float, float]) -> np.ndarray:
        """Sample texture at UV coordinates."""
        if texture is None:
            return np.array([255, 255, 255], dtype=np.uint8)
        
        u, v = uv
        height, width = texture.shape[:2]
        
        # Clamp UV to [0,1] and convert to pixel coordinates
        x = int(np.clip(u, 0, 1) * (width - 1))
        y = int(np.clip(v, 0, 1) * (height - 1))
        
        if len(texture.shape) == 2:
            # Grayscale
            return np.array([texture[y, x]], dtype=np.uint8)
        else:
            # Color
            return texture[y, x].copy()
    
    def _distribution_ggx(self, n_dot_h: float, roughness: float) -> float:
        """GGX/Trowbridge-Reitz normal distribution function."""
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        n_dot_h2 = n_dot_h * n_dot_h
        
        num = alpha2
        denom = np.pi * ((n_dot_h2 * (alpha2 - 1.0) + 1.0) ** 2)
        
        return num / max(denom, 1e-6)
    
    def _geometry_smith(self, n_dot_v: float, n_dot_l: float, roughness: float) -> float:
        """Smith geometry function for GGX."""
        k = ((roughness + 1.0) * (roughness + 1.0)) / 8.0
        
        ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k)
        ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k)
        
        return ggx1 * ggx2
    
    def _fresnel_schlick(self, cos_theta: float, f0: np.ndarray) -> np.ndarray:
        """Fresnel-Schlick approximation."""
        return f0 + (np.ones_like(f0) - f0) * ((1.0 - np.clip(cos_theta, 0.0, 1.0)) ** 5)


def create_test_materials() -> Dict[str, PbrMaterial]:
    """Create a set of test PBR materials."""
    materials = {}
    
    # Gold material
    materials['gold'] = PbrMaterial(
        base_color=(1.0, 0.86, 0.57, 1.0),
        metallic=1.0,
        roughness=0.1
    )
    
    # Silver material
    materials['silver'] = PbrMaterial(
        base_color=(0.95, 0.93, 0.88, 1.0),
        metallic=1.0,
        roughness=0.05
    )
    
    # Copper material
    materials['copper'] = PbrMaterial(
        base_color=(0.95, 0.64, 0.54, 1.0),
        metallic=1.0,
        roughness=0.15
    )
    
    # Plastic material
    materials['plastic_red'] = PbrMaterial(
        base_color=(0.8, 0.2, 0.2, 1.0),
        metallic=0.0,
        roughness=0.7
    )
    
    # Rubber material
    materials['rubber_black'] = PbrMaterial(
        base_color=(0.1, 0.1, 0.1, 1.0),
        metallic=0.0,
        roughness=0.9
    )
    
    # Wood material
    materials['wood'] = PbrMaterial(
        base_color=(0.6, 0.4, 0.2, 1.0),
        metallic=0.0,
        roughness=0.8
    )
    
    # Glass material
    materials['glass'] = PbrMaterial(
        base_color=(0.95, 0.95, 0.95, 0.1),
        metallic=0.0,
        roughness=0.0,
        alpha_cutoff=0.0
    )
    
    # Emissive material
    materials['emissive'] = PbrMaterial(
        base_color=(0.2, 0.2, 0.2, 1.0),
        metallic=0.0,
        roughness=1.0,
        emissive=(1.0, 0.5, 0.1)  # Orange glow
    )
    
    return materials


def create_test_textures() -> Dict[str, np.ndarray]:
    """Create test textures for PBR materials."""
    textures = {}
    
    # Checkerboard base color texture
    size = 64
    checker = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x // 8 + y // 8) % 2 == 0:
                checker[y, x] = [255, 255, 255]  # White
            else:
                checker[y, x] = [100, 100, 100]  # Gray
    
    textures['checker_base_color'] = checker
    
    # Metallic-roughness texture (rough metal strips)
    mr_tex = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            # Vertical strips of varying metallic/roughness
            strip = (x // (size // 4)) % 4
            metallic = strip * 85  # 0, 85, 170, 255
            roughness = 255 - (strip * 63)  # 255, 192, 129, 66
            
            mr_tex[y, x] = [roughness, roughness, metallic]  # G=roughness, B=metallic
    
    textures['metallic_roughness'] = mr_tex
    
    # Normal map texture (simple bump pattern)
    normal_tex = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            # Create simple wave pattern in normal map
            nx = np.sin(x * 0.3) * 0.5
            ny = np.sin(y * 0.3) * 0.5
            nz = np.sqrt(1.0 - nx*nx - ny*ny)
            
            # Encode normal to [0,255] range
            normal_tex[y, x] = [
                int((nx + 1.0) * 127.5),
                int((ny + 1.0) * 127.5),
                int((nz + 1.0) * 127.5)
            ]
    
    textures['normal'] = normal_tex
    
    return textures


def validate_pbr_material(material: PbrMaterial) -> Dict[str, Any]:
    """Validate PBR material properties."""
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    # Check base color range
    for i, component in enumerate(['R', 'G', 'B', 'A']):
        value = material.base_color[i]
        if not (0.0 <= value <= 1.0):
            results['errors'].append(f"Base color {component} out of range [0,1]: {value}")
            results['valid'] = False
    
    # Check metallic range
    if not (0.0 <= material.metallic <= 1.0):
        results['errors'].append(f"Metallic out of range [0,1]: {material.metallic}")
        results['valid'] = False
    
    # Check roughness range
    if material.roughness < 0.04:
        results['warnings'].append(f"Roughness {material.roughness} below recommended minimum 0.04")
    elif material.roughness > 1.0:
        results['errors'].append(f"Roughness out of range [0,1]: {material.roughness}")
        results['valid'] = False
    
    # Check occlusion strength
    if not (0.0 <= material.occlusion_strength <= 1.0):
        results['errors'].append(f"Occlusion strength out of range [0,1]: {material.occlusion_strength}")
        results['valid'] = False
    
    # Check emissive values
    for i, component in enumerate(['R', 'G', 'B']):
        value = material.emissive[i]
        if value < 0.0:
            results['errors'].append(f"Emissive {component} cannot be negative: {value}")
            results['valid'] = False
        elif value > 10.0:
            results['warnings'].append(f"High emissive {component} value: {value}")
    
    # Statistics
    results['statistics'] = {
        'is_metallic': material.metallic > 0.5,
        'is_rough': material.roughness > 0.7,
        'is_emissive': any(e > 0.01 for e in material.emissive),
        'has_transparency': material.base_color[3] < 1.0,
        'texture_count': bin(material.texture_flags).count('1'),
    }
    
    return results


def has_pbr_support() -> bool:
    """Check if PBR materials are supported."""
    try:
        # Try creating a basic PBR material
        material = PbrMaterial()
        validation = validate_pbr_material(material)
        return validation['valid']
    except Exception as e:
        logger.debug(f"PBR not supported: {e}")
        return False


# Export main classes and functions
__all__ = [
    'PbrMaterial',
    'PbrLighting', 
    'PbrRenderer',
    'PbrWorkflow',
    'create_test_materials',
    'create_test_textures',
    'validate_pbr_material',
    'has_pbr_support'
]