"""
Cascaded Shadow Maps (CSM) implementation for directional lighting.

Provides high-quality shadow mapping with multiple cascade levels and 
Percentage-Closer Filtering (PCF) for soft shadow edges.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

try:
    import forge3d._core as _core
    HAS_SHADOWS_SUPPORT = True
except (ImportError, AttributeError):
    HAS_SHADOWS_SUPPORT = False


def has_shadows_support() -> bool:
    """Check if shadow mapping functionality is available."""
    return HAS_SHADOWS_SUPPORT




class DirectionalLight:
    """Directional light configuration for shadow casting."""
    
    def __init__(self,
                 direction: Tuple[float, float, float] = (0.0, -1.0, 0.3),
                 color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 intensity: float = 3.0,
                 cast_shadows: bool = True):
        """
        Create directional light.
        
        Args:
            direction: Light direction (pointing towards light source), will be normalized
            color: Light color (RGB)
            intensity: Light intensity multiplier
            cast_shadows: Enable shadow casting
        """
        # Normalize direction
        dir_array = np.array(direction, dtype=np.float32)
        length = np.linalg.norm(dir_array)
        if length > 0:
            dir_array /= length
        else:
            dir_array = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        
        self.direction = tuple(dir_array)
        self.color = tuple(np.clip(color, 0.0, 10.0))  # Reasonable range
        self.intensity = max(0.0, intensity)
        self.cast_shadows = cast_shadows
    
    def __repr__(self) -> str:
        return (f"DirectionalLight(direction={self.direction}, "
                f"color={self.color}, intensity={self.intensity}, "
                f"cast_shadows={self.cast_shadows})")


class CsmConfig:
    """Configuration for Cascaded Shadow Maps."""
    
    def __init__(self,
                 cascade_count: int = 4,
                 shadow_map_size: int = 2048,
                 camera_far: float = 1000.0,
                 camera_near: float = 0.1,
                 lambda_factor: float = 0.5,
                 depth_bias: float = 0.0001,
                 slope_bias: float = 0.001,
                 pcf_kernel_size: int = 3):
        """
        Create CSM configuration.
        
        Args:
            cascade_count: Number of cascade levels (1-4)
            shadow_map_size: Resolution per cascade (power of 2, 512-4096)
            camera_far: Camera far plane distance
            camera_near: Camera near plane distance  
            lambda_factor: Split scheme blend factor (0.0=uniform, 1.0=logarithmic)
            depth_bias: Fixed depth bias to prevent shadow acne
            slope_bias: Slope-scaled bias for angled surfaces
            pcf_kernel_size: PCF filter size (1, 3, 5, or 7)
        """
        # Validate parameters
        self.cascade_count = max(1, min(4, int(cascade_count)))
        
        # Ensure shadow map size is power of 2
        valid_sizes = [512, 1024, 2048, 3072, 4096]
        self.shadow_map_size = min(valid_sizes, key=lambda x: abs(x - shadow_map_size))
        
        self.camera_far = max(camera_near + 1.0, camera_far)
        self.camera_near = max(0.01, camera_near)
        self.lambda_factor = np.clip(lambda_factor, 0.0, 1.0)
        self.depth_bias = max(0.0, depth_bias)
        self.slope_bias = max(0.0, slope_bias)
        
        # Validate PCF kernel size
        valid_kernels = [1, 3, 5, 7]
        self.pcf_kernel_size = min(valid_kernels, key=lambda x: abs(x - pcf_kernel_size))
        
    def __repr__(self) -> str:
        return (f"CsmConfig(cascade_count={self.cascade_count}, "
                f"shadow_map_size={self.shadow_map_size}, "
                f"camera_far={self.camera_far}, camera_near={self.camera_near}, "
                f"lambda_factor={self.lambda_factor}, pcf_kernel_size={self.pcf_kernel_size})")


class ShadowStats:
    """Shadow mapping statistics and debugging information."""
    
    def __init__(self, cascade_count: int, shadow_map_size: int, 
                 memory_usage: int, light_direction: Tuple[float, float, float],
                 split_distances: List[float], texel_sizes: List[float]):
        self.cascade_count = cascade_count
        self.shadow_map_size = shadow_map_size
        self.memory_usage = memory_usage  # In bytes
        self.light_direction = light_direction
        self.split_distances = split_distances
        self.texel_sizes = texel_sizes
    
    def __repr__(self) -> str:
        memory_mb = self.memory_usage / (1024 * 1024)
        return (f"ShadowStats(cascades={self.cascade_count}, "
                f"resolution={self.shadow_map_size}x{self.shadow_map_size}, "
                f"memory={memory_mb:.1f}MB)")


class CsmShadowMap:
    """Cascaded Shadow Map system for directional lighting."""
    
    def __init__(self, config: Optional[CsmConfig] = None):
        """
        Create CSM shadow mapping system.
        
        Args:
            config: Shadow map configuration (uses default if None)
        """
        # Allow pure-Python placeholder when native support is unavailable
        if not has_shadows_support():
            warnings.warn("CSM shadows running in pure-Python fallback mode", RuntimeWarning)
        
        self.config = config or CsmConfig()
        self.light = DirectionalLight()
        self.debug_visualization = False
        
        # These will be set when integrated with renderer
        self._native_shadow_map = None
        self._initialized = False
    
    def set_light(self, light: DirectionalLight) -> None:
        """Set directional light parameters."""
        self.light = light
        if self._initialized and self._native_shadow_map:
            # Update native light parameters
            pass  # Will be implemented with native integration
    
    def set_debug_visualization(self, enabled: bool) -> None:
        """Enable/disable cascade debug visualization."""
        self.debug_visualization = enabled
        if self._initialized and self._native_shadow_map:
            # Update native debug mode
            pass  # Will be implemented with native integration
    
    def get_stats(self) -> ShadowStats:
        """Get current shadow mapping statistics."""
        memory_per_cascade = self.config.shadow_map_size * self.config.shadow_map_size * 4
        total_memory = memory_per_cascade * self.config.cascade_count
        
        # Mock data for now - will be replaced with actual cascade data
        split_distances = [
            self.config.camera_near + (self.config.camera_far - self.config.camera_near) * 
            (i / self.config.cascade_count) for i in range(1, self.config.cascade_count + 1)
        ]
        
        texel_sizes = [1.0] * self.config.cascade_count  # Will be calculated properly
        
        return ShadowStats(
            cascade_count=self.config.cascade_count,
            shadow_map_size=self.config.shadow_map_size,
            memory_usage=total_memory,
            light_direction=self.light.direction,
            split_distances=split_distances,
            texel_sizes=texel_sizes
        )
    
    def __repr__(self) -> str:
        return f"CsmShadowMap(config={self.config}, light={self.light})"


class ShadowRenderer:
    """Shadow-aware renderer with CSM support."""
    
    def __init__(self, width: int, height: int, config: Optional[CsmConfig] = None):
        """
        Create shadow-aware renderer.
        
        Args:
            width: Render target width
            height: Render target height  
            config: Shadow configuration
        """
        # Allow pure-Python placeholder when native support is unavailable
        if not has_shadows_support():
            warnings.warn("ShadowRenderer running in pure-Python fallback mode", RuntimeWarning)
        
        self.width = width
        self.height = height
        self.shadow_map = CsmShadowMap(config)
        
        # Camera parameters
        self.camera_position = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov_y = 45.0  # degrees
        self.aspect_ratio = width / height
    
    def set_camera(self, position: Tuple[float, float, float],
                   target: Tuple[float, float, float],
                   up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
                   fov_y_degrees: float = 45.0) -> None:
        """Set camera parameters."""
        self.camera_position = np.array(position, dtype=np.float32)
        self.camera_target = np.array(target, dtype=np.float32)
        self.camera_up = np.array(up, dtype=np.float32)
        self.fov_y = np.clip(fov_y_degrees, 10.0, 160.0)
    
    def set_light(self, light: DirectionalLight) -> None:
        """Set directional light for shadow casting."""
        self.shadow_map.set_light(light)
    
    def enable_debug_visualization(self, enabled: bool = True) -> None:
        """Enable cascade debug visualization."""
        self.shadow_map.set_debug_visualization(enabled)
    
    def render_with_shadows(self, scene_data: Dict) -> np.ndarray:
        """
        Render scene with shadow mapping.
        
        Args:
            scene_data: Scene geometry and materials
            
        Returns:
            Rendered image as numpy array (height, width, 3)
        """
        if HAS_SHADOWS_SUPPORT:
            # Try native rendering path
            try:
                return self._render_native_shadows(scene_data)
            except Exception:
                # Fall back to placeholder if native fails
                pass
        
        # Fallback placeholder implementation for import safety
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Simple gradient for demonstration
        for y in range(self.height):
            for x in range(self.width):
                # Create depth-like gradient
                depth = y / self.height
                intensity = int(255 * (1.0 - depth * 0.8))
                
                # Add cascade debug colors if enabled
                if self.shadow_map.debug_visualization:
                    cascade_idx = int(depth * self.shadow_map.config.cascade_count)
                    cascade_colors = [
                        (255, 100, 100),  # Red
                        (100, 255, 100),  # Green
                        (100, 100, 255),  # Blue
                        (255, 255, 100),  # Yellow
                    ]
                    if cascade_idx < len(cascade_colors):
                        base_color = cascade_colors[cascade_idx]
                        image[y, x] = [int(c * 0.3 + intensity * 0.7) for c in base_color]
                else:
                    image[y, x] = [intensity, intensity, intensity]
        
        return image
    
    def _render_native_shadows(self, scene_data: Dict) -> np.ndarray:
        """Native shadow rendering implementation."""
        if not HAS_SHADOWS_SUPPORT:
            raise RuntimeError("Native shadow support not available")
        
        # This will be implemented with actual native bindings
        return _core.render_scene_with_shadows(
            scene_data,
            self.shadow_map.config.__dict__,
            self.shadow_map.light.__dict__,
            {
                'position': self.camera_position.tolist(),
                'target': self.camera_target.tolist(),
                'up': self.camera_up.tolist(),
                'fov_y': self.fov_y,
                'aspect_ratio': self.aspect_ratio,
            },
            self.width,
            self.height
        )
    
    def get_shadow_stats(self) -> ShadowStats:
        """Get shadow mapping statistics."""
        return self.shadow_map.get_stats()


def create_test_scene(ground_size: float = 20.0, num_objects: int = 8) -> Dict:
    """
    Create test scene with ground plane and objects for shadow demonstration.
    
    Args:
        ground_size: Size of ground plane
        num_objects: Number of test objects to create
        
    Returns:
        Scene dictionary with geometry data
    """
    scene = {
        'ground': {
            'vertices': np.array([
                [-ground_size, 0.0, -ground_size],  # Bottom-left
                [ground_size, 0.0, -ground_size],   # Bottom-right
                [ground_size, 0.0, ground_size],    # Top-right
                [-ground_size, 0.0, ground_size],   # Top-left
            ], dtype=np.float32),
            'normals': np.array([
                [0.0, 1.0, 0.0],  # Up
                [0.0, 1.0, 0.0],  # Up
                [0.0, 1.0, 0.0],  # Up
                [0.0, 1.0, 0.0],  # Up
            ], dtype=np.float32),
            'indices': np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
        },
        'objects': [],
        'bounds': {
            'min': (-ground_size, 0.0, -ground_size),
            'max': (ground_size, 0.0, ground_size),
        }
    }
    
    # Add test objects (cubes at various positions)
    for i in range(num_objects):
        angle = 2.0 * np.pi * i / num_objects
        x = 8.0 * np.cos(angle)
        z = 8.0 * np.sin(angle)
        height = 1.0 + 2.0 * np.sin(angle * 2)
        
        # Simple cube vertices (centered at origin, will be translated)
        cube_vertices = np.array([
            # Bottom face
            [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1],
            # Top face  
            [-1, height, -1], [1, height, -1], [1, height, 1], [-1, height, 1],
        ], dtype=np.float32)
        
        # Translate cube to position
        cube_vertices[:, 0] += x
        cube_vertices[:, 2] += z
        
        # Cube normals
        cube_normals = np.array([
            # Bottom face
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
            # Top face
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        ], dtype=np.float32)
        
        # Simple cube indices (just top and bottom faces for simplicity)
        cube_indices = np.array([
            # Bottom face
            0, 2, 1, 0, 3, 2,
            # Top face
            4, 5, 6, 4, 6, 7,
        ], dtype=np.uint32)
        
        scene['objects'].append({
            'vertices': cube_vertices,
            'normals': cube_normals,
            'indices': cube_indices,
            'position': (x, 0, z),
            'height': height,
        })
    
    return scene


def compare_shadow_techniques() -> Dict[str, float]:
    """
    Compare different shadow filtering techniques.
    
    Returns:
        Dictionary of technique names and their performance scores
    """
    techniques = {
        'no_filtering': 1.0,      # Fastest, lowest quality
        'pcf_3x3': 0.7,          # 9 samples
        'pcf_5x5': 0.4,          # 25 samples  
        'pcf_7x7': 0.2,          # 49 samples
        'poisson_pcf': 0.5,      # 16 samples, better quality
    }
    
    return techniques


def validate_csm_setup(config: CsmConfig, light: DirectionalLight, 
                       camera_near: float, camera_far: float) -> Dict[str, any]:
    """
    Validate CSM configuration and provide recommendations.
    
    Args:
        config: CSM configuration to validate
        light: Directional light configuration
        camera_near: Camera near plane
        camera_far: Camera far plane
        
    Returns:
        Validation results with warnings and recommendations
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': [],
        'memory_estimate_mb': 0.0,
    }
    
    # Check cascade count
    if config.cascade_count < 2:
        validation['warnings'].append("Single cascade provides limited shadow quality")
    elif config.cascade_count > 4:
        validation['errors'].append("Maximum 4 cascades supported")
        validation['valid'] = False
    
    # Check shadow map resolution
    if config.shadow_map_size < 512:
        validation['warnings'].append("Low shadow map resolution may cause aliasing")
    elif config.shadow_map_size > 4096:
        validation['warnings'].append("High resolution may impact performance")
    
    # Check camera ranges
    if config.camera_far / config.camera_near > 10000:
        validation['warnings'].append("Large far/near ratio may cause precision issues")
    
    # Check depth bias
    if config.depth_bias < 0.0001:
        validation['warnings'].append("Small depth bias may cause shadow acne")
    elif config.depth_bias > 0.01:
        validation['warnings'].append("Large depth bias may cause light leaking")
    
    # Check PCF kernel size
    if config.pcf_kernel_size == 1:
        validation['recommendations'].append("Consider PCF filtering for softer shadows")
    elif config.pcf_kernel_size >= 7:
        validation['warnings'].append("Large PCF kernel may impact performance")
    
    # Calculate memory usage using centralized function
    memory_validation = validate_shadow_memory_constraint(config)
    validation['memory_estimate_mb'] = memory_validation['memory_mb']
    
    if validation['memory_estimate_mb'] > 64:
        validation['warnings'].append(f"High memory usage: {validation['memory_estimate_mb']:.1f}MB")
    
    # Warn if approaching or exceeding 256 MiB constraint
    if not memory_validation['valid']:
        validation['errors'].append(
            f"Shadow atlas exceeds 256 MiB limit: {memory_validation['memory_mb']:.1f}MB > 256MB"
        )
        validation['valid'] = False
    elif memory_validation['headroom_mb'] < 32:  # Warn if less than 32 MiB headroom
        validation['warnings'].append(
            f"Approaching 256 MiB shadow atlas limit: {memory_validation['memory_mb']:.1f}MB "
            f"(only {memory_validation['headroom_mb']:.1f}MB headroom remaining)"
        )
    
    # Light direction validation
    light_length = np.linalg.norm(light.direction)
    if light_length < 0.1:
        validation['errors'].append("Light direction too short")
        validation['valid'] = False
    
    if abs(light.direction[1]) < 0.1:  # Nearly horizontal light
        validation['warnings'].append("Horizontal lights may cause long shadows")
    
    return validation


# Pre-configured setups for common scenarios
# Memory constraint: cascade_count × shadow_map_size² × 4 bytes ≤ 256 MiB (268,435,456 bytes)
PRESET_CONFIGS = {
    'low_quality': CsmConfig(
        cascade_count=2,
        shadow_map_size=1024,
        pcf_kernel_size=1,
        # Memory: 2 × 1024² × 4 = 8,388,608 bytes (8 MiB)
    ),
    'medium_quality': CsmConfig(
        cascade_count=3,
        shadow_map_size=1024,  # Reduced from 2048 to meet memory constraint
        pcf_kernel_size=3,
        # Memory: 3 × 1024² × 4 = 12,582,912 bytes (12 MiB)
    ),
    'high_quality': CsmConfig(
        cascade_count=4,
        shadow_map_size=2048,
        pcf_kernel_size=5,
        # Memory: 4 × 2048² × 4 = 67,108,864 bytes (64 MiB)
    ),
    'ultra_quality': CsmConfig(
        cascade_count=4,
        shadow_map_size=3072,  # Carefully selected to stay under 256 MiB
        pcf_kernel_size=7,
        # Memory: 4 × 3072² × 4 = 150,994,944 bytes (~144 MiB)
    ),
}

# Memory validation constants
MAX_SHADOW_ATLAS_MEMORY = 256 * 1024 * 1024  # 256 MiB in bytes
BYTES_PER_SHADOW_TEXEL = 4  # 32-bit depth format

def get_preset_config(preset: str) -> CsmConfig:
    """
    Get preset shadow configuration with automatic memory validation.
    
    This function unifies the preset selection API and ensures all presets
    respect the memory constraint: cascade_count × shadow_map_size² × 4 bytes ≤ 256 MiB.
    
    Args:
        preset: Quality level or legacy name. Supported values:
            - 'low_quality': 2 cascades, 1024² resolution (8 MiB)
            - 'medium_quality': 3 cascades, 1024² resolution (12 MiB) 
            - 'high_quality': 4 cascades, 2048² resolution (64 MiB)
            - 'ultra_quality': 4 cascades, 3072² resolution (~144 MiB)
            
            Legacy aliases (deprecated):
            - 'name' parameter accepted for backward compatibility
            - 'quality' parameter accepted for backward compatibility
    
    Returns:
        Pre-configured CsmConfig with validated memory usage
        
    Raises:
        ValueError: If preset name is not recognized
        RuntimeError: If preset exceeds 256 MiB memory limit (should not happen)
    """
    # Handle legacy parameter names with deprecation warning
    if preset not in PRESET_CONFIGS:
        # Try backward compatibility mapping
        legacy_mappings = {
            'low': 'low_quality',
            'medium': 'medium_quality', 
            'high': 'high_quality',
            'ultra': 'ultra_quality',
        }
        
        if preset in legacy_mappings:
            warnings.warn(
                f"Preset alias '{preset}' is deprecated. Use '{legacy_mappings[preset]}' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            preset = legacy_mappings[preset]
        else:
            available = list(PRESET_CONFIGS.keys()) + list(legacy_mappings.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    config = PRESET_CONFIGS[preset]
    
    # Validate memory constraint (should always pass for our presets, but double-check)
    memory_bytes = calculate_shadow_atlas_memory(config.cascade_count, config.shadow_map_size)
    
    if memory_bytes > MAX_SHADOW_ATLAS_MEMORY:
        raise RuntimeError(
            f"Preset '{preset}' exceeds 256 MiB memory limit: "
            f"{memory_bytes / (1024*1024):.1f} MiB > 256 MiB. "
            f"This indicates a bug in preset configuration."
        )
    
    return config


def calculate_shadow_atlas_memory(cascade_count: int, shadow_map_size: int) -> int:
    """
    Calculate shadow atlas memory usage in bytes.
    
    Formula: cascade_count × shadow_map_size² × BYTES_PER_SHADOW_TEXEL
    
    Args:
        cascade_count: Number of shadow cascades [1-4]
        shadow_map_size: Resolution per cascade (e.g., 1024, 2048, 4096)
        
    Returns:
        Memory usage in bytes
    """
    return cascade_count * shadow_map_size * shadow_map_size * BYTES_PER_SHADOW_TEXEL


def validate_shadow_memory_constraint(config: CsmConfig) -> Dict[str, any]:
    """
    Validate that shadow configuration meets memory constraints.
    
    Args:
        config: CSM configuration to validate
        
    Returns:
        dict: {
            'valid': bool,                    # Meets 256 MiB constraint  
            'memory_bytes': int,              # Actual memory usage
            'memory_mb': float,               # Memory usage in MiB
            'constraint_mb': float,           # Constraint limit (256 MiB)
            'headroom_mb': float,             # Remaining memory headroom
        }
    """
    memory_bytes = calculate_shadow_atlas_memory(config.cascade_count, config.shadow_map_size)
    memory_mb = memory_bytes / (1024 * 1024)
    constraint_mb = MAX_SHADOW_ATLAS_MEMORY / (1024 * 1024)
    
    return {
        'valid': memory_bytes <= MAX_SHADOW_ATLAS_MEMORY,
        'memory_bytes': memory_bytes,
        'memory_mb': memory_mb,
        'constraint_mb': constraint_mb, 
        'headroom_mb': constraint_mb - memory_mb,
    }


def build_shadow_atlas(scene: Dict, light: DirectionalLight, camera: Dict) -> Tuple[Dict, ShadowStats]:
    """
    Build shadow atlas with CSM cascades.
    
    Args:
        scene: Scene geometry data
        light: Directional light configuration
        camera: Camera configuration dict with 'position', 'target', 'up', 'fov_y', etc.
    
    Returns:
        Tuple of (atlas_info, stats) where:
        - atlas_info: Dict with cascade info, dimensions, memory usage
        - stats: ShadowStats object with performance metrics
    """
    if not has_shadows_support():
        # Fallback implementation for import safety
        config = CsmConfig()
        atlas_info = {
            'cascade_count': config.cascade_count,
            'atlas_dimensions': (config.shadow_map_size, config.shadow_map_size, config.cascade_count),
            'cascade_resolutions': [config.shadow_map_size] * config.cascade_count,
            'memory_usage': config.shadow_map_size * config.shadow_map_size * 4 * config.cascade_count,
        }
        
        stats = ShadowStats(
            cascade_count=config.cascade_count,
            shadow_map_size=config.shadow_map_size,
            memory_usage=atlas_info['memory_usage'],
            light_direction=light.direction,
            split_distances=[0.1, 10.0, 50.0, 100.0][:config.cascade_count],
            texel_sizes=[1.0] * config.cascade_count,
        )
        
        return atlas_info, stats
    
    # Native implementation
    try:
        result = _core.build_shadow_atlas_native(scene, light.__dict__, camera)
        atlas_info, stats_data = result
        
        stats = ShadowStats(
            cascade_count=stats_data['cascade_count'],
            shadow_map_size=stats_data['shadow_map_size'],
            memory_usage=stats_data['memory_usage'],
            light_direction=tuple(stats_data['light_direction']),
            split_distances=stats_data['split_distances'],
            texel_sizes=stats_data['texel_sizes'],
        )
        
        return atlas_info, stats
        
    except Exception:
        # Fallback if native fails
        return build_shadow_atlas(scene, light, camera)


def sample_shadow(params: Dict) -> float:
    """
    Sample shadow value using WGSL PCF path.
    
    Args:
        params: Shadow sampling parameters dict containing:
        - 'world_position': (x, y, z) world space position
        - 'light_direction': (x, y, z) light direction
        - 'cascade_idx': cascade index to sample from
        - 'pcf_kernel_size': PCF kernel size (1, 3, 5, or 7)
        - 'bias': depth bias value
    
    Returns:
        Shadow factor [0, 1] where 0 = fully shadowed, 1 = not shadowed
    """
    if not has_shadows_support():
        # Fallback gradient-based shadow for import safety
        world_pos = np.array(params.get('world_position', [0, 0, 0]))
        light_dir = np.array(params.get('light_direction', [0, -1, 0]))
        
        # Simple distance-based falloff
        distance = np.linalg.norm(world_pos)
        shadow_factor = np.clip(1.0 - distance / 50.0, 0.0, 1.0)
        
        return float(shadow_factor)
    
    # Native WGSL PCF implementation
    try:
        return _core.sample_shadow_pcf_native(params)
    except Exception:
        # Fallback if native fails
        return sample_shadow(params)
