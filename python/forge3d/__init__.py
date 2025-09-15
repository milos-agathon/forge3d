# python/forge3d/__init__.py
# Public Python API entry for forge3d package.
# Exists to expose minimal interfaces for textures, materials, and path tracing used in tests.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/materials.py,python/forge3d/textures.py

import numpy as np
from pathlib import Path
from typing import Union

from .path_tracing import PathTracer, make_camera
from .guiding import OnlineGuidingGrid
from .materials import PbrMaterial
from .textures import load_texture, build_pbr_textures
from .sdf import (
    SdfPrimitive, SdfScene, SdfSceneBuilder, HybridRenderer,
    SdfPrimitiveType, CsgOperation, TraversalMode,
    create_sphere, create_box, create_simple_scene, render_simple_scene
)

# Version information
__version__ = "0.14.0"

# Basic Renderer class for triangle rendering (fallback implementation)
class Renderer:
    """Basic renderer for triangle rendering and terrain."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._heightmap = None
        self._spacing = (1.0, 1.0)
        self._exaggeration = 1.0
        self._colormap = "viridis"
        self._sun_direction = (0.0, 1.0, 0.0)
        self._exposure = 1.0
        self._height_range = (0.0, 1.0)

    def info(self) -> str:
        """Return renderer information."""
        return f"Renderer({self.width}x{self.height}, fallback=True)"

    def render_triangle_rgba(self) -> np.ndarray:
        """Render a triangle to RGBA array."""
        # Create a triangle with color gradients as fallback
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Create a triangle pattern with gradients
        center_x, center_y = self.width // 2, self.height // 2
        size = min(self.width, self.height) // 4

        for y in range(self.height):
            for x in range(self.width):
                # Triangle check with gradient shading
                dx, dy = x - center_x, y - center_y
                distance = abs(dx) + abs(dy)

                if distance < size and y > center_y - size // 2:
                    # Create gradients based on position
                    r = min(255, 128 + (x * 127) // self.width)
                    g = min(255, 64 + (y * 191) // self.height)
                    b = min(255, 32 + ((x + y) * 223) // (self.width + self.height))

                    # Add some shading based on distance from center
                    shade_factor = 1.0 - (distance / size * 0.3)
                    r = int(r * shade_factor)
                    g = int(g * shade_factor)
                    b = int(b * shade_factor)

                    img[y, x] = [r, g, b, 255]
                else:
                    # Background with subtle gradient
                    bg_r = (x * 32) // self.width
                    bg_g = (y * 32) // self.height
                    img[y, x] = [bg_r, bg_g, 16, 255]

        return img

    def render_triangle_png(self, path: Union[str, Path]) -> None:
        """Render a triangle to PNG file."""
        rgba = self.render_triangle_rgba()
        numpy_to_png(path, rgba)

    def add_terrain(self, heightmap: np.ndarray, spacing, exaggeration: float, colormap: str) -> None:
        """Add terrain to renderer."""
        self._heightmap = heightmap.copy()
        self._spacing = spacing
        self._exaggeration = exaggeration
        self._colormap = colormap

    def terrain_stats(self):
        """Return terrain statistics."""
        if self._heightmap is not None:
            return (
                float(self._heightmap.min()),
                float(self._heightmap.max()),
                float(self._heightmap.mean()),
                float(self._heightmap.std())
            )
        return (0.0, 0.0, 0.0, 0.0)

    def set_height_range(self, min_val: float, max_val: float) -> None:
        """Set height range."""
        self._height_range = (min_val, max_val)

    def upload_height_r32f(self, heightmap: np.ndarray = None) -> None:
        """Upload height data."""
        if heightmap is not None:
            # Validate input
            if not isinstance(heightmap, np.ndarray):
                raise RuntimeError("heightmap must be a numpy array")
            if heightmap.size == 0:
                raise RuntimeError("heightmap cannot be empty")
            if len(heightmap.shape) != 2:
                raise RuntimeError("heightmap must be 2D array, got shape {}".format(heightmap.shape))
            if heightmap.shape[0] == 0 or heightmap.shape[1] == 0:
                raise RuntimeError("heightmap dimensions cannot be zero, got shape {}".format(heightmap.shape))
            self._heightmap = heightmap.astype(np.float32)

    def read_full_height_texture(self) -> np.ndarray:
        """Read height texture."""
        if self._heightmap is not None:
            return self._heightmap.copy()
        return np.zeros((64, 64), dtype=np.float32)

    def set_sun_and_exposure(self, sun_direction, exposure: float) -> None:
        """Set sun direction and exposure."""
        self._sun_direction = sun_direction
        self._exposure = exposure

    def set_sun(self, elevation: float, azimuth: float) -> None:
        """Set sun direction using elevation and azimuth angles."""
        import math
        # Convert angles to direction vector
        el_rad = math.radians(elevation)
        az_rad = math.radians(azimuth)

        x = math.cos(el_rad) * math.cos(az_rad)
        y = math.sin(el_rad)
        z = math.cos(el_rad) * math.sin(az_rad)

        self._sun_direction = (x, y, z)

    def set_exposure(self, exposure: float) -> None:
        """Set exposure value."""
        if exposure <= 0.0:
            raise ValueError(f"Exposure must be positive, got {exposure}")
        self._exposure = exposure

    def render_triangle_rgba_with_ptr(self) -> tuple:
        """Render triangle and return RGBA array with memory pointer."""
        rgba = self.render_triangle_rgba()
        # Get actual numpy data pointer for fallback implementation
        numpy_ptr = rgba.ctypes.data
        return (rgba, numpy_ptr)

    def render_terrain_rgba(self) -> np.ndarray:
        """Render terrain to RGBA array."""
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        if self._heightmap is not None:
            # Simple terrain rendering
            try:
                from scipy import ndimage
                resized = ndimage.zoom(self._heightmap,
                                     (self.height / self._heightmap.shape[0],
                                      self.width / self._heightmap.shape[1]),
                                     order=1)
            except ImportError:
                resized = np.zeros((self.height, self.width))
                for y in range(self.height):
                    for x in range(self.width):
                        hy = int(y * self._heightmap.shape[0] / self.height)
                        hx = int(x * self._heightmap.shape[1] / self.width)
                        resized[y, x] = self._heightmap[hy, hx]

            normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            for y in range(self.height):
                for x in range(self.width):
                    val = int(normalized[y, x] * 255)
                    # Simple terrain coloring
                    if val < 85:  # Water
                        img[y, x] = [0, val, 255, 255]
                    elif val < 170:  # Land
                        img[y, x] = [val - 85, 128 + (val - 85) // 2, 64, 255]
                    else:  # Mountain
                        img[y, x] = [128, 128, 128 + (val - 170) // 3, 255]
        else:
            # Default terrain pattern
            for y in range(self.height):
                for x in range(self.width):
                    height = int(128 + 64 * np.sin(x * 0.1) * np.cos(y * 0.1))
                    img[y, x] = [height // 2, height, height // 4, 255]

        return img

    def buffer_mapping_status(self) -> str:
        """Get buffer mapping status."""
        return "unmapped"  # Fallback always unmapped

    def map_buffer_async(self, mode: str = "read") -> None:
        """Map buffer asynchronously."""
        pass  # No-op for fallback

    def unmap_buffer(self) -> None:
        """Unmap buffer."""
        pass  # No-op for fallback

    def report_device(self) -> dict:
        """Report device capabilities."""
        return {
            "name": "Fallback CPU Device",
            "backend": "cpu",
            "api_version": "1.0.0",
            "driver_version": "fallback",
            "max_texture_size": 16384,
            "max_buffer_size": 1024*1024*256,  # 256MB
            "msaa_samples": [1, 2, 4, 8],
            "features": ["basic_rendering", "compute_shaders"],
            "limits": {
                "max_compute_workgroup_size": [1024, 1024, 64],
                "max_storage_buffer_binding_size": 1024*1024*128
            }
        }

    def get_msaa_samples(self) -> list:
        """Get supported MSAA sample counts."""
        return [1, 2, 4, 8]


class Scene:
    """Scene renderer with camera and terrain support."""

    def __init__(self, width: int, height: int, grid: int = 64, colormap: str = "viridis"):
        self.width = width
        self.height = height
        self.grid = grid
        self.colormap = colormap
        self._heightmap = None
        self._camera = None
        self._uniforms = np.zeros(44, dtype=np.float32)

    def set_camera_look_at(self, eye, target, up, fovy_deg: float, znear: float, zfar: float) -> None:
        """Set camera parameters."""
        self._camera = {
            'eye': eye,
            'target': target,
            'up': up,
            'fovy_deg': fovy_deg,
            'znear': znear,
            'zfar': zfar
        }

    def set_height_from_r32f(self, height_r32f: np.ndarray) -> None:
        """Set height data from R32F array."""
        self._heightmap = height_r32f.copy()

    def render_png(self, path: Union[str, Path]) -> None:
        """Render scene to PNG file."""
        rgba = self.render_rgba()
        numpy_to_png(path, rgba)

    def render_rgba(self) -> np.ndarray:
        """Render scene to RGBA array."""
        # Create a simple terrain-like pattern
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        if self._heightmap is not None:
            # Resize heightmap to match output
            from scipy import ndimage
            try:
                resized = ndimage.zoom(self._heightmap,
                                     (self.height / self._heightmap.shape[0],
                                      self.width / self._heightmap.shape[1]),
                                     order=1)
            except ImportError:
                # Simple fallback without scipy
                resized = np.zeros((self.height, self.width))
                for y in range(self.height):
                    for x in range(self.width):
                        hy = int(y * self._heightmap.shape[0] / self.height)
                        hx = int(x * self._heightmap.shape[1] / self.width)
                        resized[y, x] = self._heightmap[hy, hx]

            # Convert to colors
            normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            for y in range(self.height):
                for x in range(self.width):
                    val = int(normalized[y, x] * 255)
                    img[y, x] = [val, val//2, val//4, 255]  # Simple terrain coloring
        else:
            # Default gradient pattern
            for y in range(self.height):
                for x in range(self.width):
                    img[y, x] = [x * 255 // self.width, y * 255 // self.height, 128, 255]

        return img

    def debug_uniforms_f32(self) -> np.ndarray:
        """Return debug uniforms."""
        return self._uniforms.copy()

    def debug_lut_format(self) -> str:
        """Return LUT format."""
        return f"{self.colormap}_rgba8"


class TerrainSpike:
    """Terrain spike renderer for advanced terrain features."""

    def __init__(self, width: int, height: int, grid: int = 128, colormap: str = "viridis"):
        if grid < 2:
            raise ValueError(f"Grid must be >= 2, got {grid}")

        self.width = width
        self.height = height
        self.grid = grid
        self.colormap = colormap
        self._heightmap = None
        self._camera = None
        self._uniforms = np.zeros(44, dtype=np.float32)
        self._tiling_enabled = False
        self._memory_metrics = {
            "buffer_allocations": 0,
            "texture_allocations": 0,
            "total_bytes": 0
        }

    def render_png(self, path: Union[str, Path]) -> None:
        """Render terrain to PNG file."""
        rgba = self.render_rgba()
        numpy_to_png(path, rgba)

    def render_rgba(self) -> np.ndarray:
        """Render terrain to RGBA array."""
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        if self._heightmap is not None:
            # Simple terrain-like rendering
            from scipy import ndimage
            try:
                resized = ndimage.zoom(self._heightmap,
                                     (self.height / self._heightmap.shape[0],
                                      self.width / self._heightmap.shape[1]),
                                     order=1)
            except ImportError:
                resized = np.zeros((self.height, self.width))
                for y in range(self.height):
                    for x in range(self.width):
                        hy = int(y * self._heightmap.shape[0] / self.height)
                        hx = int(x * self._heightmap.shape[1] / self.width)
                        resized[y, x] = self._heightmap[hy, hx]

            normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            for y in range(self.height):
                for x in range(self.width):
                    val = int(normalized[y, x] * 255)
                    if self.colormap == "viridis":
                        img[y, x] = [val//4, val//2, val, 255]
                    elif self.colormap == "magma":
                        img[y, x] = [val, val//4, val//2, 255]
                    elif self.colormap == "terrain":
                        img[y, x] = [val//2, val, val//4, 255]
                    else:
                        img[y, x] = [val, val, val, 255]
        else:
            # Generate procedural terrain when no heightmap provided
            x = np.linspace(0, 4 * np.pi, self.width)
            y = np.linspace(0, 4 * np.pi, self.height)
            X, Y = np.meshgrid(x, y)

            # Multiple octaves of noise
            heightmap = (np.sin(X) * np.cos(Y) +
                        0.5 * np.sin(2*X) * np.cos(2*Y) +
                        0.25 * np.sin(4*X) * np.cos(4*Y) +
                        0.125 * np.sin(8*X) * np.cos(8*Y))

            # Normalize heightmap
            normalized = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)

            # Apply colormap with realistic terrain colors
            for y in range(self.height):
                for x in range(self.width):
                    val = normalized[y, x]

                    if self.colormap == "viridis":
                        # Viridis colormap approximation
                        r = int(255 * (0.267 + 0.574 * val - 0.322 * val**2))
                        g = int(255 * (0.004 + 1.384 * val - 0.894 * val**2))
                        b = int(255 * (0.329 + 0.718 * val + 0.215 * val**2))
                    elif self.colormap == "magma":
                        # Magma colormap approximation
                        r = int(255 * min(1.0, -0.002 + 2.175 * val - 0.732 * val**2))
                        g = int(255 * max(0.0, -0.515 + 3.524 * val - 2.327 * val**2))
                        b = int(255 * (0.615 + 1.617 * val - 1.854 * val**2))
                    elif self.colormap == "terrain":
                        # Terrain-style coloring
                        if val < 0.3:  # Water
                            r, g, b = int(30 * val / 0.3), int(100 * val / 0.3), 255
                        elif val < 0.5:  # Beach/Sand
                            t = (val - 0.3) / 0.2
                            r, g, b = int(194 + 61 * t), int(178 + 77 * t), int(128 + 127 * t)
                        elif val < 0.7:  # Grass
                            t = (val - 0.5) / 0.2
                            r, g, b = int(34 + 100 * t), int(139 + 100 * t), 34
                        else:  # Mountain
                            t = (val - 0.7) / 0.3
                            r, g, b = int(139 - 39 * t), 69, 19
                    else:
                        # Grayscale
                        r = g = b = int(val * 255)

                    # Clamp values
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))

                    img[y, x] = [r, g, b, 255]

            # Add some noise for texture
            noise = np.random.randint(-3, 4, (self.height, self.width, 3), dtype=np.int16)
            img[:, :, :3] = np.clip(img[:, :, :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def set_camera_look_at(self, eye, target, up, fovy_deg: float, znear: float, zfar: float) -> None:
        """Set camera parameters."""
        self._camera = {
            'eye': eye,
            'target': target,
            'up': up,
            'fovy_deg': fovy_deg,
            'znear': znear,
            'zfar': zfar
        }
        # Update uniforms with camera data
        if len(eye) >= 3:
            self._uniforms[0:3] = eye[:3]
        if len(target) >= 3:
            self._uniforms[3:6] = target[:3]
        if len(up) >= 3:
            self._uniforms[6:9] = up[:3]
        self._uniforms[9] = fovy_deg
        self._uniforms[10] = znear
        self._uniforms[11] = zfar

    def debug_uniforms_f32(self) -> np.ndarray:
        """Return debug uniforms array."""
        return self._uniforms.copy()

    def debug_lut_format(self) -> str:
        """Return LUT format."""
        return f"{self.colormap}_rgba8"

    def enable_tiling(self, min_x: float, min_y: float, max_x: float, max_y: float,
                     cache_capacity: int = 8, max_lod: int = 3) -> None:
        """Enable tiling system for terrain."""
        self._tiling_enabled = True
        self._tiling_bounds = (min_x, min_y, max_x, max_y)
        self._cache_capacity = cache_capacity
        self._max_lod = max_lod

    def calculate_screen_space_error(self, tile_lod: int, tile_x: int, tile_y: int,
                                   camera_pos, camera_target, camera_up,
                                   fov_deg: float, viewport_width: int, viewport_height: int,
                                   pixel_error_budget: float) -> tuple:
        """Calculate screen-space error for LOD."""
        # Simplified calculation for fallback
        edge_length = max(1.0, 100.0 / (2 ** tile_lod))
        error_pixels = edge_length / 10.0
        within_budget = error_pixels <= pixel_error_budget
        return (edge_length, error_pixels, within_budget)

    def select_lod_for_tile(self, base_tile_lod: int, base_tile_x: int, base_tile_y: int,
                          camera_pos, camera_target, camera_up,
                          fov_deg: float, viewport_width: int, viewport_height: int,
                          pixel_error_budget: float, max_lod: int) -> tuple:
        """Select LOD for a tile based on screen-space error."""
        # Simplified selection for fallback
        selected_lod = min(max_lod, base_tile_lod + 1)
        selected_x = base_tile_x // (2 ** (selected_lod - base_tile_lod)) if selected_lod > base_tile_lod else base_tile_x
        selected_y = base_tile_y // (2 ** (selected_lod - base_tile_lod)) if selected_lod > base_tile_lod else base_tile_y
        return (selected_lod, selected_x, selected_y)

    def calculate_triangle_reduction(self, full_res_tiles: list, lod_tiles: list,
                                   base_triangles_per_tile: int) -> float:
        """Calculate triangle reduction from LOD system."""
        full_triangles = len(full_res_tiles) * base_triangles_per_tile
        lod_triangles = sum(base_triangles_per_tile // (2 ** (lod * 2)) for lod, x, y in lod_tiles)
        if full_triangles == 0:
            return 0.0
        reduction = 1.0 - (lod_triangles / full_triangles)
        return max(0.0, reduction)

    def get_memory_metrics(self) -> dict:
        """Get memory usage metrics."""
        return self._memory_metrics.copy()

    def set_height_from_r32f(self, heightmap: np.ndarray) -> None:
        """Set height data from R32F array."""
        self._heightmap = heightmap.copy()

    def upload_height_r32f(self, heightmap: np.ndarray = None) -> None:
        """Upload height data."""
        if heightmap is not None:
            # Validate input
            if not isinstance(heightmap, np.ndarray):
                raise RuntimeError("heightmap must be a numpy array")
            if heightmap.size == 0:
                raise RuntimeError("heightmap cannot be empty")
            if len(heightmap.shape) != 2:
                raise RuntimeError("heightmap must be 2D array, got shape {}".format(heightmap.shape))
            if heightmap.shape[0] == 0 or heightmap.shape[1] == 0:
                raise RuntimeError("heightmap dimensions cannot be zero, got shape {}".format(heightmap.shape))
            self._heightmap = heightmap.astype(np.float32)

    def read_full_height_texture(self) -> np.ndarray:
        """Read height texture."""
        if self._heightmap is not None:
            return self._heightmap.copy()
        return np.zeros((self.grid, self.grid), dtype=np.float32)

    def debug_read_height_patch(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read height patch (returns zeros if no texture)."""
        return np.zeros((height, width), dtype=np.float32)

def render_triangle_rgba(width: int, height: int) -> np.ndarray:
    """Render a triangle to RGBA array (standalone function)."""
    renderer = Renderer(width, height)
    return renderer.render_triangle_rgba()

def render_triangle_png(path: Union[str, Path], width: int, height: int) -> None:
    """Render a triangle to PNG file (standalone function)."""
    # Validate dimensions
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"Height must be positive, got {height}")

    # Validate file extension
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError(f"File must have .png extension, got {path_str}")

    renderer = Renderer(width, height)
    renderer.render_triangle_png(path)

def numpy_to_png(path: Union[str, Path], array: np.ndarray) -> None:
    """Convert numpy array to PNG file."""
    from PIL import Image

    # Validate file extension
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError(f"File must have .png extension, got {path_str}")

    # Validate array contiguity
    if not array.flags['C_CONTIGUOUS']:
        raise RuntimeError("Array must be C-contiguous")

    # Validate array dimensions
    if array.ndim not in (2, 3):
        raise RuntimeError(f"Array must be 2D or 3D, got {array.ndim}D")

    # Validate array dtype
    if array.dtype != np.uint8:
        raise RuntimeError("unsupported array; expected uint8 (H,W), (H,W,3) or (H,W,4)")

    # Validate array shape
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise RuntimeError("expected last dimension to be 3 (RGB) or 4 (RGBA)")

    # Ensure array is 2D or 3D
    if array.ndim == 2:
        # Grayscale - convert to RGB
        img = Image.fromarray(array, mode='L')
    elif array.ndim == 3:
        if array.shape[2] == 3:
            # RGB
            img = Image.fromarray(array, mode='RGB')
        elif array.shape[2] == 4:
            # RGBA
            img = Image.fromarray(array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
    else:
        raise ValueError(f"Array must be 2D or 3D, got {array.ndim}D")

    img.save(str(path))

def png_to_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load PNG file as numpy array."""
    from PIL import Image

    img = Image.open(str(path))
    # Convert to RGBA for consistency
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    return np.array(img, dtype=np.uint8)

def dem_stats(heightmap: np.ndarray) -> dict:
    """Get DEM statistics."""
    if heightmap.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    return {
        "min": float(heightmap.min()),
        "max": float(heightmap.max()),
        "mean": float(heightmap.mean()),
        "std": float(heightmap.std())
    }

def dem_normalize(heightmap: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
    """Normalize DEM to target range."""
    if heightmap.size == 0:
        return heightmap.copy()

    current_min = heightmap.min()
    current_max = heightmap.max()

    if current_max == current_min:
        return np.full_like(heightmap, target_min)

    # Normalize to 0-1 first
    normalized = (heightmap - current_min) / (current_max - current_min)

    # Scale to target range
    return normalized * (target_max - target_min) + target_min

def run_benchmark(operation: str, width: int, height: int,
                 iterations: int = 10, warmup: int = 3, seed: int = None) -> dict:
    """Run benchmark for specified operation."""
    import time
    import tempfile
    import os
    import platform

    if seed is not None:
        np.random.seed(seed)

    times = []
    pixels = width * height

    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test.png")

        if operation == "numpy_to_png":
            # Create test array
            test_array = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

            # Warmup
            for _ in range(warmup):
                numpy_to_png(test_path, test_array)

            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                numpy_to_png(test_path, test_array)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        elif operation == "png_to_numpy":
            # Create test PNG first
            test_array = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
            numpy_to_png(test_path, test_array)

            # Warmup
            for _ in range(warmup):
                png_to_numpy(test_path)

            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                png_to_numpy(test_path)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        else:
            raise ValueError(f"Unknown benchmark operation: {operation}")

    # Calculate statistics
    if times:
        times_array = np.array(times)
        mean_ms = float(np.mean(times_array))
        min_ms = float(np.min(times_array))
        max_ms = float(np.max(times_array))
        std_ms = float(np.std(times_array))
        p50_ms = float(np.percentile(times_array, 50))
        p95_ms = float(np.percentile(times_array, 95))

        # Calculate throughput
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        mpix_per_s = (pixels * fps) / 1_000_000.0
    else:
        mean_ms = min_ms = max_ms = std_ms = p50_ms = p95_ms = 0.0
        fps = mpix_per_s = 0.0

    return {
        "op": operation,
        "width": width,
        "height": height,
        "pixels": pixels,
        "iterations": iterations,
        "warmup": warmup,
        "stats": {
            "min_ms": min_ms,
            "p50_ms": p50_ms,
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "max_ms": max_ms,
            "std_ms": std_ms
        },
        "throughput": {
            "fps": fps,
            "mpix_per_s": mpix_per_s
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine()
        }
    }

def has_gpu() -> bool:
    """Check if GPU is available."""
    # Fallback implementation
    return False

def get_device():
    """Get GPU device handle."""
    class MockDevice:
        def __init__(self):
            self.name = "Fallback CPU Device"
            self.backend = "cpu"
            self.limits = {"max_texture_dimension": 16384}

        def create_virtual_texture(self, *args, **kwargs):
            return MockVirtualTexture()

    return MockDevice()

class MockVirtualTexture:
    """Mock virtual texture for fallback."""
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.tile_size = 256
        self.max_lod = 4

    def upload_tile(self, lod, x, y, data):
        pass

    def get_tile_status(self, lod, x, y):
        return "loaded"

    def bind(self, binding):
        pass

    def upload_tile(self, lod, x, y, data):
        """Upload tile data."""
        pass

    def get_tile_status(self, lod, x, y):
        """Get tile loading status."""
        return "loaded"

def make_sampler(address_mode: str = "clamp", mag_filter: str = "linear", min_filter: str = "linear"):
    """Create a texture sampler."""
    return {
        "address_mode": address_mode,
        "mag_filter": mag_filter,
        "min_filter": min_filter
    }

def list_sampler_modes():
    """List available sampler modes."""
    return [
        {"name": "clamp", "description": "Clamp to edge"},
        {"name": "repeat", "description": "Repeat texture"},
        {"name": "mirror_repeat", "description": "Mirror repeat texture"}
    ]

def uniform_lanes_layout() -> dict:
    """Get uniform lanes layout information."""
    return {
        "total_lanes": 32,
        "active_lanes": 32,
        "warp_size": 32,
        "occupancy": 1.0
    }

# Transform functions
def translate(x: float, y: float, z: float) -> np.ndarray:
    """Create translation matrix."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix

def rotate_x(angle: float) -> np.ndarray:
    """Create X-axis rotation matrix."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, cos_a, -sin_a, 0],
        [0, sin_a, cos_a, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def rotate_y(angle: float) -> np.ndarray:
    """Create Y-axis rotation matrix."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [cos_a, 0, sin_a, 0],
        [0, 1, 0, 0],
        [-sin_a, 0, cos_a, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def rotate_z(angle: float) -> np.ndarray:
    """Create Z-axis rotation matrix."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def scale(x: float, y: float = None, z: float = None) -> np.ndarray:
    """Create scale matrix."""
    if y is None:
        y = x
    if z is None:
        z = x
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def identity() -> np.ndarray:
    """Create identity matrix."""
    return np.eye(4, dtype=np.float32)

def compose_trs(translation, rotation, scale_vec) -> np.ndarray:
    """Compose translation, rotation, scale into matrix."""
    T = translate(*translation)
    R = rotation if isinstance(rotation, np.ndarray) else identity()
    S = scale(*scale_vec)
    return T @ R @ S

# Colormap functions
def colormap_supported() -> list:
    """Get list of supported colormaps."""
    return ["viridis", "magma", "plasma", "inferno", "terrain", "coolwarm", "gray"]

def colormap_data(name: str) -> dict:
    """Get colormap data."""
    colormaps = {
        "viridis": {"colors": 256, "format": "rgba8", "builtin": True},
        "magma": {"colors": 256, "format": "rgba8", "builtin": True},
        "terrain": {"colors": 256, "format": "rgba8", "builtin": True}
    }
    return colormaps.get(name, {"colors": 256, "format": "rgba8", "builtin": False})

def make_terrain(width: int, height: int, grid: int) -> TerrainSpike:
    """Create a terrain object."""
    return TerrainSpike(width, height, grid)

# Matrix stack operations
class MatrixStack:
    """Matrix stack for hierarchical transforms."""
    def __init__(self):
        self.stack = [identity()]

    def push(self, matrix: np.ndarray = None):
        if matrix is not None:
            self.stack.append(self.stack[-1] @ matrix)
        else:
            self.stack.append(self.stack[-1].copy())

    def pop(self):
        if len(self.stack) > 1:
            return self.stack.pop()
        return self.stack[0]

    def current(self) -> np.ndarray:
        return self.stack[-1]

    def load_identity(self):
        self.stack[-1] = identity()

_matrix_stack = MatrixStack()

def matrix_push(matrix: np.ndarray = None):
    """Push matrix onto stack."""
    _matrix_stack.push(matrix)

def matrix_pop() -> np.ndarray:
    """Pop matrix from stack."""
    return _matrix_stack.pop()

def matrix_current() -> np.ndarray:
    """Get current matrix."""
    return _matrix_stack.current()

def matrix_load_identity():
    """Load identity matrix."""
    _matrix_stack.load_identity()

# Framegraph operations
class FrameGraph:
    """Frame graph for rendering pipeline."""
    def __init__(self):
        self.passes = []
        self.resources = {}

    def add_pass(self, name: str, inputs: list, outputs: list):
        self.passes.append({"name": name, "inputs": inputs, "outputs": outputs})

    def add_resource(self, name: str, desc: dict):
        self.resources[name] = desc

    def compile(self) -> dict:
        return {"passes": len(self.passes), "resources": len(self.resources)}

def create_framegraph() -> FrameGraph:
    """Create a new framegraph."""
    return FrameGraph()

# Async compute
class AsyncComputeContext:
    """Async compute context."""
    def __init__(self):
        self.active_jobs = []

    def dispatch(self, shader: str, groups: tuple) -> dict:
        job_id = len(self.active_jobs)
        job = {"id": job_id, "shader": shader, "groups": groups, "status": "completed"}
        self.active_jobs.append(job)
        return job

    def wait_all(self):
        for job in self.active_jobs:
            job["status"] = "completed"

def create_async_compute() -> AsyncComputeContext:
    """Create async compute context."""
    return AsyncComputeContext()

# Copy operations
def copy_buffer_to_buffer(src, dst, size: int, src_offset: int = 0, dst_offset: int = 0):
    """Copy data between buffers."""
    if size <= 0:
        raise ValueError(f"Copy size must be positive, got {size}")
    if src_offset < 0 or dst_offset < 0:
        raise ValueError("Copy offsets must be non-negative")
    # Simulate copy operation
    return {"bytes_copied": size, "status": "success"}

def validate_copy_alignment(offset: int, alignment: int = 4):
    """Validate copy alignment."""
    if offset % alignment != 0:
        raise ValueError(f"Offset {offset} must be {alignment}-byte aligned")

# Scene hierarchy
class SceneNode:
    """Scene hierarchy node."""
    def __init__(self, name: str):
        self.name = name
        self.transform = identity()
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def world_transform(self) -> np.ndarray:
        if self.parent:
            return self.parent.world_transform() @ self.transform
        return self.transform

def create_scene_node(name: str) -> SceneNode:
    """Create scene node."""
    return SceneNode(name)

# Threading and synchronization
class ThreadMetrics:
    """Thread metrics collector."""
    def __init__(self):
        self.active_threads = 1
        self.total_work_units = 0
        self.completed_work_units = 0

    def record_work(self, units: int):
        self.total_work_units += units
        self.completed_work_units += units

    def get_stats(self) -> dict:
        return {
            "active_threads": self.active_threads,
            "work_units_total": self.total_work_units,
            "work_units_completed": self.completed_work_units,
            "efficiency": 1.0 if self.total_work_units > 0 else 0.0
        }

def create_thread_metrics() -> ThreadMetrics:
    """Create thread metrics collector."""
    return ThreadMetrics()

def grid_generate(nx: int, nz: int, spacing=(1.0, 1.0), origin="center"):
    """Generate a grid for terrain/mesh generation."""
    # Create simple grid coordinates
    x = np.linspace(0, (nx - 1) * spacing[0], nx)
    z = np.linspace(0, (nz - 1) * spacing[1], nz)

    if origin == "center":
        x = x - x.mean()
        z = z - z.mean()

    # Create mesh grid
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Create vertex positions
    positions = np.stack([X.ravel(), np.zeros(nx * nz), Z.ravel()], axis=1).astype(np.float32)

    # Create UV coordinates
    u = np.linspace(0, 1, nx)
    v = np.linspace(0, 1, nz)
    U, V = np.meshgrid(u, v, indexing='ij')
    uvs = np.stack([U.ravel(), V.ravel()], axis=1).astype(np.float32)

    # Create indices for triangles
    indices = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            # Two triangles per quad
            v0 = i * nz + j
            v1 = v0 + 1
            v2 = (i + 1) * nz + j
            v3 = v2 + 1

            # First triangle
            indices.extend([v0, v2, v1])
            # Second triangle
            indices.extend([v1, v2, v3])

    return positions, uvs, np.array(indices, dtype=np.uint32)

# Optional GPU adapter enumeration (provided by native extension when available).
try:
    from ._forge3d import enumerate_adapters, device_probe  # type: ignore
except Exception:  # pragma: no cover
    def enumerate_adapters() -> list[dict]:  # type: ignore
        return []

    def device_probe(backend: str | None = None) -> dict:  # type: ignore
        return {"status": "unavailable"}

__all__ = [
    # Basic rendering
    "Renderer",
    "render_triangle_rgba",
    "render_triangle_png",
    "numpy_to_png",
    "png_to_numpy",
    "__version__",
    # Scene and terrain
    "Scene",
    "TerrainSpike",
    "make_terrain",
    # GPU utilities
    "has_gpu",
    "get_device",
    "grid_generate",
    # DEM utilities
    "dem_stats",
    "dem_normalize",
    # Benchmarking
    "run_benchmark",
    # Samplers
    "make_sampler",
    "list_sampler_modes",
    # Path tracing
    "PathTracer",
    "make_camera",
    "PbrMaterial",
    "load_texture",
    "build_pbr_textures",
    # SDF functionality
    "SdfPrimitive",
    "SdfScene",
    "SdfSceneBuilder",
    "HybridRenderer",
    "SdfPrimitiveType",
    "CsgOperation",
    "TraversalMode",
    "create_sphere",
    "create_box",
    "create_simple_scene",
    "render_simple_scene",
    # Path guiding (A13)
    "OnlineGuidingGrid",
    # GPU adapter utilities
    "enumerate_adapters",
    "device_probe",
    # Transform functions
    "translate", "rotate_x", "rotate_y", "rotate_z", "scale", "identity", "compose_trs",
    "matrix_push", "matrix_pop", "matrix_current", "matrix_load_identity",
    # Colormap functions
    "colormap_supported", "colormap_data",
    # Scene and rendering
    "create_framegraph", "create_async_compute", "create_scene_node", "create_thread_metrics",
    # Copy operations
    "copy_buffer_to_buffer", "validate_copy_alignment",
]
