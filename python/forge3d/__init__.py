# python/forge3d/__init__.py
# Public Python API for forge3d terrain renderer
"""
forge3d - GPU-accelerated terrain rendering library.

Core API:
    render_raster       - High-level terrain rendering from DEM
    TerrainRenderer     - Native GPU terrain renderer
    Renderer            - Fallback CPU renderer
    
Configuration:
    TerrainRenderParams - Terrain rendering parameters
    RendererConfig      - Renderer configuration
    
Utilities:
    numpy_to_png        - Save numpy array as PNG
    png_to_numpy        - Load PNG as numpy array
    has_gpu             - Check GPU availability
"""

__version__ = "1.11.0"
version = __version__

import numpy as np

# -----------------------------------------------------------------------------
# Native module loading
# -----------------------------------------------------------------------------
from ._native import (
    NATIVE_AVAILABLE as _NATIVE_AVAILABLE,
    get_native_module as _get_native_module,
)
from ._gpu import (
    enumerate_adapters,
    device_probe,
    has_gpu,
    get_device,
)
from .mem import (
    memory_metrics,
    budget_remaining,
    utilization_ratio,
    override_memory_limit,
)

_NATIVE_MODULE = _get_native_module()

# -----------------------------------------------------------------------------
# Native exports (when available)
# -----------------------------------------------------------------------------
if _NATIVE_MODULE is not None:
    for _name in (
        "Scene",
        "Session",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "TerrainRenderParams",
        "TerrainRenderer",
        "Light",
        "Atmosphere",
        "open_viewer",
        "open_terrain_viewer",
        "PickResult",  # Feature B: Picking system (Plan 1)
        "TerrainQueryResult",  # Feature B: Plan 2
        "SelectionStyle",  # Feature B: Plan 2
        "RichPickResult",  # Feature B: Plan 3
        "HighlightStyle",  # Feature B: Plan 3
        "LassoState",  # Feature B: Plan 3
        "HeightfieldHit",  # Feature B: Plan 3
        "CameraAnimation",  # Feature C: Camera animation (Plan 1 MVP)
        "CameraState",  # Feature C: Camera animation (Plan 1 MVP)
        "SunPosition",  # P0.3/M2: Sun ephemeris
        "sun_position",  # P0.3/M2: Sun ephemeris function
        "sun_position_utc",  # P0.3/M2: Sun ephemeris function (components)
        "ClipmapConfig",  # P2.1/M5: Clipmap terrain
        "ClipmapMesh",  # P2.1/M5: Clipmap terrain
        "clipmap_generate_py",  # P2.1/M5: Clipmap generation function
        "calculate_triangle_reduction_py",  # P2.1/M5: Triangle reduction calculation
    ):
        if hasattr(_NATIVE_MODULE, _name):
            globals()[_name] = getattr(_NATIVE_MODULE, _name)

# -----------------------------------------------------------------------------
# Colormaps
# -----------------------------------------------------------------------------
from .colormaps import (
    get as get_colormap,
    available as available_colormaps,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from .config import RendererConfig, load_renderer_config
from .terrain_params import (
    TerrainRenderParams as TerrainRenderParamsConfig,
    LightSettings,
    IblSettings,
    ShadowSettings,
    FogSettings,
    ReflectionSettings,
    HeightAoSettings,
    SunVisibilitySettings,
    DetailSettings,
    PomSettings,
    TriplanarSettings,
    LodSettings,
    SamplingSettings,
    ClampSettings,
)
from . import presets
from . import animation

# -----------------------------------------------------------------------------
# Core rendering API
# -----------------------------------------------------------------------------
from .render import render_raster, render_polygons, render_raytrace_mesh
from .path_tracing import PathTracer, make_camera

# -----------------------------------------------------------------------------
# Interactive Viewer API
# -----------------------------------------------------------------------------
from .viewer import open_viewer, open_viewer_async, ViewerHandle

# -----------------------------------------------------------------------------
# Fallback Renderer class
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Mapping, Sequence


class Renderer:
    """Fallback CPU renderer for terrain.
    
    Args:
        width: Output image width in pixels
        height: Output image height in pixels
        config: Optional renderer configuration
        **kwargs: Override keywords (brdf, shadows, etc.)
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        config: RendererConfig | Mapping[str, Any] | str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        from .config import split_renderer_overrides
        
        self.width = int(width)
        self.height = int(height)
        overrides, remaining = split_renderer_overrides(dict(kwargs))
        if remaining:
            raise TypeError(f"Unexpected arguments: {', '.join(sorted(str(k) for k in remaining))}")
        self._config = load_renderer_config(config, overrides)
        self._exposure = float(self._config.lighting.exposure)

    def get_config(self) -> dict:
        """Return renderer configuration as dict."""
        return self._config.to_dict()

    def apply_preset(self, name: str, **overrides: Any) -> None:
        """Apply a preset to the renderer configuration."""
        preset_map = presets.get(name)
        self._config = RendererConfig.from_mapping(preset_map, self._config)
        if overrides:
            self._config = load_renderer_config(self._config, overrides)

    def render_triangle_rgba(self) -> np.ndarray:
        """Render a basic triangle pattern (fallback test method)."""
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2
        size = min(self.width, self.height) // 4
        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - cx, y - cy
                if abs(dx) + abs(dy) < size and y > cy - size // 2:
                    img[y, x] = [128, 64, 32, 255]
                else:
                    img[y, x] = [16, 16, 24, 255]
        return img

    def render_triangle_png(self, path) -> None:
        """Render triangle to PNG file."""
        numpy_to_png(path, self.render_triangle_rgba())


# -----------------------------------------------------------------------------
# Image I/O utilities
# -----------------------------------------------------------------------------
def numpy_to_png(path, array: np.ndarray) -> None:
    """Save numpy array as PNG file."""
    from PIL import Image
    
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError(f"File must have .png extension, got {path_str}")
    
    arr = np.ascontiguousarray(array)
    if arr.dtype != np.uint8:
        raise RuntimeError("Array must be uint8")
    
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode='RGB')
    elif arr.ndim == 3 and arr.shape[2] == 4:
        img = Image.fromarray(arr, mode='RGBA')
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    
    img.save(path_str)


def png_to_numpy(path) -> np.ndarray:
    """Load PNG file as numpy array."""
    from PIL import Image
    
    img = Image.open(str(path))
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
        "std": float(heightmap.std()),
    }


# -----------------------------------------------------------------------------
# Geometry module
# -----------------------------------------------------------------------------
from . import geometry
from . import io

# -----------------------------------------------------------------------------
# P4: Map Plate / Creator Workflow
# -----------------------------------------------------------------------------
from .map_plate import MapPlate, MapPlateConfig, BBox, PlateRegion
from .legend import Legend, LegendConfig
from .scale_bar import ScaleBar, ScaleBarConfig
from .north_arrow import NorthArrow, NorthArrowConfig

# -----------------------------------------------------------------------------
# Helpers (offscreen rendering, frame dumping)
# -----------------------------------------------------------------------------
from .helpers.offscreen import (
    render_offscreen_rgba,
    save_png_deterministic,
    rgba_to_png_bytes,
)
from .helpers.frame_dump import FrameDumper, dump_frame_sequence

# -----------------------------------------------------------------------------
# Scene Bundle (.forge3d)
# -----------------------------------------------------------------------------
from .bundle import (
    save_bundle,
    load_bundle,
    is_bundle,
    BundleManifest,
    LoadedBundle,
    CameraBookmark,
    TerrainMeta,
    BUNDLE_VERSION,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # Version
    "__version__",
    "version",
    # Core rendering
    "render_raster",
    "render_polygons", 
    "render_raytrace_mesh",
    "Renderer",
    "PathTracer",
    "make_camera",
    # Native types (when available)
    "Scene",
    "Session",
    "Colormap1D",
    "MaterialSet",
    "IBL",
    "OverlayLayer",
    "TerrainRenderParams",
    "TerrainRenderer",
    # P0.3/M2: Sun ephemeris
    "SunPosition",
    "sun_position",
    "sun_position_utc",
    # Configuration
    "RendererConfig",
    "TerrainRenderParamsConfig",
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "PomSettings",
    "TriplanarSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "presets",
    # Colormaps
    "get_colormap",
    "available_colormaps",
    # GPU utilities
    "has_gpu",
    "get_device",
    "enumerate_adapters",
    "device_probe",
    "memory_metrics",
    "budget_remaining",
    # Image I/O
    "numpy_to_png",
    "png_to_numpy",
    "dem_stats",
    # Helpers
    "render_offscreen_rgba",
    "save_png_deterministic",
    "FrameDumper",
    # Modules
    "geometry",
    "io",
    # P4: Map Plate / Creator Workflow
    "MapPlate",
    "MapPlateConfig",
    "BBox",
    "PlateRegion",
    "Legend",
    "LegendConfig",
    "ScaleBar",
    "ScaleBarConfig",
    "NorthArrow",
    "NorthArrowConfig",
    # Viewer utilities
    "viewer_ipc",
    "colors",
    "interactive",
    # Scene Bundle
    "save_bundle",
    "load_bundle",
    "is_bundle",
    "BundleManifest",
    "LoadedBundle",
    "CameraBookmark",
    "TerrainMeta",
    "BUNDLE_VERSION",
]
