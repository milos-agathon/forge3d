# Re-export specific items from the compiled extension.
import importlib
try:
    import importlib.resources as resources
except ImportError:
    # Python < 3.9 compatibility
    import importlib_resources as resources

_ext = importlib.import_module("forge3d._forge3d")

# Add files function for typing stubs test compatibility
def files(package):
    """Get package files using importlib.resources.files."""
    return resources.files(package)
Renderer = _ext.Renderer
Scene = _ext.Scene
png_to_numpy = _ext.png_to_numpy
numpy_to_png = _ext.numpy_to_png
enumerate_adapters = _ext.enumerate_adapters
device_probe = _ext.device_probe
colormap_supported = _ext.colormap_supported
camera_look_at = _ext.camera_look_at
camera_perspective = _ext.camera_perspective
camera_view_proj = _ext.camera_view_proj
# Test helper functions for Workstream C validation
c5_build_framegraph_report = _ext.c5_build_framegraph_report
c6_parallel_record_metrics = _ext.c6_parallel_record_metrics
c7_run_compute_prepass = _ext.c7_run_compute_prepass
c9_push_pop_roundtrip = _ext.c9_push_pop_roundtrip
c10_parent_z90_child_unitx_world = _ext.c10_parent_z90_child_unitx_world
__version__ = _ext.__version__
# Convenience functions for backward compatibility
def render_triangle_rgba(width: int, height: int):
    """Render a deterministic triangle and return (H, W, 4) uint8."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    r = Renderer(width, height)
    return r.render_triangle_rgba()

def render_triangle_png(path, width: int, height: int) -> None:
    """Render a deterministic triangle and write it as a PNG file to `path`."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    from pathlib import Path
    path_obj = Path(path)
    if path_obj.suffix.lower() not in ['.png']:
        raise ValueError("Only PNG files are supported")
    r = Renderer(width, height)
    r.render_triangle_png(path_obj)

def make_terrain(width: int, height: int, grid: int = 128):
    """Helper constructor for TerrainSpike (only available when the crate is built with --features terrain_spike)."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    if grid < 2:
        raise ValueError("grid must be >= 2")
    try:
        return TerrainSpike(width, height, grid)  # noqa: F821
    except NameError:
        raise RuntimeError("TerrainSpike unavailable; build crate with --features terrain_spike")

# DEM helper functions
def dem_stats(heightmap):
    import numpy as np
    a = np.asarray(heightmap)
    if a.ndim != 2 or a.dtype not in (np.float32, np.float64) or not a.flags['C_CONTIGUOUS']:
        raise RuntimeError("heightmap must be 2-D float32/float64 and C-contiguous")
    a = a.astype(np.float32, copy=False)
    mn = float(a.min()); mx = float(a.max())
    mean = float(a.mean()); std = float(a.std(dtype=np.float32))
    return mn, mx, mean, std

def dem_normalize(heightmap, *, mode="minmax", out_range=(0.0, 1.0), eps=1e-8, return_stats=False):
    import numpy as np
    mn, mx, mean, std = dem_stats(heightmap)
    a = np.asarray(heightmap).astype(np.float32, copy=False)
    if mode == "minmax":
        lo, hi = map(float, out_range)
        scale = 0.0 if mx == mn else (hi - lo) / max(mx - mn, float(eps))
        out = (a - mn) * scale + lo
    elif mode == "zscore":
        out = (a - mean) / max(std, float(eps))
    else:
        raise ValueError("mode must be 'minmax' or 'zscore'")
    if return_stats:
        return out, (mn, mx, mean, std)
    return out

# T42-BEGIN:grid_generate
def grid_generate(nx: int, nz: int, spacing=(1.0, 1.0), origin: str = "center"):
    """
    Generate a regular grid mesh for heightmaps.

    Args:
        nx, nz (int): Grid dimensions in vertices. Must be >= 2.
        spacing (tuple[float, float]): (dx, dy) world-space spacing. Must be > 0.
        origin ("center" | "corner"): Grid origin convention.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            xy: (N, 2) float32, uv: (N, 2) float32, indices: (M,) uint32
    """
    if not isinstance(nx, int) or not isinstance(nz, int):
        raise TypeError("nx and nz must be integers")
    if nx < 2 or nz < 2:
        raise ValueError("nx and nz must be >= 2")
    try:
        dx, dy = spacing
        dx = float(dx); dy = float(dy)
    except Exception as e:
        raise TypeError("spacing must be a (dx, dy) pair of numbers") from e
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("spacing components must be finite and > 0")
    if origin not in ("center", "corner"):
        raise ValueError("origin must be 'center' or 'corner'")
    # Use the module reference we already have
    import importlib
    _ext = importlib.import_module("forge3d._forge3d")
    return _ext.grid_generate(nx, nz, (dx, dy), origin)
# T42-END:grid_generate

# T52-BEGIN:bench-export
# Import lazily and safely so packaging/import errors don't break base module import
try:
    from .bench import run_benchmark as _vf_run_benchmark
except Exception as _e:  # pragma: no cover
    def _vf_run_benchmark(*_args, **_kwargs):
        raise RuntimeError("forge3d.bench unavailable: " + str(_e))

# Ensure curated __all__ includes 'run_benchmark' without leaking internals
try:
    __all__  # type: ignore[name-defined]
except NameError:
    __all__ = []
if "run_benchmark" not in __all__:
    __all__.append("run_benchmark")

# Public alias
run_benchmark = _vf_run_benchmark
# T52-END:bench-export

# T42-BEGIN:__all__
# Curated public surface (dynamic inclusion of feature-gated TerrainSpike below)
__all__ = [
    "Renderer",
    "Scene",
    "png_to_numpy",
    "numpy_to_png",
    "grid_generate",
    "render_triangle_rgba",
    "render_triangle_png",
    "dem_stats",
    "dem_normalize",
    "enumerate_adapters",
    "device_probe",
    "run_benchmark",
    "files",
    "__version__",
    "c5_build_framegraph_report",
    "c6_parallel_record_metrics",
    "c7_run_compute_prepass",
    "c9_push_pop_roundtrip",
    "c10_parent_z90_child_unitx_world",
]

# If TerrainSpike is compiled in, export it too
try:
    TerrainSpike = _ext.TerrainSpike  # type: ignore[attr-defined]
    __all__.append("TerrainSpike")
except Exception:
    pass

# Clean up private symbols from namespace to prevent leakage
del _ext, importlib
# T42-END:__all__