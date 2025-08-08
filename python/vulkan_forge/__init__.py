# A1.5-BEGIN:vulkan_forge-shim
"""
Thin Python shim for the compiled extension module `_vulkan_forge`.

Public API:
- Renderer(width:int, height:int)
- render_triangle_rgba(width:int, height:int) -> np.ndarray[H,W,4], dtype=uint8
- render_triangle_png(path:str, width:int, height:int) -> None
- Optional (feature 'terrain_spike'): TerrainSpike(width:int, height:int, grid:int=128).render_png(path)
- __version__: str

<!-- T02-BEGIN:doc -->
`Renderer.set_height_range(min, max)` overrides the auto-computed `[h_min, h_max]`
used to normalize heights into `[0, 1]` for colormap & lighting.
<!-- T02-END:doc -->

<!-- T22-BEGIN:doc -->
`Renderer.set_sun(elevation_deg, azimuth_deg)` sets the light direction (degrees).
`Renderer.set_exposure(value)` controls tonemapping exposure (> 0).
<!-- T22-END:doc -->
"""
from __future__ import annotations
import importlib, importlib.util, sys
from ._validate import size_wh, png_path, grid as _grid

def _load_extension():
    # Try top-level mixed-project layout first
    spec = importlib.util.find_spec("_vulkan_forge")
    if spec is not None:
        return importlib.import_module("_vulkan_forge")
    # Then package-local layout
    spec = importlib.util.find_spec("vulkan_forge._vulkan_forge")
    if spec is not None:
        return importlib.import_module("vulkan_forge._vulkan_forge")
    raise ImportError(
        "Failed to import compiled module '_vulkan_forge'. "
        f"python={sys.executable}. Reinstall in THIS venv."
    )

_ext = _load_extension()
Renderer = _ext.Renderer

# IMPORTANT: only define TerrainSpike when the extension actually exposes it.
# When the cargo feature is not built, we do NOT create a TerrainSpike name at all,
# so `hasattr(vulkan_forge, "TerrainSpike")` is False and tests can skip.
if hasattr(_ext, "TerrainSpike"):
    TerrainSpike = _ext.TerrainSpike  # type: ignore[attr-defined]

def render_triangle_rgba(width: int, height: int):
    """Render a deterministic triangle and return (H, W, 4) uint8."""
    w, h = size_wh(width, height)
    r = Renderer(w, h)
    return r.render_triangle_rgba()

def render_triangle_png(path: str, width: int, height: int) -> None:
    """Render a deterministic triangle and write it as a PNG file to `path`."""
    w, h = size_wh(width, height)
    r = Renderer(w, h)
    r.render_triangle_png(png_path(path))

def make_terrain(width: int, height: int, grid: int = 128):
    """
    Helper constructor for TerrainSpike (only available when the crate
    is built with `--features terrain_spike`).
    """
    if "TerrainSpike" not in globals():
        raise RuntimeError("TerrainSpike unavailable; build crate with --features terrain_spike")
    w, h = size_wh(width, height)
    g = _grid(grid)
    return TerrainSpike(w, h, g)  # type: ignore[name-defined]

# Version metadata (best-effort)
try:
    from importlib.metadata import version
    __version__ = version("vulkan-forge")
except Exception:
    try:
        __version__ = version("vulkan_forge")
    except Exception:
        __version__ = "0.0.0.dev0"

# Public export list
__all__ = ["Renderer", "render_triangle_rgba", "render_triangle_png", "make_terrain", "__version__"]
if "TerrainSpike" in globals():
    __all__.append("TerrainSpike")
# A1.5-END:vulkan_forge-shim

# T02-BEGIN:dem-python-helpers
import numpy as _np

def dem_stats(heightmap):
    a = _np.asarray(heightmap)
    if a.ndim != 2 or a.dtype not in (_np.float32, _np.float64) or not a.flags['C_CONTIGUOUS']:
        raise RuntimeError("heightmap must be 2-D float32/float64 and C-contiguous")
    a = a.astype(_np.float32, copy=False)
    mn = float(a.min()); mx = float(a.max())
    mean = float(a.mean()); std = float(a.std(dtype=_np.float32))
    return mn, mx, mean, std

def dem_normalize(heightmap, *, mode="minmax", out_range=(0.0, 1.0), eps=1e-8, return_stats=False):
    mn, mx, mean, std = dem_stats(heightmap)
    a = _np.asarray(heightmap).astype(_np.float32, copy=False)
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
# T02-END:dem-python-helpers

# T02-BEGIN:dem-all
try:
    __all__ += ["dem_stats", "dem_normalize"]
except NameError:
    __all__ = ["dem_stats", "dem_normalize"]
# T02-END:dem-all

# T11-BEGIN:grid-python-helpers
def generate_grid(nx: int, nz: int, spacing=(1.0, 1.0), origin="center"):
    """
    Return (positions[N,3], uvs[N,2], indices[M]) for a regular grid on XZ.
    positions are float32, uvs float32 in [0,1], indices uint32 (triangle list).
    """
    from ._vulkan_forge import grid_generate  # type: ignore
    return grid_generate(int(nx), int(nz), tuple(map(float, spacing)), str(origin))
try:
    __all__ += ["generate_grid"]
except NameError:
    __all__ = ["generate_grid"]
# T11-END:grid-python-helpers
