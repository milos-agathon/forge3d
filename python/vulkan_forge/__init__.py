# A1.5-BEGIN:vulkan_forge-shim
"""
Thin Python shim for the compiled extension module `_vulkan_forge`.

Exports:
- Renderer: core class (always)
- TerrainSpike: optional (only when built with `--features terrain_spike`)
- render_triangle_rgba(width, height) -> array (H,W,4) uint8
- render_triangle_png(path, width, height) -> None
- __version__: best-effort package version
"""
from __future__ import annotations
import importlib, importlib.util, sys

def _load_extension():
    # Try top-level first (common for maturin mixed projects)
    spec = importlib.util.find_spec("_vulkan_forge")
    if spec is not None:
        return importlib.import_module("_vulkan_forge")

    # Fallback: package-local extension
    spec = importlib.util.find_spec("vulkan_forge._vulkan_forge")
    if spec is not None:
        return importlib.import_module("vulkan_forge._vulkan_forge")

    raise ImportError(
        "Failed to import compiled module '_vulkan_forge'.\n"
        f"python: {sys.executable}\n"
        f"sys.path[0]: {sys.path[0] if sys.path else '<empty>'}\n"
        "Ensure you built in THIS venv:\n"
        "  (if Python 3.13) export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1\n"
        "  maturin develop --release --features terrain_spike"
    )

# --- Load the compiled extension, then alias public API ---
_ext = _load_extension()
Renderer = _ext.Renderer

# Optional TerrainSpike
TerrainSpike = getattr(_ext, "TerrainSpike", None)

# --- Convenience functions wrapping the core API ---
def render_triangle_rgba(width: int, height: int):
    r = Renderer(int(width), int(height))
    return r.render_triangle_rgba()

def render_triangle_png(path: str, width: int, height: int) -> None:
    r = Renderer(int(width), int(height))
    r.render_triangle_png(str(path))

# --- Version metadata (best-effort) ---
try:
    from importlib.metadata import version
    __version__ = version("vulkan-forge")
except Exception:
    try:
        __version__ = version("vulkan_forge")
    except Exception:
        __version__ = "0.0.0.dev0"

__all__ = ["Renderer", "render_triangle_rgba", "render_triangle_png", "__version__"]
if TerrainSpike is not None:
    __all__.append("TerrainSpike")
# A1.5-END:vulkan_forge-shim
