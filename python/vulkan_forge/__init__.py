# A1.5-BEGIN:vulkan_forge-shim
"""
Thin Python shim for the compiled extension `_vulkan_forge`.

Exports:
- Renderer: the core class implemented in Rust (PyO3)
- render_triangle_rgba(width, height) -> np.ndarray[H,W,4] (uint8)
- render_triangle_png(path, width, height) -> None
- __version__: best-effort package version
"""
from __future__ import annotations

# Robust import that works whether the extension is top-level or inside this package
try:
    import _vulkan_forge as _mod          # top-level (some maturin installs)
except ModuleNotFoundError:
    try:
        from . import _vulkan_forge as _mod  # packaged inside vulkan_forge/
    except Exception as e:
        import sys
        raise ImportError(
            "Failed to import compiled module '_vulkan_forge'.\n"
            f"Python: {sys.executable}\n"
            "Fix:\n"
            "  • Activate the venv used for 'maturin develop'\n"
            "  • Reinstall:  pip install -U pip maturin && maturin develop --release\n"
        ) from e

# IMPORTANT: alias, do NOT subclass (PyO3 class not declared with #[pyclass(subclass)])
Renderer = _mod.Renderer

def render_triangle_rgba(width: int, height: int):
    """Render a deterministic triangle and return (H,W,4) uint8 array."""
    r = Renderer(int(width), int(height))
    return r.render_triangle_rgba()

def render_triangle_png(path: str, width: int, height: int) -> None:
    """Render a deterministic triangle and write `path` as PNG."""
    r = Renderer(int(width), int(height))
    r.render_triangle_png(str(path))

# Version metadata (best-effort)
try:
    from importlib.metadata import version
    __version__ = version("vulkan-forge")
except Exception:
    try:
        __version__ = version("vulkan_forge")
    except Exception:
        __version__ = "0.0.0.dev0"

__all__ = ["Renderer", "render_triangle_rgba", "render_triangle_png", "__version__"]
# A1.5-END:vulkan_forge-shim
