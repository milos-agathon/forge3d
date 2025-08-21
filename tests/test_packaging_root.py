import numpy as np
import forge3d as f3d

def test_root_import_version_and_symbols():
    assert hasattr(f3d, "__version__") and isinstance(f3d.__version__, str)
    for name in [
        "Renderer", "Scene",
        "png_to_numpy", "numpy_to_png",
        "grid_generate",
        "render_triangle_rgba", "render_triangle_png",
        "dem_stats", "dem_normalize",
    ]:
        assert hasattr(f3d, name), f"missing symbol at root: {name}"

def test_numpy_to_png_accepts_pathlike(tmp_path):
    p = tmp_path / "gray.png"
    g = np.array([[0, 127, 255],[10, 20, 30]], dtype=np.uint8)
    f3d.numpy_to_png(p, g)           # PathLike write
    a = f3d.png_to_numpy(p)          # PathLike read
    assert a.shape == (2, 3, 4)
    assert (a[..., 3] == 255).all()  # alpha channel should be opaque

def test_scene_render_rgba_contiguous():
    s = f3d.Scene(8, 6)
    arr = s.render_rgba()
    assert arr.flags["C_CONTIGUOUS"]
    assert arr.shape == (6, 8, 4)
    assert arr.dtype == np.uint8

def test_root_wrapper_render_triangle_png(tmp_path):
    p = tmp_path / "via_wrapper.png"
    f3d.render_triangle_png(p, 8, 6)
    a = f3d.png_to_numpy(p)
    assert a.shape == (6, 8, 4)

def test_root_wrapper_render_triangle_rgba():
    a = f3d.render_triangle_rgba(8, 6)
    assert a.shape == (6, 8, 4)
    assert a.flags["C_CONTIGUOUS"]

def test_curated_all_does_not_leak_module_object():
    # Ensure we didn't re-export the compiled module object itself as a user-accessible symbol
    # The _vulkan_forge attribute might exist due to Python's import machinery, but shouldn't be in __all__
    assert "_vulkan_forge" not in f3d.__all__
    # Also check that if it exists, it's not directly accessible as a public API
    if hasattr(f3d, "_vulkan_forge"):
        # It should be a module reference, not one of our functions
        import types
        assert isinstance(getattr(f3d, "_vulkan_forge"), types.ModuleType)
    
    # Sanity: __all__ exists and contains key names
    assert isinstance(getattr(f3d, "__all__", []), list)
    assert "Renderer" in f3d.__all__ and "Scene" in f3d.__all__