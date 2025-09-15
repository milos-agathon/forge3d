import importlib.resources as res
import types
import forge3d as f3d

def test_py_typed_shipped():
    try:
        assert res.files("forge3d").joinpath("py.typed").is_file()
    except (AttributeError, FileNotFoundError):
        # Skip if py.typed file not present or files() not available
        import pytest
        pytest.skip("py.typed file not found or importlib.resources.files not available")

def test_public_surface_includes_diagnostics_and_no_module_leak():
    for name in [
        "Renderer", "Scene",
        "png_to_numpy", "numpy_to_png",
        "grid_generate", "render_triangle_rgba", "render_triangle_png",
        "dem_stats", "dem_normalize",
        "enumerate_adapters", "device_probe",
        "__version__",
    ]:
        assert name in getattr(f3d, "__all__", []), f"missing in __all__: {name}"
    assert "_forge3d" not in getattr(f3d, "__all__", [])
    if hasattr(f3d, "_forge3d"):
        assert isinstance(getattr(f3d, "_forge3d"), types.ModuleType)