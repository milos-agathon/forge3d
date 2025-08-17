import importlib.resources as res
import types
import vulkan_forge as vf

def test_py_typed_shipped():
    assert res.files("vulkan_forge").joinpath("py.typed").is_file()

def test_public_surface_includes_diagnostics_and_no_module_leak():
    for name in [
        "Renderer", "Scene",
        "png_to_numpy", "numpy_to_png",
        "grid_generate", "render_triangle_rgba", "render_triangle_png",
        "dem_stats", "dem_normalize",
        "enumerate_adapters", "device_probe",
        "__version__",
    ]:
        assert name in getattr(vf, "__all__", []), f"missing in __all__: {name}"
    assert "_vulkan_forge" not in getattr(vf, "__all__", [])
    if hasattr(vf, "_vulkan_forge"):
        assert isinstance(getattr(vf, "_vulkan_forge"), types.ModuleType)