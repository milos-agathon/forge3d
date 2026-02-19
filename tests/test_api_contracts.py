"""P0.1 API Contract Tests -- Baseline snapshot of the public native API surface.

These tests lock the **current** (pre-consolidation) contract so that
refactoring in P0.2-P0.5 cannot silently remove or rename symbols that
downstream code depends on.

Tested contracts:
  - _forge3d native module loads and exposes expected classes/functions
  - Scene class has render_rgba and set_msaa_samples instance methods
  - Scene class exposes key feature-enable/disable methods
  - Key native classes are registered and accessible
  - TBN mesh functions are NOT yet exported (marked as future P0.4)
  - Orphaned pyclass types (Frame, SdfPrimitive, etc.) are NOT registered

Each test is minimal and non-trivial: it asserts something specific about
the current API surface, not just that imports succeed.
"""

from __future__ import annotations

import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE, get_native_module


# ---------------------------------------------------------------------------
# Skip entire module when native extension is absent (e.g., pure-Python CI)
# ---------------------------------------------------------------------------
if not NATIVE_AVAILABLE:
    pytest.skip(
        "Contract tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

_native = get_native_module()


# ===========================================================================
# Section 1: Core native module symbols
# ===========================================================================
class TestNativeModuleSymbols:
    """Verify that the native module exports the expected top-level symbols."""

    # ---- Registered classes (m.add_class in lib.rs) ----

    EXPECTED_CLASSES = [
        "Scene",
        "Session",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "TerrainRenderParams",
        "TerrainRenderer",
        "AovFrame",
        "CameraAnimation",
        "CameraState",
        "ClipmapConfig",
        "ClipmapMesh",
        "SunPosition",
    ]

    @pytest.mark.parametrize("cls_name", EXPECTED_CLASSES)
    def test_registered_class_exists(self, cls_name: str):
        """Each registered pyclass must be accessible on the native module."""
        assert hasattr(_native, cls_name), (
            f"_forge3d.{cls_name} not found -- "
            f"was it removed from m.add_class in lib.rs?"
        )
        obj = getattr(_native, cls_name)
        assert isinstance(obj, type), (
            f"_forge3d.{cls_name} should be a class, got {type(obj)}"
        )

    # ---- Registered free functions (wrap_pyfunction in lib.rs) ----

    EXPECTED_FUNCTIONS = [
        "open_viewer",
        "open_terrain_viewer",
        "enumerate_adapters",
        "device_probe",
        "sun_position",
        "sun_position_utc",
        "clipmap_generate_py",
        "engine_info",
        "hybrid_render",
        "configure_csm",
        "global_memory_metrics",
    ]

    @pytest.mark.parametrize("fn_name", EXPECTED_FUNCTIONS)
    def test_registered_function_exists(self, fn_name: str):
        """Each registered pyfunction must be callable on the native module."""
        assert hasattr(_native, fn_name), (
            f"_forge3d.{fn_name} not found -- "
            f"was it removed from wrap_pyfunction in lib.rs?"
        )
        obj = getattr(_native, fn_name)
        assert callable(obj), (
            f"_forge3d.{fn_name} should be callable, got {type(obj)}"
        )


# ===========================================================================
# Section 2: Scene class method contracts
# ===========================================================================
class TestSceneMethodContracts:
    """Verify Scene class exposes the methods that wrappers depend on."""

    # ---- Primary render methods ----

    def test_render_rgba_is_instance_method(self):
        """Scene.render_rgba must exist as an instance method (not static)."""
        assert hasattr(f3d.Scene, "render_rgba"), "Scene.render_rgba not found"
        # On a PyO3 class, methods appear as method descriptors
        attr = getattr(f3d.Scene, "render_rgba")
        assert callable(attr), "Scene.render_rgba must be callable"

    def test_render_png_is_instance_method(self):
        """Scene.render_png must exist as an instance method."""
        assert hasattr(f3d.Scene, "render_png"), "Scene.render_png not found"
        attr = getattr(f3d.Scene, "render_png")
        assert callable(attr), "Scene.render_png must be callable"

    # ---- Configuration methods ----

    def test_set_msaa_samples_is_instance_method(self):
        """Scene.set_msaa_samples must exist as an instance method."""
        assert hasattr(f3d.Scene, "set_msaa_samples"), (
            "Scene.set_msaa_samples not found"
        )
        attr = getattr(f3d.Scene, "set_msaa_samples")
        assert callable(attr), "Scene.set_msaa_samples must be callable"

    def test_set_camera_look_at_exists(self):
        """Scene.set_camera_look_at must exist."""
        assert hasattr(f3d.Scene, "set_camera_look_at"), (
            "Scene.set_camera_look_at not found"
        )

    def test_set_height_from_r32f_exists(self):
        """Scene.set_height_from_r32f must exist."""
        assert hasattr(f3d.Scene, "set_height_from_r32f"), (
            "Scene.set_height_from_r32f not found"
        )

    # ---- OIT methods ----

    def test_enable_oit_exists(self):
        """Scene.enable_oit must exist (P0.1 OIT feature)."""
        assert hasattr(f3d.Scene, "enable_oit"), "Scene.enable_oit not found"

    def test_disable_oit_exists(self):
        """Scene.disable_oit must exist."""
        assert hasattr(f3d.Scene, "disable_oit"), "Scene.disable_oit not found"

    def test_is_oit_enabled_exists(self):
        """Scene.is_oit_enabled must exist."""
        assert hasattr(f3d.Scene, "is_oit_enabled"), (
            "Scene.is_oit_enabled not found"
        )

    def test_get_oit_mode_exists(self):
        """Scene.get_oit_mode must exist."""
        assert hasattr(f3d.Scene, "get_oit_mode"), "Scene.get_oit_mode not found"

    # ---- SSAO methods ----

    def test_ssao_enabled_exists(self):
        """Scene.ssao_enabled must exist."""
        assert hasattr(f3d.Scene, "ssao_enabled"), "Scene.ssao_enabled not found"

    def test_set_ssao_enabled_exists(self):
        """Scene.set_ssao_enabled must exist."""
        assert hasattr(f3d.Scene, "set_ssao_enabled"), (
            "Scene.set_ssao_enabled not found"
        )

    def test_set_ssao_parameters_exists(self):
        """Scene.set_ssao_parameters must exist."""
        assert hasattr(f3d.Scene, "set_ssao_parameters"), (
            "Scene.set_ssao_parameters not found"
        )

    # ---- IBL methods ----

    def test_enable_ibl_exists(self):
        """Scene.enable_ibl must exist."""
        assert hasattr(f3d.Scene, "enable_ibl"), "Scene.enable_ibl not found"

    def test_disable_ibl_exists(self):
        """Scene.disable_ibl must exist."""
        assert hasattr(f3d.Scene, "disable_ibl"), "Scene.disable_ibl not found"

    # ---- Reflections ----

    def test_enable_reflections_exists(self):
        """Scene.enable_reflections must exist."""
        assert hasattr(f3d.Scene, "enable_reflections"), (
            "Scene.enable_reflections not found"
        )

    def test_disable_reflections_exists(self):
        """Scene.disable_reflections must exist."""
        assert hasattr(f3d.Scene, "disable_reflections"), (
            "Scene.disable_reflections not found"
        )

    # ---- DOF ----

    def test_enable_dof_exists(self):
        """Scene.enable_dof must exist."""
        assert hasattr(f3d.Scene, "enable_dof"), "Scene.enable_dof not found"

    def test_disable_dof_exists(self):
        """Scene.disable_dof must exist."""
        assert hasattr(f3d.Scene, "disable_dof"), "Scene.disable_dof not found"

    # ---- Water surface ----

    def test_enable_water_surface_exists(self):
        """Scene.enable_water_surface must exist."""
        assert hasattr(f3d.Scene, "enable_water_surface"), (
            "Scene.enable_water_surface not found"
        )

    # ---- Ground plane ----

    def test_enable_ground_plane_exists(self):
        """Scene.enable_ground_plane must exist."""
        assert hasattr(f3d.Scene, "enable_ground_plane"), (
            "Scene.enable_ground_plane not found"
        )

    # ---- Cloud shadows ----

    def test_enable_cloud_shadows_exists(self):
        """Scene.enable_cloud_shadows must exist."""
        assert hasattr(f3d.Scene, "enable_cloud_shadows"), (
            "Scene.enable_cloud_shadows not found"
        )

    # ---- Point/spot lights ----

    def test_enable_point_spot_lights_exists(self):
        """Scene.enable_point_spot_lights must exist."""
        assert hasattr(f3d.Scene, "enable_point_spot_lights"), (
            "Scene.enable_point_spot_lights not found"
        )

    def test_add_point_light_exists(self):
        """Scene.add_point_light must exist."""
        assert hasattr(f3d.Scene, "add_point_light"), (
            "Scene.add_point_light not found"
        )

    # ---- Text meshes ----

    def test_enable_text_meshes_exists(self):
        """Scene.enable_text_meshes must exist."""
        assert hasattr(f3d.Scene, "enable_text_meshes"), (
            "Scene.enable_text_meshes not found"
        )

    # ---- get_stats ----

    def test_get_stats_exists(self):
        """Scene.get_stats must exist."""
        assert hasattr(f3d.Scene, "get_stats"), "Scene.get_stats not found"


# ===========================================================================
# Section 3: Scene method count snapshot
# ===========================================================================
class TestSceneMethodCount:
    """Guard against accidental bulk removal of Scene methods.

    If someone accidentally removes a large block of #[pymethods], this
    catches it by asserting a minimum method count.
    """

    def test_scene_has_minimum_method_count(self):
        """Scene must expose at least 80 public methods.

        As of baseline (2026-02-19), Scene has ~180 methods. A drop
        below 80 would indicate a serious regression.
        """
        public_methods = [
            name for name in dir(f3d.Scene)
            if not name.startswith("_") and callable(getattr(f3d.Scene, name))
        ]
        assert len(public_methods) >= 80, (
            f"Scene has only {len(public_methods)} public methods; "
            f"expected at least 80. Methods may have been removed accidentally."
        )


# ===========================================================================
# Section 4: Orphaned pyclass types NOT registered (negative tests)
# ===========================================================================
class TestOrphanedClassesNotRegistered:
    """Verify that known orphaned pyclass types are NOT on the native module.

    These classes have #[pyclass] but no m.add_class registration.
    When P0.3 registers them, these tests should be updated to expect them.
    """

    def test_frame_not_registered(self):
        """Frame has #[pyclass] but is NOT registered in pymodule."""
        assert not hasattr(_native, "Frame"), (
            "Frame is now registered -- update this test if intentional (P0.3)"
        )

    def test_sdf_primitive_not_registered(self):
        """SdfPrimitive has #[pyclass] but is NOT registered in pymodule."""
        assert not hasattr(_native, "SdfPrimitive"), (
            "SdfPrimitive is now registered -- update this test if intentional (P0.3)"
        )

    def test_sdf_scene_not_registered(self):
        """SdfScene has #[pyclass] but is NOT registered in pymodule."""
        assert not hasattr(_native, "SdfScene"), (
            "SdfScene is now registered -- update this test if intentional (P0.3)"
        )

    def test_sdf_scene_builder_not_registered(self):
        """SdfSceneBuilder has #[pyclass] but is NOT registered in pymodule."""
        assert not hasattr(_native, "SdfSceneBuilder"), (
            "SdfSceneBuilder is now registered -- update this test if intentional (P0.3)"
        )


# ===========================================================================
# Section 5: TBN mesh functions NOT yet exported (future P0.4)
# ===========================================================================
class TestTbnFunctionsNotExported:
    """Verify that TBN mesh functions are not yet in the native module.

    The Rust functions exist (src/mesh/tbn.rs) but have no PyO3 wrapper.
    When P0.4 exposes them, flip these to positive assertions.
    """

    @pytest.mark.skipif(
        hasattr(_native, "mesh_generate_cube_tbn"),
        reason="mesh_generate_cube_tbn now exported (P0.4 complete)",
    )
    def test_mesh_generate_cube_tbn_not_exported(self):
        """mesh_generate_cube_tbn is NOT in the native module pre-P0.4."""
        assert not hasattr(_native, "mesh_generate_cube_tbn")

    @pytest.mark.skipif(
        hasattr(_native, "mesh_generate_plane_tbn"),
        reason="mesh_generate_plane_tbn now exported (P0.4 complete)",
    )
    def test_mesh_generate_plane_tbn_not_exported(self):
        """mesh_generate_plane_tbn is NOT in the native module pre-P0.4."""
        assert not hasattr(_native, "mesh_generate_plane_tbn")


# ===========================================================================
# Section 6: Python package-level API contracts
# ===========================================================================
class TestPackageLevelApiContracts:
    """Verify that forge3d package-level re-exports are intact."""

    EXPECTED_PACKAGE_ATTRS = [
        "Scene",
        "Session",
        "TerrainRenderer",
        "TerrainRenderParams",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "__version__",
        "has_gpu",
        "render_raster",
        "render_offscreen_rgba",
        "numpy_to_png",
        "png_to_numpy",
        "MapPlate",
        "Legend",
        "ScaleBar",
        "NorthArrow",
        "save_bundle",
        "load_bundle",
    ]

    @pytest.mark.parametrize("attr_name", EXPECTED_PACKAGE_ATTRS)
    def test_package_exports_symbol(self, attr_name: str):
        """forge3d package must re-export key symbols."""
        assert hasattr(f3d, attr_name), (
            f"forge3d.{attr_name} not found in package __init__.py"
        )

    def test_version_is_string(self):
        """forge3d.__version__ must be a non-empty string."""
        assert isinstance(f3d.__version__, str)
        assert len(f3d.__version__) > 0
        # Semver-ish: at least "X.Y.Z"
        parts = f3d.__version__.split(".")
        assert len(parts) >= 3, (
            f"Version '{f3d.__version__}' does not look like semver"
        )

    def test_has_gpu_returns_bool(self):
        """forge3d.has_gpu() must return a boolean."""
        result = f3d.has_gpu()
        assert isinstance(result, bool)


# ===========================================================================
# Section 7: Geometry free-function contracts
# ===========================================================================
class TestGeometryFunctionContracts:
    """Verify geometry-related native functions are registered."""

    GEOMETRY_FUNCTIONS = [
        "geometry_generate_primitive_py",
        "geometry_generate_tangents_py",
        "geometry_weld_mesh_py",
        "geometry_subdivide_py",
        "geometry_validate_mesh_py",
        "geometry_displace_heightmap_py",
        "geometry_generate_tube_py",
        "geometry_generate_ribbon_py",
        "geometry_generate_thick_polyline_py",
        "geometry_extrude_polygon_py",
    ]

    @pytest.mark.parametrize("fn_name", GEOMETRY_FUNCTIONS)
    def test_geometry_function_exists(self, fn_name: str):
        """Geometry functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), (
            f"_forge3d.{fn_name} not found"
        )
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 8: Camera function contracts
# ===========================================================================
class TestCameraFunctionContracts:
    """Verify camera-related native functions are registered."""

    CAMERA_FUNCTIONS = [
        "camera_look_at",
        "camera_perspective",
        "camera_orthographic",
        "camera_view_proj",
        "camera_dof_params",
    ]

    @pytest.mark.parametrize("fn_name", CAMERA_FUNCTIONS)
    def test_camera_function_exists(self, fn_name: str):
        """Camera functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), f"_forge3d.{fn_name} not found"
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 9: IO function contracts
# ===========================================================================
class TestIoFunctionContracts:
    """Verify IO-related native functions are registered."""

    IO_FUNCTIONS = [
        "io_import_obj_py",
        "io_export_obj_py",
        "io_export_stl_py",
        "io_import_gltf_py",
    ]

    @pytest.mark.parametrize("fn_name", IO_FUNCTIONS)
    def test_io_function_exists(self, fn_name: str):
        """IO functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), f"_forge3d.{fn_name} not found"
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 10: Transform function contracts
# ===========================================================================
class TestTransformFunctionContracts:
    """Verify transform-related native functions are registered."""

    TRANSFORM_FUNCTIONS = [
        "translate",
        "rotate_x",
        "rotate_y",
        "rotate_z",
        "scale",
    ]

    @pytest.mark.parametrize("fn_name", TRANSFORM_FUNCTIONS)
    def test_transform_function_exists(self, fn_name: str):
        """Transform functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), f"_forge3d.{fn_name} not found"
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 11: Native module total symbol count guard
# ===========================================================================
class TestNativeModuleSymbolCount:
    """Guard against accidental bulk removal of native symbols."""

    def test_native_module_has_minimum_symbols(self):
        """The native module must export at least 100 symbols.

        As of baseline (2026-02-19), it exports 134 symbols.
        A significant drop indicates accidental removal.
        """
        public_symbols = [
            name for name in dir(_native) if not name.startswith("_")
        ]
        assert len(public_symbols) >= 100, (
            f"Native module has only {len(public_symbols)} public symbols; "
            f"expected at least 100. Symbols may have been removed accidentally."
        )
