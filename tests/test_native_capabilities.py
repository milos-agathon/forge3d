"""Live native dispatch, using fake extensions without a GPU context."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def native_runtime(monkeypatch):
    from forge3d import _native

    # Restore shared state and the legacy alias even when refresh changes them.
    for name in ("NATIVE_MODULE", "NATIVE_AVAILABLE", "NATIVE_IMPORT_ERROR"):
        monkeypatch.setattr(_native, name, getattr(_native, name))
    monkeypatch.setitem(sys.modules, "python.forge3d._forge3d", object())
    original_import = importlib.import_module
    state = SimpleNamespace(module=None, error=ImportError("extension absent"), imports=0, invalidations=0)

    def importer(name, package=None):
        if name == "forge3d._forge3d":
            state.imports += 1
            if state.module is None:
                raise state.error
            return state.module
        return original_import(name, package)

    def invalidate():
        state.invalidations += 1

    monkeypatch.setattr(importlib, "import_module", importer)
    monkeypatch.setattr(importlib, "invalidate_caches", invalidate)

    def refresh(module=None, error=None):
        state.module = module
        if error is not None:
            state.error = error
        return _native.refresh_native_module()

    refresh()
    return _native, state, refresh


def test_refresh_lifecycle_and_legacy_import_compatibility(native_runtime):
    native, state, refresh = native_runtime
    from forge3d._native import NATIVE_AVAILABLE

    assert NATIVE_AVAILABLE is False
    marker = object()
    assert native.get_native_symbol("missing", marker) is marker
    module = ModuleType("forge3d._forge3d")
    module.feature = marker
    assert refresh(module) is module
    assert native.NATIVE_AVAILABLE is True
    assert native.native_import_error() is None
    assert native.get_native_symbol("feature") is marker
    assert sys.modules["python.forge3d._forge3d"] is module
    assert NATIVE_AVAILABLE is False  # An explicit by-value import is a snapshot.

    replacement = ModuleType("forge3d._forge3d")
    assert refresh(replacement) is replacement
    assert sys.modules["python.forge3d._forge3d"] is replacement
    assert native.get_native_symbol("feature") is None
    failure = OSError("missing dependent DLL")
    assert refresh(error=failure) is None
    assert native.native_import_error() is failure
    assert native.NATIVE_AVAILABLE is False
    assert "python.forge3d._forge3d" not in sys.modules
    with pytest.raises(RuntimeError, match="build the extension") as raised:
        native.require_native_module()
    assert raised.value.__cause__ is failure

    before = state.imports, state.invalidations
    native.get_native_module()
    native.get_native_symbol("feature")
    native.native_import_error()
    assert (state.imports, state.invalidations) == before
    assert state.imports == state.invalidations


def test_legacy_consumer_names_are_importable_and_capabilities_stay_live(native_runtime):
    _, _, refresh = native_runtime
    from forge3d import mesh
    from forge3d.mesh import _HAS_TBN
    from forge3d.helpers.offscreen import _ImportedNativeScene, _NativeScene

    assert _HAS_TBN is False
    assert _NativeScene is _ImportedNativeScene
    refresh(SimpleNamespace(mesh_generate_cube_tbn=lambda: None))
    assert mesh._HAS_TBN is True
    from forge3d.mesh import _HAS_TBN as refreshed_tbn
    assert refreshed_tbn is True
    assert _HAS_TBN is False
    refresh()
    assert mesh._HAS_TBN is False


def test_crs_transform_preserves_native_import_failure_cause(native_runtime):
    _, _, refresh = native_runtime
    from forge3d import crs

    failure = ImportError("DLL load failed while importing _forge3d: fake")
    refresh(error=failure)
    with pytest.raises(ImportError, match="cannot import name '_forge3d'") as raised:
        crs._crs_transform("EPSG:4326", "EPSG:3857")
    assert raised.value.__cause__ is failure


def test_required_geometry_and_io_dispatch_follow_refresh(native_runtime):
    _, _, refresh = native_runtime
    from forge3d import geometry, io

    positions = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    mesh = geometry.MeshBuffers(positions, positions, np.zeros((3, 2)), indices)
    for module in (None, SimpleNamespace(
        geometry_validate_mesh_py=lambda *_: {"native": True},
        io_export_stl_py=lambda *_: True,
    ), SimpleNamespace(), None):
        refresh(module)
        if module is None:
            with pytest.raises(RuntimeError, match="native module is not available"):
                geometry.validate_mesh(positions, indices)
            with pytest.raises(RuntimeError, match="native module is not available"):
                io.save_stl(mesh, "unused.stl")
        elif hasattr(module, "geometry_validate_mesh_py"):
            assert geometry.validate_mesh(positions, indices) == {"native": True}
            assert io.save_stl(mesh, "unused.stl") is True
        else:
            with pytest.raises(AttributeError, match="geometry_validate_mesh_py"):
                geometry.validate_mesh(positions, indices)
            with pytest.raises(AttributeError, match="io_export_stl_py"):
                io.save_stl(mesh, "unused.stl")


def test_optional_dispatch_and_missing_symbols_follow_refresh(native_runtime):
    _, _, refresh = native_runtime
    from forge3d import _gpu, _memory, buildings, cog, crs, map_scene, mesh, text, tiles3d
    from forge3d.helpers import offscreen

    class NativeScene:
        pass

    class NativeCog:
        def __init__(self, *args):
            self.args = args

    writer = lambda *_: None
    module = SimpleNamespace(
        Scene=NativeScene, CogDataset=NativeCog,
        proj_available=lambda: True,
        enumerate_adapters=lambda: [{"name": "fake adapter"}],
        device_probe=lambda: {"status": "ok", "name": "fake adapter"},
        global_memory_metrics=lambda: {"buffer_bytes": 17},
        infer_roof_type_py=lambda _: "native-roof",
        mesh_generate_cube_tbn=lambda: {"vertices": ["native"], "indices": [], "tbn_data": []},
        text_shape=lambda *_: "native-text",
        decode_b3dm_py=lambda _: {"native": True},
        numpy_to_exr=writer,
    )
    for current in (module, SimpleNamespace(), None, module):
        refresh(current)
        present = current is module
        assert _gpu.has_gpu() is present
        assert cog.is_cog_available() is present
        assert crs.HAS_NATIVE_PROJ is present
        assert mesh.has_tbn_support() is present
        assert offscreen._is_native_scene(NativeScene()) is present
        assert map_scene._numpy_to_exr_writer() is (writer if present else None)
        if present:
            assert _memory.memory_metrics()["buffer_bytes"] == 17
            assert buildings.infer_roof_type({}) == "native-roof"
            assert mesh.generate_cube_tbn()[0] == ["native"]
            assert text.shape("text", [], 12) == "native-text"
            assert tiles3d.decode_b3dm(b"") == {"native": True}
            assert cog.CogDataset("unused")._native.args == ("unused", 256, None, 256)
        else:
            assert _gpu.device_probe()["status"] == "native_missing"
            assert buildings.infer_roof_type({}) != "native-roof"
            assert mesh.generate_cube_tbn()[0]
            with pytest.raises(text.TextShapingError, match="unavailable"):
                text.shape("text", [], 12)
            with pytest.raises(ValueError, match="too small"):
                tiles3d.decode_b3dm(b"")
            with pytest.raises(RuntimeError, match="COG streaming is not available"):
                cog.CogDataset("unused")


def test_native_handles_keep_their_owning_module(native_runtime):
    _, _, refresh = native_runtime
    from forge3d import lighting

    handle = object()
    calls = []
    original = SimpleNamespace(
        create_restir_di=lambda **_: handle,
        restir_clear_lights=lambda value: calls.append(value),
    )
    refresh(original)
    existing = lighting.RestirDI()
    refresh(SimpleNamespace(
        create_restir_di=lambda **_: object(),
        restir_clear_lights=lambda _: pytest.fail("used a different module's native handle"),
    ))
    existing.clear_lights()
    assert calls == [handle]


def test_text_atlas_distinguishes_strings_from_live_shaped_types(native_runtime, monkeypatch):
    _, _, refresh = native_runtime
    from forge3d import text

    # Characterize an import without native text, whose exported type is object.
    monkeypatch.setattr(text, "ShapedText", object)

    class LiveShapedText:
        def to_dict(self):
            return {"font_sources": [], "font_sha256": []}

    refresh(SimpleNamespace(
        ShapedText=LiveShapedText,
        bake_msdf_atlas=lambda *args: {"route": "charset", "metrics": {}},
        bake_msdf_atlas_shaped=lambda *args: {"route": "shaped", "metrics": {}},
    ))
    assert text.bake_msdf_atlas([], "abc", 12)["route"] == "charset"
    assert text.bake_msdf_atlas([], LiveShapedText(), 12)["route"] == "shaped"
    assert text.ShapedText is object


def test_other_runtime_dispatch_follows_refresh(native_runtime, monkeypatch):
    _, _, refresh = native_runtime
    from forge3d import lighting, path_tracing, pointcloud, sdf, tiles3d

    setter_calls = []
    module = SimpleNamespace(
        set_exposure_scale=lambda value: setter_calls.append(value),
        hybrid_render_terrain_reference=lambda *_, **__: {"native": True},
        read_laz_point_attributes=lambda *_: {"coords": [], "intensities": [], "classifications": []},
        tiles3d_traverse_py=lambda *_: [{"native": True}],
        decode_pnts_py=lambda _: {"positions": [], "point_count": 0},
        hybrid_render=lambda *_: None,
    )
    refresh(module)
    lighting.set_exposure_stops(1)
    assert setter_calls == [2.0]
    assert path_tracing.hybrid_render_terrain_reference(np.zeros((2, 2)), 8, 8, {}) == {
        "native": True,
        "sun_source": "manual_angles",
        "solar_azimuth_deg": 315.0,
        "solar_elevation_deg": 45.0,
    }
    assert pointcloud.read_laz_point_attributes("unused")["coords"].shape == (0, 3)
    dataset = tiles3d.Tiles3dDataset(Path("unused"), None)
    assert dataset.traverse((0, 0, 0)) == [{"native": True}]
    assert tiles3d.decode_pnts(b"")["positions"].shape == (0, 3)
    monkeypatch.setattr(sdf, "USE_NATIVE_SDF", True)
    renderer = sdf.HybridRenderer(8, 8)
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    monkeypatch.setattr(renderer, "_render_native", lambda *_: image)
    assert renderer.render_sdf_scene(sdf.SdfScene()) is image
    assert sdf.NATIVE_AVAILABLE is True
    assert sdf.forge3d_native is module
    refresh()
    assert sdf.NATIVE_AVAILABLE is False
    assert not hasattr(sdf, "forge3d_native")
    with pytest.raises(RuntimeError, match="requires the native"):
        path_tracing.hybrid_render_terrain_reference(np.zeros((2, 2)), 8, 8, {})
    with pytest.raises(RuntimeError, match="compiled native extension"):
        pointcloud.read_laz_point_attributes("unused")


@pytest.mark.parametrize("initially_present", [False, True])
def test_initial_native_state_and_frozen_export_identities(initially_present):
    # A fresh interpreter proves both initial native states without a real binary.
    script = r'''
import importlib
from types import ModuleType
original_import = importlib.import_module
def module():
    result = ModuleType("forge3d._forge3d")
    for name in ("Scene", "MemoryBudgetExceeded", "CameraAnimation", "CameraKeyframe",
                 "CameraState", "ShapedText", "RasterInfo", "VectorInfo",
                 "AffineTransform", "CrsTransform"):
        setattr(result, name, type(name, (), {}))
    result.mesh_generate_cube_tbn = lambda: None
    return result
current = module() if INITIAL else None
def importer(name, package=None):
    if name == "forge3d._forge3d":
        if current is None:
            raise ImportError("fixture extension absent")
        return current
    return original_import(name, package)
importlib.import_module = importer
import forge3d
from forge3d import _native, animation, gis, text, sdf
from forge3d.helpers import offscreen
from forge3d.helpers.offscreen import _NativeScene
from forge3d.mesh import _HAS_TBN
assert _HAS_TBN is INITIAL
assert _NativeScene is (current.Scene if INITIAL else None)
wildcard = {}
exec("from forge3d.sdf import *", wildcard)
assert wildcard["NATIVE_AVAILABLE"] is INITIAL
assert ("forge3d_native" in wildcard) is INITIAL
if INITIAL:
    assert wildcard["forge3d_native"] is current
bindings = animation.CameraAnimation, gis.RasterInfo, text.ShapedText, forge3d.MemoryBudgetExceeded
old_scene = forge3d.Scene if INITIAL else None
current = module()
_native.refresh_native_module()
wildcard = {}
exec("from forge3d.sdf import *", wildcard)
assert wildcard["forge3d_native"] is current
assert wildcard["NATIVE_AVAILABLE"] is True
assert bindings == (animation.CameraAnimation, gis.RasterInfo, text.ShapedText, forge3d.MemoryBudgetExceeded)
assert forge3d.Scene is (old_scene if INITIAL else current.Scene)
assert offscreen._is_native_scene(current.Scene())
if old_scene:
    assert offscreen._is_native_scene(old_scene())
current = None
_native.refresh_native_module()
assert bindings == (animation.CameraAnimation, gis.RasterInfo, text.ShapedText, forge3d.MemoryBudgetExceeded)
assert _native.get_native_module() is None
wildcard = {}
exec("from forge3d.sdf import *", wildcard)
assert wildcard["NATIVE_AVAILABLE"] is False
assert "forge3d_native" not in wildcard
'''
    script = script.replace("INITIAL", repr(initially_present))
    script = "import sys\nsys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[1] / "python")) + ")\n" + script
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("import_before_refresh", [False, True])
def test_lazy_consumer_import_order_with_real_import_machinery(import_before_refresh):
    script = r'''
import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType

class NativeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    present = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "forge3d._forge3d":
            return None
        if not self.present:
            raise ModuleNotFoundError("fixture extension absent", name=fullname)
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return ModuleType(spec.name)

    def exec_module(self, module):
        module.CogDataset = type("CogDataset", (), {})
        module.mesh_generate_cube_tbn = lambda: None
        module.decode_b3dm_py = lambda _: {"native": True}

finder = NativeFinder()
sys.meta_path.insert(0, finder)
import forge3d
from forge3d import _native
names = ("forge3d.mesh", "forge3d.cog", "forge3d.tiles3d")
assert not any(name in sys.modules for name in names)

def consumers():
    return tuple(importlib.import_module(name) for name in names)

def check_absent(modules):
    mesh, cog, tiles3d = modules
    assert mesh.has_tbn_support() is False
    assert mesh._HAS_TBN is False
    assert cog.is_cog_available() is False
    try:
        tiles3d.decode_b3dm(b"")
    except ValueError as error:
        assert "too small" in str(error)
    else:
        raise AssertionError("absent native decoder did not use Python validation")

if BEFORE:
    modules = consumers()
    check_absent(modules)
finder.present = True
native = _native.refresh_native_module()
assert native is not None
if not BEFORE:
    modules = consumers()
mesh, cog, tiles3d = modules
assert mesh._HAS_TBN is True
assert cog.is_cog_available() is True
assert tiles3d.decode_b3dm(b"") == {"native": True}
assert sys.modules["python.forge3d._forge3d"] is native
finder.present = False
sys.modules.pop("forge3d._forge3d")
del forge3d._forge3d
assert _native.refresh_native_module() is None
assert isinstance(_native.native_import_error(), ModuleNotFoundError)
assert "python.forge3d._forge3d" not in sys.modules
check_absent(modules)
'''
    script = script.replace("BEFORE", repr(import_before_refresh))
    script = "import sys\nsys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[1] / "python")) + ")\n" + script
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
