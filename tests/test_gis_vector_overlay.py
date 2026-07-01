"""G-002c C4.0 vector overlay public surface and backend-gate tests."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE, get_native_module


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS vector overlay tests require the compiled _forge3d extension",
)

_native = get_native_module()

C4_APIS = (
    "union_geometries",
    "dissolve_vector",
    "buffer_geometry",
    "clip_vector",
    "intersect_vectors",
    "simplify_geometry",
    "load_boundary",
)


def _unit_square():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        ],
    }


def _feature_collection():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "a"},
                "geometry": _unit_square(),
            }
        ],
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }


def _assert_backend_unavailable(call):
    with pytest.raises(RuntimeError) as exc:
        call()
    text = str(exc.value)
    assert "backend_unavailable" in text
    assert "geos-topology" in text


def test_c4_apis_are_importable_from_wrapper_all_stub_and_native():
    stub = (Path(gis.__file__).with_suffix(".pyi")).read_text(encoding="utf-8")

    for name in C4_APIS:
        assert hasattr(gis, name)
        assert callable(getattr(gis, name))
        assert name in gis.__all__
        assert f"def {name}(" in stub
        assert hasattr(_native, name)
        assert callable(getattr(_native, name))


def test_c4_apis_raise_backend_unavailable_without_topology_backend(tmp_path: Path):
    path = tmp_path / "boundary.geojson"
    path.write_text(
        '{"type":"FeatureCollection","features":[],"crs":{"type":"name","properties":{"name":"EPSG:4326"}}}',
        encoding="utf-8",
    )

    _assert_backend_unavailable(lambda: gis.union_geometries([_unit_square()]))
    _assert_backend_unavailable(lambda: gis.dissolve_vector(_feature_collection()))
    _assert_backend_unavailable(lambda: gis.buffer_geometry(_unit_square(), 1.0))
    _assert_backend_unavailable(lambda: gis.clip_vector(_feature_collection(), _unit_square()))
    _assert_backend_unavailable(
        lambda: gis.intersect_vectors(_feature_collection(), _feature_collection())
    )
    _assert_backend_unavailable(lambda: gis.simplify_geometry(_unit_square(), 0.1))
    _assert_backend_unavailable(lambda: gis.load_boundary(path))


def test_union_geometries_cheap_validation_and_empty_result():
    for bad in (None, 42, {"type": "Point", "coordinates": [0.0, 0.0]}, "not-sequence"):
        with pytest.raises(ValueError, match="invalid_argument"):
            gis.union_geometries(bad)

    result = gis.union_geometries([])

    assert result["geometry"] is None
    assert result["operation"]["name"] == "union_geometries"
    assert result["operation"]["input_count"] == 0
    assert result["operation"]["output_count"] == 0
    assert result["operation"]["warnings"][0]["code"] == "empty_input"


def test_buffer_geometry_cheap_validation_precedes_backend_gate():
    for distance in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="invalid_argument"):
            gis.buffer_geometry(_unit_square(), distance)

    with pytest.raises(ValueError, match="invalid_argument"):
        gis.buffer_geometry(_unit_square(), 1.0, quad_segs=0)


def test_simplify_geometry_cheap_validation_precedes_backend_gate():
    for tolerance in (math.inf, -math.inf, math.nan, -0.1):
        with pytest.raises(ValueError, match="invalid_argument"):
            gis.simplify_geometry(_unit_square(), tolerance)


@pytest.mark.parametrize("suffixes", [(), ("_x",), ("_x", "_y", "_z"), ("_x", 7)])
def test_intersect_vectors_invalid_suffixes_precede_backend_gate(suffixes):
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.intersect_vectors(_feature_collection(), _feature_collection(), suffixes=suffixes)


def test_load_boundary_where_unsupported_precedes_backend_gate(tmp_path: Path):
    path = tmp_path / "boundary.geojson"
    path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported_option"):
        gis.load_boundary(path, where="name = 'a'")


def test_no_python_gis_backend_libraries_in_overlay_wrapper():
    source = inspect.getsource(gis)
    for banned in ("rasterio", "geopandas", "shapely", "rioxarray", "xarray", "terra"):
        assert banned not in source

