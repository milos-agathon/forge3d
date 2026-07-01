"""G-002c C4.0 vector overlay public surface and backend-gate tests."""

from __future__ import annotations

import inspect
import math
import os
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


def _shifted_square(x0: float, y0: float, size: float = 1.0):
    x1 = x0 + size
    y1 = y0 + size
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _feature(geometry):
    return {"type": "Feature", "properties": {"ignored": True}, "geometry": geometry}


def _has_topology_backend() -> bool:
    try:
        gis.union_geometries([_unit_square()])
    except RuntimeError as exc:
        if "backend_unavailable" in str(exc) and "geos-topology" in str(exc):
            return False
        raise
    return True


def _require_topology_backend():
    if not _has_topology_backend():
        if os.environ.get("FORGE3D_EXPECT_GEOS_TOPOLOGY") == "1":
            pytest.fail("geos-topology backend was expected but is unavailable")
        pytest.skip("requires forge3d built with geos-topology")


def _ring_area(ring) -> float:
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(ring, ring[1:])
        )
        / 2.0
    )


def _geometry_area(geometry) -> float:
    kind = geometry["type"]
    if kind == "Polygon":
        rings = geometry["coordinates"]
        return _ring_area(rings[0]) - sum(_ring_area(ring) for ring in rings[1:])
    if kind == "MultiPolygon":
        return sum(
            _ring_area(poly[0]) - sum(_ring_area(ring) for ring in poly[1:])
            for poly in geometry["coordinates"]
        )
    raise AssertionError(f"unexpected geometry type {kind!r}")


def _warning_codes(result):
    return {warning["code"] for warning in result["operation"]["warnings"]}


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
    if _has_topology_backend():
        pytest.skip("backend_unavailable contract is default/no-topology only")

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


def test_union_geometries_overlapping_squares_with_topology_backend():
    _require_topology_backend()

    result = gis.union_geometries([_unit_square(), _shifted_square(0.5, 0.0)])

    assert result["geometry"]["type"] == "Polygon"
    assert _geometry_area(result["geometry"]) == pytest.approx(1.5)
    assert result["operation"]["name"] == "union_geometries"
    assert result["operation"]["input_count"] == 2
    assert result["operation"]["output_count"] == 1


def test_union_geometries_disjoint_polygons_report_type_change():
    _require_topology_backend()

    result = gis.union_geometries([_unit_square(), _shifted_square(2.0, 0.0)])

    assert result["geometry"]["type"] in {"MultiPolygon", "GeometryCollection"}
    assert _geometry_area(result["geometry"]) == pytest.approx(2.0)
    assert result["operation"]["changed"] is True
    assert "geometry_type_changed" in _warning_codes(result)


def test_union_geometries_feature_inputs_use_geometry_only():
    _require_topology_backend()

    result = gis.union_geometries([
        _feature(_unit_square()),
        _feature(_shifted_square(0.5, 0.0)),
    ])

    assert result["geometry"]["type"] == "Polygon"
    assert _geometry_area(result["geometry"]) == pytest.approx(1.5)


def test_union_geometries_feature_collection_shorthand():
    _require_topology_backend()
    collection = {
        "type": "FeatureCollection",
        "features": [
            _feature(_unit_square()),
            _feature(_shifted_square(0.5, 0.0)),
        ],
    }

    result = gis.union_geometries(collection)

    assert result["geometry"]["type"] == "Polygon"
    assert _geometry_area(result["geometry"]) == pytest.approx(1.5)


def test_union_geometries_rejects_invalid_bowtie_with_topology_backend():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.union_geometries([bowtie])


def test_union_geometries_rejects_unsupported_type_with_topology_backend():
    _require_topology_backend()

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.union_geometries([{"type": "Point", "coordinates": [0.0, 0.0]}])


def test_union_geometries_empty_input_stays_empty_with_topology_backend():
    _require_topology_backend()

    result = gis.union_geometries([])

    assert result["geometry"] is None
    assert result["operation"]["warnings"][0]["code"] == "empty_input"


def test_union_geometries_optional_shapely_reference_area():
    _require_topology_backend()
    shapely_geometry = pytest.importorskip("shapely.geometry")
    shapely_ops = pytest.importorskip("shapely.ops")

    left = _unit_square()
    right = _shifted_square(0.5, 0.0)
    result = gis.union_geometries([left, right])
    expected = shapely_ops.unary_union([
        shapely_geometry.shape(left),
        shapely_geometry.shape(right),
    ])

    assert _geometry_area(result["geometry"]) == pytest.approx(expected.area)


def test_non_union_c4_apis_stay_gated_with_topology_backend(tmp_path: Path):
    _require_topology_backend()
    path = tmp_path / "boundary.geojson"
    path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    _assert_backend_unavailable(lambda: gis.dissolve_vector(_feature_collection()))
    _assert_backend_unavailable(lambda: gis.buffer_geometry(_unit_square(), 1.0))
    _assert_backend_unavailable(lambda: gis.clip_vector(_feature_collection(), _unit_square()))
    _assert_backend_unavailable(
        lambda: gis.intersect_vectors(_feature_collection(), _feature_collection())
    )
    _assert_backend_unavailable(lambda: gis.simplify_geometry(_unit_square(), 0.1))
    _assert_backend_unavailable(lambda: gis.load_boundary(path))

