"""G-002c C4.0 vector overlay public surface and backend-gate tests."""

from __future__ import annotations

import inspect
import json
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


def _point():
    return {"type": "Point", "coordinates": [0.0, 0.0]}


def _zigzag_line():
    return {
        "type": "LineString",
        "coordinates": [
            [0.0, 0.0],
            [1.0, 0.01],
            [2.0, 0.0],
            [3.0, 0.01],
            [4.0, 0.0],
        ],
    }


def _geojson_crs(code: int = 4326):
    return {"type": "name", "properties": {"name": f"EPSG:{code}"}}


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
        "crs": _geojson_crs(),
    }


def _feature_collection_from(features, *, crs=True):
    out = {"type": "FeatureCollection", "features": features}
    if crs:
        out["crs"] = _geojson_crs()
    return out


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


def _feature(geometry, properties=None):
    return {
        "type": "Feature",
        "properties": {"ignored": True} if properties is None else properties,
        "geometry": geometry,
    }


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


def _geometry_bounds(geometry):
    points = []

    def collect(value):
        if isinstance(value, (list, tuple)):
            if (
                len(value) >= 2
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))
            ):
                points.append((float(value[0]), float(value[1])))
            else:
                for item in value:
                    collect(item)

    collect(geometry.get("coordinates"))
    if not points:
        raise AssertionError("geometry has no coordinates")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _coordinate_count(geometry):
    if geometry["type"] == "LineString":
        return len(geometry["coordinates"])
    if geometry["type"] == "MultiLineString":
        return sum(len(line) for line in geometry["coordinates"])
    if geometry["type"] == "Polygon":
        return sum(len(ring) for ring in geometry["coordinates"])
    if geometry["type"] == "MultiPolygon":
        return sum(len(ring) for polygon in geometry["coordinates"] for ring in polygon)
    raise AssertionError(f"unexpected geometry type {geometry['type']!r}")


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
    _assert_backend_unavailable(
        lambda: gis.clip_vector(_feature_collection(), _unit_square(), clip_crs="EPSG:4326")
    )
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


def test_buffer_geometry_point_with_topology_backend_returns_polygonal():
    _require_topology_backend()

    result = gis.buffer_geometry(_point(), 1.0, quad_segs=4)

    assert result["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    assert _geometry_area(result["geometry"]) > 2.0
    assert result["operation"]["name"] == "buffer_geometry"
    assert result["operation"]["input_geometry_type"] == "Point"
    assert result["operation"]["output_geometry_type"] in {"Polygon", "MultiPolygon"}
    assert result["operation"]["input_count"] == 1
    assert result["operation"]["output_count"] == 1
    assert result["operation"]["changed"] is True
    assert "geometry_type_changed" in _warning_codes(result)


def test_buffer_geometry_polygon_positive_increases_area_and_bounds():
    _require_topology_backend()

    result = gis.buffer_geometry(_unit_square(), 0.25, quad_segs=8)
    bounds = _geometry_bounds(result["geometry"])

    assert _geometry_area(result["geometry"]) > _geometry_area(_unit_square())
    assert bounds[0] < 0.0
    assert bounds[1] < 0.0
    assert bounds[2] > 1.0
    assert bounds[3] > 1.0


def test_buffer_geometry_zero_distance_valid_polygon_is_deterministic():
    _require_topology_backend()

    result = gis.buffer_geometry(_unit_square(), 0.0)

    assert result["geometry"]["type"] == "Polygon"
    assert _geometry_area(result["geometry"]) == pytest.approx(1.0)
    assert result["operation"]["changed"] is False
    assert result["operation"]["warnings"] == []


def test_buffer_geometry_negative_point_returns_empty_output():
    _require_topology_backend()

    result = gis.buffer_geometry(_point(), -1.0)

    assert result["geometry"] is None
    assert result["operation"]["output_geometry_type"] is None
    assert result["operation"]["output_count"] == 0
    assert result["operation"]["changed"] is True
    assert "empty_output" in _warning_codes(result)


def test_buffer_geometry_feature_input_uses_geometry_only():
    _require_topology_backend()

    result = gis.buffer_geometry(_feature(_point()), 1.0)

    assert result["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    assert result["operation"]["input_geometry_type"] == "Point"
    assert result["operation"]["input_count"] == 1


def test_buffer_geometry_rejects_invalid_bowtie_polygon():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.buffer_geometry(bowtie, 1.0)


def test_buffer_geometry_rejects_malformed_unsupported_and_feature_collection():
    _require_topology_backend()

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.buffer_geometry({"type": "Polygon", "coordinates": [[["bad", 0.0]]]}, 1.0)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.buffer_geometry({"type": "Triangle", "coordinates": []}, 1.0)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.buffer_geometry({"type": "FeatureCollection", "features": [_feature(_point())]}, 1.0)


def test_buffer_geometry_optional_shapely_reference_area():
    _require_topology_backend()
    shapely_geometry = pytest.importorskip("shapely.geometry")

    result = gis.buffer_geometry(_unit_square(), 0.25, quad_segs=8)
    source = shapely_geometry.shape(_unit_square())
    try:
        expected = source.buffer(0.25, quad_segs=8)
    except TypeError:
        expected = source.buffer(0.25, resolution=8)

    assert _geometry_area(result["geometry"]) == pytest.approx(expected.area, rel=0.15)


def test_clip_vector_polygon_source_clipped_by_polygon_recomputes_metadata():
    _require_topology_backend()
    source = _feature_collection_from([
        _feature(_unit_square(), {"name": "parcel", "value": 7})
    ])

    result = gis.clip_vector(source, _shifted_square(0.5, 0.0), clip_crs="EPSG:4326")

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"name": "parcel", "value": 7}
    assert result["features"][0]["geometry"]["type"] == "Polygon"
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(0.5)
    assert result["info"]["feature_count"] == 1
    assert result["info"]["geometry_type"] == "Polygon"
    assert [field["name"] for field in result["info"]["schema"]] == ["name", "value"]
    assert result["info"]["bounds"] == pytest.approx((0.5, 0.0, 1.0, 1.0))
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert result["warnings"] == []
    assert result["operation"]["name"] == "clip_vector"
    assert result["operation"]["input_count"] == 1
    assert result["operation"]["output_count"] == 1


def test_clip_vector_no_intersections_returns_empty_output():
    _require_topology_backend()

    result = gis.clip_vector(
        _feature_collection(),
        _shifted_square(3.0, 3.0),
        clip_crs="EPSG:4326",
    )

    assert result["features"] == []
    assert result["info"]["feature_count"] == 0
    assert result["info"]["geometry_type"] == "Empty"
    assert result["info"]["bounds"] is None
    assert "empty_output" in _warning_codes(result)


def test_clip_vector_empty_source_returns_empty_feature_set():
    _require_topology_backend()
    source = _feature_collection_from([])

    result = gis.clip_vector(source, _unit_square(), clip_crs="EPSG:4326")

    assert result["features"] == []
    assert result["info"]["feature_count"] == 0
    assert "empty_feature_set" in _warning_codes(result)


def test_clip_vector_path_input_works(tmp_path: Path):
    _require_topology_backend()
    path = tmp_path / "clip-source.geojson"
    path.write_text(
        json.dumps(_feature_collection_from([_feature(_unit_square(), {"id": 1})])),
        encoding="utf-8",
    )

    result = gis.clip_vector(path, _shifted_square(0.5, 0.0), clip_crs="EPSG:4326")

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"id": 1}
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(0.5)


def test_clip_vector_read_vector_result_input_works(tmp_path: Path):
    _require_topology_backend()
    path = tmp_path / "clip-read-result.geojson"
    path.write_text(
        json.dumps(_feature_collection_from([_feature(_unit_square(), {"id": 2})])),
        encoding="utf-8",
    )
    source = gis.read_vector(path)

    result = gis.clip_vector(source, _shifted_square(0.5, 0.0), clip_crs="EPSG:4326")

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"id": 2}


def test_clip_vector_clip_feature_input_uses_geometry_only():
    _require_topology_backend()

    result = gis.clip_vector(
        _feature_collection(),
        _feature(_shifted_square(0.5, 0.0), {"ignored": True}),
        clip_crs="EPSG:4326",
    )

    assert len(result["features"]) == 1
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(0.5)


def test_clip_vector_clip_feature_collection_input_is_union_mask():
    _require_topology_backend()
    mask = _feature_collection_from(
        [
            _feature(_shifted_square(-0.5, 0.0)),
            _feature(_shifted_square(0.5, 0.0)),
        ]
    )

    result = gis.clip_vector(_feature_collection(), mask)

    assert len(result["features"]) == 1
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(1.0)


def test_clip_vector_source_crs_missing_raises_missing_crs():
    _require_topology_backend()
    source = _feature_collection_from([_feature(_unit_square())], crs=False)

    with pytest.raises(ValueError, match="missing_crs"):
        gis.clip_vector(source, _unit_square(), clip_crs="EPSG:4326")


def test_clip_vector_clip_crs_missing_raises_missing_crs():
    _require_topology_backend()

    with pytest.raises(ValueError, match="missing_crs"):
        gis.clip_vector(_feature_collection(), _unit_square())


def test_clip_vector_mismatched_crs_raises_crs_mismatch():
    _require_topology_backend()

    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.clip_vector(_feature_collection(), _unit_square(), clip_crs="EPSG:3857")


def test_clip_vector_invalid_source_geometry_raises_invalid_geometry():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }
    source = _feature_collection_from([_feature(bowtie)])

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.clip_vector(source, _unit_square(), clip_crs="EPSG:4326")


def test_clip_vector_invalid_clip_geometry_raises_invalid_geometry():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.clip_vector(_feature_collection(), bowtie, clip_crs="EPSG:4326")


def test_clip_vector_non_polygonal_geometry_raises_unsupported_type():
    _require_topology_backend()
    source = _feature_collection_from([_feature(_point())])

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.clip_vector(source, _unit_square(), clip_crs="EPSG:4326")
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.clip_vector(_feature_collection(), _point(), clip_crs="EPSG:4326")


def test_intersect_vectors_one_polygon_overlap_recomputes_metadata():
    _require_topology_backend()
    left = _feature_collection_from([
        _feature(_unit_square(), {"left_id": 1, "name": "left"})
    ])
    right = _feature_collection_from([
        _feature(_shifted_square(0.5, 0.0), {"right_id": 2})
    ])

    result = gis.intersect_vectors(left, right)

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {
        "left_id": 1,
        "name": "left",
        "right_id": 2,
    }
    assert result["features"][0]["geometry"]["type"] == "Polygon"
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(0.5)
    assert result["info"]["feature_count"] == 1
    assert result["info"]["geometry_type"] == "Polygon"
    assert {field["name"] for field in result["info"]["schema"]} == {
        "left_id",
        "name",
        "right_id",
    }
    assert result["info"]["bounds"] == pytest.approx((0.5, 0.0, 1.0, 1.0))
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert result["warnings"] == []
    assert result["operation"]["name"] == "intersect_vectors"
    assert result["operation"]["input_count"] == 2
    assert result["operation"]["output_count"] == 1


def test_intersect_vectors_no_overlap_returns_empty_output():
    _require_topology_backend()

    result = gis.intersect_vectors(
        _feature_collection(),
        _feature_collection_from([_feature(_shifted_square(3.0, 3.0))]),
    )

    assert result["features"] == []
    assert result["info"]["feature_count"] == 0
    assert result["info"]["geometry_type"] == "Empty"
    assert result["info"]["bounds"] is None
    assert "empty_output" in _warning_codes(result)


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_intersect_vectors_empty_input_returns_empty_feature_set(empty_side):
    _require_topology_backend()
    empty = _feature_collection_from([])
    nonempty = _feature_collection()
    left, right = (empty, nonempty) if empty_side == "left" else (nonempty, empty)

    result = gis.intersect_vectors(left, right)

    assert result["features"] == []
    assert result["info"]["feature_count"] == 0
    assert "empty_feature_set" in _warning_codes(result)


def test_intersect_vectors_path_input_works(tmp_path: Path):
    _require_topology_backend()
    left_path = tmp_path / "intersect-left.geojson"
    right_path = tmp_path / "intersect-right.geojson"
    left_path.write_text(
        json.dumps(_feature_collection_from([_feature(_unit_square(), {"id": 1})])),
        encoding="utf-8",
    )
    right_path.write_text(
        json.dumps(_feature_collection_from([_feature(_shifted_square(0.5, 0.0), {"kind": "r"})])),
        encoding="utf-8",
    )

    result = gis.intersect_vectors(left_path, right_path)

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"id": 1, "kind": "r"}
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(0.5)


def test_intersect_vectors_read_vector_result_input_works(tmp_path: Path):
    _require_topology_backend()
    left_path = tmp_path / "intersect-read-left.geojson"
    left_path.write_text(
        json.dumps(_feature_collection_from([_feature(_unit_square(), {"id": 3})])),
        encoding="utf-8",
    )
    left = gis.read_vector(left_path)
    right = _feature_collection_from([
        _feature(_shifted_square(0.5, 0.0), {"kind": "right"})
    ])

    result = gis.intersect_vectors(left, right)

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"id": 3, "kind": "right"}


def test_intersect_vectors_property_collision_applies_suffixes():
    _require_topology_backend()
    left = _feature_collection_from([
        _feature(_unit_square(), {"id": 1, "name": "left", "left_only": True})
    ])
    right = _feature_collection_from([
        _feature(_shifted_square(0.5, 0.0), {"id": 2, "name": "right", "right_only": True})
    ])

    result = gis.intersect_vectors(left, right, suffixes=("_l", "_r"))

    assert result["features"][0]["properties"] == {
        "id_l": 1,
        "id_r": 2,
        "left_only": True,
        "name_l": "left",
        "name_r": "right",
        "right_only": True,
    }


def test_intersect_vectors_generated_property_collision_raises():
    _require_topology_backend()
    left = _feature_collection_from([
        _feature(_unit_square(), {"id": 1, "id_left": "existing"})
    ])
    right = _feature_collection_from([
        _feature(_shifted_square(0.5, 0.0), {"id": 2})
    ])

    with pytest.raises(ValueError, match="property_collision"):
        gis.intersect_vectors(left, right)


def test_intersect_vectors_missing_crs_raises_missing_crs():
    _require_topology_backend()

    with pytest.raises(ValueError, match="missing_crs"):
        gis.intersect_vectors(_feature_collection_from([_feature(_unit_square())], crs=False), _feature_collection())
    with pytest.raises(ValueError, match="missing_crs"):
        gis.intersect_vectors(_feature_collection(), _feature_collection_from([_feature(_unit_square())], crs=False))


def test_intersect_vectors_crs_mismatch_raises_crs_mismatch():
    _require_topology_backend()
    right = _feature_collection_from([_feature(_unit_square())])
    right["crs"] = _geojson_crs(3857)

    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.intersect_vectors(_feature_collection(), right)


def test_intersect_vectors_invalid_geometry_raises_invalid_geometry():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.intersect_vectors(_feature_collection_from([_feature(bowtie)]), _feature_collection())
    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.intersect_vectors(_feature_collection(), _feature_collection_from([_feature(bowtie)]))


def test_intersect_vectors_non_polygonal_geometry_raises_unsupported_type():
    _require_topology_backend()

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.intersect_vectors(_feature_collection_from([_feature(_point())]), _feature_collection())
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.intersect_vectors(_feature_collection(), _feature_collection_from([_feature(_point())]))


def test_simplify_geometry_linestring_reduces_points_and_preserves_endpoints():
    _require_topology_backend()
    source = _zigzag_line()

    result = gis.simplify_geometry(source, 0.05, preserve_topology=False)

    assert result["geometry"]["type"] == "LineString"
    assert len(result["geometry"]["coordinates"]) < len(source["coordinates"])
    assert result["geometry"]["coordinates"][0] == source["coordinates"][0]
    assert result["geometry"]["coordinates"][-1] == source["coordinates"][-1]
    assert result["operation"]["name"] == "simplify_geometry"
    assert result["operation"]["input_geometry_type"] == "LineString"
    assert result["operation"]["output_geometry_type"] == "LineString"
    assert result["operation"]["input_count"] == 1
    assert result["operation"]["output_count"] == 1
    assert result["operation"]["changed"] is True


def test_simplify_geometry_polygon_preserve_topology_stays_valid():
    _require_topology_backend()
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [1.0, 0.02],
                [2.0, 0.0],
                [2.0, 1.0],
                [1.0, 1.02],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ],
    }

    result = gis.simplify_geometry(polygon, 0.05, preserve_topology=True)

    assert result["geometry"]["type"] == "Polygon"
    assert gis.validate_geometry(result["geometry"])["valid"] is True
    assert result["operation"]["output_count"] == 1


@pytest.mark.parametrize(
    "geometry",
    [
        _zigzag_line(),
        {"type": "MultiLineString", "coordinates": [_zigzag_line()["coordinates"]]},
        _unit_square(),
        {"type": "MultiPolygon", "coordinates": [_unit_square()["coordinates"]]},
    ],
)
def test_simplify_geometry_preserve_false_supported_types(geometry):
    _require_topology_backend()

    result = gis.simplify_geometry(geometry, 0.01, preserve_topology=False)

    assert result["geometry"] is not None
    assert result["geometry"]["type"] == geometry["type"]
    assert result["operation"]["output_count"] == 1


def test_simplify_geometry_zero_tolerance_is_deterministic():
    _require_topology_backend()
    source = _zigzag_line()

    result = gis.simplify_geometry(source, 0.0, preserve_topology=True)

    assert result["geometry"] == source
    assert result["operation"]["changed"] is False
    assert result["operation"]["warnings"] == []


def test_simplify_geometry_feature_input_uses_geometry_only():
    _require_topology_backend()

    result = gis.simplify_geometry(_feature(_zigzag_line(), {"ignored": True}), 0.05)

    assert result["geometry"]["type"] == "LineString"
    assert "properties" not in result["geometry"]
    assert _coordinate_count(result["geometry"]) < _coordinate_count(_zigzag_line())


def test_simplify_geometry_rejects_invalid_bowtie():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.simplify_geometry(bowtie, 0.1)


def test_simplify_geometry_rejects_unsupported_inputs():
    _require_topology_backend()

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.simplify_geometry(_point(), 0.1)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.simplify_geometry(_feature_collection(), 0.1)


def test_dissolve_vector_all_returns_one_feature_with_empty_properties():
    _require_topology_backend()
    source = _feature_collection_from(
        [
            _feature(_unit_square(), {"group": "a", "value": 1}),
            _feature(_shifted_square(2.0, 0.0), {"group": "b", "value": 2}),
        ]
    )

    result = gis.dissolve_vector(source)

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {}
    assert result["features"][0]["geometry"]["type"] in {"MultiPolygon", "GeometryCollection"}
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(2.0)
    assert result["info"]["feature_count"] == 1
    assert result["info"]["geometry_type"] == result["features"][0]["geometry"]["type"]
    assert result["info"]["bounds"] == pytest.approx((0.0, 0.0, 3.0, 1.0))
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert result["operation"]["name"] == "dissolve_vector"
    assert result["operation"]["input_count"] == 2
    assert result["operation"]["output_count"] == 1
    assert "geometry_type_changed" in _warning_codes(result)


def test_dissolve_vector_by_one_field_returns_group_features():
    _require_topology_backend()
    source = _feature_collection_from(
        [
            _feature(_unit_square(), {"zone": "a", "drop": 1}),
            _feature(_shifted_square(0.5, 0.0), {"zone": "a", "drop": 2}),
            _feature(_shifted_square(3.0, 0.0), {"zone": "b", "drop": 3}),
        ]
    )

    result = gis.dissolve_vector(source, by="zone")

    assert len(result["features"]) == 2
    features = {feature["properties"]["zone"]: feature for feature in result["features"]}
    assert set(features) == {"a", "b"}
    assert features["a"]["properties"] == {"zone": "a"}
    assert _geometry_area(features["a"]["geometry"]) == pytest.approx(1.5)
    assert features["b"]["properties"] == {"zone": "b"}
    assert _geometry_area(features["b"]["geometry"]) == pytest.approx(1.0)
    assert [field["name"] for field in result["info"]["schema"]] == ["zone"]


def test_dissolve_vector_by_multiple_fields_groups_exact_values():
    _require_topology_backend()
    source = _feature_collection_from(
        [
            _feature(_unit_square(), {"zone": "a", "kind": 1, "drop": "x"}),
            _feature(_shifted_square(0.5, 0.0), {"zone": "a", "kind": 1, "drop": "y"}),
            _feature(_shifted_square(3.0, 0.0), {"zone": "a", "kind": 2, "drop": "z"}),
        ]
    )

    result = gis.dissolve_vector(source, by=("zone", "kind"))

    properties = [feature["properties"] for feature in result["features"]]
    assert properties == [{"kind": 1, "zone": "a"}, {"kind": 2, "zone": "a"}]
    assert all(set(props) == {"zone", "kind"} for props in properties)
    assert result["info"]["feature_count"] == 2


def test_dissolve_vector_missing_field_raises_missing_field():
    _require_topology_backend()
    source = _feature_collection_from([_feature(_unit_square(), {"zone": "a"})])

    with pytest.raises(ValueError, match="missing_field"):
        gis.dissolve_vector(source, by="missing")


@pytest.mark.parametrize("by", [(), [], ("zone", 7), 7])
def test_dissolve_vector_invalid_by_values_raise_invalid_argument(by):
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.dissolve_vector(_feature_collection(), by=by)


def test_dissolve_vector_empty_source_returns_empty_feature_set():
    _require_topology_backend()

    result = gis.dissolve_vector(_feature_collection_from([]))

    assert result["features"] == []
    assert result["info"]["feature_count"] == 0
    assert result["info"]["geometry_type"] == "Empty"
    assert result["info"]["bounds"] is None
    assert "empty_feature_set" in _warning_codes(result)


def test_dissolve_vector_invalid_geometry_raises_invalid_geometry():
    _require_topology_backend()
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }

    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.dissolve_vector(_feature_collection_from([_feature(bowtie)]))


def test_dissolve_vector_non_polygonal_geometry_raises_unsupported_type():
    _require_topology_backend()

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.dissolve_vector(_feature_collection_from([_feature(_point())]))


def test_dissolve_vector_path_input_works(tmp_path: Path):
    _require_topology_backend()
    path = tmp_path / "dissolve-source.geojson"
    path.write_text(
        json.dumps(
            _feature_collection_from(
                [
                    _feature(_unit_square(), {"zone": "a"}),
                    _feature(_shifted_square(0.5, 0.0), {"zone": "a"}),
                ]
            )
        ),
        encoding="utf-8",
    )

    result = gis.dissolve_vector(path, by="zone")

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"zone": "a"}
    assert _geometry_area(result["features"][0]["geometry"]) == pytest.approx(1.5)


def test_dissolve_vector_read_vector_result_input_works(tmp_path: Path):
    _require_topology_backend()
    path = tmp_path / "dissolve-read-result.geojson"
    path.write_text(
        json.dumps(_feature_collection_from([_feature(_unit_square(), {"zone": "read"})])),
        encoding="utf-8",
    )

    result = gis.dissolve_vector(gis.read_vector(path), by="zone")

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"zone": "read"}


def test_dissolve_vector_raw_feature_collection_input_works():
    _require_topology_backend()

    result = gis.dissolve_vector(
        _feature_collection_from([_feature(_unit_square(), {"zone": "raw"})]),
        by="zone",
    )

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"zone": "raw"}


def test_dissolve_vector_missing_crs_preserves_warning_metadata():
    _require_topology_backend()
    source = _feature_collection_from([_feature(_unit_square(), {"zone": "a"})], crs=False)

    result = gis.dissolve_vector(source, by="zone")

    assert result["info"]["crs_authority"] is None
    assert result["info"]["crs_wkt"] is None
    assert result["info"]["is_georeferenced"] is False
    assert "missing_crs" in _warning_codes(result)


def test_remaining_c4_apis_stay_gated_with_topology_backend(tmp_path: Path):
    _require_topology_backend()
    path = tmp_path / "boundary.geojson"
    path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    _assert_backend_unavailable(lambda: gis.load_boundary(path))

