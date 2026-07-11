"""G-002c C3 vector geometry validity and measurement tests."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS vector geometry tests require the compiled _forge3d extension",
)


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _unit_square():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        ],
    }


def _bowtie():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        ],
    }


def _line():
    return {"type": "LineString", "coordinates": [[0.0, 0.0], [3.0, 4.0]]}


def _feature(geometry):
    return {"type": "Feature", "properties": {"name": "test"}, "geometry": geometry}


def _feature_collection(geometries):
    return {"type": "FeatureCollection", "features": [_feature(geom) for geom in geometries]}


def _read_vector_result(geometries):
    payload = _feature_collection(geometries)
    payload["info"] = {
        "path": "",
        "driver": "GeoJSON",
        "layer_name": None,
        "layer_count": 1,
        "geometry_type": "Mixed",
        "feature_count": len(geometries),
        "schema": [],
        "crs_wkt": None,
        "crs_authority": {"name": "EPSG", "code": "4326"},
        "bounds": None,
        "is_georeferenced": True,
        "warnings": [],
    }
    return payload


def _polygon_with_hole():
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
            [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]],
        ],
    }


def test_validate_geometry_valid_unit_square():
    result = gis.validate_geometry(_unit_square())

    assert result == {
        "valid": True,
        "reason": None,
        "geometry_type": "Polygon",
        "warnings": [],
    }


def test_validate_geometry_detects_bowtie_self_intersection():
    result = gis.validate_geometry(_bowtie())

    assert result["valid"] is False
    assert "invalid_geometry" in result["reason"]
    assert "invalid_geometry" in _codes(result["warnings"])


def test_validate_geometry_empty_geometry_reports_empty_geometry():
    result = gis.validate_geometry({"type": "LineString", "coordinates": []})

    assert result["valid"] is False
    assert "empty_geometry" in result["reason"]
    assert "empty_geometry" in _codes(result["warnings"])


def test_validate_geometry_malformed_and_unsupported_geometry_errors():
    malformed = gis.validate_geometry({"type": "Point", "coordinates": ["x", 1.0]})
    unsupported = gis.validate_geometry(
        {"type": "CircularString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}
    )

    assert malformed["valid"] is False
    assert "invalid_geometry" in malformed["reason"]
    assert unsupported["valid"] is False
    assert "unsupported_geometry_type" in unsupported["reason"]


def test_feature_inputs_are_normalized_for_c3_operations():
    line_feature = _feature(_line())

    assert gis.validate_geometry(line_feature)["geometry_type"] == "LineString"
    assert gis.geometry_measure(line_feature, crs=32631)["length"] == pytest.approx(5.0)
    assert gis.geometry_centroid(line_feature)["geometry"]["coordinates"] == pytest.approx(
        [1.5, 2.0]
    )
    assert gis.representative_point(line_feature)["geometry"]["coordinates"] == pytest.approx(
        [1.5, 2.0]
    )
    assert gis.interpolate_line(line_feature, 2.5)["geometry"]["coordinates"] == pytest.approx(
        [1.5, 2.0]
    )


def test_feature_collection_validation_measurement_and_centroid():
    payload = _feature_collection(
        [
            {"type": "Point", "coordinates": [100.0, 100.0]},
            _line(),
            _unit_square(),
        ]
    )

    validation = gis.validate_geometry(payload)
    measure = gis.geometry_measure(payload, crs=32631)
    centroid = gis.geometry_centroid(payload)

    assert validation["valid"] is True
    assert validation["geometry_type"] == "FeatureCollection"
    assert measure["area"] == pytest.approx(1.0)
    assert measure["length"] == pytest.approx(9.0)
    assert measure["operation"]["input_count"] == 3
    assert centroid["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])
    assert centroid["operation"]["input_count"] == 3


def test_read_vector_result_style_crs_propagates_to_operations():
    payload = _read_vector_result([_unit_square()])

    measure = gis.geometry_measure(payload, crs=32631)
    centroid = gis.geometry_centroid(payload)

    assert measure["operation"]["crs"]["source_kind"] == "vector"
    assert measure["operation"]["crs"]["authority"] == {"name": "EPSG", "code": "4326"}
    assert centroid["operation"]["crs"]["source_kind"] == "vector"
    assert centroid["operation"]["crs"]["authority"] == {"name": "EPSG", "code": "4326"}


def test_declared_projected_crs_does_not_trigger_dateline_unwrap_from_numeric_ranges():
    payload = _read_vector_result(
        [{"type": "LineString", "coordinates": [[179.0, 0.0], [-179.0, 0.0]]}]
    )
    payload["info"]["crs_authority"] = {"name": "EPSG", "code": "3857"}
    centroid = gis.geometry_centroid(payload)
    assert centroid["geometry"]["coordinates"] == pytest.approx([0.0, 0.0])


def test_dateline_representative_and_interpolated_points_stay_local():
    line = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "LineString", "coordinates": [[179.0, 0.0], [-179.0, 0.0]]},
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }
    representative = gis.representative_point(line)
    rep_lon = representative["geometry"]["coordinates"][0]
    assert abs(rep_lon) > 179.0

    interpolated = gis.interpolate_line(line, 0.5, normalized=True)
    interp_lon = interpolated["geometry"]["coordinates"][0]
    assert abs(interp_lon) > 179.0


def test_geometry_measure_polygon_area_and_boundary_length():
    result = gis.geometry_measure(_unit_square(), crs=32631)

    assert result["area"] == pytest.approx(1.0)
    assert result["length"] == pytest.approx(4.0)
    assert result["units"] == "source_crs_planar"
    assert result["operation"]["name"] == "geometry_measure"


def test_geometry_measure_polygon_with_hole_area_boundary_and_centroid():
    measure = gis.geometry_measure(_polygon_with_hole(), crs=32631)
    centroid = gis.geometry_centroid(_polygon_with_hole())

    assert measure["area"] == pytest.approx(12.0)
    assert measure["length"] == pytest.approx(24.0)
    assert centroid["geometry"]["coordinates"] == pytest.approx([2.0, 2.0])


def test_geometry_measure_metric_subsets():
    area_only = gis.geometry_measure(_unit_square(), crs=32631, metrics=("area",))
    length_only = gis.geometry_measure(_unit_square(), crs=32631, metrics=("length",))

    assert area_only["area"] == pytest.approx(1.0)
    assert area_only["length"] is None
    assert length_only["area"] is None
    assert length_only["length"] == pytest.approx(4.0)


def test_geometry_measure_line_length():
    result = gis.geometry_measure(_line(), crs=32631)

    assert result["area"] is None
    assert result["length"] == pytest.approx(5.0)


def test_geometry_measure_points_return_none_for_area_and_length():
    result = gis.geometry_measure({"type": "Point", "coordinates": [1.0, 2.0]}, crs=32631)

    assert result["area"] is None
    assert result["length"] is None


def test_geometry_measure_geometry_collection_sums_applicable_metrics():
    result = gis.geometry_measure(
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [10.0, 10.0]},
                _line(),
                _unit_square(),
            ],
        },
        crs=32631,
    )

    assert result["area"] == pytest.approx(1.0)
    assert result["length"] == pytest.approx(9.0)
    assert result["operation"]["input_count"] == 3


def test_geometry_measure_rejects_unsupported_metric():
    with pytest.raises(ValueError, match="unsupported_option"):
        gis.geometry_measure(_unit_square(), crs=32631, metrics=("area", "volume"))


def test_geometry_centroid_point_line_polygon_and_collection():
    point = gis.geometry_centroid({"type": "Point", "coordinates": [2.0, 3.0]})
    line = gis.geometry_centroid({"type": "LineString", "coordinates": [[0.0, 0.0], [4.0, 0.0]]})
    polygon = gis.geometry_centroid(_unit_square())
    collection = gis.geometry_centroid(
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [100.0, 100.0]},
                _line(),
                _unit_square(),
            ],
        }
    )

    assert point["geometry"]["coordinates"] == pytest.approx([2.0, 3.0])
    assert line["geometry"]["coordinates"] == pytest.approx([2.0, 0.0])
    assert polygon["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])
    assert collection["geometry"]["coordinates"] == pytest.approx([0.5, 0.5])


def test_representative_point_point_and_line_without_topology_backend():
    point = gis.representative_point({"type": "Point", "coordinates": [2.0, 3.0]})
    line = gis.representative_point(_line())

    assert point["geometry"]["coordinates"] == pytest.approx([2.0, 3.0])
    assert line["geometry"]["coordinates"] == pytest.approx([1.5, 2.0])


def test_representative_point_geometry_collection_uses_input_order_for_points_and_lines():
    point_first = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": [9.0, 9.0]},
            {"type": "LineString", "coordinates": [[0.0, 0.0], [4.0, 0.0]]},
        ],
    }
    line_first = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "LineString", "coordinates": [[0.0, 0.0], [4.0, 0.0]]},
            {"type": "Point", "coordinates": [9.0, 9.0]},
        ],
    }

    assert gis.representative_point(point_first)["geometry"]["coordinates"] == pytest.approx(
        [9.0, 9.0]
    )
    assert gis.representative_point(line_first)["geometry"]["coordinates"] == pytest.approx(
        [2.0, 0.0]
    )


def test_representative_point_polygon_requires_or_uses_geos_topology():
    with pytest.raises(RuntimeError, match="backend_unavailable|geos-topology"):
        gis.representative_point(_unit_square())


def test_interpolate_line_linestring_distance():
    result = gis.interpolate_line(_line(), 2.5)
    normalized = gis.interpolate_line(_line(), 0.5, normalized=True)

    assert result["geometry"]["coordinates"] == pytest.approx([1.5, 2.0])
    assert result["distance"] == pytest.approx(2.5)
    assert result["normalized"] is False
    assert normalized["geometry"]["coordinates"] == pytest.approx([1.5, 2.0])
    assert normalized["normalized"] is True


def test_interpolate_line_boundary_distances():
    line = _line()

    assert gis.interpolate_line(line, 0.0)["geometry"]["coordinates"] == pytest.approx([0.0, 0.0])
    assert gis.interpolate_line(line, 5.0)["geometry"]["coordinates"] == pytest.approx([3.0, 4.0])
    assert gis.interpolate_line(line, 0.0, normalized=True)["geometry"][
        "coordinates"
    ] == pytest.approx([0.0, 0.0])
    assert gis.interpolate_line(line, 1.0, normalized=True)["geometry"][
        "coordinates"
    ] == pytest.approx([3.0, 4.0])


def test_interpolate_line_multilinestring_cumulative_order():
    geometry = {
        "type": "MultiLineString",
        "coordinates": [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 2.0]],
        ],
    }

    result = gis.interpolate_line(geometry, 3.0)

    assert result["geometry"]["coordinates"] == pytest.approx([2.0, 1.0])


def test_interpolate_line_rejects_negative_out_of_range_and_unsupported_geometry():
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.interpolate_line(_line(), -0.1)
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.interpolate_line(_line(), 6.0)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.interpolate_line(_unit_square(), 0.5)


def test_repair_geometry_make_valid_requires_or_uses_geos_topology():
    with pytest.raises(RuntimeError, match="backend_unavailable|geos-topology"):
        gis.repair_geometry(_bowtie())


def test_repair_geometry_unsupported_method_returns_unsupported_option():
    with pytest.raises(ValueError, match="unsupported_option"):
        gis.repair_geometry(_unit_square(), method="buffer_zero")


def test_feature_collection_rejected_for_repair_representative_and_interpolate():
    payload = _feature_collection([_line()])

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.repair_geometry(payload)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.representative_point(payload)
    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.interpolate_line(payload, 0.0)


def test_no_python_gis_backend_imports_in_forge3d_gis():
    source = inspect.getsource(gis)
    tree = ast.parse(source)
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    for banned in ("rasterio", "geopandas", "shapely", "rioxarray", "xarray", "terra"):
        assert banned not in imports


