"""G-002c C1 vector metadata/read contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS vector tests require the compiled _forge3d extension",
)


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _write_geojson(
    path: Path,
    *,
    features: list[dict],
    crs: bool = True,
    name: str = "sample_layer",
):
    payload = {
        "type": "FeatureCollection",
        "name": name,
        "features": features,
    }
    if crs:
        payload["crs"] = {"type": "name", "properties": {"name": "EPSG:4326"}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sample_features() -> list[dict]:
    return [
        {
            "type": "Feature",
            "properties": {
                "name": "alpha",
                "population": 10,
                "temperature": 21.5,
                "active": True,
            },
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
        },
        {
            "type": "Feature",
            "properties": {
                "name": "beta",
                "population": 12,
                "temperature": 19.25,
                "active": False,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [0.0, 0.0],
                        [4.0, 0.0],
                        [4.0, 5.0],
                        [0.0, 5.0],
                        [0.0, 0.0],
                    ]
                ],
            },
        },
    ]


def _feature(kind: str, coordinates):
    return {
        "type": "Feature",
        "properties": {"kind": kind},
        "geometry": {"type": kind, "coordinates": coordinates},
    }


def test_read_vector_geojson_happy_path_and_vector_info(tmp_path: Path):
    path = _write_geojson(tmp_path / "sample.geojson", features=_sample_features())

    result = gis.read_vector(path)

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2
    assert result["features"][0]["geometry"]["type"] == "Point"
    assert result["info"]["driver"] == "GeoJSON"
    assert result["info"]["layer_name"] == "sample_layer"
    assert result["info"]["layer_count"] == 1
    assert result["info"]["geometry_type"] == "Mixed"
    assert result["info"]["feature_count"] == 2
    assert result["info"]["bounds"] == pytest.approx((0.0, 0.0, 4.0, 5.0))
    assert result["info"]["is_georeferenced"] is True
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert result["warnings"] == []

    vector_info = result["vector_info"]
    assert isinstance(vector_info, gis.VectorInfo)
    assert vector_info.path == str(path)
    assert vector_info.driver == "GeoJSON"
    assert vector_info.layer_name == "sample_layer"
    assert vector_info.feature_count == 2
    assert vector_info.bounds == pytest.approx((0.0, 0.0, 4.0, 5.0))
    assert vector_info.as_dict().keys() == {
        "path",
        "driver",
        "layer_name",
        "layer_count",
        "geometry_type",
        "feature_count",
        "schema",
        "crs_wkt",
        "crs_authority",
        "bounds",
        "is_georeferenced",
        "warnings",
    }
    assert not {
        "layers",
        "columns",
        "is_empty",
        "invalid_geometry_count",
    } & set(vector_info.as_dict())


def test_vector_metadata_helpers_read_path_info_and_read_result(tmp_path: Path):
    path = _write_geojson(tmp_path / "sample.geojson", features=_sample_features())
    result = gis.read_vector(path)

    assert gis.geometry_type(path) == "Mixed"
    assert gis.geometry_type(result) == "Mixed"
    assert gis.geometry_type({"type": "Point", "coordinates": [1.0, 2.0]}) == "Point"
    assert gis.feature_count(path) == 2
    assert gis.feature_count(result) == 2
    assert gis.vector_bounds(path) == pytest.approx((0.0, 0.0, 4.0, 5.0))
    assert gis.vector_bounds(result) == pytest.approx((0.0, 0.0, 4.0, 5.0))
    assert gis.vector_crs(path)["authority"] == {"name": "EPSG", "code": "4326"}
    assert gis.vector_crs(result)["authority"] == {"name": "EPSG", "code": "4326"}

    schema = {field["name"]: field for field in gis.vector_schema(path)}
    assert schema["name"]["type"] == "string"
    assert schema["population"]["type"] == "integer"
    assert schema["temperature"]["type"] == "float"
    assert schema["active"]["type"] == "boolean"


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        ({"type": "Point", "coordinates": [1.0, 2.0]}, "Point"),
        ({"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}, "LineString"),
        (
            {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]
                ],
            },
            "Polygon",
        ),
        ({"type": "MultiPoint", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}, "MultiPoint"),
        (
            {
                "type": "MultiLineString",
                "coordinates": [[[0.0, 0.0], [1.0, 1.0]]],
            },
            "MultiLineString",
        ),
        (
            {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]]
                ],
            },
            "MultiPolygon",
        ),
        (
            {
                "type": "GeometryCollection",
                "geometries": [{"type": "Point", "coordinates": [1.0, 2.0]}],
            },
            "GeometryCollection",
        ),
        ({"type": "GeometryCollection", "geometries": []}, "Empty"),
        ({"type": "UnknownThing", "coordinates": []}, "Unknown"),
    ],
)
def test_geometry_type_supports_c1_geojson_subset(geometry: dict, expected: str):
    assert gis.geometry_type(geometry) == expected


def test_geometry_type_reports_mixed_and_empty_from_feature_collections(tmp_path: Path):
    mixed = _write_geojson(tmp_path / "mixed.geojson", features=_sample_features())
    empty = _write_geojson(tmp_path / "empty.geojson", features=[])

    assert gis.geometry_type(mixed) == "Mixed"
    assert gis.geometry_type(empty) == "Empty"


def test_raw_feature_and_geometry_helper_paths():
    raw_geometry = {"type": "Point", "coordinates": [3.0, 4.0]}
    raw_feature = {
        "type": "Feature",
        "properties": {"name": "raw", "value": 7},
        "geometry": raw_geometry,
    }

    assert gis.geometry_type(raw_feature) == "Point"
    assert gis.geometry_type(raw_geometry) == "Point"
    assert gis.feature_count(raw_feature) == 1
    assert gis.feature_count(raw_geometry) == 1
    assert gis.vector_schema(raw_feature) == [
        {
            "name": "name",
            "type": "string",
            "nullable": False,
            "width": None,
            "precision": None,
        },
        {
            "name": "value",
            "type": "integer",
            "nullable": False,
            "width": None,
            "precision": None,
        },
    ]
    assert gis.vector_bounds(raw_feature) == pytest.approx((3.0, 4.0, 3.0, 4.0))
    assert gis.vector_bounds(raw_geometry) == pytest.approx((3.0, 4.0, 3.0, 4.0))
    assert gis.vector_crs(raw_geometry)["missing"] is True


def test_feature_count_zero_one_many_filtered_and_limited(tmp_path: Path):
    empty = _write_geojson(tmp_path / "empty.geojson", features=[])
    one = _write_geojson(
        tmp_path / "one.geojson",
        features=[{"type": "Feature", "properties": {}, "geometry": None}],
    )
    many = _write_geojson(tmp_path / "many.geojson", features=_sample_features())

    assert gis.feature_count(empty) == 0
    assert gis.feature_count(one) == 1
    assert gis.feature_count(many) == 2

    filtered = gis.read_vector(many, bbox=(100.0, 100.0, 101.0, 101.0))
    assert gis.feature_count(filtered) == 0

    limited = gis.read_vector(many, limit=1)
    assert len(limited["features"]) == 1
    assert gis.feature_count(limited) == 1

    limit_zero = gis.read_vector(many, limit=0)
    assert limit_zero["features"] == []
    assert gis.feature_count(limit_zero) == 0
    assert "empty_feature_set" in _codes(limit_zero["warnings"])


def test_vector_bounds_cover_point_and_polygon_sources(tmp_path: Path):
    point = _write_geojson(
        tmp_path / "point.geojson",
        features=[_feature("Point", [2.5, -1.25])],
    )
    polygon = _write_geojson(
        tmp_path / "polygon.geojson",
        features=[
            _feature(
                "Polygon",
                [[[0.0, -2.0], [4.0, -2.0], [4.0, 5.0], [0.0, -2.0]]],
            )
        ],
    )

    assert gis.vector_bounds(point) == pytest.approx((2.5, -1.25, 2.5, -1.25))
    assert gis.vector_bounds(polygon) == pytest.approx((0.0, -2.0, 4.0, 5.0))


def test_read_vector_missing_path_missing_layer_and_unsupported_driver(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="path not found"):
        gis.read_vector(tmp_path / "missing.geojson")

    path = _write_geojson(tmp_path / "sample.geojson", features=_sample_features())
    with pytest.raises(ValueError, match="missing_layer"):
        gis.read_vector(path, layer="absent")

    unsupported = tmp_path / "sample.csv"
    unsupported.write_text("x,y\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported_driver"):
        gis.read_vector(unsupported)

    backend_required = tmp_path / "sample.gpkg"
    backend_required.write_bytes(b"not a real gpkg")
    with pytest.raises(RuntimeError, match="backend_unavailable"):
        gis.read_vector(backend_required)


def test_empty_feature_collection_and_bbox_empty_result_warn(tmp_path: Path):
    empty_path = _write_geojson(tmp_path / "empty.geojson", features=[])
    empty = gis.read_vector(empty_path)

    assert empty["features"] == []
    assert empty["info"]["feature_count"] == 0
    assert empty["info"]["geometry_type"] == "Empty"
    assert empty["info"]["bounds"] is None
    assert "empty_feature_set" in _codes(empty["warnings"])
    with pytest.raises(ValueError, match="empty_feature_set"):
        gis.vector_bounds(empty)

    sample_path = _write_geojson(tmp_path / "sample.geojson", features=_sample_features())
    filtered = gis.read_vector(sample_path, bbox=(100.0, 100.0, 101.0, 101.0))

    assert filtered["features"] == []
    assert filtered["info"]["feature_count"] == 0
    assert "empty_feature_set" in _codes(filtered["warnings"])


def test_column_filter_updates_features_and_schema(tmp_path: Path):
    path = _write_geojson(tmp_path / "sample.geojson", features=_sample_features())

    result = gis.read_vector(path, columns=["name"])

    assert [feature["properties"] for feature in result["features"]] == [
        {"name": "alpha"},
        {"name": "beta"},
    ]
    assert result["info"]["schema"] == [
        {
            "name": "name",
            "type": "string",
            "nullable": False,
            "width": None,
            "precision": None,
        }
    ]


def test_missing_crs_reports_warning_without_guessing(tmp_path: Path):
    path = _write_geojson(tmp_path / "missing_crs.geojson", features=_sample_features(), crs=False)

    result = gis.read_vector(path)
    crs = gis.vector_crs(result)

    assert result["info"]["crs_authority"] is None
    assert result["info"]["crs_wkt"] is None
    assert result["info"]["is_georeferenced"] is False
    assert "missing_crs" in _codes(result["warnings"])
    assert crs["missing"] is True
    assert crs["authority"] is None
    assert "missing_crs" in _codes(crs["warnings"])


def test_invalid_crs_metadata_reports_stable_token(tmp_path: Path):
    path = tmp_path / "invalid_crs.geojson"
    payload = {
        "type": "FeatureCollection",
        "name": "bad_crs",
        "crs": {"type": "name", "properties": {"name": "EPSG:999999"}},
        "features": _sample_features(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid_crs"):
        gis.read_vector(path)


def test_invalid_geojson_and_structurally_invalid_payload_report_invalid_geometry(tmp_path: Path):
    bad_json = tmp_path / "bad.geojson"
    bad_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.read_vector(bad_json)

    bad_payload = tmp_path / "bad_payload.geojson"
    bad_payload.write_text(
        json.dumps({"type": "FeatureCollection", "features": [42]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid_geometry"):
        gis.read_vector(bad_payload)


def test_vector_bounds_order_is_left_bottom_right_top(tmp_path: Path):
    path = _write_geojson(tmp_path / "bounds.geojson", features=_sample_features())

    left, bottom, right, top = gis.vector_bounds(path)

    assert (left, bottom, right, top) == pytest.approx((0.0, 0.0, 4.0, 5.0))
    assert left < right
    assert bottom < top


def test_optional_shapely_reference_bounds(tmp_path: Path):
    shape = pytest.importorskip("shapely.geometry")
    path = _write_geojson(tmp_path / "sample.geojson", features=_sample_features())

    result = gis.read_vector(path)
    bounds = [shape.shape(feature["geometry"]).bounds for feature in result["features"]]
    expected = (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )

    assert result["info"]["bounds"] == pytest.approx(expected)
