"""G-002c C2 vector CRS reprojection contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS vector CRS tests require the compiled _forge3d extension",
)


WEB_MERCATOR_X_1 = 111_319.49079327357
WEB_MERCATOR_Y_1 = 111_325.1428663851


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _feature(kind: str, coordinates, **extra):
    feature = {
        "type": "Feature",
        "id": extra.pop("feature_id", f"{kind.lower()}-1"),
        "properties": {"kind": kind, "name": extra.pop("name", kind.lower())},
        "geometry": {"type": kind, "coordinates": coordinates},
    }
    feature.update(extra)
    return feature


def _collection(features: list[dict], *, crs: bool = True, **extra) -> dict:
    payload = {"type": "FeatureCollection", "features": features}
    if crs:
        payload["crs"] = {"type": "name", "properties": {"name": "EPSG:4326"}}
    payload.update(extra)
    return payload


def _write_geojson(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _point_coordinates(result: dict, index: int = 0):
    return result["features"][index]["geometry"]["coordinates"]


def test_reproject_vector_geojson_path_epsg4326_to_3857(tmp_path: Path):
    payload = _collection(
        [
            _feature(
                "Point",
                [1.0, 1.0],
                feature_id="point-a",
                source_member="kept",
                bbox=[0.0, 0.0, 0.0, 0.0],
            ),
            _feature("LineString", [[0.0, 0.0], [1.0, 1.0]]),
            _feature(
                "Polygon",
                [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
            ),
        ],
        bbox=[-999.0, -999.0, 999.0, 999.0],
    )
    path = _write_geojson(tmp_path / "sample.geojson", payload)

    result = gis.reproject_vector(path, "EPSG:3857")

    assert result["type"] == "FeatureCollection"
    assert result["feature_count"] == 3
    assert result["info"]["feature_count"] == 3
    assert result["vector_info"].feature_count == 3
    assert result["src_crs"]["authority"] == {"name": "EPSG", "code": "4326"}
    assert result["dst_crs"]["authority"] == {"name": "EPSG", "code": "3857"}
    assert _point_coordinates(result) == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )
    assert result["bounds"] == pytest.approx(
        (0.0, 0.0, WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1), abs=1e-6
    )
    assert result["info"]["bounds"] == pytest.approx(result["bounds"])
    assert result["features"][0]["id"] == "point-a"
    assert result["features"][0]["properties"] == {"kind": "Point", "name": "point"}
    assert result["features"][0]["source_member"] == "kept"
    assert result["features"][0]["bbox"] == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1, WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )


def test_reproject_vector_dict_feature_collection_with_explicit_src_crs():
    payload = _collection([_feature("Point", [1.0, 1.0])], crs=False)

    result = gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")

    assert _point_coordinates(result) == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )
    assert result["src_crs"]["authority"] == {"name": "EPSG", "code": "4326"}


def test_reproject_vector_empty_feature_collection_with_crs_warns_consistently():
    payload = _collection([], crs=True, bbox=[-10.0, -5.0, 10.0, 5.0])

    result = gis.reproject_vector(payload, "EPSG:3857")

    assert result["features"] == []
    assert result["feature_count"] == 0
    assert result["info"]["feature_count"] == 0
    assert result["bounds"] is None
    assert result["info"]["bounds"] is None
    assert result["info"]["is_georeferenced"] is False
    assert result["src_crs"]["authority"] == {"name": "EPSG", "code": "4326"}
    assert result["dst_crs"]["authority"] == {"name": "EPSG", "code": "3857"}
    assert "empty_feature_set" in _codes(result["warnings"])
    assert "bbox" not in result


def test_reproject_vector_accepts_read_vector_result_dict(tmp_path: Path):
    path = _write_geojson(
        tmp_path / "sample.geojson",
        _collection([_feature("Point", [1.0, 1.0])]),
    )
    read_result = gis.read_vector(path)

    result = gis.reproject_vector(read_result, "EPSG:3857")

    assert _point_coordinates(result) == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )
    assert result["src_crs"]["authority"] == {"name": "EPSG", "code": "4326"}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "type": "Feature",
            "properties": {"kind": "raw"},
            "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
        },
        {"type": "Point", "coordinates": [1.0, 1.0]},
    ],
)
def test_reproject_vector_accepts_raw_feature_and_geometry_with_explicit_src_crs(payload: dict):
    result = gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")

    assert _point_coordinates(result) == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )


def test_reproject_vector_missing_source_crs_errors():
    payload = _collection([_feature("Point", [1.0, 1.0])], crs=False)

    with pytest.raises(ValueError, match="MissingCrs|missing_crs"):
        gis.reproject_vector(payload, "EPSG:3857")


def test_reproject_vector_invalid_destination_crs_errors():
    payload = _collection([_feature("Point", [1.0, 1.0])])

    with pytest.raises(ValueError, match="InvalidCrs|invalid_crs"):
        gis.reproject_vector(payload, "not-a-crs")


def test_reproject_vector_src_crs_conflict_with_metadata_errors():
    payload = _collection([_feature("Point", [1.0, 1.0])])

    with pytest.raises(ValueError, match="CrsMismatch"):
        gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:3857")


def test_reproject_vector_same_crs_copies_features_and_updates_metadata():
    payload = _collection(
        [_feature("Point", [1.0, 1.0], bbox=[0.0, 0.0, 0.0, 0.0])]
    )

    result = gis.reproject_vector(payload, "EPSG:4326")

    assert _point_coordinates(result) == pytest.approx([1.0, 1.0])
    assert result["features"][0]["bbox"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert result["dst_crs"]["authority"] == {"name": "EPSG", "code": "4326"}
    assert result["bounds"] == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_reproject_vector_recomputes_or_removes_geojson_bbox_members():
    payload = _collection(
        [
            _feature(
                "Point",
                [1.0, 1.0],
                bbox=[-999.0, -999.0, 999.0, 999.0],
            ),
            {
                "type": "Feature",
                "properties": {"name": "geometry-bbox"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [1.0, 1.0],
                    "bbox": [-999.0, -999.0, 999.0, 999.0],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "empty-with-bbox"},
                "bbox": [-999.0, -999.0, 999.0, 999.0],
                "geometry": {"type": "LineString", "coordinates": []},
            },
        ],
        crs=False,
        bbox=[-999.0, -999.0, 999.0, 999.0],
    )

    result = gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")

    expected_point_bbox = [
        WEB_MERCATOR_X_1,
        WEB_MERCATOR_Y_1,
        WEB_MERCATOR_X_1,
        WEB_MERCATOR_Y_1,
    ]
    assert result["features"][0]["bbox"] == pytest.approx(expected_point_bbox)
    assert result["features"][1]["geometry"]["bbox"] == pytest.approx(expected_point_bbox)
    assert "bbox" not in result["features"][2]
    assert "bbox" not in result
    assert result["bounds"] == pytest.approx(
        (WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1, WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1)
    )


def test_reproject_vector_preserves_z_and_additional_ordinates():
    payload = _collection([_feature("Point", [1.0, 1.0, 42.0, 7.0])], crs=False)

    result = gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")

    assert _point_coordinates(result) == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1, 42.0, 7.0]
    )


def test_reproject_vector_geometry_collection_recurses():
    payload = _collection(
        [
            {
                "type": "Feature",
                "properties": {"name": "collection"},
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "Point", "coordinates": [1.0, 1.0]},
                        {
                            "type": "LineString",
                            "coordinates": [[0.0, 0.0], [1.0, 1.0]],
                        },
                    ],
                },
            }
        ],
        crs=False,
    )

    result = gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")

    geometries = result["features"][0]["geometry"]["geometries"]
    assert geometries[0]["coordinates"] == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )
    assert geometries[1]["coordinates"][1] == pytest.approx(
        [WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1]
    )


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Point", "coordinates": ["x", 1.0]},
        {"type": "Point", "coordinates": [1.0]},
        {"type": "LineString", "coordinates": [1.0, 2.0]},
        {"type": "GeometryCollection"},
    ],
)
def test_reproject_vector_rejects_malformed_coordinates(geometry: dict):
    payload = {"type": "Feature", "properties": {}, "geometry": geometry}

    with pytest.raises(ValueError, match="InvalidGeometry|invalid_geometry"):
        gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")


def test_reproject_vector_preserves_null_and_empty_geometries_with_warning():
    payload = _collection(
        [
            {"type": "Feature", "properties": {"name": "null"}, "geometry": None},
            _feature("LineString", []),
        ],
        crs=False,
    )

    result = gis.reproject_vector(payload, "EPSG:3857", src_crs="EPSG:4326")

    assert result["feature_count"] == 2
    assert result["features"][0]["geometry"] is None
    assert result["features"][1]["geometry"]["coordinates"] == []
    assert result["bounds"] is None
    assert "empty_geometry" in _codes(result["warnings"])


def test_reproject_vector_bounds_order_is_left_bottom_right_top():
    payload = _collection(
        [
            _feature("Point", [-1.0, -1.0]),
            _feature("Point", [1.0, 1.0]),
        ],
        crs=False,
    )

    left, bottom, right, top = gis.reproject_vector(
        payload, "EPSG:3857", src_crs="EPSG:4326"
    )["bounds"]

    assert left < right
    assert bottom < top
    assert (left, bottom, right, top) == pytest.approx(
        (-WEB_MERCATOR_X_1, -WEB_MERCATOR_Y_1, WEB_MERCATOR_X_1, WEB_MERCATOR_Y_1)
    )


def test_reproject_vector_rejects_vector_info_without_feature_payload(tmp_path: Path):
    path = _write_geojson(
        tmp_path / "sample.geojson",
        _collection([_feature("Point", [1.0, 1.0])]),
    )
    result = gis.read_vector(path)

    with pytest.raises(ValueError, match="feature payload is required"):
        gis.reproject_vector(result["vector_info"], "EPSG:3857")


def test_reproject_vector_valid_but_unsupported_crs_pair_fails_backend_unavailable():
    payload = _collection([_feature("Point", [1.0, 1.0])])

    with pytest.raises(RuntimeError, match="BackendUnavailable"):
        gis.reproject_vector(payload, "EPSG:32631")
