"""G-002 Later OSM query and parser tests."""

from __future__ import annotations

import json

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS OSM tests require the compiled _forge3d extension",
)


def _codes(result) -> set[str]:
    return {warning["code"] for warning in result.get("warnings", [])}


def _payload():
    return {
        "elements": [
            {"type": "node", "id": 1, "lat": 0.0, "lon": 0.0, "tags": {"amenity": "cafe"}},
            {"type": "node", "id": 2, "lat": 0.0, "lon": 1.0},
            {"type": "node", "id": 3, "lat": 1.0, "lon": 1.0},
            {"type": "node", "id": 4, "lat": 1.0, "lon": 0.0},
            {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}},
            {"type": "way", "id": 11, "nodes": [1, 2, 3, 4, 1], "tags": {"building": "yes", "building:levels": "2"}},
            {"type": "way", "id": 12, "nodes": [1, 99], "tags": {"highway": "service"}},
            {"type": "relation", "id": 20, "tags": {"type": "multipolygon"}},
        ]
    }


def test_parse_osm_features_nodes_ways_skips_and_metadata():
    result = gis.parse_osm_features(_payload())

    geometries = [feature["geometry"]["type"] for feature in result["features"]]
    assert geometries == ["Point", "LineString", "Polygon"]
    assert result["type"] == "FeatureCollection"
    assert result["crs"] == {"name": "EPSG", "code": "4326"}
    assert result["bounds"] == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert result["skipped"]["incomplete_way"] == 1
    assert result["skipped"]["unsupported_relation"] == 1
    assert {"incomplete_way", "unsupported_relation"} <= _codes(result)
    assert result["features"][2]["properties"]["building:levels"] == "2"


def test_parse_osm_features_tag_filtering():
    result = gis.parse_osm_features(_payload(), tags={"building": True})

    assert len(result["features"]) == 1
    assert result["features"][0]["geometry"]["type"] == "Polygon"
    assert result["features"][0]["properties"]["building"] == "yes"


def test_parse_osm_features_empty_result_warns():
    result = gis.parse_osm_features({"elements": []})

    assert result["type"] == "FeatureCollection"
    assert result["features"] == []
    assert "empty_feature_set" in _codes(result)


@pytest.mark.parametrize("payload", ["not json", {"not_elements": []}])
def test_parse_osm_features_rejects_malformed_payload(payload):
    with pytest.raises(ValueError, match="malformed_payload"):
        gis.parse_osm_features(payload)


def test_query_osm_features_uses_explicit_cache_payload():
    payload = _payload()

    result = gis.query_osm_features(
        (0.0, 0.0, 1.0, 1.0),
        {"building": True},
        cache={"osm_json": payload},
    )

    assert result["osm_json"] == payload
    assert "[out:json]" in result["query"]
    assert "building" in result["query"]
    assert result["bounds"] == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert result["remote"]["status"] == "mocked"


def test_query_osm_features_without_endpoint_does_not_use_public_network():
    with pytest.raises(RuntimeError, match="backend_unavailable|cache_miss"):
        gis.query_osm_features((0.0, 0.0, 1.0, 1.0), {"highway": True})


def test_query_osm_features_cache_payload_accepts_json_string():
    payload = json.dumps(_payload())

    result = gis.query_osm_features(
        (0.0, 0.0, 1.0, 1.0),
        {"amenity": "cafe"},
        cache={"osm_json": payload},
    )

    assert result["osm_json"]["elements"][0]["tags"]["amenity"] == "cafe"
