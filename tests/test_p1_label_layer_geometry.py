from __future__ import annotations


def _feature(feature_id: str, geometry: dict, name: str) -> dict:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {"name": name},
        "geometry": geometry,
    }


def _diagnostic_codes(layer) -> list[str]:
    return [diagnostic.code for diagnostic in layer.diagnostics or ()]


def test_label_layer_from_features_ingests_point_line_and_polygon_candidates_deterministically():
    from forge3d.map_scene import LabelLayer

    features = [
        _feature(
            "road-2",
            {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
            "Main Road",
        ),
        _feature("city-1", {"type": "Point", "coordinates": [0.25, 0.5, 12.0]}, "Alpha"),
        _feature(
            "park-3",
            {
                "type": "Polygon",
                "coordinates": [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4], [0.1, 0.1]]],
            },
            "Central Park",
        ),
    ]

    first = LabelLayer.from_features(features, layer_id="labels.assets", crs="EPSG:4326")
    second = LabelLayer.from_features(reversed(features), layer_id="labels.assets", crs="EPSG:4326")

    assert _diagnostic_codes(first) == []
    assert [label["id"] for label in first.labels] == ["city-1", "park-3", "road-2"]
    assert first.labels == second.labels
    assert [label["geometry_type"] for label in first.labels] == ["Point", "Polygon", "LineString"]
    assert [label["placement_kind"] for label in first.labels] == ["point", "polygon", "line"]
    assert first.metadata["crs"] == "EPSG:4326"


def test_label_layer_from_features_reports_invalid_and_unsupported_geometry_with_feature_ids():
    from forge3d.map_scene import LabelLayer

    layer = LabelLayer.from_features(
        [
            _feature("good", {"type": "Point", "coordinates": [0.0, 0.0]}, "Valid"),
            _feature("empty", {}, "Empty"),
            _feature("bad-point", {"type": "Point", "coordinates": []}, "Bad Point"),
            _feature("multi", {"type": "MultiPoint", "coordinates": [[0.0, 0.0]]}, "Multi"),
        ],
        layer_id="labels.geometry",
    )

    assert [label["id"] for label in layer.labels] == ["good"]
    diagnostics = {diagnostic.object_id: diagnostic for diagnostic in layer.diagnostics or ()}
    assert diagnostics["empty"].code == "placeholder_fallback"
    assert diagnostics["empty"].details["feature"] == "label invalid geometry"
    assert diagnostics["bad-point"].code == "placeholder_fallback"
    assert diagnostics["multi"].code == "unsupported_feature"
    assert diagnostics["multi"].details["feature"] == "label geometry type MultiPoint"


def test_label_layer_from_geodataframe_like_object_preserves_geometry_and_crs():
    from forge3d.map_scene import LabelLayer

    class Geometry:
        __geo_interface__ = {"type": "Point", "coordinates": [0.5, 0.25]}

    class Row(dict):
        @property
        def geometry(self):
            return self["geometry"]

    class FakeGeoDataFrame:
        columns = ["name", "geometry"]
        crs = "EPSG:3857"

        def iterrows(self):
            yield "city-a", Row(name="Alpha", geometry=Geometry())

    layer = LabelLayer.from_geodataframe(FakeGeoDataFrame(), layer_id="labels.gdf")

    assert _diagnostic_codes(layer) == []
    assert layer.metadata["crs"] == "EPSG:3857"
    assert layer.labels == [
        {
            "id": "city-a",
            "source_id": "city-a",
            "text": "Alpha",
            "geometry": {"type": "Point", "coordinates": [0.5, 0.25]},
            "geometry_type": "Point",
            "placement_kind": "point",
            "properties": {"name": "Alpha"},
            "terrain_mode": "auto",
        }
    ]


def test_label_layer_from_style_layer_uses_text_field_and_validates_geometry():
    from forge3d.map_scene import LabelLayer

    style_layer = {"layout": {"text-field": "{label}"}}
    layer = LabelLayer.from_style_layer(
        [
            _feature("city-1", {"type": "Point", "coordinates": [0.0, 0.0]}, "ignored")
            | {"properties": {"label": "Styled City"}},
            _feature("unsupported", {"type": "GeometryCollection", "geometries": []}, "Unsupported"),
        ],
        style_layer,
        layer_id="labels.style",
    )

    assert [label["text"] for label in layer.labels] == ["Styled City"]
    assert _diagnostic_codes(layer) == ["unsupported_feature"]
    assert layer.diagnostics[0].object_id == "unsupported"
