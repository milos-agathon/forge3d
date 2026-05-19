from __future__ import annotations


def _feature(feature_id: str, props: dict, x: float = 0.0, y: float = 0.0) -> dict:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": props,
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def _scene_with_label_layer(layer):
    from forge3d.map_scene import LightingPreset, MapScene, OrbitCamera, OutputSpec, TerrainSource

    return MapScene(
        terrain=TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "fixture-dem"}),
        camera=OrbitCamera(),
        lighting=LightingPreset(),
        layers=[layer],
        output=OutputSpec(width=128, height=128),
    )


def test_label_layer_supports_prd_expression_subset():
    from forge3d.map_scene import LabelLayer

    features = [_feature("city-1", {"name": "Alpha", "kind": "city", "fallback": "Beta"})]

    assert LabelLayer.from_features(features, text="{name}").labels[0]["text"] == "Alpha"
    assert LabelLayer.from_features(features, text=["get", "name"]).labels[0]["text"] == "Alpha"
    assert (
        LabelLayer.from_features(features, text=["concat", ["get", "kind"], ":", ["get", "name"]]).labels[0]["text"]
        == "city:Alpha"
    )
    assert (
        LabelLayer.from_features(features, text=["coalesce", ["get", "missing"], ["get", "fallback"]]).labels[0]["text"]
        == "Beta"
    )
    assert LabelLayer.from_features(features, text=["upcase", ["get", "name"]]).labels[0]["text"] == "ALPHA"
    assert LabelLayer.from_features(features, text=["downcase", ["get", "name"]]).labels[0]["text"] == "alpha"


def test_label_layer_reports_missing_expression_field_before_render():
    from forge3d.map_scene import LabelLayer

    layer = LabelLayer.from_features(
        [_feature("city-1", {"name": "Alpha"})],
        text=["get", "missing"],
        layer_id="labels.missing-field",
    )

    assert layer.labels == []
    assert [diagnostic.code for diagnostic in layer.diagnostics or ()] == ["missing_label_field"]
    assert layer.diagnostics[0].details["field"] == "missing"
    assert layer.diagnostics[0].object_id == "city-1"


def test_label_layer_compile_labels_produces_deterministic_accepted_and_rejected_plan():
    from forge3d.map_scene import LabelLayer

    layer = LabelLayer.from_features(
        [
            _feature("accepted", {"name": "Alpha"}, x=10.0, y=10.0),
            _feature("outside", {"name": "Far"}, x=999.0, y=999.0),
        ],
        layer_id="labels.plan",
        metadata={"seed": 7},
    )

    first = layer.compile_labels(camera={}, viewport={"width": 128, "height": 128})
    second = layer.compile_labels(camera={}, viewport={"width": 128, "height": 128})

    assert first.to_dict() == second.to_dict()
    assert [label.label_id for label in first.accepted] == ["accepted"]
    assert [(label.label_id, label.reason) for label in first.rejected] == [("outside", "outside_view")]


def test_mapscene_validate_reports_line_label_incomplete_and_missing_glyphs_before_render():
    from forge3d.map_scene import LabelLayer

    layer = LabelLayer.from_features(
        [
            {
                "type": "Feature",
                "id": "road-1",
                "properties": {"name": "Road"},
                "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [10.0, 10.0]]},
            },
            _feature("city-1", {"name": "Alpha"}, x=10.0, y=10.0),
        ],
        layer_id="labels.diagnostics",
        glyph_atlas={"glyphs": list("Road")},
    )

    report = _scene_with_label_layer(layer).validate()
    codes_by_object = {(diagnostic.object_id, diagnostic.code) for diagnostic in report.diagnostics}

    assert ("road-1", "experimental_feature") in codes_by_object
    assert ("city-1", "missing_glyphs") in codes_by_object
