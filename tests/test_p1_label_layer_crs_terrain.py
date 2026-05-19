from __future__ import annotations


def _point_feature(feature_id: str = "city-1") -> dict:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {"name": "Alpha"},
        "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
    }


def test_label_layer_crs_transform_uses_forge3d_crs_utilities(monkeypatch):
    from forge3d.map_scene import LabelLayer
    import forge3d.crs as crs

    calls: list[tuple[list[list[float]], str, str]] = []

    def fake_transform(coords, from_crs, to_crs):
        calls.append((coords.tolist(), from_crs, to_crs))
        return [[1000.0, 2000.0]]

    monkeypatch.setattr(crs, "transform_coords", fake_transform)

    layer = LabelLayer.from_features(
        [_point_feature()],
        layer_id="labels.crs",
        crs="EPSG:4326",
        target_crs="EPSG:3857",
    )

    assert calls == [([[10.0, 20.0]], "EPSG:4326", "EPSG:3857")]
    assert layer.labels[0]["geometry"]["coordinates"] == [1000.0, 2000.0]
    assert layer.metadata["crs"] == "EPSG:3857"
    assert layer.metadata["source_crs"] == "EPSG:4326"
    assert [diagnostic.code for diagnostic in layer.diagnostics or ()] == []


def test_label_layer_required_terrain_sampling_applies_sampler_to_world_position():
    from forge3d.map_scene import LabelLayer

    def sampler(x, y, z=0.0):
        return {"elevation": x + y + z, "source": "unit-test", "visible": True}

    layer = LabelLayer.from_features(
        [_point_feature()],
        layer_id="labels.terrain",
        crs="EPSG:4326",
        terrain_sampling="required",
        terrain_sampler=sampler,
    )

    assert [diagnostic.code for diagnostic in layer.diagnostics or ()] == []
    assert layer.labels[0]["terrain_sample"] == {
        "elevation": 30.0,
        "source": "unit-test",
        "visible": True,
    }
    assert layer.labels[0]["geometry"]["coordinates"] == [10.0, 20.0, 30.0]


def test_label_layer_required_terrain_sampling_without_sampler_is_diagnostic_bearing():
    from forge3d.map_scene import LabelLayer

    layer = LabelLayer.from_features(
        [_point_feature()],
        layer_id="labels.terrain.missing",
        crs="EPSG:4326",
        terrain_sampling="required",
    )

    assert [diagnostic.code for diagnostic in layer.diagnostics or ()] == ["unavailable_terrain_sampler"]
    assert layer.diagnostics[0].object_id == "city-1"
    assert layer.labels[0]["terrain_mode"] == "required"
