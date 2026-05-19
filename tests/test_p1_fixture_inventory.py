from __future__ import annotations

import json
from pathlib import Path


def test_p1_label_fixture_can_be_synthesized_deterministically(tmp_path):
    fixture = tmp_path / "p1_labels.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "city-1",
                "properties": {"name": "Alpha"},
                "geometry": {"type": "Point", "coordinates": [0.25, 0.25, 0.0]},
            },
            {
                "type": "Feature",
                "id": "road-1",
                "properties": {"name": "Main"},
                "geometry": {"type": "LineString", "coordinates": [[0.1, 0.1], [0.9, 0.9]]},
            },
            {
                "type": "Feature",
                "id": "park-1",
                "properties": {"name": "Park"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4], [0.1, 0.1]]],
                },
            },
        ],
    }
    fixture.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    first = json.loads(fixture.read_text(encoding="utf-8"))
    second = json.loads(fixture.read_text(encoding="utf-8"))
    assert first == second
    assert [feature["id"] for feature in first["features"]] == ["city-1", "road-1", "park-1"]


def test_p1_font_and_building_fixtures_are_available_or_synthesizable(tmp_path):
    from forge3d.map_scene import FontAtlas

    root = Path(__file__).resolve().parents[1]
    atlas_json = root / "assets" / "fonts" / "default_atlas.json"
    atlas_png = root / "assets" / "fonts" / "default_atlas.png"
    cityjson = root / "assets" / "geojson" / "sample_buildings.city.json"
    geojson = root / "assets" / "geojson" / "mount_fuji_buildings.geojson"

    atlas = FontAtlas.default_latin()
    assert atlas.covers("A")
    assert atlas.coverage == {"start": 32, "end": 127, "name": "Basic Latin"}

    available = [path for path in (atlas_json, atlas_png, cityjson, geojson) if path.exists()]
    for path in available:
        assert path.stat().st_size > 0

    synthetic_cityjson = tmp_path / "sample_buildings.city.json"
    synthetic_cityjson.write_text(
        json.dumps(
            {
                "type": "CityJSON",
                "version": "1.1",
                "CityObjects": {"building-1": {"type": "Building", "geometry": []}},
                "vertices": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert synthetic_cityjson.exists()


def test_p1_tileset_fixture_can_be_synthesized_deterministically(tmp_path):
    tileset = tmp_path / "tileset.json"
    payload = {
        "asset": {"version": "1.0"},
        "geometricError": 10,
        "root": {
            "boundingVolume": {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
            "geometricError": 0,
            "content": {"uri": "root.b3dm"},
        },
    }
    tileset.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    loaded = json.loads(tileset.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["root"]["content"]["uri"] == "root.b3dm"


def test_p1_bundle_fixture_manifest_can_be_synthesized_deterministically(tmp_path):
    bundle = tmp_path / "fixture.forge3d"
    scene_dir = bundle / "scene"
    scene_dir.mkdir(parents=True)

    recipe = {
        "camera": {"kind": "orbit_camera", "target": [0, 0, 0]},
        "kind": "mapscene_recipe",
        "layers": [
            {"kind": "label_layer", "layer_id": "labels.cities"},
            {"kind": "building_layer", "layer_id": "buildings.city"},
            {"kind": "tiles3d_layer", "layer_id": "tiles.local"},
        ],
        "lighting": {"kind": "lighting_preset", "name": "default"},
        "output": {"kind": "output_spec", "width": 640, "height": 360, "format": "png"},
        "terrain": {"kind": "terrain_source", "path": "dem.tif", "crs": "EPSG:4326"},
    }
    review = {
        "kind": "mapscene_review_bundle",
        "schema": "forge3d.mapscene.review.v1",
        "source_layer_ids": ["buildings.city", "labels.cities", "terrain", "tiles.local"],
        "supported_features": {"mapscene.save_bundle": "supported"},
        "unsupported_features": {"tiles3d.runtime_parity": "underdeveloped"},
    }
    manifest = {
        "created_by": "test",
        "files": [
            "scene/mapscene_recipe.json",
            "scene/mapscene_review.json",
            "scene/state.json",
        ],
    }

    for name, payload in {
        "mapscene_recipe.json": recipe,
        "mapscene_review.json": review,
        "state.json": {"validation_report": {"status": "ok", "diagnostics": []}},
    }.items():
        (scene_dir / name).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    first = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert first == second
    assert first["files"] == sorted(first["files"])
    assert json.loads((scene_dir / "mapscene_review.json").read_text(encoding="utf-8"))["source_layer_ids"] == [
        "buildings.city",
        "labels.cities",
        "terrain",
        "tiles.local",
    ]
