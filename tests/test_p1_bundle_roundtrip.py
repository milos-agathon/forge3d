from __future__ import annotations

import json


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _renderable_fixture_scene():
    import forge3d as f3d

    labels = f3d.LabelLayer.from_features(
        [
            {
                "type": "Feature",
                "id": "city-1",
                "properties": {"name": "Alpha"},
                "geometry": {"type": "Point", "coordinates": [24.0, 24.0, 0.0]},
            }
        ],
        layer_id="labels.cities",
        crs="EPSG:32610",
        glyph_atlas={"glyphs": list("Alpha")},
        metadata={"source_id": "labels-source", "seed": 3},
    )
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={
                "width": 8,
                "height": 8,
                "asset_status": "fixture",
                "source_id": "terrain-source",
                "source_crs": "EPSG:4326",
                "source_geotransform": [0.0, 0.01, 0.0, 1.0, 0.0, -0.01],
                "geotransform": [500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0],
                "nodata": -9999.0,
                "alignment_transform_applied": True,
            },
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.25),
        output=f3d.OutputSpec(width=64, height=64, format="png", path="review.png"),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                metadata={"width": 8, "height": 8, "asset_status": "fixture", "source_id": "ortho-source"},
            ),
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
                        "properties": {"class": "primary"},
                    }
                ],
                metadata={"source_id": "roads-source"},
            ),
            labels,
        ],
    )


def test_bundle_persists_required_p1_review_fields_and_label_payloads(tmp_path):
    scene = _renderable_fixture_scene()
    bundle = tmp_path / "review.forge3d"

    report = scene.save_bundle(bundle)

    review = _read_json(bundle / "scene" / "mapscene_review.json")
    state = _read_json(bundle / "scene" / "state.json")
    recipe = _read_json(bundle / "scene" / "mapscene_recipe.json")
    label_plan = _read_json(bundle / "scene" / "label_plans" / "labels.cities.json")
    label_source = _read_json(bundle / "scene" / "label_sources" / "labels.cities.json")
    terrain_source = _read_json(bundle / "scene" / "layer_sources" / "terrain.json")

    assert report.status == "ok"
    assert recipe["terrain"]["metadata"]["source_id"] == "terrain-source"
    assert recipe["camera"]["kind"] == "orbit_camera"
    assert recipe["lighting"]["name"] == "daylight"
    assert recipe["output"]["format"] == "png"
    assert review["supported_export_settings"] == {
        "bundle_schema": "forge3d.mapscene.review.v1",
        "label_plan_persistence": True,
        "output_formats": ["exr", "png"],
    }
    assert review["compiled_label_plan_ids"] == ["labels.cities"]
    assert review["source_layer_ids"] == ["labels.cities", "ortho", "roads", "terrain"]
    assert state["validation_report"] == report.to_dict()
    assert label_plan["accepted"][0]["label_id"] == "city-1"
    assert label_source["source_reference"]["source_id"] == "labels-source"
    assert terrain_source["source_reference"]["source_id"] == "terrain-source"
    assert recipe["terrain"]["metadata"]["source_crs"] == "EPSG:4326"
    assert recipe["terrain"]["metadata"]["source_geotransform"] == [0.0, 0.01, 0.0, 1.0, 0.0, -0.01]
    assert recipe["terrain"]["metadata"]["geotransform"] == [500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0]
    assert recipe["terrain"]["metadata"]["nodata"] == -9999.0
    assert terrain_source["metadata"]["source_crs"] == "EPSG:4326"
    assert terrain_source["metadata"]["source_geotransform"] == [0.0, 0.01, 0.0, 1.0, 0.0, -0.01]
    assert terrain_source["metadata"]["geotransform"] == [500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0]
    assert terrain_source["metadata"]["nodata"] == -9999.0


def test_bundle_load_reconstructs_scene_and_blocks_without_native_terrain(tmp_path):
    import pytest

    import forge3d as f3d

    bundle = tmp_path / "roundtrip.forge3d"
    output = tmp_path / "loaded.png"
    _renderable_fixture_scene().save_bundle(bundle)

    loaded = f3d.MapScene.load_bundle(bundle)

    # The compiled label plan is rehydrated verbatim from the bundle.
    assert loaded.compiled_plan is not None
    assert loaded.compiled_label_plans["labels.cities"].accepted

    # The fixture DEM does not exist on disk, so rendering blocks with a
    # structured diagnostic instead of writing a CPU placeholder.
    with pytest.raises(f3d.MapSceneNativeUnavailable) as excinfo:
        loaded.render(str(output))
    assert not output.exists()
    assert excinfo.value.diagnostic["status"] == "diagnostic_block"
    assert excinfo.value.diagnostic["layer"] == "terrain"


def test_bundle_load_reports_missing_external_assets_with_structured_diagnostics(tmp_path):
    import forge3d as f3d

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "synthetic-dem"}),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=32, height=32),
        layers=[f3d.RasterOverlay(layer_id="ortho.missing", path=str(tmp_path / "missing.tif"), crs="EPSG:4326")],
    )
    bundle = tmp_path / "missing.forge3d"
    scene.save_bundle(bundle)

    loaded = f3d.MapScene.load_bundle(bundle)
    report = loaded.validate()
    diagnostics = [(diagnostic.code, diagnostic.layer_id, diagnostic.details.get("path")) for diagnostic in report.diagnostics]

    assert ("missing_external_asset", "ortho.missing", str(tmp_path / "missing.tif")) in diagnostics
    assert report.unsupported_features["raster.asset"] == "unsupported"
