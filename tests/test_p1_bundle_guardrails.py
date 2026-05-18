from __future__ import annotations

import json


def test_p1_bundle_persists_asset_layers_label_sources_and_diagnostics(tmp_path):
    import forge3d as f3d

    labels = f3d.LabelLayer.from_features(
        [
            {
                "type": "Feature",
                "id": "city-1",
                "properties": {"name": "Alpha"},
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            }
        ],
        layer_id="labels.cities",
        crs="EPSG:4326",
    )
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "synthetic-dem"}),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        layers=[
            labels,
            f3d.MapSceneBuildingLayer.from_geojson(
                "buildings.geojson",
                layer_id="buildings.city",
                support_level="underdeveloped",
                metadata={"source_id": "synthetic-buildings"},
            ),
            f3d.Tiles3DLayer.from_tileset_json(
                "tileset.json",
                layer_id="tiles.local",
                metadata={"source_id": "synthetic-tiles"},
            ),
        ],
        output=f3d.OutputSpec(width=64, height=64),
    )

    first = tmp_path / "first.forge3d"
    second = tmp_path / "second.forge3d"
    report = scene.save_bundle(first)
    scene.save_bundle(second)

    assert "mapscene.save_bundle" in report.supported_features
    for rel in (
        "scene/mapscene_recipe.json",
        "scene/mapscene_review.json",
        "scene/state.json",
        "scene/layer_sources/labels.cities.json",
        "scene/layer_sources/buildings.city.json",
        "scene/layer_sources/tiles.local.json",
        "scene/label_sources/labels.cities.json",
    ):
        assert (first / rel).exists(), rel

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["checksums"] == second_manifest["checksums"]
    assert list(first_manifest["checksums"]) == sorted(first_manifest["checksums"])

    review = json.loads((first / "scene" / "mapscene_review.json").read_text(encoding="utf-8"))
    assert review["source_layer_ids"] == ["buildings.city", "labels.cities", "terrain", "tiles.local"]
    assert review["supported_features"]["mapscene.save_bundle"] == "supported"

    loaded = f3d.MapScene.load_bundle(first)
    assert loaded.last_validation_report is not None
    assert loaded.recipe.layers[0].layer_id == "labels.cities"


def test_p1_bundle_missing_external_assets_remain_diagnostic_bearing(tmp_path):
    import forge3d as f3d

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "synthetic-dem"}),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        layers=[
            f3d.RasterOverlay(
                layer_id="raster.missing",
                path=str(tmp_path / "missing.tif"),
                crs="EPSG:4326",
            )
        ],
        output=f3d.OutputSpec(width=64, height=64),
    )

    report = scene.save_bundle(tmp_path / "missing-assets.forge3d")
    codes = [diagnostic.code for diagnostic in report.diagnostics]
    assert "missing_external_asset" in codes
    assert report.unsupported_features["raster.asset"] == "unsupported"
