from __future__ import annotations


def _scene_with_layer(layer):
    from forge3d.map_scene import LightingPreset, MapScene, OrbitCamera, OutputSpec, TerrainSource

    return MapScene(
        terrain=TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "fixture-dem"}),
        camera=OrbitCamera(),
        lighting=LightingPreset(),
        layers=[layer],
        output=OutputSpec(width=64, height=64),
    )


def _summary(report, layer_id: str):
    return next(summary for summary in report.layer_summaries if summary.layer_id == layer_id)


def test_tiles3d_public_api_loads_supported_local_tileset_fixture(tmp_path):
    from forge3d.map_scene import Tiles3DLayer

    tileset = tmp_path / "tileset.json"
    tileset.write_text('{"asset":{"version":"1.1"},"geometricError":1}', encoding="utf-8")

    layer = Tiles3DLayer.from_tileset_json(tileset, layer_id="tiles.local")
    report = _scene_with_layer(layer).validate()

    assert layer.source["path"] == str(tileset)
    assert "python_public_3dtiles_incomplete" in [diagnostic.code for diagnostic in report.diagnostics]
    assert _summary(report, "tiles.local").details["source_kind"] == "tileset.json"


def test_tiles3d_validation_distinguishes_supported_and_unsupported_formats(tmp_path):
    from forge3d.map_scene import Tiles3DLayer

    glb = tmp_path / "model.glb"
    glb.write_bytes(b"glTF")

    layer = Tiles3DLayer(layer_id="tiles.glb", source={"path": str(glb), "source_format": "glb"})
    report = _scene_with_layer(layer).validate()
    diagnostics = {(diagnostic.code, diagnostic.details.get("format")) for diagnostic in report.diagnostics}

    assert ("unsupported_tile_format", "glb") in diagnostics
    assert report.unsupported_features["tiles3d.format"] == "unsupported"


def test_tiles3d_cache_statistics_are_exposed_in_layer_summary(tmp_path):
    from forge3d.map_scene import Tiles3DLayer

    tileset = tmp_path / "tileset.json"
    tileset.write_text('{"asset":{"version":"1.1"}}', encoding="utf-8")

    layer = Tiles3DLayer.from_tileset_json(
        tileset,
        layer_id="tiles.cache",
        cache_budget=4096,
        cache_stats={"resident_tiles": 2, "bytes_used": 1024},
    )
    report = _scene_with_layer(layer).validate()

    summary = _summary(report, "tiles.cache")
    assert summary.details["cache_budget"] == 4096
    assert summary.details["cache_stats"] == {"bytes_used": 1024, "resident_tiles": 2}


def test_tiles3d_lod_and_screen_space_error_config_is_public_metadata(tmp_path):
    from forge3d.map_scene import Tiles3DLayer

    tileset = tmp_path / "tileset.json"
    tileset.write_text('{"asset":{"version":"1.1"}}', encoding="utf-8")

    layer = Tiles3DLayer.from_tileset_json(
        tileset,
        layer_id="tiles.lod",
        lod={"maximum_screen_space_error": 8.0, "maximum_depth": 3},
    )
    report = _scene_with_layer(layer).validate()

    assert _summary(report, "tiles.lod").details["lod"] == {
        "maximum_depth": 3,
        "maximum_screen_space_error": 8.0,
    }


def test_tiles3d_unsupported_b3dm_or_glb_features_produce_typed_diagnostics(tmp_path):
    from forge3d.map_scene import Tiles3DLayer

    b3dm = tmp_path / "tile.b3dm"
    b3dm.write_bytes(b"b3dm")

    layer = Tiles3DLayer.from_b3dm(
        b3dm,
        layer_id="tiles.b3dm",
        metadata={"unsupported_features": ["draco_mesh_compression", "external_glb_extension"]},
    )
    report = _scene_with_layer(layer).validate()
    diagnostics = {(diagnostic.code, diagnostic.details.get("feature")) for diagnostic in report.diagnostics}

    assert ("unsupported_tile_feature", "draco_mesh_compression") in diagnostics
    assert ("unsupported_tile_feature", "external_glb_extension") in diagnostics
    assert report.unsupported_features["tiles3d.feature"] == "unsupported"


def test_tiles3d_docs_state_no_full_cesium_runtime_parity():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / "docs/guides/data_and_scene_workflows.md").read_text(encoding="utf-8")

    assert "not full Cesium runtime parity" in workflow
    assert "python_public_3dtiles_incomplete" in workflow
