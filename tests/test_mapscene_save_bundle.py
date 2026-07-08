import json

import forge3d as f3d
from forge3d.label_plan import PriorityClass


def _label_scene() -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "asset_status": "fixture"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png", path="map.png"),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=42),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                metadata={"width": 8, "height": 8, "source_id": "ortho-fixture", "asset_status": "fixture"},
            ),
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                        "properties": {"class": "primary"},
                    }
                ],
                metadata={"source_id": "roads-fixture"},
            ),
            f3d.PointCloudLayer(
                layer_id="points",
                path="fixtures/points.las",
                crs="EPSG:32610",
                point_count=12,
                metadata={"source_id": "points-fixture", "asset_status": "fixture"},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "city",
                        "kind": "point",
                        "text": "Alpha",
                        "geometry": {"type": "Point", "coordinates": (24.0, 24.0, 0.0)},
                        "priority_class": "cities",
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Alpha"))},
                priority_rules=[PriorityClass("cities", rank=10)],
                metadata={"source_id": "labels-fixture"},
            )
        ],
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_save_bundle_writes_deterministic_review_bundle_with_label_plan(tmp_path):
    first_path = tmp_path / "first.forge3d"
    second_path = tmp_path / "second.forge3d"

    first_report = _label_scene().save_bundle(str(first_path))
    second_report = _label_scene().save_bundle(str(second_path))

    assert first_report.status == "ok"
    assert first_report.to_dict() == second_report.to_dict()
    assert _read_json(first_path / "manifest.json") == _read_json(second_path / "manifest.json")
    assert _read_json(first_path / "scene" / "mapscene_recipe.json") == _read_json(
        second_path / "scene" / "mapscene_recipe.json"
    )

    state = _read_json(first_path / "scene" / "state.json")
    review = _read_json(first_path / "scene" / "mapscene_review.json")
    label_plan = _read_json(first_path / "scene" / "label_plans" / "labels.json")
    label_source = _read_json(first_path / "scene" / "label_sources" / "labels.json")
    terrain_source = _read_json(first_path / "scene" / "layer_sources" / "terrain.json")
    raster_source = _read_json(first_path / "scene" / "layer_sources" / "ortho.json")
    vector_source = _read_json(first_path / "scene" / "layer_sources" / "roads.json")
    point_source = _read_json(first_path / "scene" / "layer_sources" / "points.json")

    assert state["validation_report"] == first_report.to_dict()
    assert review["renderable"] is True
    assert review["render_status"] == "ready_for_render"
    assert review["last_render_backend"] is None
    assert review["output"]["format"] == "png"
    assert review["source_layer_ids"] == ["labels", "ortho", "points", "roads", "terrain"]
    assert label_plan["accepted"][0]["label_id"] == "city"
    assert label_source["source_reference"]["source_id"] == "labels-fixture"
    assert label_source["labels"][0]["id"] == "city"
    assert terrain_source["source_reference"]["source_id"] == "fixtures/dem.tif"
    assert raster_source["source_reference"]["source_id"] == "ortho-fixture"
    assert vector_source["source_reference"]["source_id"] == "roads-fixture"
    assert point_source["source_reference"]["source_id"] == "points-fixture"
    assert vector_source["features"][0]["id"] == "road-1"
    # SUTURA: point clouds are classified native-required in the serialized
    # recipe; rendering without native composition blocks with a diagnostic.
    assert point_source["payload"]["support_level"] == "native-required"
    compiled_plan = _read_json(first_path / "scene" / "compiled_plan.json")
    assert "labels" in compiled_plan["compiled_label_plans"]
    assert compiled_plan["depth_cull"]["camera_terrain_key"]


def test_save_bundle_preserves_blocking_diagnostics_without_render_success_claim(tmp_path):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "asset_status": "fixture"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source="fixtures/buildings.geojson",
                support_level="Pro-gated",
            )
        ],
    )
    bundle_path = tmp_path / "blocked.forge3d"

    report = scene.save_bundle(str(bundle_path))

    assert report.status == "error"
    assert bundle_path.exists()
    review = _read_json(bundle_path / "scene" / "mapscene_review.json")
    state = _read_json(bundle_path / "scene" / "state.json")
    assert review["renderable"] is False
    assert review["render_status"] == "blocked_by_diagnostics"
    assert state["validation_report"]["diagnostics"][0]["code"] == "pro_gated_path"


def test_save_bundle_preserves_missing_asset_diagnostics_without_renderability(tmp_path):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="missing/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="missing/ortho.tif",
                crs="EPSG:32610",
                metadata={"width": 8, "height": 8},
            )
        ],
    )
    bundle_path = tmp_path / "missing-assets.forge3d"

    report = scene.save_bundle(str(bundle_path))

    review = _read_json(bundle_path / "scene" / "mapscene_review.json")
    state = _read_json(bundle_path / "scene" / "state.json")
    codes = [diagnostic["code"] for diagnostic in state["validation_report"]["diagnostics"]]
    assert report.status == "error"
    assert review["renderable"] is False
    assert review["render_status"] == "blocked_by_diagnostics"
    assert codes == ["missing_external_asset", "missing_external_asset"]
