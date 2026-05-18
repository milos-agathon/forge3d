import forge3d as f3d
from forge3d.label_plan import KeepoutRegion, PriorityClass


def _scene_with_labels() -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 16, "height": 16, "asset_status": "fixture"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=1000.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=256, height=256, format="png"),
        map_furniture=f3d.MapFurnitureLayer(
            title="Harbor overview",
            keepouts=[KeepoutRegion("title", "title", (0, 0, 64, 64), priority=100)],
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=1234),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "city",
                        "kind": "point",
                        "text": "Alpha",
                        "geometry": {"type": "Point", "coordinates": (120.0, 120.0, 0.0)},
                        "priority_class": "cities",
                    },
                    {
                        "id": "park",
                        "kind": "polygon",
                        "text": "Park",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    (150.0, 150.0),
                                    (190.0, 150.0),
                                    (190.0, 190.0),
                                    (150.0, 190.0),
                                    (150.0, 150.0),
                                ]
                            ],
                        },
                        "priority_class": "parks",
                    },
                    {
                        "id": "blocked-title",
                        "kind": "point",
                        "text": "Beta",
                        "geometry": {"type": "Point", "coordinates": (24.0, 24.0, 0.0)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("AlphaParkBeta"))},
                priority_rules=[
                    PriorityClass("cities", rank=10),
                    PriorityClass("parks", rank=5),
                ],
            )
        ],
    )


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def test_label_layer_compiles_deterministic_label_plan_and_rejection_summary():
    scene = _scene_with_labels()

    first_report = scene.validate()
    first_plan = scene.compiled_label_plans["labels"].to_dict()
    second_report = scene.validate()
    second_plan = scene.compiled_label_plans["labels"].to_dict()

    assert first_report.to_dict() == second_report.to_dict()
    assert first_plan == second_plan
    assert first_report.status == "warning"
    assert "label_rejection_summary" in _codes(first_report)

    accepted_ids = [label["label_id"] for label in first_plan["accepted"]]
    rejected = {label["label_id"]: label for label in first_plan["rejected"]}
    assert accepted_ids == ["city", "park"]
    assert rejected["blocked-title"]["reason"] == "keepout_region"

    summary = next(
        diagnostic for diagnostic in first_report.diagnostics if diagnostic.code == "label_rejection_summary"
    )
    assert summary.layer_id == "labels"
    assert summary.details["rejection_counts"] == {"keepout_region": 1}

    layer_summary = next(summary for summary in first_report.layer_summaries if summary.layer_id == "labels")
    assert layer_summary.support_level == "supported"
    assert layer_summary.details["compiled_label_plan"]["accepted_count"] == 2
    assert layer_summary.details["compiled_label_plan"]["rejected_count"] == 1
    assert layer_summary.details["compiled_label_plan"]["seed"] == 1234
