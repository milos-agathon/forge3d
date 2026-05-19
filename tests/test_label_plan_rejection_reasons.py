from forge3d.label_plan import REJECTION_REASONS


_GLYPHS_WITHOUT_BANG = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")


def _point(label_id, text, x, y, *, priority=0, **extra):
    record = {
        "id": label_id,
        "text": text,
        "geometry": {"type": "Point", "coordinates": (x, y, 0.0)},
        "priority": priority,
    }
    record.update(extra)
    return record


def _reason_fixture_labels():
    return [
        _point("empty-text", "   ", 1.0, 1.0),
        _point("missing-glyph", "Bang!", 2.0, 2.0),
        _point("outside-view", "Outside", 200.0, 5.0),
        {
            "id": "invalid-geometry",
            "text": "Invalid",
            "geometry": {"type": "Point", "coordinates": ("bad", 4.0, 0.0)},
        },
        {
            "id": "unsupported-geometry",
            "text": "Unsupported",
            "geometry": {"type": "LineString", "coordinates": [(5.0, 5.0), (6.0, 6.0)]},
        },
        _point("keepout-label", "Keepout", 20.0, 20.0),
        _point(
            "terrain-label",
            "Terrain",
            40.0,
            40.0,
            terrain_sample={"visible": False, "elevation": 10.0, "source": "fixture"},
        ),
        _point("collision-a", "One", 50.0, 50.0, priority=5),
        _point("collision-b", "Two", 50.0, 50.0, priority=5),
        _point("priority-high", "High", 60.0, 60.0, priority=20),
        _point("priority-low", "Low", 60.0, 60.0, priority=1),
    ]


def _compile_reason_fixture():
    from forge3d import KeepoutRegion, LabelPlan

    return LabelPlan.compile(
        labels=_reason_fixture_labels(),
        camera={"name": "fixed"},
        viewport={"width": 100, "height": 100},
        keepouts=[
            KeepoutRegion(
                region_id="legend",
                kind="legend",
                bounds=(10.0, 10.0, 30.0, 30.0),
            )
        ],
        glyph_atlas={"glyphs": _GLYPHS_WITHOUT_BANG},
        seed=11,
    )


def test_label_plan_retains_every_required_rejection_reason():
    from forge3d import LabelPlan

    plan = _compile_reason_fixture()
    reasons_by_label = {label.label_id: label.reason for label in plan.rejected}

    assert reasons_by_label == {
        "collision-b": "collision",
        "empty-text": "empty_text",
        "invalid-geometry": "invalid_geometry",
        "keepout-label": "keepout_region",
        "missing-glyph": "missing_glyph",
        "outside-view": "outside_view",
        "priority-low": "priority_lost",
        "terrain-label": "terrain_occluded",
        "unsupported-geometry": "unsupported_geometry_type",
    }
    assert set(reasons_by_label.values()) == set(REJECTION_REASONS)

    diagnostics_by_code = {diagnostic.code: diagnostic for diagnostic in plan.diagnostics}
    assert diagnostics_by_code["missing_glyphs"].object_id == "missing-glyph"
    assert diagnostics_by_code["missing_glyphs"].details["missing_glyphs"] == ["!"]
    assert diagnostics_by_code["label_rejection_summary"].details["rejection_counts"] == {
        reason: 1 for reason in REJECTION_REASONS
    }

    payload = plan.to_dict()
    assert LabelPlan.from_dict(payload).to_dict() == payload


def test_rejected_labels_keep_candidate_identity_and_structured_details():
    plan = _compile_reason_fixture()
    rejected_payload = {label["label_id"]: label for label in plan.to_dict()["rejected"]}

    assert rejected_payload["collision-b"]["candidate_id"] == "collision-b:center"
    assert rejected_payload["collision-b"]["details"]["collides_with"] == "collision-a"
    assert rejected_payload["priority-low"]["candidate_id"] == "priority-low:center"
    assert rejected_payload["priority-low"]["details"]["collides_with"] == "priority-high"
    assert rejected_payload["keepout-label"]["candidate_id"] == "keepout-label:center"
    assert rejected_payload["keepout-label"]["details"]["keepout_region_id"] == "legend"
    assert rejected_payload["terrain-label"]["candidate_id"] == "terrain-label:center"
    assert rejected_payload["terrain-label"]["details"]["terrain_sample"]["visible"] is False
    assert plan.to_render_payload()["rejected"] == plan.to_dict()["rejected"]
