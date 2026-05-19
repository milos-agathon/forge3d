from forge3d.diagnostics import Diagnostic


def _labels():
    return [
        {
            "id": "city-a",
            "text": "Alpha",
            "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
            "priority": 5,
        },
        {
            "id": "city-b",
            "text": "Beta",
            "geometry": {"type": "Point", "coordinates": (30.0, 40.0, 6.0)},
            "priority": 3,
        },
    ]


def test_label_plan_public_contract_and_roundtrip():
    from forge3d import AcceptedLabel, KeepoutRegion, LabelCandidate, LabelPlan, PriorityClass

    keepout = KeepoutRegion(
        region_id="legend",
        kind="legend",
        bounds=(700.0, 500.0, 780.0, 560.0),
        priority=10,
    )
    priority = PriorityClass(name="cities", rank=20)

    plan = LabelPlan.compile(
        labels=_labels(),
        camera={"name": "fixed"},
        viewport={"width": 800, "height": 600},
        keepouts=[keepout],
        priority_rules=[priority],
        typography={"family": "default", "size": 14},
        glyph_atlas={"glyphs": list("ABaehlpt")},
        seed=1234,
    )

    assert plan.seed == 1234
    assert len(plan.accepted) == 2
    assert plan.rejected == []
    assert plan.diagnostics == []
    assert plan.bounds["screen"] == [10.0, 20.0, 30.0, 40.0]
    assert isinstance(plan.accepted[0], AcceptedLabel)
    assert isinstance(plan.accepted[0].candidate, LabelCandidate)
    assert plan.accepted[0].candidate.candidate_type == "center"

    payload = plan.to_dict()
    assert payload["payload_version"] == 1
    assert [label["label_id"] for label in payload["accepted"]] == ["city-a", "city-b"]
    assert LabelPlan.from_dict(payload).to_dict() == payload

    render_payload = plan.to_render_payload()
    export_payload = plan.to_export_payload()
    assert render_payload["kind"] == "label_plan_render_payload"
    assert export_payload["kind"] == "label_plan_export_payload"
    assert render_payload["accepted"] == payload["accepted"]
    assert export_payload["accepted"] == payload["accepted"]

    assert isinstance(Diagnostic.from_dict, object)


def test_label_plan_rejects_empty_sources_without_placeholder_success():
    from forge3d import LabelPlan

    plan = LabelPlan.compile(
        labels=[
            {
                "id": "empty",
                "text": "   ",
                "geometry": {"type": "Point", "coordinates": (0.0, 0.0, 0.0)},
            }
        ],
        camera={},
        viewport=(100, 100),
        seed=0,
    )

    assert plan.accepted == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0].reason == "empty_text"
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["label_rejection_summary"]
    assert plan.to_render_payload()["accepted"] == []
    assert plan.to_render_payload()["diagnostics"][0]["code"] == "label_rejection_summary"
