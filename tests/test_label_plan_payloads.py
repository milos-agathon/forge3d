def _compile_payload_plan():
    from forge3d import LabelPlan

    return LabelPlan.compile(
        labels=[
            {
                "id": "accepted",
                "text": "Accepted",
                "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 2.0)},
                "typography": {"family": "atlas", "size": 11},
            },
            {
                "id": "missing",
                "text": "Missing!",
                "geometry": {"type": "Point", "coordinates": (30.0, 20.0, 2.0)},
            },
        ],
        camera={"name": "fixed"},
        viewport=(100, 100),
        glyph_atlas={"glyphs": set("AcceptedMisng")},
        typography={"family": "atlas", "size": 11},
        seed=12,
    )


def test_render_and_export_payloads_preserve_real_plan_data():
    plan = _compile_payload_plan()
    payload = plan.to_dict()
    render_payload = plan.to_render_payload()
    export_payload = plan.to_export_payload()

    assert render_payload["kind"] == "label_plan_render_payload"
    assert export_payload["kind"] == "label_plan_export_payload"
    for candidate_payload in (render_payload, export_payload):
        assert candidate_payload["accepted"] == payload["accepted"]
        assert candidate_payload["rejected"] == payload["rejected"]
        assert candidate_payload["diagnostics"] == payload["diagnostics"]
        assert candidate_payload["bounds"] == payload["bounds"]
        assert candidate_payload["seed"] == 12
        assert candidate_payload["accepted"][0]["glyphs"] == list("Accepted")
        typography = candidate_payload["accepted"][0]["typography"]
        assert typography["family"] == "atlas"
        assert typography["size"] == 11
        assert typography["shaping"] == "littera"
        assert typography["render_mapping"] == "positioned_glyphs_by_id"
        assert typography["glyph_ids"]
        assert typography["shaped_runs"][0]["glyphs"]
        assert candidate_payload["accepted"][0]["candidates"]
        assert candidate_payload["rejected"][0]["reason"] == "missing_glyph"


def test_unsupported_payload_backend_returns_typed_diagnostic_not_empty_success():
    plan = _compile_payload_plan()

    render_payload = plan.to_render_payload(backend="native-gpu")
    export_payload = plan.to_export_payload(backend="pdf")

    for candidate_payload in (render_payload, export_payload):
        assert candidate_payload["supported"] is False
        assert candidate_payload["accepted"]
        assert candidate_payload["rejected"]
        assert any(diagnostic["code"] == "placeholder_fallback" for diagnostic in candidate_payload["diagnostics"])
        assert candidate_payload["backend"] in {"native-gpu", "pdf"}
