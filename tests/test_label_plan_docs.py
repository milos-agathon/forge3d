from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_label_plan_guide_documents_core_contract_and_boundaries():
    guide = _read("docs/guides/label_plan_guide.md")

    required_terms = [
        "LabelPlan.compile",
        "AcceptedLabel",
        "RejectedLabel",
        "LabelCandidate",
        "KeepoutRegion",
        "PriorityClass",
        "to_render_payload",
        "to_export_payload",
        "placeholder_fallback",
        "missing_glyphs",
        "label_rejection_summary",
        "center",
        "above",
        "below",
        "left",
        "right",
        "radial",
        "centroid",
        "visual_center",
        "terrain_occluded",
        "keepout_region",
        "priority_lost",
    ]
    for term in required_terms:
        assert term in guide

    for reason in [
        "collision",
        "outside_view",
        "missing_glyph",
        "priority_lost",
        "keepout_region",
        "terrain_occluded",
        "invalid_geometry",
        "unsupported_geometry_type",
        "empty_text",
    ]:
        assert f"`{reason}`" in guide

    assert "production-ready curved" not in guide.lower()
    assert "production-ready non-latin" not in guide.lower()


def test_label_plan_support_matrix_and_api_reference_are_current():
    support_matrix = _read("docs/guides/label_support_matrix.md")
    api_reference = _read("docs/api/api_reference.rst")
    index = _read("docs/index.rst")

    assert "| Deterministic LabelPlan | `supported` |" in support_matrix
    assert "point and polygon candidates" in support_matrix
    assert "unsupported backends return `placeholder_fallback`" in support_matrix
    assert ".. automodule:: forge3d.label_plan" in api_reference
    assert "Deterministic LabelPlan" in api_reference
    assert "guides/label_plan_guide" in index
