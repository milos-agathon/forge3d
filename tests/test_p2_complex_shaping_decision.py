from __future__ import annotations

from pathlib import Path

import forge3d as f3d


ROOT = Path(__file__).resolve().parents[1]


def test_complex_script_shaping_is_non_blocking_experimental_diagnostic():
    plan = f3d.LabelPlan.compile(
        labels=[
            {
                "id": "arabic-label",
                "text": "مرحبا",
                "geometry": {"type": "Point", "coordinates": [10, 10]},
            }
        ],
        camera={},
        viewport=(100, 100),
        glyph_atlas={"glyphs": list("مرحبا")},
    )

    assert not plan.accepted
    assert plan.rejected[0].reason == "unsupported_geometry_type"
    diagnostic = next(d for d in plan.diagnostics if d.code == "experimental_feature")
    assert diagnostic.object_id == "arabic-label"
    assert diagnostic.details["feature"] == "complex-script shaping"


def test_label_support_docs_record_shaping_deferral():
    text = (ROOT / "docs/guides/label_support_matrix.md").read_text(encoding="utf-8")

    assert "HarfBuzz-compatible shaping" in text
    assert "non-MVP-blocking" in text
    assert "`experimental`" in text
