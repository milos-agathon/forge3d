from __future__ import annotations

from pathlib import Path

import forge3d as f3d
from forge3d._native import get_native_module


ROOT = Path(__file__).resolve().parents[1]


def _arabic_font_path() -> Path:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise AssertionError("Arabic shaping test requires a local TrueType font")


def test_complex_shaping_uses_approved_native_dependencies():
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")

    assert "rustybuzz" in cargo
    assert "unicode-bidi" in cargo


def test_native_text_shaper_returns_rustybuzz_glyph_metadata():
    shaped_glyphs = ["\ufe8e", "\ufe92", "\ufea3", "\ufeae", "\ufee3"]
    native = get_native_module()

    shaped = native.shape_text("مرحبا", str(_arabic_font_path()), list("مرحبا") + shaped_glyphs)

    assert shaped["engine"] == "rustybuzz"
    assert shaped["shaping"] == "rustybuzz"
    assert shaped["direction"] == "rtl"
    assert len(shaped["glyph_ids"]) == len(shaped_glyphs)
    assert len(shaped["clusters"]) == len(shaped_glyphs)
    assert len(shaped["advances"]) == len(shaped_glyphs)
    assert shaped["clusters"][0] > shaped["clusters"][-1]
    assert all(isinstance(glyph_id, int) and glyph_id > 0 for glyph_id in shaped["glyph_ids"])


def test_arabic_script_shaping_is_accepted_with_native_joined_glyphs():
    shaped_glyphs = ["\ufe8e", "\ufe92", "\ufea3", "\ufeae", "\ufee3"]
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
        glyph_atlas={"glyphs": list("مرحبا") + shaped_glyphs, "font_path": str(_arabic_font_path())},
    )

    assert not plan.rejected
    assert not [d for d in plan.diagnostics if d.code == "experimental_feature"]
    assert len(plan.accepted) == 1
    assert plan.accepted[0].glyphs == tuple(shaped_glyphs)
    assert plan.accepted[0].typography["shaping"] == "rustybuzz"
    assert plan.accepted[0].typography["engine"] == "rustybuzz"
    assert plan.accepted[0].typography["glyph_ids"]
    assert plan.accepted[0].typography["direction"] == "rtl"
    assert plan.accepted[0].typography["render_mapping"] == "arabic_presentation_forms"


def test_unsupported_complex_script_still_reports_experimental_diagnostic():
    plan = f3d.LabelPlan.compile(
        labels=[
            {
                "id": "devanagari-label",
                "text": "क्ष",
                "geometry": {"type": "Point", "coordinates": [10, 10]},
            }
        ],
        camera={},
        viewport=(100, 100),
        glyph_atlas={"glyphs": list("क्ष")},
    )

    assert not plan.accepted
    assert plan.rejected[0].reason == "unsupported_geometry_type"
    diagnostic = next(d for d in plan.diagnostics if d.code == "experimental_feature")
    assert diagnostic.object_id == "devanagari-label"
    assert diagnostic.details["feature"] == "complex-script shaping"


def test_label_support_docs_record_shaping_deferral():
    text = (ROOT / "docs/guides/label_support_matrix.md").read_text(encoding="utf-8")

    assert "Arabic joining" in text
    assert "Indic shaping" in text
    assert "`experimental`" in text
