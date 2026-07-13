from __future__ import annotations

from pathlib import Path

import forge3d as f3d
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _arabic_font_path() -> Path:
    candidates = [
        # Windows
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        # Linux (GitHub-hosted runners ship DejaVu/Noto/Liberation)
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        # macOS
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Tahoma.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise AssertionError("Arabic shaping test requires a local TrueType font")


def test_complex_shaping_has_no_external_shaping_dependencies():
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")

    assert "rustybuzz" not in cargo
    assert "unicode-bidi" not in cargo


def test_native_text_shaper_returns_littera_glyph_metadata():
    shaped = f3d.text.shape("مرحبا", [str(ROOT / "assets/fonts/NotoSansArabic-subset.ttf")], 16.0)
    run = shaped.to_dict()["runs"][0]
    assert run["direction"] == "rtl"
    assert run["glyphs"][0]["cluster"] < run["glyphs"][-1]["cluster"]
    assert all(glyph["glyph_id"] > 0 for glyph in run["glyphs"])


def test_arabic_script_shaping_is_accepted_with_native_joined_glyphs():
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
        glyph_atlas={
            "glyphs": list("مرحبا"),
            "font_path": str(ROOT / "assets/fonts/NotoSansArabic-subset.ttf"),
        },
    )

    assert not plan.rejected
    assert not [d for d in plan.diagnostics if d.code == "experimental_feature"]
    assert len(plan.accepted) == 1
    assert plan.accepted[0].glyphs == tuple("مرحبا")
    assert plan.accepted[0].typography["shaping"] == "littera"
    assert plan.accepted[0].typography["engine"] == "littera"
    assert plan.accepted[0].typography["glyph_ids"]
    assert plan.accepted[0].typography["direction"] == "rtl"
    assert plan.accepted[0].typography["render_mapping"] == "shaped_clusters"
    assert plan.accepted[0].typography["compositor"] == "deferred_task_8"
    assert plan.accepted[0].typography["shaped_runs"][0]["glyphs"]
    assert plan.accepted[0].typography["font_indices"]


def test_devanagari_is_supported_by_littera():
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
        glyph_atlas={
            "glyphs": list("क्ष"),
            "font_path": str(ROOT / "assets/fonts/NotoSansDevanagari-subset.ttf"),
        },
    )

    assert not plan.rejected
    assert plan.accepted[0].typography["shaping"] == "littera"


@pytest.mark.parametrize(
    ("text", "font"),
    [
        ("office", "NotoSans-subset.ttf"),
        ("地图", "NotoSansSC-subset.ttf"),
    ],
)
def test_valid_font_routes_every_script_through_littera(text, font):
    plan = f3d.LabelPlan.compile(
        labels=[{
            "id": "label",
            "text": text,
            "geometry": {"type": "Point", "coordinates": [10, 10]},
        }],
        camera={},
        viewport=(100, 100),
        glyph_atlas={
            "glyphs": list(text),
            "font_path": str(ROOT / "assets/fonts" / font),
        },
    )
    assert not plan.rejected
    assert plan.accepted[0].typography["shaping"] == "littera"
    assert plan.accepted[0].typography["glyph_ids"]


def test_complex_missing_font_keeps_structured_reason():
    plan = f3d.LabelPlan.compile(
        labels=[{
            "id": "label",
            "text": "مرحبا",
            "geometry": {"type": "Point", "coordinates": [10, 10]},
        }],
        camera={},
        viewport=(100, 100),
        glyph_atlas={"glyphs": list("مرحبا")},
    )
    assert plan.rejected[0].reason == "font_chain_required"
    assert plan.rejected[0].details["diagnostics"][0]["reason"] == "font_chain_required"


def test_label_plan_keeps_native_shaping_error_reason(tmp_path):
    bad_font = tmp_path / "bad.ttf"
    bad_font.write_bytes(b"not a font")
    plan = f3d.LabelPlan.compile(
        labels=[{
            "id": "label",
            "text": "مرحبا",
            "geometry": {"type": "Point", "coordinates": [10, 10]},
        }],
        camera={},
        viewport=(100, 100),
        glyph_atlas={"glyphs": list("مرحبا"), "font_path": str(bad_font)},
    )
    assert plan.rejected[0].reason == "malformed_font"
    assert plan.rejected[0].details["diagnostics"][0]["reason"] == "malformed_font"


def test_unsupported_script_has_structured_diagnostics():
    with pytest.raises(f3d.text.TextShapingError) as caught:
        f3d.text.shape(
            "A",
            [str(ROOT / "assets/fonts/NotoSans-subset.ttf")],
            12.0,
            script="beng",
        )
    assert caught.value.diagnostics[0]["reason"] == "unsupported_script"


def test_label_support_docs_record_shaping_scope():
    text = (ROOT / "docs/guides/label_support_matrix.md").read_text(encoding="utf-8")

    assert "Arabic joining" in text
    assert "Indic shaping" in text
