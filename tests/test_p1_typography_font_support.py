from __future__ import annotations


def test_default_latin_font_atlas_loads_bundled_ascii_coverage():
    from forge3d.map_scene import FontAtlas

    atlas = FontAtlas.default_latin()

    assert atlas.font_size == 24
    assert atlas.line_height == 28
    for char in (" ", "A", "Z", "a", "z", "0", "9"):
        assert atlas.covers(char), char
    assert {key: atlas.coverage[key] for key in ("start", "end", "name")} == {
        "start": 32,
        "end": 127,
        "name": "Basic Latin",
    }
    assert atlas.coverage["atlas_kind"] == "msdf_font_atlas"
    assert atlas.coverage["px_range"] > 0
    assert atlas.diagnostics == []


def test_font_atlas_from_missing_ttf_or_otf_returns_typed_diagnostic():
    from forge3d.map_scene import FontAtlas

    atlas = FontAtlas.from_font("missing-font.ttf")

    assert atlas.glyphs == set()
    assert [diagnostic.code for diagnostic in atlas.diagnostics] == ["missing_external_asset"]
    assert atlas.diagnostics[0].details["layer_type"] == "font_atlas"
    assert atlas.diagnostics[0].object_id == "missing-font.ttf"


def test_font_fallback_ranges_are_deterministic_and_queryable():
    from forge3d.map_scene import FontAtlas, FontFallbackRange

    atlas = FontAtlas.default_latin(
        fallbacks=[
            FontFallbackRange("Greek", 0x0370, 0x03FF, "GreekFallback"),
            FontFallbackRange("Latin", 0x0000, 0x007F, "DefaultLatin"),
        ]
    )

    assert [item.name for item in atlas.fallbacks] == ["Latin", "Greek"]
    assert atlas.fallback_for("A").font_family == "DefaultLatin"
    assert atlas.fallback_for("Ω").font_family == "GreekFallback"
    assert atlas.fallback_for("Ж") is None


def test_font_atlas_reports_unicode_coverage_gap_before_render():
    from forge3d.map_scene import FontAtlas

    atlas = FontAtlas.default_latin()
    diagnostics = atlas.validate_text("Alpha Ω")

    assert [diagnostic.code for diagnostic in diagnostics] == ["unicode_coverage_gap"]
    assert diagnostics[0].details["missing_glyphs"] == ["Ω"]


def test_typography_metrics_reflect_kerning_tracking_and_line_height():
    from forge3d.map_scene import TypographySettings

    plain = TypographySettings(font_size=20, kerning=False, tracking=0.0, line_height=20)
    kerned = TypographySettings(font_size=20, kerning=True, tracking=2.0, line_height=30)

    plain_metrics = plain.measure_text("AV\nA")
    kerned_metrics = kerned.measure_text("AV\nA")

    assert kerned_metrics["line_count"] == 2
    assert kerned_metrics["height"] == 60
    assert kerned_metrics["width"] > plain_metrics["width"]
    assert kerned_metrics["kerning_applied"] is True
    assert kerned_metrics["tracking"] == 2.0


def test_typography_metrics_use_native_outline_bounds_not_character_count():
    from forge3d.map_scene import TypographySettings

    metrics = TypographySettings(font_size=32, kerning=True, tracking=0.0)

    narrow = metrics.measure_text("iiiiii")
    wide = metrics.measure_text("WWWWWW")

    assert wide["width"] > narrow["width"] * 2
    assert narrow["line_count"] == 1
    assert wide["line_count"] == 1


def test_typography_multiline_and_callout_layout_metadata_is_explicit():
    from forge3d.map_scene import TypographySettings

    settings = TypographySettings(font_size=16, multiline=True, callout=True, callout_offset=(8.0, -4.0))
    layout = settings.layout_label("Peak\nElevation", anchor=(100.0, 50.0, 12.0))

    assert layout["lines"] == ["Peak", "Elevation"]
    assert layout["metrics"]["line_count"] == 2
    assert layout["callout"] == {
        "enabled": True,
        "anchor": [100.0, 50.0, 12.0],
        "label_anchor": [108.0, 46.0, 12.0],
        "offset": [8.0, -4.0],
    }


def test_docs_record_live_complex_script_shaping_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workflow = (root / "docs/guides/data_and_scene_workflows.md").read_text(encoding="utf-8")
    api_reference = (root / "docs/api/api_reference.rst").read_text(encoding="utf-8")

    assert "Complex-script shaping uses the native LITTERA" in workflow
    assert "positioned-outline" in workflow
    assert "FontAtlas.default_latin" in api_reference
    assert "TypographySettings" in api_reference
