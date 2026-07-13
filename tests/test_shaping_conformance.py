from __future__ import annotations

import json
from pathlib import Path

import pytest

import forge3d
from forge3d._native import get_native_module


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "shaping"


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(DATA.glob("*.json")):
        cases.extend(json.loads(path.read_text(encoding="utf-8")))
    return cases


CASES = load_cases()


def test_committed_corpus_has_at_least_200_cases_across_required_groups():
    assert len(CASES) >= 200
    identities = {
        (case["text"], tuple(case["fonts"]), json.dumps(case["options"], sort_keys=True))
        for case in CASES
    }
    assert len(identities) >= 200
    expectations = {
        json.dumps(case["runs"], sort_keys=True, ensure_ascii=False)
        for case in CASES
    }
    assert len(expectations) >= 200
    assert {case["size"] for case in CASES} == {12.0}
    assert any(len(case["fonts"]) > 1 for case in CASES)
    assert any(case["options"].get("language") for case in CASES)
    assert any(case["options"].get("features") == {"liga": False} for case in CASES)
    assert any(case["options"].get("features") == {"liga": True} for case in CASES)
    texts = {case["text"] for case in CASES}
    assert {"سلام", "لا", "ل\u200dا", "ل\u200cا", "क्ष", "स्त्र", "श्र"} <= texts
    assert {"A\u2067שלום\u2069", "office\noffice"} <= texts
    assert {case["id"].split("-", 1)[0] for case in CASES} == {
        "arabic", "cjk", "devanagari", "hebrew", "latin", "mixed"
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_shape_matches_harfbuzz(case):
    fonts = [str(ROOT / path) for path in case["fonts"]]
    shaped = forge3d.text.shape(
        case["text"], fonts, case["size"], **case["options"]
    )
    assert shaped.to_dict()["runs"] == case["runs"]


def test_shaped_text_is_native_and_immutable():
    shaped = forge3d.text.shape(
        "Map", [str(ROOT / "assets/fonts/NotoSans-subset.ttf")], 12.0
    )
    assert type(shaped).__module__ == "forge3d._forge3d"
    with pytest.raises(AttributeError):
        shaped.size = 24.0


def test_shape_is_size_invariant_in_q26_6_space():
    font = str(ROOT / "assets/fonts/NotoSans-subset.ttf")
    small = forge3d.text.shape("office", [font], 8.0).to_dict()["runs"]
    large = forge3d.text.shape("office", [font], 72.0).to_dict()["runs"]
    assert small == large


def test_public_payload_is_the_task_7_integer_conformance_surface():
    shaped = forge3d.text.shape(
        "Map", [str(ROOT / "assets/fonts/NotoSans-subset.ttf")], 12.0
    )
    payload = shaped.to_dict()
    assert set(payload) == {"text", "size", "levels", "legal_breaks", "runs"}
    assert set(payload["runs"][0]["glyphs"][0]) == {
        "glyph_id", "font_index", "cluster", "x_advance", "x_offset"
    }


def test_rtl_glyphs_remain_logical_until_line_ranges_are_known():
    shaped = forge3d.text.shape(
        "שלום", [str(ROOT / "assets/fonts/NotoSansHebrew-subset.ttf")], 12.0
    )
    clusters = [glyph["cluster"] for glyph in shaped.to_dict()["runs"][0]["glyphs"]]
    assert clusters == sorted(clusters)


def test_missing_font_has_structured_diagnostics():
    with pytest.raises(forge3d.text.TextShapingError) as caught:
        forge3d.text.shape("Map", [str(ROOT / "missing.ttf")], 12.0)
    assert caught.value.diagnostics == [{
        "status": "diagnostic_block",
        "reason": "font_not_found",
        "font": str(ROOT / "missing.ttf"),
    }]


def test_default_ignorables_and_bidi_isolates_do_not_emit_notdef():
    shaped = forge3d.text.shape(
        "A\u2067שלום\u2069",
        [
            str(ROOT / "assets/fonts/NotoSans-subset.ttf"),
            str(ROOT / "assets/fonts/NotoSansHebrew-subset.ttf"),
        ],
        12.0,
    ).to_dict()
    glyphs = [glyph for run in shaped["runs"] for glyph in run["glyphs"]]
    assert all(glyph["glyph_id"] != 0 for glyph in glyphs)
    assert {glyph["cluster"] for glyph in glyphs}.isdisjoint({1, 9})


def test_join_controls_do_not_require_cmap_glyphs():
    font = str(ROOT / "assets/fonts/NotoSansArabic-subset.ttf")
    outputs = {}
    for text in ("ل\u200dا", "ل\u200cا", "\u200d"):
        shaped = forge3d.text.shape(text, [font], 12.0).to_dict()
        outputs[text] = [
            glyph["glyph_id"]
            for run in shaped["runs"]
            for glyph in run["glyphs"]
        ]
        assert all(
            glyph["glyph_id"] != 0
            for run in shaped["runs"]
            for glyph in run["glyphs"]
        )
    joined = forge3d.text.shape("لا", [font], 12.0).to_dict()
    joined_ids = [
        glyph["glyph_id"]
        for run in joined["runs"]
        for glyph in run["glyphs"]
    ]
    assert outputs["ل\u200dا"] == joined_ids
    assert outputs["ل\u200cا"] != joined_ids


def test_mandatory_breaks_emit_no_glyph_and_split_shaping_runs():
    shaped = forge3d.text.shape(
        "office\noffice", [str(ROOT / "assets/fonts/NotoSans-subset.ttf")], 12.0
    ).to_dict()
    assert len(shaped["runs"]) == 2
    assert all(
        glyph["cluster"] != len("office")
        for run in shaped["runs"]
        for glyph in run["glyphs"]
    )


def test_language_enables_locl_and_explicit_features_override_defaults():
    deva = str(ROOT / "assets/fonts/NotoSansDevanagari-subset.ttf")
    default = forge3d.text.shape("शहर", [deva], 12.0, script="deva")
    marathi = forge3d.text.shape(
        "शहर", [deva], 12.0, script="deva", language="MAR "
    )
    assert default.to_dict()["runs"] != marathi.to_dict()["runs"]

    latin = str(ROOT / "assets/fonts/NotoSans-subset.ttf")
    ligature = forge3d.text.shape("office", [latin], 12.0)
    disabled = forge3d.text.shape(
        "office", [latin], 12.0, features={"liga": False}
    )
    assert ligature.to_dict()["runs"] != disabled.to_dict()["runs"]


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (lambda native, font: native.text_shape("A", [font], 0.0), "invalid_size"),
        (lambda native, font: native.text_shape("☃", [font], 12.0), "missing_glyph"),
        (lambda native, font: native.text_shape("A", [font], 12.0, "beng"), "unsupported_script"),
        (lambda native, font: native.text_shape("A", [font], 12.0, None, "x"), "malformed_tag"),
    ],
)
def test_native_shape_errors_have_stable_diagnostics(call, reason):
    native = get_native_module()
    font = str(ROOT / "assets/fonts/NotoSans-subset.ttf")
    with pytest.raises(ValueError) as caught:
        call(native, font)
    assert caught.value.diagnostics[0]["reason"] == reason


def test_native_invalid_font_has_structured_diagnostics(tmp_path):
    bad_font = tmp_path / "bad.ttf"
    bad_font.write_bytes(b"not a font")
    native = get_native_module()
    with pytest.raises(ValueError) as caught:
        native.text_shape("A", [str(bad_font)], 12.0)
    assert caught.value.diagnostics[0]["reason"] == "malformed_font"
    assert caught.value.diagnostics[0]["font"] == str(bad_font)


def test_wrapper_preserves_native_diagnostics_without_message_parsing():
    font = str(ROOT / "assets/fonts/NotoSans-subset.ttf")
    with pytest.raises(forge3d.text.TextShapingError) as caught:
        forge3d.text.shape("☃", [font], 12.0)
    assert caught.value.diagnostics[0]["reason"] == "missing_glyph"
    assert caught.value.diagnostics[0]["codepoint"] == "U+2603"


def test_three_native_text_symbols_are_registered_together():
    native = get_native_module()
    assert all(hasattr(native, name) for name in (
        "text_shape", "rasterize_shaped_run", "bake_msdf_atlas"
    ))


@pytest.mark.parametrize("name", ["rasterize_shaped_run", "bake_msdf_atlas"])
def test_future_rendering_seams_fail_explicitly(name):
    native = get_native_module()
    with pytest.raises(NotImplementedError) as native_caught:
        getattr(native, name)()
    assert native_caught.value.diagnostics[0]["operation"] == name

    function = getattr(forge3d.text, name)
    with pytest.raises(NotImplementedError) as caught:
        function()
    assert caught.value.diagnostics[0]["status"] == "diagnostic_block"
    assert caught.value.diagnostics[0]["reason"] == "littera_rendering_deferred"
