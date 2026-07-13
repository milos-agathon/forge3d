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


def test_three_native_text_symbols_are_registered_together():
    native = get_native_module()
    assert all(hasattr(native, name) for name in (
        "text_shape", "rasterize_shaped_run", "bake_msdf_atlas"
    ))


@pytest.mark.parametrize("name", ["rasterize_shaped_run", "bake_msdf_atlas"])
def test_future_rendering_seams_fail_explicitly(name):
    function = getattr(forge3d.text, name)
    with pytest.raises(NotImplementedError) as caught:
        function()
    assert caught.value.diagnostics[0]["status"] == "diagnostic_block"
    assert caught.value.diagnostics[0]["reason"] == "littera_rendering_deferred"
