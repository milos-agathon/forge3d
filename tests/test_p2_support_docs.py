from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_integrated_p2_support_docs_use_exact_support_terms_and_deferral_wording():
    docs = "\n".join(
        _doc(path)
        for path in (
            "docs/guides/virtual_texturing_support_matrix.md",
            "docs/guides/building_support_matrix.md",
            "docs/guides/label_support_matrix.md",
            "docs/guides/large_scene_support.md",
            "docs/guides/competitive_positioning.md",
        )
    )

    for term in (
        "`supported`",
        "`underdeveloped`",
        "`missing`",
        "`Pro-gated`",
        "`placeholder/fallback`",
        "`experimental`",
        "`unsupported`",
        "`non-goal`",
    ):
        assert term in docs
    assert "non-MVP-blocking" in docs
    assert "diagnosed before render" in docs


def test_integrated_p2_support_docs_do_not_overclaim_external_engine_parity():
    docs = "\n".join(
        _doc(path)
        for path in (
            "docs/guides/virtual_texturing_support_matrix.md",
            "docs/guides/building_support_matrix.md",
            "docs/guides/label_support_matrix.md",
            "docs/guides/large_scene_support.md",
            "docs/guides/competitive_positioning.md",
        )
    ).lower()

    forbidden = (
        "full mapbox gl parity",
        "full cesium parity",
        "blender replacement",
        "unreal replacement",
        "web-first engine",
        "hosted streaming support",
    )
    for phrase in forbidden:
        assert phrase not in docs
