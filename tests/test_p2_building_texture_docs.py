from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_building_texture_matrix_documents_prerequisites_and_fallbacks():
    text = _doc("docs/guides/building_support_matrix.md")

    assert "| Textured PBR buildings | `unsupported` |" in text
    assert "`missing_texture_path`" in text
    assert "`missing_uvs`" in text
    assert "`unsupported_texture_format`" in text
    assert "`placeholder_fallback`" in text
    assert "UV coordinates" in text
    assert "albedo texture path" in text
    assert "scalar fallback" in text


def test_building_docs_do_not_present_scalar_fallback_as_textured_success():
    text = (_doc("docs/guides/building_support_matrix.md") + "\n" + _doc("docs/guides/offline_3d_map_rendering.md")).lower()

    assert "textured pbr buildings | `supported`" not in text
    assert "scalar fallback is not textured pbr support" in text
    assert "does not render textured pbr buildings end to end" in text
