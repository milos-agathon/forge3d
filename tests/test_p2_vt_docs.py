from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_vt_support_matrix_states_exact_family_runtime_status():
    text = _doc("docs/guides/virtual_texturing_support_matrix.md")

    assert "| Albedo terrain VT family | `supported` | Runtime pages the albedo family" in text
    assert "| Normal terrain VT family | `unsupported` | Python accepts `normal`" in text
    assert "| Mask terrain VT family | `unsupported` | Python accepts `mask`" in text
    assert "`vt_unsupported_family`" in text
    assert "`vt.normal`" in text
    assert "`vt.mask`" in text
    assert "native runtime pages only `albedo`" in text


def test_vt_docs_disallow_silent_skip_or_support_overclaim():
    text = _doc("docs/guides/virtual_texturing_support_matrix.md").lower()

    assert "must not silently skip" in text
    assert "normal terrain vt family | `supported`" not in text
    assert "mask terrain vt family | `supported`" not in text
    assert "full normal/mask runtime support" not in text
