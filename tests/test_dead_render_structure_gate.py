"""CENSOR source gate for buried render shims and cached viewer bind groups."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_orphan_render_abstractions_are_deleted() -> None:
    deleted = [
        "src/core/framegraph.rs",
        "src/core/postfx/chain.rs",
        "src/core/postfx/effect.rs",
        "src/core/postfx/resources.rs",
        "src/core/bloom/effect.rs",
        "src/core/bloom/execute.rs",
    ]
    survivors = [path for path in deleted if (ROOT / path).exists()]
    assert survivors == [], f"orphan render abstractions survived: {survivors}"


def test_frame_loop_does_not_create_bind_groups() -> None:
    postfx = (ROOT / "src/viewer/render/main_loop/postfx.rs").read_text(
        encoding="utf-8"
    )
    tonemap = (ROOT / "src/core/tonemap.rs").read_text(encoding="utf-8")
    assert ".create_bind_group(" not in postfx
    assert ".create_bind_group(" not in tonemap


def test_viewer_bind_group_cache_is_invalidated_with_resources() -> None:
    cache = (ROOT / "src/viewer/render/main_loop/postfx_cache.rs").read_text(
        encoding="utf-8"
    )
    resize = (ROOT / "src/viewer/state/resize.rs").read_text(encoding="utf-8")
    ibl = (ROOT / "src/viewer/state/viewer_helpers/ibl.rs").read_text(
        encoding="utf-8"
    )

    assert "lit_bind_group_cache.borrow().is_some()" in cache
    assert "comp_bind_group_cache.borrow().contains_key" in cache
    assert "lit_bind_group_cache.borrow_mut().take()" in resize
    assert "comp_bind_group_cache.borrow_mut().clear()" in resize
    assert "lit_bind_group_cache.borrow_mut().take()" in ibl
