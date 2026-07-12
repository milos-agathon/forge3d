"""CENSOR source gate for buried render shims and cached viewer bind groups."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Symbols of the deleted dead structures (CENSOR item 5). Reintroducing any of
# them — under any file name — must trip this gate, not just recreating the
# original paths.
FORBIDDEN_SYMBOLS = re.compile(
    r"\bPostFxChain\b|\bexecute_chain\b|\bpostfx_apply_noop\b"
    r"|\bPostFxResourcePool\b|\bRenderBundleManager\b|\brender_bundles\b"
)

# Frame-loop files allowed to contain `.create_bind_group(`:
# - postfx_cache.rs: the identity-keyed caches themselves (miss-only creation).
# - geometry/pass.rs: lazy autogen albedo bind group, created once and stored.
FRAME_LOOP_BIND_GROUP_ALLOWLIST = {
    "postfx_cache.rs",
    "geometry/pass.rs",
}


def test_orphan_render_abstractions_are_deleted() -> None:
    deleted = [
        "src/core/framegraph.rs",
        "src/core/postfx/chain.rs",
        "src/core/postfx/effect.rs",
        "src/core/postfx/resources.rs",
        "src/core/postfx/mod.rs",
        "src/core/postfx/config.rs",
        "src/core/bloom/effect.rs",
        "src/core/bloom/execute.rs",
        "src/core/render_bundles.rs",
        "src/core/render_bundles_types.rs",
    ]
    survivors = [path for path in deleted if (ROOT / path).exists()]
    assert survivors == [], f"orphan render abstractions survived: {survivors}"


def test_forbidden_dead_structure_symbols_do_not_return() -> None:
    hits = []
    for path in (ROOT / "src").rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN_SYMBOLS.search(line) and not line.lstrip().startswith("//"):
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{i}")
    assert hits == [], f"deleted dead-structure symbols reintroduced: {hits}"


def test_frame_loop_does_not_create_bind_groups() -> None:
    main_loop = ROOT / "src/viewer/render/main_loop"
    offenders = []
    for path in main_loop.rglob("*.rs"):
        rel = path.relative_to(main_loop).as_posix()
        if rel in FRAME_LOOP_BIND_GROUP_ALLOWLIST:
            continue
        if ".create_bind_group(" in path.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert offenders == [], (
        "per-frame bind-group creation in the frame loop outside the documented "
        f"allowlist: {offenders}"
    )
    tonemap = (ROOT / "src/core/tonemap.rs").read_text(encoding="utf-8")
    assert ".create_bind_group(" not in tonemap

    routine = "\n".join(
        (main_loop / path).read_text(encoding="utf-8")
        for path in ("frame_setup.rs", "geometry/fog.rs", "geometry/fog_dispatch.rs")
    )
    assert ".create_sampler(" not in routine


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
    for cache in (
        "sky_bg0_cache",
        "fog_bg0_cache",
        "fog_bg2_cache",
        "fog_bg2_half_cache",
        "fog_upsample_bg_cache",
    ):
        assert f"{cache}.borrow_mut().take()" in resize
    assert "lit_bind_group_cache.borrow_mut().take()" in ibl
