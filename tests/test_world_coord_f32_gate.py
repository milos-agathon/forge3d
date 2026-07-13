# tests/test_world_coord_f32_gate.py
# MENSURA win 5 (CI-gated): the textual f64 -> f32 NARROWING cliff exists in
# exactly ONE place.
#
# Source-level gate in the style of tests/test_allocation_gate.py: across the
# world-coordinate surface of the tree, exactly one `as f32` cast applied to a
# world coordinate exists — inside `Anchor::narrow`, the helper used only by
# `Anchor::to_render_*` (src/camera/anchor.rs).
#
# SCOPE — be precise about what this PROVES vs. what it does NOT:
#   PROVES: no `as f32` token on a world-vocabulary source line survives
#   outside Anchor::narrow, and Anchor::narrow is the SINGLE narrowing
#   implementation.
#   DOES NOT PROVE renderer-wide f64 anchoring. A world coordinate that is
#   STORED as f32 / Vec3 / [f32;3] (via a glam f32 constructor, a PyO3 f32
#   parameter, or a numpy f32 array) emits no `as f32` token and is therefore
#   invisible to this textual gate. Giving every Scene / terrain / 3D-Tiles /
#   point-cloud / vector / culling / picking / label path an f64 object origin
#   is tracked as the remaining M-06 renderer-wide work and is NOT closed by
#   this gate alone.
# RELEVANT FILES: src/camera/anchor.rs, src/geo/units.rs, src/tiles3d/bounds.rs

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Scan every Rust source file. Candidate lines are selected by world-space
# vocabulary; harmless render scalars are ignored rather than hidden by a
# directory allowlist.
GATED = ["src"]
WORLD_HINT = re.compile(
    r"\b(world|object_origin|ecef|longitude|latitude|lon|lat|target|eye|world_coord)\w*\b",
    re.IGNORECASE,
)

# The single sanctioned world-coordinate narrowing site.
SANCTIONED = ("src/camera/anchor.rs", "value as f32")

# Casts in gated files that are demonstrably NOT world coordinates. Each entry
# is (path, exact code substring, justification). Keep this list minimal —
# every addition must name a non-coordinate operand.
ALLOWLIST = [
    ("src/core/water_surface/constructor.rs", "x as f32 * step", "grid index"),
    ("src/core/water_surface/constructor.rs", "y as f32 * step", "grid index"),
    ("src/path_tracing/restir/types.rs", "self.m as f32", "reservoir sample count"),
    ("src/path_tracing/restir/types.rs", "other.m as f32", "reservoir sample count"),
    ("src/shadows/cascade_math.rs", "shadow_map_size as f32", "texture resolution"),
    ("src/terrain/renderer/resources/resize.rs", "width as f32 * resolution_scale", "pixel width"),
    ("src/terrain/renderer/resources/resize.rs", "height as f32 * resolution_scale", "pixel height"),
    ("src/viewer/terrain/overlay/stack/composite.rs", "target_width as f32", "pixel width"),
    ("src/viewer/terrain/overlay/stack/composite.rs", "target_height as f32", "pixel height"),
    (
        "src/scene/py_api/base.rs",
        "self.width as f32 / self.height as f32",
        "u32 viewport pixel dimensions (aspect ratio), not world coordinates",
    ),
    (
        "src/gis/terrarium.rs",
        "pixel[0] as f32 * 256.0 + pixel[1] as f32 + pixel[2] as f32 / 256.0",
        "u8 Terrarium tile decode: 24-bit quantized source data, every "
        "representable value is exact in f32 (no f64 world value is narrowed)",
    ),
    (
        "src/gis/terrarium.rs",
        "values[row * tile_width + col] as f32",
        "mosaic copy of the same 24-bit quantized Terrarium values back into "
        "the f32 DEM band; exact by construction",
    ),
]


def _strip_comments(text: str) -> str:
    # Line comments and doc comments; block comments are not used for code in
    # the gated files.
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def _gated_files():
    files = []
    for entry in GATED:
        path = ROOT / entry
        if path.is_dir():
            files.extend(sorted(path.rglob("*.rs")))
        elif path.exists():
            files.append(path)
        else:
            raise AssertionError(f"gated path missing: {entry}")
    return files


def _cast_sites():
    sites = []
    for path in _gated_files():
        rel = path.relative_to(ROOT).as_posix()
        text = _strip_comments(path.read_text(encoding="utf-8"))
        for i, line in enumerate(text.splitlines(), 1):
            for _ in re.finditer(r"\bas f32\b", line):
                stripped = line.strip()
                if WORLD_HINT.search(stripped) or (rel, "value as f32") == SANCTIONED:
                    sites.append((rel, i, stripped))
    return sites


def test_exactly_one_world_coordinate_f32_cast():
    sites = _cast_sites()

    def allowed(site):
        rel, _i, line = site
        return any(rel == p and frag in line for p, frag in
                   [(a[0], a[1]) for a in ALLOWLIST])

    world_casts = [s for s in sites if not allowed(s)]
    assert len(world_casts) == 1, (
        "expected exactly ONE world-coordinate `as f32` cast "
        f"(Anchor::narrow); found {len(world_casts)}: {world_casts}"
    )
    rel, _line_no, line = world_casts[0]
    assert rel == SANCTIONED[0] and SANCTIONED[1] in line, (
        f"the single sanctioned cast moved: {world_casts[0]}"
    )


def test_allowlist_entries_still_exist_verbatim():
    # A stale allowlist entry would silently widen the gate.
    for rel, frag, _why in ALLOWLIST:
        text = _strip_comments((ROOT / rel).read_text(encoding="utf-8"))
        assert frag in text, f"stale allowlist entry: {rel}: {frag}"


def test_legacy_truncation_sites_are_gone():
    # The original f32 cliffs named by the MENSURA audit must stay dead.
    bounds = (ROOT / "src/tiles3d/bounds.rs").read_text(encoding="utf-8")
    assert "Vec3::new(x as f32, y as f32, z as f32)" not in bounds
    base = (ROOT / "src/scene/py_api/base.rs").read_text(encoding="utf-8")
    assert "eye: (f32, f32, f32)" not in base, (
        "set_camera_look_at must accept f64 world coordinates"
    )


def test_single_narrowing_implementation_lives_only_in_anchor():
    # M-06: keep exactly ONE sanctioned narrowing implementation. The camera
    # anchor module must contain exactly one textual `as f32` (inside
    # Anchor::narrow); a second narrowing helper added here fails immediately.
    anchor = _strip_comments((ROOT / SANCTIONED[0]).read_text(encoding="utf-8"))
    count = len(re.findall(r"\bas f32\b", anchor))
    assert count == 1, (
        f"{SANCTIONED[0]} must hold exactly one `as f32` (the sole narrowing "
        f"implementation); found {count}"
    )
    assert "fn narrow(value: f64) -> f32" in anchor, (
        "the sanctioned narrowing helper Anchor::narrow(value: f64) -> f32 moved"
    )


# A hidden helper that rebuilds a render Vec3 by narrowing three world
# components — the exact pre-MENSURA cliff — must not reappear in ANY file.
_TRIPLE_NARROW = re.compile(
    r"Vec3::new\(\s*\w+ as f32\s*,\s*\w+ as f32\s*,\s*\w+ as f32\s*\)"
)


def test_no_file_rebuilds_a_render_vec3_from_three_narrowed_components():
    for path in _gated_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == SANCTIONED[0]:
            continue  # anchor narrows component-by-component via Self::narrow
        text = _strip_comments(path.read_text(encoding="utf-8"))
        match = _TRIPLE_NARROW.search(text)
        assert match is None, (
            f"{rel} rebuilds a render Vec3 from three narrowed scalars "
            f"({match.group(0)!r}); route world coordinates through Anchor instead"
        )


def test_public_camera_helpers_anchor_earth_scale_targets():
    """Public camera matrices preserve a 10 m target offset at Earth radius."""
    import numpy as np
    import forge3d as f3d
    from forge3d import _forge3d

    local = np.asarray(
        _forge3d.camera_look_at((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    earth = np.asarray(
        _forge3d.camera_look_at(
            (6_378_137.0, 2_000.0, -3_000.0),
            (6_378_147.0, 2_000.0, -3_000.0),
            (0.0, 1.0, 0.0),
        )
    )
    np.testing.assert_allclose(earth, local, rtol=0.0, atol=1e-6)
