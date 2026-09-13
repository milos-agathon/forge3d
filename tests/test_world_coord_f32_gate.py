"""Fail-closed inventory for every Rust narrowing primitive.

The inventory is deliberately name-agnostic: every ``as f32``, ``.as_vec*()``,
and f64/DVec-to-f32/Vec helper is occurrence-locked across production Rust.
Separate positive contracts prove that each viewer world-position route calls
the active Anchor inside the producing function.
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANCTIONED = "src/camera/anchor.rs"
SANCTIONED_DD_SPLITS = {
    (
        "src/core/dd.rs",
        "from_f64",
        "as_f32",
        1,
        "let hi = value as f32",
    ),
    (
        "src/core/dd.rs",
        "from_f64",
        "as_f32",
        2,
        "let lo = (value - hi as f64) as f32",
    ),
}

# Updated only after reviewing the complete inventory printed by a failure.
# The digest includes (file, function, operation, ordinal, normalized statement).
# Re-frozen at the paris-eiffel-day-pt tip (9a0a7d0f) after merging it into
# this branch: the merged prometheus line carries the +46 sites over the
# AEQUITAS-era 1345 freeze, and this branch adds zero conversion sites on top
# -- inventory and digest are identical at the merge base tip and HEAD. The
# earlier AEQUITAS raster-twin transition (+22/-7 in
# src/offscreen/adjudication_raster.rs) remains recorded below.
EXPECTED_CONVERSION_COUNT = 1391
EXPECTED_CONVERSION_SHA256 = "ae37f4b7a2fe587dab04f13a6a45dc53c058874c76d87e5a1f3bcc0cea8e29cc"

# The reviewed TERMINUS reader transition remains locked below. COMPENDIUM adds
# four integer-to-f32 reconstruction conversions in predict.rs; those are
# included in the current count and digest above without weakening the reader
# transition assertion.
REVIEWED_INVENTORY_TRANSITION = {
    "current_count": EXPECTED_CONVERSION_COUNT,
    "removed": (
        "src/terrain/cog/cog_reader.rs",
        "decode_heights",
        "as_f32",
        1,
        "heights.push(f64::from_le_bytes(bytes) as f32)",
    ),
    "added": (
        "src/terrain/cog/cog_reader.rs",
        "decode_heights",
        "as_f32",
        1,
        "heights.push(f64::from_le_bytes(read_le_bytes8(data, i * 8)) as f32)",
    ),
}

# AEQUITAS replaced the adjudication raster twin's CPU supersampler
# (render_raster_reference_incremental, base_uniforms) with the production
# instanced-PBR path plus an analytic ground-GI bake. Every added site is an
# integer-index-to-f32 normalization cast in adjudication_raster.rs; the full
# removed/added site lists are frozen here so the digest bump is auditable.
REVIEWED_AEQUITAS_RASTER_TRANSITION = {
    "base_count": 1327,
    "base_digest": "d6368abc90af4f03c5d1a9f573e4d9efebd38767d9927098cdcc418fb0e42817",
    "result_count": 1345,
    "result_digest": "5ba005a550bdbb994b5a8dbfb36fd691d08bd05cecce6bd58919ed59ac5596c2",
    "path": "src/offscreen/adjudication_raster.rs",
    "removed": (
        ("base_uniforms", 1, "cam_origin_fovy: [origin.x, origin.y, origin.z, desc.fov_y_rad()], cam_right_aspect: [right.x, right.y, right.z, aspect], cam_up_w: [up.x, up.y, up.z, rw as f32], cam_forward_h: [forward.x, forward.y, forward.z, rh as f32], sun_dir_intensity: [sun.x, sun.y, sun.z, desc.sun_intensity], sun_color_pad: [desc.sun_color[0], desc.sun_color[1], desc.sun_color[2], 0.0], environment: { let e = desc.environment_raw()"),
        ("base_uniforms", 2, "cam_origin_fovy: [origin.x, origin.y, origin.z, desc.fov_y_rad()], cam_right_aspect: [right.x, right.y, right.z, aspect], cam_up_w: [up.x, up.y, up.z, rw as f32], cam_forward_h: [forward.x, forward.y, forward.z, rh as f32], sun_dir_intensity: [sun.x, sun.y, sun.z, desc.sun_intensity], sun_color_pad: [desc.sun_color[0], desc.sun_color[1], desc.sun_color[2], 0.0], environment: { let e = desc.environment_raw()"),
        ("render_raster_reference_incremental", 1, "let aspect = width as f32 / height as f32"),
        ("render_raster_reference_incremental", 2, "let aspect = width as f32 / height as f32"),
        ("render_raster_reference_incremental", 3, "u.misc = [desc.plane_half_extent, i as f32, 1.0, 0.0]"),
        ("render_raster_reference_incremental", 4, "let o = (k as f32 + 0.5) / SSAA as f32 - 0.5"),
        ("render_raster_reference_incremental", 5, "let o = (k as f32 + 0.5) / SSAA as f32 - 0.5"),
    ),
    "added": (
        ("bake_plane_indirect", 1, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 2, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 3, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 4, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 5, "let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32"),
        ("bake_plane_indirect", 6, "let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32"),
        ("bake_plane_indirect", 7, "let phi = k as f32 * 2.399_963_2"),
        ("bake_plane_indirect", 8, "let occ = 2.0 * w_sky / BAKE_RAYS as f32"),
        ("bake_plane_indirect", 9, "let v_sun = ray_hit_sphere(desc, p + Vec3::Y * 1e-3, light, None).is_none() as i32 as f32"),
        ("bake_plane_indirect", 10, "let u = ((p.x - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize"),
        ("bake_plane_indirect", 11, "let u = ((p.x - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize"),
        ("bake_plane_indirect", 12, "let v = ((p.z - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize"),
        ("bake_plane_indirect", 13, "let v = ((p.z - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize"),
        ("bake_plane_indirect", 14, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 15, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 16, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 17, "let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )"),
        ("bake_plane_indirect", 18, "let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32"),
        ("bake_plane_indirect", 19, "let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32"),
        ("bake_plane_indirect", 20, "let phi = k as f32 * 2.399_963_2"),
        ("bake_plane_indirect", 21, "} } let em = a_plane * e_bounce * (2.0 / BAKE_RAYS as f32)"),
        ("render", 1, "let projection = Mat4::perspective_rh(desc.fov_y_rad(), width as f32 / height as f32, near, far)"),
        ("render", 2, "let projection = Mat4::perspective_rh(desc.fov_y_rad(), width as f32 / height as f32, near, far)"),
        ("render", 3, "let weights: Vec<f32> = (0..SSAA) .map(|k| 1.0 - (((k as f32 + 0.5) / SSAA as f32 - 0.5).abs() / 0.5)) .collect()"),
        ("render", 4, "let weights: Vec<f32> = (0..SSAA) .map(|k| 1.0 - (((k as f32 + 0.5) / SSAA as f32 - 0.5).abs() / 0.5)) .collect()"),
    ),
}


def _strip_comments_and_strings(text: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _remove_cfg_test_modules(text: str) -> str:
    text = _strip_comments_and_strings(text)
    marker = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*mod\s+\w+\s*\{")
    while match := marker.search(text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        text = text[: match.start()] + " " * (cursor - match.start()) + text[cursor:]
    return text


def _function_spans(text: str):
    spans = []
    for match in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)[^;{]*\{", text, re.S):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        spans.append((match.start(), cursor, match.group(1)))
    return spans


def _function_name(spans, position: int) -> str:
    return next(
        (name for start, end, name in spans if start <= position < end),
        "<module>",
    )


def _statement(text: str, position: int) -> str:
    start = max(text.rfind(";", 0, position), text.rfind("{", 0, position)) + 1
    end_candidates = [candidate for candidate in (text.find(";", position), text.find("}", position)) if candidate >= 0]
    end = min(end_candidates, default=min(len(text), position + 160))
    return re.sub(r"\s+", " ", text[start:end]).strip()


CONVERSION_PATTERNS = {
    "as_f32": re.compile(r"\bas\s+f32\b"),
    "as_vec": re.compile(r"\.\s*as_vec[234]\s*\(\s*\)"),
    "f64_helper": re.compile(
        r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:<[^>{}]*>)?\s*\([^)]*(?:f64|DVec[234]|DMat[234])[^)]*\)\s*"
        r"(?:->\s*(?:f32|Vec[234]|\[\s*f32|Vec\s*<\s*f32))",
        re.S,
    ),
}


def _conversion_inventory_text(rel: str, raw: str):
    text = _remove_cfg_test_modules(raw)
    spans = _function_spans(text)
    matches = []
    for operation, pattern in CONVERSION_PATTERNS.items():
        matches.extend((match.start(), operation) for match in pattern.finditer(text))
    counters = {}
    sites = []
    for position, operation in sorted(matches):
        function = _function_name(spans, position)
        key = (function, operation)
        counters[key] = counters.get(key, 0) + 1
        sites.append(
            (
                rel,
                function,
                operation,
                counters[key],
                _statement(text, position),
            )
        )
    return sites


def _complete_conversion_inventory():
    sites = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        sites.extend(_conversion_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


def conversion_inventory():
    """Reviewed narrowing inventory, excluding the exact lossless DD split."""
    return [site for site in _complete_conversion_inventory() if site not in SANCTIONED_DD_SPLITS]


def _inventory_digest(sites) -> str:
    payload = "\n".join("\t".join(map(str, site)) for site in sites)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _function_body(rel: str, function: str) -> str:
    text = _remove_cfg_test_modules(_read(rel))
    for start, end, name in _function_spans(text):
        if name == function:
            return text[start:end]
    raise AssertionError(f"missing function {rel}::{function}")


def test_exact_production_conversion_inventory_is_frozen():
    sites = conversion_inventory()
    digest = _inventory_digest(sites)
    assert (len(sites), digest) == (
        EXPECTED_CONVERSION_COUNT,
        EXPECTED_CONVERSION_SHA256,
    ), f"conversion inventory changed: count={len(sites)} sha256={digest}\n" + "\n".join(
        repr(site) for site in sites
    )


def test_all_required_rejecting_probes_change_the_inventory():
    probes = [
        "v as f32",
        "coords[0] as f32",
        "position.as_vec3()",
        "coords.map(|v| v as f32)",
        "point.to_array().map(|v| v as f32)",
        "fn narrow(v: f64) -> f32 { v as f32 }",
        "macro_rules! narrow { ($v:expr) => { $v as f32 } }",
        "fn bad(v: f64, origin: f64) -> f32 { v as f32 - origin as f32 }",
    ]
    for probe in probes:
        assert _conversion_inventory_text("probe.rs", probe), f"scanner missed {probe}"


def test_dd_encode_has_exactly_the_two_reviewed_split_casts():
    actual = {
        site
        for site in _complete_conversion_inventory()
        if site[0] == "src/core/dd.rs" and site[1] == "from_f64"
    }
    assert actual == SANCTIONED_DD_SPLITS


def test_reviewed_checked_reader_inventory_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_INVENTORY_TRANSITION
    assert len(sites) == transition["current_count"] == EXPECTED_CONVERSION_COUNT
    assert _inventory_digest(sites) == EXPECTED_CONVERSION_SHA256
    assert transition["added"] in sites
    assert transition["removed"] not in sites


def test_reviewed_aequitas_raster_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_AEQUITAS_RASTER_TRANSITION
    assert len(sites) == EXPECTED_CONVERSION_COUNT
    assert _inventory_digest(sites) == EXPECTED_CONVERSION_SHA256
    path = transition["path"]
    for function, ordinal, statement in transition["removed"]:
        assert (path, function, "as_f32", ordinal, statement) not in sites
    for function, ordinal, statement in transition["added"]:
        assert (path, function, "as_f32", ordinal, statement) in sites
    # The transition bridges its own base/result counts; the current
    # EXPECTED_CONVERSION_COUNT additionally includes the +46 as_f32 sites the
    # paris-eiffel-day-pt merge carries in
    # src/path_tracing/hybrid_compute/render_terrain.rs (same sanctioned
    # integer-to-f32 class; identical inventory at the base tip and HEAD).
    assert len(transition["added"]) - len(transition["removed"]) == (
        transition["result_count"] - transition["base_count"]
    )


def test_anchor_narrow_is_the_only_world_conversion_implementation():
    anchor = _remove_cfg_test_modules(_read(SANCTIONED))
    narrow = _function_body(SANCTIONED, "narrow")
    assert len(re.findall(r"\bas\s+f32\b", narrow)) == 1
    assert "value as f32" in re.sub(r"\s+", " ", narrow)
    assert anchor.count("Self::narrow(") == 6

    position = _function_body(SANCTIONED, "to_render_vec3")
    assert "p - self.origin" in re.sub(r"\s+", " ", position)
    assert position.count("Self::narrow(") == 3
    direction = _function_body(SANCTIONED, "direction_to_render")
    assert direction.count("Self::narrow(") == 3


def test_anchor_dd_split_is_a_named_non_narrowing_crossing():
    body = re.sub(r"\s+", " ", _function_body(SANCTIONED, "to_dd"))
    assert "DDVec3::from_dvec3(p)" in body
    assert "Self::narrow(" not in body
    assert " as f32" not in body


def test_each_viewer_world_route_calls_its_active_anchor_in_the_same_function():
    routes = {
        ("src/viewer/viewer_types.rs", "view"): "self.anchor.view_look_at(",
        ("src/viewer/viewer_types.rs", "render_eye"): "self.anchor.to_render_vec3(",
        ("src/viewer/render/main_loop/frame_anchor.rs", "anchored_object_model"): "frame.anchor.model_offset(",
        ("src/viewer/pointcloud/state.rs", "packed_point"): "anchor.to_render_vec3(",
        ("src/viewer/terrain/vector_overlay.rs", "repack_source_vertices"): "anchor.to_render_vec3(",
        ("src/labels/mod.rs", "update_with_camera_anchored"): "anchor.to_render_vec3(",
        ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state"): "frame.anchor.to_render_vec3(",
        ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state"): "frame.anchor.to_render_vec3(",
        ("src/viewer/input/viewer_input.rs", "pick_at_screen"): ".to_world_from_render_f64(",
    }
    for (rel, function), required_call in routes.items():
        body = re.sub(r"\s+", "", _function_body(rel, function))
        normalized_call = re.sub(r"\s+", "", required_call)
        assert normalized_call in body, f"{rel}::{function} lacks {required_call}"


def test_cityjson_and_viewer_absolute_storage_types_are_explicitly_f64():
    assert "pub positions: Vec<f64>" in _read("src/import/cityjson/types.rs")
    assert "pub(crate) object_translation: glam::DVec3" in _read("src/viewer/viewer_struct.rs")
    assert "pub world_pos: DVec3" in _read("src/labels/types.rs")
    assert "pub position: DVec3" in _read("src/viewer/pointcloud/types.rs")


def test_public_camera_helper_preserves_earth_scale_offset():
    import numpy as np
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
