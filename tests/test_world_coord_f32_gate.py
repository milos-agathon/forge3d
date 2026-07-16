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

# Updated only after reviewing the complete inventory printed by a failure.
# The digest includes (file, function, operation, ordinal, normalized statement).
EXPECTED_CONVERSION_COUNT = 1278
EXPECTED_CONVERSION_SHA256 = "db001acdb19bd51042d2940de1194c13d382325dc5e550bec70e23d764f1d7ec"


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


def conversion_inventory():
    sites = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        sites.extend(_conversion_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


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
