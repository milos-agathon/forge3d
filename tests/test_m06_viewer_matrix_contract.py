"""Normalized, occurrence-level inventory of interactive-viewer matrix producers."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src/viewer", ROOT / "src/shadows")

PATTERNS = {
    "look_at": re.compile(r"\b(?:D?Mat4)::look_at_[a-z_]+\s*\("),
    "projection_ctor": re.compile(r"\b(?:D?Mat4)::(?:perspective|orthographic)_[a-z_]+\s*\("),
    "anchor_view": re.compile(
        r"\b(?:self\s*\.\s*)?anchor\s*\.\s*view_look_at\s*\("
    ),
    "frame_view": re.compile(r"\bframe\.view\s*\("),
    "frame_view_projection": re.compile(r"\bframe\.view_projection\s*\("),
    "controller_view_delegate": re.compile(r"\bself\.(?:orbit|fps)\.view_matrix\s*\("),
    "matrix_compose": re.compile(
        r"\b(?:self\.projection\s*\([^;]+?\)|frame\.projection\s*\([^;]+?\)|"
        r"camera_projection|light_projection|proj|projection)\s*\*\s*"
        r"(?:self\.view\s*\(\)|frame\.view\s*\(\)|camera_view|light_view|view_mat|model_view)"
    ),
}

# Tuple: relative file, enclosing function, operation, occurrence in function.
# Every duplicate is distinct; adding a second producer in an allowlisted file
# or function changes this inventory and fails.
EXPECTED = {
    ("src/shadows/csm_renderer.rs", "compute_cascade", "matrix_compose", 1),
    ("src/shadows/csm_renderer.rs", "compute_cascade", "projection_ctor", 1),
    ("src/shadows/csm_renderer.rs", "update_cascades", "matrix_compose", 1),
    ("src/viewer/camera_controller.rs", "view_matrix", "anchor_view", 1),
    ("src/viewer/camera_controller.rs", "view_matrix", "anchor_view", 2),
    ("src/viewer/camera_controller.rs", "view_matrix", "controller_view_delegate", 1),
    ("src/viewer/camera_controller.rs", "view_matrix", "controller_view_delegate", 2),
    ("src/viewer/input/viewer_input.rs", "handle_input", "frame_view", 1),
    ("src/viewer/input/viewer_input.rs", "handle_input", "matrix_compose", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase", "frame_view", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase", "matrix_compose", 1),
    ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame", "frame_view", 1),
    ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame", "frame_view_projection", 1),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "frame_view", 1),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "matrix_compose", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "frame_view", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "matrix_compose", 1),
    ("src/viewer/render/main_loop/secondary.rs", "render_secondary_paths", "frame_view_projection", 1),
    ("src/viewer/render/main_loop/secondary.rs", "render_secondary_paths", "frame_view_projection", 2),
    ("src/viewer/state/labels.rs", "update_labels", "frame_view_projection", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "frame_view", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "matrix_compose", 1),
    ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state", "frame_view", 1),
    ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state", "matrix_compose", 1),
    ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state", "frame_view", 1),
    ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state", "matrix_compose", 1),
    ("src/viewer/viewer_types.rs", "projection", "projection_ctor", 1),
    ("src/viewer/viewer_types.rs", "view", "anchor_view", 1),
    ("src/viewer/viewer_types.rs", "view_projection", "matrix_compose", 1),
}

NO_MATRIX_FILES = {
    "src/viewer/terrain/dof/pass/execute.rs",
    "src/viewer/terrain/overlay/stack/composite.rs",
    "src/viewer/terrain/volume_density.rs",
}


def _strip(text: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


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


def _inventory_text(rel: str, raw: str):
    text = _strip(raw)
    spans = _function_spans(text)
    sites = []
    counters = {}
    matches = []
    for operation, pattern in PATTERNS.items():
        matches.extend((match.start(), operation) for match in pattern.finditer(text))
    for position, operation in sorted(matches):
        function = next(
            (name for start, end, name in spans if start <= position < end),
            "<module>",
        )
        key = (function, operation)
        counters[key] = counters.get(key, 0) + 1
        sites.append((rel, function, operation, counters[key]))
    return sites


def inventory_sites():
    sites = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.rs")):
            rel = path.relative_to(ROOT).as_posix()
            sites.extend(_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


def test_matrix_producer_inventory_is_exact_by_callsite_and_occurrence():
    actual = set(inventory_sites())
    assert actual == EXPECTED, (
        f"missing={sorted(EXPECTED - actual)}\n"
        f"unexpected={sorted(actual - EXPECTED)}\n"
        f"actual={sorted(actual)}"
    )


def test_multiline_and_duplicate_producers_are_detected():
    probe = """
    fn render(frame: FrameCamera, proj: Mat4, view_mat: Mat4) {
        let a = proj *\n view_mat;
        let b = proj * view_mat;
        let c = frame.view();
    }
    """
    sites = _inventory_text("probe.rs", probe)
    assert ("probe.rs", "render", "matrix_compose", 1) in sites
    assert ("probe.rs", "render", "matrix_compose", 2) in sites
    assert ("probe.rs", "render", "frame_view", 1) in sites


def test_direct_look_at_is_owned_only_by_anchor_module():
    paths = [ROOT / "src/camera/mod.rs"]
    for root in SCAN_ROOTS:
        paths.extend(sorted(root.rglob("*.rs")))
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel == "src/camera/anchor.rs":
            continue
        text = _strip(path.read_text(encoding="utf-8"))
        assert not PATTERNS["look_at"].search(text), f"direct look-at producer in {rel}"


def test_declared_no_matrix_consumers_remain_matrix_free():
    for rel in NO_MATRIX_FILES:
        text = _strip((ROOT / rel).read_text(encoding="utf-8"))
        for operation, pattern in PATTERNS.items():
            assert not pattern.search(text), f"{rel} gained {operation}"
