"""Fail-closed inventory for viewer Anchor ownership and rebase mutation."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ANCHOR_INVENTORY = {
    ("src/camera/mod.rs", "anchored_view", "constructor", 1),
    ("src/camera/mod.rs", "anchored_view", "rebase_call", 1),
    ("src/labels/mod.rs", "update_with_camera", "constructor", 1),
    ("src/picking/terrain_query.rs", "<module>", "storage", 1),
    ("src/picking/terrain_query.rs", "<module>", "storage", 2),
    ("src/picking/terrain_query.rs", "<module>", "storage", 3),
    ("src/picking/terrain_query.rs", "default", "constructor", 1),
    ("src/picking/terrain_query.rs", "query_ray_heightfield", "storage", 1),
    ("src/pointcloud/renderer.rs", "default_anchor", "constructor", 1),
    ("src/pointcloud/renderer.rs", "default_anchor", "rebase_call", 1),
    ("src/py_types/pointcloud.rs", "create_gpu_buffer", "constructor", 1),
    ("src/py_types/pointcloud.rs", "create_gpu_buffer", "rebase_call", 1),
    ("src/scene/core/constructor.rs", "new_impl", "constructor", 1),
    ("src/scene/mod.rs", "<module>", "storage", 1),
    ("src/scene/py_api/base.rs", "set_camera_look_at", "rebase_call", 1),
    ("src/tiles3d/renderer.rs", "add_buildings", "constructor", 1),
    ("src/tiles3d/renderer.rs", "add_buildings", "rebase_call", 1),
    ("src/tiles3d/renderer.rs", "from_buildings", "constructor", 1),
    ("src/tiles3d/renderer.rs", "from_buildings", "rebase_call", 1),
    ("src/viewer/camera_controller.rs", "prospective_anchor", "anchor_copy", 1),
    ("src/viewer/camera_controller.rs", "prospective_anchor", "rebase_call", 1),
    ("src/viewer/event_loop/command_preflight.rs", "preflight_command_batch", "anchor_copy", 1),
    ("src/viewer/event_loop/command_preflight.rs", "preflight_command_batch", "rebase_call", 1),
    ("src/viewer/init/viewer_new.rs", "new", "constructor", 1),
    ("src/viewer/input/viewer_input.rs", "handle_input", "anchor_copy", 1),
    ("src/viewer/input/viewer_input.rs", "handle_input", "anchor_copy", 2),
    ("src/viewer/input/viewer_input.rs", "handle_input", "anchor_copy", 3),
    ("src/viewer/input/viewer_input.rs", "handle_input", "anchor_copy", 4),
    ("src/viewer/input/viewer_input.rs", "handle_input", "anchor_copy", 5),
    ("src/viewer/input/viewer_input.rs", "handle_input", "anchor_copy", 6),
    ("src/viewer/input/viewer_input.rs", "update", "anchor_copy", 1),
    ("src/viewer/input/viewer_input.rs", "update", "anchor_copy", 2),
    ("src/viewer/input/viewer_input.rs", "update", "anchor_copy", 3),
    ("src/viewer/pointcloud/state.rs", "load_from_file", "anchor_copy", 1),
    ("src/viewer/pointcloud/state.rs", "load_from_file", "rebase_call", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "prepare_frame_anchor", "rebase_call", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "prospective_frame_camera", "rebase_call", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "selected_frame_camera", "anchor_copy", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "selected_frame_camera", "anchor_copy", 2),
    ("src/viewer/render/main_loop/frame_anchor.rs", "selected_frame_camera", "anchor_copy", 3),
    ("src/viewer/scene_review.rs", "validate_scene_review_effective", "anchor_copy", 1),
    ("src/viewer/scene_review.rs", "validate_scene_review_effective", "anchor_copy", 2),
    ("src/viewer/scene_review.rs", "validate_scene_review_effective", "rebase_call", 1),
    ("src/viewer/viewer_struct.rs", "<module>", "storage", 1),
    ("src/viewer/viewer_types.rs", "<module>", "storage", 1),
}

LOCAL_NON_VIEWER_JUSTIFICATIONS = {
    ("src/camera/mod.rs", "anchored_view"): "stateless public camera helper",
    ("src/labels/mod.rs", "update_with_camera"): "legacy local-label compatibility entry",
    ("src/picking/terrain_query.rs", "<module>"): "query results retain their producing anchor snapshot",
    ("src/picking/terrain_query.rs", "default"): "empty compatibility query result",
    ("src/picking/terrain_query.rs", "query_ray_heightfield"): "caller-supplied producing anchor snapshot",
    ("src/pointcloud/renderer.rs", "default_anchor"): "standalone point-buffer helper",
    ("src/py_types/pointcloud.rs", "create_gpu_buffer"): "caller-supplied standalone point-buffer anchor",
    ("src/scene/core/constructor.rs", "new_impl"): "offscreen Scene owns its separate signed ABI",
    ("src/scene/mod.rs", "<module>"): "offscreen Scene persistent anchor",
    ("src/tiles3d/renderer.rs", "add_buildings"): "standalone CityJSON renderer helper",
    ("src/tiles3d/renderer.rs", "from_buildings"): "standalone CityJSON renderer helper",
}


def _strip(text: str) -> str:
    return re.sub(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        lambda match: " " * len(match.group(0)),
        text,
        flags=re.S,
    )


def _production_text(text: str) -> str:
    text = _strip(text)
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


def _inventory_text(rel: str, raw: str):
    text = _production_text(raw)
    spans = _function_spans(text)
    patterns = {
        "constructor": re.compile(
            r"\b(?:crate\s*::\s*camera\s*::\s*)?Anchor\s*::\s*(?:new|with_epsilon|try_with_epsilon)\s*\("
        ),
        "storage": re.compile(
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*:\s*(?!&)(?:crate\s*::\s*camera\s*::\s*)?Anchor\b(?!\s*::)"
        ),
        "anchor_copy": re.compile(
            r"\b(?:"
            r"let\s+(?:mut\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:\*\s*)?"
            r"(?:[A-Za-z_][A-Za-z0-9_]*anchor[A-Za-z0-9_]*|frame\s*\.\s*anchor|"
            r"self\s*\.\s*camera_anchor|self\s*\.\s*prospective_frame_camera\s*\(\s*\)\s*\.\s*anchor)"
            r"\s*;|[A-Za-z_][A-Za-z0-9_]*\s*=\s*self\s*\.\s*camera_anchor\s*;"
            r"|anchor\s*:\s*self\s*\.\s*camera_anchor\s*[,}])"
        ),
        "rebase_call": re.compile(r"\.\s*rebase_if_needed\s*\("),
        "indirect_live_mutation": re.compile(
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*&\s*mut\s+self\s*\.\s*camera_anchor"
        ),
    }
    matches = []
    for operation, pattern in patterns.items():
        matches.extend((match.start(), operation) for match in pattern.finditer(text))
    for start, end, _function in spans:
        body = text[start:end]
        anchor_refs = re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*&\s*(?:crate\s*::\s*camera\s*::\s*)?Anchor\b",
            body,
        )
        for name in anchor_refs:
            pattern = re.compile(
                rf"\blet\s+(?:mut\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*\*\s*{re.escape(name)}\b"
            )
            matches.extend((start + match.start(), "anchor_copy") for match in pattern.finditer(body))
    counters = {}
    sites = []
    for position, operation in sorted(matches):
        function = next(
            (name for start, end, name in spans if start <= position < end),
            "<module>",
        )
        key = (function, operation)
        counters[key] = counters.get(key, 0) + 1
        sites.append((rel, function, operation, counters[key]))
    return sites


def anchor_inventory():
    sites = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        sites.extend(_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_anchor_storage_construction_copy_and_rebase_inventory_is_exact():
    actual = set(anchor_inventory())
    assert actual == EXPECTED_ANCHOR_INVENTORY, (
        f"missing={sorted(EXPECTED_ANCHOR_INVENTORY - actual)}\n"
        f"unexpected={sorted(actual - EXPECTED_ANCHOR_INVENTORY)}\n"
        f"actual={sorted(actual)}"
    )


def test_viewer_has_one_persistent_anchor_one_frozen_copy_and_one_live_mutation():
    viewer_sites = [site for site in anchor_inventory() if site[0].startswith("src/viewer/")]
    assert (
        "src/viewer/viewer_struct.rs",
        "<module>",
        "storage",
        1,
    ) in viewer_sites
    assert (
        "src/viewer/viewer_types.rs",
        "<module>",
        "storage",
        1,
    ) in viewer_sites

    live = []
    for path in (ROOT / "src/viewer").rglob("*.rs"):
        rel = path.relative_to(ROOT).as_posix()
        text = _production_text(path.read_text(encoding="utf-8"))
        for match in re.finditer(
            r"self\s*\.\s*camera_anchor\s*\.\s*rebase_if_needed\s*\(", text
        ):
            function = next(
                (name for start, end, name in _function_spans(text) if start <= match.start() < end),
                "<module>",
            )
            live.append((rel, function))
    assert live == [("src/viewer/render/main_loop/frame_anchor.rs", "prepare_frame_anchor")]


def test_every_non_viewer_constructor_or_storage_site_is_justified():
    owned = {
        (rel, function)
        for rel, function, operation, _ordinal in EXPECTED_ANCHOR_INVENTORY
        if operation in {"constructor", "storage"} and not rel.startswith("src/viewer/")
    }
    assert owned == set(LOCAL_NON_VIEWER_JUSTIFICATIONS)
    manual = _src("src/viewer/state/labels.rs")
    assert "let frame = self.current_frame_camera()" in manual
    assert "frame.view_projection" in manual
    assert ".update_with_camera(" not in manual


def test_alias_local_constructor_and_indirect_mutation_probes_are_rejected():
    probes = [
        "fn f(anchor: &mut Anchor) { anchor.rebase_if_needed(focus); }",
        "fn f() { let anchor = Anchor::new(); }",
        "fn f(&mut self) { rebase_anchor(&mut self.camera_anchor, focus); }",
    ]
    for probe in probes:
        assert _inventory_text("probe.rs", probe), probe


def test_frame_anchor_is_prepared_before_any_render_state_or_pass():
    main_loop = _src("src/viewer/render/main_loop.rs")
    body = main_loop.split("pub fn render", 1)[1]
    prepare = body.index("self.prepare_frame_anchor()")
    assert prepare < body.index("self.prepare_render_frame()")
    assert prepare < body.index("self.render_geometry_stage")


def test_screen_snapshot_and_motion_blur_receive_the_copied_frame():
    secondary = _src("src/viewer/render/main_loop/secondary.rs")
    assert "let frame = self.current_frame_camera()" in secondary
    assert "render_to_texture" in secondary
    motion = _src("src/viewer/terrain/render/motion_blur.rs")
    assert "frame: crate::viewer::viewer_types::FrameCamera" in motion
    assert "frame.with_pose" in motion
    assert "rebase_if_needed" not in motion
