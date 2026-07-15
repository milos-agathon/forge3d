"""Source lock for the viewer-owned anchor and once-per-frame rebase boundary."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", text)


def test_viewer_anchor_has_one_live_mutating_rebase_site():
    sites = []
    for path in (ROOT / "src/viewer").rglob("*.rs"):
        text = _normalized(path.read_text(encoding="utf-8"))
        count = len(re.findall(r"self\s*\.\s*camera_anchor\s*\.\s*rebase_if_needed\s*\(", text))
        sites.extend([path.relative_to(ROOT).as_posix()] * count)
    assert sites == ["src/viewer/render/main_loop/frame_anchor.rs"]


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


def test_duplicate_rebase_probe_is_counted():
    probe = "self.camera_anchor.rebase_if_needed(focus);\nself.camera_anchor\n .rebase_if_needed(focus);"
    normalized = _normalized(probe)
    assert len(re.findall(r"self\s*\.\s*camera_anchor\s*\.\s*rebase_if_needed", normalized)) == 2
