"""Option-2 absolute-world boundary and historical-contract regression gates."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_historical_option_one_contract_is_absent_from_production_and_tests():
    forbidden = [
        "VIEWER_" + "LOCAL_FRAME_MAX_COORD",
        "coord_within_" + "local_frame",
        "sanitize_terrain_" + "target",
    ]
    paths = list((ROOT / "src").rglob("*.rs")) + list((ROOT / "tests").glob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for symbol in forbidden:
            assert symbol not in text, f"historical Option-1 symbol {symbol} in {path}"


def test_option_two_accepts_absolute_world_and_bounds_only_anchor_residual():
    camera = _read("src/viewer/camera_controller.rs")
    assert "pub const VIEWER_RENDER_FRAME_MAX_COORD: f64 = 1_000_000.0" in camera
    assert "let residual = (v - anchor.origin()).abs().max_element()" in camera
    assert "pub target: DVec3" in camera
    assert "pub position: DVec3" in camera
    assert "set_look_at_accepts_earth_scale_world_coordinates" in camera
    assert "render_frame_bound_is_relative_to_prospective_anchor" in camera


def test_one_prospective_frame_validates_all_content_points():
    frame = _read("src/viewer/render/main_loop/frame_anchor.rs")
    body = frame.split("pub(crate) fn validate_content_points", 1)[1].split(
        "/// Select the one camera pose", 1
    )[0]
    assert body.count("let frame = self.prospective_frame_camera()") == 1
    assert "for point in points" in body
    assert "&frame.anchor" in body
    assert "rebase_if_needed" not in body


def test_absolute_sources_remain_f64_until_anchor_packing():
    assert "pub position: DVec3" in _read("src/viewer/pointcloud/types.rs")
    assert "pub source_vertices: Vec<VectorSourceVertex>" in _read(
        "src/viewer/terrain/vector_overlay.rs"
    )
    labels = _read("src/labels/types.rs")
    assert "pub world_pos: DVec3" in labels
    assert "pub polyline: Vec<DVec3>" in labels
    city = _read("src/import/cityjson/types.rs")
    assert "pub positions: Vec<f64>" in city
    city_geometry = _read("src/import/cityjson/geometry.rs")
    assert "direction_to_render" in city_geometry
    assert "point[0] as f32" not in city_geometry


def test_object_vertices_stay_local_and_translation_is_anchored_once():
    command = _read("src/viewer/cmd/ipc_command.rs")
    assert "viewer.object_translation = candidate_translation" in command
    assert "prospective_anchor(&viewer.camera_anchor, candidate_translation)" in command
    assert "validate_world_point(CoordRole::Object, candidate_translation, &validation_anchor)" in command
    assert "transformed_positions" not in command
    assert "transform_point3" not in command
    frame = _read("src/viewer/render/main_loop/frame_anchor.rs")
    assert "frame.anchor.model_offset(self.object_translation)" in frame
    geometry = _read("src/viewer/render/main_loop/geometry/pass.rs")
    assert geometry.count("anchored_object_model(frame)") == 1


def test_object_preflight_ignores_unloaded_identity_placeholder():
    preflight = _read("src/viewer/event_loop/command_preflight.rs")
    assert "let object_present = !self.object_source_positions.is_empty()" in preflight
    assert "if object_present {\n            validate_points(&anchor, CoordRole::Object, [object_translation])?;" in preflight
    assert "} else if object_transform_requested {\n            object_translation" in preflight
    assert "else if !(object_transform_requested && requested_general_pose.is_none())" in preflight


def test_frame_camera_precedence_is_full_pose_not_anchor_only():
    frame = _read("src/viewer/render/main_loop/frame_anchor.rs")
    terrain = frame.index("kind: ActiveCameraKind::Terrain")
    point_cloud = frame.index("kind: ActiveCameraKind::PointCloud")
    general = frame.index("kind: ActiveCameraKind::General")
    assert terrain < point_cloud < general
    assert "current_frame_camera" in _read("src/viewer/input/viewer_input.rs")
    assert "current_frame_camera" in _read("src/viewer/render/main_loop/secondary.rs")


def test_mouse_pick_ray_uses_render_frame_not_object_local_model():
    source = _read("src/viewer/input/viewer_input.rs")
    picking = source.split("pub(crate) fn pick_at_screen", 1)[1].split(
        "pub fn handle_input", 1
    )[0]
    assert "let frame = self.current_frame_camera()" in picking
    assert "frame.projection(self.config.width, self.config.height) * frame.view()" in picking
    assert "anchored_object_model" not in picking
