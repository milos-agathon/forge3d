# tests/test_m06_anchoring_boundary.py
# MENSURA M-06 boundary lock. The acceptance is: every ABSOLUTE geospatial
# coordinate (ECEF / projected UTM / Web Mercator, magnitudes 5e5..6.4e6 m,
# where f32 costs 0.03-1 m) stays f64 until it passes through the single
# Anchor::narrow site. This test encodes the audited boundary so a regression
# (a NEW absolute-world f32 path, or an anchored path silently de-anchored)
# fails loudly. It complements the textual `as f32` gate in
# tests/test_world_coord_f32_gate.py.
#
# Key finding (verified): the interactive viewer and the standalone vector
# renderer are NOT absolute-world paths — they render in a terrain-local /
# normalized / clip frame where f32 is correct. Only Scene, 3D-Tiles, point
# clouds, and CityJSON carry absolute coordinates, and those are anchored.
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / "src" / rel).read_text(encoding="utf-8")


def test_absolute_geospatial_paths_route_through_the_anchor():
    # Each path that holds absolute ECEF/projected coordinates must narrow via
    # the Anchor (f64 origin -> Anchor::to_render_*), never a bare cast.
    assert "create_gpu_buffer_anchored" in _src("pointcloud/renderer.rs")
    assert "anchor.to_render_vec3" in _src("pointcloud/renderer.rs")
    assert "render_positions" in _src("tiles3d/pnts.rs")
    assert "anchor.to_render_vec3" in _src("tiles3d/pnts.rs")
    # The offscreen Scene anchors both its camera and its object model matrices.
    scene = _src("scene/py_api/base.rs")
    assert "camera_anchor" in scene or "Anchor" in scene
    # tiles3d bounding volumes carry f64 (DVec3/[f64;N]) end to end.
    bounds = _src("tiles3d/bounds.rs")
    assert "Height<Ellipsoidal>" in bounds and "DVec3" in bounds


def test_cityjson_keeps_an_f64_origin_before_narrowing():
    # CityJSON decode keeps the CRS translate in f64 and tessellates relative to
    # an f64 origin (sub(vertex, origin)), so absolute easting/northing never
    # lands directly in f32.
    geom = _src("import/cityjson/geometry.rs")
    assert "origin: [f64; 3]" in geom
    assert re.search(r"sub\(\s*ring\[[ij]\]\s*,\s*origin\s*\)", geom)


def test_viewer_terrain_render_path_does_no_absolute_earth_scale_geodesy():
    # HONEST SCOPE: this asserts only that the viewer's terrain render path never
    # *derives* absolute ECEF/geodetic world coordinates itself. It is NOT a proof
    # of full anchoring: the viewer's CAMERA path is now a Rust contract (see the
    # test below), but the overlay/label/transform IPC world fields remain f32 and
    # rely on the Python-side normalized/terrain-local convention. Widening those
    # to f64 + a viewer anchor is the remaining M-06 residual (option 2 in
    # mensura-m06-world-coord-anchoring.md).
    forbidden = re.compile(r"\b(6_?378_?137|wgs84_to_ecef|geodetic_to_ecef)\b", re.IGNORECASE)
    for rel in ("viewer/terrain/scene.rs", "viewer/cmd/terrain_command.rs",
                "viewer/camera_controller.rs"):
        assert not forbidden.search(_src(rel)), f"absolute geodesy leaked into {rel}"
    # The whole boundary rests on the single narrowing site (its uniqueness is
    # locked by test_world_coord_f32_gate.py); confirm the sanctioned helper.
    assert "fn narrow(value: f64) -> f32" in _src("camera/anchor.rs")


def test_viewer_camera_enforces_local_frame_contract():
    # M-06 (option 1): the interactive viewer's camera setters no longer accept an
    # arbitrary f32 world target. `set_look_at` / `set_orbit_pose_target` validate
    # every eye/target component against VIEWER_LOCAL_FRAME_MAX_COORD and reject
    # (without mutating state) any absolute geospatial coordinate — turning the
    # former Python-side convention into a Rust contract. The behavioural proof is
    # the Rust unit tests in `src/viewer/camera_controller.rs`
    # (`camera_controller::tests`); this locks that the enforcement is wired and
    # cannot be silently removed.
    cam = _src("viewer/camera_controller.rs")
    assert "pub const VIEWER_LOCAL_FRAME_MAX_COORD" in cam
    # The public setters return a Result carrying the contract error, not `()`.
    assert re.search(r"fn set_look_at\([^)]*\)\s*->\s*Result<\(\),\s*CameraFrameError>", cam, re.S)
    assert re.search(r"fn set_orbit_pose_target\([^)]*\)\s*->\s*Result<\(\),\s*CameraFrameError>", cam, re.S)
    assert "validate_local_frame(CoordRole::Eye" in cam and "validate_local_frame(CoordRole::Target" in cam
    # The external IPC boundary handles the rejection rather than ignoring it.
    ipc = _src("viewer/cmd/ipc_command.rs")
    assert "set_look_at(eye, target, up)" in ipc
    assert "SetCamLookAt rejected" in ipc
    # The *terrain* orbit target is a second camera world-coordinate entry; it is
    # guarded through the same predicate so the contract cannot be bypassed.
    terrain = _src("viewer/cmd/terrain_command.rs")
    assert "coord_within_local_frame" in terrain
    assert "sanitize_terrain_target" in terrain
