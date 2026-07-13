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


def test_interactive_viewer_is_a_local_frame_not_absolute_world():
    # The viewer never holds absolute geospatial coordinates: its orbit camera
    # is distance-bounded (a local frame), and the viewer terrain path contains
    # no ECEF/geodetic conversion. If a future change introduces absolute
    # coordinates here, it must anchor them and this guard should be revisited.
    cam = _src("viewer/camera_controller.rs")
    assert "clamp(0.1, 1000.0)" in cam, "orbit camera is no longer a bounded local frame"
    # No absolute-Earth-scale geodesy in the viewer terrain render path.
    forbidden = re.compile(r"\b(6_?378_?137|wgs84_to_ecef|geodetic_to_ecef)\b", re.IGNORECASE)
    for rel in ("viewer/terrain/scene.rs", "viewer/cmd/terrain_command.rs",
                "viewer/camera_controller.rs"):
        assert not forbidden.search(_src(rel)), f"absolute geodesy leaked into {rel}"
    # The whole boundary rests on the single narrowing site (its uniqueness is
    # locked by test_world_coord_f32_gate.py); confirm the sanctioned helper.
    assert "fn narrow(value: f64) -> f32" in _src("camera/anchor.rs")
