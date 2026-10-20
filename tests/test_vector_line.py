"""Analytical contracts for the bounded local/viewer line helper."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import forge3d
from forge3d import VectorLineLayer, VectorLineStyleError
from forge3d.style import parse_style, validate_style_support, vector_overlay_configs_from_style
from forge3d.viewer import ViewerHandle


PATH = [(0, 0, 0), (10, 0, 0), (10, 0, 10)]


def triangles(config, color):
    xyz = np.array([[v.x, v.z] for v in config.vertices])
    selected = []
    for a, b, c in np.array(config.indices).reshape(-1, 3):
        if tuple(config.vertices[a].to_array()[3:7]) == color:
            selected.append(xyz[[a, b, c]])
    return np.array(selected)


def area(mesh):
    a, b = mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2


def coverage(mesh, point):
    # Strict interior counting avoids double-counting shared triangle boundaries.
    edge = np.roll(mesh, -1, axis=1) - mesh
    delta = np.array(point) - mesh
    cross = edge[..., 0] * delta[..., 1] - edge[..., 1] * delta[..., 0]
    return int(((cross > 0).all(axis=1) | (cross < 0).all(axis=1)).sum())


@pytest.mark.parametrize("cap,expected", [("butt", 20), ("square", 24)])
def test_width_caps_and_disjoint_halo(cap, expected):
    line = VectorLineLayer("road", PATH[:2], width=2, halo=1, cap=cap)
    config = line.to_overlay_config()
    core = triangles(config, line.color)
    border = triangles(config, line.halo_color)
    assert area(core) == pytest.approx(expected)
    assert area(border) == pytest.approx(4 * (12 if cap == "butt" else 14) - expected)
    assert core[..., 1].min() == -1
    assert core[..., 1].max() == 1
    assert border[..., 1].min() == -2
    assert border[..., 1].max() == 2
    assert coverage(core, (5.13, .23)) == 1
    assert coverage(border, (5.13, .23)) == 0
    assert coverage(border, (5.13, 1.23)) == 1
    assert coverage(core, (-.47, .17)) == (1 if cap == "square" else 0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("join,expected", [("miter", 40), ("bevel", 39.5)])
def test_join_geometry_and_no_overlap(join, expected, reverse):
    line = VectorLineLayer("road", PATH[::-1] if reverse else PATH, width=2, halo=1, join=join)
    config = line.to_overlay_config()
    core, border = triangles(config, line.color), triangles(config, line.halo_color)
    assert area(core) == pytest.approx(expected)
    assert area(core) + area(border) == pytest.approx(88 if join == "miter" else 86)
    assert coverage(core, (10.83, -.81)) == (1 if join == "miter" else 0)
    for point in [(9.37, .23), (10.13, .21), (9.81, -.63), (10.71, 1.13)]:
        assert coverage(core, point) == 1
        assert coverage(border, point) == 0


@pytest.mark.parametrize("turn", [150, 165])
@pytest.mark.parametrize("reverse", [False, True])
def test_miter_clamp_applies_to_stroke_and_halo(turn, reverse):
    angle = math.radians(turn)
    points = [(-1000, 0, 0), (0, 0, 0), (1000 * math.cos(angle), 0, 1000 * math.sin(angle))]
    line = VectorLineLayer("road", points[::-1] if reverse else points, width=2, halo=1)
    config = line.to_overlay_config()
    core, border = triangles(config, line.color), triangles(config, line.halo_color)
    direction = np.array([-math.sin(angle / 2), math.cos(angle / 2)])
    # Analytical bisector length: unchanged below ratio 4, clamped above it.
    ratio = min(1 / math.cos(angle / 2), 4)
    for mesh, radius in ((core, 1), (border, 2)):
        corners = mesh.reshape(-1, 2)
        corners = corners[np.linalg.norm(corners, axis=1) < 10]
        for sign in (-1, 1):
            expected = sign * direction * radius * ratio
            assert np.min(np.linalg.norm(corners - expected, axis=1)) == pytest.approx(0, abs=1e-12)
        assert np.linalg.norm(corners, axis=1).max() == pytest.approx(radius * ratio)
    # At a clamped corner, the halo still surrounds a disjoint stroke.
    for distance, stroke_count, halo_count in ((.5 * ratio, 1, 0), (1.5 * ratio, 0, 1)):
        point = direction * distance + np.array([.01, .01])
        assert coverage(core, point) == stroke_count
        assert coverage(border, point) == halo_count


@pytest.mark.parametrize("drape", [False, True])
def test_display_height_offset_and_native_drape_contract(drape):
    points = [(0, 3, 0), (10, 5, 0)]
    config = VectorLineLayer("road", points, drape=drape, z_offset=-.75).to_overlay_config()
    assert {v.y for v in config.vertices} == ({3, 5} if drape else {2.25, 4.25})
    assert config.drape is drape
    assert config.drape_offset == (-.75 if drape else 0)
    assert config.to_ipc_dict()["primitive"] == "triangles"
    assert config.depth_bias == .1  # Existing VectorOverlayConfig default.


@pytest.mark.parametrize("field,value", [
    ("width", 0), ("width", .09), ("width", float("nan")),
    ("halo", -1), ("halo", float("inf")), ("z_offset", float("nan")),
    ("opacity", -1), ("opacity", 2), ("opacity", float("nan")),
    ("drape", 1), ("color", (1, 0, float("nan"), 1)),
    ("halo_color", (1, 1, 1, 2)), ("feature_id", True),
    ("feature_id", 1.5), ("feature_id", 2**32), ("name", ""),
])
def test_invalid_inputs(field, value):
    values = {"name": "road", "points": PATH, field: value}
    with pytest.raises(ValueError):
        VectorLineLayer(**values).to_overlay_config()


@pytest.mark.parametrize("points", [
    [], [(0, 0, 0)], [(0, 0, 0), (0, 1, 0)],
    [(0, 0, 0), (float("inf"), 0, 0)], [(0, 0), (1, 1)],
    [(0, 0, 0), (10, 0, 0), (5, 0, 0)],
    [(0, 0, 0), (10, 0, 0), (0, 0, 0)],
    [(0, 0, 0), (10, 0, 10), (0, 0, 10), (10, 0, 0)],
])
def test_invalid_paths(points):
    with pytest.raises(ValueError):
        VectorLineLayer("road", points).to_overlay_config()


def test_frozen_inputs_and_exact_feature_id():
    points = [list(p) for p in PATH]
    line = VectorLineLayer("road", points, feature_id=2**32 - 1)
    points[0][0] = 100
    assert line.points[0][0] == 0
    assert all(v.feature_id == 2**32 - 1 for v in line.to_overlay_config().vertices)


def test_public_submission_and_no_ipc_on_invalid_geometry(monkeypatch):
    viewer = ViewerHandle.__new__(ViewerHandle)
    commands = []
    def submit(command):
        commands.append(command)
        return {"ok": True, "id": command["id"]}
    monkeypatch.setattr(viewer, "_send_command", submit)
    assert viewer.add_vector_line(VectorLineLayer("road", PATH, halo=1)) == 1
    assert commands[0]["primitive"] == "triangles"
    assert commands[0]["drape"] is True
    with pytest.raises(ValueError):
        viewer.add_vector_line(VectorLineLayer("bad", [(0, 0, 0), (0, 0, 0)]))
    assert len(commands) == 1


@pytest.mark.parametrize("section,field,value", [
    ("paint", "line-dasharray", [1, 2]), ("paint", "line-gradient", "red"),
    ("layout", "line-cap", "round"), ("layout", "line-join", "round"),
    ("layout", "line-sort-key", 1),
    ("layout", "line-miter-limit", 2),
])
@pytest.mark.parametrize("parsed", [False, True])
def test_style_diagnostics_before_geometry(section, field, value, parsed):
    layer = {"id": "road", "type": "line", section: {field: value}}
    if parsed:
        layer = parse_style({"version": 8, "layers": [layer]}).layers[0]
    with pytest.raises(VectorLineStyleError) as caught:
        VectorLineLayer.from_style(layer, PATH)
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "unsupported_style_field"
    assert diagnostic.layer_id == "road"
    assert diagnostic.details["section"] == section
    assert any(field in name for name in diagnostic.details["fields"])


def test_style_expression_context_and_defaults():
    layer = {"id": "road", "type": "line", "paint": {
        "line-color": "#ff0000", "line-width": ["get", "width"], "line-opacity": .5,
    }, "layout": {"line-cap": "square", "line-join": "miter"}}
    line = VectorLineLayer.from_style(layer, PATH, properties={"width": 4}, halo=1)
    assert (line.width, line.color, line.opacity, line.cap, line.join) == (4, (1, 0, 0, 1), .5, "square", "miter")
    default = VectorLineLayer.from_style({"id": "road", "type": "line"}, PATH)
    assert (default.width, default.cap, default.join) == (1, "butt", "miter")
    assert VectorLineLayer("road", PATH).join == "miter"
    with pytest.raises(ValueError, match="cannot evaluate"):
        VectorLineLayer.from_style(layer, PATH)


@pytest.mark.parametrize("paint", [
    {"line-width": True}, {"line-width": float("nan")},
    {"line-opacity": float("inf")}, {"line-color": "not-a-color"},
])
def test_invalid_style_values(paint):
    with pytest.raises(ValueError):
        VectorLineLayer.from_style({"id": "road", "type": "line", "paint": paint}, PATH)


def test_width_that_folds_a_short_corner_is_rejected():
    with pytest.raises(ValueError, match="crossing or folded"):
        VectorLineLayer("road", [(0, 0, 0), (1, 0, 0), (1, 0, 1)], width=6).to_overlay_config()


@pytest.mark.parametrize("spacing,folds", [(2, True), (5, False)])
def test_densifying_terrain_path_requires_room_for_halo_at_corner(spacing, folds):
    original = np.array([(25, 0, 25), (70, 0, 25), (100, 0, 95)], dtype=float)
    dense = [original[0]]
    for a, b in zip(original, original[1:]):
        segments = math.ceil(np.linalg.norm((b - a)[[0, 2]]) / spacing)
        dense.extend(a + (b - a) * i / segments for i in range(1, segments + 1))
    line = VectorLineLayer("road", dense, width=6, halo=2, cap="square", join="bevel")
    if folds:
        with pytest.raises(ValueError, match="crossing or folded"):
            line.to_overlay_config()
    else:
        assert line.to_overlay_config().vertex_count > 0


def test_existing_translation_defaults_and_diagnostics_unchanged():
    style = {"version": 8, "layers": [{"id": "road", "type": "line", "paint": {"line-width": 3}}]}
    feature = {"geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 10]]}}
    old = vector_overlay_configs_from_style(style, [feature])[0]
    assert old.primitive.value == "lines"
    assert old.line_width == 3
    assert old.drape is False
    style["layers"][0]["paint"]["line-gradient"] = "red"
    assert validate_style_support(style).diagnostics[0].code == "unsupported_style_field"


def test_exports_and_stub_contract():
    assert forge3d.VectorLineLayer is VectorLineLayer
    assert {"VectorLineLayer", "VectorLineStyleError"} <= set(forge3d.__all__)
    root = Path(forge3d.__file__).parent
    stub = ast.parse((root / "vector_line.pyi").read_text())
    assert any(isinstance(n, ast.ClassDef) and n.name == "VectorLineLayer" for n in stub.body)
    viewer = ast.parse((root / "viewer.pyi").read_text())
    handle = next(n for n in viewer.body if isinstance(n, ast.ClassDef) and n.name == "ViewerHandle")
    assert any(isinstance(n, ast.FunctionDef) and n.name == "add_vector_line" for n in handle.body)
