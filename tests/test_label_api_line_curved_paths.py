import math

from forge3d.viewer import LabelOperationResult, ViewerHandle


def _fake_viewer():
    viewer = object.__new__(ViewerHandle)
    commands = []

    def fake_send(cmd):
        commands.append(cmd)
        if cmd["cmd"] == "add_line_label":
            return {"ok": True, "id": cmd["id"]}
        return {"ok": True}

    viewer._send_command = fake_send
    return viewer, commands


def _glyphs_for(viewer, label_id):
    state = viewer.label_configuration_state()
    return state["line_label_glyph_instances"][str(label_id)]


def test_line_label_fixtures_emit_ordered_glyph_instances_with_tangent_rotation():
    viewer, commands = _fake_viewer()

    horizontal_id = viewer.add_line_label("AB", [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)])
    vertical_id = viewer.add_line_label("C", [(0.0, 0.0, 0.0), (0.0, 0.0, 100.0)])
    diagonal_id = viewer.add_line_label("D", [(0.0, 0.0, 0.0), (100.0, 0.0, 100.0)])

    horizontal = _glyphs_for(viewer, horizontal_id)
    vertical = _glyphs_for(viewer, vertical_id)
    diagonal = _glyphs_for(viewer, diagonal_id)

    assert [glyph["glyph"] for glyph in horizontal] == ["A", "B"]
    assert [glyph["ordering_key"] for glyph in horizontal] == [
        f"{horizontal_id}:0000",
        f"{horizontal_id}:0001",
    ]
    assert horizontal[0]["position"][0] < horizontal[1]["position"][0]
    assert all(abs(glyph["rotation"]) < 1e-6 for glyph in horizontal)

    assert abs(vertical[0]["rotation"] - math.pi / 2.0) < 1e-6
    assert abs(diagonal[0]["rotation"] - math.pi / 4.0) < 1e-6
    assert [cmd["cmd"] for cmd in commands] == [
        "add_line_label",
        "add_line_label",
        "add_line_label",
    ]


def test_curved_label_returns_experimental_diagnostic_instead_of_success():
    viewer, commands = _fake_viewer()

    result = viewer.add_curved_label(
        "River",
        [(0.0, 0.0, 0.0), (20.0, 0.0, 10.0), (40.0, 0.0, 0.0)],
    )

    assert isinstance(result, LabelOperationResult)
    assert result.ok is False
    assert commands == []
    assert [diag.code for diag in result.diagnostics] == ["experimental_feature"]
    assert result.diagnostics[0].support_level == "experimental"
    assert result.diagnostics[0].details["feature"] == "curved labels"
