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


def test_reverse_line_label_applies_upside_down_avoidance():
    viewer, commands = _fake_viewer()

    label_id = viewer.add_line_label("W", [(10.0, 0.0, 0.0), (0.0, 0.0, 0.0)])

    glyph = _glyphs_for(viewer, label_id)[0]
    assert abs(glyph["rotation"]) < 1e-6
    assert glyph["upside_down_adjusted"] is True
    assert [cmd["cmd"] for cmd in commands] == ["add_line_label"]


def test_terrain_elevated_line_label_returns_diagnostic_when_sampling_unavailable():
    viewer, commands = _fake_viewer()

    result = viewer.add_line_label(
        "Trail",
        [(0.0, 0.0, 0.0), (10.0, 0.0, 10.0)],
        terrain_mode="sample",
    )

    assert isinstance(result, LabelOperationResult)
    assert result.ok is False
    assert commands == []
    assert [diag.code for diag in result.diagnostics] == ["experimental_feature"]
    assert result.diagnostics[0].details["feature"] == "terrain-elevated line labels"
