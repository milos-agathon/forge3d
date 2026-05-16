from forge3d.viewer import ViewerHandle


def _fake_viewer():
    viewer = object.__new__(ViewerHandle)
    commands = []

    def fake_send(cmd):
        commands.append(cmd)
        return {"ok": True}

    viewer._send_command = fake_send
    return viewer, commands


def test_set_label_typography_returns_diagnostic_instead_of_noop_success():
    viewer, commands = _fake_viewer()

    result = viewer.set_label_typography(
        tracking=0.25,
        kerning=True,
        line_height=1.3,
        word_spacing=2.0,
    )

    assert result.ok is False
    assert commands == []
    assert [diag.code for diag in result.diagnostics] == ["experimental_feature"]
    assert result.diagnostics[0].support_level == "experimental"
    assert result.state["typography"] is None


def test_set_declutter_algorithm_returns_diagnostic_instead_of_noop_success():
    viewer, commands = _fake_viewer()

    result = viewer.set_declutter_algorithm("annealing", seed=123, max_iterations=50)

    assert result.ok is False
    assert commands == []
    assert [diag.code for diag in result.diagnostics] == ["experimental_feature"]
    assert result.diagnostics[0].details["feature"] == "label declutter algorithm"
    assert result.state["declutter_algorithm"] is None
