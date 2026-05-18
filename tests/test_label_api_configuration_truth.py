from forge3d.viewer import ViewerHandle


def _fake_viewer():
    viewer = object.__new__(ViewerHandle)
    commands = []

    def fake_send(cmd):
        commands.append(cmd)
        return {"ok": True}

    viewer._send_command = fake_send
    return viewer, commands


def test_set_label_typography_mutates_state_and_layout_metrics():
    viewer, commands = _fake_viewer()

    result = viewer.set_label_typography(
        tracking=0.25,
        kerning=True,
        line_height=1.3,
        word_spacing=2.0,
    )

    assert result.ok is True
    assert result.diagnostics == []
    assert commands == [
        {
            "cmd": "set_label_typography",
            "tracking": 0.25,
            "kerning": True,
            "line_height": 1.3,
            "word_spacing": 2.0,
        }
    ]
    assert result.state["typography"] == {
        "tracking": 0.25,
        "kerning": True,
        "line_height": 1.3,
        "word_spacing": 2.0,
    }
    assert result.state["layout_metrics"]["sample_text"] == "AV label"
    assert result.state["layout_metrics"]["typography_width"] > result.state["layout_metrics"]["default_width"]
    assert result.state["layout_metrics"]["line_height_px"] == 20.8


def test_set_declutter_algorithm_mutates_placement_policy_state():
    viewer, commands = _fake_viewer()

    result = viewer.set_declutter_algorithm("annealing", seed=123, max_iterations=50)

    assert result.ok is True
    assert result.diagnostics == []
    assert commands == [
        {
            "cmd": "set_declutter_algorithm",
            "algorithm": "annealing",
            "seed": 123,
            "max_iterations": 50,
        }
    ]
    assert result.state["declutter_algorithm"] == {
        "algorithm": "annealing",
        "seed": 123,
        "max_iterations": 50,
        "placement_order": "priority_then_energy",
    }
