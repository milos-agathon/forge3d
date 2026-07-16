import pytest

from forge3d.viewer import ViewerError, ViewerHandle


def _fake_viewer(response_factory=None):
    viewer = object.__new__(ViewerHandle)
    commands = []

    def fake_send(cmd):
        commands.append(cmd)
        if response_factory is not None:
            return response_factory(cmd)
        if cmd["cmd"] in {"add_label", "add_line_label", "add_callout", "add_vector_overlay"}:
            return {"ok": True, "id": cmd["id"]}
        return {"ok": True}

    viewer._send_command = fake_send
    return viewer, commands


def test_add_label_returns_stable_created_id_and_sends_it_to_viewer():
    viewer, commands = _fake_viewer()

    label_id = viewer.add_label("City", (1.0, 2.0, 3.0), priority=4)

    assert isinstance(label_id, int)
    assert commands == [
        {
            "cmd": "add_label",
            "id": label_id,
            "text": "City",
            "world_pos": [1.0, 2.0, 3.0],
            "priority": 4,
        }
    ]


def test_add_label_rejects_success_response_without_created_id():
    viewer, _commands = _fake_viewer(lambda _cmd: {"ok": True})

    with pytest.raises(ViewerError, match="stable label id"):
        viewer.add_label("City", (1.0, 2.0, 3.0))


def test_add_labels_preserves_input_order_with_per_input_diagnostics():
    viewer, commands = _fake_viewer()

    result = viewer.add_labels(
        [
            {"text": "A", "world_pos": (0.0, 0.0, 0.0)},
            {"text": "", "world_pos": (1.0, 0.0, 0.0)},
            {"text": "C", "world_pos": (2.0, 0.0, 0.0)},
        ]
    )

    assert result.ids == [commands[0]["id"], None, commands[1]["id"]]
    assert [cmd["text"] for cmd in commands] == ["A", "C"]
    assert [diag.code for diag in result.diagnostics] == ["placeholder_fallback"]
    assert result.diagnostics[0].object_id == "1"


def test_line_label_callout_and_overlay_creation_return_stable_ids():
    viewer, commands = _fake_viewer()

    line_id = viewer.add_line_label(
        "Road",
        [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        placement="along",
    )
    callout_id = viewer.add_callout("Peak", (1.0, 2.0, 3.0))
    overlay_id = viewer.add_vector_overlay(
        "label-halo",
        vertices=[
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1],
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1],
        ],
        indices=[0, 1, 2],
    )

    assert line_id == commands[0]["id"]
    assert callout_id == commands[1]["id"]
    assert overlay_id == commands[2]["id"]
    assert [cmd["cmd"] for cmd in commands] == [
        "add_line_label",
        "add_callout",
        "add_vector_overlay",
    ]
