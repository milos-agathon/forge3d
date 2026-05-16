import inspect

from forge3d.viewer import ViewerHandle


def _fake_viewer():
    viewer = object.__new__(ViewerHandle)
    commands = []

    def fake_send(cmd):
        commands.append(cmd)
        if cmd["cmd"] in {"add_label", "add_line_label", "add_callout"}:
            return {"ok": True, "id": cmd["id"]}
        return {"ok": True}

    viewer._send_command = fake_send
    return viewer, commands


def test_high_level_label_workflow_uses_public_viewer_methods():
    viewer, commands = _fake_viewer()

    viewer.load_label_atlas("atlas.png", "metrics.json")
    label_id = viewer.add_label("Summit", (1.0, 2.0, 3.0), size=18.0)
    batch = viewer.add_labels(
        [
            {"text": "North", "world_pos": (0.0, 0.0, 0.0)},
            {"text": "South", "world_pos": (1.0, 0.0, 0.0)},
        ]
    )
    viewer.set_labels_enabled(False)
    viewer.clear_labels()

    assert isinstance(label_id, int)
    assert batch.ids == [commands[2]["id"], commands[3]["id"]]
    assert batch.diagnostics == []
    assert [cmd["cmd"] for cmd in commands] == [
        "load_label_atlas",
        "add_label",
        "add_label",
        "add_label",
        "set_labels_enabled",
        "clear_labels",
    ]


def test_public_label_workflow_does_not_call_raw_viewer_ipc_helpers():
    public_methods = [
        ViewerHandle.load_label_atlas,
        ViewerHandle.add_label,
        ViewerHandle.add_labels,
        ViewerHandle.set_labels_enabled,
        ViewerHandle.clear_labels,
    ]

    for method in public_methods:
        source = inspect.getsource(method)
        assert "viewer_ipc" not in source
        assert "send_ipc(" not in source
