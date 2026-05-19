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


def test_atlas_enabled_and_clear_update_serializable_label_state():
    viewer, commands = _fake_viewer()

    atlas_result = viewer.load_label_atlas("atlas.png", "metrics.json")
    enabled_result = viewer.set_labels_enabled(False)
    label_id = viewer.add_label("Summit", (1.0, 2.0, 3.0))
    clear_result = viewer.clear_labels()

    assert atlas_result.ok is True
    assert atlas_result.state["active_atlas"] == {
        "atlas_png_path": "atlas.png",
        "metrics_json_path": "metrics.json",
    }
    assert enabled_result.ok is True
    assert enabled_result.state["enabled"] is False
    assert label_id not in clear_result.state["label_ids"]
    assert clear_result.state["label_count"] == 0
    assert [cmd["cmd"] for cmd in commands] == [
        "load_label_atlas",
        "set_labels_enabled",
        "add_label",
        "clear_labels",
    ]


def test_remove_label_changes_known_state_and_diagnoses_missing_id():
    viewer, commands = _fake_viewer()

    label_id = viewer.add_label("City", (0.0, 0.0, 0.0))
    remove_result = viewer.remove_label(label_id)
    missing_result = viewer.remove_label(label_id)

    assert remove_result.ok is True
    assert remove_result.state["label_count"] == 0
    assert [cmd["cmd"] for cmd in commands] == ["add_label", "remove_label"]
    assert commands[1]["id"] == label_id

    assert missing_result.ok is False
    assert [diag.code for diag in missing_result.diagnostics] == ["placeholder_fallback"]
    assert missing_result.diagnostics[0].object_id == str(label_id)
