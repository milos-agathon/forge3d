import pytest

from forge3d.viewer import LabelOperationResult, ViewerHandle


def _fake_viewer():
    viewer = object.__new__(ViewerHandle)
    commands = []

    def fake_send(cmd):
        commands.append(cmd)
        if cmd["cmd"] in {"add_label", "add_line_label"}:
            return {"ok": True, "id": cmd["id"]}
        return {"ok": True}

    viewer._send_command = fake_send
    return viewer, commands


def _assert_diagnostic_result(result, expected_code):
    assert isinstance(result, LabelOperationResult)
    assert result.ok is False
    assert [diag.code for diag in result.diagnostics] == [expected_code]
    return result.diagnostics[0]


def test_empty_label_text_returns_placeholder_diagnostic_without_sending_command():
    viewer, commands = _fake_viewer()

    result = viewer.add_label("", (0.0, 0.0, 0.0))

    diagnostic = _assert_diagnostic_result(result, "placeholder_fallback")
    assert diagnostic.object_id == "pending"
    assert commands == []


@pytest.mark.parametrize(
    "path",
    [
        [(0.0, 0.0, 0.0)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    ],
)
def test_invalid_line_geometry_returns_placeholder_diagnostic(path):
    viewer, commands = _fake_viewer()

    result = viewer.add_line_label("Road", path)

    diagnostic = _assert_diagnostic_result(result, "placeholder_fallback")
    assert diagnostic.details["feature"] == "invalid line label path"
    assert commands == []


def test_missing_glyphs_return_structured_diagnostic_before_render():
    viewer, commands = _fake_viewer()
    viewer.load_label_atlas("atlas.png", "metrics.json")

    result = viewer.add_label("Café", (0.0, 0.0, 0.0))

    diagnostic = _assert_diagnostic_result(result, "missing_glyphs")
    assert diagnostic.details["missing_glyphs"] == ["é"]
    assert [cmd["cmd"] for cmd in commands] == ["load_label_atlas"]


def test_curved_label_and_terrain_unavailable_paths_are_typed_diagnostics():
    viewer, commands = _fake_viewer()

    curved = viewer.add_curved_label(
        "Bend",
        [(0.0, 0.0, 0.0), (5.0, 0.0, 2.0), (10.0, 0.0, 0.0)],
    )
    terrain = viewer.add_line_label(
        "Ridge",
        [(0.0, 0.0, 0.0), (10.0, 0.0, 5.0)],
        terrain_mode="sample",
    )

    curved_diagnostic = _assert_diagnostic_result(curved, "experimental_feature")
    terrain_diagnostic = _assert_diagnostic_result(terrain, "experimental_feature")
    assert curved_diagnostic.details["feature"] == "curved labels"
    assert terrain_diagnostic.details["feature"] == "terrain-elevated line labels"
    assert commands == []
