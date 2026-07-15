import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE = Path("examples/label_api_truth_basic.py")
PUBLIC_DOCS = [
    Path("docs/api/api_reference.rst"),
    Path("docs/guides/label_support_matrix.md"),
]

if not EXAMPLE.exists():
    pytest.skip(
        "example 'label_api_truth_basic.py' is untracked/local-only",
        allow_module_level=True,
    )


def _load_example_module():
    examples_dir = str(EXAMPLE.parent.resolve())
    added_examples_dir = examples_dir not in sys.path
    if added_examples_dir:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("label_api_truth_basic", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_label_api_truth_example_exists_and_avoids_raw_ipc_helpers():
    assert EXAMPLE.exists()
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "viewer_ipc" not in source
    assert "send_ipc" not in source
    assert "ViewerHandle" in source
    assert "add_label" in source
    assert "add_labels" in source
    assert "add_line_label" in source


def test_label_api_truth_example_runs_and_reports_expected_support_statuses():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["point_label_id"] == 1
    assert payload["batch_ids"] == [2, 3]
    assert payload["line_label_id"] == 4
    assert payload["callout_id"] == 5
    assert payload["overlay_id"] == 1
    assert payload["state_after_clear"]["label_count"] == 0
    assert payload["state_after_clear"]["enabled"] is True
    assert payload["diagnostics"]["curved_labels"] == ["experimental_feature"]
    assert payload["diagnostics"]["terrain_elevated_line_labels"] == ["experimental_feature"]
    assert payload["diagnostics"]["typography_controls"] == []
    assert payload["diagnostics"]["decluttering_controls"] == []
    assert payload["typography_state"]["layout_metrics"]["typography_width"] > payload["typography_state"]["layout_metrics"]["default_width"]
    assert payload["declutter_state"]["declutter_algorithm"]["placement_order"] == "priority_then_energy"
    assert payload["diagnostics"]["missing_glyphs"] == ["missing_glyphs"]


def test_label_api_truth_workflow_is_deterministic_for_fixed_inputs():
    module = _load_example_module()

    first = module.run_basic_workflow()
    second = module.run_basic_workflow()

    assert first == second
    assert first["line_glyph_ordering_keys"] == ["4:0000", "4:0001", "4:0002", "4:0003"]
    assert first["line_glyph_rotations"] == [0.0, 0.0, 0.0, 0.0]


def test_label_api_truth_quickstart_points_to_runnable_example():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)

    assert "examples/label_api_truth_basic.py --json" in text
    assert "ViewerHandle.add_label" in text
    assert "experimental_feature" in text
    assert "layout metrics" in text
    assert "placement policy" in text
    assert "missing_glyphs" in text
