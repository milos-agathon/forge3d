"""Smoke example for the feature 002 high-level label API contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _import_shim import ensure_repo_import

ensure_repo_import()

from forge3d.viewer import LabelOperationResult, ViewerHandle


def _fake_viewer() -> tuple[ViewerHandle, list[dict[str, Any]]]:
    viewer = object.__new__(ViewerHandle)
    commands: list[dict[str, Any]] = []

    def fake_command(cmd: dict[str, Any]) -> dict[str, Any]:
        commands.append(dict(cmd))
        if cmd["cmd"] in {"add_label", "add_line_label", "add_callout", "add_vector_overlay"}:
            return {"ok": True, "id": cmd["id"]}
        return {"ok": True}

    viewer._send_command = fake_command  # type: ignore[attr-defined]
    return viewer, commands


def _codes(result: LabelOperationResult) -> list[str]:
    return [diagnostic.code for diagnostic in result.diagnostics]


def _require_id(value: int | LabelOperationResult, name: str) -> int:
    if isinstance(value, LabelOperationResult):
        raise RuntimeError(f"{name} was rejected: {_codes(value)}")
    return int(value)


def run_basic_workflow() -> dict[str, Any]:
    viewer, commands = _fake_viewer()

    viewer.load_label_atlas("examples/assets/fonts/default_atlas.png", "examples/assets/fonts/default_atlas.json")
    point_label_id = _require_id(
        viewer.add_label("Summit", (1.0, 2.0, 3.0), priority=100),
        "point label",
    )
    batch = viewer.add_labels(
        [
            {"text": "North", "world_pos": (0.0, 0.0, 0.0), "priority": 10},
            {"text": "South", "world_pos": (10.0, 0.0, 0.0), "priority": 9},
        ]
    )
    line_label_id = _require_id(
        viewer.add_line_label("ROAD", [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]),
        "line label",
    )
    line_state = viewer.label_configuration_state()
    line_glyphs = line_state["line_label_glyph_instances"][str(line_label_id)]

    curved = viewer.add_curved_label(
        "River",
        [(0.0, 0.0, 0.0), (20.0, 0.0, 10.0), (40.0, 0.0, 0.0)],
    )
    terrain = viewer.add_line_label(
        "Trail",
        [(0.0, 0.0, 0.0), (20.0, 0.0, 10.0)],
        terrain_mode="sample",
    )
    typography = viewer.set_label_typography(tracking=0.05, kerning=True, line_height=1.2)
    declutter = viewer.set_declutter_algorithm("annealing", seed=123)
    missing_glyph = viewer.add_label("Cafe accented: café", (0.0, 0.0, 0.0))
    callout_id = _require_id(viewer.add_callout("Peak", (5.0, 5.0, 5.0)), "callout")
    overlay_id = viewer.add_vector_overlay(
        "label-halo",
        vertices=[
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
        indices=[0, 1, 2],
    )

    viewer.set_labels_enabled(False)
    state_disabled = viewer.label_configuration_state()
    viewer.set_labels_enabled(True)
    viewer.clear_labels()
    state_after_clear = viewer.label_configuration_state()

    if not isinstance(curved, LabelOperationResult):
        raise RuntimeError("curved labels unexpectedly returned a production id")
    if not isinstance(terrain, LabelOperationResult):
        raise RuntimeError("terrain line labels unexpectedly returned a production id")
    if not isinstance(missing_glyph, LabelOperationResult):
        raise RuntimeError("missing glyph example unexpectedly returned a production id")

    return {
        "point_label_id": point_label_id,
        "batch_ids": [int(label_id) if label_id is not None else None for label_id in batch.ids],
        "line_label_id": line_label_id,
        "callout_id": callout_id,
        "overlay_id": int(overlay_id),
        "line_glyph_ordering_keys": [glyph["ordering_key"] for glyph in line_glyphs],
        "line_glyph_rotations": [round(float(glyph["rotation"]), 6) for glyph in line_glyphs],
        "typography_state": typography.state,
        "declutter_state": declutter.state,
        "state_disabled": state_disabled,
        "state_after_clear": state_after_clear,
        "diagnostics": {
            "curved_labels": _codes(curved),
            "terrain_elevated_line_labels": _codes(terrain),
            "typography_controls": _codes(typography),
            "decluttering_controls": _codes(declutter),
            "missing_glyphs": _codes(missing_glyph),
            "batch": [diagnostic.code for diagnostic in batch.diagnostics],
        },
        "commands": [command["cmd"] for command in commands],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the high-level label API truth smoke workflow.")
    parser.add_argument("--json", action="store_true", help="Print the deterministic workflow payload as JSON.")
    args = parser.parse_args()

    payload = run_basic_workflow()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("label ids:", payload["point_label_id"], payload["batch_ids"], payload["line_label_id"])
        print("diagnostics:", payload["diagnostics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
