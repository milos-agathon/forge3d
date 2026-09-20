from __future__ import annotations

import forge3d as f3d

# CARTOGRAPHER-PRIME (da4d5d1e): LabelPlan performs no line/curve re-layout.
# Repeated line placements come from the compute_line_label_placement geometry
# authority and are consumed verbatim; without authority the label is rejected
# with an explicit diagnostic instead of a synthetic placement.
_GLYPHS = [{"font_index": 0, "glyph_id": 7, "origin": [2.0, 3.0], "rotation": 0.0}]


def _line_authority(label_id: str, xs: list[float], y: float) -> dict:
    return {
        "source": "compute_line_label_placement",
        "positioned_glyphs": _GLYPHS,
        "candidates": [
            {
                "candidate_id": f"{label_id}:repeat-{index}",
                "candidate_type": "line_repeat",
                "anchor": [x, y, 0.5],
                "bounds": [x - 4.0, y - 2.0, x + 4.0, y + 2.0],
            }
            for index, x in enumerate(xs)
        ],
    }


def test_repeated_line_labels_are_deterministic_from_geometry_authority():
    labels = [
        {
            "id": "road-a",
            "text": "A1",
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
            "priority_class": "roads",
            "geometry_authority": _line_authority("road-a", [4.0, 40.0, 80.0], 10.0),
        }
    ]

    first = f3d.LabelPlan.compile(labels=labels, camera={}, viewport=(200, 100), seed=7)
    second = f3d.LabelPlan.compile(labels=labels, camera={}, viewport=(200, 100), seed=7)

    assert first.to_dict() == second.to_dict()
    assert not first.rejected
    assert len(first.accepted) == 1
    accepted = first.accepted[0]
    assert accepted.geometry_type == "LineString"
    assert accepted.candidate.candidate_type == "line_repeat"
    assert [candidate.anchor[:2] for candidate in accepted.candidates] == [
        (4.0, 10.0),
        (40.0, 10.0),
        (80.0, 10.0),
    ]
    assert all(
        candidate.details["geometry_authority"] == "compute_line_label_placement"
        for candidate in accepted.candidates
    )


def test_repeated_line_label_without_authority_is_rejected_not_synthesized():
    plan = f3d.LabelPlan.compile(
        labels=[
            {
                "id": "road-a",
                "text": "A1",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
                "repeat_distance": 40,
            }
        ],
        camera={},
        viewport=(200, 100),
        seed=7,
    )

    assert not plan.accepted
    assert plan.rejected[0].reason == "missing_geometry_authority"
    assert plan.rejected[0].details["required_authority"] == "compute_line_label_placement"


def test_curved_line_labels_are_explicitly_diagnosed_not_silent_success():
    plan = f3d.LabelPlan.compile(
        labels=[
            {
                "id": "river-curve",
                "text": "River",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [20, 10], [40, 0]]},
                "curved_text": True,
            }
        ],
        camera={},
        viewport=(100, 100),
    )

    assert not plan.accepted
    assert plan.rejected[0].reason == "missing_geometry_authority"
    diagnostic = next(
        d for d in plan.diagnostics if d.code == "label_geometry_authority_missing"
    )
    assert diagnostic.object_id == "river-curve"
    assert diagnostic.severity == "error"
    assert diagnostic.details["required_authority"] == "layout_curved_text"
