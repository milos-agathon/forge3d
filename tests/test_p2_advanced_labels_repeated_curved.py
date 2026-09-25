from __future__ import annotations

import forge3d as f3d


def test_repeated_line_labels_are_deterministic_with_repeat_distance():
    labels = [
        {
            "id": "road-a",
            "text": "A1",
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
            "repeat_distance": 40,
            "priority_class": "roads",
        }
    ]

    first = f3d.LabelPlan.compile(labels=labels, camera={}, viewport=(200, 100), seed=7)
    second = f3d.LabelPlan.compile(labels=labels, camera={}, viewport=(200, 100), seed=7)

    assert first.to_dict() == second.to_dict()
    assert len(first.accepted) == 1
    accepted = first.accepted[0]
    assert accepted.geometry_type == "LineString"
    assert accepted.candidate.candidate_type == "line_repeat"
    assert [candidate.anchor[:2] for candidate in accepted.candidates] == [
        (0.0, 0.0),
        (40.0, 0.0),
        (80.0, 0.0),
    ]
    assert accepted.candidate.details["repeat_distance"] == 40.0


def test_curved_line_labels_use_native_geometry_authority():
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

    assert [label.label_id for label in plan.accepted] == ["river-curve"]
    candidate = plan.accepted[0].candidate
    assert candidate.candidate_type == "curved_layout"
    assert candidate.details["geometry_authority"] == "layout_curved_text"
    assert candidate.details["glyph_placements"]
