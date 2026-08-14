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
            "geometry_authority": {
                "source": "compute_line_label_placement",
                "positioned_glyphs": [
                    {"font_index": 0, "glyph_id": 1, "origin": [0, 0], "rotation": 0}
                ],
                "candidates": [
                    {
                        "candidate_id": f"road-a:repeat-{index}",
                        "candidate_type": "line_repeat",
                        "anchor": [x, 0, 0],
                        "bounds": [x, 0, x + 10, 10],
                        "details": {"repeat_distance": 40},
                    }
                    for index, x in enumerate((0, 40, 80))
                ],
            },
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


def test_curved_line_labels_require_authoritative_positioned_geometry():
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
    assert diagnostic.details["required_authority"] == "layout_curved_text"
