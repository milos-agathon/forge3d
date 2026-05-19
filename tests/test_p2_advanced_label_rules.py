from __future__ import annotations

import forge3d as f3d


def test_cartographic_priority_preset_orders_multi_class_labels():
    labels = [
        {
            "id": "city",
            "text": "City",
            "geometry": {"type": "Point", "coordinates": [10, 10]},
            "priority_class": "cities",
        },
        {
            "id": "capital",
            "text": "Capital",
            "geometry": {"type": "Point", "coordinates": [10, 10]},
            "priority_class": "capitals",
        },
        {
            "id": "road",
            "text": "Road",
            "geometry": {"type": "LineString", "coordinates": [[0, 30], [80, 30]]},
            "priority_class": "roads",
            "placement_preset": "road",
            "repeat_distance": 80,
        },
    ]

    plan = f3d.LabelPlan.compile(
        labels=labels,
        camera={},
        viewport=(100, 100),
        priority_rules="cartographic",
        seed=3,
    )

    accepted_ids = [label.label_id for label in plan.accepted]
    rejected_ids = [label.label_id for label in plan.rejected]
    assert "capital" in accepted_ids
    assert "road" in accepted_ids
    assert "city" in rejected_ids
    city = next(item for item in plan.rejected if item.label_id == "city")
    assert city.reason == "priority_lost"
    assert city.details["winner_priority_class"] == "capitals"
    assert plan.bounds["priority_rules"][0]["name"] == "annotations"


def test_leader_line_callout_records_deterministic_candidate_details():
    plan = f3d.LabelPlan.compile(
        labels=[
            {
                "id": "peak-callout",
                "text": "Peak",
                "geometry": {"type": "Point", "coordinates": [20, 20]},
                "placement_preset": "callout",
                "leader_line": True,
                "priority_class": "peaks",
            }
        ],
        camera={},
        viewport=(100, 100),
        priority_rules="cartographic",
    )

    assert len(plan.accepted) == 1
    candidate = plan.accepted[0].candidate
    assert candidate.candidate_type == "leader_line"
    assert candidate.details["leader_line"] is True
    assert candidate.details["placement_preset"] == "callout"
