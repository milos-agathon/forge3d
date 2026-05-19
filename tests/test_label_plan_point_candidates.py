def _compile_point(seed):
    from forge3d import LabelPlan

    return LabelPlan.compile(
        labels=[
            {
                "id": "poi",
                "text": "Point",
                "geometry": {"type": "Point", "coordinates": (100.0, 120.0, 5.0)},
                "priority": 7,
                "candidate_policy": {
                    "offset_px": 10.0,
                    "radial_count": 4,
                    "radial_radius_px": 20.0,
                    "radial_jitter_deg": 6.0,
                },
            }
        ],
        camera={"name": "fixed"},
        viewport={"width": 400, "height": 300},
        glyph_atlas={"glyphs": list("Pinot")},
        seed=seed,
    )


def test_point_label_generates_required_candidate_types_in_stable_order():
    plan = _compile_point(seed=22)
    assert len(plan.accepted) == 1

    accepted = plan.accepted[0]
    candidates = [candidate.to_dict() for candidate in accepted.candidates]

    assert accepted.candidate.candidate_id == "poi:center"
    assert [candidate["candidate_type"] for candidate in candidates] == [
        "center",
        "above",
        "below",
        "left",
        "right",
        "radial",
        "radial",
        "radial",
        "radial",
    ]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "poi:center",
        "poi:above",
        "poi:below",
        "poi:left",
        "poi:right",
        "poi:radial-0",
        "poi:radial-1",
        "poi:radial-2",
        "poi:radial-3",
    ]
    assert candidates[0]["anchor"] == [100.0, 120.0, 5.0]
    assert candidates[1]["anchor"] == [100.0, 110.0, 5.0]
    assert candidates[2]["anchor"] == [100.0, 130.0, 5.0]
    assert candidates[3]["anchor"] == [90.0, 120.0, 5.0]
    assert candidates[4]["anchor"] == [110.0, 120.0, 5.0]
    assert all(candidate["bounds"] is not None for candidate in candidates)
    assert all(candidate["ordering_key"].startswith("poi:") for candidate in candidates)


def test_radial_candidates_are_seeded_and_roundtrip_deterministic():
    first = _compile_point(seed=33).to_dict()
    second = _compile_point(seed=33).to_dict()
    different_seed = _compile_point(seed=34).to_dict()

    first_radials = [
        candidate
        for candidate in first["accepted"][0]["candidates"]
        if candidate["candidate_type"] == "radial"
    ]
    different_radials = [
        candidate
        for candidate in different_seed["accepted"][0]["candidates"]
        if candidate["candidate_type"] == "radial"
    ]

    assert second == first
    assert [candidate["details"]["angle_deg"] for candidate in first_radials] != [
        candidate["details"]["angle_deg"] for candidate in different_radials
    ]
    assert [candidate["details"]["radial_index"] for candidate in first_radials] == [0, 1, 2, 3]
