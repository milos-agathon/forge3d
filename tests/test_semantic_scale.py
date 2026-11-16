import math
from dataclasses import replace

import numpy as np
import pytest

from forge3d import semantic_scale
from forge3d._map_scene_render import _compose_semantic_scale_locator
from forge3d.map_scene import LightingPreset, MapScene, OrbitCamera, OutputSpec, TerrainSource, VectorOverlay
from forge3d.semantic_scale import Connection, Landmark, RouteEdge, ScaleFrame, ScalePolicy, compile_scale


def test_locator_reserves_a_readable_footer_without_moving_geography():
    locator = {"rect": (0.77, 0.66, 0.98, 0.96), "aspect": 1.1,
               "routes": {"rail": ((0.1, 0.9), (0.9, 1.0))},
               "landmarks": {"station": (0.5, 1.0)}, "focus": (0.8, 1.0)}
    plain = np.full((360, 640, 4), 255, dtype=np.uint8)
    with_notice = plain.copy()
    _compose_semantic_scale_locator(plain, locator)
    _compose_semantic_scale_locator(with_notice, {**locator, "route_continues_in_locator": True})

    x0, y0, x1, y1 = locator["rect"]
    top, bottom = round(y0 * 360), round(y1 * 360)
    left, right = round(x0 * 640), round(x1 * 640)
    row = top + round((bottom - top) * 0.8)
    np.testing.assert_array_equal(with_notice[:row], plain[:row])
    footer = plain[row:bottom, left:right]
    assert np.all(footer == footer[0, 0])
    ink = np.any(with_notice[row:bottom, left:right] != footer, axis=2)
    ys, xs = np.nonzero(ink)
    assert 0 < ys.min() <= ys.max() < footer.shape[0] - 1
    assert 0 < xs.min() <= xs.max() < footer.shape[1] - 1


def inputs():
    terrain = TerrainSource(
        crs="EPSG:32632", data=np.zeros((41, 41), dtype=np.float32),
        metadata={"width": 41, "height": 41, "bounds": [0, 0, 410, 410],
                  "geotransform": [0, 10, 0, 410, 0, -10], "resolution": [10, 10],
                  "units": "m", "vertical_reference": "LN02"},
    )
    base = MapScene(
        terrain=terrain, camera=OrbitCamera(),
        lighting=LightingPreset(settings={"cli_params": {"camera_mode": "mesh:zup"}}),
        output=OutputSpec(320, 240), target_crs="EPSG:32632",
    )
    policy = ScalePolicy(
        crs="EPSG:32632", vertical_reference="LN02",
        routes=(RouteEdge("edge-a", ((140, 200), (175, 210), (210, 200))),
                RouteEdge("edge-b", ((210, 200), (260, 220)))),
        connections=(Connection("Through route", ("edge-a", "edge-b")),),
        landmarks=(Landmark("place-a", "Start", (140, 200), True, "Route end"),
                   Landmark("place-b", "Local", (260, 220), False, "Local orientation", 35)),
        locator_bounds=(5, 5, 405, 405), locator_rect=(0.72, 0.68, 0.98, 0.98),
        overview_tolerance_m=10, pixel_tolerance=2, stroke_width_px=3, label_size_px=16,
        reasons=("10 m DEM spacing and 2 px fixture tolerance",),
        omissions=("Unselected branches outside this itinerary",),
    )
    frame = ScaleFrame(OrbitCamera(distance=260, elevation_deg=75, fov_deg=45), (205, 205))
    return base, policy, frame


def test_compiles_output_boundary_availability_and_random_access():
    base, policy, close = inputs()
    far = replace(close, camera=replace(close.camera, distance=1800))
    partial = replace(close, unavailable_detail=("fine height tile missing",))
    result = compile_scale(base, policy, (far, close, partial, far))
    seq = result.to_dict()["frames"]
    assert [result.frame_record(i) for i in (2, 0, 3, 1)] == [seq[i] for i in (2, 0, 3, 1)]
    assert [x["routes"]["edge-a"]["representation"] for x in seq] == ["overview", "full", "locator_only", "overview"]
    assert "ground_span_smaller_than_label" == seq[0]["labels_omitted"]["place-b"]
    assert "detail_unavailable" == seq[2]["labels_omitted"]["place-b"]
    assert any(d["reason"] == "overview_exceeds_output_tolerance" for d in seq[2]["diagnostics"])
    assert seq[0]["locator"]["routes"] == seq[2]["locator"]["routes"]
    assert seq[2]["locator"]["route_in_locator_only"] is True
    assert all(route["representation"] == "locator_only" for route in seq[2]["routes"].values())
    assert not any(isinstance(layer, VectorOverlay) for layer in result.scene(2).recipe.layers)
    far_outage = compile_scale(base, policy, (replace(far, unavailable_detail=partial.unavailable_detail),))
    assert far_outage.frame_record(0)["routes"]["edge-a"]["representation"] == "overview"
    assert any(isinstance(layer, VectorOverlay) for layer in far_outage.scene(0).recipe.layers)
    assert seq[2]["locator"]["focus"] == [0.5, 0.5]
    independent = compile_scale(base, policy, (partial,)).frame_record(0)
    assert independent == {**seq[2], "frame": 0}
    assert result.frame_record(2) == seq[2]
    for i, expected in enumerate(seq):
        frame = result.scene(i)
        assert expected["label_rationale"]
        assert "rationale" not in frame.recipe.layers[-1].plan
        plan = frame.compile_plan().label_plans["semantic-scale-labels"]
        assert [label.source_id for label in plan.accepted] == expected["labels_accepted"]
        for label in plan.accepted:
            assert label.candidate.candidate_type == "center"
            assert all(candidate.anchor == label.candidate.anchor for candidate in label.candidates)
        if not expected["locator"].get("route_in_locator_only"):
            features = frame.recipe.layers[0].features
            assert tuple(features[0]["geometry"]["coordinates"][-1]) == tuple(features[1]["geometry"]["coordinates"][0])

    assert "place-a" in seq[0]["labels_accepted"]
    error = seq[1]["routes"]["edge-a"]["overview_error_px"]
    for threshold, expected in ((error, "overview"), (math.nextafter(error, 0), "full")):
        at_boundary = compile_scale(base, replace(policy, pixel_tolerance=threshold), (close,))
        assert at_boundary.frame_record(0)["routes"]["edge-a"]["representation"] == expected

    highres = MapScene(replace(base.recipe, output=OutputSpec(640, 480)))
    decision = compile_scale(highres, policy, (close,)).frame_record(0)
    assert decision["routes"]["edge-a"]["overview_error_px"] == pytest.approx(2 * error)


@pytest.mark.parametrize("change, error", [
    ({"vertical_reference": "ellipsoidal"}, "datum"),
    ({"crs": "EPSG:4326"}, "projected metre"),
    ({"pixel_tolerance": float("nan")}, "finite and positive"),
    ({"connections": (Connection("Missing", ("unknown",)),)}, "known edges"),
    ({"routes": (RouteEdge("edge-a", ((15, 15), (25, 25))),
                 RouteEdge("edge-b", ((35, 35), (45, 45))))}, "disconnected"),
    ({"locator_bounds": (200, 200, 400, 400)}, "locator does not contain"),
])
def test_invalid_policy(change, error):
    base, policy, frame = inputs()
    with pytest.raises(ValueError, match=error):
        compile_scale(base, replace(policy, **change), (frame,))



def test_required_label_rejection_and_invalid_camera():
    base, policy, frame = inputs()
    # Reserve the full viewport to exercise the existing placement rejection.
    policy = replace(policy, locator_rect=(0, 0, 1, 1))
    record = compile_scale(base, policy, (frame,)).frame_record(0)
    assert record["labels_rejected"]["place-a"]
    assert {"source_id": "place-a", "reason": "required_label_unplaced",
            "detail": record["labels_rejected"]["place-a"]} in record["diagnostics"]
    assert "place-a" in record["locator"]["landmarks"]
    with pytest.raises(ValueError, match="finite perspective camera"):
        compile_scale(base, policy, (replace(frame, camera=replace(frame.camera, distance=0)),))


def test_suppresses_only_enclosed_one_frame_optional_labels():
    base, policy, close = inputs()
    far = replace(close, camera=replace(close.camera, distance=1800))
    once = compile_scale(base, policy, (far, close, far))
    record = once.frame_record(1)
    assert record["labels_omitted"]["place-b"] == "single_frame_admission"
    assert "place-b" not in record["labels_accepted"]
    assert "place-a" in record["labels_accepted"]
    assert once.scene(1).compile_plan().label_plans["semantic-scale-labels"].accepted[0].source_id == "place-a"
    twice = compile_scale(base, policy, (far, close, close, far))
    assert all("place-b" in twice.frame_record(i)["labels_accepted"] for i in (1, 2))
    assert "place-b" in compile_scale(base, policy, (close,)).frame_record(0)["labels_accepted"]


def test_offscreen_edge_cannot_trigger_outage_fallback():
    base, policy, close = inputs()
    routes = (
        RouteEdge("edge-a", ((140, 200), (210, 200))),
        RouteEdge("edge-b", ((280, 380), (305, 390), (330, 380))),
    )
    policy = replace(policy, routes=routes, pixel_tolerance=0.5,
                     connections=(Connection("Visible connection", ("edge-a",)),))
    frame = replace(close, unavailable_detail=("fine route detail missing",))
    record = semantic_scale.compile_scale(base, policy, (frame,)).frame_record(0)
    assert record["routes"]["edge-a"]["representation"] == "overview"
    assert record["routes"]["edge-b"] == {"representation": "locator_only", "reason": "outside_viewport"}
    assert not record["locator"].get("route_in_locator_only")
    assert not any(d["reason"] == "overview_exceeds_output_tolerance" for d in record["diagnostics"])


def test_unprojectable_edge_announces_full_route_in_locator(monkeypatch):
    base, policy, frame = inputs()
    project = semantic_scale.TerrainProjector.path

    def project_path(projector, path):
        if path[-1] == policy.routes[1].path[-1]:
            raise ValueError("world overlay anchor is behind the terrain camera")
        return project(projector, path)

    monkeypatch.setattr(semantic_scale.TerrainProjector, "path", project_path)
    result = compile_scale(base, policy, (frame,))
    record = result.frame_record(0)
    assert record["routes"]["edge-a"]["representation"] == "full"
    assert record["routes"]["edge-b"]["representation"] == "locator_only"
    assert record["locator"]["route_continues_in_locator"]
    assert set(record["locator"]["routes"]) == {"edge-a", "edge-b"}
    assert any(d["reason"] == "route_projection_unavailable" for d in record["diagnostics"])


def test_viewport_crossing_counts_with_both_endpoints_outside():
    points = semantic_scale._viewport_samples(np.array([[-4, 3], [14, 3]], dtype=float), 10, 8)
    np.testing.assert_array_equal(points, [[0, 3], [10, 3]])
