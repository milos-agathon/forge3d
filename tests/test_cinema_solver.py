"""AUTOCHIRON's physical GPU acceptance suite. Acceptance never skips."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

import forge3d as f3d
from forge3d.cinema import Subject, FramingSpec, Shot, ShotPlan, CinemaInfeasible, CinemaNativeUnavailable, plan
from forge3d.terrain_scatter import TerrainScatterSource

FIXTURES = Path(__file__).parent / "data/cinema_scenes"
FEASIBLE = ("knife_edge", "valley_col", "summit_cliff", "canyon")


def test_frozen_jerk_budgets_are_derived_only_from_declared_inputs():
    from cinema_reference import reference
    frozen = json.loads((FIXTURES.parent / "autochiron_jerk_budgets.json").read_text())
    assert frozen["status"] == "frozen_before_acceptance"
    for name, value in frozen["fixtures"].items():
        data = json.loads((FIXTURES / f"{name}.json").read_text())
        declared = {"heightmap": data["heightmap"], "terrain_width": data["terrain_width"],
                    "shot": {k: v for k, v in data["shot"].items() if k != "max_jerk"}}
        assert hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest() == value["geometry_duration_sha256"]
        assert reference(declared)["proposed_max_jerk"] == value["proposed_max_jerk"] == data["shot"]["max_jerk"]
    control = frozen["insufficient_budget_control"]
    assert control["max_jerk"] < control["analytic_required_minimum"]


def load_scene(name):
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    fields = dict(data["shot"])
    fields["subject"] = Subject(**fields["subject"])
    fields["framing"] = FramingSpec(**fields["framing"])
    shot = Shot(**fields)
    source = TerrainScatterSource(np.asarray(data["heightmap"], np.float32), terrain_width=data["terrain_width"])
    return source, shot


def assert_report(report, shot):
    assert not report["software_fallback"]
    assert report["trace_rays"] > 0
    assert report["max_dt_s"] <= 1 / 240
    assert report["heightfield_violations"] == report["trace_clearance_violations"] == 0
    assert report["minimum_clearance"] >= shot.clearance
    assert report["framing_fraction"] >= .95
    assert report["framing_outside_transitions"] == 0
    assert report["knot_continuity_relative"] < 1e-6
    assert report["max_jerk"] <= shot.max_jerk
    assert report["analytic_max_jerk"] <= report["max_jerk"]
    assert report["consumer_max_acceleration_jump"] >= 0
    assert report["spline_max_jerk"] <= shot.max_jerk
    assert report["max_speed"] <= shot.max_speed
    assert report["resampling_relative_error"] < .001
    assert report["occlusion_fraction"] <= .05
    assert report["longest_occlusion_outside_transition_s"] <= .5


def record_case_report(case, reports):
    print("case-report", json.dumps({"source": "knife_edge", "case": case,
                                         "reports": reports}, sort_keys=True))


@pytest.mark.parametrize("name", FEASIBLE)
def test_four_feasible_scenes(name, tmp_path):
    source, shot = load_scene(name)
    started = time.perf_counter()
    first = plan([shot], source)
    elapsed = time.perf_counter() - started
    second = plan([shot], source)
    assert first.keyframes_json == second.keyframes_json
    replay = ShotPlan.from_json(first.to_json())
    assert replay.to_json() == first.to_json()
    assert replay.keyframes_json == first.keyframes_json
    assert replay.bake().keyframe_count > 2
    assert_report(first.validation[0], shot)
    assert elapsed <= 10
    data = json.loads(first.to_json())
    assert data["compiled"][0]["solve_time_s"] <= 10
    report = dict(first.validation[0], scene=name, total_solve_time_s=elapsed,
                  keyframes_sha256=hashlib.sha256(first.keyframes_json.encode()).hexdigest(),
                  spline_jerk_headroom=shot.max_jerk-first.validation[0]["spline_max_jerk"],
                  consumer_interval_jerk_headroom=shot.max_jerk-first.validation[0]["max_jerk"],
                  limiting_jerk_headroom_fraction=(shot.max_jerk-max(first.validation[0]["max_jerk"],first.validation[0]["spline_max_jerk"]))/shot.max_jerk)
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def test_linear_ablation_is_load_bearing():
    reports = []
    for name in FEASIBLE:
        source, shot = load_scene(name)
        report = plan([shot], source).validation[0]
        reports.append(report)
        print(name, "linear violations", report["ablation_violations"], "depth", report["ablation_depth"])
    assert sum(r["ablation_violations"] > 0 for r in reports) >= 3


def test_infeasible_names_binding_constraint():
    source, shot = load_scene("infeasible")
    with pytest.raises(CinemaInfeasible) as caught:
        plan([shot], source)
    assert caught.value.binding_constraint == "clearance"
    print(str(caught.value))


def test_obstructed_dolly_reports_clearance_without_detouring():
    source, shot = load_scene("canyon")
    heights = source.heightmap.copy()
    heights[32, 12] = 54.  # x=18.75: interior ridge, clear of both endpoints.
    source = TerrainScatterSource(heights, terrain_width=source.terrain_width)
    with pytest.raises(CinemaInfeasible) as caught:
        plan([shot], source)
    assert caught.value.binding_constraint == "clearance"
    assert "straight dolly" in caught.value.diagnostic["detail"]


def test_canyon_fixture_keeps_a_feasible_straight_dolly():
    source, shot = load_scene("canyon")
    assert shot.kind == "dolly"
    result = plan([shot], source)
    assert_report(result.validation[0], shot)
    segments = json.loads(result.to_json())["compiled"][0]["solution"]["segments"]
    assert len(segments) == 1 and segments[0].get("arc") is None
    assert tuple(segments[0]["eye0"]) == shot.start_eye
    assert tuple(segments[0]["eye1"]) == shot.end_eye
    assert result.validation[0]["ablation_violations"] == 0
    # The camera enters below the walls and clears the floor ridge. A global
    # enclosing-box check would include the 64-unit wall and reject this line.
    assert shot.start_eye[0] < 12.5 < shot.end_eye[0]
    assert np.max(source.heightmap) == 64. > shot.start_eye[1]
    floor = source.heightmap[32]
    x = np.linspace(shot.start_eye[0], shot.end_eye[0], int(shot.duration_s * 240) + 1)
    heights = np.interp(x, np.linspace(0., source.terrain_width, source.width), floor)
    assert shot.start_eye[1] - np.max(heights) == 4. > shot.clearance
    assert result.validation[0]["minimum_clearance"] >= shot.clearance
    print("canyon-straight-dolly", json.dumps(result.validation[0], sort_keys=True))


def test_dolly_clears_a_line_with_an_off_path_obstacle_in_its_bounding_box():
    original, shot = load_scene("knife_edge")
    heights = np.zeros_like(original.heightmap)
    heights[2, 5] = 50.
    source = TerrainScatterSource(heights, terrain_width=100.)
    shot = replace(shot, kind="dolly", start_eye=(10., 12., 10.), end_eye=(35., 12., 35.))
    result = plan([shot], source)
    assert_report(result.validation[0], shot)
    segments = json.loads(result.to_json())["compiled"][0]["solution"]["segments"]
    assert len(segments) == 1
    assert tuple(segments[0]["eye0"]) == shot.start_eye
    assert tuple(segments[0]["eye1"]) == shot.end_eye


@pytest.mark.parametrize("region,start,end,constraint", (
    ((.2,.2,.8,.8),(10.,100.,50.),(80.,100.,50.),"framing_orientation"),
    ((.65,.45,.75,.55),(49.,100.,50.),(49.5,100.,50.),"framing_region"),
))
def test_dolly_distinguishes_orientation_and_region_from_distance(region, start, end, constraint):
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=100.)
    shot = replace(shot, kind="dolly", start_eye=start, end_eye=end,
                   framing=replace(shot.framing, region=region))
    with pytest.raises(CinemaInfeasible) as caught:
        plan([shot], source)
    assert caught.value.binding_constraint == constraint


def test_dolly_consumer_clearance_failure_keeps_its_binding_diagnostic():
    source, shot = load_scene("canyon")
    result = plan([shot], source)
    data = json.loads(result.to_json())
    for key in data["keyframes"]:
        key["target"][1] -= 5.
    lowered = ShotPlan.from_json(json.dumps(data))
    with pytest.raises(CinemaInfeasible) as caught:
        lowered.validate(source)
    assert caught.value.binding_constraint == "clearance"


def test_straight_dolly_cannot_route_around_subject_framing_shell():
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=100.)
    shot = replace(shot, kind="dolly", start_eye=(10., 55., 50.), end_eye=(80., 55., 50.))
    with pytest.raises(CinemaInfeasible) as caught:
        plan([shot], source)
    assert caught.value.binding_constraint == "framing_distance"
    assert "straight dolly" in caught.value.diagnostic["detail"]


def test_dolly_solved_and_emitted_path_stays_on_straight_approach():
    source, shot = load_scene("knife_edge")
    shot = replace(shot, kind="dolly", start_eye=(10., 55., 50.), end_eye=(25., 55., 50.))
    result = plan([shot], source)
    assert_report(result.validation[0], shot)
    segments = json.loads(result.to_json())["compiled"][0]["solution"]["segments"]
    assert len(segments) == 1 and segments[0].get("arc") is None
    animation = result.bake()
    start, end = np.asarray(shot.start_eye), np.asarray(shot.end_eye)
    direction = end - start
    # Use the specification's physical 240 Hz verification cadence.
    for t in np.linspace(0., shot.duration_s, int(np.ceil(shot.duration_s * 240)) + 1):
        state = animation.evaluate(float(t))
        phi, theta = np.deg2rad([state.phi_deg, state.theta_deg])
        eye = np.asarray(state.target) + state.radius * np.array([
            np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
        u = np.dot(eye - start, direction) / np.dot(direction, direction)
        relative_error = np.linalg.norm(eye - (start + u * direction)) / np.linalg.norm(eye - np.mean(shot.subject.bounds, axis=0))
        assert relative_error < .001


def test_canonical_spec_and_rejection_boundaries():
    _, shot = load_scene("knife_edge")
    spec = ShotPlan((replace(shot, bearing_deg=(-0.0, 0.0)),))
    text = spec.to_json()
    assert "-0.0" not in text
    assert ShotPlan.from_json(text).to_json() == text
    with pytest.raises(ValueError):
        ShotPlan.from_json(text.replace('"headroom":0.05', '"headroom":NaN'))
    with pytest.raises(ValueError):
        replace(shot, duration_s=float("inf"))
    with pytest.raises(ValueError):
        FramingSpec((.8,.2,.1,.9), (.1,.2), 0, 55, 1)
    assert "cinema" in f3d.__all__
    for name in ("Subject", "FramingSpec", "Shot", "ShotPlan", "plan_shots"):
        assert name in f3d.__all__ and hasattr(f3d, name)
    for name in ("cinema_solve", "cinema_validate"):
        assert name in f3d.__all__ and hasattr(f3d, name)
    geographic = Subject.from_lon_lat_bounds(((1,2,3),(2,3,4)), to_world=lambda x,y,z:(x*2,z,y*3))
    assert geographic.bounds == ((2.,3.,6.),(4.,4.,9.))
    with pytest.raises(ValueError):
        replace(shot, duration_s=True)


def test_existing_verified_certificate_link_roundtrips():
    source, shot = load_scene("knife_edge")
    result = plan([shot], source)
    root = Path(__file__).parent / "golden/certificates"
    linked = result.link_certificate(root / "mapscene_terrain_raster.json", pubkey=root / "signing.pub", frame_time=1.0)
    assert ShotPlan.from_json(linked.to_json()).to_json() == linked.to_json()
    link = json.loads(linked.to_json())["certificate_links"][0]
    assert link["plan_keyframes_sha256"] == hashlib.sha256(result.keyframes_json.encode()).hexdigest()
    with pytest.raises(ValueError):
        result.link_certificate(root / "mapscene_terrain_raster.json", pubkey=bytes(32), frame_time=1.0)


def test_replayed_plan_is_revalidated_on_actual_consumer():
    source, shot = load_scene("knife_edge")
    result = plan([shot], source)
    artifact = json.loads(result.to_json())
    for k in artifact["keyframes"]:
        k["target"][1] -= 100
    forged = ShotPlan.from_json(json.dumps(artifact))
    with pytest.raises(CinemaInfeasible):
        forged.validate(source)
    source.heightmap[0,0] = 1
    changed = TerrainScatterSource(source.heightmap, terrain_width=100)
    with pytest.raises(ValueError, match="terrain differs"):
        result.validate(changed)


def test_trace_validator_detects_known_clearance_hit():
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=original.terrain_width)
    shot = replace(shot, kind="flyover", start_eye=(10.,shot.clearance+1.,50.), end_eye=(90.,shot.clearance+1.,50.))
    result = plan([shot], source)
    record_case_report("positive-trace-control", result.validation)
    artifact = json.loads(result.to_json())
    for key in artifact["keyframes"]:
        key["target"][1] -= shot.clearance
    with pytest.raises(CinemaInfeasible) as caught:
        ShotPlan.from_json(json.dumps(artifact)).validate(source)
    assert caught.value.binding_constraint == "clearance"
    report = json.loads(caught.value.diagnostic["detail"])
    assert report["heightfield_violations"] == report["samples"]
    assert report["trace_clearance_violations"] == report["samples"]
    assert report["adapter"] == result.validation[0]["adapter"]
    print("positive trace control", report["trace_clearance_violations"], report["adapter"], report["backend"])


def test_joined_animation_and_saved_position_join_consistency():
    source, shot = load_scene("knife_edge")
    second = replace(shot, kind="orbit", start_eye=shot.end_eye, end_eye=(50.0,44.0,90.0), bearing_deg=(0.0,90.0), duration_s=30.0)
    result = plan([shot, second], source)
    record_case_report("joined-reveal-orbit", result.validation)
    for report, part in zip(result.validation, result.shots):
        assert_report(report, part)
    # Numerical consistency probe of the saved rest-to-rest quintic contract.
    data=json.loads(result.to_json())
    segments = [segment for compiled in data["compiled"] for segment in compiled["solution"]["segments"]]
    for a,b in zip(segments,segments[1:]):
            assert a["eye1"] == b["eye0"]
            assert a["target1"] == b["target0"]
            # Consistency of the declared rest-to-rest basis and position joins.
            # The positions use that basis; this is not an independent C2 proof.
            for segment,u in ((a,1.0),(b,0.0)):
                if segment.get("arc") is not None:
                    continue  # Analytic arc derivatives are independently tested in Rust.
                h=segment["t1"]-segment["t0"]
                xs=np.linspace(0,1,6)
                for prefix in ("eye","target"):
                    first=np.asarray(segment[prefix+"0"])
                    delta=np.asarray(segment[prefix+"1"])-first
                    positions=first[None,:]+(xs**3*(10-15*xs+6*xs**2))[:,None]*delta[None,:]
                    coefficients=np.polynomial.polynomial.polyfit(xs,positions,5)
                    for order in (1,2):
                        derivative=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coefficients,order))/h**order
                        assert np.linalg.norm(derivative)/max(1.,np.linalg.norm(first),np.linalg.norm(first+delta)) < 1e-6
    # A corrupted position join must fail the native continuity validator,
    # independently of the unchanged animation keyframes.
    data["compiled"][0]["solution"]["segments"][1]["eye0"][0] += shot.subject.bounds[1][0]-shot.subject.bounds[0][0]
    with pytest.raises(CinemaInfeasible) as error:
        ShotPlan.from_json(json.dumps(data)).validate(source)
    assert error.value.binding_constraint == "C2_continuity"
    assert json.loads(error.value.diagnostic["detail"])["knot_continuity_relative"] >= 1e-6


def test_terrain_attached_subject_on_smooth_mountain():
    source,shot=load_scene("terrain_attached_mountain")
    result=plan([shot],source)
    report=result.validation[0]
    assert_report(report,shot)
    assert report["subject_visibility_samples"] == 4
    assert report["buried_subject_samples"] == 5
    print("terrain-attached mountain",json.dumps(report,sort_keys=True))


@pytest.mark.parametrize("kind,constraint", (("reveal","reveal_intent"),("dolly","dolly_intent")))
def test_fixed_endpoints_cannot_contradict_shot_intent(kind,constraint):
    original,shot=load_scene("knife_edge")
    source=TerrainScatterSource(np.zeros_like(original.heightmap),terrain_width=100.)
    invalid=replace(shot,kind=kind,start_eye=(10.,55.,50.),end_eye=(90.,55.,50.))
    with pytest.raises(CinemaInfeasible) as error:plan([invalid],source)
    assert error.value.binding_constraint == constraint


def test_full_orbit_preserves_radius_on_emitted_consumer():
    original,shot=load_scene("knife_edge")
    source=TerrainScatterSource(np.zeros_like(original.heightmap),terrain_width=100.)
    orbit=replace(shot,kind="orbit",start_eye=(90.,55.,50.),end_eye=(90.,55.,50.),
                  bearing_deg=(0.,360.),duration_s=2*shot.duration_s)
    result=plan([orbit],source)
    assert_report(result.validation[0],orbit)
    saved=json.loads(result.to_json())["compiled"][0]["solution"]["segments"]
    assert len(saved) == 1 and saved[0]["arc"]["sweep"] == 2*np.pi
    # Reconstruct the public animation's returned spherical pose, independently
    # of the validator's radius metric, at the acceptance sampling cadence.
    animation=result.bake()
    center=np.mean(orbit.subject.bounds,axis=0)
    radius=40.
    radii=[]
    for time_s in np.linspace(0,orbit.duration_s,int(orbit.duration_s*240)+1):
        state=animation.evaluate(float(time_s))
        phi,theta=np.deg2rad([state.phi_deg,state.theta_deg])
        eye=np.asarray(state.target)+state.radius*np.array([np.sin(theta)*np.cos(phi),np.cos(theta),np.sin(theta)*np.sin(phi)])
        radii.append(np.linalg.norm((eye-center)[[0,2]]))
    deviation=max(abs(np.asarray(radii)/radius-1))
    # Here subject distance equals horizontal radius, so the original
    # resampling contract bounds radius error by the same 0.1%.
    assert deviation < .001
    for fraction in (.25,.5,.75):
        t=orbit.duration_s*fraction
        a=animation.evaluate(t-1/240)
        b=animation.evaluate(t+1/240)
        assert b.phi_deg > a.phi_deg
    print("full orbit consumer radius",json.dumps({"minimum":min(radii),"maximum":max(radii),"relative_error":deviation}))


@pytest.mark.parametrize("kind", ("reveal", "orbit"))
def test_off_center_framing_is_solved_and_physically_validated(kind):
    source,shot=load_scene("knife_edge")
    shot=replace(shot,framing=replace(shot.framing,region=(.65,.45,.75,.55),size_frac_band=(.04,.25)))
    if kind == "orbit":
        shot=replace(shot,kind=kind,start_eye=(99.,55.,50.),end_eye=(50.,55.,99.),
                     bearing_deg=(0.,90.),duration_s=30.)
    result=plan([shot],source)
    assert_report(result.validation[0],shot)
    record_case_report("off-center-framing-"+kind, result.validation)
    assert json.loads(result.keyframes_json)[0]["target"] != list(np.mean(shot.subject.bounds,axis=0))


def test_fixed_and_joined_orbit_preserve_declared_winding():
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=100.)
    orbit = replace(shot, kind="orbit", start_eye=(50.,55.,10.), end_eye=None,
                    bearing_deg=(270.,360.), duration_s=30.)
    reveal = replace(shot, start_eye=(50.,54.,20.), end_eye=orbit.start_eye)
    individual = plan([orbit], source)
    joined = plan([reveal, replace(orbit, start_eye=None)], source)
    spans = []
    for result in (individual, joined):
        saved = json.loads(result.to_json())["compiled"][-1]["solution"]["segments"]
        assert len(saved) == 1
        assert saved[0]["arc"]["sweep"] == np.pi / 2
        assert saved[0]["eye0"] == list(orbit.start_eye)
        animation = result.bake()
        start_time = sum(part.duration_s for part in result.shots[:-1])
        span = animation.evaluate(start_time + orbit.duration_s).phi_deg - animation.evaluate(start_time).phi_deg
        # Public angles are float32; use that representation's relative precision.
        np.testing.assert_allclose(span, 90., rtol=np.finfo(np.float32).eps, atol=0.)
        spans.append(span)
        for report, part in zip(result.validation, result.shots):
            assert_report(report, part)
        assert ShotPlan.from_json(result.to_json()).to_json() == result.to_json()
    assert joined.shots[1].start_eye == orbit.start_eye
    assert plan([orbit], source).keyframes_json == individual.keyframes_json
    record_case_report("fixed-and-joined-270-to-360", joined.validation)
    print("orbit-winding-span", json.dumps({"individual_deg": spans[0], "joined_deg": spans[1]}))
    with pytest.raises(CinemaInfeasible) as caught:
        plan([replace(orbit, bearing_deg=(180.,360.))], source)
    assert caught.value.binding_constraint == "orbit_start_bearing"


@pytest.mark.parametrize("bearings", ((0.,90.), (360.,450.), (0.,-90.), (360.,-360.)))
def test_orbit_signed_sweep_and_equivalent_windings(bearings):
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=100.)
    orbit = replace(shot, kind="orbit", start_eye=(90.,55.,50.), end_eye=None,
                    bearing_deg=bearings, duration_s=60.)
    result = plan([orbit], source)
    report = result.orbit_adjustments[0]
    sweep = bearings[1] - bearings[0]
    assert report["signed_sweep_deg"] == sweep
    assert report["effective_bearing_deg"] == list(bearings)
    segments = json.loads(result.to_json())["compiled"][0]["solution"]["segments"]
    assert sum(s.get("arc", {}).get("sweep", 0.) for s in segments) == np.radians(sweep)
    animation = result.bake()
    np.testing.assert_allclose(animation.evaluate(orbit.duration_s).phi_deg - animation.evaluate(0.).phi_deg,
                               sweep, rtol=np.finfo(np.float32).eps)
    assert_report(result.validation[0], orbit)
    assert ShotPlan.from_json(result.to_json()).to_json() == result.to_json()


def test_rounded_orbit_endpoints_preserve_start_and_report_projection():
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=100.)
    orbit = replace(shot, kind="orbit", start_eye=(90.01,55.,50.01),
                    end_eye=(50.,55.,90.01), bearing_deg=(0.,90.), duration_s=30.)
    result = plan([orbit], source)
    adjustment = result.orbit_adjustments[0]
    assert adjustment["effective_start_eye"] == list(orbit.start_eye)
    np.testing.assert_allclose(adjustment["effective_end_eye"], (49.99,55.,90.01), atol=1e-12)
    assert 0 < adjustment["end_adjustment_distance"] < .001 * np.linalg.norm(np.asarray(orbit.end_eye)-np.mean(orbit.subject.bounds,axis=0))
    assert adjustment["effective_bearing_deg"][0] != 0.
    assert adjustment["signed_sweep_deg"] == 90.
    assert_report(result.validation[0], orbit)
    for invalid in (replace(orbit, start_eye=(90.,55.,51.)), replace(orbit, end_eye=(51.,55.,90.))):
        with pytest.raises(CinemaInfeasible) as caught:
            plan([invalid], source)
        assert caught.value.binding_constraint in ("orbit_start_bearing", "orbit_endpoint")


def test_joined_orbit_derives_bearing_from_exact_preceding_eye():
    original, shot = load_scene("knife_edge")
    source = TerrainScatterSource(np.zeros_like(original.heightmap), terrain_width=100.)
    reveal = replace(shot, start_eye=(50.,54.,20.), end_eye=(50.,55.,10.))
    orbit = replace(shot, kind="orbit", start_eye=None, end_eye=None,
                    bearing_deg=(0.,-90.), duration_s=30.)
    result = plan([reveal, orbit], source)
    compiled = json.loads(result.to_json())["compiled"]
    assert compiled[1]["solution"]["segments"][0]["eye0"] == compiled[0]["solution"]["segments"][-1]["eye1"]
    adjustment = result.orbit_adjustments[1]
    assert adjustment["joined_start"]
    assert adjustment["requested_bearing_deg"] == [0.,-90.]
    assert adjustment["effective_bearing_deg"] == [-90.,-180.]
    assert adjustment["signed_sweep_deg"] == -90.
    for report, effective in zip(result.validation, result.shots):
        assert_report(report, effective)
    assert ShotPlan.from_json(result.to_json()).to_json() == result.to_json()
    with pytest.raises(CinemaInfeasible) as caught:
        plan([reveal, replace(orbit, start_eye=(50.,55.,10.01))], source)
    assert caught.value.binding_constraint == "shot_join"


def test_motion_conditioned_flyover_joins_orbit_in_the_same_consumer_chart():
    source, flyover = load_scene("valley_col")
    orbit = replace(flyover, kind="orbit", start_eye=None, end_eye=None,
                    bearing_deg=(360.,450.), duration_s=60.)
    result = plan([flyover, orbit], source)
    saved = json.loads(result.to_json())
    assert saved["compiled"][0]["solution"]["segments"][-1]["eye1"] == saved["compiled"][1]["solution"]["segments"][0]["eye0"]
    for report, shot in zip(result.validation, result.shots):
        assert_report(report, shot)
    assert result.orbit_adjustments[1]["signed_sweep_deg"] == 90.
    assert ShotPlan.from_json(result.to_json()).to_json() == result.to_json()


def test_disconnected_corridor_names_binding_constraint():
    _,shot=load_scene("knife_edge")
    source=TerrainScatterSource(np.zeros((17,17),np.float32),terrain_width=100.)
    shot=replace(shot,kind="flyover",subject=Subject(((47.,67.,47.),(53.,73.,53.))),
                 framing=replace(shot.framing,size_frac_band=(.055,.13)),
                 start_eye=(5.,3.,5.),end_eye=(95.,3.,95.))
    with pytest.raises(CinemaInfeasible) as error:plan([shot],source)
    assert error.value.binding_constraint == "clearance_framing_corridor"


def test_fully_embedded_subject_is_explicitly_rejected():
    source,shot=load_scene("knife_edge")
    shot=replace(shot,subject=Subject(((47.,-6.,47.),(53.,-1.,53.))))
    with pytest.raises(CinemaInfeasible) as error:plan([shot],source)
    assert error.value.binding_constraint == "subject_embedded"


def test_zero_transitions_do_not_hide_endpoint_framing_failures():
    source,shot=load_scene("knife_edge")
    artifact=json.loads(plan([shot],source).to_json())
    for key in artifact["keyframes"]:key["fov"]=1.
    with pytest.raises(CinemaInfeasible) as error:ShotPlan.from_json(json.dumps(artifact)).validate(source)
    assert error.value.binding_constraint == "framing"
    report=json.loads(error.value.diagnostic["detail"])
    assert report["framing_outside_transitions"] == report["samples"]


def test_gpu_and_capacity_errors_are_environment_failures():
    from forge3d.cinema import _call
    for constraint in ("gpu_trace","validation_resolution"):
        def unavailable():
            raise ValueError(json.dumps({"status":"infeasible","binding_constraint":constraint,"detail":"unavailable test environment"}))
        with pytest.raises(CinemaNativeUnavailable):_call(unavailable)
    # Exercise the real native resolution boundary as well as error routing.
    source,shot=load_scene("knife_edge")
    too_long=replace(shot,kind="dolly",start_eye=(10.,55.,50.),end_eye=(25.,55.,50.),
                     duration_s=float(np.finfo(np.float32).max))
    with pytest.raises(CinemaNativeUnavailable,match="validation_resolution"):
        plan([too_long],source)


def test_motion_budget_is_below_native_roundoff_and_rejects_impossible_duration():
    source,shot=load_scene("knife_edge")
    shot=replace(shot,kind="dolly",start_eye=(10.,55.,50.),end_eye=(25.,55.,50.))
    # Independent analytic budget: duplicated endpoint controls of a sampled
    # quintic have limiting cubic jerk 150*displacement/duration^3 (versus 60
    # for the quintic itself). Declare it before solving; never fit to a report.
    constrained=replace(shot,max_jerk=150*15/shot.duration_s**3)
    result=plan([constrained],source)
    report=result.validation[0]
    assert_report(report,constrained)
    assert report["max_jerk"] < report["sampled_max_jerk"]
    assert report["consumer_max_acceleration_jump"] > 0
    assert report["consumer_fd_max_jerk"] > report["max_jerk"]
    print("analytic-budget-control",json.dumps({"max_jerk":constrained.max_jerk,
          "source":"150*displacement/duration^3, declared before solving",
          "report":report},sort_keys=True))
    # Half the analytically required rest-to-rest jerk cannot fit this duration.
    minimum=60*15/shot.duration_s**3
    with pytest.raises(CinemaInfeasible) as error:plan([replace(shot,max_jerk=minimum/2)],source)
    assert error.value.binding_constraint == "duration_speed_jerk"


def test_declared_zero_refinement_budget_refuses_unproven_consumer():
    source,shot=load_scene("knife_edge")
    shot=replace(shot,kind="dolly",start_eye=(10.,55.,50.),end_eye=(25.,55.,50.),max_refine_passes=0)
    with pytest.raises(CinemaInfeasible) as error:plan([shot],source)
    assert error.value.binding_constraint == "consumer_resampling"


@pytest.mark.parametrize("kind", ("reveal", "orbit", "flyover", "dolly"))
def test_intent_generates_endpoints_inside_terrain(kind):
    source, shot = load_scene("knife_edge")
    shot = replace(shot, kind=kind, start_eye=None, end_eye=None,
                   bearing_deg=(180.,180. if kind == "dolly" else 0.),
                   elevation_deg=(-20.,20. if kind == "reveal" else -20.), duration_s=30.)
    result = plan([shot], source)
    assert_report(result.validation[0], shot)
    record_case_report("generated-"+kind, result.validation)
    segments = json.loads(result.to_json())["compiled"][0]["solution"]["segments"]
    if kind == "reveal":
        assert segments[-1]["eye1"][1] > segments[0]["eye0"][1]
    if kind == "dolly":
        center=np.mean(shot.subject.bounds,axis=0)
        assert np.linalg.norm(np.asarray(segments[-1]["eye1"])-center) < np.linalg.norm(np.asarray(segments[0]["eye0"])-center)
