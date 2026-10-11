use super::{solver::Terrain, Segment};
use glam::DVec3;

#[test]
fn analytic_arc_keeps_moving_and_matches_hand_derivatives() {
    let segment = Segment {
        t0: 0.0,
        t1: 2.0,
        eye0: [10.0, 3.0, 0.0],
        eye1: [0.0, 3.0, 10.0],
        target0: [0.0; 3],
        target1: [0.0; 3],
        arc: Some(super::Arc {
            center: [0.0; 3],
            sweep: std::f64::consts::FRAC_PI_2,
        }),
    };
    assert!(segment.arc.as_ref().unwrap().valid(&segment));
    let radial = DVec3::new(5.0 * 2.0_f64.sqrt(), 0.0, 5.0 * 2.0_f64.sqrt());
    let tangent = DVec3::new(-radial.z, 0.0, radial.x);
    let omega = std::f64::consts::FRAC_PI_2 * 1.875 / 2.0;
    let angular_jerk = -30.0 * std::f64::consts::FRAC_PI_2 / 8.0;
    assert!(segment.sample(1.0, 0).0.distance(radial + DVec3::Y * 3.0) < 1e-6);
    assert!(segment.sample(1.0, 1).0.distance(tangent * omega) < 1e-6);
    assert!(segment.sample(1.0, 2).0.distance(-radial * omega.powi(2)) < 1e-6);
    assert!(
        segment
            .sample(1.0, 3)
            .0
            .distance(tangent * (angular_jerk - omega.powi(3)))
            < 1e-6
    );
    for i in 1..480 {
        let t = i as f64 / 240.0;
        assert!(segment.sample(t, 1).0.length() > 0.0);
        assert!(segment.sample(t, 3).0.length() <= segment.jerk_bound());
    }
    for t in [0.0, 2.0] {
        assert_eq!(segment.sample(t, 1).0, DVec3::ZERO);
        assert_eq!(segment.sample(t, 2).0, DVec3::ZERO);
    }
    let mut broken = segment.clone();
    broken.eye1[0] = 1.0;
    assert!(!broken.arc.as_ref().unwrap().valid(&broken));
}

#[test]
fn legacy_segment_json_remains_byte_identical() {
    let json = r#"{"t0":0.0,"t1":1.0,"eye0":[1.0,2.0,3.0],"eye1":[2.0,3.0,4.0],"target0":[0.0,0.0,0.0],"target1":[0.0,0.0,0.0]}"#;
    let segment: Segment = serde_json::from_str(json).unwrap();
    assert!(segment.arc.is_none());
    assert_eq!(serde_json::to_string(&segment).unwrap(), json);
}

#[test]
fn off_center_arc_keeps_exact_shared_endpoint_targets() {
    let mut s = shot();
    s.kind = "orbit".into();
    s.duration_s = 30.0;
    s.framing.region = [0.65, 0.45, 0.75, 0.55];
    s.framing.size_frac_band = [0.04, 0.25];
    s.start_eye = Some([99.0, 55.0, 50.0]);
    s.end_eye = Some([50.0, 55.0, 99.0]);
    s.bearing_deg = [0.0, 90.0];
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let solution = super::solver::solve(&s, &terrain).unwrap();
    assert_eq!(solution.segments.len(), 1);
    let target = super::solver::target(DVec3::from(s.end_eye.unwrap()), &s).unwrap();
    assert_eq!(solution.sample(s.duration_s, 0).1, target);
    assert!(solution.segments[0]
        .arc
        .as_ref()
        .unwrap()
        .valid(&solution.segments[0]));
    s.start_eye = None;
    s.end_eye = None;
    s.bearing_deg = [270.0, 360.0];
    s.elevation_deg = [20.0, 20.0];
    let generated = super::solver::solve(&s, &terrain).unwrap();
    let arc = generated
        .segments
        .iter()
        .find_map(|segment| segment.arc.as_ref())
        .unwrap();
    assert_eq!(arc.sweep, std::f64::consts::FRAC_PI_2);
}

#[test]
fn fixed_orbit_start_retains_declared_winding_and_rejects_wrong_bearing() {
    let mut s = shot();
    s.kind = "orbit".into();
    s.duration_s = 30.0;
    s.start_eye = Some([50.0, 55.0, 10.0]);
    s.end_eye = None;
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    for bearings in [[270.0, 360.0], [-90.0, 0.0], [630.0, 720.0], [270.0, 180.0]] {
        s.bearing_deg = bearings;
        let solution = super::solver::solve(&s, &terrain).unwrap();
        let arc = solution.segments[0].arc.as_ref().unwrap();
        assert_eq!(arc.sweep, (bearings[1] - bearings[0]).to_radians());
        assert_eq!(solution.segments[0].eye0, s.start_eye.unwrap());
        assert!(arc.valid(&solution.segments[0]));
    }
    s.bearing_deg = [180.0, 360.0];
    assert!(super::solver::solve(&s, &terrain)
        .unwrap_err()
        .contains("orbit_start_bearing"));
}

#[test]
fn rounded_orbit_inputs_project_end_without_moving_start_or_changing_sweep() {
    let mut s = shot();
    s.kind = "orbit".into();
    s.duration_s = 30.0;
    s.start_eye = Some([90.01, 55.0, 50.01]);
    s.end_eye = Some([50.0, 55.0, 90.01]);
    s.bearing_deg = [360.0, 450.0];
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let solution = super::solver::solve(&s, &terrain).unwrap();
    assert_eq!(solution.segments[0].eye0, s.start_eye.unwrap());
    let final_eye = DVec3::from(solution.segments.last().unwrap().eye1);
    assert!(final_eye.distance(DVec3::new(49.99, 55.0, 90.01)) < 1e-12);
    let arc = solution
        .segments
        .iter()
        .find_map(|s| s.arc.as_ref())
        .unwrap();
    assert_eq!(arc.sweep, 90.0_f64.to_radians());
    s.end_eye = Some([51.0, 55.0, 90.0]);
    assert!(super::solver::solve(&s, &terrain)
        .unwrap_err()
        .contains("orbit_endpoint"));
}

#[test]
fn consumer_pole_crossing_uses_a_continuous_equivalent_spherical_chart() {
    let s = shot();
    let center = s.subject.center();
    let segment = Segment {
        t0: 0.0,
        t1: s.duration_s,
        eye0: (center + DVec3::new(-40.0, -32.0, 0.0)).to_array(),
        eye1: (center + DVec3::new(40.0, -32.0, 0.0)).to_array(),
        target0: center.to_array(),
        target1: center.to_array(),
        arc: None,
    };
    let solution = super::Solution {
        segments: vec![segment],
        expanded_nodes: 0,
        corridor: "polar control".into(),
    };
    let keys = super::validate::resample(&solution, &s, 60).unwrap();
    assert!(keys.iter().all(|k| k.phi == 180.0));
    assert!(keys.windows(2).all(|k| k[1].theta >= k[0].theta));
    assert!(super::validate::approximation(&s, &solution, &keys, 0.0).unwrap() < 0.001);
    let consumer = super::motion::Consumer::new(&keys).unwrap();
    assert!(consumer.jerk_bound(0.0, s.duration_s) <= s.max_jerk);
}

#[test]
fn motion_alternative_preserves_intent_endpoints_budgets_and_flat_trace_control() {
    let mut s = shot();
    s.kind = "flyover".into();
    s.start_eye = Some([10.0, 4.0, 50.0]);
    s.end_eye = Some([90.0, 4.0, 50.0]);
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let original = super::solver::solve(&s, &terrain).unwrap();
    let alternative = super::solver::motion_alternative(&s, &terrain, &original).unwrap();
    assert_eq!(alternative.segments[0].eye0, s.start_eye.unwrap());
    assert_eq!(
        alternative.segments.last().unwrap().eye1,
        s.end_eye.unwrap()
    );
    assert_eq!(alternative.segments.len(), 1);
    assert!(alternative.segments[0].arc.is_some());
    assert!(alternative.segments[0].jerk_bound() <= s.max_jerk);
    for i in 0..=240 {
        assert_eq!(
            alternative.sample(s.duration_s * i as f64 / 240.0, 0).0.y,
            4.0
        );
    }
    let keys = super::validate::resample(&alternative, &s, 20).unwrap();
    assert!(super::validate::approximation(&s, &alternative, &keys, 0.0).unwrap() < 0.001);
    assert!(
        super::motion::Consumer::new(&keys)
            .unwrap()
            .jerk_bound(0.0, s.duration_s)
            <= s.max_jerk
    );
    s.kind = "dolly".into();
    assert!(super::solver::motion_alternative(&s, &terrain, &original).is_err());
}

#[test]
fn orbit_clearance_rectangles_enclose_arc_and_do_not_create_motion_knots() {
    let mut f: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/data/cinema_scenes/summit_cliff.json"
    ))
    .unwrap();
    f["shot"]["max_refine_passes"] = serde_json::json!(0);
    let s: super::Shot = serde_json::from_value(f["shot"].clone()).unwrap();
    let rows: Vec<Vec<f32>> = serde_json::from_value(f["heightmap"].clone()).unwrap();
    let heights = rows.into_iter().flatten().collect();
    let terrain = Terrain::new(heights, 17, 17, 100.0).unwrap();
    let solution = super::solver::solve(&s, &terrain).unwrap();
    assert_eq!(
        solution
            .segments
            .iter()
            .filter(|segment| segment.arc.is_some())
            .count(),
        1
    );
    assert_eq!(solution.segments.len(), 3);
    for i in 0..=7200 {
        let (eye, _) = solution.sample(i as f64 / 240.0, 0);
        assert!(eye.y - terrain.height(eye) >= s.clearance);
    }
    for pair in solution.segments.windows(2) {
        for derivative in 0..=2 {
            let a = pair[0].sample(pair[0].t1, derivative);
            let b = pair[1].sample(pair[1].t0, derivative);
            assert!(a.0.distance(b.0) < 1e-6);
            assert!(a.1.distance(b.1) < 1e-6);
        }
    }
}

fn shot() -> super::Shot {
    let mut f: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/data/cinema_scenes/knife_edge.json"
    ))
    .unwrap();
    f["shot"]["max_refine_passes"] = serde_json::json!(0);
    serde_json::from_value(f["shot"].clone()).unwrap()
}

#[test]
fn empty_transitions_do_not_exempt_endpoints() {
    let mut s = shot();
    assert!(!super::validate::in_transition(&s, 0.0));
    assert!(!super::validate::in_transition(&s, s.duration_s));
    s.transition = [1.0, 1.0];
    assert!(super::validate::in_transition(&s, 0.0));
    assert!(super::validate::in_transition(&s, s.duration_s));
    assert!(!super::validate::in_transition(&s, s.duration_s / 2.0));
}

#[test]
fn buried_subject_samples_are_explicit_and_fully_embedded_subjects_reject() {
    let t = Terrain::new(vec![40.0; 9], 3, 3, 100.0).unwrap();
    let mut s = shot();
    s.subject.bounds = [[47.0, 36.0, 47.0], [53.0, 42.0, 53.0]];
    let points = t.subject_points(&s).unwrap();
    assert_eq!(points.len(), 4);
    assert!(points.iter().all(|p| p.y == 42.0));
    s.subject.bounds[1][1] = 39.0;
    assert!(t
        .subject_points(&s)
        .unwrap_err()
        .contains("subject_embedded"));
}

#[test]
fn sight_ray_uses_bilinear_surface_without_neighbor_cell_false_block() {
    let mut heights = vec![0.0; 17 * 17];
    for row in heights.chunks_mut(17) {
        row[8] = 40.0;
    }
    let t = Terrain::new(heights, 17, 17, 100.0).unwrap();
    let eye = DVec3::new(10.0, 8.0, 50.0);
    assert!(t.visible(eye, DVec3::new(53.0, 52.0, 47.0)));
    assert!(!t.visible(eye, DVec3::new(90.0, 8.0, 50.0)));
    // Along a diagonal a bilinear saddle has an interior quadratic maximum.
    let saddle = Terrain::new(vec![0.0, 10.0, 10.0, 0.0], 2, 2, 1.0).unwrap();
    assert!(!saddle.visible(DVec3::new(0.0, 4.0, 0.0), DVec3::new(1.0, 4.0, 1.0)));
    assert!(saddle.visible(DVec3::new(0.0, 6.0, 0.0), DVec3::new(1.0, 6.0, 1.0)));
}

#[test]
fn disconnected_framing_corridor_rejects_after_valid_endpoints() {
    let t = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let mut s = shot();
    s.kind = "flyover".into();
    s.subject.bounds = [[47.0, 67.0, 47.0], [53.0, 73.0, 53.0]];
    s.framing.size_frac_band = [0.055, 0.13];
    s.start_eye = Some([5.0, 3.0, 5.0]);
    s.end_eye = Some([95.0, 3.0, 95.0]);
    assert!(super::solver::solve(&s, &t)
        .unwrap_err()
        .contains("clearance_framing_corridor"));
}

#[test]
fn consumer_knot_acceleration_jump_is_separate_from_interval_jerk() {
    let keys: Vec<_> = [0.0, 0.0, 1.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(i, x)| super::validate::Keyframe {
            time: i as f64,
            phi: 0.0,
            theta: 90.0,
            radius: 10.0,
            fov: 55.0,
            target: [x, 0.0, 0.0],
        })
        .collect();
    let c = super::motion::Consumer::new(&keys).unwrap();
    // Cubic controls [0,0,1,1] yield jerk -6. Adjacent controls yield +3;
    // their accelerations jump by 1 at each interior knot.
    assert!((c.jerk_bound(0.0, 3.0) - 6.0).abs() < 1e-6);
    assert!((c.acceleration_jump(0.0, 3.0) - 1.0).abs() < 1e-6);
    // The shared pose has distinct one-sided jerks. A shot ending at t=1
    // must report the left interval, not the next shot's right derivative.
    assert_eq!(c.sample_before(1.0).0[0], c.sample(1.0).0[0]);
    assert!((c.sample_before(1.0).0[3].length() - 3.0).abs() < 1e-6);
    assert!((c.sample(1.0).0[3].length() - 6.0).abs() < 1e-6);
    assert!((c.jerk_bound(0.0, 1.0) - 3.0).abs() < 1e-6);
    let maximum = |h: f64| {
        (0..4)
            .map(|i| c.sample(1.0 + (i as f64 - 1.0) * h).0[0])
            .collect::<Vec<_>>()
    };
    let third_difference = |h: f64| {
        let p = maximum(h);
        (p[3] - 3.0 * p[2] + 3.0 * p[1] - p[0]).length() / h.powi(3)
    };
    assert!(third_difference(1.0 / 480.0) > third_difference(1.0 / 240.0));
    assert!((c.jerk_bound(0.0, 3.0) - 6.0).abs() < 1e-6);
}

#[test]
fn invalid_reveal_and_dolly_endpoints_name_the_intent() {
    let mut s = shot();
    let a = DVec3::new(10.0, 55.0, 50.0);
    let b = DVec3::new(90.0, 55.0, 50.0);
    assert!(super::solver::check_intent(&s, a, b)
        .unwrap_err()
        .contains("reveal_intent"));
    s.kind = "dolly".into();
    assert!(super::solver::check_intent(&s, a, b)
        .unwrap_err()
        .contains("dolly_intent"));
    assert!(super::solver::check_intent(&s, a, DVec3::new(25.0, 55.0, 50.0)).is_ok());
}

#[test]
fn dolly_is_straight_and_obstructions_cannot_trigger_a_detour() {
    let mut s = shot();
    s.kind = "dolly".into();
    s.start_eye = Some([10.0, 55.0, 50.0]);
    s.end_eye = Some([25.0, 55.0, 50.0]);
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let solution = super::solver::solve(&s, &terrain).unwrap();
    assert_eq!(solution.segments.len(), 1);
    for i in 0..=240 {
        let eye = solution.sample(s.duration_s * i as f64 / 240.0, 0).0;
        assert_eq!(eye.y, 55.0);
        assert_eq!(eye.z, 50.0);
        assert!((10.0..=25.0).contains(&eye.x));
    }
    // Put the low obstacle strictly between clear endpoints. It violates the
    // camera's clearance, but is below every endpoint-to-subject sight ray.
    s.start_eye = Some([10.0, 12.0, 50.0]);
    s.end_eye = Some([35.0, 12.0, 50.0]);
    let clear = super::solver::solve(&s, &terrain).unwrap();
    let mut heights = vec![0.0; 17 * 17];
    heights[8 * 17 + 4] = 10.0;
    let blocked = Terrain::new(heights, 17, 17, 100.0).unwrap();
    super::solver::endpoints(
        s.start_eye.unwrap().into(),
        s.end_eye.unwrap().into(),
        &blocked,
        &s,
        super::solver::shell(&s),
    )
    .unwrap();
    let error = super::solver::solve(&s, &blocked).unwrap_err();
    assert!(error.contains("clearance") && error.contains("straight dolly approach"));
    assert_eq!(
        blocked.line_clearance(s.start_eye.unwrap().into(), s.end_eye.unwrap().into()),
        2.0
    );
    let error = super::solver::repair(&s, &blocked, &clear, s.duration_s * 0.55, true).unwrap_err();
    assert!(error.contains("clearance") && !error.contains("dolly_intent"));
    let mut heights = vec![0.0; 17 * 17];
    heights[8 * 17 + 7] = 50.0;
    let occluded = Terrain::new(heights, 17, 17, 100.0).unwrap();
    let error = super::solver::repair(&s, &occluded, &clear, s.duration_s / 2.0, true).unwrap_err();
    assert!(error.contains("occlusion") && !error.contains("dolly_intent"));
    let mut misaimed = clear.clone();
    for segment in &mut misaimed.segments {
        segment.target0 = [10.0, 12.0, 0.0];
        segment.target1 = [35.0, 12.0, 0.0];
    }
    let error =
        super::solver::repair(&s, &terrain, &misaimed, s.duration_s / 2.0, false).unwrap_err();
    assert!(error.contains("framing") && !error.contains("dolly_intent"));
    assert!(
        super::solver::repair(&s, &terrain, &solution, s.duration_s / 2.0, true)
            .unwrap_err()
            .contains("repair_not_applicable")
    );
    for time in [0.0, s.duration_s] {
        assert!(super::solver::repair(&s, &terrain, &clear, time, true)
            .unwrap_err()
            .contains("fixed_endpoint"));
    }
    let mut outside = clear.clone();
    for segment in &mut outside.segments {
        segment.eye0[0] = -10.0;
        segment.eye1[0] = -5.0;
    }
    assert!(
        super::solver::repair(&s, &terrain, &outside, s.duration_s / 2.0, true)
            .unwrap_err()
            .contains("terrain_bounds")
    );
    s.start_eye = Some([10.0, 55.0, 50.0]);
    s.end_eye = Some([80.0, 55.0, 50.0]);
    assert!(super::solver::solve(&s, &terrain)
        .unwrap_err()
        .contains("framing_distance"));
}

#[test]
fn dolly_uses_the_actual_line_not_the_enclosing_terrain_box() {
    let mut s = shot();
    s.kind = "dolly".into();
    s.start_eye = Some([10.0, 12.0, 10.0]);
    s.end_eye = Some([35.0, 12.0, 35.0]);
    let mut heights = vec![0.0; 17 * 17];
    heights[2 * 17 + 5] = 50.0;
    let terrain = Terrain::new(heights, 17, 17, 100.0).unwrap();
    let a = DVec3::from(s.start_eye.unwrap());
    let b = DVec3::from(s.end_eye.unwrap());
    assert!(terrain.ceiling(a, b, s.clearance) > a.y);
    assert_eq!(terrain.line_clearance(a, b), 12.0);
    let solution = super::solver::solve(&s, &terrain).unwrap();
    assert_eq!(solution.segments.len(), 1);
    assert_eq!(solution.segments[0].eye0, a.to_array());
    assert_eq!(solution.segments[0].eye1, b.to_array());
}

#[test]
fn straight_clearance_finds_bilinear_interior_extrema_on_sloping_lines() {
    let terrain = Terrain::new(
        vec![0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        3,
        3,
        100.0,
    )
    .unwrap();
    // In this cell, H(u)=40u(1-u). G(u)=12-20u+40u² has its
    // minimum 9.5 at u=1/4, neither an endpoint nor the midpoint.
    assert_eq!(
        terrain.line_clearance(DVec3::new(0.0, 12.0, 0.0), DVec3::new(50.0, 32.0, 50.0)),
        9.5
    );
    assert_eq!(
        terrain.line_clearance(DVec3::new(0.0, 10.0, 0.0), DVec3::new(50.0, 10.0, 50.0)),
        0.0
    );
}

#[test]
fn dolly_names_orientation_and_region_constraints_without_distance_mislabeling() {
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let mut s = shot();
    s.kind = "dolly".into();
    s.start_eye = Some([10.0, 100.0, 50.0]);
    s.end_eye = Some([80.0, 100.0, 50.0]);
    // Distances stay inside the shell, but the interior eye is vertically
    // above the subject: Y-up orientation, rather than framing distance, binds.
    let error = super::solver::solve(&s, &terrain).unwrap_err();
    assert!(error.contains("framing_orientation") && !error.contains("framing_distance"));
    s.start_eye = Some([49.0, 100.0, 50.0]);
    s.end_eye = Some([49.5, 100.0, 50.0]);
    s.framing.region = [0.65, 0.45, 0.75, 0.55];
    let error = super::solver::solve(&s, &terrain).unwrap_err();
    assert!(error.contains("framing_region") && !error.contains("framing_distance"));
}

#[test]
fn consumer_circular_jerk_matches_chain_rule_and_interval_bound() {
    let keys: Vec<_> = (0..4)
        .map(|i| super::validate::Keyframe {
            time: i as f64,
            phi: i as f64 * 10.0,
            theta: 90.0,
            radius: 10.0,
            fov: 55.0,
            target: [0.0; 3],
        })
        .collect();
    let c = super::motion::Consumer::new(&keys).unwrap();
    let expected = 10.0 * 10.0_f64.to_radians().powi(3);
    let (pose, target) = c.sample(1.5);
    assert!(target.iter().all(|v| *v == DVec3::ZERO));
    assert!((pose[3].length() - expected).abs() < 1e-6);
    assert!((c.jerk_bound(1.0, 2.0) - expected).abs() < 1e-6);
    // Independent symmetric third divided difference, with scale fixed by the
    // representable float64 interval for a third derivative.
    let h = f64::EPSILON.powf(1.0 / 5.0) / 10.0_f64.to_radians();
    let p = [-2.0, -1.0, 1.0, 2.0].map(|u| c.sample(1.5 + h * u).0[0]);
    let fd = (-p[0] + 2.0 * p[1] - 2.0 * p[2] + p[3]) / (2.0 * h.powi(3));
    assert!((fd.length() - expected).abs() < 1e-6);
    let translated: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let mut key = k.clone();
            key.radius += i as f64;
            key.target[0] = i as f64;
            key
        })
        .collect();
    let varying = super::motion::Consumer::new(&translated).unwrap();
    let w = 10.0_f64.to_radians();
    let expected = (9.0 * w.powi(4) + (11.5 * w.powi(3)).powi(2)).sqrt();
    assert!((varying.sample(1.5).0[3].length() - expected).abs() < 1e-6);
    assert!(varying.jerk_bound(1.0, 2.0) >= expected);
    let polar: Vec<_> = keys
        .iter()
        .map(|k| {
            let mut key = k.clone();
            key.theta = 179.0;
            key
        })
        .collect();
    let polar = super::motion::Consumer::new(&polar).unwrap();
    let expected = 10.0 * 179.0_f64.to_radians().sin() * w.powi(3);
    assert!((polar.sample(1.5).0[3].length() - expected).abs() < 1e-6);
    assert!((polar.jerk_bound(1.0, 2.0) - expected).abs() < 1e-6);
}

#[test]
fn minmax_corridor_dominates_independent_bilinear_probes() {
    let t = Terrain::new(vec![0.0, 1.0, 8.0, 2.0, 3.0, 9.0, 0.0, 4.0, 2.0], 3, 3, 2.0).unwrap();
    for z in 0..=32 {
        for x in 0..=32 {
            let p = DVec3::new(x as f64 / 16.0, 0.0, z as f64 / 16.0);
            assert!(t.ceiling(p, p, 0.0) >= t.height(p));
            let q = DVec3::new(2.0 - p.x, 0.0, 2.0 - p.z);
            let bound = t.ceiling(p, q, 0.0);
            for i in 0..=32 {
                assert!(t.height(p.lerp(q, i as f64 / 32.0)) <= bound);
            }
        }
    }
}

#[test]
fn quintic_is_c2_and_has_declared_derivative_extrema() {
    let s = Segment {
        arc: None,
        t0: 0.0,
        t1: 2.0,
        eye0: [0.0; 3],
        eye1: [2.0, 0.0, 0.0],
        target0: [0.0; 3],
        target1: [0.0, 2.0, 0.0],
    };
    for t in [s.t0, s.t1] {
        assert_eq!(s.sample(t, 1).0, DVec3::ZERO);
        assert_eq!(s.sample(t, 2).0, DVec3::ZERO);
    }
    assert_eq!(s.sample(1.0, 1).0.length(), 1.875);
    assert_eq!(s.sample(0.0, 3).0.length(), 15.0);
    let h = 1e-4;
    let velocity = (s.sample(1.0 + h, 0).0 - s.sample(1.0 - h, 0).0) / (2.0 * h);
    assert!((velocity - s.sample(1.0, 1).0).length() < 1e-6);
}

#[test]
fn framing_matches_hand_computed_perspective_and_off_center_target() {
    let mut fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/data/cinema_scenes/knife_edge.json"
    ))
    .unwrap();
    // This projection-only test requests no solver refinement.
    fixture["shot"]["max_refine_passes"] = serde_json::json!(0);
    let mut shot: super::Shot = serde_json::from_value(fixture["shot"].clone()).unwrap();
    shot.subject.bounds = [[-1.0; 3], [1.0; 3]];
    shot.framing.fov_deg = 90.0;
    shot.framing.aspect = 1.0;
    shot.framing.size_frac_band = [0.11, 0.112];
    shot.framing.headroom = 0.4;
    // At z=10 the nearest cube face is z=1: its screen width is 1/9.
    let eye = DVec3::new(0.0, 0.0, 10.0);
    assert!(super::validate::framing(&shot, eye, DVec3::ZERO, 90.0));
    assert!(!super::validate::framing(
        &shot,
        eye * 2.0,
        DVec3::ZERO,
        90.0
    ));
    shot.framing.region = [0.65, 0.45, 0.75, 0.55];
    shot.framing.size_frac_band = [0.1, 0.2];
    shot.framing.headroom = 0.0;
    let aim = super::solver::target(eye, &shot).unwrap();
    assert!(super::validate::framing(&shot, eye, aim, 90.0));
    assert!(!super::validate::framing(&shot, eye, DVec3::ZERO, 90.0));
}

#[test]
fn waypoint_repair_refits_a_clear_c2_path_within_motion_budgets() {
    let mut fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/data/cinema_scenes/knife_edge.json"
    ))
    .unwrap();
    fixture["shot"]["max_refine_passes"] = serde_json::json!(0);
    let shot: super::Shot = serde_json::from_value(fixture["shot"].clone()).unwrap();
    let terrain = Terrain::new(vec![0.0; 17 * 17], 17, 17, 100.0).unwrap();
    let before = super::solver::solve(&shot, &terrain).unwrap();
    let after =
        super::solver::repair(&shot, &terrain, &before, shot.duration_s * 0.25, true).unwrap();
    assert!(after.segments.len() > before.segments.len());
    for segment in &after.segments {
        let (eye_jerk, target_jerk) = segment.sample(segment.t0, 3);
        assert!(eye_jerk.length().max(target_jerk.length()) <= shot.max_jerk);
        for u in [0.0, 0.5, 1.0] {
            let t = segment.t0 + u * (segment.t1 - segment.t0);
            let (eye, _) = segment.sample(t, 0);
            assert!(terrain.inside(eye) && eye.y - terrain.height(eye) >= shot.clearance);
            let (velocity, _) = segment.sample(t, 1);
            assert!(velocity.length() <= shot.max_speed);
        }
    }
    for pair in after.segments.windows(2) {
        for order in 0..=2 {
            let (a, b) = pair[0].sample(pair[0].t1, order);
            let (c, d) = pair[1].sample(pair[1].t0, order);
            for (left, right) in [(a, c), (b, d)] {
                assert!(left.distance(right) / left.length().max(right.length()).max(1.0) < 1e-6);
            }
        }
    }
}
