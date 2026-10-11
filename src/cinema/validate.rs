//! Independent dense validation of the actual native consumer cubic.
use super::{binding, gpu_trace, motion, solver::Terrain, Shot, Solution};
use crate::animation::{CameraAnimation, CameraKeyframe};
use crate::camera::Anchor;
use crate::geo::units::{SceneCoord, SceneOffset};
use glam::{DVec3, Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Keyframe {
    pub time: f64,
    pub phi: f64,
    pub theta: f64,
    pub radius: f64,
    pub fov: f64,
    pub target: [f64; 3],
}
impl Keyframe {
    pub fn native(&self) -> CameraKeyframe {
        CameraKeyframe::new(
            self.time as f32,
            self.phi as f32,
            self.theta as f32,
            self.radius,
            self.fov as f32,
            Some(self.target),
        )
    }
}
pub fn animation(keys: &[Keyframe]) -> CameraAnimation {
    let mut a = CameraAnimation::new();
    a.replace_keyframes(keys.iter().map(Keyframe::native).collect());
    a
}
pub fn consumer(a: &CameraAnimation, t: f64) -> Result<(DVec3, DVec3), String> {
    let state = a
        .evaluate(t as f32)
        .ok_or_else(|| binding("consumer", "animation returned no state"))?;
    let target = DVec3::from(
        state
            .target
            .ok_or_else(|| binding("consumer", "animation lost target"))?,
    );
    let phi = f64::from(state.phi_deg).to_radians();
    let theta = f64::from(state.theta_deg).to_radians();
    let eye = target
        + DVec3::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        ) * state.radius;
    Ok((eye, target))
}
pub fn resample(solution: &Solution, shot: &Shot, panels: usize) -> Result<Vec<Keyframe>, String> {
    let mut keys = Vec::new();
    keys.try_reserve(
        panels
            .checked_add(1)
            .ok_or_else(|| binding("consumer_precision", "keyframe count overflow"))?,
    )
    .map_err(|e| binding("consumer_precision", e))?;
    let mut previous = 0.0;
    for i in 0..=panels {
        let t = shot.duration_s * i as f64 / panels as f64;
        let (eye, target) = solution.sample(t, 0);
        let d = eye - target;
        let r = d.length();
        let mut phi = d.z.atan2(d.x).to_degrees();
        if i > 0 {
            while phi - previous > 180.0 {
                phi -= 360.0;
            }
            while phi - previous < -180.0 {
                phi += 360.0;
            }
        }
        previous = phi;
        keys.push(Keyframe {
            time: f64::from(t as f32),
            phi: f64::from(phi as f32),
            theta: f64::from((d.y / r).clamp(-1.0, 1.0).acos().to_degrees() as f32),
            radius: r,
            fov: f64::from(shot.framing.fov_deg as f32),
            target: target.to_array(),
        });
        if i > 0 && shot.kind != "orbit" {
            // The same spherical eye has two charts: (phi, theta) and
            // (phi+180, -theta). Keep the nearest continuous chart across a
            // pole, instead of injecting an azimuth spin into the consumer.
            // Operate on the already narrowed keys; native() remains the
            // authoritative float32 boundary and its jerk gate is unchanged.
            let prior = &keys[i - 1];
            let mut current = keys[i].clone();
            current.theta += 360.0 * ((prior.theta - current.theta) / 360.0).round();
            let alternate_phi =
                current.phi + 180.0 + 360.0 * ((prior.phi - current.phi - 180.0) / 360.0).round();
            let alternate_theta =
                -current.theta + 360.0 * ((prior.theta + current.theta) / 360.0).round();
            if (alternate_phi - prior.phi).powi(2) + (alternate_theta - prior.theta).powi(2)
                < (current.phi - prior.phi).powi(2) + (current.theta - prior.theta).powi(2)
            {
                current.phi = alternate_phi;
                current.theta = alternate_theta;
            }
            previous = current.phi;
            keys[i] = current;
        }
    }
    Ok(keys)
}

pub fn framing(shot: &Shot, eye: DVec3, target: DVec3, fov_deg: f64) -> bool {
    let f = &shot.framing;
    let view = Mat4::look_at_rh(
        Vec3::ZERO,
        Anchor::offset_to_render(SceneOffset::scene(target - eye)),
        Vec3::Y,
    );
    let projection = crate::camera::perspective_wgpu(
        (fov_deg as f32).to_radians(),
        f.aspect as f32,
        0.01,
        eye.distance(shot.subject.center()) as f32 * 4.0 + 1.0,
    );
    let vp = projection * view;
    let project = |p: DVec3| {
        let q = vp * Anchor::offset_to_render(SceneOffset::scene(p - eye)).extend(1.0);
        if q.w <= 0.0 || !q.is_finite() {
            return None;
        }
        Some([
            0.5 + 0.5 * f64::from(q.x / q.w),
            0.5 - 0.5 * f64::from(q.y / q.w),
        ])
    };
    let Some(center) = project(shot.subject.center()) else {
        return false;
    };
    if center[0] < f.region[0]
        || center[0] > f.region[2]
        || center[1] < f.region[1]
        || center[1] > f.region[3]
    {
        return false;
    }
    let mut lower = [f64::INFINITY; 2];
    let mut upper = [f64::NEG_INFINITY; 2];
    for p in shot.subject.corners() {
        let Some(q) = project(p) else {
            return false;
        };
        for i in 0..2 {
            lower[i] = lower[i].min(q[i]);
            upper[i] = upper[i].max(q[i]);
        }
    }
    let size = (upper[0] - lower[0]).max(upper[1] - lower[1]);
    size >= f.size_frac_band[0] && size <= f.size_frac_band[1] && lower[1] >= f.headroom
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub subject_visibility_samples: usize,
    pub buried_subject_samples: usize,
    pub repair_times: Vec<f64>,
    pub samples: usize,
    pub max_dt_s: f64,
    pub minimum_clearance: f64,
    pub heightfield_violations: usize,
    pub trace_clearance_violations: usize,
    pub framing_fraction: f64,
    pub framing_outside_transitions: usize,
    pub occlusion_fraction: f64,
    pub longest_occlusion_outside_transition_s: f64,
    pub max_speed: f64,
    pub max_jerk: f64,
    pub analytic_max_jerk: f64,
    pub sampled_max_jerk: f64,
    pub consumer_fd_max_jerk: f64,
    #[serde(default)]
    pub consumer_max_acceleration_jump: f64,
    #[serde(default)]
    pub orbit_radius_relative_error: f64,
    pub spline_max_jerk: f64,
    pub knot_continuity_relative: f64,
    pub resampling_relative_error: f64,
    pub ablation_violations: usize,
    pub ablation_depth: f64,
    pub adapter: String,
    pub backend: String,
    pub software_fallback: bool,
    pub trace_rays: usize,
}
fn sample_panels(shot: &Shot, offset: f64) -> f64 {
    // Leave one global-time rounding interval below the requested maximum dt.
    // At magnitude T, float32 spacing is bounded by EPSILON*T.
    let period = 1.0 / 240.0 - f64::from(f32::EPSILON) * (offset + shot.duration_s);
    if period <= 0.0 {
        return f64::INFINITY;
    }
    (shot.duration_s / period).ceil() + 1.0
}
fn times(shot: &Shot, offset: f64) -> Vec<f64> {
    let n = sample_panels(shot, offset) as usize;
    (0..=n)
        .map(|i| f64::from((offset + shot.duration_s * i as f64 / n as f64) as f32) - offset)
        .collect()
}
pub fn check_resolution(shot: &Shot, offset: f64) -> Result<(), String> {
    let count = sample_panels(shot, offset) + 1.0;
    let capacity = gpu_trace::sample_capacity().map_err(|e| binding("gpu_trace", e))?;
    if !count.is_finite() || count > capacity as f64 {
        return Err(binding(
            "validation_resolution",
            "shot duration exceeds negotiated GPU trace batch capacity",
        ));
    }
    Ok(())
}
pub fn approximation(
    shot: &Shot,
    solution: &Solution,
    keys: &[Keyframe],
    offset: f64,
) -> Result<f64, String> {
    if keys.windows(2).any(|k| k[1].time <= k[0].time) {
        return Err(binding(
            "consumer_precision",
            "native f32 keyframe times coalesced",
        ));
    }
    let a = animation(keys);
    let mut error: f64 = 0.0;
    let mut samples = times(shot, offset);
    samples.extend(
        keys.iter()
            .map(|k| k.time - offset)
            .filter(|t| *t >= 0.0 && *t <= shot.duration_s),
    );
    for t in samples {
        let (eye, target) = solution.sample(t, 0);
        let (actual, aim) = consumer(&a, t + offset)?;
        error = error.max(
            actual.distance(eye).max(aim.distance(target)) / eye.distance(shot.subject.center()),
        );
    }
    Ok(error)
}
fn add_ray(rays: &mut Vec<gpu_trace::Ray>, origin: DVec3, direction: DVec3, length: f64) {
    let d = direction.normalize();
    // PROMETHEUS's heightfield is in the declared local scene frame. Its
    // texture origin and ray origin share the same zero anchor.
    let p = Anchor::new().to_render_f32(SceneCoord::scene(origin));
    let d = Anchor::offset_to_render(SceneOffset::scene(d));
    rays.push(gpu_trace::Ray {
        origin_tmin: [p.x, p.y, p.z, 0.0],
        direction_tmax: [d.x, d.y, d.z, length as f32],
    });
}
pub fn validate(
    shot: &Shot,
    solution: &Solution,
    keys: &[Keyframe],
    terrain: &Terrain,
    offset: f64,
) -> Result<Report, String> {
    let subject_points = terrain.subject_points(shot)?;
    let a = animation(keys);
    let ts = times(shot, offset);
    let start_time = ts[0] + offset;
    let end_time = ts.last().unwrap() + offset;
    let continuous = motion::Consumer::new(keys)?;
    let dt = ts.windows(2).map(|t| t[1] - t[0]).fold(0.0, f64::max);
    let mut minimum = f64::INFINITY;
    let mut repair_times = Vec::new();
    let mut height_bad = 0;
    let mut frame_good = 0;
    let mut frame_bad_outside = 0;
    let mut positions = Vec::new();
    let mut targets = Vec::new();
    let mut continuous_positions = Vec::new();
    let mut continuous_targets = Vec::new();
    let mut analytic_max_jerk = continuous.knot_jerk(start_time, end_time);
    let mut rays = Vec::new();
    let mut ray_ranges = Vec::new();
    let mut ablation_bad = 0;
    let mut ablation_depth: f64 = 0.0;
    let spline_jerk = solution
        .segments
        .iter()
        .map(super::Segment::jerk_bound)
        .fold(0.0, f64::max);
    let (first, _) = consumer(&a, offset)?;
    let (last, _) = consumer(&a, shot.duration_s + offset)?;
    super::solver::check_intent(
        shot,
        DVec3::from(solution.segments[0].eye0),
        DVec3::from(solution.segments.last().unwrap().eye1),
    )?;
    let center = shot.subject.center();
    let orbit_radius = (first.x - center.x).hypot(first.z - center.z);
    let mut orbit_error: f64 = 0.0;
    for &t in &ts {
        let (eye, aim) = consumer(&a, t + offset)?;
        if shot.kind == "orbit" {
            orbit_error = orbit_error
                .max(((eye.x - center.x).hypot(eye.z - center.z) / orbit_radius - 1.0).abs());
        }
        // The last pose belongs to this shot's left-hand interval. A joined
        // next shot may have a different derivative at the same position.
        let (pose, look) = if t == *ts.last().unwrap() {
            continuous.sample_before(t + offset)
        } else {
            continuous.sample(t + offset)
        };
        continuous_positions.push(pose[0]);
        continuous_targets.push(look[0]);
        analytic_max_jerk = analytic_max_jerk
            .max(pose[3].length())
            .max(look[3].length());
        let clearance = eye.y - terrain.height(eye);
        minimum = minimum.min(clearance);
        if !terrain.inside(eye) || clearance < shot.clearance {
            height_bad += 1;
            repair_times.push(t);
        }
        let in_transition = in_transition(shot, t);
        let actual_fov = f64::from(
            a.evaluate((t + offset) as f32)
                .ok_or_else(|| binding("consumer", "missing camera state"))?
                .fov_deg,
        );
        if framing(shot, eye, aim, actual_fov) {
            frame_good += 1;
        } else if !in_transition {
            frame_bad_outside += 1;
            repair_times.push(t);
        }
        let linear = first.lerp(last, t / shot.duration_s);
        let penetration = terrain.height(linear) + shot.clearance - linear.y;
        if penetration > 0.0 {
            ablation_bad += 1;
            ablation_depth = ablation_depth.max(penetration);
        }
        let begin = rays.len();
        for d in [
            DVec3::NEG_Y,
            DVec3::new(1.0, -1.0, 0.0),
            DVec3::new(-1.0, -1.0, 0.0),
            DVec3::new(0.0, -1.0, 1.0),
            DVec3::new(0.0, -1.0, -1.0),
        ] {
            add_ray(&mut rays, eye, d, shot.clearance);
        }
        let forward = (aim - eye).normalize();
        let right = forward.cross(DVec3::Y).normalize();
        let up = right.cross(forward);
        let tangent = (actual_fov.to_radians() * 0.5).tan();
        add_ray(&mut rays, eye, forward, shot.clearance);
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                add_ray(
                    &mut rays,
                    eye,
                    forward + right * x * tangent * shot.framing.aspect + up * y * tangent,
                    shot.clearance,
                );
            }
        }
        let occlusion_begin = rays.len();
        for subject in &subject_points {
            let delta = *subject - eye;
            // Trim only the endpoint rounding interval of f32 traversal.
            let trim = delta.length() * f64::from(f32::EPSILON) * 8.0;
            add_ray(&mut rays, eye, delta, delta.length() - trim);
        }
        ray_ranges.push((begin, occlusion_begin, rays.len(), in_transition));
        positions.push(eye);
        targets.push(aim);
    }
    let trace = gpu_trace::trace(terrain, &rays).map_err(|e| binding("gpu_trace", e))?;
    let mut trace_bad = 0;
    let mut occluded = 0;
    let mut run = 0;
    let mut longest = 0;
    for (index, (begin, middle, end, transition)) in ray_ranges.into_iter().enumerate() {
        if trace.hits[begin..middle].iter().any(|v| *v != 0) {
            trace_bad += 1;
            repair_times.push(ts[index]);
        }
        // Any sampled part of the declared subject hidden by terrain counts.
        let blocked = trace.hits[middle..end].iter().any(|v| *v != 0);
        if blocked {
            occluded += 1;
            if !transition {
                repair_times.push(ts[index]);
            }
        }
        if blocked && !transition {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    let mut speed: f64 = 0.0;
    let mut sampled_jerk: f64 = 0.0;
    let mut continuous_jerk: f64 = 0.0;
    for (i, path) in [
        &positions,
        &targets,
        &continuous_positions,
        &continuous_targets,
    ]
    .into_iter()
    .enumerate()
    {
        for (p, t) in path.windows(2).zip(ts.windows(2)) {
            if i < 2 {
                speed = speed.max(p[0].distance(p[1]) / (t[1] - t[0]));
            }
        }
        for (p, t) in path.windows(4).zip(ts.windows(4)) {
            let a = (p[1] - p[0]) / (t[1] - t[0]);
            let b = (p[2] - p[1]) / (t[2] - t[1]);
            let c = (p[3] - p[2]) / (t[3] - t[2]);
            let ab = (b - a) / (t[2] - t[0]);
            let bc = (c - b) / (t[3] - t[1]);
            let jerk = 6.0 * (bc - ab).length() / (t[3] - t[0]);
            if i < 2 {
                sampled_jerk = sampled_jerk.max(jerk);
            } else {
                continuous_jerk = continuous_jerk.max(jerk);
            }
        }
    }
    // A difference spanning a consumer knot measures its acceleration jump / dt,
    // not an interval jerk. Keep it diagnostic; gate the interval bound only.
    let jerk = continuous.jerk_bound(start_time, end_time);
    if !jerk.is_finite() || !analytic_max_jerk.is_finite() {
        return Err(binding(
            "consumer_precision",
            "consumer motion derivatives are not finite",
        ));
    }
    let mut continuity: f64 = 0.0;
    for s in solution.segments.windows(2) {
        for d in 0..=2 {
            let (a, b) = s[0].sample(s[0].t1, d);
            let (c, e) = s[1].sample(s[1].t0, d);
            continuity = continuity
                .max(a.distance(c) / a.length().max(c.length()).max(1.0))
                .max(b.distance(e) / b.length().max(e.length()).max(1.0));
        }
    }
    Ok(Report {
        subject_visibility_samples: subject_points.len(),
        buried_subject_samples: 9 - subject_points.len(),
        repair_times,
        samples: ts.len(),
        max_dt_s: dt,
        minimum_clearance: minimum,
        heightfield_violations: height_bad,
        trace_clearance_violations: trace_bad,
        framing_fraction: frame_good as f64 / ts.len() as f64,
        framing_outside_transitions: frame_bad_outside,
        occlusion_fraction: occluded as f64 / ts.len() as f64,
        longest_occlusion_outside_transition_s: longest as f64 * dt,
        max_speed: speed,
        max_jerk: jerk,
        analytic_max_jerk,
        sampled_max_jerk: sampled_jerk,
        consumer_fd_max_jerk: continuous_jerk,
        consumer_max_acceleration_jump: continuous.acceleration_jump(start_time, end_time),
        orbit_radius_relative_error: orbit_error,
        spline_max_jerk: spline_jerk,
        knot_continuity_relative: continuity,
        resampling_relative_error: approximation(shot, solution, keys, offset)?,
        ablation_violations: ablation_bad,
        ablation_depth,
        adapter: trace.adapter,
        backend: trace.backend,
        software_fallback: trace.software_fallback,
        trace_rays: trace.ray_count,
    })
}
pub fn in_transition(shot: &Shot, time: f64) -> bool {
    (shot.transition[0] > 0.0 && time <= shot.transition[0])
        || (shot.transition[1] > 0.0 && time >= shot.duration_s - shot.transition[1])
}
pub fn failure(shot: &Shot, r: &Report) -> Option<&'static str> {
    if r.max_dt_s > 1.0 / 240.0 {
        Some("validation_resolution")
    } else if r.heightfield_violations > 0 || r.trace_clearance_violations > 0 {
        Some("clearance")
    } else if r.framing_fraction < 0.95 || r.framing_outside_transitions > 0 {
        Some("framing")
    } else if r.occlusion_fraction > 0.05 || r.longest_occlusion_outside_transition_s > 0.5 {
        Some("occlusion")
    } else if r.knot_continuity_relative >= 1e-6 {
        Some("C2_continuity")
    } else if r.max_speed > shot.max_speed {
        Some("speed")
    } else if r.max_jerk > shot.max_jerk || r.spline_max_jerk > shot.max_jerk {
        Some("jerk")
    } else if r.resampling_relative_error >= 0.001 {
        Some("consumer_resampling")
    } else {
        None
    }
}
pub fn require(shot: &Shot, r: &Report) -> Result<(), String> {
    failure(shot, r).map_or(Ok(()), |constraint| {
        Err(binding(constraint, serde_json::to_string(r).unwrap()))
    })
}
