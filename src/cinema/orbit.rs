//! Analytic circular motion: one minimum-jerk angle, no interior arc stops.
use super::{binding, solver, Arc, Segment, Shot, Solution};
use glam::DVec3;
use std::f64::consts::{FRAC_PI_2, TAU};

impl Arc {
    fn radius(&self, point: [f64; 3]) -> f64 {
        (point[0] - self.center[0]).hypot(point[2] - self.center[2])
    }
    pub fn unit_bounds(&self, segment: &Segment) -> (f64, f64) {
        let radius = self.radius(segment.eye0).max(self.radius(segment.target0));
        // Exact extrema of q', q'', q''' for q=10u³-15u⁴+6u⁵.
        let v = self.sweep.abs() * 15.0 / 8.0;
        let a = self.sweep.abs() * 10.0 / 3.0_f64.sqrt();
        let j = self.sweep.abs() * 60.0;
        (radius * v, radius * (j + v.powi(3) + 3.0 * v * a))
    }
    pub fn sample(&self, segment: &Segment, u: f64, derivative: usize) -> (DVec3, DVec3) {
        let h = segment.t1 - segment.t0;
        let angle = self.sweep * u.powi(3) * (10.0 + u * (-15.0 + 6.0 * u));
        let v = self.sweep * 30.0 * u.powi(2) * (1.0 - u).powi(2) / h;
        let a = self.sweep * 60.0 * u * (1.0 - 3.0 * u + 2.0 * u * u) / h.powi(2);
        let j = self.sweep * (60.0 - 360.0 * u + 360.0 * u * u) / h.powi(3);
        let (sin, cos) = angle.sin_cos();
        let center = DVec3::from(self.center);
        let rotate = |start: [f64; 3], end: [f64; 3]| {
            let p = DVec3::from(start) - center;
            let radial = DVec3::new(p.x * cos - p.z * sin, 0.0, p.x * sin + p.z * cos);
            let tangent = DVec3::new(-radial.z, 0.0, radial.x);
            match derivative {
                0 if u == 0.0 => DVec3::from(start),
                0 if u == 1.0 => DVec3::from(end),
                0 => center + radial + DVec3::Y * p.y,
                1 => tangent * v,
                2 => tangent * a - radial * v.powi(2),
                _ => tangent * (j - v.powi(3)) - radial * (3.0 * v * a),
            }
        };
        (
            rotate(segment.eye0, segment.eye1),
            rotate(segment.target0, segment.target1),
        )
    }
    pub fn valid(&self, segment: &Segment) -> bool {
        if !self.sweep.is_finite()
            || self.sweep == 0.0
            || self.center.iter().any(|v| !v.is_finite())
        {
            return false;
        }
        let center = DVec3::from(self.center);
        let (s, c) = self.sweep.sin_cos();
        [
            (segment.eye0, segment.eye1),
            (segment.target0, segment.target1),
        ]
        .into_iter()
        .all(|(a, b)| {
            let p = DVec3::from(a) - center;
            let expected = center + DVec3::new(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
            // Floating-point backward-error guard, not a shape tolerance.
            let rounding = f64::EPSILON
                * (1.0 + self.sweep.abs())
                * (DVec3::from(a).length() + center.length() + p.length());
            (a[1] - b[1]).abs() <= rounding && expected.distance(DVec3::from(b)) <= rounding
        })
    }
}

pub fn solve(
    shot: &Shot,
    terrain: &solver::Terrain,
    start: DVec3,
    end: DVec3,
    extra_floor: f64,
) -> Result<Solution, String> {
    let shell = solver::shell(shot);
    let center = shot.subject.center();
    let radius = (start.x - center.x).hypot(start.z - center.z);
    let declared = shot.bearing_deg[0].to_radians();
    let angle = if shot.start_eye.is_none() {
        declared
    } else {
        let principal = (start.z - center.z).atan2(start.x - center.x);
        let unwrapped = principal + TAU * ((declared - principal) / TAU).round();
        // Preserve the supplied eye and derive its nearest equivalent bearing.
        let expected = center
            + DVec3::new(
                radius * declared.cos(),
                start.y - center.y,
                radius * declared.sin(),
            );
        if start.distance(expected) > 0.001 * start.distance(center) {
            return Err(binding(
                "orbit_start_bearing",
                "fixed start eye does not match the declared starting bearing",
            ));
        }
        unwrapped
    };
    let sweep = (shot.bearing_deg[1] - shot.bearing_deg[0]).to_radians();
    if sweep == 0.0 {
        return Err(binding("orbit_angle", "orbit must sweep a nonzero angle"));
    }
    let at = |a: f64, y: f64| center + DVec3::new(radius * a.cos(), y - center.y, radius * a.sin());
    let projected = at(angle + sweep, end.y);
    if shot.end_eye.is_some() && projected.distance(end) > 0.001 * end.distance(center) {
        return Err(binding(
            "orbit_endpoint",
            "fixed endpoint exceeds 0.1% of subject distance from the requested signed arc",
        ));
    }
    let end = projected;
    solver::check_intent(shot, start, end)?;
    solver::endpoints(start, end, terrain, shot, shell)?;
    let margin = solver::consumer_margin(terrain, shell.1);
    let mut floor = start.y.max(end.y).max(extra_floor);
    // Cardinal extrema divide an arc into intervals whose exact X/Z extrema
    // are at their endpoints. Their dilated rectangles enclose the full arcs.
    // Repeated turns visit the same four rectangles: no angular sampling cap.
    let (lo, hi) = if sweep.abs() >= TAU {
        (0.0, TAU)
    } else {
        (angle.min(angle + sweep), angle.max(angle + sweep))
    };
    let mut angles = vec![lo, hi];
    let mut cardinal = (lo / FRAC_PI_2).floor() + 1.0;
    while cardinal * FRAC_PI_2 < hi {
        angles.push(cardinal * FRAC_PI_2);
        cardinal += 1.0;
    }
    angles.sort_by(f64::total_cmp);
    let subjects = terrain.subject_points(shot)?;
    for bounds in angles.windows(2) {
        let a = at(bounds[0], floor);
        let b = at(bounds[1], floor);
        if !terrain.inside(a) || !terrain.inside(b) {
            return Err(binding(
                "terrain_bounds",
                "orbit arc exits the heightfield footprint",
            ));
        }
        floor = floor.max(terrain.ceiling(a, b, shot.clearance) + shot.clearance + margin);
        for p in [a, b, at((bounds[0] + bounds[1]) * 0.5, floor)] {
            for subject in &subjects {
                floor = floor.max(terrain.visibility_floor(p, *subject) + margin);
            }
        }
    }
    let up = DVec3::new(start.x, floor, start.z);
    let down = DVec3::new(end.x, floor, end.z);
    if !solver::family(up, shot, shell)
        || !solver::family(down, shot, shell)
        || !solver::edge(start, up, terrain, shot, shell)
        || !solver::edge(down, end, terrain, shot, shell)
    {
        return Err(binding(
            "clearance_framing_corridor",
            "orbit clearance loft leaves the framing shell",
        ));
    }
    let make = |a: DVec3, b: DVec3, arc| -> Result<Segment, String> {
        Ok(Segment {
            t0: 0.0,
            t1: 1.0,
            eye0: a.to_array(),
            eye1: b.to_array(),
            target0: solver::target(a, shot)?.to_array(),
            target1: solver::target(b, shot)?.to_array(),
            arc,
        })
    };
    let mut segments = Vec::new();
    if start != up {
        segments.push(make(start, up, None)?);
    }
    let arc = make(
        up,
        down,
        Some(Arc {
            center: center.to_array(),
            sweep,
        }),
    )?;
    if !arc.arc.as_ref().unwrap().valid(&arc) {
        return Err(binding(
            "orbit_endpoint",
            "fixed endpoint does not match the declared bearing and radius",
        ));
    }
    segments.push(arc);
    if down != end {
        segments.push(make(down, end, None)?);
    }
    let bounds: Vec<_> = segments
        .iter()
        .map(|s| {
            let speed = s.arc.as_ref().map_or_else(
                || {
                    1.875
                        * DVec3::from(s.eye0)
                            .distance(DVec3::from(s.eye1))
                            .max(DVec3::from(s.target0).distance(DVec3::from(s.target1)))
                },
                |arc| arc.unit_bounds(s).0,
            );
            (speed / shot.max_speed).max((s.jerk_bound() / shot.max_jerk).cbrt())
        })
        .collect();
    let needed: f64 = bounds.iter().sum();
    if needed > shot.duration_s {
        return Err(binding(
            "duration_speed_jerk",
            format!(
                "minimum continuous-orbit duration {needed} exceeds {}",
                shot.duration_s
            ),
        ));
    }
    // Allocate spare time in proportion to the motion bounds, independently
    // of the DEM resolution or the number of clearance intervals.
    let mut time = 0.0;
    let count = segments.len();
    for (i, (segment, lower)) in segments.iter_mut().zip(bounds).enumerate() {
        segment.t0 = time;
        time = if i + 1 == count {
            shot.duration_s
        } else {
            time + lower * shot.duration_s / needed
        };
        segment.t1 = time;
    }
    Ok(Solution { segments, expanded_nodes: 0,
        corridor: "PROMETHEUS dilated arc rectangles; analytic minimum-jerk angle with endpoint clearance lifts".into() })
}
