//! Visibility graph on the DEM lattice, clipped to the framing distance shell.
//! Routed edges use dilated PROMETHEUS bounds; dolly checks its exact straight trace.
use super::{binding, Segment, Shot, Solution};
use crate::path_tracing::hybrid_compute::terrain_heightfield::{build_minmax_mips, MinMaxMips};
use glam::DVec3;
use std::{cmp::Ordering, collections::BinaryHeap};

pub(super) fn consumer_margin(terrain: &Terrain, radius: f64) -> f64 {
    // Reserve the contract's maximum resampling displacement, plus the native
    // angle/position rounding guard, so a valid cubic need not graze the floor.
    0.001 * radius + (terrain.width + radius) * f64::from(f32::EPSILON) * std::f64::consts::TAU
}

pub struct Terrain {
    pub heights: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub width: f64,
    pub pyramid: MinMaxMips,
}
impl Terrain {
    pub fn new(heights: Vec<f32>, w: u32, h: u32, width: f64) -> Result<Self, String> {
        if !width.is_finite() || width <= 0.0 {
            return Err(binding("terrain", "width must be positive and finite"));
        }
        let pyramid = build_minmax_mips(&heights, w, h).map_err(|e| binding("terrain", e))?;
        Ok(Self {
            heights,
            w: w as usize,
            h: h as usize,
            width,
            pyramid,
        })
    }
    pub fn inside(&self, p: DVec3) -> bool {
        p.is_finite() && p.x >= 0.0 && p.z >= 0.0 && p.x <= self.width && p.z <= self.width
    }
    pub fn height(&self, p: DVec3) -> f64 {
        let x = (p.x / self.width * (self.w - 1) as f64).clamp(0.0, (self.w - 1) as f64);
        let z = (p.z / self.width * (self.h - 1) as f64).clamp(0.0, (self.h - 1) as f64);
        let (ix, iz) = (x.floor() as usize, z.floor() as usize);
        let at = |dx: usize, dz: usize| {
            f64::from(self.heights[(iz + dz).min(self.h - 1) * self.w + (ix + dx).min(self.w - 1)])
        };
        let (u, v) = (x - ix as f64, z - iz as f64);
        (at(0, 0) * (1.0 - u) + at(1, 0) * u) * (1.0 - v)
            + (at(0, 1) * (1.0 - u) + at(1, 1) * u) * v
    }
    pub fn subject_points(&self, shot: &Shot) -> Result<Vec<DVec3>, String> {
        let points: Vec<_> = std::iter::once(shot.subject.center())
            .chain(shot.subject.corners())
            .filter(|p| p.y > self.height(*p))
            .collect();
        if points.is_empty() {
            return Err(binding(
                "subject_embedded",
                "no declared subject reference point is above the heightfield",
            ));
        }
        Ok(points)
    }
    pub fn ceiling(&self, a: DVec3, b: DVec3, dilation: f64) -> f64 {
        let lower = a.min(b) - DVec3::splat(dilation);
        let upper = a.max(b) + DVec3::splat(dilation);
        let ix = |x: f64| {
            (x / self.width * (self.w - 1) as f64)
                .floor()
                .clamp(0.0, (self.w - 2) as f64) as usize
        };
        let iz = |z: f64| {
            (z / self.width * (self.h - 1) as f64)
                .floor()
                .clamp(0.0, (self.h - 2) as f64) as usize
        };
        let rect = [ix(lower.x), iz(lower.z), ix(upper.x), iz(upper.z)];
        self.node_max(self.pyramid.levels.len() - 1, 0, 0, rect)
    }
    fn node_max(&self, level: usize, x: usize, z: usize, rect: [usize; 4]) -> f64 {
        let side = 1usize << level;
        let (lo_x, lo_z) = (x * side, z * side);
        if lo_x > rect[2] || lo_z > rect[3] || lo_x + side <= rect[0] || lo_z + side <= rect[1] {
            return f64::NEG_INFINITY;
        }
        let (w, h) = self.pyramid.dims[level];
        if x >= w as usize || z >= h as usize {
            return f64::NEG_INFINITY;
        }
        if level == 0
            || (lo_x >= rect[0]
                && lo_z >= rect[1]
                && lo_x + side - 1 <= rect[2]
                && lo_z + side - 1 <= rect[3])
        {
            return f64::from(self.pyramid.levels[level][z * w as usize + x][1]);
        }
        let mut top = f64::NEG_INFINITY;
        for dz in 0..2 {
            for dx in 0..2 {
                top = top.max(self.node_max(level - 1, x * 2 + dx, z * 2 + dz, rect));
            }
        }
        top
    }
    fn cell_intervals(&self, a: DVec3, b: DVec3) -> Vec<f64> {
        let d = b - a;
        let mut ts = vec![0.0, 1.0];
        for (origin, delta, count) in [(a.x, d.x, self.w), (a.z, d.z, self.h)] {
            if delta != 0.0 {
                for i in 1..count - 1 {
                    let t = (i as f64 * self.width / (count - 1) as f64 - origin) / delta;
                    if t > 0.0 && t < 1.0 {
                        ts.push(t);
                    }
                }
            }
        }
        ts.sort_by(f64::total_cmp);
        ts.dedup();
        ts
    }
    /// Exact vertical clearance extrema of a line over each bilinear DEM cell.
    /// The line visits only these cells, rather than its entire enclosing box.
    pub(super) fn line_clearance(&self, a: DVec3, b: DVec3) -> f64 {
        let d = b - a;
        self.cell_intervals(a, b)
            .windows(2)
            .fold(f64::INFINITY, |minimum, t| {
                let gap = |u: f64| {
                    let p = a + d * (t[0] + (t[1] - t[0]) * u);
                    p.y - self.height(p)
                };
                let g0 = gap(0.0);
                let gm = gap(0.5);
                let g1 = gap(1.0);
                let quadratic = 2.0 * (g0 + g1 - 2.0 * gm);
                let linear = g1 - g0 - quadratic;
                let mut least = minimum.min(g0).min(g1);
                if quadratic > 0.0 {
                    let u = -linear / (2.0 * quadratic);
                    if u > 0.0 && u < 1.0 {
                        least = least.min(g0 + u * (linear + u * quadratic));
                    }
                }
                least
            })
    }
    // A bilinear heightfield restricted to a straight sight ray is quadratic;
    // solve its eye-height constraint exactly on the same cell partition.
    pub fn visible(&self, a: DVec3, b: DVec3) -> bool {
        a.y >= self.visibility_floor(a, b)
    }
    pub(super) fn visibility_floor(&self, a: DVec3, b: DVec3) -> f64 {
        let d = b - a;
        let ts = self.cell_intervals(a, b);
        ts.windows(2).fold(f64::NEG_INFINITY, |mut floor, t| {
            let h = t[1] - t[0];
            let y0 = self.height(a + d * t[0]);
            let ym = self.height(a + d * (t[0] + h / 2.0));
            let y1 = self.height(a + d * t[1]);
            // N(v)/(1-t0-h*v), where N is height(v)-subject_y*(t0+h*v).
            let n0 = y0 - b.y * t[0];
            let n2 = 2.0 * (y0 + y1 - 2.0 * ym);
            let n1 = y1 - y0 - n2 - b.y * h;
            let denominator = 1.0 - t[0];
            let aa = -n2 * h;
            let bb = 2.0 * n2 * denominator;
            let cc = n1 * denominator + h * n0;
            let mut candidates = vec![0.0, 1.0];
            if aa != 0.0 {
                let discriminant = bb * bb - 4.0 * aa * cc;
                if discriminant >= 0.0 {
                    let q = -0.5 * (bb + discriminant.sqrt().copysign(bb));
                    if q != 0.0 {
                        candidates.extend([q / aa, cc / q]);
                    } else {
                        candidates.push(-bb / (2.0 * aa));
                    }
                }
            } else if bb != 0.0 {
                candidates.push(-cc / bb);
            }
            for v in candidates.into_iter().filter(|v| *v >= 0.0 && *v <= 1.0) {
                let u = t[0] + h * v;
                if u == 1.0 {
                    if b.y < y1 {
                        return f64::INFINITY;
                    }
                } else {
                    floor = floor.max((n0 + v * (n1 + v * n2)) / (1.0 - u));
                }
            }
            floor
        })
    }
}

pub fn target(eye: DVec3, shot: &Shot) -> Result<DVec3, String> {
    let delta = shot.subject.center() - eye;
    if !delta.is_finite() || delta.x.hypot(delta.z) == 0.0 {
        return Err(binding(
            "framing_orientation",
            "Y-up projection needs a nonvertical look direction",
        ));
    }
    if shot.framing.region[0] + shot.framing.region[2] == 1.0
        && shot.framing.region[1] + shot.framing.region[3] == 1.0
    {
        return Ok(shot.subject.center());
    }
    let length = delta.length();
    let direction = delta / length;
    let f = &shot.framing;
    let u = (f.region[0] + f.region[2]) * 0.5;
    let v = (f.region[1] + f.region[3]) * 0.5;
    let tan = (f.fov_deg.to_radians() * 0.5).tan();
    let px = (2.0 * u - 1.0) * tan * f.aspect;
    let py = (1.0 - 2.0 * v) * tan;
    let horizontal = direction.x.hypot(direction.z);
    let sine = px / (1.0 + px * px + py * py).sqrt() / horizontal;
    if !sine.is_finite() || sine.abs() >= 1.0 {
        return Err(binding(
            "framing_region",
            "Y-up camera cannot attain the requested screen center",
        ));
    }
    let offset = sine.asin();
    let yaw = direction.z.atan2(direction.x) - offset;
    let pitch = direction.y.atan2(horizontal * offset.cos()) - py.atan();
    Ok(eye
        + DVec3::new(
            pitch.cos() * yaw.cos(),
            pitch.sin(),
            pitch.cos() * yaw.sin(),
        ) * length)
}

pub(super) fn shell(shot: &Shot) -> (f64, f64) {
    let diameter =
        (DVec3::from(shot.subject.bounds[1]) - DVec3::from(shot.subject.bounds[0])).length();
    let f = &shot.framing;
    let tan = (f.fov_deg.to_radians() * 0.5).tan();
    let tan_x = tan * f.aspect;
    let px = (f.region[0] + f.region[2] - 1.0) * tan_x;
    let py = (1.0 - f.region[1] - f.region[3]) * tan;
    let projection_bound = ((1.0 + px * px).sqrt() / tan_x).max((1.0 + py * py).sqrt() / tan);
    let [lo, hi] = shot.framing.size_frac_band;
    let extent = DVec3::from(shot.subject.bounds[1]) - DVec3::from(shot.subject.bounds[0]);
    (
        (diameter * projection_bound / (2.0 * hi) + diameter * 0.5)
            * (1.0 + px * px + py * py).sqrt(),
        extent.min_element() / (2.0 * lo * tan.min(tan_x)) - diameter * 0.5,
    )
}
pub(super) fn family(p: DVec3, shot: &Shot, radius: (f64, f64)) -> bool {
    let d = p.distance(shot.subject.center());
    d >= radius.0 && d <= radius.1 && target(p, shot).is_ok()
}
fn framing_constraint(p: DVec3, shot: &Shot, radius: (f64, f64)) -> Result<(), String> {
    let distance = p.distance(shot.subject.center());
    if distance < radius.0 || distance > radius.1 {
        return Err(binding(
            "framing_distance",
            "straight dolly approach leaves the projection-derived distance shell",
        ));
    }
    target(p, shot).map(|_| ())
}

fn straight_dolly(
    a: DVec3,
    b: DVec3,
    terrain: &Terrain,
    shot: &Shot,
    radius: (f64, f64),
) -> Result<(), String> {
    let clearance = terrain.line_clearance(a, b);
    if clearance < shot.clearance {
        return Err(binding(
            "clearance",
            format!(
                "straight dolly approach has minimum vertical clearance {clearance}, requires {}",
                shot.clearance
            ),
        ));
    }
    let d = b - a;
    let q = a - shot.subject.center();
    let mut candidates = vec![0.0, 1.0];
    if d.length_squared() > 0.0 {
        candidates.push((-q.dot(d) / d.length_squared()).clamp(0.0, 1.0));
    }
    let horizontal = DVec3::new(d.x, 0.0, d.z);
    if horizontal.length_squared() > 0.0 {
        candidates.push((-q.dot(horizontal) / horizontal.length_squared()).clamp(0.0, 1.0));
    }
    // Attainability of an off-center Y-up target depends on horizontal²/radius².
    // Its stationary points solve a quadratic; include them, not arbitrary probes.
    let n = [
        q.x * q.x + q.z * q.z,
        2.0 * (q.x * d.x + q.z * d.z),
        horizontal.length_squared(),
    ];
    let den = [q.length_squared(), 2.0 * q.dot(d), d.length_squared()];
    let c = n[1] * den[0] - n[0] * den[1];
    let linear = 2.0 * (n[2] * den[0] - n[0] * den[2]);
    let quadratic = n[2] * den[1] - n[1] * den[2];
    if quadratic != 0.0 {
        let discriminant = linear * linear - 4.0 * quadratic * c;
        if discriminant >= 0.0 {
            let root = -0.5 * (linear + discriminant.sqrt().copysign(linear));
            if root != 0.0 {
                candidates.extend([root / quadratic, c / root]);
            } else {
                candidates.push(-linear / (2.0 * quadratic));
            }
        }
    } else if linear != 0.0 {
        candidates.push(-c / linear);
    }
    for u in candidates.into_iter().filter(|u| *u >= 0.0 && *u <= 1.0) {
        framing_constraint(a + d * u, shot, radius)?;
    }
    Ok(())
}
pub(super) fn edge(a: DVec3, b: DVec3, t: &Terrain, shot: &Shot, r: (f64, f64)) -> bool {
    if a.y.min(b.y) < t.ceiling(a, b, shot.clearance) + shot.clearance {
        return false;
    }
    let d = b - a;
    if d == DVec3::ZERO {
        return family(a, shot, r);
    }
    // The native spherical camera has no continuous azimuth through a vertical
    // look-axis pole. Route around it rather than emitting an unresampleable edge.
    let horizontal = DVec3::new(d.x, 0.0, d.z);
    if horizontal.length_squared() > 0.0 {
        let v = ((shot.subject.center() - a).dot(horizontal) / horizontal.length_squared())
            .clamp(0.0, 1.0);
        if !family(a + d * v, shot, r) {
            return false;
        }
    }
    let u = ((shot.subject.center() - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
    family(a, shot, r) && family(b, shot, r) && (a + d * u).distance(shot.subject.center()) >= r.0
}
#[derive(Clone, Copy, PartialEq)]
struct Queue(f64, usize);
impl Eq for Queue {}
impl Ord for Queue {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .total_cmp(&self.0)
            .then_with(|| other.1.cmp(&self.1))
    }
}
impl PartialOrd for Queue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn endpoints(
    start: DVec3,
    end: DVec3,
    terrain: &Terrain,
    shot: &Shot,
    r: (f64, f64),
) -> Result<(), String> {
    let subject_points = terrain.subject_points(shot)?;
    for p in [start, end] {
        if !terrain.inside(p) {
            return Err(binding(
                "terrain_bounds",
                "shot endpoint is outside the heightfield",
            ));
        }
        let floor = if shot.kind == "dolly" {
            terrain.height(p)
        } else {
            terrain.ceiling(p, p, shot.clearance)
        };
        if p.y < floor + shot.clearance {
            return Err(binding(
                "clearance",
                "fixed endpoint is below its required terrain clearance",
            ));
        }
        if p.distance(shot.subject.center()) < r.0 || p.distance(shot.subject.center()) > r.1 {
            return Err(binding(
                "framing_distance",
                "fixed endpoint is outside the projection-derived distance shell",
            ));
        }
        target(p, shot)?;
        if !subject_points.iter().all(|q| terrain.visible(p, *q)) {
            return Err(binding(
                "occlusion",
                "fixed endpoint cannot see the subject",
            ));
        }
    }
    Ok(())
}

fn route(
    start: DVec3,
    end: DVec3,
    terrain: &Terrain,
    shot: &Shot,
    r: (f64, f64),
) -> Result<(Vec<DVec3>, usize), String> {
    endpoints(start, end, terrain, shot, r)?;
    if shot.kind == "dolly" {
        straight_dolly(start, end, terrain, shot, r)?;
        return Ok((vec![start, end], 0));
    }
    let subject_points = terrain.subject_points(shot)?;
    if edge(start, end, terrain, shot, r) {
        return Ok((vec![start, end], 0));
    }
    // Three altitude states per DEM cell: its conservative floor, the route's
    // clearance plane and the upper framing-shell boundary.
    let margin = consumer_margin(terrain, r.1);
    let ceiling = |p: DVec3| {
        let c = shot.subject.center();
        let h2 = (p.x - c.x).powi(2) + (p.z - c.z).powi(2);
        DVec3::new(
            p.x,
            c.y + ((r.1 - margin).powi(2) - h2).max(0.0).sqrt(),
            p.z,
        )
    };
    let plane = terrain.ceiling(start, end, shot.clearance) + shot.clearance;
    // Quantization guard derived from the float32 angle pair and world extent.
    let raised = |p: DVec3| DVec3::new(p.x, p.y.max(plane + margin), p.z);
    let mut nodes = vec![
        start,
        end,
        raised(start),
        raised(end),
        ceiling(start),
        ceiling(end),
    ];
    // The declared corridor dilation sets a physical search granularity. DEM
    // detail still participates in every conservative edge query.
    let cells = (terrain.width / shot.clearance).ceil() as usize;
    let nx = (terrain.w - 1).min(cells).max(1);
    let nz = (terrain.h - 1).min(cells).max(1);
    for z in 0..nz {
        for x in 0..nx {
            let p = DVec3::new(
                (x as f64 + 0.5) * terrain.width / nx as f64,
                0.0,
                (z as f64 + 0.5) * terrain.width / nz as f64,
            );
            nodes.push(DVec3::new(
                p.x,
                terrain.ceiling(p, p, shot.clearance) + shot.clearance + margin,
                p.z,
            ));
            nodes.push(raised(p));
            nodes.push(ceiling(p));
        }
    }
    // Visibility is the expensive part; evaluate it only for reached neighbors.
    let mut admissible = vec![None; nodes.len()];
    let mut distance = vec![f64::INFINITY; nodes.len()];
    let mut parent = vec![usize::MAX; nodes.len()];
    let mut queue = BinaryHeap::new();
    distance[0] = 0.0;
    queue.push(Queue(start.distance(end), 0));
    let mut expanded = 0;
    while let Some(Queue(cost, i)) = queue.pop() {
        if cost > distance[i] + nodes[i].distance(end) {
            continue;
        }
        if i == 1 {
            break;
        }
        expanded += 1;
        let mut neighbors = vec![0, 1, 2, 3, 4, 5];
        let (cx, cz) = if i < 6 {
            (
                ((nodes[i].x / terrain.width * nx as f64).floor() as usize).min(nx - 1),
                ((nodes[i].z / terrain.width * nz as f64).floor() as usize).min(nz - 1),
            )
        } else {
            let cell = (i - 6) / 3;
            (cell % nx, cell / nx)
        };
        for dz in -1isize..=1 {
            for dx in -1isize..=1 {
                let x = cx as isize + dx;
                let z = cz as isize + dz;
                if x >= 0 && z >= 0 && x < nx as isize && z < nz as isize {
                    let j = 6 + 3 * (z as usize * nx + x as usize);
                    neighbors.extend([j, j + 1, j + 2]);
                }
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        for j in neighbors {
            if i == j || !edge(nodes[i], nodes[j], terrain, shot, r) {
                continue;
            }
            let allowed = *admissible[j].get_or_insert_with(|| {
                family(nodes[j], shot, r)
                    && nodes[j].y
                        >= terrain.ceiling(nodes[j], nodes[j], shot.clearance) + shot.clearance
                    && subject_points.iter().all(|q| terrain.visible(nodes[j], *q))
            });
            if !allowed {
                continue;
            }
            let candidate = distance[i] + nodes[i].distance(nodes[j]);
            if candidate < distance[j] {
                distance[j] = candidate;
                parent[j] = i;
                queue.push(Queue(candidate + nodes[j].distance(end), j));
            }
        }
    }
    if !distance[1].is_finite() {
        return Err(binding(
            "clearance_framing_corridor",
            "no connected conservative corridor inside the framing shell",
        ));
    }
    let mut path = vec![end];
    let mut i = 1;
    while i != 0 {
        i = parent[i];
        path.push(nodes[i]);
    }
    path.reverse();
    let mut simplified = vec![start];
    let mut i = 0;
    while i + 1 < path.len() {
        let j = (i + 1..path.len())
            .rev()
            .find(|j| edge(path[i], path[*j], terrain, shot, r))
            .unwrap();
        simplified.push(path[j]);
        i = j;
    }
    Ok((simplified, expanded))
}

pub fn solve(shot: &Shot, terrain: &Terrain) -> Result<Solution, String> {
    if !terrain.inside(DVec3::from(shot.subject.bounds[0]))
        || !terrain.inside(DVec3::from(shot.subject.bounds[1]))
    {
        return Err(binding(
            "subject_bounds",
            "subject is outside the local heightfield footprint",
        ));
    }
    let r = shell(shot);
    let subject_points = terrain.subject_points(shot)?;
    let center = shot.subject.center();
    let eye = |bearing: f64, elevation: f64, radius: f64| {
        center
            + DVec3::new(
                elevation.cos() * bearing.cos(),
                elevation.sin(),
                elevation.cos() * bearing.sin(),
            ) * radius
    };
    let generated = |index: usize, approach: bool| -> Result<DVec3, String> {
        let bearing = shot.bearing_deg[index].to_radians();
        let elevation = shot.elevation_deg[index].to_radians();
        let direction = eye(bearing, elevation, 1.0) - center;
        let margin = consumer_margin(terrain, r.1);
        let lower = r.0 + margin;
        let mut upper = r.1 - margin;
        for (coordinate, delta) in [(center.x, direction.x), (center.z, direction.z)] {
            if delta > 0.0 {
                upper = upper.min((terrain.width - coordinate) / delta - margin);
            }
            if delta < 0.0 {
                upper = upper.min(-coordinate / delta - margin);
            }
        }
        if upper < lower {
            return Err(binding(
                "framing_terrain_bounds",
                "declared bearing has no framing radius inside terrain",
            ));
        }
        let middle = (lower + upper) * 0.5;
        let candidates = if approach {
            [lower, middle, upper]
        } else {
            [middle, upper, lower]
        };
        for radius in candidates {
            let mut p = eye(bearing, elevation, radius);
            p.y =
                p.y.max(terrain.ceiling(p, p, shot.clearance) + shot.clearance + margin);
            for corner in &subject_points {
                p.y = p.y.max(terrain.visibility_floor(p, *corner) + margin);
            }
            if terrain.inside(p) && family(p, shot, r) {
                return Ok(p);
            }
        }
        Err(binding(
            "framing_clearance_endpoint",
            "generated endpoint cannot intersect clearance and framing families",
        ))
    };
    let start = match shot.start_eye {
        Some(p) => DVec3::from(p),
        None => generated(0, false)?,
    };
    let end = match shot.end_eye {
        Some(p) => DVec3::from(p),
        None if shot.kind == "orbit" => {
            let bearing = shot.bearing_deg[1].to_radians();
            let radius = (start.x - center.x).hypot(start.z - center.z);
            let mut p = center
                + DVec3::new(
                    radius * bearing.cos(),
                    start.y - center.y,
                    radius * bearing.sin(),
                );
            let margin = consumer_margin(terrain, r.1);
            p.y =
                p.y.max(terrain.ceiling(p, p, shot.clearance) + shot.clearance + margin);
            for corner in &subject_points {
                p.y = p.y.max(terrain.visibility_floor(p, *corner) + margin);
            }
            p
        }
        None => generated(1, shot.kind == "dolly")?,
    };
    if shot.kind == "orbit" {
        return super::orbit::solve(shot, terrain, start, end, f64::NEG_INFINITY);
    }
    let anchors = [start, end];
    endpoints(start, end, terrain, shot, r)?;
    check_intent(shot, start, end)?;
    let solution = fit(shot, terrain, &anchors)?;
    Ok(solution)
}

pub fn check_intent(shot: &Shot, start: DVec3, end: DVec3) -> Result<(), String> {
    let center = shot.subject.center();
    match shot.kind.as_str() {
        "reveal" if end.y <= start.y && end.distance(center) <= start.distance(center) => {
            Err(binding(
                "reveal_intent",
                "reveal must rise or pull back from the subject",
            ))
        }
        "dolly" if end.distance(center) >= start.distance(center) => {
            Err(binding("dolly_intent", "dolly must approach the subject"))
        }
        "orbit" => {
            let radius = (start.x - center.x).hypot(start.z - center.z);
            let end_radius = (end.x - center.x).hypot(end.z - center.z);
            if radius == 0.0
                || (end_radius - radius).abs()
                    > f64::EPSILON * (start.length() + end.length() + center.length())
            {
                Err(binding(
                    "orbit_radius",
                    "orbit endpoints must share their horizontal framing radius",
                ))
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

/// Retry a feasible geometric route whose spherical consumer exceeds its jerk
/// budget. Candidate geometry comes from the declared framing shell and the
/// terrain pyramid, never from a fitted or increased motion budget.
pub fn motion_alternative(
    shot: &Shot,
    terrain: &Terrain,
    previous: &Solution,
) -> Result<Solution, String> {
    if !matches!(shot.kind.as_str(), "reveal" | "flyover") {
        return Err(binding(
            "shot_intent",
            "this intent cannot change its route",
        ));
    }
    let start = DVec3::from(previous.segments[0].eye0);
    let end = DVec3::from(previous.segments.last().unwrap().eye1);
    let center = shot.subject.center();
    let r = shell(shot);
    let margin = consumer_margin(terrain, r.1);
    let floor = start.y.max(end.y).max(
        terrain.ceiling(
            DVec3::ZERO,
            DVec3::new(terrain.width, 0.0, terrain.width),
            0.0,
        ) + shot.clearance
            + margin,
    );
    if shot.framing.region[0] + shot.framing.region[2] == 1.0
        && shot.framing.region[1] + shot.framing.region[3] == 1.0
    {
        let mut circular = shot.clone();
        circular.kind = "orbit".into();
        circular.start_eye = Some(start.to_array());
        circular.end_eye = Some(end.to_array());
        let angle = (start.z - center.z).atan2(start.x - center.x).to_degrees();
        let end_angle = (end.z - center.z).atan2(end.x - center.x).to_degrees();
        let mut sweep = end_angle - angle;
        while sweep > 180.0 {
            sweep -= 360.0;
        }
        while sweep < -180.0 {
            sweep += 360.0;
        }
        if sweep.abs() == 180.0 {
            sweep = sweep
                .abs()
                .copysign(shot.bearing_deg[1] - shot.bearing_deg[0]);
        }
        circular.bearing_deg = [angle, angle + sweep];
        let mut result = super::orbit::solve(&circular, terrain, start, end, floor)?;
        // Match the independently declared equal-phase reference. Empty lifts
        // leave additional time to the remaining phases. All bounds still gate.
        let count = result.segments.len();
        for (i, segment) in result.segments.iter_mut().enumerate() {
            segment.t0 = shot.duration_s * i as f64 / count as f64;
            segment.t1 = shot.duration_s * (i + 1) as f64 / count as f64;
            let speed = segment.arc.as_ref().map_or_else(
                || {
                    1.875
                        * DVec3::from(segment.eye0)
                            .distance(DVec3::from(segment.eye1))
                            .max(
                                DVec3::from(segment.target0).distance(DVec3::from(segment.target1)),
                            )
                },
                |arc| arc.unit_bounds(segment).0,
            ) / (segment.t1 - segment.t0);
            if speed > shot.max_speed || segment.jerk_bound() > shot.max_jerk {
                return Err(binding(
                    "duration_speed_jerk",
                    "motion alternative exceeds the declared budget",
                ));
            }
        }
        let last = result.segments.last_mut().unwrap();
        last.eye1 = end.to_array();
        last.target1 = target(end, shot)?.to_array();
        if result
            .segments
            .iter()
            .any(|segment| segment.arc.as_ref().is_some_and(|arc| !arc.valid(segment)))
        {
            return Err(binding(
                "orbit_endpoint",
                "motion alternative must preserve exact fixed endpoints",
            ));
        }
        result
            .corridor
            .push_str("; centered framing motion alternative, equal reference phases");
        return Ok(result);
    }
    // Off-center framing needs a nonvertical look direction. Use its existing
    // minimum framing radius as a lateral candidate offset, not a new gate.
    for sign in [-1.0, 1.0] {
        let z = center.z + sign * r.0;
        let a = DVec3::new(start.x, floor, z);
        let b = DVec3::new(end.x, floor, z);
        if !terrain.inside(a) || !terrain.inside(b) {
            continue;
        }
        if let Ok(mut result) = fit(shot, terrain, &[start, a, b, end]) {
            result
                .corridor
                .push_str("; lateral motion alternative from declared framing radius");
            return Ok(result);
        }
    }
    Err(binding(
        "jerk",
        "no safe motion alternative in the declared framing shell",
    ))
}

fn fit(shot: &Shot, terrain: &Terrain, anchors: &[DVec3]) -> Result<Solution, String> {
    let r = shell(shot);
    let start = anchors[0];
    let mut points = vec![start];
    let mut expanded = 0;
    for a in anchors.windows(2) {
        let (path, n) = route(a[0], a[1], terrain, shot, r)?;
        points.extend(path.into_iter().skip(1));
        expanded += n;
    }
    points.dedup();
    if points.len() < 2 {
        return Err(binding("shot_geometry", "shot has no displacement"));
    }
    let targets: Vec<DVec3> = points
        .iter()
        .map(|p| target(*p, shot))
        .collect::<Result<_, _>>()?;
    let lower: Vec<f64> = points
        .windows(2)
        .zip(targets.windows(2))
        .map(|(p, q)| {
            let length = p[0].distance(p[1]).max(q[0].distance(q[1]));
            (1.875 * length / shot.max_speed).max((60.0 * length / shot.max_jerk).cbrt())
        })
        .collect();
    let needed: f64 = lower.iter().sum();
    if needed > shot.duration_s {
        return Err(binding(
            "duration_speed_jerk",
            format!(
                "minimum rest-to-rest duration {needed} exceeds {}",
                shot.duration_s
            ),
        ));
    }
    // Integral squared jerk is 720*(|delta_eye|²+|delta_target|²)/h⁵.
    // Convex time allocation has h proportional to energy^(1/6), with the
    // speed/jerk lower bounds held active by finite water filling.
    let energy_weights: Vec<f64> = points
        .windows(2)
        .zip(targets.windows(2))
        .map(|(p, q)| (p[0].distance_squared(p[1]) + q[0].distance_squared(q[1])).powf(1.0 / 6.0))
        .collect();
    let mut active = vec![false; lower.len()];
    let weights = loop {
        let locked: f64 = lower
            .iter()
            .zip(&active)
            .filter(|(_, a)| **a)
            .map(|(v, _)| *v)
            .sum();
        let free: f64 = energy_weights
            .iter()
            .zip(&active)
            .filter(|(_, a)| !**a)
            .map(|(v, _)| *v)
            .sum();
        if free == 0.0 {
            break lower.clone();
        }
        let scale = (shot.duration_s - locked) / free;
        let mut changed = false;
        for i in 0..active.len() {
            if !active[i] && scale * energy_weights[i] < lower[i] {
                active[i] = true;
                changed = true;
            }
        }
        if !changed {
            break lower
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if active[i] {
                        *v
                    } else {
                        scale * energy_weights[i]
                    }
                })
                .collect::<Vec<_>>();
        }
    };
    let mut segments = Vec::new();
    let mut t0 = 0.0;
    for i in 0..weights.len() {
        let t1 = if i + 1 == weights.len() {
            shot.duration_s
        } else {
            t0 + weights[i]
        };
        segments.push(Segment {
            t0,
            t1,
            eye0: points[i].to_array(),
            eye1: points[i + 1].to_array(),
            target0: targets[i].to_array(),
            target1: targets[i + 1].to_array(),
            arc: None,
        });
        t0 = t1;
    }
    Ok(Solution {
        segments,
        expanded_nodes: expanded,
        corridor: if shot.kind == "dolly" {
            "Exact bilinear-cell straight dolly clearance; rest-to-rest minimum-jerk quintic"
        } else {
            "PROMETHEUS min-max dilated visibility graph; rest-to-rest minimum-jerk quintics"
        }
        .into(),
    })
}

/// Insert the independently reported failing pose, lift it within the declared
/// framing shell, then search and refit all connecting corridors. The caller
/// applies the shot's existing, explicit refinement-pass budget.
pub fn repair(
    shot: &Shot,
    terrain: &Terrain,
    solution: &Solution,
    time: f64,
    lift: bool,
) -> Result<Solution, String> {
    if time <= 0.0 || time >= shot.duration_s {
        return Err(binding(
            "fixed_endpoint",
            "validation cannot repair a fixed endpoint",
        ));
    }
    if shot.kind == "dolly" {
        let (eye, look) = solution.sample(time, 0);
        if !terrain.inside(eye) {
            return Err(binding(
                "terrain_bounds",
                "straight dolly validation leaves the declared terrain bounds",
            ));
        }
        let radius = shell(shot);
        let clearance = eye.y - terrain.height(eye);
        if clearance < shot.clearance {
            return Err(binding(
                "clearance",
                "straight dolly validation failed terrain clearance; detouring is unavailable",
            ));
        }
        framing_constraint(eye, shot, radius)?;
        if !super::validate::framing(shot, eye, look, shot.framing.fov_deg) {
            return Err(binding(
                "framing",
                "straight dolly validation failed projected framing; detouring is unavailable",
            ));
        }
        if !terrain
            .subject_points(shot)?
            .iter()
            .all(|q| terrain.visible(eye, *q))
        {
            return Err(binding(
                "occlusion",
                "straight dolly validation failed subject visibility; detouring is unavailable",
            ));
        }
        if solution
            .sample(time, 1)
            .0
            .length()
            .max(solution.sample(time, 1).1.length())
            > shot.max_speed
        {
            return Err(binding(
                "speed",
                "straight dolly validation failed its speed budget",
            ));
        }
        if solution
            .segments
            .iter()
            .any(|s| s.jerk_bound() > shot.max_jerk)
        {
            return Err(binding(
                "jerk",
                "straight dolly validation failed its jerk budget",
            ));
        }
        return Err(binding(
            "repair_not_applicable",
            "no solved-pose violation at the requested time; a dolly cannot detour or lift, and consumer validation retains its original diagnostic",
        ));
    }
    let r = shell(shot);
    let (mut eye, _) = solution.sample(time, 0);
    let center = shot.subject.center();
    let margin = consumer_margin(terrain, r.1);
    let upper = center.y
        + (r.1 * r.1 - (eye.x - center.x).powi(2) - (eye.z - center.z).powi(2))
            .max(0.0)
            .sqrt();
    eye.y = eye
        .y
        .max(terrain.ceiling(eye, eye, shot.clearance) + shot.clearance + margin);
    // Clearance/occlusion repair lifts within the shell. Framing-only repair
    // adds an aiming waypoint on the existing path, avoiding an unrelated climb.
    if lift {
        eye.y += (upper - eye.y) * 0.5;
    }
    if !family(eye, shot, r) {
        return Err(binding(
            "clearance_framing_corridor",
            "repair exhausted the framing shell",
        ));
    }
    if shot.kind == "orbit" {
        return super::orbit::solve(
            shot,
            terrain,
            DVec3::from(solution.segments[0].eye0),
            DVec3::from(solution.segments.last().unwrap().eye1),
            eye.y,
        );
    }
    let mut anchors = vec![DVec3::from(solution.segments[0].eye0)];
    for s in &solution.segments {
        if time > s.t0 && time <= s.t1 {
            anchors.push(eye);
        }
        anchors.push(DVec3::from(s.eye1));
    }
    fit(shot, terrain, &anchors)
}
