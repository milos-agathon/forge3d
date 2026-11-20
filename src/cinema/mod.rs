//! AUTOCHIRON: deterministic terrain-constrained shot planning.
#[cfg(feature = "extension-module")]
pub mod bindings;
mod gpu_trace;
mod motion;
mod orbit;
pub mod solver;
#[cfg(test)]
mod tests;
pub mod validate;

use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    pub bounds: [[f64; 3]; 2],
}
impl Subject {
    pub fn center(&self) -> DVec3 {
        (DVec3::from(self.bounds[0]) + DVec3::from(self.bounds[1])) * 0.5
    }
    pub fn corners(&self) -> Vec<DVec3> {
        (0..8)
            .map(|i| {
                DVec3::new(
                    self.bounds[(i & 1) as usize][0],
                    self.bounds[((i >> 1) & 1) as usize][1],
                    self.bounds[((i >> 2) & 1) as usize][2],
                )
            })
            .collect()
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FramingSpec {
    pub region: [f64; 4],
    pub size_frac_band: [f64; 2],
    pub headroom: f64,
    pub fov_deg: f64,
    pub aspect: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Shot {
    pub kind: String,
    pub subject: Subject,
    pub duration_s: f64,
    pub framing: FramingSpec,
    pub transition: [f64; 2],
    pub clearance: f64,
    pub max_speed: f64,
    pub max_jerk: f64,
    pub start_eye: Option<[f64; 3]>,
    pub end_eye: Option<[f64; 3]>,
    pub bearing_deg: [f64; 2],
    pub elevation_deg: [f64; 2],
    pub max_refine_passes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub eye0: [f64; 3],
    pub eye1: [f64; 3],
    pub target0: [f64; 3],
    pub target1: [f64; 3],
    /// Absent on legacy straight quintics, preserving their serialized bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arc: Option<Arc>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arc {
    pub center: [f64; 3],
    pub sweep: f64,
}
impl Segment {
    pub fn sample(&self, t: f64, derivative: usize) -> (DVec3, DVec3) {
        let h = self.t1 - self.t0;
        let u = ((t - self.t0) / h).clamp(0.0, 1.0);
        if let Some(arc) = &self.arc {
            return arc.sample(self, u, derivative);
        }
        let q = match derivative {
            0 => u.powi(3) * (10.0 + u * (-15.0 + 6.0 * u)),
            1 => 30.0 * u.powi(2) * (1.0 - u).powi(2) / h,
            2 => 60.0 * u * (1.0 - 3.0 * u + 2.0 * u * u) / h.powi(2),
            _ => (60.0 - 360.0 * u + 360.0 * u * u) / h.powi(3),
        };
        let a = DVec3::from(self.eye0);
        let b = DVec3::from(self.target0);
        let eye = (DVec3::from(self.eye1) - a) * q;
        let target = (DVec3::from(self.target1) - b) * q;
        if derivative == 0 {
            (a + eye, b + target)
        } else {
            (eye, target)
        }
    }
    /// Conservative bound over the entire segment, including arc interior.
    pub fn jerk_bound(&self) -> f64 {
        if let Some(arc) = &self.arc {
            arc.unit_bounds(self).1 / (self.t1 - self.t0).powi(3)
        } else {
            let (eye, target) = self.sample(self.t0, 3);
            eye.length().max(target.length())
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Solution {
    pub segments: Vec<Segment>,
    pub expanded_nodes: usize,
    pub corridor: String,
}
impl Solution {
    pub fn sample(&self, t: f64, derivative: usize) -> (DVec3, DVec3) {
        let i = self
            .segments
            .partition_point(|s| s.t1 < t)
            .min(self.segments.len() - 1);
        self.segments[i].sample(t, derivative)
    }
}
pub fn binding(constraint: &str, detail: impl std::fmt::Display) -> String {
    serde_json::json!({"status":"infeasible", "binding_constraint":constraint,
        "detail":detail.to_string()})
    .to_string()
}
