use super::{
    binding,
    solver::{self, Terrain},
    validate::{self, Keyframe, Report},
    Shot, Solution,
};
use pyo3::prelude::*;
use serde::Serialize;

fn checked_shot(text: &str) -> Result<Shot, String> {
    let s: Shot = serde_json::from_str(text).map_err(|e| binding("spec", e))?;
    let f = &s.framing;
    let scalars = [
        s.duration_s,
        s.clearance,
        s.max_speed,
        s.max_jerk,
        f.fov_deg,
        f.aspect,
    ];
    if scalars.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || f.fov_deg >= 180.0
        || !["reveal", "orbit", "flyover", "dolly"].contains(&s.kind.as_str())
        || f.region
            .iter()
            .chain(f.size_frac_band.iter())
            .chain(s.transition.iter())
            .any(|v| !v.is_finite())
        || f.region[0] < 0.0
        || f.region[1] < 0.0
        || f.region[2] > 1.0
        || f.region[3] > 1.0
        || f.region[0] >= f.region[2]
        || f.region[1] >= f.region[3]
        || f.size_frac_band[0] <= 0.0
        || f.size_frac_band[1] > 1.0
        || f.size_frac_band[0] >= f.size_frac_band[1]
        || !f.headroom.is_finite()
        || f.headroom < 0.0
        || f.headroom >= 1.0
        || s.transition.iter().any(|v| *v < 0.0)
        || s.transition.iter().sum::<f64>() > s.duration_s
        || s.subject
            .bounds
            .iter()
            .flatten()
            .chain(s.start_eye.iter().flatten())
            .chain(s.end_eye.iter().flatten())
            .chain(s.bearing_deg.iter())
            .chain(s.elevation_deg.iter())
            .any(|v| !v.is_finite())
        || (0..3).any(|i| s.subject.bounds[1][i] <= s.subject.bounds[0][i])
    {
        return Err(binding(
            "spec",
            "invalid shot geometry, framing, transition or motion budget",
        ));
    }
    Ok(s)
}
fn py_error(e: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e)
}
#[derive(Serialize)]
struct Planned {
    solution: Solution,
    keyframes: Vec<Keyframe>,
    validation: Report,
    solve_time_s: f64,
    refinement_passes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    orbit_adjustments: Option<serde_json::Value>,
}

fn orbit_adjustments(shot: &Shot, solution: &Solution) -> Option<serde_json::Value> {
    if shot.kind != "orbit" {
        return None;
    }
    let start = glam::DVec3::from(solution.segments[0].eye0);
    let end = glam::DVec3::from(solution.segments.last().unwrap().eye1);
    let center = shot.subject.center();
    let principal = (start.z - center.z).atan2(start.x - center.x).to_degrees();
    let bearing = principal + 360.0 * ((shot.bearing_deg[0] - principal) / 360.0).round();
    let sweep = shot.bearing_deg[1] - shot.bearing_deg[0];
    Some(serde_json::json!({
        "requested_bearing_deg": shot.bearing_deg,
        "effective_bearing_deg": [bearing, bearing + sweep],
        "signed_sweep_deg": sweep,
        "requested_start_eye": shot.start_eye, "effective_start_eye": start.to_array(),
        "requested_end_eye": shot.end_eye, "effective_end_eye": end.to_array(),
        "end_adjustment_distance": shot.end_eye.map_or(0.0, |p| end.distance(p.into())),
        "fixed_endpoint_relative_tolerance": 0.001
    }))
}

#[pyfunction]
pub fn cinema_solve(
    spec_json: &str,
    heights: Vec<f32>,
    width: u32,
    height: u32,
    terrain_width: f64,
) -> PyResult<String> {
    let shot = checked_shot(spec_json).map_err(py_error)?;
    let terrain = Terrain::new(heights, width, height, terrain_width).map_err(py_error)?;
    let started = std::time::Instant::now();
    let mut solution = solver::solve(&shot, &terrain).map_err(py_error)?;
    let mut motion_alternative_tried = false;
    validate::check_resolution(&shot, 0.0).map_err(py_error)?;
    // Six samples resolve a degree-five segment; refine only as required by
    // the independent consumer error/clearance checks, rather than quantizing
    // orbit angles into thousands of nearly identical float32 keys.
    let mut panels = solution
        .segments
        .len()
        .checked_mul(5)
        .ok_or_else(|| py_error(binding("consumer_precision", "keyframe count overflow")))?;
    for pass in 0..=shot.max_refine_passes {
        let keys = validate::resample(&solution, &shot, panels).map_err(py_error)?;
        if validate::approximation(&shot, &solution, &keys, 0.0).map_err(py_error)? >= 0.001 {
            if pass == shot.max_refine_passes {
                return Err(py_error(binding(
                    "consumer_resampling",
                    "declared refinement passes exhausted",
                )));
            }
            panels = panels.checked_mul(2).ok_or_else(|| {
                py_error(binding("consumer_precision", "keyframe count overflow"))
            })?;
            continue;
        }
        let validation =
            validate::validate(&shot, &solution, &keys, &terrain, 0.0).map_err(py_error)?;
        if let Err(diagnostic) = validate::require(&shot, &validation) {
            if matches!(validate::failure(&shot, &validation), Some("jerk"))
                && !motion_alternative_tried
                && matches!(shot.kind.as_str(), "reveal" | "flyover")
                && pass < shot.max_refine_passes
            {
                motion_alternative_tried = true;
                if let Ok(alternative) = solver::motion_alternative(&shot, &terrain, &solution) {
                    solution = alternative;
                    panels = solution.segments.len().checked_mul(5).ok_or_else(|| {
                        py_error(binding("consumer_precision", "keyframe count overflow"))
                    })?;
                    continue;
                }
            }
            if matches!(validate::failure(&shot, &validation), Some("jerk"))
                && validation.spline_max_jerk <= shot.max_jerk
                && pass < shot.max_refine_passes
            {
                panels = panels.checked_mul(2).ok_or_else(|| {
                    py_error(binding("consumer_precision", "keyframe count overflow"))
                })?;
                continue;
            }
            if pass == shot.max_refine_passes
                || shot.kind == "dolly"
                || validation.repair_times.is_empty()
                || !matches!(
                    validate::failure(&shot, &validation),
                    Some("clearance" | "framing" | "occlusion")
                )
            {
                return Err(py_error(diagnostic));
            }
            solution = solver::repair(
                &shot,
                &terrain,
                &solution,
                validation.repair_times[0],
                validate::failure(&shot, &validation) != Some("framing"),
            )
            .map_err(py_error)?;
            continue;
        }
        return serde_json::to_string(&Planned {
            orbit_adjustments: orbit_adjustments(&shot, &solution),
            solution,
            keyframes: keys,
            validation,
            solve_time_s: started.elapsed().as_secs_f64(),
            refinement_passes: pass,
        })
        .map_err(|e| py_error(e.to_string()));
    }
    unreachable!("inclusive refinement loop either returns or retries")
}
#[pyfunction]
pub fn cinema_validate(
    spec_json: &str,
    solution_json: &str,
    keyframes_json: &str,
    heights: Vec<f32>,
    width: u32,
    height: u32,
    terrain_width: f64,
    time_offset: f64,
) -> PyResult<String> {
    let shot = checked_shot(spec_json).map_err(py_error)?;
    let solution: Solution =
        serde_json::from_str(solution_json).map_err(|e| py_error(binding("artifact", e)))?;
    let keys: Vec<Keyframe> =
        serde_json::from_str(keyframes_json).map_err(|e| py_error(binding("artifact", e)))?;
    if solution.segments.is_empty()
        || keys.len() < 2
        || solution
            .segments
            .iter()
            .any(|s| s.t1 <= s.t0 || s.arc.as_ref().is_some_and(|arc| !arc.valid(s)))
        || keys.windows(2).any(|k| k[1].time <= k[0].time)
    {
        return Err(py_error(binding(
            "artifact",
            "empty or unordered spline/keyframes",
        )));
    }
    if !time_offset.is_finite()
        || time_offset < 0.0
        || solution.segments[0].t0 != 0.0
        || solution.segments.last().unwrap().t1 != shot.duration_s
        || solution.segments.windows(2).any(|s| s[0].t1 != s[1].t0)
        || keys[0].time > time_offset
        || keys.last().unwrap().time < f64::from((time_offset + shot.duration_s) as f32)
        || keys
            .iter()
            .any(|k| k.radius <= 0.0 || k.fov <= 0.0 || k.fov >= 180.0)
    {
        return Err(py_error(binding(
            "artifact",
            "invalid duration, coverage or camera projection",
        )));
    }
    let terrain = Terrain::new(heights, width, height, terrain_width).map_err(py_error)?;
    validate::check_resolution(&shot, time_offset).map_err(py_error)?;
    let report =
        validate::validate(&shot, &solution, &keys, &terrain, time_offset).map_err(py_error)?;
    validate::require(&shot, &report).map_err(py_error)?;
    serde_json::to_string(&report).map_err(|e| py_error(e.to_string()))
}
