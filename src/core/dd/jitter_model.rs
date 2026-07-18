use super::gpu_report::DdSelftestReport;
use super::DDVec3;
use crate::core::error::{RenderError, RenderResult};
use bytemuck::{Pod, Zeroable};
use glam::DVec3;

pub(super) const DEFAULT_FRAMES: u32 = 1_000;
pub(super) const CAMERA_STEP_METRES: f64 = 0.001;

#[derive(Clone, Debug)]
pub struct DdJitterReport {
    pub dd_errors_px: Vec<f64>,
    pub f32_errors_px: Vec<f64>,
    pub dd_max_error_px: f64,
    pub f32_max_error_px: f64,
    pub raw_over_one_px: u32,
    pub dd_hash_a: String,
    pub dd_hash_b: String,
    pub backend: String,
    pub shader_label: String,
    pub certificate_json: String,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraPair {
    pub camera_dd: DDVec3,
    pub padding: [f32; 2],
    pub raw_f32: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub render_frame: u32,
}

pub(super) struct JitterModel {
    pub original_points: [DVec3; 3],
    pub points: [DDVec3; 3],
    pub camera_values: Vec<DVec3>,
    pub cameras: Vec<CameraPair>,
    pub globals: Globals,
}

pub(super) fn build_model(frames: u32) -> RenderResult<JitterModel> {
    let base =
        crate::geo::projections::geocentric::wgs84_geodetic_to_ecef(86.9250, 27.9881, 8_848.86)
            .map_err(|error| {
                RenderError::render(format!("Everest ECEF conversion failed: {error}"))
            })?;
    let original_points = [
        base + DVec3::new(0.0, -0.1, 0.0),
        base + DVec3::new(0.2, 0.1, 0.0),
        base + DVec3::new(-0.2, 0.1, 0.0),
    ];
    let points = original_points.map(DDVec3::from_dvec3);
    let camera_values: Vec<DVec3> = (0..frames)
        .map(|frame| base + DVec3::new(0.0, -0.25 + frame as f64 * CAMERA_STEP_METRES, 0.0))
        .collect();
    let cameras = camera_values
        .iter()
        .map(|value| CameraPair {
            camera_dd: DDVec3::from_dvec3(*value),
            padding: [0.0; 2],
            raw_f32: [value.x as f32, value.y as f32, value.z as f32, 0.0],
        })
        .collect();
    Ok(JitterModel {
        original_points,
        points,
        camera_values,
        cameras,
        globals: Globals {
            view_proj: glam::Mat4::from_scale(glam::Vec3::new(2.0, 2.0, 1.0)).to_cols_array_2d(),
            frame_count: frames,
            width: 64,
            height: 64,
            render_frame: 0,
        },
    })
}

pub(super) struct JitterMetrics {
    pub dd_errors_px: Vec<f64>,
    pub f32_errors_px: Vec<f64>,
    pub dd_max_error_px: f64,
    pub f32_max_error_px: f64,
    pub raw_over_one_px: u32,
}

pub(super) fn reduce_measurements(
    measured: &[[f32; 2]],
    model: &JitterModel,
) -> RenderResult<JitterMetrics> {
    let mut dd_errors_px = Vec::with_capacity(measured.len());
    let mut f32_errors_px = Vec::with_capacity(measured.len());
    for (frame, values) in measured.iter().enumerate() {
        let local = model.original_points[0] - model.camera_values[frame];
        let reference = (1.0 - 2.0 * local.y) * 0.5 * model.globals.height as f64;
        if !values[0].is_finite() || !values[1].is_finite() || !reference.is_finite() {
            return Err(RenderError::render(format!(
                "non-finite jitter measurement at frame {frame}"
            )));
        }
        dd_errors_px.push((values[0] as f64 - reference).abs());
        f32_errors_px.push((values[1] as f64 - reference).abs());
    }
    let dd_max_error_px = dd_errors_px.iter().copied().fold(0.0, f64::max);
    let f32_max_error_px = f32_errors_px.iter().copied().fold(0.0, f64::max);
    let raw_over_one_px = f32_errors_px.iter().filter(|value| **value > 1.0).count() as u32;
    Ok(JitterMetrics {
        dd_errors_px,
        f32_errors_px,
        dd_max_error_px,
        f32_max_error_px,
        raw_over_one_px,
    })
}

pub(super) fn jitter_evidence(
    capability: &DdSelftestReport,
    frames: u32,
    metrics: &JitterMetrics,
    shader_hash: String,
    hashes: (&str, &str),
) -> crate::core::certificate::JitterEvidence {
    crate::core::certificate::JitterEvidence {
        unit: "px".to_string(),
        backend: capability.backend.clone(),
        adapter: capability.adapter.clone(),
        two_prod_variant: capability.two_prod_variant.as_str().to_string(),
        shader_label: super::jitter::LABEL.to_string(),
        shader_hash,
        frame_count: frames,
        camera_step_metres: CAMERA_STEP_METRES,
        dd_max_error_px: metrics.dd_max_error_px,
        threshold_px: 0.01,
        raw_max_error_px: metrics.f32_max_error_px,
        raw_over_one_px: metrics.raw_over_one_px,
        dd_hash_a: hashes.0.to_string(),
        dd_hash_b: hashes.1.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_measurement_fails_closed() {
        let model = build_model(1).unwrap();
        assert!(reduce_measurements(&[[f32::NAN, 0.0]], &model).is_err());
        assert!(reduce_measurements(&[[0.0, f32::INFINITY]], &model).is_err());
    }
}
