// src/viewer/viewer_types.rs
// GPU uniform structs and mesh types for the interactive viewer
// RELEVANT FILES: shaders/viewer_lit.wgsl, shaders/volumetric.wgsl

use crate::geometry::MeshBuffers;
use glam::{Mat3, Mat4, Vec2, Vec3};

/// Sky rendering uniforms (P6-01)
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniforms {
    pub sun_direction_turbidity: [f32; 4],
    pub ground_albedo_sun_size_sun_intensity_exposure: [f32; 4],
    pub model_pad: [u32; 4], // x=model (0=Preetham, 1=Hosek-Wilkie)
    pub hosek_coeffs_a_d: [[f32; 4]; 3],
    pub hosek_coeffs_e_h: [[f32; 4]; 3],
    pub hosek_coeff_i: [f32; 4],
    pub hosek_radiance: [f32; 4],
}

impl SkyUniforms {
    pub fn new(
        sun_direction: [f32; 3],
        turbidity: f32,
        ground_albedo: f32,
        sun_size: f32,
        sun_intensity: f32,
        exposure: f32,
        model: u32,
    ) -> Self {
        let turbidity = turbidity.clamp(1.0, 10.0);
        let ground_albedo = ground_albedo.clamp(0.0, 1.0);
        let solar_elevation = sun_direction[1]
            .clamp(0.0, 1.0)
            .asin()
            .clamp(0.0, std::f32::consts::FRAC_PI_2);
        let hosek =
            crate::terrain::hosek_sky::hosek_rgb_sky(turbidity, ground_albedo, solar_elevation);

        Self {
            sun_direction_turbidity: [
                sun_direction[0],
                sun_direction[1],
                sun_direction[2],
                turbidity,
            ],
            ground_albedo_sun_size_sun_intensity_exposure: [
                ground_albedo,
                sun_size.max(0.0),
                sun_intensity.max(0.0),
                exposure.max(0.0),
            ],
            model_pad: [model, 0, 0, 0],
            hosek_coeffs_a_d: hosek.uniform_a_d(),
            hosek_coeffs_e_h: hosek.uniform_e_h(),
            hosek_coeff_i: hosek.uniform_i(),
            hosek_radiance: hosek.uniform_radiance(),
        }
    }
}

#[cfg(test)]
mod sky_tests {
    use super::SkyUniforms;

    #[test]
    fn sky_uniforms_match_wgsl_size() {
        assert_eq!(std::mem::size_of::<SkyUniforms>(), 176);
    }
}

/// Std140-compatible packed layout for VolumetricUniforms
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricUniformsStd140 {
    pub density: f32,
    pub height_falloff: f32,
    pub phase_g: f32,
    pub max_steps: u32,
    pub start_distance: f32,
    pub max_distance: f32,
    pub _pad_a0: f32,
    pub _pad_a1: f32,
    pub scattering_color: [f32; 3],
    pub absorption: f32,
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub ambient_color: [f32; 3],
    pub temporal_alpha: f32,
    pub use_shadows: u32,
    pub jitter_strength: f32,
    pub frame_index: u32,
    pub _pad0: u32,
}

/// Camera uniforms for fog rendering
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FogCameraUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub eye_position: [f32; 3],
    pub near: f32,
    pub far: f32,
    pub _pad: [f32; 3],
}

/// Std140-compatible upsample params for fog_upsample.wgsl
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FogUpsampleParamsStd140 {
    pub sigma: f32,
    pub use_bilateral: u32,
    pub _pad: [f32; 2],
}

/// Packed vertex for viewer scene geometry
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub rough_metal: [f32; 2],
}

/// State saved before P5.1 Cornell scene setup, restored after capture
pub struct P51CornellSceneState {
    pub geom_vb: Option<crate::core::resource_tracker::TrackedBuffer>,
    pub geom_ib: Option<crate::core::resource_tracker::TrackedBuffer>,
    pub geom_index_count: u32,
    pub sky_enabled: bool,
    pub fog_enabled: bool,
    pub viz_mode: super::viewer_enums::VizMode,
    pub gi_viz_mode: crate::cli::args::GiVizMode,
    pub camera_eye: Vec3,
    pub camera_target: Vec3,
}

/// Scene mesh container for viewer geometry
#[derive(Default)]
pub struct SceneMesh {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

impl SceneMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extend_with_mesh(
        &mut self,
        mesh: &MeshBuffers,
        transform: Mat4,
        roughness: f32,
        metallic: f32,
    ) {
        let base = self.vertices.len() as u32;
        let normal_matrix = Mat3::from_mat4(transform).inverse().transpose();
        for i in 0..mesh.positions.len() {
            let pos = Vec3::from_array(mesh.positions[i]);
            let pos_w = (transform * pos.extend(1.0)).truncate();
            let normal_src = if mesh.normals.len() == mesh.positions.len() {
                Vec3::from_array(mesh.normals[i])
            } else {
                Vec3::Y
            };
            let normal_w = (normal_matrix * normal_src).normalize_or_zero();
            let uv = if mesh.uvs.len() == mesh.positions.len() {
                Vec2::from_array(mesh.uvs[i])
            } else {
                Vec2::ZERO
            };
            self.vertices.push(PackedVertex {
                position: pos_w.to_array(),
                normal: normal_w.to_array(),
                uv: uv.to_array(),
                rough_metal: [roughness, metallic],
            });
        }
        for &idx in &mesh.indices {
            self.indices.push(base + idx);
        }
    }
}
