// src/viewer/viewer_types.rs
// GPU uniform structs and mesh types for the interactive viewer
// RELEVANT FILES: shaders/viewer_lit.wgsl, shaders/volumetric.wgsl

use crate::geometry::MeshBuffers;
use glam::{Mat3, Mat4, Vec2, Vec3};

/// Sky rendering uniforms (P6-01)
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniforms {
    pub sun_direction: [f32; 3],
    pub turbidity: f32,
    pub ground_albedo: f32,
    pub model: u32, // 0=Preetham, 1=Hosek-Wilkie
    pub sun_intensity: f32,
    pub exposure: f32,
    pub _pad: [f32; 4],
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

/// P6: Volumetric fog uniforms matching shaders/volumetric.wgsl
#[repr(C, align(16))]
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct VolumetricUniforms {
    pub density: f32,
    pub height_falloff: f32,
    pub phase_g: f32,
    pub max_steps: u32,
    pub start_distance: f32,
    pub max_distance: f32,
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
    pub _pad1: u32,
    pub _pad2: u32,
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
    pub geom_vb: Option<wgpu::Buffer>,
    pub geom_ib: Option<wgpu::Buffer>,
    pub geom_index_count: u32,
    pub sky_enabled: bool,
    pub fog_enabled: bool,
    pub viz_mode: super::viewer_enums::VizMode,
    pub gi_viz_mode: crate::cli::args::GiVizMode,
    pub camera_mode: super::camera_controller::CameraMode,
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

    pub fn push_vertex(
        &mut self,
        position: Vec3,
        normal: Vec3,
        uv: Vec2,
        roughness: f32,
        metallic: f32,
    ) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(PackedVertex {
            position: position.to_array(),
            normal: normal.normalize_or_zero().to_array(),
            uv: uv.to_array(),
            rough_metal: [roughness, metallic],
        });
        idx
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
