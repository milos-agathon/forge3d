use bytemuck::{Pod, Zeroable};
use glam::Mat4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshUniforms {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub light_dir_ws: [f32; 4],
    pub mr: [f32; 2], // metallic, roughness
    pub _pad_mr: [f32; 2],
}

impl Default for MeshUniforms {
    fn default() -> Self {
        Self {
            model: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            color: [1.0, 1.0, 1.0, 1.0],
            light_dir_ws: [0.0, -1.0, 0.0, 0.0],
            mr: [0.0, 1.0],
            _pad_mr: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPN {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}
