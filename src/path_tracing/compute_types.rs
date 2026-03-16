use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub frame_index: u32,
    pub aov_flags: u32,
    pub cam_origin: [f32; 3],
    pub cam_fov_y: f32,
    pub cam_right: [f32; 3],
    pub cam_aspect: f32,
    pub cam_up: [f32; 3],
    pub cam_exposure: f32,
    pub cam_forward: [f32; 3],
    pub seed_hi: u32,
    pub seed_lo: u32,
    pub _pad_end: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub emissive: [f32; 3],
    pub roughness: f32,
    pub ior: f32,
    pub ax: f32,
    pub ay: f32,
    pub _pad1: f32,
}
