#[derive(Debug, Clone)]
pub struct SceneGlobals {
    pub globals: crate::terrain::Globals,
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
}

impl Default for SceneGlobals {
    fn default() -> Self {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0, 2.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = crate::camera::perspective_wgpu(45f32.to_radians(), 4.0 / 3.0, 0.1, 100.0);
        Self {
            globals: crate::terrain::Globals::default(),
            view,
            proj,
        }
    }
}

pub(super) struct Text3DInstance {
    pub(super) vbuf: wgpu::Buffer,
    pub(super) ibuf: wgpu::Buffer,
    pub(super) index_count: u32,
    pub(super) vertex_count: u32,
    pub(super) model: glam::Mat4,
    pub(super) color: [f32; 4],
    pub(super) light_dir: [f32; 3],
    pub(super) light_intensity: f32,
    pub(super) metallic: f32,
    pub(super) roughness: f32,
}

// F16: GPU Instancing batch description
#[cfg(feature = "enable-gpu-instancing")]
pub(super) struct InstancedBatch {
    pub(super) vbuf: wgpu::Buffer,
    pub(super) ibuf: wgpu::Buffer,
    pub(super) instbuf: wgpu::Buffer,
    pub(super) index_count: u32,
    pub(super) instance_count: u32,
    pub(super) color: [f32; 4],
    pub(super) light_dir: [f32; 3],
    pub(super) light_intensity: f32,
}
