use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoSettingsUniform {
    radius: f32,
    intensity: f32,
    bias: f32,
    num_samples: u32,
    technique: u32,
    frame_index: u32,
    inv_resolution: [f32; 2],
    proj_scale: f32,
    ao_min: f32,
}

pub(super) struct SsaoResources {
    radius: f32,
    intensity: f32,
    bias: f32,
    pub(super) width: u32,
    pub(super) height: u32,
    sampler: wgpu::Sampler,
    blur_sampler: wgpu::Sampler,
    settings_buffer: wgpu::Buffer,
    blur_settings_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    ao_texture: wgpu::Texture,
    ao_view: wgpu::TextureView,
    blur_texture: wgpu::Texture,
    blur_view: wgpu::TextureView,
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,
    noise_sampler: wgpu::Sampler,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    ssao_bind_group_layout: wgpu::BindGroupLayout,
    ssao_output_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    ssao_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
}

include!("ssao/setup.rs");
include!("ssao/helpers.rs");
include!("ssao/constructor.rs");
include!("ssao/runtime.rs");

#[cfg(test)]
mod ssao_uniform_tests {
    use super::compute_ssao_proj_scale;

    #[test]
    fn proj_scale_matches_documented_formula_for_fov() {
        let fov_y = 60.0_f32.to_radians();
        let h = 480u32;
        let proj = crate::camera::perspective_wgpu(fov_y, 1.0, 0.1, 100.0);
        let expected = 0.5 * h as f32 * (1.0 / (fov_y * 0.5).tan());
        let got = compute_ssao_proj_scale(h, &proj);
        assert!((got - expected).abs() < 1e-4 * expected.max(1.0));
    }
}
