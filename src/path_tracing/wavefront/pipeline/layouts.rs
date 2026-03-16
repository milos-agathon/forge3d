use super::*;

impl WavefrontPipelines {
    pub(super) fn create_uniforms_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniforms-layout"),
            entries: &[uniform_entry(0)],
        })
    }

    pub(super) fn create_accum_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accum-layout"),
            entries: &[aov_entry(0, false)],
        })
    }
}
