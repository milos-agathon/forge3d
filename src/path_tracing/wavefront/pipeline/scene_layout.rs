use super::*;

impl WavefrontPipelines {
    pub(super) fn create_scene_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene-layout"),
            entries: &[
                aov_entry(0, true),
                aov_entry(1, true),
                aov_entry(2, true),
                aov_entry(3, true),
                aov_entry(4, true),
                aov_entry(5, true),
                aov_entry(6, true),
                aov_entry(7, true),
                aov_entry(8, true),
                aov_entry(9, true),
                aov_entry(10, false),
                aov_entry(11, false),
                uniform_entry(12),
                aov_entry(13, false),
                aov_entry(14, true),
                aov_entry(15, true),
                aov_entry(16, false),
                aov_entry(17, false),
                aov_entry(18, false),
                uniform_entry(19),
                aov_entry(20, true),
            ],
        })
    }
}
