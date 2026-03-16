fn create_ssao_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // AO textures are single-channel float; match ssao.wgsl's r32float storage textures.
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn compute_ssao_proj_scale(height: u32, projection: &glam::Mat4) -> f32 {
    (0.5 * height.max(1) as f32 * projection.y_axis.y.abs()).max(1e-4)
}

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
