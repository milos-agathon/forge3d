fn create_ssao_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> Result<(TrackedTexture, wgpu::TextureView), RenderError> {
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}

fn compute_ssao_proj_scale(height: u32, projection: &glam::Mat4) -> f32 {
    (0.5 * height.max(1) as f32 * projection.y_axis.y.abs()).max(1e-4)
}
