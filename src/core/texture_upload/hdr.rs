use super::types::{HdrFormat, HdrTexture, HdrTextureConfig};
use crate::core::error::{RenderError, RenderResult};
use crate::core::memory_tracker::{global_tracker, MemoryMetrics};
use crate::core::resource_tracker::tracked_create_texture;
use crate::core::texture_layout::TransferLayout;

pub fn create_texture_rgba32f(data: &[f32], config: HdrTextureConfig) -> RenderResult<HdrTexture> {
    if config.format != HdrFormat::Rgba32Float {
        return Err(RenderError::upload(
            "create_texture_rgba32f requires Rgba32Float format".to_string(),
        ));
    }

    create_hdr_texture_internal(bytemuck::cast_slice(data), config)
}

pub fn create_texture_rgba16f(data: &[u16], config: HdrTextureConfig) -> RenderResult<HdrTexture> {
    if config.format != HdrFormat::Rgba16Float {
        return Err(RenderError::upload(
            "create_texture_rgba16f requires Rgba16Float format".to_string(),
        ));
    }

    create_hdr_texture_internal(bytemuck::cast_slice(data), config)
}

pub fn create_texture_rgb32f_with_alpha(
    rgb_data: &[f32],
    alpha: f32,
    config: HdrTextureConfig,
) -> RenderResult<HdrTexture> {
    if config.format != HdrFormat::Rgba32Float {
        return Err(RenderError::upload(
            "create_texture_rgb32f_with_alpha requires Rgba32Float format".to_string(),
        ));
    }

    let pixel_count = (config.width * config.height) as usize;
    if rgb_data.len() != pixel_count * 3 {
        return Err(RenderError::upload(format!(
            "RGB data length mismatch: expected {} ({}x{}x3), got {}",
            pixel_count * 3,
            config.width,
            config.height,
            rgb_data.len()
        )));
    }

    let mut rgba_data = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let base = i * 3;
        rgba_data.push(rgb_data[base]);
        rgba_data.push(rgb_data[base + 1]);
        rgba_data.push(rgb_data[base + 2]);
        rgba_data.push(alpha);
    }

    create_texture_rgba32f(&rgba_data, config)
}

fn create_hdr_texture_internal(data: &[u8], config: HdrTextureConfig) -> RenderResult<HdrTexture> {
    validate_config(&config)?;

    let g = crate::core::gpu::try_ctx()?;
    let format = config.format.to_wgpu();

    let texture_size =
        crate::core::texture_layout::resident_2d_size(config.width, config.height, format);
    let tracker = global_tracker();
    let current_metrics = tracker.get_metrics();
    check_hdr_budget(texture_size, &current_metrics)?;

    let mip_level_count = if config.generate_mipmaps {
        (config.width.max(config.height) as f32).log2().floor() as u32 + 1
    } else {
        1
    };

    let texture = tracked_create_texture(
        &g.device,
        &wgpu::TextureDescriptor {
            label: config.label.as_deref(),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: config.usage,
            view_formats: &[],
        },
    )?;

    upload_texture_data(
        &g.queue,
        &texture,
        data,
        config.width,
        config.height,
        config.format,
    )?;

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(HdrTexture {
        texture,
        view,
        format: config.format,
        width: config.width,
        height: config.height,
    })
}

fn upload_texture_data(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    data: &[u8],
    width: u32,
    height: u32,
    format: HdrFormat,
) -> RenderResult<()> {
    let layout = TransferLayout::queue_write(
        format.to_wgpu(),
        wgpu::TextureAspect::All,
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    )
    .map_err(RenderError::upload)?;
    let expected_size = layout.payload_bytes as usize;

    if data.len() != expected_size {
        return Err(RenderError::upload(format!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_size,
            data.len()
        )));
    }

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        layout.image_data_layout(),
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    Ok(())
}

pub(super) fn check_hdr_budget(texture_size: u64, metrics: &MemoryMetrics) -> RenderResult<()> {
    if metrics.total_bytes > metrics.limit_bytes
        || texture_size > metrics.limit_bytes.saturating_sub(metrics.total_bytes)
    {
        return Err(RenderError::upload(format!(
            "HDR texture ({} bytes) would exceed memory budget. Current: {} bytes, limit: {} bytes",
            texture_size, metrics.total_bytes, metrics.limit_bytes
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn align_copy_bytes_per_row(bytes_per_row: u32) -> u32 {
    crate::core::texture_layout::align_copy_row(bytes_per_row).expect("Texture row exceeds u32")
}

pub(super) fn validate_config(config: &HdrTextureConfig) -> RenderResult<()> {
    if config.width == 0 || config.height == 0 {
        return Err(RenderError::upload(
            "Texture dimensions must be > 0".to_string(),
        ));
    }

    const MAX_TEXTURE_SIZE: u32 = 16384;
    if config.width > MAX_TEXTURE_SIZE || config.height > MAX_TEXTURE_SIZE {
        return Err(RenderError::upload(format!(
            "Texture dimensions too large: {}x{}, maximum is {}x{}",
            config.width, config.height, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE
        )));
    }

    Ok(())
}

pub fn create_hdr_lut_1d(data: &[f32], width: u32, format: HdrFormat) -> RenderResult<HdrTexture> {
    let config = HdrTextureConfig {
        label: Some("hdr-lut-1d".to_string()),
        width,
        height: 1,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        generate_mipmaps: false,
    };

    match format {
        HdrFormat::Rgba32Float => create_texture_rgba32f(data, config),
        HdrFormat::Rgba16Float => {
            let f16_data: Vec<u16> = data
                .iter()
                .map(|&f| half::f16::from_f32(f).to_bits())
                .collect();
            create_texture_rgba16f(&f16_data, config)
        }
    }
}

pub fn create_hdr_environment_map(
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<HdrTexture> {
    let config = HdrTextureConfig {
        label: Some("hdr-environment-map".to_string()),
        width,
        height,
        format: HdrFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        generate_mipmaps: true,
    };

    create_texture_rgba32f(data, config)
}
