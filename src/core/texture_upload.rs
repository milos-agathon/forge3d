//! Texture upload helpers for HDR and high-precision formats
//!
//! Provides utilities for creating and uploading RGBA16F and RGBA32F textures
//! with proper memory budget checking and alignment handling.

use super::error::{RenderError, RenderResult};
use super::gpu::ctx;
use crate::core::memory_tracker::global_tracker;
use std::num::NonZeroU32;

/// Configuration for HDR texture creation
#[derive(Debug, Clone)]
pub struct HdrTextureConfig {
    /// Texture label for debugging
    pub label: Option<String>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Texture format (RGBA16F or RGBA32F)
    pub format: HdrFormat,
    /// Texture usage flags
    pub usage: wgpu::TextureUsages,
    /// Whether to generate mipmaps
    pub generate_mipmaps: bool,
}

/// HDR texture format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrFormat {
    /// 16-bit floating point per component (half precision)
    Rgba16Float,
    /// 32-bit floating point per component (full precision)
    Rgba32Float,
}

impl HdrFormat {
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            HdrFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            HdrFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            HdrFormat::Rgba16Float => 8,  // 4 components × 2 bytes
            HdrFormat::Rgba32Float => 16, // 4 components × 4 bytes
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            HdrFormat::Rgba16Float => "Rgba16Float",
            HdrFormat::Rgba32Float => "Rgba32Float",
        }
    }
}

impl Default for HdrTextureConfig {
    fn default() -> Self {
        Self {
            label: None,
            width: 1,
            height: 1,
            format: HdrFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            generate_mipmaps: false,
        }
    }
}

/// Created HDR texture with associated resources
pub struct HdrTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: HdrFormat,
    pub width: u32,
    pub height: u32,
}

impl HdrTexture {
    /// Get pixel count
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Get texture size in bytes
    pub fn size_bytes(&self) -> usize {
        self.pixel_count() * self.format.bytes_per_pixel()
    }

    /// Create a sampler suitable for HDR textures
    pub fn create_sampler(&self, linear_filtering: bool) -> wgpu::Sampler {
        let g = ctx();

        let filter = if linear_filtering {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        };

        g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr-texture-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        })
    }
}

/// Create HDR texture from RGBA32F data
pub fn create_texture_rgba32f(data: &[f32], config: HdrTextureConfig) -> RenderResult<HdrTexture> {
    if config.format != HdrFormat::Rgba32Float {
        return Err(RenderError::upload(
            "create_texture_rgba32f requires Rgba32Float format".to_string(),
        ));
    }

    create_hdr_texture_internal(bytemuck::cast_slice(data), config)
}

/// Create HDR texture from RGBA16F data
pub fn create_texture_rgba16f(data: &[u16], config: HdrTextureConfig) -> RenderResult<HdrTexture> {
    if config.format != HdrFormat::Rgba16Float {
        return Err(RenderError::upload(
            "create_texture_rgba16f requires Rgba16Float format".to_string(),
        ));
    }

    create_hdr_texture_internal(bytemuck::cast_slice(data), config)
}

/// Create HDR texture from RGB32F data (add alpha channel)
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

    // Convert RGB to RGBA
    let mut rgba_data = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let base = i * 3;
        rgba_data.push(rgb_data[base]); // R
        rgba_data.push(rgb_data[base + 1]); // G
        rgba_data.push(rgb_data[base + 2]); // B
        rgba_data.push(alpha); // A
    }

    create_texture_rgba32f(&rgba_data, config)
}

/// Internal HDR texture creation
fn create_hdr_texture_internal(data: &[u8], config: HdrTextureConfig) -> RenderResult<HdrTexture> {
    validate_config(&config)?;

    let g = ctx();
    let format = config.format.to_wgpu();

    // Check memory budget
    let texture_size =
        (config.width as u64) * (config.height as u64) * (config.format.bytes_per_pixel() as u64);
    let tracker = global_tracker();

    // Check if we have enough budget for this texture
    let current_metrics = tracker.get_metrics();
    if texture_size > current_metrics.limit_bytes - current_metrics.total_bytes {
        return Err(RenderError::upload(format!(
            "HDR texture ({} bytes) would exceed memory budget. Current: {} bytes, limit: {} bytes",
            texture_size, current_metrics.total_bytes, current_metrics.limit_bytes
        )));
    }

    // Calculate mip levels
    let mip_level_count = if config.generate_mipmaps {
        (config.width.max(config.height) as f32).log2().floor() as u32 + 1
    } else {
        1
    };

    // Create texture
    let texture = g.device.create_texture(&wgpu::TextureDescriptor {
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
    });

    // Track texture allocation
    tracker.track_texture_allocation(config.width, config.height, format);

    // Upload data with proper alignment
    upload_texture_data(
        &g.device,
        &g.queue,
        &texture,
        data,
        config.width,
        config.height,
        config.format,
    )?;

    // Create view
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(HdrTexture {
        texture,
        view,
        format: config.format,
        width: config.width,
        height: config.height,
    })
}

/// Upload texture data with proper alignment
fn upload_texture_data(
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    data: &[u8],
    width: u32,
    height: u32,
    format: HdrFormat,
) -> RenderResult<()> {
    let bytes_per_pixel = format.bytes_per_pixel();
    let row_bytes = width as usize * bytes_per_pixel;
    let expected_size = height as usize * row_bytes;

    if data.len() != expected_size {
        return Err(RenderError::upload(format!(
            "Data size mismatch: expected {} bytes, got {}",
            expected_size,
            data.len()
        )));
    }

    // Calculate padded bytes per row for GPU alignment
    let padded_bytes_per_row = align_copy_bytes_per_row(row_bytes as u32);

    let image_data = if padded_bytes_per_row == row_bytes as u32 {
        // No padding needed, use data directly
        data
    } else {
        // Need to create padded data
        return Err(RenderError::upload(
            "Texture row padding not yet implemented for HDR formats".to_string(),
        ));
    };

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        image_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(padded_bytes_per_row).unwrap().into()),
            rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    Ok(())
}

/// Align bytes per row to GPU requirements
fn align_copy_bytes_per_row(bytes_per_row: u32) -> u32 {
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((bytes_per_row + alignment - 1) / alignment) * alignment
}

/// Validate HDR texture configuration
fn validate_config(config: &HdrTextureConfig) -> RenderResult<()> {
    if config.width == 0 || config.height == 0 {
        return Err(RenderError::upload(
            "Texture dimensions must be > 0".to_string(),
        ));
    }

    // Check maximum texture size (implementation-specific, but 16384 is common)
    const MAX_TEXTURE_SIZE: u32 = 16384;
    if config.width > MAX_TEXTURE_SIZE || config.height > MAX_TEXTURE_SIZE {
        return Err(RenderError::upload(format!(
            "Texture dimensions too large: {}x{}, maximum is {}x{}",
            config.width, config.height, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE
        )));
    }

    Ok(())
}

/// Utility functions for common HDR texture patterns

/// Create a 1D HDR LUT texture
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
            // Convert f32 to f16 (this is a simplified conversion)
            let f16_data: Vec<u16> = data
                .iter()
                .map(|&f| half::f16::from_f32(f).to_bits())
                .collect();
            create_texture_rgba16f(&f16_data, config)
        }
    }
}

/// Create HDR environment map from equirectangular data
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
        generate_mipmaps: true, // Useful for environment maps
    };

    create_texture_rgba32f(data, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr_format_bytes_per_pixel() {
        assert_eq!(HdrFormat::Rgba16Float.bytes_per_pixel(), 8);
        assert_eq!(HdrFormat::Rgba32Float.bytes_per_pixel(), 16);
    }

    #[test]
    fn test_align_copy_bytes_per_row() {
        assert_eq!(align_copy_bytes_per_row(100), 256); // Should align to 256
        assert_eq!(align_copy_bytes_per_row(256), 256); // Already aligned
        assert_eq!(align_copy_bytes_per_row(300), 512); // Next alignment boundary
    }

    #[test]
    fn test_validate_config() {
        let mut config = HdrTextureConfig::default();

        // Valid config should pass
        config.width = 512;
        config.height = 512;
        assert!(validate_config(&config).is_ok());

        // Zero dimensions should fail
        config.width = 0;
        assert!(validate_config(&config).is_err());

        config.width = 512;
        config.height = 0;
        assert!(validate_config(&config).is_err());

        // Too large dimensions should fail
        config.width = 20000;
        config.height = 20000;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_rgb_to_rgba_conversion() {
        let rgb_data = vec![1.0, 0.5, 0.25, 0.75, 1.0, 0.5]; // 2 RGB pixels
        let alpha = 0.8;

        // This would be tested with actual GPU context in integration tests
        // For now, just verify the logic would work
        let expected_rgba = vec![
            1.0, 0.5, 0.25, 0.8, // First pixel + alpha
            0.75, 1.0, 0.5, 0.8, // Second pixel + alpha
        ];

        // Simulate the conversion logic
        let pixel_count = 2;
        let mut rgba_data = Vec::with_capacity(pixel_count * 4);
        for i in 0..pixel_count {
            let base = i * 3;
            rgba_data.push(rgb_data[base]); // R
            rgba_data.push(rgb_data[base + 1]); // G
            rgba_data.push(rgb_data[base + 2]); // B
            rgba_data.push(alpha); // A
        }

        assert_eq!(rgba_data, expected_rgba);
    }
}

/// Create R32Float height texture from float32 data
pub fn create_r32f_height_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<(wgpu::Texture, wgpu::TextureView)> {
    let size = (width * height) as usize;
    if data.len() != size {
        return Err(RenderError::upload(format!(
            "Data length {} != {} (width*height)",
            data.len(),
            size
        )));
    }

    // Check memory budget (≤ 512 MiB)
    let texture_bytes = (width as u64) * (height as u64) * 4; // R32Float = 4 bytes per pixel
    let limit_bytes = 512 * 1024 * 1024; // 512 MiB
    if texture_bytes > limit_bytes {
        return Err(RenderError::upload(format!(
            "Height texture {}x{} ({} bytes) exceeds 512 MiB limit",
            width, height, texture_bytes
        )));
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("height_r32f"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Upload data with proper row alignment
    upload_r32f_data(queue, &texture, data, width, height)?;

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}

/// Upload R32F data to texture with proper row alignment
fn upload_r32f_data(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<()> {
    let row_bytes = width * 4; // 4 bytes per f32
    let padded_bpr = align_copy_bytes_per_row(row_bytes);

    if padded_bpr == row_bytes {
        // No padding needed, upload directly
        let bytes: &[u8] = bytemuck::cast_slice(data);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(row_bytes).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        // Need padding - create padded upload buffer
        let mut padded_data = vec![0u8; (padded_bpr * height) as usize];
        let input_data = bytemuck::cast_slice::<f32, u8>(data);

        for y in 0..height {
            let src_offset = (y * row_bytes) as usize;
            let dst_offset = (y * padded_bpr) as usize;
            let src_end = src_offset + row_bytes as usize;
            let dst_end = dst_offset + row_bytes as usize;

            padded_data[dst_offset..dst_end].copy_from_slice(&input_data[src_offset..src_end]);
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(())
}

pub fn create_r32f_height_texture_padded(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<(wgpu::Texture, wgpu::TextureView)> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(RenderError::upload(format!(
            "data length {} != {} (width*height)",
            data.len(),
            expected
        )));
    }

    let bytes_per_row = (width * 4) as usize;
    let padded_bpr = ((bytes_per_row + 255) / 256) * 256;
    let mut staged = vec![0u8; padded_bpr * (height as usize)];
    let src = bytemuck::cast_slice::<f32, u8>(data);

    for row in 0..height as usize {
        let src_off = row * bytes_per_row;
        let dst_off = row * padded_bpr;
        staged[dst_off..dst_off + bytes_per_row]
            .copy_from_slice(&src[src_off..src_off + bytes_per_row]);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("height_r32f"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &staged,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr as u32).unwrap().into()),
            rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}
