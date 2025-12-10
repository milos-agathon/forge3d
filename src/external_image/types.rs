//! External image import types and configuration.

/// Configuration for external image import operations.
#[derive(Debug, Clone)]
pub struct ImageImportConfig {
    /// Target texture format (always RGBA8UnormSrgb for WebGPU parity).
    pub target_format: wgpu::TextureFormat,
    /// Texture usage flags.
    pub usage: wgpu::TextureUsages,
    /// Whether to generate mipmaps.
    pub generate_mipmaps: bool,
    /// Texture label for debugging.
    pub label: Option<String>,
    /// Maximum allowed texture dimension (for safety).
    pub max_dimension: u32,
    /// Whether to premultiply alpha.
    pub premultiply_alpha: bool,
}

impl Default for ImageImportConfig {
    fn default() -> Self {
        Self {
            target_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            generate_mipmaps: false,
            label: None,
            max_dimension: 8192,
            premultiply_alpha: false,
        }
    }
}

/// Information about an imported texture.
#[derive(Debug)]
pub struct ImportedTextureInfo {
    /// The created WGPU texture.
    pub texture: wgpu::Texture,
    /// Texture view for binding.
    pub view: wgpu::TextureView,
    /// Original image dimensions.
    pub width: u32,
    pub height: u32,
    /// Detected source format.
    pub source_format: ImageSourceFormat,
    /// Final texture format.
    pub texture_format: wgpu::TextureFormat,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// Detected source image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSourceFormat {
    /// PNG with RGBA channels.
    PngRgba,
    /// PNG with RGB channels (converted to RGBA).
    PngRgb,
    /// PNG grayscale (converted to RGBA).
    PngGrayscale,
    /// JPEG RGB (converted to RGBA).
    JpegRgb,
}

impl ImageSourceFormat {
    /// Get human-readable format name.
    pub fn name(self) -> &'static str {
        match self {
            ImageSourceFormat::PngRgba => "PNG RGBA",
            ImageSourceFormat::PngRgb => "PNG RGB",
            ImageSourceFormat::PngGrayscale => "PNG Grayscale",
            ImageSourceFormat::JpegRgb => "JPEG RGB",
        }
    }

    /// Get number of channels in source format.
    pub fn channels(self) -> u32 {
        match self {
            ImageSourceFormat::PngRgba => 4,
            ImageSourceFormat::PngRgb => 3,
            ImageSourceFormat::PngGrayscale => 1,
            ImageSourceFormat::JpegRgb => 3,
        }
    }
}
