use wgpu::TextureFormat;

use crate::core::texture_format::{CompressionQuality, TextureUseCase};

/// Compressed image data with metadata
#[derive(Debug, Clone)]
pub struct CompressedImage {
    /// Raw compressed data
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Number of mip levels
    pub mip_levels: u32,
    /// Texture format
    pub format: TextureFormat,
    /// Whether this is sRGB format
    pub is_srgb: bool,
    /// Original file format
    pub source_format: String,
}

/// Compression statistics and metrics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original uncompressed size in bytes
    pub uncompressed_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Compression ratio (uncompressed / compressed)
    pub compression_ratio: f32,
    /// Time taken to compress in milliseconds
    pub compression_time_ms: f64,
    /// Quality metric (0.0-1.0, higher is better)
    pub quality_score: f32,
    /// Peak Signal-to-Noise Ratio in dB
    pub psnr_db: f32,
}

/// Texture compression options
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Target format (None = auto-select)
    pub target_format: Option<TextureFormat>,
    /// Quality level
    pub quality: CompressionQuality,
    /// Generate mip maps
    pub generate_mipmaps: bool,
    /// Use case for format selection
    pub use_case: TextureUseCase,
    /// Maximum texture size (power of 2)
    pub max_size: u32,
    /// Whether to enforce power-of-2 dimensions
    pub force_power_of_2: bool,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            target_format: None,
            quality: CompressionQuality::Normal,
            generate_mipmaps: true,
            use_case: TextureUseCase::Albedo,
            max_size: 4096,
            force_power_of_2: false,
        }
    }
}
