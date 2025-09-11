//! O3: Texture format detection and validation
//!
//! This module provides comprehensive texture format detection, validation,
//! and conversion utilities for compressed and uncompressed textures.

use wgpu::TextureFormat;
use std::collections::HashMap;

/// Comprehensive texture format information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFormatInfo {
    /// WGPU texture format
    pub format: TextureFormat,
    /// Whether this is a compressed format
    pub is_compressed: bool,
    /// Bytes per pixel (or block for compressed formats)
    pub bytes_per_pixel: u32,
    /// Block size for compressed formats (e.g., 4 for 4x4 blocks)
    pub block_size: u32,
    /// Number of color channels
    pub channels: u32,
    /// Bit depth per channel
    pub bit_depth: u32,
    /// Whether format supports linear filtering
    pub supports_linear: bool,
    /// Whether format is sRGB
    pub is_srgb: bool,
}

impl TextureFormatInfo {
    /// Calculate texture size in bytes
    pub fn calculate_size(&self, width: u32, height: u32) -> u64 {
        if self.is_compressed {
            let blocks_x = (width + self.block_size - 1) / self.block_size;
            let blocks_y = (height + self.block_size - 1) / self.block_size;
            (blocks_x as u64) * (blocks_y as u64) * (self.bytes_per_pixel as u64)
        } else {
            (width as u64) * (height as u64) * (self.bytes_per_pixel as u64)
        }
    }
    
    /// Check if format is suitable for a given use case
    pub fn is_suitable_for_use(&self, use_case: TextureUseCase) -> bool {
        match use_case {
            TextureUseCase::Albedo => true, // Most formats work for albedo
            TextureUseCase::Normal => !self.is_compressed || matches!(self.format,
                TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm),
            TextureUseCase::Height => !self.is_srgb && self.supports_linear,
            TextureUseCase::HDR => matches!(self.format,
                TextureFormat::Bc6hRgbFloat | TextureFormat::Bc6hRgbUfloat |
                TextureFormat::Rgba16Float | TextureFormat::Rgba32Float |
                TextureFormat::Rg11b10Float),
            TextureUseCase::UI => !self.is_compressed, // UI needs pixel-perfect access
        }
    }
}

/// Texture use case for format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureUseCase {
    /// Diffuse/albedo textures
    Albedo,
    /// Normal maps
    Normal,
    /// Height/displacement maps
    Height,
    /// HDR/tone mapped content
    HDR,
    /// UI/text rendering
    UI,
}

/// Compression quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionQuality {
    /// Fastest compression, lower quality
    Fast,
    /// Balanced compression and quality
    Normal,
    /// Highest quality, slower compression
    High,
}

/// Texture format registry with comprehensive format information
pub struct TextureFormatRegistry {
    formats: HashMap<TextureFormat, TextureFormatInfo>,
}

impl Default for TextureFormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureFormatRegistry {
    /// Create new registry with all supported formats
    pub fn new() -> Self {
        let mut formats = HashMap::new();
        
        // Uncompressed formats
        formats.insert(TextureFormat::R8Unorm, TextureFormatInfo {
            format: TextureFormat::R8Unorm,
            is_compressed: false,
            bytes_per_pixel: 1,
            block_size: 1,
            channels: 1,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Rg8Unorm, TextureFormatInfo {
            format: TextureFormat::Rg8Unorm,
            is_compressed: false,
            bytes_per_pixel: 2,
            block_size: 1,
            channels: 2,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Rgba8Unorm, TextureFormatInfo {
            format: TextureFormat::Rgba8Unorm,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Rgba8UnormSrgb, TextureFormatInfo {
            format: TextureFormat::Rgba8UnormSrgb,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: false,
            is_srgb: true,
        });
        
        formats.insert(TextureFormat::Bgra8Unorm, TextureFormatInfo {
            format: TextureFormat::Bgra8Unorm,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::R16Float, TextureFormatInfo {
            format: TextureFormat::R16Float,
            is_compressed: false,
            bytes_per_pixel: 2,
            block_size: 1,
            channels: 1,
            bit_depth: 16,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Rgba16Float, TextureFormatInfo {
            format: TextureFormat::Rgba16Float,
            is_compressed: false,
            bytes_per_pixel: 8,
            block_size: 1,
            channels: 4,
            bit_depth: 16,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::R32Float, TextureFormatInfo {
            format: TextureFormat::R32Float,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 1,
            bit_depth: 32,
            supports_linear: false, // Some backends don't support linear sampling on 32-bit float
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Rgba32Float, TextureFormatInfo {
            format: TextureFormat::Rgba32Float,
            is_compressed: false,
            bytes_per_pixel: 16,
            block_size: 1,
            channels: 4,
            bit_depth: 32,
            supports_linear: true,
            is_srgb: false,
        });
        
        // BC (DirectX) compressed formats
        formats.insert(TextureFormat::Bc1RgbaUnorm, TextureFormatInfo {
            format: TextureFormat::Bc1RgbaUnorm,
            is_compressed: true,
            bytes_per_pixel: 8, // 8 bytes per 4x4 block
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Bc1RgbaUnormSrgb, TextureFormatInfo {
            format: TextureFormat::Bc1RgbaUnormSrgb,
            is_compressed: true,
            bytes_per_pixel: 8,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: false,
            is_srgb: true,
        });
        
        formats.insert(TextureFormat::Bc3RgbaUnorm, TextureFormatInfo {
            format: TextureFormat::Bc3RgbaUnorm,
            is_compressed: true,
            bytes_per_pixel: 16, // 16 bytes per 4x4 block
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Bc3RgbaUnormSrgb, TextureFormatInfo {
            format: TextureFormat::Bc3RgbaUnormSrgb,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: false,
            is_srgb: true,
        });
        
        formats.insert(TextureFormat::Bc4RUnorm, TextureFormatInfo {
            format: TextureFormat::Bc4RUnorm,
            is_compressed: true,
            bytes_per_pixel: 8, // 8 bytes per 4x4 block
            block_size: 4,
            channels: 1,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Bc5RgUnorm, TextureFormatInfo {
            format: TextureFormat::Bc5RgUnorm,
            is_compressed: true,
            bytes_per_pixel: 16, // 16 bytes per 4x4 block
            block_size: 4,
            channels: 2,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Bc6hRgbUfloat, TextureFormatInfo {
            format: TextureFormat::Bc6hRgbUfloat,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 3,
            bit_depth: 16,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Bc7RgbaUnorm, TextureFormatInfo {
            format: TextureFormat::Bc7RgbaUnorm,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        // ETC2 compressed formats (mobile)
        formats.insert(TextureFormat::Etc2Rgb8Unorm, TextureFormatInfo {
            format: TextureFormat::Etc2Rgb8Unorm,
            is_compressed: true,
            bytes_per_pixel: 8, // 8 bytes per 4x4 block
            block_size: 4,
            channels: 3,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        formats.insert(TextureFormat::Etc2Rgba8Unorm, TextureFormatInfo {
            format: TextureFormat::Etc2Rgba8Unorm,
            is_compressed: true,
            bytes_per_pixel: 16, // 16 bytes per 4x4 block
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        });
        
        Self { formats }
    }
    
    /// Get format information
    pub fn get_format_info(&self, format: TextureFormat) -> Option<&TextureFormatInfo> {
        self.formats.get(&format)
    }
    
    /// Detect best compressed format for a given use case and device capabilities
    pub fn select_best_compressed_format(
        &self,
        use_case: TextureUseCase,
        device_features: &wgpu::Features,
        quality: CompressionQuality,
    ) -> Option<TextureFormat> {
        let candidates = match use_case {
            TextureUseCase::Albedo => vec![
                TextureFormat::Bc7RgbaUnorm,
                TextureFormat::Bc3RgbaUnorm,
                TextureFormat::Bc1RgbaUnorm,
                TextureFormat::Etc2Rgba8Unorm,
                TextureFormat::Etc2Rgb8Unorm,
            ],
            TextureUseCase::Normal => vec![
                TextureFormat::Bc5RgUnorm,
                TextureFormat::Bc3RgbaUnorm,
                TextureFormat::EacRg11Unorm,
            ],
            TextureUseCase::Height => vec![
                TextureFormat::Bc4RUnorm,
                TextureFormat::EacR11Unorm,
            ],
            TextureUseCase::HDR => vec![
                TextureFormat::Bc6hRgbUfloat,
                TextureFormat::Bc6hRgbFloat,
            ],
            TextureUseCase::UI => return None, // UI should use uncompressed
        };
        
        // Filter by device support and quality preference
        for format in candidates {
            if self.is_format_supported(format, device_features) {
                let format_info = self.get_format_info(format).unwrap();
                
                // Apply quality filtering
                let quality_ok = match quality {
                    CompressionQuality::Fast => true, // Accept any format for fast
                    CompressionQuality::Normal => !matches!(format, 
                        TextureFormat::Bc1RgbaUnorm | TextureFormat::Etc2Rgb8Unorm),
                    CompressionQuality::High => matches!(format,
                        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc6hRgbUfloat | 
                        TextureFormat::Bc5RgUnorm),
                };
                
                if quality_ok && format_info.is_suitable_for_use(use_case) {
                    return Some(format);
                }
            }
        }
        
        None
    }
    
    /// Check if a format is supported by device
    pub fn is_format_supported(&self, format: TextureFormat, device_features: &wgpu::Features) -> bool {
        match format {
            // BC formats require specific feature
            TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb |
            TextureFormat::Bc2RgbaUnorm | TextureFormat::Bc2RgbaUnormSrgb |
            TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb |
            TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm |
            TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm |
            TextureFormat::Bc6hRgbFloat | TextureFormat::Bc6hRgbUfloat |
            TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => {
                device_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
            },
            
            // ETC2 formats
            TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb |
            TextureFormat::Etc2Rgb8A1Unorm | TextureFormat::Etc2Rgb8A1UnormSrgb |
            TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb |
            TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm |
            TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => {
                device_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
            },
            
            // ASTC formats
            TextureFormat::Astc { .. } => {
                device_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
            },
            
            // Most uncompressed formats are always supported
            _ => true,
        }
    }
    
    /// Get list of all supported formats for a device
    pub fn get_supported_formats(&self, device_features: &wgpu::Features) -> Vec<TextureFormat> {
        self.formats.keys()
            .filter(|&&format| self.is_format_supported(format, device_features))
            .cloned()
            .collect()
    }
    
    /// Calculate compression ratio compared to RGBA8
    pub fn calculate_compression_ratio(&self, format: TextureFormat, width: u32, height: u32) -> f32 {
        let rgba8_size = (width * height * 4) as f32;
        
        if let Some(info) = self.get_format_info(format) {
            let compressed_size = info.calculate_size(width, height) as f32;
            rgba8_size / compressed_size
        } else {
            1.0 // Unknown format, assume no compression
        }
    }
    
    /// Get format family (BC, ETC2, ASTC, etc.)
    pub fn get_format_family(&self, format: TextureFormat) -> &'static str {
        match format {
            TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb |
            TextureFormat::Bc2RgbaUnorm | TextureFormat::Bc2RgbaUnormSrgb |
            TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb |
            TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm |
            TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm |
            TextureFormat::Bc6hRgbFloat | TextureFormat::Bc6hRgbUfloat |
            TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => "BC",
            
            TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb |
            TextureFormat::Etc2Rgb8A1Unorm | TextureFormat::Etc2Rgb8A1UnormSrgb |
            TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb |
            TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm |
            TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => "ETC2",
            
            TextureFormat::Astc { .. } => "ASTC",
            
            _ => "Uncompressed",
        }
    }
}

/// Global texture format registry
static GLOBAL_FORMAT_REGISTRY: std::sync::OnceLock<TextureFormatRegistry> = std::sync::OnceLock::new();

/// Get reference to global format registry
pub fn global_format_registry() -> &'static TextureFormatRegistry {
    GLOBAL_FORMAT_REGISTRY.get_or_init(|| TextureFormatRegistry::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_registry_creation() {
        let registry = TextureFormatRegistry::new();
        
        // Should have information for basic formats
        assert!(registry.get_format_info(TextureFormat::Rgba8Unorm).is_some());
        assert!(registry.get_format_info(TextureFormat::Bc1RgbaUnorm).is_some());
        assert!(registry.get_format_info(TextureFormat::Etc2Rgb8Unorm).is_some());
    }
    
    #[test]
    fn test_compressed_size_calculation() {
        let registry = TextureFormatRegistry::new();
        
        // Test BC1 format (8 bytes per 4x4 block)
        let bc1_info = registry.get_format_info(TextureFormat::Bc1RgbaUnorm).unwrap();
        let bc1_size = bc1_info.calculate_size(16, 16); // 4x4 blocks = 16 blocks total
        assert_eq!(bc1_size, 4 * 4 * 8); // 16 blocks * 8 bytes per block
        
        // Test uncompressed format
        let rgba8_info = registry.get_format_info(TextureFormat::Rgba8Unorm).unwrap();
        let rgba8_size = rgba8_info.calculate_size(16, 16);
        assert_eq!(rgba8_size, 16 * 16 * 4); // 16x16 * 4 bytes per pixel
    }
    
    #[test]
    fn test_format_suitability() {
        let registry = TextureFormatRegistry::new();
        
        let bc5_info = registry.get_format_info(TextureFormat::Bc5RgUnorm).unwrap();
        assert!(bc5_info.is_suitable_for_use(TextureUseCase::Normal));
        assert!(!bc5_info.is_suitable_for_use(TextureUseCase::UI));
        
        let rgba8_info = registry.get_format_info(TextureFormat::Rgba8Unorm).unwrap();
        assert!(rgba8_info.is_suitable_for_use(TextureUseCase::UI));
        assert!(rgba8_info.is_suitable_for_use(TextureUseCase::Albedo));
    }
    
    #[test]
    fn test_compression_ratio_calculation() {
        let registry = TextureFormatRegistry::new();
        
        // BC1 should have 4:1 compression ratio vs RGBA8
        let bc1_ratio = registry.calculate_compression_ratio(TextureFormat::Bc1RgbaUnorm, 64, 64);
        assert!((bc1_ratio - 4.0).abs() < 0.1);
        
        // Uncompressed should have 1:1 ratio
        let rgba_ratio = registry.calculate_compression_ratio(TextureFormat::Rgba8Unorm, 64, 64);
        assert!((rgba_ratio - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_format_family_detection() {
        let registry = TextureFormatRegistry::new();
        
        assert_eq!(registry.get_format_family(TextureFormat::Bc1RgbaUnorm), "BC");
        assert_eq!(registry.get_format_family(TextureFormat::Bc7RgbaUnorm), "BC");
        assert_eq!(registry.get_format_family(TextureFormat::Etc2Rgb8Unorm), "ETC2");
        assert_eq!(registry.get_format_family(TextureFormat::Rgba8Unorm), "Uncompressed");
    }
    
    #[test]
    fn test_global_registry_access() {
        let registry = global_format_registry();
        assert!(registry.get_format_info(TextureFormat::Rgba8Unorm).is_some());
    }
}