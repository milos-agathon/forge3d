//! O3: Texture format detection and validation
//!
//! This module provides comprehensive texture format detection, validation,
//! and conversion utilities for compressed and uncompressed textures.

use std::collections::HashMap;
use wgpu::TextureFormat;

use super::texture_format_defs::all_format_definitions;

/// Comprehensive texture format information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFormatInfo {
    pub format: TextureFormat,
    pub is_compressed: bool,
    pub bytes_per_pixel: u32,
    pub block_size: u32,
    pub channels: u32,
    pub bit_depth: u32,
    pub supports_linear: bool,
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
            TextureUseCase::Albedo => true,
            TextureUseCase::Normal => {
                !self.is_compressed
                    || matches!(
                        self.format,
                        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm
                    )
            }
            TextureUseCase::Height => !self.is_srgb && self.supports_linear,
            TextureUseCase::HDR => matches!(
                self.format,
                TextureFormat::Bc6hRgbFloat
                    | TextureFormat::Bc6hRgbUfloat
                    | TextureFormat::Rgba16Float
                    | TextureFormat::Rgba32Float
                    | TextureFormat::Rg11b10Float
            ),
            TextureUseCase::UI => !self.is_compressed,
        }
    }
}

/// Texture use case for format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureUseCase {
    Albedo,
    Normal,
    Height,
    HDR,
    UI,
}

/// Compression quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionQuality {
    Fast,
    Normal,
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
        for info in all_format_definitions() {
            formats.insert(info.format, info);
        }
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
        let candidates = get_format_candidates(use_case)?;
        find_best_format(&candidates, self, device_features, quality, use_case)
    }

    /// Check if a format is supported by device
    pub fn is_format_supported(&self, format: TextureFormat, features: &wgpu::Features) -> bool {
        is_format_supported_by_device(format, features)
    }

    /// Get list of all supported formats for a device
    pub fn get_supported_formats(&self, device_features: &wgpu::Features) -> Vec<TextureFormat> {
        self.formats
            .keys()
            .filter(|&&format| self.is_format_supported(format, device_features))
            .cloned()
            .collect()
    }

    /// Calculate compression ratio compared to RGBA8
    pub fn calculate_compression_ratio(&self, format: TextureFormat, w: u32, h: u32) -> f32 {
        let rgba8_size = (w * h * 4) as f32;
        self.get_format_info(format)
            .map(|info| rgba8_size / info.calculate_size(w, h) as f32)
            .unwrap_or(1.0)
    }

    /// Get format family (BC, ETC2, ASTC, etc.)
    pub fn get_format_family(&self, format: TextureFormat) -> &'static str {
        get_format_family(format)
    }
}

fn get_format_candidates(use_case: TextureUseCase) -> Option<Vec<TextureFormat>> {
    Some(match use_case {
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
        TextureUseCase::Height => vec![TextureFormat::Bc4RUnorm, TextureFormat::EacR11Unorm],
        TextureUseCase::HDR => vec![TextureFormat::Bc6hRgbUfloat, TextureFormat::Bc6hRgbFloat],
        TextureUseCase::UI => return None,
    })
}

fn find_best_format(
    candidates: &[TextureFormat],
    registry: &TextureFormatRegistry,
    features: &wgpu::Features,
    quality: CompressionQuality,
    use_case: TextureUseCase,
) -> Option<TextureFormat> {
    for &format in candidates {
        if !is_format_supported_by_device(format, features) {
            continue;
        }
        let Some(info) = registry.get_format_info(format) else {
            continue;
        };
        if is_quality_acceptable(format, quality) && info.is_suitable_for_use(use_case) {
            return Some(format);
        }
    }
    None
}

fn is_quality_acceptable(format: TextureFormat, quality: CompressionQuality) -> bool {
    match quality {
        CompressionQuality::Fast => true,
        CompressionQuality::Normal => !matches!(
            format,
            TextureFormat::Bc1RgbaUnorm | TextureFormat::Etc2Rgb8Unorm
        ),
        CompressionQuality::High => matches!(
            format,
            TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc5RgUnorm
        ),
    }
}

fn is_format_supported_by_device(format: TextureFormat, features: &wgpu::Features) -> bool {
    use TextureFormat::*;
    match format {
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm
        | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm | Bc6hRgbFloat
        | Bc6hRgbUfloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => {
            features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        }
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb
        | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
        | EacRg11Snorm => features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
        Astc { .. } => features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
        _ => true,
    }
}

fn get_format_family(format: TextureFormat) -> &'static str {
    use TextureFormat::*;
    match format {
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm
        | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm | Bc6hRgbFloat
        | Bc6hRgbUfloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => "BC",
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb
        | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
        | EacRg11Snorm => "ETC2",
        Astc { .. } => "ASTC",
        _ => "Uncompressed",
    }
}

/// Global texture format registry
static GLOBAL_FORMAT_REGISTRY: std::sync::OnceLock<TextureFormatRegistry> =
    std::sync::OnceLock::new();

/// Get reference to global format registry
pub fn global_format_registry() -> &'static TextureFormatRegistry {
    GLOBAL_FORMAT_REGISTRY.get_or_init(TextureFormatRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_registry_creation() {
        let registry = TextureFormatRegistry::new();
        assert!(registry
            .get_format_info(TextureFormat::Rgba8Unorm)
            .is_some());
        assert!(registry
            .get_format_info(TextureFormat::Bc1RgbaUnorm)
            .is_some());
        assert!(registry
            .get_format_info(TextureFormat::Etc2Rgb8Unorm)
            .is_some());
    }

    #[test]
    fn test_compressed_size_calculation() {
        let registry = TextureFormatRegistry::new();
        let bc1_info = registry
            .get_format_info(TextureFormat::Bc1RgbaUnorm)
            .unwrap();
        assert_eq!(bc1_info.calculate_size(16, 16), 4 * 4 * 8);
        let rgba8_info = registry.get_format_info(TextureFormat::Rgba8Unorm).unwrap();
        assert_eq!(rgba8_info.calculate_size(16, 16), 16 * 16 * 4);
    }

    #[test]
    fn test_format_suitability() {
        let registry = TextureFormatRegistry::new();
        let bc5 = registry.get_format_info(TextureFormat::Bc5RgUnorm).unwrap();
        assert!(bc5.is_suitable_for_use(TextureUseCase::Normal));
        assert!(!bc5.is_suitable_for_use(TextureUseCase::UI));
    }

    #[test]
    fn test_compression_ratio() {
        let registry = TextureFormatRegistry::new();
        let bc1_ratio = registry.calculate_compression_ratio(TextureFormat::Bc1RgbaUnorm, 64, 64);
        assert!((bc1_ratio - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_format_family_detection() {
        let registry = TextureFormatRegistry::new();
        assert_eq!(
            registry.get_format_family(TextureFormat::Bc1RgbaUnorm),
            "BC"
        );
        assert_eq!(
            registry.get_format_family(TextureFormat::Etc2Rgb8Unorm),
            "ETC2"
        );
        assert_eq!(
            registry.get_format_family(TextureFormat::Rgba8Unorm),
            "Uncompressed"
        );
    }

    #[test]
    fn test_global_registry_access() {
        let registry = global_format_registry();
        assert!(registry
            .get_format_info(TextureFormat::Rgba8Unorm)
            .is_some());
    }
}
