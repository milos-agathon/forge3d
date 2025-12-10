//! Mipmap generation utilities
//!
//! Provides CPU-based mipmap generation with box filtering and optional gamma correction.

use super::error::{RenderError, RenderResult};

/// Represents a single mipmap level
#[derive(Debug, Clone)]
pub struct Level {
    /// Width of this mip level
    pub width: u32,
    /// Height of this mip level  
    pub height: u32,
    /// RGBA32F pixel data (row-major, 4 components per pixel)
    pub data: Vec<f32>,
}

impl Level {
    /// Create a new mipmap level
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), (width * height * 4) as usize);
        Self {
            width,
            height,
            data,
        }
    }

    /// Get pixel count for this level
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Get data size in bytes
    pub fn data_size_bytes(&self) -> usize {
        self.data.len() * 4 // 4 bytes per f32
    }
}

/// Configuration for mipmap generation
#[derive(Debug, Clone)]
pub struct MipmapConfig {
    /// Use gamma-aware downsampling (sRGB -> linear -> sRGB)
    pub gamma_aware: bool,
    /// Gamma value for correction (typically 2.2)
    pub gamma: f32,
}

impl Default for MipmapConfig {
    fn default() -> Self {
        Self {
            gamma_aware: false,
            gamma: 2.2,
        }
    }
}

/// Build a complete mipmap chain using CPU box filtering
///
/// Parameters:
/// - `data`: Input RGBA32F pixel data (row-major, 4 components per pixel)
/// - `width`: Width of the base level
/// - `height`: Height of the base level
/// - `config`: Mipmap generation configuration
///
/// Returns a Vec of mipmap levels, where level 0 is the original image
pub fn build_mip_chain_rgba32f(
    data: &[f32],
    width: u32,
    height: u32,
    config: &MipmapConfig,
) -> RenderResult<Vec<Level>> {
    if width == 0 || height == 0 {
        return Err(RenderError::upload(
            "width and height must be > 0".to_string(),
        ));
    }

    let expected_len = (width * height * 4) as usize;
    if data.len() != expected_len {
        return Err(RenderError::upload(format!(
            "data length mismatch: expected {} ({}x{}x4), got {}",
            expected_len,
            width,
            height,
            data.len()
        )));
    }

    let mut levels = Vec::new();
    let mut current_data = data.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    // Add the base level
    levels.push(Level::new(
        current_width,
        current_height,
        current_data.clone(),
    ));

    // Generate mip levels until we reach 1x1
    while current_width > 1 || current_height > 1 {
        let next_width = (current_width / 2).max(1);
        let next_height = (current_height / 2).max(1);

        let next_data = downsample_box_filter(
            &current_data,
            current_width,
            current_height,
            next_width,
            next_height,
            config,
        )?;

        levels.push(Level::new(next_width, next_height, next_data.clone()));

        current_data = next_data;
        current_width = next_width;
        current_height = next_height;
    }

    Ok(levels)
}

/// Convenience function using default config
pub fn build_mip_chain_rgba32f_default(
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<Vec<Level>> {
    build_mip_chain_rgba32f(data, width, height, &MipmapConfig::default())
}

/// Downsample using box filtering
fn downsample_box_filter(
    src_data: &[f32],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    config: &MipmapConfig,
) -> RenderResult<Vec<f32>> {
    let mut dst_data = vec![0.0f32; (dst_width * dst_height * 4) as usize];

    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // Calculate source region
            let src_x_start = (dst_x as f32 * x_ratio) as u32;
            let src_y_start = (dst_y as f32 * y_ratio) as u32;
            let src_x_end = ((dst_x + 1) as f32 * x_ratio).ceil() as u32;
            let src_y_end = ((dst_y + 1) as f32 * y_ratio).ceil() as u32;

            let src_x_end = src_x_end.min(src_width);
            let src_y_end = src_y_end.min(src_height);

            // Accumulate samples in the box
            let mut rgba_sum = [0.0f32; 4];
            let mut sample_count = 0;

            for src_y in src_y_start..src_y_end {
                for src_x in src_x_start..src_x_end {
                    let src_idx = ((src_y * src_width + src_x) * 4) as usize;

                    if src_idx + 3 < src_data.len() {
                        let mut rgba = [
                            src_data[src_idx],
                            src_data[src_idx + 1],
                            src_data[src_idx + 2],
                            src_data[src_idx + 3],
                        ];

                        // Apply gamma correction if enabled
                        if config.gamma_aware {
                            // Convert sRGB to linear for RGB channels (leave alpha linear)
                            for i in 0..3 {
                                rgba[i] = srgb_to_linear(rgba[i]);
                            }
                        }

                        rgba_sum[0] += rgba[0];
                        rgba_sum[1] += rgba[1];
                        rgba_sum[2] += rgba[2];
                        rgba_sum[3] += rgba[3];
                        sample_count += 1;
                    }
                }
            }

            // Average the samples
            if sample_count > 0 {
                let inv_count = 1.0 / sample_count as f32;
                rgba_sum[0] *= inv_count;
                rgba_sum[1] *= inv_count;
                rgba_sum[2] *= inv_count;
                rgba_sum[3] *= inv_count;

                // Apply inverse gamma correction if enabled
                if config.gamma_aware {
                    // Convert linear back to sRGB for RGB channels
                    for i in 0..3 {
                        rgba_sum[i] = linear_to_srgb(rgba_sum[i]);
                    }
                }
            }

            // Store the result
            let dst_idx = ((dst_y * dst_width + dst_x) * 4) as usize;
            dst_data[dst_idx] = rgba_sum[0];
            dst_data[dst_idx + 1] = rgba_sum[1];
            dst_data[dst_idx + 2] = rgba_sum[2];
            dst_data[dst_idx + 3] = rgba_sum[3];
        }
    }

    Ok(dst_data)
}

/// Convert sRGB to linear space
#[inline]
fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear to sRGB space
#[inline]
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Calculate the number of mip levels for given dimensions
pub fn calculate_mip_levels(width: u32, height: u32) -> u32 {
    if width == 0 || height == 0 {
        return 0;
    }
    let max_dim = width.max(height);
    (max_dim as f32).log2().ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_mip_levels() {
        assert_eq!(calculate_mip_levels(1, 1), 1);
        assert_eq!(calculate_mip_levels(2, 2), 2);
        assert_eq!(calculate_mip_levels(4, 4), 3);
        assert_eq!(calculate_mip_levels(8, 8), 4);
        assert_eq!(calculate_mip_levels(256, 256), 9);
        assert_eq!(calculate_mip_levels(256, 128), 9);
        assert_eq!(calculate_mip_levels(128, 256), 9);
    }

    #[test]
    fn test_build_mip_chain_small() {
        // Create a simple 2x2 RGBA image (red, green, blue, white)
        let data = vec![
            1.0, 0.0, 0.0, 1.0, // red
            0.0, 1.0, 0.0, 1.0, // green
            0.0, 0.0, 1.0, 1.0, // blue
            1.0, 1.0, 1.0, 1.0, // white
        ];

        let levels = build_mip_chain_rgba32f_default(&data, 2, 2).unwrap();
        assert_eq!(levels.len(), 2); // 2x2 -> 1x1

        // Check base level
        assert_eq!(levels[0].width, 2);
        assert_eq!(levels[0].height, 2);
        assert_eq!(levels[0].data.len(), 16); // 2*2*4

        // Check 1x1 level
        assert_eq!(levels[1].width, 1);
        assert_eq!(levels[1].height, 1);
        assert_eq!(levels[1].data.len(), 4); // 1*1*4

        // The 1x1 level should be the average of all pixels
        let avg_r = (1.0 + 0.0 + 0.0 + 1.0) / 4.0;
        let avg_g = (0.0 + 1.0 + 0.0 + 1.0) / 4.0;
        let avg_b = (0.0 + 0.0 + 1.0 + 1.0) / 4.0;
        let avg_a = 1.0; // All alpha values are 1.0

        assert!((levels[1].data[0] - avg_r).abs() < 1e-6);
        assert!((levels[1].data[1] - avg_g).abs() < 1e-6);
        assert!((levels[1].data[2] - avg_b).abs() < 1e-6);
        assert!((levels[1].data[3] - avg_a).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_aware_config() {
        let config = MipmapConfig {
            gamma_aware: true,
            gamma: 2.2,
        };

        // Test with a simple gradient
        let data = vec![
            0.0, 0.0, 0.0, 1.0, // black
            1.0, 1.0, 1.0, 1.0, // white
        ];

        let levels = build_mip_chain_rgba32f(&data, 2, 1, &config).unwrap();
        assert_eq!(levels.len(), 2); // 2x1 -> 1x1

        // The gamma-aware average should be different from linear average
        let linear_avg = 0.5;
        let gamma_aware_result = levels[1].data[0]; // R channel of 1x1 result

        // With gamma correction, the result should be darker than linear average
        // because we convert to linear space, average, then convert back
        assert!(gamma_aware_result < linear_avg);
    }

    #[test]
    fn test_error_cases() {
        // Test empty dimensions
        let data = vec![];
        assert!(build_mip_chain_rgba32f_default(&data, 0, 0).is_err());
        assert!(build_mip_chain_rgba32f_default(&data, 1, 0).is_err());
        assert!(build_mip_chain_rgba32f_default(&data, 0, 1).is_err());

        // Test mismatched data length
        let data = vec![1.0; 8]; // Only 2 pixels worth of data
        assert!(build_mip_chain_rgba32f_default(&data, 2, 2).is_err()); // Expects 16 values
    }
}
