//! Image denoising for both raster and ray-traced outputs
//!
//! Provides multiple denoising strategies:
//! - Intel Open Image Denoise (OIDN) - high-quality ML-based denoiser
//! - Bilateral filter - fast edge-preserving spatial filter
//! - None - passthrough for baseline comparison
//!
//! The denoiser operates on CPU with pinned staging buffers to minimize overhead.

use std::sync::Once;
use log::{warn, info};

/// Denoiser selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiserType {
    None,
    Oidn,
    Bilateral,
}

impl DenoiserType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "oidn" => Some(Self::Oidn),
            "bilateral" => Some(Self::Bilateral),
            _ => None,
        }
    }
}

/// Denoiser configuration
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    pub denoiser: DenoiserType,
    pub strength: f32,  // 0.0 - 1.0, controls denoising intensity
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            denoiser: DenoiserType::Oidn,
            strength: 0.8,
        }
    }
}

static OIDN_WARNING: Once = Once::new();

/// Apply denoising to RGBA image data (in-place or copy)
///
/// # Arguments
/// * `rgba_data` - Input RGBA8 image data (width * height * 4 bytes)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Denoiser configuration
///
/// # Returns
/// Denoised RGBA8 data (same format as input)
pub fn denoise_rgba(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    config: &DenoiseConfig,
) -> Result<Vec<u8>, String> {
    match config.denoiser {
        DenoiserType::None => {
            // Passthrough - just copy data
            Ok(rgba_data.to_vec())
        }
        DenoiserType::Oidn => {
            // Try OIDN first, fallback to bilateral if unavailable
            match denoise_with_oidn(rgba_data, width, height, config.strength) {
                Ok(denoised) => Ok(denoised),
                Err(e) => {
                    OIDN_WARNING.call_once(|| {
                        warn!("OIDN unavailable: {}. Falling back to bilateral filter.", e);
                    });
                    denoise_with_bilateral(rgba_data, width, height, config.strength)
                }
            }
        }
        DenoiserType::Bilateral => {
            denoise_with_bilateral(rgba_data, width, height, config.strength)
        }
    }
}

/// Denoise using Intel Open Image Denoise (OIDN)
///
/// OIDN expects RGB32F input, so we convert RGBA8 -> RGB32F -> denoise -> RGBA8
fn denoise_with_oidn(
    _rgba_data: &[u8],
    _width: u32,
    _height: u32,
    _strength: f32,
) -> Result<Vec<u8>, String> {
    #[cfg(feature = "oidn")]
    {
        use oidn::{Device, RayTracing};
        
        let pixels = (_width * _height) as usize;
        
        // Convert RGBA8 to RGB32F for OIDN
        let mut rgb_f32 = vec![0.0f32; pixels * 3];
        for i in 0..pixels {
            let r = _rgba_data[i * 4] as f32 / 255.0;
            let g = _rgba_data[i * 4 + 1] as f32 / 255.0;
            let b = _rgba_data[i * 4 + 2] as f32 / 255.0;
            rgb_f32[i * 3] = r;
            rgb_f32[i * 3 + 1] = g;
            rgb_f32[i * 3 + 2] = b;
        }
        
        // Create OIDN device
        let device = Device::new();
        
        // Check if HDR denoising is supported (better for our use case)
        let filter_type = if device.get::<RayTracing>() {
            oidn::Format::Float3
        } else {
            oidn::Format::Float3
        };
        
        // Create RT denoiser filter
        let mut filter = oidn::RayTracing::new(&device);
        
        // Prepare output buffer
        let mut output = vec![0.0f32; pixels * 3];
        
        // Execute denoising
        filter
            .image_dimensions(_width as usize, _height as usize)
            .hdr(true)  // Enable HDR mode for better color preservation
            .srgb(false)  // We handle gamma correction separately
            .filter(&rgb_f32, &mut output)
            .map_err(|e| format!("OIDN filtering failed: {:?}", e))?;
        
        // Blend between original and denoised based on strength
        let inv_strength = 1.0 - _strength;
        for i in 0..(pixels * 3) {
            output[i] = output[i] * _strength + rgb_f32[i] * inv_strength;
        }
        
        // Convert RGB32F back to RGBA8
        let mut result = vec![0u8; pixels * 4];
        for i in 0..pixels {
            let r = (output[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (output[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (output[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
            let a = _rgba_data[i * 4 + 3];  // Preserve alpha
            result[i * 4] = r;
            result[i * 4 + 1] = g;
            result[i * 4 + 2] = b;
            result[i * 4 + 3] = a;
        }
        
        info!("Applied OIDN denoising (strength={:.2})", _strength);
        Ok(result)
    }
    
    #[cfg(not(feature = "oidn"))]
    {
        Err("OIDN feature not enabled at compile time".to_string())
    }
}

/// Denoise using bilateral filter (edge-preserving spatial filter)
///
/// Fast CPU implementation suitable for real-time or fallback use
fn denoise_with_bilateral(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    strength: f32,
) -> Result<Vec<u8>, String> {
    let w = width as usize;
    let h = height as usize;
    let pixels = w * h;
    
    // Bilateral filter parameters scaled by strength
    let spatial_sigma = 2.0 * strength + 0.5;  // 0.5 - 2.5 pixel radius
    let range_sigma = 0.3 * strength + 0.1;     // 0.1 - 0.4 intensity range
    
    let kernel_radius = (spatial_sigma * 2.0).ceil() as i32;
    let spatial_coef = -0.5 / (spatial_sigma * spatial_sigma);
    let range_coef = -0.5 / (range_sigma * range_sigma);
    
    let mut result = vec![0u8; pixels * 4];
    
    // Process each pixel
    for y in 0..h {
        for x in 0..w {
            let center_idx = (y * w + x) * 4;
            let center_r = rgba_data[center_idx] as f32 / 255.0;
            let center_g = rgba_data[center_idx + 1] as f32 / 255.0;
            let center_b = rgba_data[center_idx + 2] as f32 / 255.0;
            let center_a = rgba_data[center_idx + 3];
            
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut weight_sum = 0.0f32;
            
            // Iterate over kernel window
            for ky in -kernel_radius..=kernel_radius {
                for kx in -kernel_radius..=kernel_radius {
                    let ny = (y as i32 + ky).clamp(0, h as i32 - 1) as usize;
                    let nx = (x as i32 + kx).clamp(0, w as i32 - 1) as usize;
                    let neighbor_idx = (ny * w + nx) * 4;
                    
                    let neighbor_r = rgba_data[neighbor_idx] as f32 / 255.0;
                    let neighbor_g = rgba_data[neighbor_idx + 1] as f32 / 255.0;
                    let neighbor_b = rgba_data[neighbor_idx + 2] as f32 / 255.0;
                    
                    // Spatial weight (Gaussian based on distance)
                    let spatial_dist_sq = (kx * kx + ky * ky) as f32;
                    let spatial_weight = (spatial_coef * spatial_dist_sq).exp();
                    
                    // Range weight (Gaussian based on color difference)
                    let color_diff_r = neighbor_r - center_r;
                    let color_diff_g = neighbor_g - center_g;
                    let color_diff_b = neighbor_b - center_b;
                    let range_dist_sq = color_diff_r * color_diff_r 
                                      + color_diff_g * color_diff_g 
                                      + color_diff_b * color_diff_b;
                    let range_weight = (range_coef * range_dist_sq).exp();
                    
                    let weight = spatial_weight * range_weight;
                    
                    sum_r += neighbor_r * weight;
                    sum_g += neighbor_g * weight;
                    sum_b += neighbor_b * weight;
                    weight_sum += weight;
                }
            }
            
            // Normalize and blend with original based on strength
            let inv_strength = 1.0 - strength;
            let filtered_r = sum_r / weight_sum;
            let filtered_g = sum_g / weight_sum;
            let filtered_b = sum_b / weight_sum;
            
            let final_r = (filtered_r * strength + center_r * inv_strength).clamp(0.0, 1.0);
            let final_g = (filtered_g * strength + center_g * inv_strength).clamp(0.0, 1.0);
            let final_b = (filtered_b * strength + center_b * inv_strength).clamp(0.0, 1.0);
            
            result[center_idx] = (final_r * 255.0) as u8;
            result[center_idx + 1] = (final_g * 255.0) as u8;
            result[center_idx + 2] = (final_b * 255.0) as u8;
            result[center_idx + 3] = center_a;  // Preserve alpha
        }
    }
    
    info!("Applied bilateral filter denoising (strength={:.2}, spatial_sigma={:.2})", 
          strength, spatial_sigma);
    Ok(result)
}

/// Compute variance of a flat patch in the image (for testing)
///
/// Returns variance across RGB channels in a 32x32 patch at center
pub fn compute_patch_variance(rgba_data: &[u8], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;
    
    // Sample 32x32 patch at center
    let patch_size = 32.min(w).min(h);
    let start_x = (w - patch_size) / 2;
    let start_y = (h - patch_size) / 2;
    
    let mut values = Vec::new();
    
    for y in start_y..(start_y + patch_size) {
        for x in start_x..(start_x + patch_size) {
            let idx = (y * w + x) * 4;
            let r = rgba_data[idx] as f32 / 255.0;
            let g = rgba_data[idx + 1] as f32 / 255.0;
            let b = rgba_data[idx + 2] as f32 / 255.0;
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            values.push(luminance);
        }
    }
    
    // Compute variance
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    
    variance
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_denoise_none_passthrough() {
        let data = vec![128u8; 64 * 64 * 4];
        let config = DenoiseConfig {
            denoiser: DenoiserType::None,
            strength: 1.0,
        };
        
        let result = denoise_rgba(&data, 64, 64, &config).unwrap();
        assert_eq!(result, data);
    }
    
    #[test]
    fn test_bilateral_preserves_dimensions() {
        let data = vec![128u8; 64 * 64 * 4];
        let config = DenoiseConfig {
            denoiser: DenoiserType::Bilateral,
            strength: 0.8,
        };
        
        let result = denoise_rgba(&data, 64, 64, &config).unwrap();
        assert_eq!(result.len(), data.len());
    }
    
    #[test]
    fn test_bilateral_reduces_noise() {
        let mut rng_state = 12345u64;
        let mut simple_rand = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng_state / 65536) % 256) as u8
        };
        
        // Create noisy image (random values around 128)
        let mut data = vec![0u8; 64 * 64 * 4];
        for i in 0..(64 * 64) {
            let noise = simple_rand();
            data[i * 4] = noise;
            data[i * 4 + 1] = noise;
            data[i * 4 + 2] = noise;
            data[i * 4 + 3] = 255;
        }
        
        let original_variance = compute_patch_variance(&data, 64, 64);
        
        let config = DenoiseConfig {
            denoiser: DenoiserType::Bilateral,
            strength: 0.8,
        };
        
        let result = denoise_rgba(&data, 64, 64, &config).unwrap();
        let denoised_variance = compute_patch_variance(&result, 64, 64);
        
        // Denoising should reduce variance
        assert!(denoised_variance < original_variance * 0.8);
    }
    
    #[test]
    fn test_compute_patch_variance() {
        // Uniform image should have zero variance
        let uniform = vec![128u8; 64 * 64 * 4];
        let var = compute_patch_variance(&uniform, 64, 64);
        assert!(var < 0.001);
        
        // Checkerboard should have high variance
        let mut checkerboard = vec![0u8; 64 * 64 * 4];
        for y in 0..64 {
            for x in 0..64 {
                let val = if (x + y) % 2 == 0 { 0 } else { 255 };
                let idx = (y * 64 + x) * 4;
                checkerboard[idx] = val;
                checkerboard[idx + 1] = val;
                checkerboard[idx + 2] = val;
                checkerboard[idx + 3] = 255;
            }
        }
        let var = compute_patch_variance(&checkerboard, 64, 64);
        assert!(var > 0.1);
    }
}
