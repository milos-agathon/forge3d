// src/path_tracing/oidn_runner.rs
// OIDN denoising with tiled mode for memory-bounded operation

use crate::post::denoise::{DenoiseConfig, DenoiserType};

/// OIDN execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OidnMode {
    /// Denoise the entire image at once (requires full-frame buffer)
    Final,
    /// Denoise per tile with overlap and stitching (bounded memory)
    Tiled,
    /// No denoising
    Off,
}

impl OidnMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "final" => Some(OidnMode::Final),
            "tiled" => Some(OidnMode::Tiled),
            "off" | "none" => Some(OidnMode::Off),
            _ => None,
        }
    }
}

impl Default for OidnMode {
    fn default() -> Self {
        OidnMode::Final
    }
}

/// Configuration for tiled OIDN processing
#[derive(Debug, Clone)]
pub struct TiledOidnConfig {
    pub mode: OidnMode,
    pub overlap: u32,  // Pixels of overlap for seamless stitching (default 32)
    pub strength: f32,
}

impl Default for TiledOidnConfig {
    fn default() -> Self {
        Self {
            mode: OidnMode::Final,
            overlap: 32,
            strength: 0.8,
        }
    }
}

/// Apply OIDN denoising with tiled mode support
pub fn denoise_tiled(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    config: &TiledOidnConfig,
) -> Result<Vec<u8>, String> {
    match config.mode {
        OidnMode::Off => Ok(rgba_data.to_vec()),
        OidnMode::Final => {
            // Single-pass denoising on entire image
            let denoise_config = DenoiseConfig {
                denoiser: DenoiserType::Oidn,
                strength: config.strength,
            };
            crate::post::denoise::denoise_rgba(rgba_data, width, height, &denoise_config)
        }
        OidnMode::Tiled => {
            denoise_tiled_with_overlap(rgba_data, width, height, tile_width, tile_height, config)
        }
    }
}

/// Denoise using tiled approach with overlap for seamless stitching
fn denoise_tiled_with_overlap(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    config: &TiledOidnConfig,
) -> Result<Vec<u8>, String> {
    let overlap = config.overlap;
    let denoise_config = DenoiseConfig {
        denoiser: DenoiserType::Oidn,
        strength: config.strength,
    };
    
    // Output buffer
    let mut output = vec![0u8; (width * height * 4) as usize];
    
    // Process tiles with overlap
    let tiles_x = ((width + tile_width - 1) / tile_width) as usize;
    let tiles_y = ((height + tile_height - 1) / tile_height) as usize;
    
    eprintln!("[OidnTiled] Processing {}x{} tiles with {}px overlap", tiles_x, tiles_y, overlap);
    
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            // Calculate tile bounds with overlap
            let tile_x_start = if tx > 0 {
                (tx as u32 * tile_width).saturating_sub(overlap)
            } else {
                0
            };
            let tile_y_start = if ty > 0 {
                (ty as u32 * tile_height).saturating_sub(overlap)
            } else {
                0
            };
            
            let tile_x_end = ((tx as u32 + 1) * tile_width + overlap).min(width);
            let tile_y_end = ((ty as u32 + 1) * tile_height + overlap).min(height);
            
            let tile_w = tile_x_end - tile_x_start;
            let tile_h = tile_y_end - tile_y_start;
            
            // Extract tile data with overlap
            let mut tile_data = vec![0u8; (tile_w * tile_h * 4) as usize];
            for y in 0..tile_h {
                let src_y = tile_y_start + y;
                let src_offset = (src_y * width + tile_x_start) as usize * 4;
                let dst_offset = (y * tile_w) as usize * 4;
                let row_bytes = (tile_w * 4) as usize;
                
                tile_data[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&rgba_data[src_offset..src_offset + row_bytes]);
            }
            
            // Denoise tile
            let denoised_tile = crate::post::denoise::denoise_rgba(
                &tile_data, tile_w, tile_h, &denoise_config
            )?;
            
            // Calculate center crop region (excluding overlap margins)
            let crop_left = if tx > 0 { overlap } else { 0 };
            let crop_top = if ty > 0 { overlap } else { 0 };
            let crop_right = if tx < tiles_x - 1 { overlap } else { 0 };
            let crop_bottom = if ty < tiles_y - 1 { overlap } else { 0 };
            
            let crop_w = tile_w - crop_left - crop_right;
            let crop_h = tile_h - crop_top - crop_bottom;
            
            // Copy center crop to output
            for y in 0..crop_h {
                let src_offset = ((crop_top + y) * tile_w + crop_left) as usize * 4;
                let dst_x = tile_x_start + crop_left;
                let dst_y = tile_y_start + crop_top + y;
                let dst_offset = (dst_y * width + dst_x) as usize * 4;
                let row_bytes = (crop_w * 4) as usize;
                
                output[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&denoised_tile[src_offset..src_offset + row_bytes]);
            }
        }
    }
    
    eprintln!("[OidnTiled] Denoising complete");
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_oidn_mode_from_str() {
        assert_eq!(OidnMode::from_str("final"), Some(OidnMode::Final));
        assert_eq!(OidnMode::from_str("tiled"), Some(OidnMode::Tiled));
        assert_eq!(OidnMode::from_str("off"), Some(OidnMode::Off));
        assert_eq!(OidnMode::from_str("none"), Some(OidnMode::Off));
        assert_eq!(OidnMode::from_str("invalid"), None);
    }
    
    #[test]
    fn test_denoise_off_passthrough() {
        let data = vec![128u8; 64 * 64 * 4];
        let config = TiledOidnConfig {
            mode: OidnMode::Off,
            overlap: 32,
            strength: 0.8,
        };
        
        let result = denoise_tiled(&data, 64, 64, 32, 32, &config).unwrap();
        assert_eq!(result, data);
    }
}
