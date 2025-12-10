use std::collections::HashMap;
use wgpu::{TextureFormat, TextureUsages};

/// Post-processing effect configuration
#[derive(Debug, Clone)]
pub struct PostFxConfig {
    /// Effect name/identifier
    pub name: String,
    /// Whether the effect is enabled
    pub enabled: bool,
    /// Effect-specific parameters
    pub parameters: HashMap<String, f32>,
    /// Priority for effect ordering (higher = later)
    pub priority: i32,
    /// Whether this effect needs temporal data
    pub temporal: bool,
    /// Number of ping-pong buffers needed
    pub ping_pong_count: u32,
}

impl Default for PostFxConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            parameters: HashMap::new(),
            priority: 0,
            temporal: false,
            ping_pong_count: 2,
        }
    }
}

/// Resource descriptor for post-processing textures
#[derive(Debug, Clone)]
pub struct PostFxResourceDesc {
    /// Resource width (0 = match input)
    pub width: u32,
    /// Resource height (0 = match input)  
    pub height: u32,
    /// Texture format
    pub format: TextureFormat,
    /// Usage flags
    pub usage: TextureUsages,
    /// Mip level count
    pub mip_count: u32,
    /// Sample count for MSAA
    pub sample_count: u32,
}

impl Default for PostFxResourceDesc {
    fn default() -> Self {
        Self {
            width: 0,  // Match input
            height: 0, // Match input
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            mip_count: 1,
            sample_count: 1,
        }
    }
}
