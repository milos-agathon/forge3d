use crate::core::error::RenderError;
use bytemuck::{Pod, Zeroable};
use once_cell::sync::Lazy;
use std::sync::Mutex;

// Global configuration for point rendering (shape mode & LOD threshold)
static GLOBAL_POINT_CONFIG: Lazy<Mutex<(u32, f32)>> = Lazy::new(|| Mutex::new((0u32, 1.0f32)));

pub fn set_global_shape_mode(mode: u32) {
    let mut cfg = GLOBAL_POINT_CONFIG.lock().expect("point cfg poisoned");
    cfg.0 = mode;
}

pub fn set_global_lod_threshold(threshold: f32) {
    let mut cfg = GLOBAL_POINT_CONFIG.lock().expect("point cfg poisoned");
    cfg.1 = threshold.max(0.0);
}

pub fn get_global_config() -> (u32, f32) {
    let cfg = GLOBAL_POINT_CONFIG.lock().expect("point cfg poisoned");
    *cfg
}

/// Point rendering uniforms with H20,H21,H22 enhancements
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointUniform {
    pub transform: [[f32; 4]; 4],   // View-projection matrix
    pub viewport_size: [f32; 2],    // Viewport dimensions for size calculation
    pub pixel_scale: f32,           // Pixels per world unit
    pub debug_mode: u32,            // H20: Debug rendering flags
    pub atlas_size: [f32; 2],       // H21: Texture atlas dimensions
    pub enable_clip_w_scaling: u32, // H22: Enable clip.w aware sizing
    pub _pad0: f32,                 // Align depth_range to 8-byte boundary (std140/std430)
    pub depth_range: [f32; 2],      // H22: Near/far planes for clip.w scaling
    pub shape_mode: u32, // H2: shape/material mode (0=circle,4=texture,5=sphere impostor)
    pub lod_threshold: f32, // H2: pixel-size threshold for LOD
}

/// H20: Debug rendering mode flags
#[derive(Debug, Clone, Copy)]
pub struct DebugFlags {
    pub show_bounds: bool,    // Show point bounding boxes
    pub show_centers: bool,   // Show point centers as dots
    pub color_by_depth: bool, // Color points by depth
    pub show_normals: bool,   // Show surface normals (if available)
}

impl Default for DebugFlags {
    fn default() -> Self {
        Self {
            show_bounds: false,
            show_centers: false,
            color_by_depth: false,
            show_normals: false,
        }
    }
}

impl DebugFlags {
    /// Convert debug flags to u32 bitfield for shader
    pub fn to_bitfield(&self) -> u32 {
        let mut flags = 0u32;
        if self.show_bounds {
            flags |= 1 << 0;
        }
        if self.show_centers {
            flags |= 1 << 1;
        }
        if self.color_by_depth {
            flags |= 1 << 2;
        }
        if self.show_normals {
            flags |= 1 << 3;
        }
        flags
    }
}

/// H21: Texture atlas configuration
#[derive(Debug)]
pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,     // Size of each tile in pixels
    pub tiles_per_row: u32, // Number of tiles per row
}

impl TextureAtlas {
    /// Create a new texture atlas
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        tile_size: u32,
        data: &[u8],
    ) -> Result<Self, RenderError> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Point.TextureAtlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4), // RGBA8
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vf.Vector.Point.AtlasSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let tiles_per_row = width / tile_size;

        Ok(Self {
            texture,
            view,
            sampler,
            width,
            height,
            tile_size,
            tiles_per_row,
        })
    }

    /// Get UV coordinates for a tile index
    pub fn get_tile_uv(&self, tile_index: u32) -> ([f32; 2], [f32; 2]) {
        let tile_x = tile_index % self.tiles_per_row;
        let tile_y = tile_index / self.tiles_per_row;

        let u_size = self.tile_size as f32 / self.width as f32;
        let v_size = self.tile_size as f32 / self.height as f32;

        let u_min = tile_x as f32 * u_size;
        let v_min = tile_y as f32 * v_size;
        let u_max = u_min + u_size;
        let v_max = v_min + v_size;

        ([u_min, v_min], [u_max, v_max])
    }
}

/// Point shape types for different rendering modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointShape {
    /// Simple circle (default)
    Circle = 0,
    /// Square/rectangle
    Square = 1,
    /// Diamond shape
    Diamond = 2,
    /// Triangle pointing up
    Triangle = 3,
    /// Custom texture from atlas (H21)
    Texture = 4,
}
