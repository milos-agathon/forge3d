//! GBuffer system for deferred rendering and screen-space effects
//!
//! Provides depth, normals, and material data for screen-space techniques

use crate::error::RenderResult;
use wgpu::*;

/// GBuffer configuration
#[derive(Debug, Clone)]
pub struct GBufferConfig {
    /// Width of GBuffer
    pub width: u32,
    /// Height of GBuffer
    pub height: u32,
    /// Format for depth buffer
    pub depth_format: TextureFormat,
    /// Format for normal buffer
    pub normal_format: TextureFormat,
    /// Format for material/albedo buffer
    pub material_format: TextureFormat,
    /// Whether to use half-precision formats
    pub use_half_precision: bool,
}

impl Default for GBufferConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            // Use a widely color-renderable single-channel float format to store view-space depth
            // R32Float as a color attachment may not be renderable on some backends (e.g., Metal)
            depth_format: TextureFormat::R16Float,
            normal_format: TextureFormat::Rgba16Float,
            material_format: TextureFormat::Rgba8Unorm,
            use_half_precision: true,
        }
    }
}

/// GBuffer textures for deferred rendering
pub struct GBuffer {
    /// Depth texture (view-space depth or world-space depth)
    pub depth_texture: Texture,
    pub depth_view: TextureView,
    
    /// Normal texture (view-space normals, encoded)
    pub normal_texture: Texture,
    pub normal_view: TextureView,
    
    /// Material/albedo texture
    pub material_texture: Texture,
    pub material_view: TextureView,
    
    /// Optional position reconstruction texture (if not reconstructing from depth)
    pub position_texture: Option<Texture>,
    pub position_view: Option<TextureView>,
    
    /// Configuration
    config: GBufferConfig,
}

impl GBuffer {
    /// Create new GBuffer
    pub fn new(device: &Device, config: GBufferConfig) -> RenderResult<Self> {
        // Create depth texture
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("gbuffer_depth"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.depth_format,
            usage: TextureUsages::TEXTURE_BINDING 
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());
        
        // Create normal texture
        let normal_texture = device.create_texture(&TextureDescriptor {
            label: Some("gbuffer_normal"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.normal_format,
            usage: TextureUsages::TEXTURE_BINDING 
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let normal_view = normal_texture.create_view(&TextureViewDescriptor::default());
        
        // Create material texture
        let material_texture = device.create_texture(&TextureDescriptor {
            label: Some("gbuffer_material"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.material_format,
            usage: TextureUsages::TEXTURE_BINDING 
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let material_view = material_texture.create_view(&TextureViewDescriptor::default());
        
        Ok(Self {
            depth_texture,
            depth_view,
            normal_texture,
            normal_view,
            material_texture,
            material_view,
            position_texture: None,
            position_view: None,
            config,
        })
    }
    
    /// Resize GBuffer
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> RenderResult<()> {
        self.config.width = width;
        self.config.height = height;
        
        // Recreate textures
        let new_gbuffer = Self::new(device, self.config.clone())?;
        *self = new_gbuffer;
        
        Ok(())
    }
    
    /// Get configuration
    pub fn config(&self) -> &GBufferConfig {
        &self.config
    }
    
    /// Get dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}
