//! GBuffer system for deferred rendering and screen-space effects
//!
//! Provides depth, normals, and material data for screen-space techniques
//! such as SSAO/GTAO, SSGI, and SSR. The default configuration encodes:
//!   - `depth_texture`   : `R32Float` linear view-space depth (>0 for geometry)
//!   - `normal_texture`  : `Rgba16Float` with view-space normal in `xyz` and
//!                         perceptual roughness in `w` (both mapped to [0,1])
//!   - `material_texture`: `Rgba8Unorm` with diffuse/base color in `rgb` and
//!                         metallic factor in `a` (both in [0,1])

use super::error::RenderResult;
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
            // R32Float supports both RENDER_ATTACHMENT and STORAGE_BINDING (needed for HZB mip generation)
            // R16Float doesn't support STORAGE_BINDING on most platforms
            depth_format: TextureFormat::R32Float,
            normal_format: TextureFormat::Rgba16Float,
            material_format: TextureFormat::Rgba8Unorm,
            use_half_precision: true,
        }
    }
}

/// GBuffer textures for deferred rendering
pub struct GBuffer {
    /// Depth texture (linear view-space depth; cleared to 0.0 for background)
    pub depth_texture: Texture,
    pub depth_view: TextureView,

    /// Normal texture (view-space normals encoded into [0,1], plus roughness in alpha)
    pub normal_texture: Texture,
    pub normal_view: TextureView,

    /// Material/albedo texture (diffuse/base color in RGB, metallic in alpha)
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
        // Create depth texture (single mip; HZB is built into a separate texture)
        let depth_mips = 1u32;
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("gbuffer_depth"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: depth_mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.depth_format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        // Render attachments must have exactly one mip level
        let depth_view = depth_texture.create_view(&TextureViewDescriptor {
            label: Some("gbuffer_depth_mip0"),
            format: Some(config.depth_format),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

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
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC,
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
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC,
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
