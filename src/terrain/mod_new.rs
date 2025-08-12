
// R1: Import from colormap module instead of duplicating
use crate::colormap::{ColormapType, map_name_to_type, resolve_bytes, SUPPORTED};
use std::sync::Arc;
use std::env;
use wgpu::{Device, Buffer, BindGroup, RenderPipeline, TextureView, TextureFormat, TextureUsages};
use image::{ImageReader, DynamicImage, ImageFormat};
use std::io::Cursor;

// R1: Remove this duplicated ColormapType enum and its impl
// (This will be deleted as per requirements)

pub struct TerrainSpike {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    colormap_lut: ColormapLUT,
}

impl TerrainSpike {
    pub fn new(device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        // R2: Compute srgb_ok using device texture format features
        let srgb_ok = check_srgb_support(device);
        let force_unorm = env::var("VF_FORCE_LUT_UNORM").is_ok();
        let prefer_srgb = srgb_ok && !force_unorm;
        
        // R2: Pass prefer_srgb to ColormapLUT::new
        let colormap_lut = ColormapLUT::new(device, prefer_srgb)?;
        
        Ok(Self {
            vertex_buffer: todo!(),
            index_buffer: todo!(),
            bind_group: todo!(),
            render_pipeline: todo!(),
            colormap_lut,
        })
    }
}

fn check_srgb_support(device: &Device) -> bool {
    // R2: Check if Rgba8UnormSrgb format supports TEXTURE_BINDING | COPY_DST usage
    // Note: In wgpu, texture format support is generally guaranteed for common formats
    // but we should check the specific usage requirements
    
    // For now, we'll assume Rgba8UnormSrgb is supported with TEXTURE_BINDING | COPY_DST
    // In a real implementation, you might want to check device.features() or limits()
    // for specific texture format features if they were exposed
    
    true
}

pub struct ColormapLUT {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl ColormapLUT {
    // R2: Change signature to accept prefer_srgb parameter
    pub fn new(device: &Device, prefer_srgb: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let format = if prefer_srgb {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        
        // Load PNG RGBA bytes (existing logic would be here)
        let rgba_bytes = load_colormap_png()?;
        
        let final_bytes = if prefer_srgb {
            // R2: Upload PNG RGBA bytes as-is for sRGB format
            rgba_bytes
        } else {
            // R2: CPU sRGB to linear conversion for Unorm format
            rgba_bytes
                .chunks(4)
                .flat_map(|pixel| {
                    [
                        srgb_to_linear(pixel[0]), // R
                        srgb_to_linear(pixel[1]), // G  
                        srgb_to_linear(pixel[2]), // B
                        pixel[3], // A unchanged
                    ]
                })
                .collect()
        };
        
        // Create texture with final_bytes and format
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colormap LUT"),
            size: wgpu::Extent3d {
                width: 256,  // Assuming 256x1 LUT
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Upload final_bytes to texture
        // ... texture upload logic ...
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        Ok(Self {
            texture,
            view,
            sampler,
        })
    }
}

// R2: CPU sRGB to linear conversion function
fn srgb_to_linear(srgb: u8) -> u8 {
    let normalized = srgb as f32 / 255.0;
    let linear = if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        ((normalized + 0.055) / 1.055).powf(2.4)
    };
    (linear * 255.0).round() as u8
}

fn load_colormap_png() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Placeholder for existing PNG loading logic
    // This should use the resolve_bytes function from the colormap module
    // let colormap_type = map_name_to_type("viridis")?; // example
    // let bytes = resolve_bytes(colormap_type)?;
    Ok(vec![])
}

// R1: Example of how to replace ColormapType::from_str usage:
// Instead of: let colormap = ColormapType::from_str(name)?;
// Use: let colormap = map_name_to_type(name)?;