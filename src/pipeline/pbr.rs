//! PBR GPU pipeline implementation
//!
//! Provides GPU resource management, texture handling, and bind group creation
//! for PBR materials using the metallic-roughness workflow.

use crate::core::material::{texture_flags, PbrMaterial};
use glam::Vec3;
use std::collections::HashMap;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode, ImageCopyTexture,
    ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

/// PBR texture set for a material
#[derive(Debug)]
pub struct PbrTextures {
    /// Base color (albedo) texture - RGBA8
    pub base_color: Option<Texture>,

    /// Metallic-roughness texture - RG format (B=metallic, G=roughness)
    pub metallic_roughness: Option<Texture>,

    /// Normal map texture - RGB format (tangent space)
    pub normal: Option<Texture>,

    /// Ambient occlusion texture - R format
    pub occlusion: Option<Texture>,

    /// Emissive texture - RGB format
    pub emissive: Option<Texture>,
}

/// PBR material with GPU resources
#[derive(Debug)]
pub struct PbrMaterialGpu {
    /// Material properties
    pub material: PbrMaterial,

    /// Material uniform buffer
    pub uniform_buffer: Buffer,

    /// PBR textures
    pub textures: PbrTextures,

    /// Texture views for binding
    pub texture_views: HashMap<String, TextureView>,

    /// Material bind group
    pub bind_group: Option<BindGroup>,
}

impl PbrMaterialGpu {
    /// Create PBR material GPU resources
    pub fn new(device: &Device, material: PbrMaterial) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("pbr_material_uniforms"),
            size: std::mem::size_of::<PbrMaterial>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            material,
            uniform_buffer,
            textures: PbrTextures {
                base_color: None,
                metallic_roughness: None,
                normal: None,
                occlusion: None,
                emissive: None,
            },
            texture_views: HashMap::new(),
            bind_group: None,
        }
    }

    /// Update material properties on GPU
    pub fn update_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.material]),
        );
    }

    /// Set base color texture
    pub fn set_base_color_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_base_color",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8UnormSrgb, // sRGB for color textures
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.base_color = Some(texture);
        self.texture_views.insert("base_color".to_string(), view);
        self.material.texture_flags |= texture_flags::BASE_COLOR;
    }

    /// Set metallic-roughness texture
    pub fn set_metallic_roughness_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_metallic_roughness",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8Unorm, // Linear for material properties
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.metallic_roughness = Some(texture);
        self.texture_views
            .insert("metallic_roughness".to_string(), view);
        self.material.texture_flags |= texture_flags::METALLIC_ROUGHNESS;
    }

    /// Set normal map texture
    pub fn set_normal_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_normal",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8Unorm, // Linear for normal maps
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.normal = Some(texture);
        self.texture_views.insert("normal".to_string(), view);
        self.material.texture_flags |= texture_flags::NORMAL;
    }

    /// Set occlusion texture
    pub fn set_occlusion_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_occlusion",
            texture_data,
            width,
            height,
            TextureFormat::R8Unorm, // Single channel for AO
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.occlusion = Some(texture);
        self.texture_views.insert("occlusion".to_string(), view);
        self.material.texture_flags |= texture_flags::OCCLUSION;
    }

    /// Set emissive texture
    pub fn set_emissive_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        texture_data: &[u8],
        width: u32,
        height: u32,
    ) {
        let texture = create_texture_from_data(
            device,
            queue,
            "pbr_emissive",
            texture_data,
            width,
            height,
            TextureFormat::Rgba8UnormSrgb, // sRGB for emissive colors
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.textures.emissive = Some(texture);
        self.texture_views.insert("emissive".to_string(), view);
        self.material.texture_flags |= texture_flags::EMISSIVE;
    }

    /// Create bind group for material
    pub fn create_bind_group(
        &mut self,
        device: &Device,
        queue: &Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &Sampler,
    ) {
        // Create default textures for missing ones
        let default_white = create_default_texture(device, queue, "default_white", [255, 255, 255, 255]);
        let default_normal = create_default_texture(device, queue, "default_normal", [128, 128, 255, 255]);
        let default_metallic_roughness =
            create_default_texture(device, queue, "default_mr", [0, 255, 0, 255]); // No metallic, full roughness
        let default_black = create_default_texture(device, queue, "default_black", [0, 0, 0, 0]);

        // Create views for default textures
        let default_white_view = default_white.create_view(&TextureViewDescriptor::default());
        let default_normal_view = default_normal.create_view(&TextureViewDescriptor::default());
        let default_mr_view =
            default_metallic_roughness.create_view(&TextureViewDescriptor::default());
        let default_black_view = default_black.create_view(&TextureViewDescriptor::default());

        let base_color_view = self
            .texture_views
            .get("base_color")
            .unwrap_or(&default_white_view);
        let metallic_roughness_view = self
            .texture_views
            .get("metallic_roughness")
            .unwrap_or(&default_mr_view);
        let normal_view = self
            .texture_views
            .get("normal")
            .unwrap_or(&default_normal_view);
        let occlusion_view = self
            .texture_views
            .get("occlusion")
            .unwrap_or(&default_white_view);
        let emissive_view = self
            .texture_views
            .get("emissive")
            .unwrap_or(&default_black_view);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pbr_material_bind_group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(base_color_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(metallic_roughness_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(occlusion_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(emissive_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }
}

/// Create texture from raw data
fn create_texture_from_data(
    device: &Device,
    queue: &Queue,
    label: &str,
    data: &[u8],
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Texture {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Calculate bytes per pixel based on format
    let bytes_per_pixel = match format {
        TextureFormat::R8Unorm => 1,
        TextureFormat::Rg8Unorm => 2,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => 4,
        _ => 4, // Default to 4 bytes
    };

    let bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((bytes_per_row + alignment - 1) / alignment) * alignment
    };

    // Create padded data if necessary
    if bytes_per_row == padded_bytes_per_row {
        // No padding needed
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        // Need to pad rows
        let mut padded_data = vec![0u8; (padded_bytes_per_row * height) as usize];
        for y in 0..height {
            let src_offset = (y * bytes_per_row) as usize;
            let dst_offset = (y * padded_bytes_per_row) as usize;
            let src_range = src_offset..(src_offset + bytes_per_row as usize);
            let dst_range = dst_offset..(dst_offset + bytes_per_row as usize);

            if src_range.end <= data.len() && dst_range.end <= padded_data.len() {
                padded_data[dst_range].copy_from_slice(&data[src_range]);
            }
        }

        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    texture
}

/// Create default 1x1 texture with specific color
fn create_default_texture(device: &Device, queue: &Queue, label: &str, color: [u8; 4]) -> Texture {
    create_texture_from_data(device, queue, label, &color, 1, 1, TextureFormat::Rgba8Unorm)
}

/// Create PBR material sampler
pub fn create_pbr_sampler(device: &Device) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_material_sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    })
}
