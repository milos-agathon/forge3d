//! PBR GPU pipeline implementation
//!
//! Provides GPU resource management, texture handling, and bind group creation
//! for PBR materials using the metallic-roughness workflow.

use crate::core::material::{texture_flags, PbrMaterial};
use crate::shadows::{CsmConfig, CsmRenderer};
use std::collections::HashMap;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode, ImageCopyTexture,
    ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

/// Tone mapping operators available to the PBR pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMappingMode {
    /// Filmic ACES approximation
    Aces,
    /// Classic Reinhard curve
    Reinhard,
    /// Hable filmic curve (Uncharted 2)
    Hable,
}

impl ToneMappingMode {
    /// Map enum variant to shader index for tone mapping selection
    pub fn as_index(self) -> u32 {
        match self {
            ToneMappingMode::Aces => 0,
            ToneMappingMode::Reinhard => 1,
            ToneMappingMode::Hable => 2,
        }
    }
}

/// Tone mapping configuration shared between CPU previews and GPU passes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMappingConfig {
    pub mode: ToneMappingMode,
    pub exposure: f32,
}

impl ToneMappingConfig {
    /// Create new tone mapping configuration with explicit exposure
    pub fn new(mode: ToneMappingMode, exposure: f32) -> Self {
        let exposure = exposure.max(1e-6);
        Self { mode, exposure }
    }

    /// Create config from exposure stops (2**stops multiplier)
    pub fn with_stops(mode: ToneMappingMode, stops: f32) -> Self {
        Self::new(mode, exposure_from_stops(stops))
    }

    /// Update exposure using stops value
    pub fn set_exposure_stops(&mut self, stops: f32) {
        self.exposure = exposure_from_stops(stops);
    }

    /// Convert stored exposure back to stops for UI readback
    pub fn exposure_stops(&self) -> f32 {
        self.exposure.max(1e-6).log2()
    }
}

/// Convert exposure stops to scalar multiplier
pub fn exposure_from_stops(stops: f32) -> f32 {
    2.0_f32.powf(stops)
}

const HABLE_WHITE_POINT: f32 = 11.2;

fn tone_curve_reinhard(value: f32) -> f32 {
    value / (1.0 + value)
}

fn tone_curve_aces(value: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let numerator = value * (a * value + b);
    let denominator = value * (c * value + d) + e;
    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

fn tone_curve_hable(value: f32) -> f32 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    let numerator = value * (value * a + c * b) + d * e;
    let denominator = value * (value * a + b) + d * f;
    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        let tone = numerator / denominator - e / f;
        let white_scale = {
            let w_num = HABLE_WHITE_POINT * (HABLE_WHITE_POINT * a + c * b) + d * e;
            let w_den = HABLE_WHITE_POINT * (HABLE_WHITE_POINT * a + b) + d * f;
            if w_den.abs() < f32::EPSILON {
                1.0
            } else {
                (w_num / w_den) - e / f
            }
        };
        if white_scale.abs() < f32::EPSILON {
            0.0
        } else {
            (tone / white_scale).clamp(0.0, 1.0)
        }
    }
}

fn tone_map_scalar(value: f32, config: ToneMappingConfig) -> f32 {
    let exposed = (value * config.exposure).max(0.0);
    match config.mode {
        ToneMappingMode::Aces => tone_curve_aces(exposed),
        ToneMappingMode::Reinhard => tone_curve_reinhard(exposed),
        ToneMappingMode::Hable => tone_curve_hable(exposed),
    }
}

/// Apply tone mapping to RGB color using provided configuration
pub fn tone_map_color(color: [f32; 3], config: ToneMappingConfig) -> [f32; 3] {
    [
        tone_map_scalar(color[0], config),
        tone_map_scalar(color[1], config),
        tone_map_scalar(color[2], config),
    ]
}

/// Provide WGSL source for the tone mapping pass used by shaders
pub fn tone_map_shader_source() -> &'static str {
    include_str!("../../shaders/tone_map.wgsl")
}

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
        let default_white =
            create_default_texture(device, queue, "default_white", [255, 255, 255, 255]);
        let default_normal =
            create_default_texture(device, queue, "default_normal", [128, 128, 255, 255]);
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
    create_texture_from_data(
        device,
        queue,
        label,
        &color,
        1,
        1,
        TextureFormat::Rgba8Unorm,
    )
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

/// Enhanced PBR pipeline with integrated Cascaded Shadow Maps support
#[derive(Debug)]
pub struct PbrPipelineWithShadows {
    /// Base PBR material
    pub material: PbrMaterialGpu,
    /// CSM renderer for shadow mapping
    pub csm_renderer: Option<CsmRenderer>,
    /// Combined bind group including shadows
    pub shadow_bind_group: Option<BindGroup>,
    /// Tone mapping configuration
    pub tone_mapping: ToneMappingConfig,
}

impl PbrPipelineWithShadows {
    /// Create new PBR pipeline with optional shadow support
    pub fn new(device: &Device, material: PbrMaterial, enable_shadows: bool) -> Self {
        let material_gpu = PbrMaterialGpu::new(device, material);

        let csm_renderer = if enable_shadows {
            let config = CsmConfig {
                cascade_count: 3,
                shadow_map_size: 2048,
                max_shadow_distance: 200.0,
                pcf_kernel_size: 3,
                depth_bias: 0.005,
                slope_bias: 0.01,
                peter_panning_offset: 0.001,
                enable_evsm: false,
                debug_mode: 0,
                ..Default::default()
            };
            Some(CsmRenderer::new(device, config))
        } else {
            None
        };

        Self {
            material: material_gpu,
            csm_renderer,
            shadow_bind_group: None,
            tone_mapping: ToneMappingConfig::new(ToneMappingMode::Reinhard, 1.0),
        }
    }

    /// Enable/disable shadow casting
    pub fn set_shadow_enabled(&mut self, device: &Device, enabled: bool) {
        if enabled && self.csm_renderer.is_none() {
            let config = CsmConfig::default();
            self.csm_renderer = Some(CsmRenderer::new(device, config));
        } else if !enabled {
            self.csm_renderer = None;
        }
        // Invalidate bind group to force recreation
        self.shadow_bind_group = None;
    }

    /// Configure shadow quality settings
    pub fn configure_shadows(
        &mut self,
        pcf_kernel_size: u32,
        shadow_map_size: u32,
        debug_mode: u32,
    ) {
        if let Some(ref mut csm) = self.csm_renderer {
            csm.config.pcf_kernel_size = pcf_kernel_size;
            csm.config.shadow_map_size = shadow_map_size;
            csm.config.debug_mode = debug_mode;
            csm.set_debug_mode(debug_mode);
        }
    }

    /// Update tone mapping configuration
    pub fn set_tone_mapping(&mut self, config: ToneMappingConfig) {
        self.tone_mapping = config;
    }

    /// Update shadow cascades for current frame
    pub fn update_shadows(
        &mut self,
        queue: &Queue,
        camera_view: glam::Mat4,
        camera_projection: glam::Mat4,
        light_direction: glam::Vec3,
        near_plane: f32,
        far_plane: f32,
    ) {
        if let Some(ref mut csm) = self.csm_renderer {
            csm.update_cascades(
                camera_view,
                camera_projection,
                light_direction,
                near_plane,
                far_plane,
            );
            csm.upload_uniforms(queue);
        }
    }

    /// Get shadow bind group for rendering (recreates if necessary)
    pub fn get_or_create_shadow_bind_group(
        &mut self,
        device: &Device,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<&BindGroup> {
        if let Some(ref csm) = self.csm_renderer {
            if self.shadow_bind_group.is_none() {
                let shadow_view = csm.shadow_texture_view();

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("pbr_shadow_bind_group"),
                    layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: csm.uniform_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(&shadow_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Sampler(&csm.shadow_sampler),
                        },
                    ],
                });

                self.shadow_bind_group = Some(bind_group);
            }

            self.shadow_bind_group.as_ref()
        } else {
            None
        }
    }

    /// Check if shadows are enabled
    pub fn has_shadows(&self) -> bool {
        self.csm_renderer.is_some()
    }

    /// Get cascade information for debugging
    pub fn get_cascade_info(&self, cascade_idx: usize) -> Option<(f32, f32, f32)> {
        self.csm_renderer
            .as_ref()
            .and_then(|csm| csm.get_cascade_info(cascade_idx))
    }

    /// Validate that peter-panning prevention is working
    pub fn validate_peter_panning_prevention(&self) -> bool {
        self.csm_renderer
            .as_ref()
            .map(|csm| csm.validate_peter_panning_prevention())
            .unwrap_or(true)
    }
}

/// Create CSM renderer with predefined quality presets
pub fn create_csm_with_preset(device: &Device, preset: CsmQualityPreset) -> CsmRenderer {
    let config = match preset {
        CsmQualityPreset::Low => CsmConfig {
            cascade_count: 3,
            shadow_map_size: 1024,
            pcf_kernel_size: 1, // No PCF
            depth_bias: 0.01,
            slope_bias: 0.02,
            peter_panning_offset: 0.002,
            enable_evsm: false,
            debug_mode: 0,
            ..Default::default()
        },
        CsmQualityPreset::Medium => CsmConfig {
            cascade_count: 3,
            shadow_map_size: 2048,
            pcf_kernel_size: 3, // 3x3 PCF
            depth_bias: 0.005,
            slope_bias: 0.01,
            peter_panning_offset: 0.001,
            enable_evsm: false,
            debug_mode: 0,
            ..Default::default()
        },
        CsmQualityPreset::High => CsmConfig {
            cascade_count: 4,
            shadow_map_size: 4096,
            pcf_kernel_size: 5, // 5x5 PCF
            depth_bias: 0.003,
            slope_bias: 0.005,
            peter_panning_offset: 0.0005,
            enable_evsm: false,
            debug_mode: 0,
            ..Default::default()
        },
        CsmQualityPreset::Ultra => CsmConfig {
            cascade_count: 4,
            shadow_map_size: 4096,
            pcf_kernel_size: 7, // Poisson disk PCF
            depth_bias: 0.002,
            slope_bias: 0.003,
            peter_panning_offset: 0.0003,
            enable_evsm: true,
            debug_mode: 0,
            ..Default::default()
        },
    };

    CsmRenderer::new(device, config)
}

/// CSM quality presets for different performance/quality tradeoffs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsmQualityPreset {
    /// Low quality: 3 cascades, 1024px, no PCF
    Low,
    /// Medium quality: 3 cascades, 2048px, 3x3 PCF
    Medium,
    /// High quality: 4 cascades, 4096px, 5x5 PCF
    High,
    /// Ultra quality: 4 cascades, 4096px, Poisson PCF + EVSM
    Ultra,
}

/// Get WGSL source for CSM integration
pub fn csm_shader_source() -> &'static str {
    include_str!("../../shaders/csm.wgsl")
}
