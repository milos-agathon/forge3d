//! PBR GPU pipeline implementation
//!
//! Provides GPU resource management, texture handling, and bind group creation
//! for PBR materials using the metallic-roughness workflow.

use crate::core::material::{texture_flags, PbrLighting, PbrMaterial};
use crate::mesh::vertex::TbnVertex;
use crate::lighting::types::{ShadowTechnique, MaterialShading};
use crate::lighting::LightBuffer;
use crate::shadows::{ShadowManager, ShadowManagerConfig};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::collections::HashMap;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource,
    Buffer, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode, ImageCopyTexture,
    ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};
use wgpu::util::DeviceExt;

// P2-06: Use centralized MaterialShading from lighting::types instead of duplicate definition
// MaterialShading is GPU-aligned and matches WGSL ShadingParamsGPU exactly

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

/// Scene uniforms for PBR pipeline (model, view, projection, normal matrices)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PbrSceneUniforms {
    pub model_matrix: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4],
    pub projection_matrix: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
}

impl PbrSceneUniforms {
    /// Construct uniforms from model, view, and projection matrices.
    pub fn from_matrices(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        let normal_matrix = compute_normal_matrix(model);
        Self {
            model_matrix: model.to_cols_array_2d(),
            view_matrix: view.to_cols_array_2d(),
            projection_matrix: projection.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
        }
    }
}

impl Default for PbrSceneUniforms {
    fn default() -> Self {
        Self::from_matrices(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY)
    }
}

fn compute_normal_matrix(model: Mat4) -> Mat4 {
    let determinant = model.determinant();
    if determinant.abs() < 1e-6 {
        Mat4::IDENTITY
    } else {
        model.inverse().transpose()
    }
}

#[derive(Debug)]
struct PbrIblResources {
    irradiance_texture: Texture,
    irradiance_view: TextureView,
    irradiance_sampler: Sampler,
    prefilter_texture: Texture,
    prefilter_view: TextureView,
    prefilter_sampler: Sampler,
    brdf_lut_texture: Texture,
    brdf_lut_view: TextureView,
    brdf_lut_sampler: Sampler,
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

fn create_fallback_ibl_resources(device: &Device, queue: &Queue) -> PbrIblResources {
    let irradiance_texture = create_default_texture(
        device,
        queue,
        "pbr_fallback_irradiance",
        [255, 255, 255, 255],
    );
    let prefilter_texture = create_default_texture(
        device,
        queue,
        "pbr_fallback_prefilter",
        [255, 255, 255, 255],
    );
    let brdf_lut_texture = create_default_texture(
        device,
        queue,
        "pbr_fallback_brdf_lut",
        [255, 255, 255, 255],
    );

    let irradiance_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_irradiance_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let prefilter_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_prefilter_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });
    let brdf_lut_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_fallback_ibl_lut_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });

    PbrIblResources {
        irradiance_view: irradiance_texture.create_view(&TextureViewDescriptor::default()),
        irradiance_sampler,
        irradiance_texture,
        prefilter_view: prefilter_texture.create_view(&TextureViewDescriptor::default()),
        prefilter_sampler,
        prefilter_texture,
        brdf_lut_view: brdf_lut_texture.create_view(&TextureViewDescriptor::default()),
        brdf_lut_sampler,
        brdf_lut_texture,
    }
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
pub struct PbrPipelineWithShadows {
    /// Base PBR material
    pub material: PbrMaterialGpu,
    /// CPU copy of scene transform uniforms
    pub scene_uniforms: PbrSceneUniforms,
    /// GPU buffer storing scene transform uniforms
    pub scene_uniform_buffer: Buffer,
    /// CPU copy of lighting parameters
    pub lighting_uniforms: PbrLighting,
    /// GPU buffer storing lighting parameters
   pub lighting_uniform_buffer: Buffer,
    /// CPU copy of shading parameters (BRDF routing) - P2-06
    pub shading_uniforms: MaterialShading,
    /// GPU buffer storing shading parameters
    pub shading_uniform_buffer: Buffer,
    /// Cached bind group for global uniforms
    pub globals_bind_group: Option<BindGroup>,
    /// IBL resources (fallback or user-provided)
    ibl_resources: PbrIblResources,
    /// Cached bind group for IBL sampling resources
    pub ibl_bind_group: Option<BindGroup>,
    /// Cached shadow configuration (reused when recreating managers)
    pub shadow_config: ShadowManagerConfig,
    /// Shadow manager providing atlas + technique uniforms
    pub shadow_manager: Option<ShadowManager>,
    /// Combined bind group including shadows
    pub shadow_bind_group: Option<BindGroup>,
    /// Layout describing the shadow bind group bindings
    pub shadow_bind_group_layout: Option<BindGroupLayout>,
    /// Bind group layout for global uniforms (model/view/projection + lighting)
    pub globals_bind_group_layout: BindGroupLayout,
    /// Bind group layout for material properties/textures
    pub material_bind_group_layout: BindGroupLayout,
    /// Bind group layout for IBL resources (irradiance/prefilter/LUT)
    pub ibl_bind_group_layout: BindGroupLayout,
    /// Cached render pipeline built from combined PBR + shadow shader
    pub render_pipeline: Option<wgpu::RenderPipeline>,
    /// Surface format associated with the cached pipeline
    pub pipeline_format: Option<TextureFormat>,
    /// Tone mapping configuration
    pub tone_mapping: ToneMappingConfig,
    /// P1-06: Light buffer for multi-light support with triple-buffering
    pub light_buffer: LightBuffer,
}

impl PbrPipelineWithShadows {
    /// Create new PBR pipeline with optional shadow support
    pub fn new(
        device: &Device,
        queue: &Queue,
        material: PbrMaterial,
        enable_shadows: bool,
    ) -> Self {
        let material_gpu = PbrMaterialGpu::new(device, material);
        let scene_uniforms = PbrSceneUniforms::default();
        let scene_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbr_scene_uniforms_buffer"),
            contents: bytemuck::bytes_of(&scene_uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let lighting_uniforms = PbrLighting::default();
        let lighting_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbr_lighting_uniforms_buffer"),
                contents: bytemuck::bytes_of(&lighting_uniforms),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // P2-06: Shading (BRDF selection) uniforms using MaterialShading
        let shading_uniforms = MaterialShading::default();
        let shading_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbr_shading_uniforms_buffer"),
            contents: bytemuck::bytes_of(&shading_uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let mut shadow_config = ShadowManagerConfig::default();
        shadow_config.technique = ShadowTechnique::PCF;
        shadow_config.csm.cascade_count = 3;
        shadow_config.csm.shadow_map_size = 2048;
        shadow_config.csm.max_shadow_distance = 200.0;
        shadow_config.csm.pcf_kernel_size = 3;
        shadow_config.csm.depth_bias = 0.005;
        shadow_config.csm.slope_bias = 0.01;
        shadow_config.csm.peter_panning_offset = 0.001;
        shadow_config.csm.debug_mode = 0;

        let globals_bind_group_layout = Self::create_globals_bind_group_layout(device);
        let material_bind_group_layout = Self::create_material_bind_group_layout(device);
        let ibl_bind_group_layout = Self::create_ibl_bind_group_layout(device);
        let ibl_resources = create_fallback_ibl_resources(device, queue);
        
        // P1-06: Initialize light buffer for multi-light support
        let light_buffer = LightBuffer::new(device);
        let ibl_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pbr_ibl_bind_group"),
            layout: &ibl_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&ibl_resources.irradiance_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&ibl_resources.irradiance_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&ibl_resources.prefilter_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&ibl_resources.prefilter_sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&ibl_resources.brdf_lut_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&ibl_resources.brdf_lut_sampler),
                },
            ],
        });
        let globals_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pbr_globals_bind_group"),
            layout: &globals_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lighting_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: shading_uniform_buffer.as_entire_binding(),
                },
                // P1-06: Light buffer bindings (3, 4, 5)
                BindGroupEntry {
                    binding: 3,
                    resource: light_buffer.current_light_buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: light_buffer.current_count_buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: light_buffer.environment_buffer().as_entire_binding(),
                },
            ],
        });

        let mut shadow_manager = None;
        let mut shadow_bind_group_layout = None;

        if enable_shadows {
            let manager = ShadowManager::new(device, shadow_config.clone());
            shadow_config = manager.config().clone();
            shadow_bind_group_layout = Some(manager.create_bind_group_layout(device));
            shadow_manager = Some(manager);
        }

        Self {
            material: material_gpu,
            scene_uniforms,
            scene_uniform_buffer,
            lighting_uniforms,
            lighting_uniform_buffer,
            shading_uniforms,
            shading_uniform_buffer,
            globals_bind_group: Some(globals_bind_group),
            ibl_resources,
            ibl_bind_group: Some(ibl_bind_group),
            shadow_config,
            shadow_manager,
            shadow_bind_group: None,
            shadow_bind_group_layout,
            globals_bind_group_layout,
            material_bind_group_layout,
            ibl_bind_group_layout,
            render_pipeline: None,
            pipeline_format: None,
            tone_mapping: ToneMappingConfig::new(ToneMappingMode::Reinhard, 1.0),
            light_buffer,
        }
    }

    /// Enable/disable shadow casting
    pub fn set_shadow_enabled(&mut self, device: &Device, enabled: bool) {
        if enabled {
            if self.shadow_manager.is_none() {
                self.rebuild_shadow_resources(device);
            }
        } else if self.shadow_manager.is_some() {
            self.drop_shadow_resources();
        }
    }

    /// Configure shadow quality settings
    pub fn configure_shadows(
        &mut self,
        device: &Device,
        pcf_kernel_size: u32,
        shadow_map_size: u32,
        debug_mode: u32,
    ) {
        self.shadow_config.csm.pcf_kernel_size = pcf_kernel_size;
        self.shadow_config.csm.shadow_map_size = shadow_map_size;
        self.shadow_config.csm.debug_mode = debug_mode;

        if self.shadow_manager.is_some() {
            self.rebuild_shadow_resources(device);
        }
        self.shadow_bind_group = None;
    }

    /// Change the active shadow technique and recreate resources if necessary
    pub fn set_shadow_technique(&mut self, device: &Device, technique: ShadowTechnique) {
        if self.shadow_config.technique == technique {
            return;
        }

        self.shadow_config.technique = technique;
        if self.shadow_manager.is_some() {
            self.rebuild_shadow_resources(device);
        } else {
            self.shadow_bind_group_layout = None;
        }

        self.shadow_bind_group = None;
    }

    /// Update tone mapping configuration
    pub fn set_tone_mapping(&mut self, config: ToneMappingConfig) {
        self.tone_mapping = config;
    }

    /// Overwrite scene uniforms and upload to GPU.
    pub fn update_scene_uniforms(&mut self, queue: &Queue, uniforms: &PbrSceneUniforms) {
        self.scene_uniforms = *uniforms;
        queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.scene_uniforms),
        );
    }

    /// Convenience helper to derive uniforms from matrices and upload them.
    pub fn update_scene_from_matrices(
        &mut self,
        queue: &Queue,
        model: Mat4,
        view: Mat4,
        projection: Mat4,
    ) {
        let uniforms = PbrSceneUniforms::from_matrices(model, view, projection);
        self.update_scene_uniforms(queue, &uniforms);
    }

    /// Overwrite lighting uniforms and upload to GPU.
    pub fn update_lighting_uniforms(&mut self, queue: &Queue, lighting: &PbrLighting) {
        self.lighting_uniforms = *lighting;
        queue.write_buffer(
            &self.lighting_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );
    }

    /// P1-06: Advance light buffer frame counter and refresh bind group
    /// Call this at the start of each frame before rendering
    pub fn advance_light_frame(&mut self, device: &Device) {
        // Advance to next triple-buffered index
        self.light_buffer.next_frame();
        
        // Invalidate bind group to force recreation with new buffers
        self.globals_bind_group = None;
    }
    
    /// P1-06: Upload lights to GPU via LightBuffer
    /// Returns error if more than MAX_LIGHTS (16) are provided
    pub fn update_lights(&mut self, device: &Device, queue: &Queue, lights: &[crate::lighting::types::Light]) -> Result<(), String> {
        self.light_buffer.update(device, queue, lights)?;
        
        // Invalidate bind group to pick up new light data
        self.globals_bind_group = None;
        
        Ok(())
    }
    
    /// P1-06: Get current light count for debugging
    pub fn light_count(&self) -> usize {
        self.light_buffer.last_uploaded_lights().len()
    }
    
    /// P1-06: Get debug info string from light buffer
    pub fn light_debug_info(&self) -> String {
        self.light_buffer.debug_info()
    }

    /// Ensure the material bind group exists, creating defaults if needed.
    pub fn ensure_material_bind_group(
        &mut self,
        device: &Device,
        queue: &Queue,
        sampler: &Sampler,
    ) {
        if self.material.bind_group.is_none() {
            self.material
                .create_bind_group(device, queue, &self.material_bind_group_layout, sampler);
        }
    }

    /// Get the bind group layout used for scene/global uniforms.
    pub fn globals_layout(&self) -> &BindGroupLayout {
        &self.globals_bind_group_layout
    }

    /// Get the bind group layout used for PBR material data.
    pub fn material_layout(&self) -> &BindGroupLayout {
        &self.material_bind_group_layout
    }

    /// Get the bind group layout used for IBL resources.
    pub fn ibl_layout(&self) -> &BindGroupLayout {
        &self.ibl_bind_group_layout
    }

    /// Ensure the render pipeline exists for the requested surface format.
    pub fn ensure_pipeline(
        &mut self,
        device: &Device,
        surface_format: TextureFormat,
    ) -> &wgpu::RenderPipeline {
        if self.shadow_manager.is_none() {
            self.rebuild_shadow_resources(device);
        }

        if self.shadow_bind_group_layout.is_none() {
            if let Some(manager) = self.shadow_manager.as_ref() {
                let layout = manager.create_bind_group_layout(device);
                self.shadow_bind_group_layout = Some(layout);
                self.render_pipeline = None;
            }
        }

        if self.pipeline_format != Some(surface_format) {
            self.render_pipeline = None;
            self.pipeline_format = Some(surface_format);
        }

        if self.render_pipeline.is_none() {
            let pipeline = self.build_render_pipeline(device, surface_format);
            self.render_pipeline = Some(pipeline);
        }

        self.render_pipeline
            .as_ref()
            .expect("PBR render pipeline should be initialized")
    }

    /// Bind the shadow resources at bind group slot 3 for the current render pass.
    pub fn bind_shadow_resources<'a>(
        &'a mut self,
        device: &Device,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        if let Some(bind_group) = self.get_or_create_shadow_bind_group(device) {
            pass.set_bind_group(3, bind_group, &[]);
        }
    }

    /// Bind IBL textures/samplers at bind group slot 2.
    pub fn bind_ibl_resources<'a>(
        &'a mut self,
        device: &Device,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        let bind_group = self.ensure_ibl_bind_group(device);
        pass.set_bind_group(2, bind_group, &[]);
    }

    /// Set pipeline state and bind all dependent resources for a render pass.
    pub fn begin_render<'a>(
        &'a mut self,
        device: &Device,
        surface_format: TextureFormat,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        self.ensure_pipeline(device, surface_format);
        let render_pipeline_ptr = self
            .render_pipeline
            .as_ref()
            .expect("render pipeline should be initialized")
            as *const wgpu::RenderPipeline;
        let globals_bind_group_ptr =
            self.ensure_globals_bind_group(device) as *const BindGroup;
        let material_bind_group_ptr = self
            .material
            .bind_group
            .as_ref()
            .map(|bg| bg as *const BindGroup);
        let ibl_bind_group_ptr = self.ensure_ibl_bind_group(device) as *const BindGroup;
        let shadow_bind_group_ptr = self
            .get_or_create_shadow_bind_group(device)
            .map(|bg| bg as *const BindGroup);

        // Safety: render pipeline lives as long as `self`; pass only reads it.
        pass.set_pipeline(unsafe { &*render_pipeline_ptr });
        pass.set_bind_group(0, unsafe { &*globals_bind_group_ptr }, &[]);
        if let Some(ptr) = material_bind_group_ptr {
            // Safety: bind group pointer derived from stored Option; lifetime tied to `self`.
            pass.set_bind_group(1, unsafe { &*ptr }, &[]);
        }
        pass.set_bind_group(2, unsafe { &*ibl_bind_group_ptr }, &[]);
        if let Some(ptr) = shadow_bind_group_ptr {
            pass.set_bind_group(3, unsafe { &*ptr }, &[]);
        }
    }

    /// Bind global scene + lighting uniforms at bind group slot 0.
    pub fn bind_global_uniforms<'a>(
        &'a mut self,
        device: &Device,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        let bind_group = self.ensure_globals_bind_group(device);
        pass.set_bind_group(0, bind_group, &[]);
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
        if let Some(ref mut manager) = self.shadow_manager {
            manager.update_cascades(
                camera_view,
                camera_projection,
                light_direction,
                near_plane,
                far_plane,
            );
            manager.upload_uniforms(queue);
        }
    }

    /// Get shadow bind group for rendering (recreates if necessary)
    pub fn get_or_create_shadow_bind_group(
        &mut self,
        device: &Device,
    ) -> Option<&BindGroup> {
        let manager = match self.shadow_manager.as_ref() {
            Some(manager) => manager,
            None => return None,
        };

        if self.shadow_bind_group_layout.is_none() {
            let layout = manager.create_bind_group_layout(device);
            self.shadow_bind_group_layout = Some(layout);
            self.render_pipeline = None;
        }

        let layout = self.shadow_bind_group_layout.as_ref()?;

        if self.shadow_bind_group.is_none() {
            let shadow_view = manager.shadow_view();
            let shadow_sampler = manager.shadow_sampler();
            let moment_view = manager.moment_view();
            let moment_sampler = manager.moment_sampler();

            let entries = [
                BindGroupEntry {
                    binding: 0,
                    resource: manager
                        .renderer()
                        .uniform_buffer
                        .as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&shadow_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(shadow_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&moment_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(moment_sampler),
                },
            ];

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("pbr_shadow_bind_group"),
                layout,
                entries: &entries,
            });

            self.shadow_bind_group = Some(bind_group);
        }

        self.shadow_bind_group.as_ref()
    }

    /// Check if shadows are enabled
    pub fn has_shadows(&self) -> bool {
        self.shadow_manager.is_some()
    }

    /// Get cascade information for debugging
    pub fn get_cascade_info(&self, cascade_idx: usize) -> Option<(f32, f32, f32)> {
        self.shadow_manager
            .as_ref()
            .and_then(|mgr| mgr.renderer().get_cascade_info(cascade_idx))
    }

    /// Validate that peter-panning prevention is working
    pub fn validate_peter_panning_prevention(&self) -> bool {
        self.shadow_manager
            .as_ref()
            .map(|mgr| mgr.renderer().validate_peter_panning_prevention())
            .unwrap_or(true)
    }

    /// Get the bind group layout required for shadow sampling.
    pub fn shadow_layout(&self) -> Option<&BindGroupLayout> {
        self.shadow_bind_group_layout.as_ref()
    }

    fn rebuild_shadow_resources(&mut self, device: &Device) {
        let manager = ShadowManager::new(device, self.shadow_config.clone());
        self.shadow_config = manager.config().clone();
        let layout = manager.create_bind_group_layout(device);
        self.shadow_bind_group = None;
        self.shadow_bind_group_layout = Some(layout);
        self.shadow_manager = Some(manager);
        self.render_pipeline = None;
        self.pipeline_format = None;
    }

    fn drop_shadow_resources(&mut self) {
        self.shadow_manager = None;
        self.shadow_bind_group = None;
        self.shadow_bind_group_layout = None;
        self.render_pipeline = None;
        self.pipeline_format = None;
    }

    fn build_render_pipeline(
        &mut self,
        device: &Device,
        surface_format: TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shadow_layout = self
            .shadow_bind_group_layout
            .as_ref()
            .expect("shadow layout must exist before building pipeline");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_pipeline_layout"),
            bind_group_layouts: &[
                &self.globals_bind_group_layout,
                &self.material_bind_group_layout,
                &self.ibl_bind_group_layout,
                shadow_layout,
            ],
            push_constant_ranges: &[],
        });

        let shader_source = format!(
            "{}\n{}",
            include_str!("../../shaders/shadows.wgsl"),
            include_str!("../../shaders/pbr.wgsl")
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr_shader_module"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[TbnVertex::buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    fn create_globals_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_globals_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // M2: ShadingParamsGPU (BRDF selection)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // P1-06: Binding 3 - Light array SSBO (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // P1-06: Binding 4 - Light metadata uniform (count, frame, seeds)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // P1-06: Binding 5 - Environment params uniform (stub for P4 IBL)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn ensure_globals_bind_group(&mut self, device: &Device) -> &BindGroup {
        if self.globals_bind_group.is_none() {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("pbr_globals_bind_group"),
                layout: &self.globals_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.scene_uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.lighting_uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.shading_uniform_buffer.as_entire_binding(),
                    },
                    // P1-06: Light buffer bindings (3, 4, 5)
                    BindGroupEntry {
                        binding: 3,
                        resource: self.light_buffer.current_light_buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: self.light_buffer.current_count_buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: self.light_buffer.environment_buffer().as_entire_binding(),
                    },
                ],
            });
            self.globals_bind_group = Some(bind_group);
        }
        self.globals_bind_group
            .as_ref()
            .expect("global bind group should exist")
    }

    /// Update BRDF model by index (matches WGSL constants)
    pub fn set_brdf_index(&mut self, queue: &Queue, brdf_index: u32) {
        self.shading_uniforms.brdf = brdf_index;
        queue.write_buffer(
            &self.shading_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.shading_uniforms),
        );
    }

    /// P2-06: Update full MaterialShading parameters from CPU to GPU
    /// 
    /// This uploads all BRDF dispatch parameters to the GPU uniform buffer,
    /// allowing dynamic control over shading model, metallic, roughness,
    /// and extended parameters (sheen, clearcoat, subsurface, anisotropy).
    /// 
    /// # Example
    /// ```ignore
    /// use forge3d::lighting::types::{MaterialShading, BrdfModel};
    /// 
    /// let mut shading = MaterialShading::default();
    /// shading.brdf = BrdfModel::DisneyPrincipled.as_u32();
    /// shading.metallic = 1.0;
    /// shading.roughness = 0.3;
    /// shading.sheen = 0.2;
    /// 
    /// pbr_state.update_shading_uniforms(&queue, &shading);
    /// ```
    pub fn update_shading_uniforms(&mut self, queue: &Queue, shading: &MaterialShading) {
        self.shading_uniforms = *shading;
        queue.write_buffer(
            &self.shading_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.shading_uniforms),
        );
    }

    fn ensure_ibl_bind_group(&mut self, device: &Device) -> &BindGroup {
        if self.ibl_bind_group.is_none() {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("pbr_ibl_bind_group"),
                layout: &self.ibl_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&self.ibl_resources.irradiance_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.ibl_resources.irradiance_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&self.ibl_resources.prefilter_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&self.ibl_resources.prefilter_sampler),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(&self.ibl_resources.brdf_lut_view),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::Sampler(&self.ibl_resources.brdf_lut_sampler),
                    },
                ],
            });
            self.ibl_bind_group = Some(bind_group);
        }
        self.ibl_bind_group
            .as_ref()
            .expect("ibl bind group should exist")
    }

    fn create_material_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_material_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_ibl_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_ibl_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
}

/// Create shadow manager with predefined quality presets
pub fn create_csm_with_preset(device: &Device, preset: CsmQualityPreset) -> ShadowManager {
    let mut config = ShadowManagerConfig::default();

    match preset {
        CsmQualityPreset::Low => {
            config.csm.cascade_count = 3;
            config.csm.shadow_map_size = 1024;
            config.csm.pcf_kernel_size = 1;
            config.csm.depth_bias = 0.01;
            config.csm.slope_bias = 0.02;
            config.csm.peter_panning_offset = 0.002;
            config.technique = ShadowTechnique::Hard;
        }
        CsmQualityPreset::Medium => {
            config.csm.cascade_count = 3;
            config.csm.shadow_map_size = 2048;
            config.csm.pcf_kernel_size = 3;
            config.csm.depth_bias = 0.005;
            config.csm.slope_bias = 0.01;
            config.csm.peter_panning_offset = 0.001;
            config.technique = ShadowTechnique::PCF;
        }
        CsmQualityPreset::High => {
            config.csm.cascade_count = 4;
            config.csm.shadow_map_size = 4096;
            config.csm.pcf_kernel_size = 5;
            config.csm.depth_bias = 0.003;
            config.csm.slope_bias = 0.005;
            config.csm.peter_panning_offset = 0.0005;
            config.technique = ShadowTechnique::PCF;
        }
        CsmQualityPreset::Ultra => {
            config.csm.cascade_count = 4;
            config.csm.shadow_map_size = 4096;
            config.csm.pcf_kernel_size = 7;
            config.csm.depth_bias = 0.002;
            config.csm.slope_bias = 0.003;
            config.csm.peter_panning_offset = 0.0003;
            config.technique = ShadowTechnique::EVSM;
            config.pcss_blocker_radius = 0.02;
            config.pcss_filter_radius = 0.05;
            config.moment_bias = 0.0002;
        }
    };

    ShadowManager::new(device, config)
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
