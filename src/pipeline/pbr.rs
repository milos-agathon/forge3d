//! PBR GPU pipeline implementation
//!
//! Provides GPU resource management, texture handling, and bind group creation
//! for PBR materials using the metallic-roughness workflow.

use crate::core::material::{texture_flags, PbrLighting, PbrMaterial};
use crate::lighting::types::{MaterialShading, ShadowTechnique};
use crate::lighting::LightBuffer;
use crate::mesh::vertex::TbnVertex;
use crate::shadows::{ShadowManager, ShadowManagerConfig};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource,
    Buffer, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode, ImageCopyTexture,
    ImageDataLayout, Origin3d, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

// P2-06: Use centralized MaterialShading from lighting::types instead of duplicate definition
// MaterialShading is GPU-aligned and matches WGSL ShadingParamsGPU exactly

mod bindings;
mod constructor;
mod ibl;
mod material;
mod rendering;
mod scene_uniforms;
mod shadow;
mod state;
mod textures;
mod tone_mapping;

pub use material::PbrMaterialGpu;
pub use scene_uniforms::PbrSceneUniforms;
pub use shadow::{create_csm_with_preset, csm_shader_source, CsmQualityPreset};
pub use textures::{create_pbr_sampler, PbrTextures};
pub use tone_mapping::{
    exposure_from_stops, tone_map_color, tone_map_shader_source, ToneMappingConfig, ToneMappingMode,
};

use ibl::{create_fallback_ibl_resources, PbrIblResources};
use textures::{create_default_texture, create_texture_from_data};

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
