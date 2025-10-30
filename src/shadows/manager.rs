// src/shadows/manager.rs
// Shadow technique orchestration with atlas budgeting and parameter uniforms
// Exists to unify cascaded depth resources with pluggable filtering techniques
// RELEVANT FILES: src/shadows/csm.rs, shaders/shadows.wgsl, src/pipeline/pbr.rs, python/forge3d/config.py

use glam::{Mat4, Vec3};
use log::warn;
use std::num::NonZeroU64;

use wgpu::{
    AddressMode, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, Device, Extent3d, FilterMode, Queue, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderStages, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureAspect, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

use crate::lighting::types::ShadowTechnique;

use super::{CsmConfig, CsmRenderer, CsmUniforms};

const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
const MIN_SHADOW_RESOLUTION: u32 = 256;
const MAX_SEARCH_TEXELS: f32 = 6.0;

/// High-level configuration used to instantiate the shadow manager.
#[derive(Debug, Clone)]
pub struct ShadowManagerConfig {
    pub csm: CsmConfig,
    pub technique: ShadowTechnique,
    pub pcss_blocker_radius: f32,
    pub pcss_filter_radius: f32,
    pub light_size: f32,
    pub moment_bias: f32,
    pub max_memory_bytes: u64,
}

impl Default for ShadowManagerConfig {
    fn default() -> Self {
        Self {
            csm: CsmConfig::default(),
            technique: ShadowTechnique::PCF,
            pcss_blocker_radius: 0.03,
            pcss_filter_radius: 0.06,
            light_size: 0.25,
            moment_bias: 0.0005,
            max_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

/// Runtime controller that keeps the CSM atlas and technique uniforms in sync.
pub struct ShadowManager {
    config: ShadowManagerConfig,
    renderer: CsmRenderer,
    moment_sampler: Sampler,
    fallback_moment_texture: Option<Texture>,
    requires_moments: bool,
    memory_bytes: u64,
}

impl ShadowManager {
    /// Construct manager while respecting memory constraints and technique requirements.
    pub fn new(device: &Device, mut config: ShadowManagerConfig) -> Self {
        let requires_moments = matches!(
            config.technique,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        );
        config.csm.enable_evsm = requires_moments;

        Self::enforce_memory_budget(&mut config);

        let mut renderer = CsmRenderer::new(device, config.csm.clone());

        let moment_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_moment_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        let fallback_moment_texture = if requires_moments {
            None
        } else {
            Some(Self::create_fallback_moment_texture(device))
        };

        let memory_bytes = renderer.total_memory_bytes();

        let mut manager = Self {
            config,
            renderer,
            moment_sampler,
            fallback_moment_texture,
            requires_moments,
            memory_bytes,
        };

        manager.apply_uniform_overrides();
        manager
    }

    /// Access underlying configuration.
    pub fn config(&self) -> &ShadowManagerConfig {
        &self.config
    }

    /// Mutable access to configuration for dynamic adjustments.
    pub fn config_mut(&mut self) -> &mut ShadowManagerConfig {
        &mut self.config
    }

    /// Underlying cascaded shadow renderer.
    pub fn renderer(&self) -> &CsmRenderer {
        &self.renderer
    }

    /// Mutable renderer access.
    pub fn renderer_mut(&mut self) -> &mut CsmRenderer {
        &mut self.renderer
    }

    /// Depth sampler used for comparisons.
    pub fn shadow_sampler(&self) -> &Sampler {
        &self.renderer.shadow_sampler
    }

    /// Sampler used for moment-based filtering (also bound for fallbacks).
    pub fn moment_sampler(&self) -> &Sampler {
        &self.moment_sampler
    }

    /// Create a bind group layout that matches the active shadow technique.
    pub fn create_bind_group_layout(&self, device: &Device) -> BindGroupLayout {
        let entries = [
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<CsmUniforms>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Comparison),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ];

        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_manager_bind_group_layout"),
            entries: &entries,
        })
    }

    /// Depth atlas view covering all cascades.
    pub fn shadow_view(&self) -> wgpu::TextureView {
        self.renderer.shadow_texture_view()
    }

    /// Moment atlas view or a fallback texture when moments are unused.
    pub fn moment_view(&self) -> wgpu::TextureView {
        if let Some(view) = self.renderer.moment_texture_view() {
            view
        } else if let Some(ref texture) = self.fallback_moment_texture {
            let layer_count = self.config.csm.cascade_count.max(1).min(4);
            texture.create_view(&TextureViewDescriptor {
                label: Some("shadow_moment_fallback_view"),
                format: Some(TextureFormat::Rgba32Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(layer_count),
                ..Default::default()
            })
        } else {
            self.renderer.shadow_texture_view()
        }
    }

    /// Active technique enumeration.
    pub fn technique(&self) -> ShadowTechnique {
        self.config.technique
    }

    /// Total GPU memory currently consumed by the atlas.
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Update cascades with latest camera/light state and refresh technique parameters.
    pub fn update_cascades(
        &mut self,
        camera_view: Mat4,
        camera_projection: Mat4,
        light_direction: Vec3,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.renderer.update_cascades(
            camera_view,
            camera_projection,
            light_direction,
            near_plane,
            far_plane,
        );
        self.apply_uniform_overrides();
    }

    /// Upload uniforms to GPU.
    pub fn upload_uniforms(&self, queue: &Queue) {
        self.renderer.upload_uniforms(queue);
    }

    /// Returns true if the active technique reads the moment atlas.
    pub fn uses_moments(&self) -> bool {
        self.requires_moments
    }

    /// Re-apply technique parameters after external configuration tweaks.
    pub fn refresh_uniforms(&mut self) {
        self.apply_uniform_overrides();
    }

    fn apply_uniform_overrides(&mut self) {
        self.renderer
            .set_debug_mode(self.config.csm.debug_mode);
        self.renderer.uniforms.technique = self.config.technique.as_u32();
        self.renderer.uniforms.technique_flags =
            Self::compute_flags(self.config.technique, self.requires_moments);
        self.renderer.uniforms.technique_params = [
            self.config.pcss_blocker_radius,
            self.config.pcss_filter_radius,
            self.config.moment_bias,
            self.config.light_size,
        ];
        self.renderer.uniforms.technique_reserved = [0.0; 4];

        if matches!(self.config.technique, ShadowTechnique::PCSS) {
            self.clamp_pcss_radius();
        }
    }

    fn clamp_pcss_radius(&mut self) {
        let cascade_count = self.renderer.config.cascade_count as usize;
        if cascade_count == 0 {
            return;
        }

        let min_texel_size = self
            .renderer
            .uniforms
            .cascades
            .iter()
            .take(cascade_count)
            .map(|c| c.texel_size)
            .fold(f32::MAX, f32::min);

        let max_radius = min_texel_size * MAX_SEARCH_TEXELS;

        self.renderer.uniforms.technique_params[0] =
            self.config.pcss_blocker_radius.min(max_radius);
        self.renderer.uniforms.technique_params[1] =
            self.config.pcss_filter_radius.min(max_radius * 2.0);
    }

    fn compute_flags(technique: ShadowTechnique, requires_moments: bool) -> u32 {
        let mut flags = 0u32;
        if requires_moments {
            flags |= 0b01;
        }
        if matches!(technique, ShadowTechnique::PCSS) {
            flags |= 0b10;
        }
        flags
    }

    fn create_fallback_moment_texture(device: &Device) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some("shadow_moment_fallback"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn enforce_memory_budget(config: &mut ShadowManagerConfig) {
        loop {
            let usage = Self::estimate_memory_bytes(
                config.csm.shadow_map_size,
                config.csm.cascade_count,
                config.technique,
            );

            if usage <= config.max_memory_bytes {
                break;
            }

            let next_res = (config.csm.shadow_map_size / 2).max(MIN_SHADOW_RESOLUTION);
            if next_res == config.csm.shadow_map_size {
                warn!(
                    "Shadow atlas exceeds {} MiB budget; continuing with {} px maps ({:.1} MiB).",
                    config.max_memory_bytes as f64 / (1024.0 * 1024.0),
                    next_res,
                    usage as f64 / (1024.0 * 1024.0)
                );
                break;
            }

            warn!(
                "Shadow atlas over budget ({:.1} MiB); downscaling resolution {} -> {}.",
                usage as f64 / (1024.0 * 1024.0),
                config.csm.shadow_map_size,
                next_res
            );
            config.csm.shadow_map_size = next_res;
        }
    }

    fn estimate_memory_bytes(
        map_resolution: u32,
        cascades: u32,
        technique: ShadowTechnique,
    ) -> u64 {
        let res = map_resolution as u64;
        let casc = cascades as u64;
        let depth_bytes = res * res * casc * 4;

        let moment_bytes = match technique {
            ShadowTechnique::VSM => res * res * casc * 8,
            ShadowTechnique::EVSM | ShadowTechnique::MSM => res * res * casc * 16,
            _ => 0,
        };

        depth_bytes + moment_bytes
    }
}
