//! Shadow mapping implementation with Cascaded Shadow Maps (CSM)
//!
//! Provides GPU-accelerated shadow mapping with multiple cascades,
//! depth texture arrays, and PCF filtering support.

use crate::core::cascade_split::{generate_cascades, CascadeSplitConfig, ShadowCascade};
use glam::{Mat4, Vec3, Vec4};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CompareFunction, Device, Extent3d, FilterMode, Queue, Sampler,
    SamplerDescriptor, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

/// PCF quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcfQuality {
    /// No filtering (single sample)
    None = 1,
    /// 3x3 PCF kernel
    Low = 3,
    /// 5x5 PCF kernel
    Medium = 5,
    /// 7x7 PCF kernel or Poisson disk sampling
    High = 7,
}

/// Shadow mapping configuration
#[derive(Debug, Clone)]
pub struct ShadowMappingConfig {
    /// Resolution of each shadow map
    pub shadow_map_size: u32,

    /// PCF quality setting
    pub pcf_quality: PcfQuality,

    /// Depth bias to prevent shadow acne
    pub depth_bias: f32,

    /// Slope-scaled bias factor
    pub slope_bias: f32,

    /// Shadow debug visualization mode:
    ///   0 = disabled
    ///   1 = cascade boundary overlay (color-coded by cascade)
    ///   2 = raw shadow visibility (grayscale)
    /// Set via FORGE3D_TERRAIN_SHADOW_DEBUG env var: "cascades" or "raw"
    pub debug_mode: u32,

    /// Shadow map format (D24Plus or D32Float)
    pub depth_format: TextureFormat,
}

impl Default for ShadowMappingConfig {
    fn default() -> Self {
        Self {
            shadow_map_size: 1024,
            pcf_quality: PcfQuality::Medium,
            depth_bias: 0.005,
            slope_bias: 1.0,
            debug_mode: 0,
            depth_format: TextureFormat::Depth24Plus,
        }
    }
}

/// CSM uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsmUniforms {
    /// Light direction in world space
    pub light_direction: [f32; 4],

    /// Light view matrix  
    pub light_view: [[f32; 4]; 4],

    /// Shadow cascade data (up to 4 cascades)
    pub cascades: [CsmCascadeData; 4],

    /// Number of active cascades
    pub cascade_count: u32,

    /// PCF kernel size
    pub pcf_kernel_size: u32,

    /// Depth bias to prevent acne
    pub depth_bias: f32,

    /// Slope-scaled bias
    pub slope_bias: f32,

    /// Shadow map resolution
    pub shadow_map_size: f32,

    /// Debug visualization mode
    pub debug_mode: u32,

    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,

    /// PCSS light radius (optional softness control)
    pub pcss_light_radius: f32,
}

/// GPU representation of a shadow cascade
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsmCascadeData {
    /// Light-space projection matrix
    pub light_projection: [[f32; 4]; 4],

    /// Combined light_view_proj matrix (projection * view)
    /// Pre-computed for efficiency and to ensure consistency with shadow depth pass
    pub light_view_proj: [[f32; 4]; 4],

    /// Near plane distance
    pub near_distance: f32,

    /// Far plane distance
    pub far_distance: f32,

    /// Texel size in world space
    pub texel_size: f32,

    /// Padding for alignment
    pub _padding: f32,
}

/// Shadow atlas information for debugging
#[derive(Debug)]
pub struct ShadowAtlasInfo {
    /// Number of cascades
    pub cascade_count: u32,

    /// Atlas dimensions (width, height, depth)
    pub atlas_dimensions: (u32, u32, u32),

    /// Individual cascade resolutions
    pub cascade_resolutions: Vec<u32>,

    /// Memory usage in bytes
    pub memory_usage: u64,
}

/// Statistics from shadow map generation
#[derive(Debug)]
pub struct ShadowStats {
    /// Number of draw calls for shadow generation
    pub draw_calls: u32,

    /// Number of triangles rendered to shadow maps
    pub triangles_rendered: u64,

    /// Time taken for shadow map generation (ms)
    pub generation_time_ms: f32,

    /// GPU memory usage for shadow maps (bytes)
    pub memory_usage_bytes: u64,
}

/// Main shadow mapping system
pub struct ShadowMapping {
    /// Configuration
    config: ShadowMappingConfig,

    /// Cascade configuration
    cascade_config: CascadeSplitConfig,

    /// Shadow map texture array
    shadow_maps: Option<Texture>,

    /// Shadow map views for each cascade
    cascade_views: Vec<TextureView>,

    /// Shadow sampler
    shadow_sampler: Option<Sampler>,

    /// Uniform buffer
    uniform_buffer: Option<Buffer>,

    /// Bind group for shadow resources
    bind_group: Option<BindGroup>,

    /// Current cascade data
    cascades: Vec<ShadowCascade>,

    /// Light parameters
    light_direction: Vec3,
    light_view_matrix: Mat4,
}

impl ShadowMapping {
    /// Create new shadow mapping system
    pub fn new(config: ShadowMappingConfig, cascade_config: CascadeSplitConfig) -> Self {
        Self {
            config,
            cascade_config,
            shadow_maps: None,
            cascade_views: Vec::new(),
            shadow_sampler: None,
            uniform_buffer: None,
            bind_group: None,
            cascades: Vec::new(),
            light_direction: Vec3::new(0.0, -1.0, 0.3).normalize(),
            light_view_matrix: Mat4::IDENTITY,
        }
    }

    /// Initialize GPU resources
    pub fn initialize(&mut self, device: &Device) -> Result<(), String> {
        // Create shadow map texture array
        let shadow_maps = device.create_texture(&TextureDescriptor {
            label: Some("shadow_maps"),
            size: Extent3d {
                width: self.config.shadow_map_size,
                height: self.config.shadow_map_size,
                depth_or_array_layers: self.cascade_config.cascade_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: self.config.depth_format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create individual cascade views
        let mut cascade_views = Vec::new();
        for i in 0..self.cascade_config.cascade_count {
            let view = shadow_maps.create_view(&TextureViewDescriptor {
                label: Some(&format!("shadow_cascade_{}", i)),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: i,
                array_layer_count: Some(1),
            });
            cascade_views.push(view);
        }

        // Create shadow sampler
        let shadow_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            border_color: None,
            anisotropy_clamp: 1,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("csm_uniforms"),
            size: std::mem::size_of::<CsmUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.shadow_maps = Some(shadow_maps);
        self.cascade_views = cascade_views;
        self.shadow_sampler = Some(shadow_sampler);
        self.uniform_buffer = Some(uniform_buffer);

        Ok(())
    }

    /// Update cascades for current frame
    pub fn update_cascades(
        &mut self,
        light_direction: Vec3,
        camera_view: Mat4,
        camera_projection: Mat4,
    ) {
        self.light_direction = light_direction.normalize();

        // Generate cascades
        self.cascades = generate_cascades(
            &self.cascade_config,
            self.light_direction,
            camera_view,
            camera_projection,
            self.config.shadow_map_size as f32,
        );

        // Create light view matrix
        let light_up = if self.light_direction.dot(Vec3::Y).abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };

        self.light_view_matrix = Mat4::look_at_rh(Vec3::ZERO, self.light_direction, light_up);
    }

    /// Update GPU uniforms
    pub fn update_uniforms(&self, queue: &Queue) {
        if let Some(uniform_buffer) = &self.uniform_buffer {
            // Convert cascades to GPU format
            let mut cascade_data = [CsmCascadeData {
                light_projection: [[0.0; 4]; 4],
                light_view_proj: [[0.0; 4]; 4],
                near_distance: 0.0,
                far_distance: 0.0,
                texel_size: 0.0,
            _padding: 0.0,
            }; 4];

            for (i, cascade) in self.cascades.iter().enumerate() {
                if i < 4 {
                    let light_view_proj = cascade.light_projection * self.light_view_matrix;
                    cascade_data[i] = CsmCascadeData {
                        light_projection: cascade.light_projection.to_cols_array_2d(),
                        light_view_proj: light_view_proj.to_cols_array_2d(),
                        near_distance: cascade.near_distance,
                        far_distance: cascade.far_distance,
                        texel_size: cascade.texel_size,
                        _padding: 0.0,
                    };
                }
            }

            let uniforms = CsmUniforms {
                light_direction: Vec4::new(
                    self.light_direction.x,
                    self.light_direction.y,
                    self.light_direction.z,
                    0.0,
                )
                .to_array(),
                light_view: self.light_view_matrix.to_cols_array_2d(),
                cascades: cascade_data,
                cascade_count: self.cascades.len().min(4) as u32,
                pcf_kernel_size: self.config.pcf_quality as u32,
                depth_bias: self.config.depth_bias,
                slope_bias: self.config.slope_bias,
                shadow_map_size: self.config.shadow_map_size as f32,
                debug_mode: self.config.debug_mode,
                peter_panning_offset: 0.001, // Default peter-panning offset
                pcss_light_radius: 0.0,
            };

            queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        }
    }

    /// Create bind group for shadow resources
    pub fn create_bind_group(&mut self, device: &Device, layout: &wgpu::BindGroupLayout) {
        if let (Some(uniform_buffer), Some(shadow_maps), Some(shadow_sampler)) = (
            &self.uniform_buffer,
            &self.shadow_maps,
            &self.shadow_sampler,
        ) {
            let shadow_maps_view = shadow_maps.create_view(&TextureViewDescriptor {
                label: Some("shadow_maps_array"),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(self.cascade_config.cascade_count),
            });

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("shadow_bind_group"),
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&shadow_maps_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(shadow_sampler),
                    },
                ],
            });

            self.bind_group = Some(bind_group);
        }
    }

    /// Generate shadow atlas info for debugging
    pub fn build_shadow_atlas(&self) -> ShadowAtlasInfo {
        let cascade_resolutions: Vec<u32> = self
            .cascades
            .iter()
            .map(|_| self.config.shadow_map_size)
            .collect();

        let memory_usage = self.calculate_memory_usage();

        ShadowAtlasInfo {
            cascade_count: self.cascades.len() as u32,
            atlas_dimensions: (
                self.config.shadow_map_size,
                self.config.shadow_map_size,
                self.cascades.len() as u32,
            ),
            cascade_resolutions,
            memory_usage,
        }
    }

    /// Calculate memory usage of shadow maps
    fn calculate_memory_usage(&self) -> u64 {
        let bytes_per_pixel = match self.config.depth_format {
            TextureFormat::Depth16Unorm => 2,
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => 4,
            TextureFormat::Depth32Float | TextureFormat::Depth32FloatStencil8 => 4,
            _ => 4,
        };

        let pixels_per_cascade = (self.config.shadow_map_size * self.config.shadow_map_size) as u64;
        let total_pixels = pixels_per_cascade * self.cascades.len() as u64;

        total_pixels * bytes_per_pixel
    }

    /// Get cascade views for rendering
    pub fn get_cascade_views(&self) -> &[TextureView] {
        &self.cascade_views
    }

    /// Get bind group for shadow sampling
    pub fn get_bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get number of cascades
    pub fn cascade_count(&self) -> u32 {
        self.cascades.len() as u32
    }

    /// Get cascade data
    pub fn get_cascades(&self) -> &[ShadowCascade] {
        &self.cascades
    }

    /// Get shadow map texture
    pub fn get_shadow_maps(&self) -> Option<&Texture> {
        self.shadow_maps.as_ref()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ShadowMappingConfig) {
        self.config = config;
    }

    /// Update cascade configuration  
    pub fn update_cascade_config(&mut self, config: CascadeSplitConfig) {
        self.cascade_config = config;
    }

    /// Check if shadow mapping is properly initialized
    pub fn is_initialized(&self) -> bool {
        self.shadow_maps.is_some() && self.shadow_sampler.is_some() && self.uniform_buffer.is_some()
    }
}

/// Create default shadow mapping bind group layout
pub fn create_shadow_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_bind_group_layout"),
        entries: &[
            // CSM uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<CsmUniforms>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            },
            // Shadow map texture array
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    sample_type: wgpu::TextureSampleType::Depth,
                },
                count: None,
            },
            // Shadow sampler
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_mapping_creation() {
        let config = ShadowMappingConfig::default();
        let cascade_config = CascadeSplitConfig::default();

        let shadow_mapping = ShadowMapping::new(config, cascade_config);

        assert!(!shadow_mapping.is_initialized());
        assert_eq!(shadow_mapping.cascade_count(), 0);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let config = ShadowMappingConfig {
            shadow_map_size: 512,
            depth_format: TextureFormat::Depth24Plus,
            ..Default::default()
        };

        let cascade_config = CascadeSplitConfig {
            cascade_count: 3,
            ..Default::default()
        };

        let mut shadow_mapping = ShadowMapping::new(config, cascade_config);

        // Simulate having cascades
        shadow_mapping.cascades = vec![
            ShadowCascade {
                near_distance: 0.1,
                far_distance: 10.0,
                light_projection: Mat4::IDENTITY,
                texel_size: 0.01,
                cascade_index: 0,
            };
            3
        ];

        let memory_usage = shadow_mapping.calculate_memory_usage();

        // 512x512 pixels, 4 bytes per pixel, 3 cascades
        let expected = 512 * 512 * 4 * 3;
        assert_eq!(memory_usage, expected);
    }
}
