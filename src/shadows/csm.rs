// src/shadows/csm.rs
// Cascaded Shadow Maps implementation with 3-4 cascades and PCF/EVSM filtering
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

use crate::lighting::types::ShadowTechnique;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::{
    AddressMode, BindGroup, Buffer, BufferDescriptor, BufferUsages, CompareFunction, Device,
    Extent3d, FilterMode, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

/// Shadow cascade configuration for a single cascade level
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowCascade {
    /// Light-space projection matrix for this cascade
    pub light_projection: [f32; 16],
    /// Near plane distance in view space
    pub near_distance: f32,
    /// Far plane distance in view space
    pub far_distance: f32,
    /// Texel size in world space for this cascade
    pub texel_size: f32,
    /// Padding for alignment
    pub _padding: f32,
}

impl ShadowCascade {
    /// Create a new shadow cascade
    pub fn new(near: f32, far: f32, light_projection: Mat4, texel_size: f32) -> Self {
        Self {
            light_projection: light_projection.to_cols_array(),
            near_distance: near,
            far_distance: far,
            texel_size,
            _padding: 0.0,
        }
    }

    /// Get the projection matrix as Mat4
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::from_cols_array(&self.light_projection)
    }
}

/// CSM configuration and uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CsmUniforms {
    /// Directional light direction in world space
    pub light_direction: [f32; 4],
    /// Light view matrix
    pub light_view: [f32; 16],
    /// Shadow cascades (up to 4)
    pub cascades: [ShadowCascade; 4],
    /// Number of active cascades
    pub cascade_count: u32,
    /// PCF kernel size: 1=none, 3=3x3, 5=5x5, 7=poisson
    pub pcf_kernel_size: u32,
    /// Base depth bias to prevent shadow acne
    pub depth_bias: f32,
    /// Slope-scaled bias factor
    pub slope_bias: f32,
    /// Shadow map resolution
    pub shadow_map_size: f32,
    /// Debug visualization mode
    pub debug_mode: u32,
    /// EVSM positive exponent
    pub evsm_positive_exp: f32,
    /// EVSM negative exponent
    pub evsm_negative_exp: f32,
    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,
    /// Enable unclipped depth where supported (B17)
    pub enable_unclipped_depth: u32,
    /// Depth clipping distance factor for cascade adjustment
    pub depth_clip_factor: f32,
    /// Active shadow technique identifier
    pub technique: u32,
    /// Technique feature flags (bitmask)
    pub technique_flags: u32,
    /// Primary technique parameters (pcss radius/filter, moment bias, light size)
    pub technique_params: [f32; 4],
    /// Reserved for future expansions (e.g., MSM tuning)
    pub technique_reserved: [f32; 4],
    /// Cascade blend range (0.0 = no blend, 0.1 = 10% blend at boundaries)
    pub cascade_blend_range: f32,
    /// Padding for alignment
    pub _padding2: [f32; 3],
}

impl Default for CsmUniforms {
    fn default() -> Self {
        Self {
            light_direction: [0.0, -1.0, 0.0, 0.0],
            light_view: Mat4::IDENTITY.to_cols_array(),
            cascades: [ShadowCascade {
                light_projection: Mat4::IDENTITY.to_cols_array(),
                near_distance: 0.0,
                far_distance: 100.0,
                texel_size: 0.1,
                _padding: 0.0,
            }; 4],
            cascade_count: 3,
            pcf_kernel_size: 3,
            depth_bias: 0.005,
            slope_bias: 0.01,
            shadow_map_size: 2048.0,
            debug_mode: 0,
            evsm_positive_exp: 40.0,
            evsm_negative_exp: 5.0,
            peter_panning_offset: 0.001,
            enable_unclipped_depth: 0, // Disabled by default, enabled when supported
            depth_clip_factor: 1.0,    // Default factor for depth clipping
            technique: ShadowTechnique::PCF.as_u32(),
            technique_flags: 0,
            technique_params: [0.0; 4],
            technique_reserved: [0.0; 4],
            cascade_blend_range: 0.0,
            _padding2: [0.0; 3],
        }
    }
}

/// CSM configuration parameters
#[derive(Debug, Clone)]
pub struct CsmConfig {
    /// Number of cascades (3 or 4)
    pub cascade_count: u32,
    /// Shadow map resolution per cascade
    pub shadow_map_size: u32,
    /// Maximum shadow distance
    pub max_shadow_distance: f32,
    /// Cascade split distances (if empty, calculated automatically)
    pub cascade_splits: Vec<f32>,
    /// PCF kernel size
    pub pcf_kernel_size: u32,
    /// Base depth bias
    pub depth_bias: f32,
    /// Slope-scaled bias
    pub slope_bias: f32,
    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,
    /// Enable EVSM filtering
    pub enable_evsm: bool,
    /// EVSM positive exponent (typical: 20-80)
    pub evsm_positive_exp: f32,
    /// EVSM negative exponent (typical: 20-80)
    pub evsm_negative_exp: f32,
    /// Debug visualization mode
    pub debug_mode: u32,
    /// Enable unclipped depth (B17)
    pub enable_unclipped_depth: bool,
    /// Depth clipping distance factor
    pub depth_clip_factor: f32,
    /// Enable cascade stabilization (texel snapping)
    pub stabilize_cascades: bool,
    /// Cascade blend range (0.0 = no blend, 0.1 = 10% blend at boundaries)
    pub cascade_blend_range: f32,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self {
            cascade_count: 3,
            shadow_map_size: 2048,
            max_shadow_distance: 200.0,
            cascade_splits: vec![],
            pcf_kernel_size: 3,
            depth_bias: 0.005,
            slope_bias: 0.01,
            peter_panning_offset: 0.001,
            enable_evsm: false,
            evsm_positive_exp: 40.0,
            evsm_negative_exp: 40.0,
            debug_mode: 0,
            enable_unclipped_depth: false, // Will be enabled based on hardware support
            depth_clip_factor: 1.0,
            stabilize_cascades: true,      // Enable by default to prevent shimmer
            cascade_blend_range: 0.0,      // No blend by default (can be enabled for smoother transitions)
        }
    }
}

/// Statistics for cascade performance monitoring (B17)
#[derive(Debug, Clone)]
pub struct CascadeStatistics {
    /// Total texel area covered by all cascades
    pub total_texel_area: f32,
    /// Total depth range covered by all cascades
    pub depth_range_coverage: f32,
    /// Number of overlapping cascade transitions
    pub cascade_overlaps: u32,
    /// Whether unclipped depth is enabled
    pub unclipped_depth_enabled: bool,
    /// Current depth clip factor
    pub depth_clip_factor: f32,
    /// Effective shadow distance with clipping factor
    pub effective_shadow_distance: f32,
}

/// Cascaded Shadow Maps renderer
#[derive(Debug)]
pub struct CsmRenderer {
    /// Configuration
    pub config: CsmConfig,
    /// Uniform data
    pub uniforms: CsmUniforms,
    /// Uniform buffer
    pub uniform_buffer: Buffer,
    /// Shadow map texture array
    pub shadow_maps: Texture,
    /// Shadow map views
    pub shadow_map_views: Vec<TextureView>,
    /// Shadow comparison sampler
    pub shadow_sampler: Sampler,
    /// EVSM moment maps (optional)
    pub evsm_maps: Option<Texture>,
    /// Shadow bind group
    pub bind_group: Option<BindGroup>,
}

impl CsmRenderer {
    /// Create a new CSM renderer
    pub fn new(device: &Device, config: CsmConfig) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("csm_uniforms"),
            size: std::mem::size_of::<CsmUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shadow map texture array
        let shadow_maps = device.create_texture(&TextureDescriptor {
            label: Some("csm_shadow_maps"),
            size: Extent3d {
                width: config.shadow_map_size,
                height: config.shadow_map_size,
                depth_or_array_layers: config.cascade_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create individual shadow map views for rendering
        let shadow_map_views: Vec<TextureView> = (0..config.cascade_count)
            .map(|i| {
                shadow_maps.create_view(&TextureViewDescriptor {
                    label: Some(&format!("csm_shadow_map_view_{}", i)),
                    format: Some(TextureFormat::Depth32Float),
                    dimension: Some(TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                })
            })
            .collect();

        // Create shadow comparison sampler
        let shadow_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("csm_shadow_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        });

        // Create EVSM maps if enabled
        let evsm_maps = if config.enable_evsm {
            Some(device.create_texture(&TextureDescriptor {
                label: Some("csm_evsm_maps"),
                size: Extent3d {
                    width: config.shadow_map_size,
                    height: config.shadow_map_size,
                    depth_or_array_layers: config.cascade_count,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            }))
        } else {
            None
        };

        let uniforms = CsmUniforms::default();

        Self {
            config,
            uniforms,
            uniform_buffer,
            shadow_maps,
            shadow_map_views,
            shadow_sampler,
            evsm_maps,
            bind_group: None,
        }
    }

    /// Update light direction and view matrix
    pub fn set_light_direction(&mut self, direction: Vec3, view_matrix: Mat4) {
        self.uniforms.light_direction =
            Vec4::new(direction.x, direction.y, direction.z, 0.0).to_array();
        self.uniforms.light_view = view_matrix.to_cols_array();
    }

    /// Calculate automatic cascade splits using practical split scheme
    pub fn calculate_cascade_splits(&self, near_plane: f32, far_plane: f32) -> Vec<f32> {
        let mut splits = Vec::with_capacity(self.config.cascade_count as usize + 1);
        splits.push(near_plane);

        let range = far_plane - near_plane;
        let ratio = far_plane / near_plane;

        // Practical Split Scheme (PSS) - blend between logarithmic and uniform
        let lambda = 0.75; // Blend factor (0.0 = uniform, 1.0 = logarithmic)

        for i in 1..self.config.cascade_count {
            let i_f = i as f32;
            let count_f = self.config.cascade_count as f32;

            // Uniform split
            let uniform_split = near_plane + (i_f / count_f) * range;

            // Logarithmic split
            let log_split = near_plane * ratio.powf(i_f / count_f);

            // Blend the two schemes
            let split = lambda * log_split + (1.0 - lambda) * uniform_split;
            splits.push(split);
        }

        splits.push(far_plane);
        splits
    }

    /// Update cascade configuration
    pub fn update_cascades(
        &mut self,
        camera_view: Mat4,
        camera_projection: Mat4,
        light_direction: Vec3,
        near_plane: f32,
        far_plane: f32,
    ) {
        // Calculate cascade splits if not provided
        let splits = if self.config.cascade_splits.is_empty() {
            self.calculate_cascade_splits(
                near_plane,
                far_plane.min(self.config.max_shadow_distance),
            )
        } else {
            self.config.cascade_splits.clone()
        };

        // Light view matrix (looking down the light direction)
        let light_up = if light_direction.y.abs() > 0.99 {
            Vec3::X // Use X as up vector if light is nearly vertical
        } else {
            Vec3::Y
        };
        let light_view = Mat4::look_to_rh(Vec3::ZERO, light_direction, light_up);

        // Update uniforms
        self.set_light_direction(light_direction, light_view);
        self.uniforms.cascade_count = self.config.cascade_count;
        self.uniforms.pcf_kernel_size = self.config.pcf_kernel_size;
        self.uniforms.depth_bias = self.config.depth_bias;
        self.uniforms.slope_bias = self.config.slope_bias;
        self.uniforms.shadow_map_size = self.config.shadow_map_size as f32;
        self.uniforms.debug_mode = self.config.debug_mode;
        self.uniforms.evsm_positive_exp = self.config.evsm_positive_exp;
        self.uniforms.evsm_negative_exp = self.config.evsm_negative_exp;
        self.uniforms.peter_panning_offset = self.config.peter_panning_offset;
        self.uniforms.cascade_blend_range = self.config.cascade_blend_range;

        // Calculate frustum corners for each cascade
        let inv_view_proj = (camera_projection * camera_view).inverse();

        for i in 0..self.config.cascade_count as usize {
            let near_dist = splits[i];
            let far_dist = splits[i + 1];

            // Calculate frustum corners in world space
            let frustum_corners = calculate_frustum_corners(
                inv_view_proj,
                near_dist / far_plane, // Normalized depth
                far_dist / far_plane,
            );

            // Transform corners to light space
            let mut light_space_corners = Vec::new();
            for corner in &frustum_corners {
                let light_space_pos = light_view * corner.extend(1.0);
                light_space_corners.push(light_space_pos.truncate());
            }

            // Calculate AABB in light space
            let mut min_bounds = light_space_corners[0];
            let mut max_bounds = light_space_corners[0];

            for corner in &light_space_corners[1..] {
                min_bounds = min_bounds.min(*corner);
                max_bounds = max_bounds.max(*corner);
            }

            // Expand bounds slightly to prevent edge cases
            let expand = 0.01;
            min_bounds -= Vec3::splat(expand);
            max_bounds += Vec3::splat(expand);

            // Calculate texel size for stable sampling
            let world_units_per_texel =
                (max_bounds.x - min_bounds.x) / self.config.shadow_map_size as f32;

            // TEXEL SNAPPING: Snap bounds to texel grid in world space
            // This prevents shimmering when camera moves by keeping shadow map
            // pixels aligned to world space coordinates
            if self.config.stabilize_cascades {
                // Round min bounds to nearest texel boundary
                min_bounds.x = (min_bounds.x / world_units_per_texel).floor() * world_units_per_texel;
                min_bounds.y = (min_bounds.y / world_units_per_texel).floor() * world_units_per_texel;
                
                // Recalculate max bounds to maintain exact shadow map size
                max_bounds.x = min_bounds.x + world_units_per_texel * self.config.shadow_map_size as f32;
                max_bounds.y = min_bounds.y + world_units_per_texel * self.config.shadow_map_size as f32;
            }

            // Create orthographic projection for this cascade
            let light_projection = Mat4::orthographic_rh(
                min_bounds.x,
                max_bounds.x,
                min_bounds.y,
                max_bounds.y,
                -max_bounds.z - 50.0, // Extended near plane to catch occluders
                -min_bounds.z,
            );

            // Create cascade
            self.uniforms.cascades[i] =
                ShadowCascade::new(near_dist, far_dist, light_projection, world_units_per_texel);
        }
    }

    /// Upload uniform data to GPU
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Get the full shadow texture view for binding
    pub fn shadow_texture_view(&self) -> TextureView {
        self.shadow_maps.create_view(&TextureViewDescriptor {
            label: Some("csm_full_shadow_view"),
            format: Some(TextureFormat::Depth32Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(self.config.cascade_count),
        })
    }

    /// Get optional view into the EVSM/VSM moment texture array
    pub fn moment_texture_view(&self) -> Option<TextureView> {
        self.evsm_maps.as_ref().map(|texture| {
            texture.create_view(&TextureViewDescriptor {
                label: Some("csm_moment_texture_view"),
                format: Some(TextureFormat::Rgba32Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(self.config.cascade_count),
            })
        })
    }

    /// Calculate total GPU memory used by the shadow resources
    pub fn total_memory_bytes(&self) -> u64 {
        let depth_bytes = (self.config.shadow_map_size as u64)
            * (self.config.shadow_map_size as u64)
            * (self.config.cascade_count as u64)
            * 4;

        let moment_bytes = if self.evsm_maps.is_some() {
            (self.config.shadow_map_size as u64)
                * (self.config.shadow_map_size as u64)
                * (self.config.cascade_count as u64)
                * 16
        } else {
            0
        };

        depth_bytes + moment_bytes
    }

    /// Helper to expose current shadow map resolution
    pub fn shadow_map_resolution(&self) -> u32 {
        self.config.shadow_map_size
    }

    /// Get WGSL shader source for CSM
    pub fn shader_source() -> &'static str {
        include_str!("../../shaders/shadows.wgsl")
    }

    /// Enable/disable debug visualization
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.config.debug_mode = mode;
        self.uniforms.debug_mode = mode;
    }

    /// Get cascade information for debugging
    pub fn get_cascade_info(&self, cascade_idx: usize) -> Option<(f32, f32, f32)> {
        if cascade_idx < self.config.cascade_count as usize {
            let cascade = &self.uniforms.cascades[cascade_idx];
            Some((
                cascade.near_distance,
                cascade.far_distance,
                cascade.texel_size,
            ))
        } else {
            None
        }
    }

    /// Check if peter-panning artifacts should be visible
    pub fn validate_peter_panning_prevention(&self) -> bool {
        self.uniforms.peter_panning_offset > 0.0001 && self.uniforms.depth_bias > 0.0001
    }

    // B17: Depth-clip control methods

    /// Detect hardware support for unclipped depth
    pub fn detect_unclipped_depth_support(device: &Device) -> bool {
        // Check device features for unclipped depth support
        let _features = device.features();

        // WebGPU currently doesn't directly expose unclipped depth as a feature
        // In practice, this would check for:
        // - Vulkan: depthClamp device feature
        // - D3D12: D3D12_FEATURE_D3D12_OPTIONS::DepthBoundsTestSupported
        // - Metal: Depth clipping is typically always available

        // For now, we'll assume modern hardware supports it and provide conservative fallback
        // This would be refined based on actual device capabilities in a production implementation

        // Conservative approach: enable for discrete GPUs, disable for integrated
        // This is a placeholder - actual implementation would query specific capabilities
        true // Assume support for demo purposes
    }

    /// Enable or disable unclipped depth rendering
    pub fn set_unclipped_depth_enabled(&mut self, enabled: bool, device: &Device) {
        let supported = Self::detect_unclipped_depth_support(device);

        // Only enable if hardware supports it
        self.config.enable_unclipped_depth = enabled && supported;
        self.uniforms.enable_unclipped_depth = if self.config.enable_unclipped_depth {
            1
        } else {
            0
        };

        // Adjust depth clip factor for better cascade coverage when unclipped depth is enabled
        if self.config.enable_unclipped_depth {
            // With unclipped depth, we can extend cascade coverage
            self.config.depth_clip_factor = 1.5; // Extend 50% beyond normal clip range
        } else {
            self.config.depth_clip_factor = 1.0; // Standard clipping
        }

        self.uniforms.depth_clip_factor = self.config.depth_clip_factor;
    }

    /// Check if unclipped depth is currently enabled
    pub fn is_unclipped_depth_enabled(&self) -> bool {
        self.config.enable_unclipped_depth
    }

    /// Get current depth clip factor
    pub fn get_depth_clip_factor(&self) -> f32 {
        self.config.depth_clip_factor
    }

    /// Set custom depth clip factor (for advanced tuning)
    pub fn set_depth_clip_factor(&mut self, factor: f32) {
        self.config.depth_clip_factor = factor.clamp(0.5, 3.0); // Reasonable bounds
        self.uniforms.depth_clip_factor = self.config.depth_clip_factor;
    }

    /// Retune cascades for optimal unclipped depth performance
    pub fn retune_cascades_for_unclipped_depth(&mut self) {
        if self.config.enable_unclipped_depth {
            // Adjust cascade parameters for better performance with unclipped depth

            // Reduce peter-panning offset since unclipped depth reduces artifacts
            self.config.peter_panning_offset *= 0.5;
            self.uniforms.peter_panning_offset = self.config.peter_panning_offset;

            // Slightly reduce depth bias as unclipped depth improves precision
            self.config.depth_bias *= 0.8;
            self.uniforms.depth_bias = self.config.depth_bias;

            // Adjust slope bias for better contact shadows
            self.config.slope_bias *= 0.9;
            self.uniforms.slope_bias = self.config.slope_bias;

            // Increase cascade count if performance allows (better quality)
            if self.config.cascade_count < 4 {
                self.config.cascade_count = (self.config.cascade_count + 1).min(4);
                self.uniforms.cascade_count = self.config.cascade_count;
            }
        }
    }

    /// Calculate optimal cascade splits for unclipped depth
    pub fn calculate_unclipped_cascade_splits(&self, near_plane: f32, far_plane: f32) -> Vec<f32> {
        if !self.config.enable_unclipped_depth {
            return self.calculate_cascade_splits(near_plane, far_plane);
        }

        let effective_far = far_plane * self.config.depth_clip_factor;
        let mut splits = Vec::with_capacity(self.config.cascade_count as usize + 1);
        splits.push(near_plane);

        let range = effective_far - near_plane;
        let ratio = effective_far / near_plane;

        // More aggressive split scheme for unclipped depth (favors close-up detail)
        let lambda = 0.85; // More logarithmic distribution

        for i in 1..self.config.cascade_count {
            let i_f = i as f32;
            let count_f = self.config.cascade_count as f32;

            // Uniform split
            let uniform_split = near_plane + (i_f / count_f) * range;

            // More aggressive logarithmic split for unclipped depth
            let log_split = near_plane * ratio.powf(i_f / count_f);

            // Blend with more emphasis on logarithmic
            let split = lambda * log_split + (1.0 - lambda) * uniform_split;
            splits.push(split);
        }

        splits.push(effective_far);
        splits
    }

    /// Get cascade statistics for performance monitoring
    pub fn get_cascade_statistics(&self) -> CascadeStatistics {
        let mut total_texel_area = 0.0;
        let mut depth_range_coverage = 0.0;
        let mut cascade_overlaps = 0;

        for i in 0..self.config.cascade_count as usize {
            let cascade = &self.uniforms.cascades[i];
            let texel_area = cascade.texel_size * cascade.texel_size;
            total_texel_area += texel_area;

            let range = cascade.far_distance - cascade.near_distance;
            depth_range_coverage += range;

            // Check for overlap with next cascade
            if i + 1 < self.config.cascade_count as usize {
                let next_cascade = &self.uniforms.cascades[i + 1];
                if cascade.far_distance > next_cascade.near_distance {
                    cascade_overlaps += 1;
                }
            }
        }

        CascadeStatistics {
            total_texel_area,
            depth_range_coverage,
            cascade_overlaps,
            unclipped_depth_enabled: self.config.enable_unclipped_depth,
            depth_clip_factor: self.config.depth_clip_factor,
            effective_shadow_distance: self.config.max_shadow_distance
                * self.config.depth_clip_factor,
        }
    }
}

/// Calculate frustum corners in world space for a given depth range
fn calculate_frustum_corners(inv_view_proj: Mat4, near_norm: f32, far_norm: f32) -> Vec<Vec3> {
    let mut corners = Vec::with_capacity(8);

    // NDC corners for near and far planes
    let ndc_corners = [
        // Near plane corners
        [-1.0, -1.0, near_norm],
        [1.0, -1.0, near_norm],
        [1.0, 1.0, near_norm],
        [-1.0, 1.0, near_norm],
        // Far plane corners
        [-1.0, -1.0, far_norm],
        [1.0, -1.0, far_norm],
        [1.0, 1.0, far_norm],
        [-1.0, 1.0, far_norm],
    ];

    // Transform NDC corners to world space
    for ndc in &ndc_corners {
        let world_pos = inv_view_proj * Vec4::new(ndc[0], ndc[1], ndc[2], 1.0);
        corners.push((world_pos / world_pos.w).truncate());
    }

    corners
}

/// Peter-panning detection utility
pub fn detect_peter_panning(
    shadow_factor: f32,
    surface_normal: Vec3,
    light_direction: Vec3,
) -> bool {
    let n_dot_l = surface_normal.dot(-light_direction);

    // Peter-panning occurs when shadows are cast on surfaces facing away from light
    // or when there's insufficient depth bias
    n_dot_l <= 0.01 && shadow_factor < 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_splits() {
        let config = CsmConfig::default();
        let device = crate::gpu::create_device_for_test();
        let renderer = CsmRenderer::new(&device, config);

        let splits = renderer.calculate_cascade_splits(0.1, 100.0);

        // Should have cascade_count + 1 splits
        assert_eq!(splits.len(), 4); // 3 cascades + 1

        // Splits should be monotonically increasing
        for i in 1..splits.len() {
            assert!(splits[i] > splits[i - 1]);
        }

        // First split should be near plane, last should be far plane
        assert!((splits[0] - 0.1).abs() < f32::EPSILON);
        assert!((splits[3] - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_peter_panning_detection() {
        let surface_normal = Vec3::new(0.0, 1.0, 0.0); // Upward facing
        let light_direction = Vec3::new(0.0, -1.0, 0.0); // Downward light

        // Should not detect peter-panning for properly lit surface
        assert!(!detect_peter_panning(1.0, surface_normal, light_direction));

        // Should detect peter-panning for back-facing surface in shadow
        let back_facing_normal = Vec3::new(0.0, -1.0, 0.0);
        assert!(detect_peter_panning(
            0.2,
            back_facing_normal,
            light_direction
        ));
    }
}
