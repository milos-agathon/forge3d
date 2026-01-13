// src/shadows/csm_renderer.rs
// Cascaded Shadow Maps renderer implementation
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

use super::cascade_math::{
    calculate_cascade_splits, calculate_frustum_corners, calculate_light_space_bounds,
    snap_bounds_to_texel_grid,
};
use super::csm_types::{CascadeStatistics, CsmConfig, CsmUniforms, ShadowCascade};
use glam::{Mat4, Vec3, Vec4};
use wgpu::{
    AddressMode, BindGroup, Buffer, BufferDescriptor, BufferUsages, CompareFunction, Device,
    Extent3d, FilterMode, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

/// Cascaded Shadow Maps renderer
#[derive(Debug)]
pub struct CsmRenderer {
    pub config: CsmConfig,
    pub uniforms: CsmUniforms,
    pub uniform_buffer: Buffer,
    pub shadow_maps: Texture,
    pub shadow_map_views: Vec<TextureView>,
    pub shadow_sampler: Sampler,
    pub evsm_maps: Option<Texture>,
    pub bind_group: Option<BindGroup>,
}

impl CsmRenderer {
    /// Create a new CSM renderer
    pub fn new(device: &Device, config: CsmConfig) -> Self {
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("csm_uniforms"),
            size: std::mem::size_of::<CsmUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_maps = create_shadow_map_texture(device, &config);
        let shadow_map_views = create_shadow_map_views(&shadow_maps, &config);
        let shadow_sampler = create_shadow_sampler(device);
        let evsm_maps = create_evsm_maps(device, &config);

        Self {
            config,
            uniforms: CsmUniforms::default(),
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
        calculate_cascade_splits(self.config.cascade_count, near_plane, far_plane)
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
        let splits = if self.config.cascade_splits.is_empty() {
            self.calculate_cascade_splits(
                near_plane,
                far_plane.min(self.config.max_shadow_distance),
            )
        } else {
            self.config.cascade_splits.clone()
        };

        let light_up = if light_direction.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let light_view = Mat4::look_to_rh(Vec3::ZERO, light_direction, light_up);

        self.set_light_direction(light_direction, light_view);
        self.sync_config_to_uniforms();

        let inv_view_proj = (camera_projection * camera_view).inverse();

        for i in 0..self.config.cascade_count as usize {
            let cascade = self.compute_cascade(i, &splits, far_plane, inv_view_proj, light_view);
            self.uniforms.cascades[i] = cascade;
        }
    }

    fn sync_config_to_uniforms(&mut self) {
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
    }

    fn compute_cascade(
        &self,
        idx: usize,
        splits: &[f32],
        far_plane: f32,
        inv_view_proj: Mat4,
        light_view: Mat4,
    ) -> ShadowCascade {
        let near_dist = splits[idx];
        let far_dist = splits[idx + 1];

        // Convert view distances to NDC depth for WGPU's [0,1] depth range
        // For perspective projection: ndc_z = (far * (z - near)) / (z * (far - near))
        // Using near=1.0 (matches render_shadow_passes) and far=far_plane
        let proj_near = 1.0_f32;
        let proj_far = far_plane;
        let near_ndc = (proj_far * (near_dist - proj_near)) / (near_dist * (proj_far - proj_near));
        let far_ndc = (proj_far * (far_dist - proj_near)) / (far_dist * (proj_far - proj_near));
        
        let frustum_corners =
            calculate_frustum_corners(inv_view_proj, near_ndc.clamp(0.0, 1.0), far_ndc.clamp(0.0, 1.0));

        let (mut min_bounds, mut max_bounds) =
            calculate_light_space_bounds(&frustum_corners, light_view);

        let world_units_per_texel =
            (max_bounds.x - min_bounds.x) / self.config.shadow_map_size as f32;

        if self.config.stabilize_cascades {
            (min_bounds, max_bounds) =
                snap_bounds_to_texel_grid(min_bounds, max_bounds, self.config.shadow_map_size);
        }

        // Expand bounds significantly to include shadow casters behind the camera frustum
        // For terrain, objects far from the camera can still cast shadows into the visible area
        let z_expansion = (max_bounds.z - min_bounds.z).abs().max(1000.0) * 2.0;
        let xy_expansion = (max_bounds.x - min_bounds.x).abs().max(500.0) * 0.5;
        
        let light_projection = Mat4::orthographic_rh(
            min_bounds.x - xy_expansion,
            max_bounds.x + xy_expansion,
            min_bounds.y - xy_expansion,
            max_bounds.y + xy_expansion,
            -max_bounds.z - z_expansion,
            -min_bounds.z + z_expansion,
        );

        let light_view_proj = light_projection * light_view;

        ShadowCascade::new(
            near_dist,
            far_dist,
            light_projection,
            light_view_proj,
            world_units_per_texel,
        )
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
                format: Some(TextureFormat::Rgba16Float),
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
                * 8 // Rgba16Float = 8 bytes per pixel
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
        include_str!("../shaders/shadows.wgsl")
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

    /// Get cascade statistics for performance monitoring
    pub fn get_cascade_statistics(&self) -> CascadeStatistics {
        let mut total_texel_area = 0.0;
        let mut depth_range_coverage = 0.0;
        let mut cascade_overlaps = 0;

        for i in 0..self.config.cascade_count as usize {
            let cascade = &self.uniforms.cascades[i];
            total_texel_area += cascade.texel_size * cascade.texel_size;
            depth_range_coverage += cascade.far_distance - cascade.near_distance;

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

// GPU resource creation helpers

fn create_shadow_map_texture(device: &Device, config: &CsmConfig) -> Texture {
    device.create_texture(&TextureDescriptor {
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
    })
}

fn create_shadow_map_views(shadow_maps: &Texture, config: &CsmConfig) -> Vec<TextureView> {
    (0..config.cascade_count)
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
        .collect()
}

fn create_shadow_sampler(device: &Device) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some("csm_shadow_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Nearest,
        compare: Some(CompareFunction::LessEqual),
        ..Default::default()
    })
}

fn create_evsm_maps(device: &Device, config: &CsmConfig) -> Option<Texture> {
    if config.enable_evsm {
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
            // Use Rgba16Float for moment maps (VSM/EVSM/MSM)
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        }))
    } else {
        None
    }
}
