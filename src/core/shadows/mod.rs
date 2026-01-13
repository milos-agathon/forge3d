/*!
 * Cascaded Shadow Maps (CSM) implementation with PCF filtering
 *
 * Provides high-quality shadows for directional lights across large view distances
 * using cascaded shadow maps with percentage-closer filtering for soft edges.
 */

pub mod frustum;
pub mod resources;
pub mod types;

use bytemuck::Zeroable;
use glam::{Mat4, Vec3};

pub use frustum::CameraFrustum;
pub use resources::{
    create_cascade_depth_views, create_shadow_array_view, create_shadow_sampler,
    create_uniform_buffer,
};
pub use types::{
    parse_shadow_debug_env, CsmConfig, CsmUniforms, DirectionalLight, ShadowCascade, ShadowStats,
};

/// Cascaded Shadow Map manager
pub struct CsmShadowMap {
    /// Configuration parameters
    config: CsmConfig,
    /// Directional light
    light: DirectionalLight,
    /// Shadow map texture array
    #[allow(dead_code)]
    shadow_maps: wgpu::Texture,
    /// Shadow map depth views (one per cascade)
    shadow_depth_views: Vec<wgpu::TextureView>,
    /// Combined shadow map array view for sampling
    shadow_array_view: wgpu::TextureView,
    /// Shadow sampler with PCF
    shadow_sampler: wgpu::Sampler,
    /// CSM uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Current cascade data
    cascades: Vec<ShadowCascade>,
    /// Shadow debug mode
    debug_mode: u32,
}

impl CsmShadowMap {
    /// Create new CSM shadow map system
    pub fn new(device: &wgpu::Device, config: CsmConfig) -> Self {
        let cascade_count = config.cascade_count as usize;

        let shadow_maps = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CSM Shadow Maps"),
            size: wgpu::Extent3d {
                width: config.shadow_map_size,
                height: config.shadow_map_size,
                depth_or_array_layers: config.cascade_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let shadow_depth_views = create_cascade_depth_views(&shadow_maps, &config);
        let shadow_array_view = create_shadow_array_view(&shadow_maps, &config);
        let shadow_sampler = create_shadow_sampler(device);
        let uniform_buffer = create_uniform_buffer(device);

        Self {
            config,
            light: DirectionalLight::default(),
            shadow_maps,
            shadow_depth_views,
            shadow_array_view,
            shadow_sampler,
            uniform_buffer,
            cascades: vec![ShadowCascade::zeroed(); cascade_count],
            debug_mode: parse_shadow_debug_env(),
        }
    }

    /// Set directional light parameters
    pub fn set_light(&mut self, light: DirectionalLight) {
        self.light = light;
    }

    /// Enable/disable debug cascade visualization (legacy API)
    pub fn set_debug_visualization(&mut self, enabled: bool) {
        self.debug_mode = if enabled { 1 } else { 0 };
    }

    /// Set shadow debug mode (0=off, 1=cascade overlay, 2=raw visibility)
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
    }

    /// Update shadow cascades for current camera
    pub fn update_cascades(&mut self, queue: &wgpu::Queue, camera_frustum: &CameraFrustum) {
        let split_distances = calculate_split_distances(&self.config, camera_frustum);
        let (light_up, light_right) = calculate_light_basis(&self.light);

        for (cascade_idx, cascade) in self.cascades.iter_mut().enumerate() {
            update_single_cascade(
                cascade,
                cascade_idx,
                &split_distances,
                camera_frustum,
                &self.light,
                light_up,
                light_right,
                &self.config,
            );
        }

        let uniforms = build_csm_uniforms(
            &self.config,
            &self.light,
            &self.cascades,
            light_up,
            self.debug_mode,
        );
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Get read-only access to the current shadow cascades
    pub fn cascades(&self) -> &[ShadowCascade] {
        &self.cascades
    }

    /// Get shadow map texture array view for binding
    pub fn shadow_array_view(&self) -> &wgpu::TextureView {
        &self.shadow_array_view
    }

    /// Get shadow sampler for binding
    pub fn shadow_sampler(&self) -> &wgpu::Sampler {
        &self.shadow_sampler
    }

    /// Get uniform buffer for binding
    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    /// Get depth view for specific cascade (for rendering)
    pub fn cascade_depth_view(&self, cascade_idx: usize) -> Option<&wgpu::TextureView> {
        self.shadow_depth_views.get(cascade_idx)
    }

    /// Get number of active cascades
    pub fn cascade_count(&self) -> u32 {
        self.config.cascade_count
    }

    /// Get light-space projection matrix for cascade
    pub fn cascade_projection(&self, cascade_idx: usize) -> Option<Mat4> {
        self.cascades
            .get(cascade_idx)
            .map(|c| Mat4::from_cols_array_2d(&c.light_projection))
    }

    /// Get shadow map resolution
    pub fn shadow_map_size(&self) -> u32 {
        self.config.shadow_map_size
    }

    /// Get current shadow mapping statistics
    pub fn get_stats(&self) -> ShadowStats {
        let memory_per_cascade =
            (self.config.shadow_map_size * self.config.shadow_map_size * 4) as u64;
        let total_memory = memory_per_cascade * self.config.cascade_count as u64;

        ShadowStats {
            cascade_count: self.config.cascade_count,
            shadow_map_size: self.config.shadow_map_size,
            memory_usage: total_memory,
            light_direction: self.light.direction,
            split_distances: self.cascades.iter().map(|c| c.far_distance).collect(),
            texel_sizes: self.cascades.iter().map(|c| c.texel_size).collect(),
        }
    }
}

fn calculate_split_distances(config: &CsmConfig, frustum: &CameraFrustum) -> Vec<f32> {
    let mut splits = vec![frustum.near];
    for i in 1..=config.cascade_count {
        let i_norm = i as f32 / config.cascade_count as f32;
        let uniform = frustum.near + (frustum.far - frustum.near) * i_norm;
        let logarithmic = frustum.near * (frustum.far / frustum.near).powf(i_norm);
        let distance = config.lambda * logarithmic + (1.0 - config.lambda) * uniform;
        splits.push(distance);
    }
    splits
}

fn calculate_light_basis(light: &DirectionalLight) -> (Vec3, Vec3) {
    let light_dir = light.direction.normalize();
    let light_up = if light_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let light_right = light_dir.cross(light_up).normalize();
    let light_up = light_right.cross(light_dir).normalize();
    (light_up, light_right)
}

fn update_single_cascade(
    cascade: &mut ShadowCascade,
    idx: usize,
    splits: &[f32],
    frustum: &CameraFrustum,
    light: &DirectionalLight,
    light_up: Vec3,
    light_right: Vec3,
    config: &CsmConfig,
) {
    let near_dist = splits[idx];
    let far_dist = splits[idx + 1];
    let light_dir = light.direction.normalize();

    let mut corners = frustum.get_corners_at_depth(far_dist);
    let near_corners = frustum.get_corners_at_depth(near_dist);
    corners[0..4].copy_from_slice(&near_corners[0..4]);

    let (min, max) = compute_light_space_bounds(&corners, light_right, light_up, light_dir);
    let (min_x, max_x, min_y, max_y, min_z, max_z) = add_padding_and_snap(min, max, config);

    let ortho = Mat4::orthographic_rh(min_x, max_x, min_y, max_y, min_z, max_z);
    let light_pos = Vec3::ZERO - light_dir * (max_z + 100.0);
    let light_view = Mat4::look_at_rh(light_pos, light_pos + light_dir, light_up);

    cascade.light_projection = (ortho * light_view).to_cols_array_2d();
    cascade.near_distance = near_dist;
    cascade.far_distance = far_dist;
    cascade.texel_size = (max_x - min_x) / config.shadow_map_size as f32;
}

fn compute_light_space_bounds(
    corners: &[Vec3; 8],
    right: Vec3,
    up: Vec3,
    dir: Vec3,
) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for c in corners {
        let ls = Vec3::new(c.dot(right), c.dot(up), c.dot(dir));
        min = min.min(ls);
        max = max.max(ls);
    }
    (min, max)
}

fn add_padding_and_snap(
    min: Vec3,
    max: Vec3,
    config: &CsmConfig,
) -> (f32, f32, f32, f32, f32, f32) {
    let padding = (max.x - min.x).max(max.y - min.y) * 0.05;
    let texel = (max.x - min.x + 2.0 * padding) / config.shadow_map_size as f32;
    let min_x = ((min.x - padding) / texel).floor() * texel;
    let max_x = ((max.x + padding) / texel).ceil() * texel;
    let min_y = ((min.y - padding) / texel).floor() * texel;
    let max_y = ((max.y + padding) / texel).ceil() * texel;
    let min_z = min.z - (max.z - min.z) * 0.1;
    (min_x, max_x, min_y, max_y, min_z, max.z)
}

fn build_csm_uniforms(
    config: &CsmConfig,
    light: &DirectionalLight,
    cascades: &[ShadowCascade],
    light_up: Vec3,
    debug_mode: u32,
) -> CsmUniforms {
    let mut cascade_array = [ShadowCascade::zeroed(); 4];
    for (i, c) in cascades.iter().enumerate().take(4) {
        cascade_array[i] = *c;
    }
    CsmUniforms {
        light_direction: [light.direction.x, light.direction.y, light.direction.z, 0.0],
        light_view: Mat4::look_at_rh(Vec3::ZERO, light.direction, light_up).to_cols_array_2d(),
        cascades: cascade_array,
        cascade_count: config.cascade_count,
        pcf_kernel_size: config.pcf_kernel_size,
        depth_bias: config.depth_bias,
        slope_bias: config.slope_bias,
        shadow_map_size: config.shadow_map_size as f32,
        debug_mode,
        evsm_positive_exp: 40.0,
        evsm_negative_exp: 5.0,
        peter_panning_offset: 0.001,
        enable_unclipped_depth: 0,
        depth_clip_factor: 1.0,
        technique: 1, // PCF
        technique_flags: 0,
        technique_params: [0.0; 4],
        technique_reserved: [0.0; 4],
        cascade_blend_range: 0.0,
        _padding2: [0.0; 14],
    }
}
