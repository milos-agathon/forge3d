/*!
 * Cascaded Shadow Maps (CSM) implementation with PCF filtering
 *
 * Provides high-quality shadows for directional lights across large view distances
 * using cascaded shadow maps with percentage-closer filtering for soft edges.
 */

use bytemuck::Zeroable;
use glam::{Mat4, Vec3};

/// Configuration for cascaded shadow maps
#[derive(Debug, Clone)]
pub struct CsmConfig {
    /// Number of cascade levels (typically 2-4)
    pub cascade_count: u32,
    /// Shadow map resolution per cascade
    pub shadow_map_size: u32,
    /// Far plane distance for camera
    pub camera_far: f32,
    /// Near plane distance for camera  
    pub camera_near: f32,
    /// Lambda factor for cascade split scheme (0.0 = uniform, 1.0 = logarithmic)
    pub lambda: f32,
    /// Bias to prevent shadow acne
    pub depth_bias: f32,
    /// Slope-scaled bias for angled surfaces
    pub slope_bias: f32,
    /// PCF filter kernel size (1, 3, 5, or 7)
    pub pcf_kernel_size: u32,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            shadow_map_size: 2048,
            camera_far: 1000.0,
            camera_near: 0.1,
            lambda: 0.5,
            depth_bias: 0.0001,
            slope_bias: 0.001,
            pcf_kernel_size: 3,
        }
    }
}

/// Shadow cascade data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowCascade {
    /// Light-space projection matrix for this cascade
    pub light_projection: [[f32; 4]; 4],
    /// Far plane distance for this cascade
    pub far_distance: f32,
    /// Near plane distance for this cascade  
    pub near_distance: f32,
    /// Texel size in world space
    pub texel_size: f32,
    /// Padding for alignment
    pub _padding: f32,
}

/// CSM uniform buffer data sent to GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsmUniforms {
    /// Light direction in world space
    pub light_direction: [f32; 4],
    /// Light view matrix (world to light space)
    pub light_view: [[f32; 4]; 4],
    /// Shadow cascades data
    pub cascades: [ShadowCascade; 4],
    /// Number of active cascades
    pub cascade_count: u32,
    /// PCF kernel size
    pub pcf_kernel_size: u32,
    /// Depth bias
    pub depth_bias: f32,
    /// Slope-scaled bias
    pub slope_bias: f32,
    /// Shadow map texture array size
    pub shadow_map_size: f32,
    /// Debug visualization mode (0=off, 1=cascade colors)
    pub debug_mode: u32,
    /// Padding for alignment
    pub _padding: [f32; 2],
}

/// Directional light configuration for shadow casting
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    /// Light direction (normalized, pointing towards light source)
    pub direction: Vec3,
    /// Light color and intensity
    pub color: Vec3,
    /// Light intensity multiplier
    pub intensity: f32,
    /// Enable shadow casting
    pub cast_shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.3).normalize(),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 3.0,
            cast_shadows: true,
        }
    }
}

/// Camera frustum representation for cascade calculation
#[derive(Debug, Clone)]
pub struct CameraFrustum {
    /// Camera position
    pub position: Vec3,
    /// Camera forward direction
    pub forward: Vec3,
    /// Camera up direction
    pub up: Vec3,
    /// Camera right direction
    pub right: Vec3,
    /// Vertical field of view (radians)
    pub fov_y: f32,
    /// Aspect ratio (width/height)
    pub aspect: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
}

impl CameraFrustum {
    /// Create camera frustum from view and projection matrices
    pub fn from_matrices(view: &Mat4, projection: &Mat4) -> Self {
        // Extract camera position from inverse view matrix
        let inv_view = view.inverse();
        let position = Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z);

        // Extract camera directions from view matrix
        let forward = -Vec3::new(view.z_axis.x, view.z_axis.y, view.z_axis.z);
        let up = Vec3::new(view.y_axis.x, view.y_axis.y, view.y_axis.z);
        let right = Vec3::new(view.x_axis.x, view.x_axis.y, view.x_axis.z);

        // Extract FOV and aspect from projection matrix
        let fov_y = 2.0 * (1.0 / projection.y_axis.y).atan();
        let aspect = projection.y_axis.y / projection.x_axis.x;

        // Extract near/far planes from projection matrix (assuming reverse Z)
        let near = projection.w_axis.z / (projection.z_axis.z - 1.0);
        let far = projection.w_axis.z / (projection.z_axis.z + 1.0);

        Self {
            position,
            forward: forward.normalize(),
            up: up.normalize(),
            right: right.normalize(),
            fov_y,
            aspect,
            near,
            far,
        }
    }

    /// Get frustum corners at specific depth
    pub fn get_corners_at_depth(&self, depth: f32) -> [Vec3; 8] {
        let h_near = (self.fov_y * 0.5).tan() * self.near;
        let w_near = h_near * self.aspect;
        let h_far = (self.fov_y * 0.5).tan() * depth;
        let w_far = h_far * self.aspect;

        let near_center = self.position + self.forward * self.near;
        let far_center = self.position + self.forward * depth;

        [
            // Near plane corners
            near_center + self.up * h_near - self.right * w_near, // top-left
            near_center + self.up * h_near + self.right * w_near, // top-right
            near_center - self.up * h_near - self.right * w_near, // bottom-left
            near_center - self.up * h_near + self.right * w_near, // bottom-right
            // Far plane corners
            far_center + self.up * h_far - self.right * w_far, // top-left
            far_center + self.up * h_far + self.right * w_far, // top-right
            far_center - self.up * h_far - self.right * w_far, // bottom-left
            far_center - self.up * h_far + self.right * w_far, // bottom-right
        ]
    }
}

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
    /// Debug visualization enabled
    debug_visualization: bool,
}

impl CsmShadowMap {
    /// Create new CSM shadow map system
    pub fn new(device: &wgpu::Device, config: CsmConfig) -> Self {
        let cascade_count = config.cascade_count as usize;
        // Create shadow map texture array
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

        // Create depth views for each cascade
        let mut shadow_depth_views = Vec::with_capacity(config.cascade_count as usize);
        for i in 0..config.cascade_count {
            let view = shadow_maps.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("CSM Shadow Map Cascade {}", i)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: i,
                array_layer_count: Some(1),
            });
            shadow_depth_views.push(view);
        }

        // Create array view for shader sampling
        let shadow_array_view = shadow_maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some("CSM Shadow Map Array"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(config.cascade_count),
        });

        // Create shadow sampler with comparison for PCF
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("CSM Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CSM Uniforms"),
            size: std::mem::size_of::<CsmUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            config,
            light: DirectionalLight::default(),
            shadow_maps,
            shadow_depth_views,
            shadow_array_view,
            shadow_sampler,
            uniform_buffer,
            cascades: vec![ShadowCascade::zeroed(); cascade_count],
            debug_visualization: false,
        }
    }

    /// Set directional light parameters
    pub fn set_light(&mut self, light: DirectionalLight) {
        self.light = light;
    }

    /// Enable/disable debug cascade visualization
    pub fn set_debug_visualization(&mut self, enabled: bool) {
        self.debug_visualization = enabled;
    }

    /// Update shadow cascades for current camera
    pub fn update_cascades(&mut self, queue: &wgpu::Queue, camera_frustum: &CameraFrustum) {
        // Calculate cascade split distances using practical split scheme
        let mut split_distances = vec![camera_frustum.near];

        for i in 1..=self.config.cascade_count {
            let i_norm = i as f32 / self.config.cascade_count as f32;

            // Uniform split
            let uniform = camera_frustum.near + (camera_frustum.far - camera_frustum.near) * i_norm;

            // Logarithmic split
            let logarithmic =
                camera_frustum.near * (camera_frustum.far / camera_frustum.near).powf(i_norm);

            // Blend between uniform and logarithmic
            let distance = self.config.lambda * logarithmic + (1.0 - self.config.lambda) * uniform;

            split_distances.push(distance);
        }

        // Calculate light space matrices for each cascade
        let light_dir = self.light.direction.normalize();
        let light_up = if light_dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let light_right = light_dir.cross(light_up).normalize();
        let light_up = light_right.cross(light_dir).normalize();

        // Update each cascade
        for (cascade_idx, cascade) in self.cascades.iter_mut().enumerate() {
            let near_dist = split_distances[cascade_idx];
            let far_dist = split_distances[cascade_idx + 1];

            // Get frustum corners for this cascade
            let mut corners = camera_frustum.get_corners_at_depth(far_dist);

            // Update near corners
            let near_corners = camera_frustum.get_corners_at_depth(near_dist);
            corners[0..4].copy_from_slice(&near_corners[0..4]);

            // Transform corners to light space
            let mut light_space_corners = Vec::with_capacity(8);
            for corner in &corners {
                let light_space = Vec3::new(
                    corner.dot(light_right),
                    corner.dot(light_up),
                    corner.dot(light_dir),
                );
                light_space_corners.push(light_space);
            }

            // Calculate tight bounding box in light space
            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            let mut min_z = f32::INFINITY;
            let mut max_z = f32::NEG_INFINITY;

            for corner in &light_space_corners {
                min_x = min_x.min(corner.x);
                max_x = max_x.max(corner.x);
                min_y = min_y.min(corner.y);
                max_y = max_y.max(corner.y);
                min_z = min_z.min(corner.z);
                max_z = max_z.max(corner.z);
            }

            // Add padding to prevent edge sampling issues
            let padding = (max_x - min_x).max(max_y - min_y) * 0.05;
            min_x -= padding;
            max_x += padding;
            min_y -= padding;
            max_y += padding;

            // Extend depth range to include potential casters
            min_z -= (max_z - min_z) * 0.1;

            // Snap to texel boundaries to reduce shimmering
            let texel_size = (max_x - min_x) / self.config.shadow_map_size as f32;
            min_x = (min_x / texel_size).floor() * texel_size;
            max_x = (max_x / texel_size).ceil() * texel_size;
            min_y = (min_y / texel_size).floor() * texel_size;
            max_y = (max_y / texel_size).ceil() * texel_size;

            // Create orthographic projection matrix
            let ortho_projection = Mat4::orthographic_rh(min_x, max_x, min_y, max_y, min_z, max_z);

            // Create light view matrix
            let light_pos = Vec3::ZERO - light_dir * (max_z + 100.0);
            let light_view = Mat4::look_at_rh(light_pos, light_pos + light_dir, light_up);

            let light_projection = ortho_projection * light_view;

            // Update cascade data
            cascade.light_projection = light_projection.to_cols_array_2d();
            cascade.near_distance = near_dist;
            cascade.far_distance = far_dist;
            cascade.texel_size = texel_size;
        }

        // Update uniform buffer
        let uniforms = CsmUniforms {
            light_direction: [
                self.light.direction.x,
                self.light.direction.y,
                self.light.direction.z,
                0.0,
            ],
            light_view: Mat4::look_at_rh(Vec3::ZERO, self.light.direction, light_up)
                .to_cols_array_2d(),
            cascades: {
                let mut cascade_array = [ShadowCascade::zeroed(); 4];
                for (i, cascade) in self.cascades.iter().enumerate() {
                    if i < 4 {
                        cascade_array[i] = *cascade;
                    }
                }
                cascade_array
            },
            cascade_count: self.config.cascade_count,
            pcf_kernel_size: self.config.pcf_kernel_size,
            depth_bias: self.config.depth_bias,
            slope_bias: self.config.slope_bias,
            shadow_map_size: self.config.shadow_map_size as f32,
            debug_mode: if self.debug_visualization { 1 } else { 0 },
            _padding: [0.0; 2],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Get read-only access to the current shadow cascades.
    /// This is useful for downstream systems (e.g., volumetric fog)
    /// that want to reuse the light-space projection matrices without
    /// duplicating CSM logic.
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
}

/// Shadow mapping statistics and debugging info
#[derive(Debug, Clone)]
pub struct ShadowStats {
    /// Number of active cascades
    pub cascade_count: u32,
    /// Shadow map resolution per cascade  
    pub shadow_map_size: u32,
    /// Total memory usage in bytes
    pub memory_usage: u64,
    /// Light direction
    pub light_direction: Vec3,
    /// Cascade split distances
    pub split_distances: Vec<f32>,
    /// Texel sizes per cascade
    pub texel_sizes: Vec<f32>,
}

impl CsmShadowMap {
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
