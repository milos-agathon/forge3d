// src/core/reflections.rs
// Planar Reflections implementation with render-to-texture and clip plane support (B5)
// RELEVANT FILES: shaders/planar_reflections.wgsl, python/forge3d/lighting.py, tests/test_b5_reflections.py

use glam::{Mat4, Vec3};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, FilterMode, Queue,
    RenderPass, Sampler, SamplerDescriptor, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

// Re-export types and math helpers
pub use super::reflections_math::{
    calculate_fresnel, clip_frustum_to_plane, create_reflection_matrix, distance_to_plane,
    is_above_plane, reflect_point_across_plane,
};
pub use super::reflections_types::{
    PlanarReflectionUniforms, ReflectionPlane, ReflectionQuality, REFLECTION_DISABLED,
    REFLECTION_ENABLED, REFLECTION_PASS,
};

/// Planar reflection renderer
pub struct PlanarReflectionRenderer {
    /// Configuration
    pub uniforms: PlanarReflectionUniforms,
    /// Uniform buffer
    pub uniform_buffer: Buffer,
    /// Reflection render target (color)
    pub reflection_texture: Texture,
    /// Reflection depth buffer
    pub reflection_depth: Texture,
    /// Reflection texture view for sampling
    pub reflection_view: TextureView,
    /// Reflection depth view for rendering
    pub reflection_depth_view: TextureView,
    /// Reflection render target view for rendering
    pub reflection_render_view: TextureView,
    /// Reflection sampler
    pub reflection_sampler: Sampler,
    /// Bind group for reflection resources
    pub bind_group: Option<BindGroup>,
    /// Quality setting
    pub quality: ReflectionQuality,
}

impl PlanarReflectionRenderer {
    /// Create a new planar reflection renderer
    pub fn new(device: &Device, quality: ReflectionQuality) -> Self {
        let resolution = quality.resolution();

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("planar_reflection_uniforms"),
            size: std::mem::size_of::<PlanarReflectionUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create reflection render target
        let reflection_texture = device.create_texture(&TextureDescriptor {
            label: Some("planar_reflection_texture"),
            size: Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // Use 8-bit UNORM for performance; reflection pass content does not require HDR
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create reflection depth buffer
        let reflection_depth = device.create_texture(&TextureDescriptor {
            label: Some("planar_reflection_depth"),
            size: Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create texture views
        let reflection_view = reflection_texture.create_view(&TextureViewDescriptor::default());
        let reflection_depth_view = reflection_depth.create_view(&TextureViewDescriptor::default());
        let reflection_render_view =
            reflection_texture.create_view(&TextureViewDescriptor::default());

        // Create reflection sampler with linear filtering for smooth reflections
        let reflection_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("planar_reflection_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            compare: None,
            ..Default::default()
        });

        let mut uniforms = PlanarReflectionUniforms::default();
        uniforms.reflection_resolution = resolution as f32;
        uniforms.blur_kernel_size = quality.blur_kernel_size();
        uniforms.max_blur_radius = quality.max_blur_radius();
        uniforms.camera_position = [0.0, 0.0, 0.0, 1.0];

        Self {
            uniforms,
            uniform_buffer,
            reflection_texture,
            reflection_depth,
            reflection_view,
            reflection_depth_view,
            reflection_render_view,
            reflection_sampler,
            bind_group: None,
            quality,
        }
    }

    /// Set reflection plane
    pub fn set_reflection_plane(&mut self, normal: Vec3, point: Vec3, size: Vec3) {
        self.uniforms.reflection_plane = ReflectionPlane::new(normal, point, size);
    }

    /// Update reflection camera matrices
    pub fn update_reflection_camera(
        &mut self,
        camera_pos: Vec3,
        camera_target: Vec3,
        camera_up: Vec3,
        projection: Mat4,
    ) {
        self.uniforms.reflection_plane.update_matrices(
            camera_pos,
            camera_target,
            camera_up,
            projection,
        );
        self.uniforms.camera_position = [camera_pos.x, camera_pos.y, camera_pos.z, 1.0];
    }

    /// Set reflection intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.uniforms.reflection_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set Fresnel power
    pub fn set_fresnel_power(&mut self, power: f32) {
        self.uniforms.fresnel_power = power.max(0.1);
    }

    /// Set distance fade parameters
    pub fn set_distance_fade(&mut self, start: f32, end: f32) {
        self.uniforms.distance_fade_start = start;
        self.uniforms.distance_fade_end = end.max(start);
    }

    /// Enable/disable reflections
    pub fn set_enabled(&mut self, enabled: bool) {
        self.uniforms.enable_reflections = if enabled {
            REFLECTION_ENABLED
        } else {
            REFLECTION_DISABLED
        };
    }

    /// Mark uniforms for reflection render pass (clip plane active, no sampling)
    pub fn set_reflection_pass_mode(&mut self) {
        self.uniforms.enable_reflections = REFLECTION_PASS;
    }

    /// Set debug mode
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.uniforms.debug_mode = mode;
    }

    /// Upload uniform data to GPU
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Begin reflection render pass
    pub fn begin_reflection_pass<'a>(&'a self, encoder: &'a mut CommandEncoder) -> RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("planar_reflection_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.reflection_render_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.reflection_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    /// Create bind group for reflection resources
    pub fn create_bind_group(&mut self, device: &Device, layout: &wgpu::BindGroupLayout) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("planar_reflection_bind_group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.reflection_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.reflection_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.reflection_depth_view),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }

    /// Get bind group for rendering
    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    /// Calculate estimated frame cost percentage
    pub fn estimate_frame_cost(&self) -> f32 {
        let resolution_factor = (self.quality.resolution() as f32 / 1024.0).powi(2);
        let blur_factor = self.uniforms.blur_kernel_size as f32 / 5.0;

        // Base cost is ~5% for medium quality
        let base_cost = 5.0;
        base_cost * resolution_factor * blur_factor
    }

    /// Check if estimated frame cost meets B5 requirement (≤15%)
    pub fn meets_performance_requirement(&self) -> bool {
        self.estimate_frame_cost() <= 15.0
    }

    /// Get reflection texture resolution
    pub fn resolution(&self) -> u32 {
        self.quality.resolution()
    }

    /// Get WGSL shader source
    pub fn shader_source() -> &'static str {
        include_str!("../shaders/planar_reflections.wgsl")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_matrix_creation() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let distance = 0.0;

        let reflection_matrix = create_reflection_matrix(normal, distance);
        let point = Vec3::new(1.0, 2.0, 3.0);
        let reflected = reflection_matrix.transform_point3(point);

        assert!((reflected.x - 1.0).abs() < f32::EPSILON);
        assert!((reflected.y - (-2.0)).abs() < f32::EPSILON);
        assert!((reflected.z - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reflect_point_across_plane() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let plane_normal = Vec3::new(0.0, 1.0, 0.0);
        let plane_distance = 0.0;

        let reflected = reflect_point_across_plane(point, plane_normal, plane_distance);

        assert!((reflected.x - 1.0).abs() < f32::EPSILON);
        assert!((reflected.y - (-2.0)).abs() < f32::EPSILON);
        assert!((reflected.z - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distance_to_plane() {
        let point = Vec3::new(0.0, 5.0, 0.0);
        let plane_normal = Vec3::new(0.0, 1.0, 0.0);
        let plane_distance = -2.0;

        let distance = distance_to_plane(point, plane_normal, plane_distance);
        assert!((distance - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fresnel_calculation() {
        let view_dir = Vec3::new(0.0, 1.0, 0.0);
        let surface_normal = Vec3::new(0.0, 1.0, 0.0);
        let fresnel_power = 5.0;

        let fresnel = calculate_fresnel(view_dir, surface_normal, fresnel_power);
        assert!(fresnel < 0.1);

        let grazing_view = Vec3::new(1.0, 0.1, 0.0).normalize();
        let grazing_fresnel = calculate_fresnel(grazing_view, surface_normal, fresnel_power);
        assert!(grazing_fresnel > 0.8);
    }

    #[test]
    fn test_quality_settings() {
        assert_eq!(ReflectionQuality::Low.resolution(), 512);
        assert!(
            ReflectionQuality::Ultra.blur_kernel_size() > ReflectionQuality::Low.blur_kernel_size()
        );
    }
}
