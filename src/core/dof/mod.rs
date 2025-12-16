//! Realtime Depth of Field with circle-of-confusion and gather blur.
//!
//! Provides GPU-based DOF rendering with configurable quality and methods.

mod pipeline;
mod types;

pub use types::{utils, CameraDofParams, DofMethod, DofQuality, DofUniforms};

use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, Device, Extent3d, FilterMode,
    Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};

/// Depth of Field renderer.
pub struct DofRenderer {
    pub uniforms: DofUniforms,
    pub uniform_buffer: Buffer,
    pub dof_texture: Texture,
    pub dof_view: TextureView,
    pub dof_storage_view: TextureView,
    pub temp_texture: Option<Texture>,
    pub temp_view: Option<TextureView>,
    pub sampler: Sampler,
    pub gather_pipeline: ComputePipeline,
    pub separable_h_pipeline: ComputePipeline,
    pub separable_v_pipeline: ComputePipeline,
    pub bind_group: Option<BindGroup>,
    pub quality: DofQuality,
    pub method: DofMethod,
}

impl DofRenderer {
    /// Create a new DOF renderer.
    pub fn new(device: &Device, width: u32, height: u32, quality: DofQuality) -> Self {
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("dof_uniforms"),
            size: std::mem::size_of::<DofUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dof_texture = device.create_texture(&TextureDescriptor {
            label: Some("dof_output_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let dof_view = dof_texture.create_view(&TextureViewDescriptor::default());
        let dof_storage_view = dof_texture.create_view(&TextureViewDescriptor::default());

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("dof_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            compare: None,
            ..Default::default()
        });

        let (gather_pipeline, separable_h_pipeline, separable_v_pipeline) =
            pipeline::create_pipelines(device);

        let mut uniforms = DofUniforms::default();
        uniforms.screen_size = [width as f32, height as f32];
        uniforms.inv_screen_size = [1.0 / width as f32, 1.0 / height as f32];
        uniforms.quality_level = quality.level();
        uniforms.sample_count = quality.sample_count();
        uniforms.max_blur_radius = quality.max_blur_radius();

        Self {
            uniforms,
            uniform_buffer,
            dof_texture,
            dof_view,
            dof_storage_view,
            temp_texture: None,
            temp_view: None,
            sampler,
            gather_pipeline,
            separable_h_pipeline,
            separable_v_pipeline,
            bind_group: None,
            quality,
            method: DofMethod::Gather,
        }
    }

    /// Set camera DOF parameters.
    pub fn set_camera_params(&mut self, params: CameraDofParams) {
        self.uniforms.aperture = params.aperture;
        self.uniforms.focus_distance = params.focus_distance;
        self.uniforms.focal_length = params.focal_length;
    }

    /// Set aperture (f-stop reciprocal).
    pub fn set_aperture(&mut self, aperture: f32) {
        self.uniforms.aperture = aperture.max(0.001);
    }

    /// Set focus distance.
    pub fn set_focus_distance(&mut self, distance: f32) {
        self.uniforms.focus_distance = distance.max(0.1);
    }

    /// Set focal length.
    pub fn set_focal_length(&mut self, focal_length: f32) {
        self.uniforms.focal_length = focal_length.max(10.0);
    }

    /// Set bokeh rotation angle.
    pub fn set_bokeh_rotation(&mut self, rotation: f32) {
        self.uniforms.bokeh_rotation = rotation;
    }

    /// Set near/far transition ranges.
    pub fn set_transition_ranges(&mut self, near_range: f32, far_range: f32) {
        self.uniforms.near_transition_range = near_range.max(0.1);
        self.uniforms.far_transition_range = far_range.max(0.1);
    }

    /// Set CoC bias for fine-tuning.
    pub fn set_coc_bias(&mut self, bias: f32) {
        self.uniforms.coc_bias = bias;
    }

    /// Set debug mode.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.uniforms.debug_mode = mode;
    }

    /// Enable/disable CoC visualization.
    pub fn set_show_coc(&mut self, show: bool) {
        self.uniforms.show_coc = if show { 1 } else { 0 };
    }

    /// Set DOF method (gather vs separable).
    pub fn set_method(&mut self, method: DofMethod) {
        self.method = method;
    }

    /// Upload uniform data to GPU.
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Create bind group for DOF resources.
    pub fn create_bind_group(
        &mut self,
        device: &Device,
        color_texture: &TextureView,
        depth_texture: &TextureView,
        output_override: Option<&TextureView>,
    ) {
        let output_view = output_override.unwrap_or(&self.dof_storage_view);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("dof_bind_group"),
            layout: &self.gather_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(color_texture),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(depth_texture),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(output_view),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }

    /// Dispatch DOF computation.
    pub fn dispatch(&self, encoder: &mut CommandEncoder) {
        let Some(ref bind_group) = self.bind_group else {
            return;
        };

        let workgroup_count_x = (self.uniforms.screen_size[0] as u32 + 7) / 8;
        let workgroup_count_y = (self.uniforms.screen_size[1] as u32 + 7) / 8;

        match self.method {
            DofMethod::Gather => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dof_gather_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gather_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
            }
            DofMethod::Separable => {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dof_separable_h_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.separable_h_pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dof_separable_v_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.separable_v_pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
                }
            }
        }
    }

    /// Get DOF output texture.
    pub fn output_texture(&self) -> &Texture {
        &self.dof_texture
    }

    /// Get DOF output texture view.
    pub fn output_view(&self) -> &TextureView {
        &self.dof_view
    }

    /// Resize DOF textures.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.uniforms.screen_size = [width as f32, height as f32];
        self.uniforms.inv_screen_size = [1.0 / width as f32, 1.0 / height as f32];

        self.dof_texture = device.create_texture(&TextureDescriptor {
            label: Some("dof_output_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.dof_view = self
            .dof_texture
            .create_view(&TextureViewDescriptor::default());
        self.dof_storage_view = self
            .dof_texture
            .create_view(&TextureViewDescriptor::default());
        self.bind_group = None;
    }

    /// Calculate circle of confusion for a given depth.
    pub fn calculate_coc(&self, depth: f32) -> f32 {
        let distance_diff = (depth - self.uniforms.focus_distance).abs();
        let denominator = depth * (self.uniforms.focus_distance + self.uniforms.focal_length);

        if denominator < 0.001 {
            return 0.0;
        }

        let coc =
            (self.uniforms.aperture * self.uniforms.focal_length * distance_diff) / denominator;
        let coc_pixels = coc * 36.0 * self.uniforms.blur_radius_scale;
        (coc_pixels + self.uniforms.coc_bias).clamp(0.0, self.uniforms.max_blur_radius)
    }

    /// Get WGSL shader source.
    pub fn shader_source() -> &'static str {
        include_str!("../../shaders/dof.wgsl")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dof_uniforms_size() {
        assert_eq!(std::mem::size_of::<DofUniforms>(), 80);
    }

    #[test]
    fn test_quality_settings() {
        assert_eq!(DofQuality::Low.sample_count(), 8);
        assert_eq!(DofQuality::Medium.sample_count(), 16);
        assert_eq!(DofQuality::High.sample_count(), 24);
        assert_eq!(DofQuality::Ultra.sample_count(), 32);
    }

    #[test]
    fn test_f_stop_conversion() {
        let f2_8 = utils::f_stop_to_aperture(2.8);
        assert!((f2_8 - (1.0 / 2.8)).abs() < 0.001);
        let back = utils::aperture_to_f_stop(f2_8);
        assert!((back - 2.8).abs() < 0.001);
    }
}
