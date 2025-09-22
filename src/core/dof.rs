// src/core/dof.rs
// Realtime Depth of Field implementation with circle-of-confusion and gather blur (B6)
// RELEVANT FILES: shaders/dof.wgsl, python/forge3d/camera.py, tests/test_b6_dof.py

use bytemuck::{Pod, Zeroable};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, ComputePipelineDescriptor,
    Device, Extent3d, FilterMode, PipelineLayoutDescriptor, Queue, Sampler, SamplerDescriptor,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

/// DOF parameters matching WGSL DofUniforms structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DofUniforms {
    // Camera parameters
    pub aperture: f32,       // Aperture size (f-stop reciprocal)
    pub focus_distance: f32, // Focus distance in world units
    pub focal_length: f32,   // Camera focal length
    pub sensor_size: f32,    // Sensor size for CoC calculations

    // Quality and performance settings
    pub blur_radius_scale: f32, // Scale factor for blur radius
    pub max_blur_radius: f32,   // Maximum blur radius in pixels
    pub sample_count: u32,      // Number of samples for gather
    pub quality_level: u32,     // Quality level (0=low, 1=medium, 2=high, 3=ultra)

    // Near and far field settings
    pub near_transition_range: f32, // Transition range for near field
    pub far_transition_range: f32,  // Transition range for far field
    pub coc_bias: f32,              // CoC bias for fine-tuning
    pub bokeh_rotation: f32,        // Bokeh shape rotation

    // Screen space parameters
    pub screen_size: [f32; 2],     // Screen resolution
    pub inv_screen_size: [f32; 2], // 1.0 / screen_size

    // Debug and visualization
    pub debug_mode: u32,    // Debug visualization mode
    pub show_coc: u32,      // Show circle-of-confusion
    pub _padding: [f32; 2], // Padding for alignment
}

impl Default for DofUniforms {
    fn default() -> Self {
        Self {
            aperture: 0.1,        // f/10 equivalent
            focus_distance: 10.0, // 10 units focus distance
            focal_length: 50.0,   // 50mm focal length
            sensor_size: 36.0,    // 35mm full frame sensor

            blur_radius_scale: 1.0, // Default scale
            max_blur_radius: 16.0,  // Max 16 pixel blur
            sample_count: 16,       // 16 samples for gather
            quality_level: 1,       // Medium quality

            near_transition_range: 2.0, // Near field transition
            far_transition_range: 5.0,  // Far field transition
            coc_bias: 0.0,              // No bias
            bokeh_rotation: 0.0,        // No rotation

            screen_size: [1920.0, 1080.0],
            inv_screen_size: [1.0 / 1920.0, 1.0 / 1080.0],

            debug_mode: 0, // Normal rendering
            show_coc: 0,   // Don't show CoC
            _padding: [0.0; 2],
        }
    }
}

/// DOF quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DofQuality {
    /// Low quality: 8 samples, fast but lower quality
    Low,
    /// Medium quality: 16 samples, balanced
    Medium,
    /// High quality: 24 samples, high quality
    High,
    /// Ultra quality: 32 samples, best quality
    Ultra,
}

impl DofQuality {
    /// Get sample count for this quality setting
    pub fn sample_count(self) -> u32 {
        match self {
            DofQuality::Low => 8,
            DofQuality::Medium => 16,
            DofQuality::High => 24,
            DofQuality::Ultra => 32,
        }
    }

    /// Get quality level index
    pub fn level(self) -> u32 {
        match self {
            DofQuality::Low => 0,
            DofQuality::Medium => 1,
            DofQuality::High => 2,
            DofQuality::Ultra => 3,
        }
    }

    /// Get max blur radius for this quality
    pub fn max_blur_radius(self) -> f32 {
        match self {
            DofQuality::Low => 8.0,
            DofQuality::Medium => 12.0,
            DofQuality::High => 16.0,
            DofQuality::Ultra => 20.0,
        }
    }
}

/// DOF rendering method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DofMethod {
    /// Single-pass gather blur (higher quality)
    Gather,
    /// Two-pass separable blur (better performance)
    Separable,
}

/// Camera DOF parameters
#[derive(Debug, Clone, Copy)]
pub struct CameraDofParams {
    pub aperture: f32,         // Aperture size (f-stop reciprocal)
    pub focus_distance: f32,   // Focus distance in world units
    pub focal_length: f32,     // Camera focal length in mm
    pub auto_focus: bool,      // Enable auto-focus
    pub auto_focus_speed: f32, // Auto-focus transition speed
}

impl Default for CameraDofParams {
    fn default() -> Self {
        Self {
            aperture: 0.1,         // f/10
            focus_distance: 10.0,  // 10 units
            focal_length: 50.0,    // 50mm
            auto_focus: false,     // Manual focus
            auto_focus_speed: 2.0, // Smooth transitions
        }
    }
}

/// Depth of Field renderer
pub struct DofRenderer {
    /// Configuration
    pub uniforms: DofUniforms,
    /// Uniform buffer
    pub uniform_buffer: Buffer,
    /// DOF output texture
    pub dof_texture: Texture,
    /// DOF texture view for reading
    pub dof_view: TextureView,
    /// DOF texture view for compute writing
    pub dof_storage_view: TextureView,
    /// Temporary texture for separable blur
    pub temp_texture: Option<Texture>,
    /// Temporary texture view
    pub temp_view: Option<TextureView>,
    /// Sampler for texture sampling
    pub sampler: Sampler,
    /// Compute pipeline for gather blur
    pub gather_pipeline: ComputePipeline,
    /// Compute pipeline for separable horizontal blur
    pub separable_h_pipeline: ComputePipeline,
    /// Compute pipeline for separable vertical blur
    pub separable_v_pipeline: ComputePipeline,
    /// Bind group for resources
    pub bind_group: Option<BindGroup>,
    /// Current quality setting
    pub quality: DofQuality,
    /// Current method
    pub method: DofMethod,
}

impl DofRenderer {
    /// Create a new DOF renderer
    pub fn new(device: &Device, width: u32, height: u32, quality: DofQuality) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("dof_uniforms"),
            size: std::mem::size_of::<DofUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create DOF output texture
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
            format: TextureFormat::Rgba16Float, // HDR format for better quality
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create texture views
        let dof_view = dof_texture.create_view(&TextureViewDescriptor::default());
        let dof_storage_view = dof_texture.create_view(&TextureViewDescriptor::default());

        // Create sampler
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

        // Load shader and create pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("dof_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dof_gather_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_dof",
        });

        let separable_h_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dof_separable_h_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_dof_separable_h",
        });

        let separable_v_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dof_separable_v_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_dof_separable_v",
        });

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

    /// Set camera DOF parameters
    pub fn set_camera_params(&mut self, params: CameraDofParams) {
        self.uniforms.aperture = params.aperture;
        self.uniforms.focus_distance = params.focus_distance;
        self.uniforms.focal_length = params.focal_length;
    }

    /// Set aperture (f-stop reciprocal)
    pub fn set_aperture(&mut self, aperture: f32) {
        self.uniforms.aperture = aperture.max(0.001); // Prevent division by zero
    }

    /// Set focus distance
    pub fn set_focus_distance(&mut self, distance: f32) {
        self.uniforms.focus_distance = distance.max(0.1); // Minimum focus distance
    }

    /// Set focal length
    pub fn set_focal_length(&mut self, focal_length: f32) {
        self.uniforms.focal_length = focal_length.max(10.0); // Minimum 10mm
    }

    /// Set bokeh rotation angle
    pub fn set_bokeh_rotation(&mut self, rotation: f32) {
        self.uniforms.bokeh_rotation = rotation;
    }

    /// Set near/far transition ranges
    pub fn set_transition_ranges(&mut self, near_range: f32, far_range: f32) {
        self.uniforms.near_transition_range = near_range.max(0.1);
        self.uniforms.far_transition_range = far_range.max(0.1);
    }

    /// Set CoC bias for fine-tuning
    pub fn set_coc_bias(&mut self, bias: f32) {
        self.uniforms.coc_bias = bias;
    }

    /// Set debug mode
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.uniforms.debug_mode = mode;
    }

    /// Enable/disable CoC visualization
    pub fn set_show_coc(&mut self, show: bool) {
        self.uniforms.show_coc = if show { 1 } else { 0 };
    }

    /// Set DOF method (gather vs separable)
    pub fn set_method(&mut self, method: DofMethod) {
        self.method = method;
    }

    /// Upload uniform data to GPU
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Create bind group for DOF resources
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

    /// Dispatch DOF computation
    pub fn dispatch(&self, encoder: &mut CommandEncoder) {
        let Some(ref bind_group) = self.bind_group else {
            return; // No bind group created
        };

        let workgroup_count_x = (self.uniforms.screen_size[0] as u32 + 7) / 8;
        let workgroup_count_y = (self.uniforms.screen_size[1] as u32 + 7) / 8;

        match self.method {
            DofMethod::Gather => {
                // Single-pass gather blur
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dof_gather_pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.gather_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
            }
            DofMethod::Separable => {
                // Two-pass separable blur
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("dof_separable_h_pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.separable_h_pipeline);
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
                }

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("dof_separable_v_pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.separable_v_pipeline);
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
                }
            }
        }
    }

    /// Get DOF output texture
    pub fn output_texture(&self) -> &Texture {
        &self.dof_texture
    }

    /// Get DOF output texture view
    pub fn output_view(&self) -> &TextureView {
        &self.dof_view
    }

    /// Resize DOF textures
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        // Update uniforms
        self.uniforms.screen_size = [width as f32, height as f32];
        self.uniforms.inv_screen_size = [1.0 / width as f32, 1.0 / height as f32];

        // Recreate DOF texture
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

        // Recreate views
        self.dof_view = self
            .dof_texture
            .create_view(&TextureViewDescriptor::default());
        self.dof_storage_view = self
            .dof_texture
            .create_view(&TextureViewDescriptor::default());

        // Clear bind group to force recreation
        self.bind_group = None;
    }

    /// Calculate circle of confusion for a given depth
    pub fn calculate_coc(&self, depth: f32) -> f32 {
        let object_distance = depth;
        let distance_diff = (object_distance - self.uniforms.focus_distance).abs();
        let denominator =
            object_distance * (self.uniforms.focus_distance + self.uniforms.focal_length);

        if denominator < 0.001 {
            return 0.0;
        }

        let coc =
            (self.uniforms.aperture * self.uniforms.focal_length * distance_diff) / denominator;
        let coc_pixels = coc * 36.0 * self.uniforms.blur_radius_scale; // 36mm sensor size

        (coc_pixels + self.uniforms.coc_bias).clamp(0.0, self.uniforms.max_blur_radius)
    }

    /// Get WGSL shader source
    pub fn shader_source() -> &'static str {
        include_str!("../../shaders/dof.wgsl")
    }
}

/// Utility functions for DOF calculations
pub mod utils {
    /// Convert f-stop to aperture value
    pub fn f_stop_to_aperture(f_stop: f32) -> f32 {
        1.0 / f_stop.max(1.0)
    }

    /// Convert aperture to f-stop
    pub fn aperture_to_f_stop(aperture: f32) -> f32 {
        1.0 / aperture.max(0.001)
    }

    /// Calculate hyperfocal distance
    pub fn hyperfocal_distance(focal_length: f32, f_stop: f32, circle_of_confusion: f32) -> f32 {
        (focal_length * focal_length) / (f_stop * circle_of_confusion) + focal_length
    }

    /// Calculate depth of field range
    pub fn depth_of_field_range(
        focal_length: f32,
        f_stop: f32,
        focus_distance: f32,
        circle_of_confusion: f32,
    ) -> (f32, f32) {
        let h = hyperfocal_distance(focal_length, f_stop, circle_of_confusion);

        let near = (h * focus_distance) / (h + focus_distance - focal_length);
        let far = if focus_distance < (h - focal_length) {
            (h * focus_distance) / (h - focus_distance + focal_length)
        } else {
            f32::INFINITY
        };

        (near, far)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dof_uniforms_size() {
        // Ensure uniforms match WGSL layout
        assert_eq!(std::mem::size_of::<DofUniforms>(), 80); // Expected size for alignment
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

        let back_to_f_stop = utils::aperture_to_f_stop(f2_8);
        assert!((back_to_f_stop - 2.8).abs() < 0.001);
    }

    #[test]
    fn test_coc_calculation() {
        let mut uniforms = DofUniforms::default();
        uniforms.aperture = 0.1; // f/10
        uniforms.focus_distance = 10.0;
        uniforms.focal_length = 50.0;

        let renderer = DofRenderer {
            uniforms,
            // ... other fields would be created with device
            uniform_buffer: unsafe { std::mem::zeroed() },
            dof_texture: unsafe { std::mem::zeroed() },
            dof_view: unsafe { std::mem::zeroed() },
            dof_storage_view: unsafe { std::mem::zeroed() },
            temp_texture: None,
            temp_view: None,
            sampler: unsafe { std::mem::zeroed() },
            gather_pipeline: unsafe { std::mem::zeroed() },
            separable_h_pipeline: unsafe { std::mem::zeroed() },
            separable_v_pipeline: unsafe { std::mem::zeroed() },
            bind_group: None,
            quality: DofQuality::Medium,
            method: DofMethod::Gather,
        };

        // Test CoC at focus distance (should be near zero)
        let coc_at_focus = renderer.calculate_coc(10.0);
        assert!(coc_at_focus < 0.1);

        // Test CoC at different distances
        let coc_near = renderer.calculate_coc(5.0);
        let coc_far = renderer.calculate_coc(20.0);
        assert!(coc_near > 0.0);
        assert!(coc_far > 0.0);
    }
}
