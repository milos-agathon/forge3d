// src/core/cloud_shadows.rs
// Cloud shadow overlay system for B7 - 2D shadow texture modulation over terrain
// RELEVANT FILES: shaders/cloud_shadows.wgsl, src/scene/mod.rs, examples/cloud_shadows_demo.py

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, ComputePipelineDescriptor,
    Device, Extent3d, FilterMode, PipelineLayoutDescriptor, Queue, Sampler, SamplerDescriptor,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

/// Cloud shadow parameters matching WGSL CloudShadowUniforms structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CloudShadowUniforms {
    // Cloud movement parameters
    pub cloud_speed: [f32; 2], // Cloud movement speed (x, y)
    pub time: f32,             // Current time for animation
    pub cloud_scale: f32,      // Scale of cloud patterns

    // Cloud appearance parameters
    pub cloud_density: f32,    // Base cloud density [0, 1]
    pub cloud_coverage: f32,   // Cloud coverage amount [0, 1]
    pub shadow_intensity: f32, // Shadow strength [0, 1]
    pub shadow_softness: f32,  // Shadow edge softness

    // Texture parameters
    pub texture_size: [f32; 2],     // Size of cloud shadow texture
    pub inv_texture_size: [f32; 2], // 1.0 / texture_size

    // Noise parameters
    pub noise_octaves: u32,   // Number of noise octaves
    pub noise_frequency: f32, // Base noise frequency
    pub noise_amplitude: f32, // Noise amplitude
    pub wind_direction: f32,  // Wind direction in radians

    // Debug and visualization
    pub debug_mode: u32,       // Debug visualization mode
    pub show_clouds_only: u32, // Show only cloud patterns
    pub _padding: [f32; 2],    // Padding for alignment
}

impl Default for CloudShadowUniforms {
    fn default() -> Self {
        Self {
            cloud_speed: [0.02, 0.01], // Slow drift
            time: 0.0,
            cloud_scale: 2.0, // Medium scale clouds

            cloud_density: 0.6,    // Moderate density
            cloud_coverage: 0.4,   // 40% coverage
            shadow_intensity: 0.7, // Strong shadows
            shadow_softness: 0.3,  // Soft edges

            texture_size: [512.0, 512.0], // Default texture size
            inv_texture_size: [1.0 / 512.0, 1.0 / 512.0],

            noise_octaves: 4,     // 4 octaves of noise
            noise_frequency: 1.0, // Base frequency
            noise_amplitude: 1.0, // Full amplitude
            wind_direction: 0.0,  // No initial wind

            debug_mode: 0,       // Normal rendering
            show_clouds_only: 0, // Show shadows
            _padding: [0.0; 2],
        }
    }
}

/// Cloud shadow quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudShadowQuality {
    /// Low quality: 256x256 texture, 3 octaves
    Low,
    /// Medium quality: 512x512 texture, 4 octaves
    Medium,
    /// High quality: 1024x1024 texture, 5 octaves
    High,
    /// Ultra quality: 2048x2048 texture, 6 octaves
    Ultra,
}

impl CloudShadowQuality {
    /// Get texture size for this quality setting
    pub fn texture_size(self) -> u32 {
        match self {
            CloudShadowQuality::Low => 256,
            CloudShadowQuality::Medium => 512,
            CloudShadowQuality::High => 1024,
            CloudShadowQuality::Ultra => 2048,
        }
    }

    /// Get noise octaves for this quality
    pub fn noise_octaves(self) -> u32 {
        match self {
            CloudShadowQuality::Low => 3,
            CloudShadowQuality::Medium => 4,
            CloudShadowQuality::High => 5,
            CloudShadowQuality::Ultra => 6,
        }
    }
}

/// Cloud shadow animation parameters
#[derive(Debug, Clone, Copy)]
pub struct CloudAnimationParams {
    pub speed: Vec2,         // Cloud movement speed
    pub wind_direction: f32, // Wind direction in radians
    pub wind_strength: f32,  // Wind strength multiplier
    pub turbulence: f32,     // Amount of turbulence
}

impl Default for CloudAnimationParams {
    fn default() -> Self {
        Self {
            speed: Vec2::new(0.02, 0.01),
            wind_direction: 0.0,
            wind_strength: 1.0,
            turbulence: 0.1,
        }
    }
}

/// Cloud shadow renderer
pub struct CloudShadowRenderer {
    /// Configuration
    pub uniforms: CloudShadowUniforms,
    /// Uniform buffer
    pub uniform_buffer: Buffer,
    /// Cloud shadow texture
    pub shadow_texture: Texture,
    /// Cloud shadow texture view for reading
    pub shadow_view: TextureView,
    /// Cloud shadow texture view for compute writing
    pub shadow_storage_view: TextureView,
    /// Sampler for texture sampling
    pub sampler: Sampler,
    /// Compute pipeline for cloud generation
    pub compute_pipeline: ComputePipeline,
    /// Bind group for resources
    pub bind_group: Option<BindGroup>,
    /// Current quality setting
    pub quality: CloudShadowQuality,
    /// Animation parameters
    pub animation_params: CloudAnimationParams,
    /// Current time for animation
    pub current_time: f32,
}

impl CloudShadowRenderer {
    /// Create a new cloud shadow renderer
    pub fn new(device: &Device, quality: CloudShadowQuality) -> Self {
        let texture_size = quality.texture_size();

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_shadow_uniforms"),
            size: std::mem::size_of::<CloudShadowUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create cloud shadow texture
        let shadow_texture = device.create_texture(&TextureDescriptor {
            label: Some("cloud_shadow_texture"),
            size: Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create texture views
        let shadow_view = shadow_texture.create_view(&TextureViewDescriptor::default());
        let shadow_storage_view = shadow_texture.create_view(&TextureViewDescriptor::default());

        // Create sampler
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_shadow_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            compare: None,
            ..Default::default()
        });

        // Load shader and create pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud_shadow_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cloud_shadows.wgsl").into(),
            ),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_shadow_bind_group_layout"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_shadow_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("cloud_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_generate_cloud_shadows",
        });

        let mut uniforms = CloudShadowUniforms::default();
        uniforms.texture_size = [texture_size as f32, texture_size as f32];
        uniforms.inv_texture_size = [1.0 / texture_size as f32, 1.0 / texture_size as f32];
        uniforms.noise_octaves = quality.noise_octaves();

        Self {
            uniforms,
            uniform_buffer,
            shadow_texture,
            shadow_view,
            shadow_storage_view,
            sampler,
            compute_pipeline,
            bind_group: None,
            quality,
            animation_params: CloudAnimationParams::default(),
            current_time: 0.0,
        }
    }

    /// Set cloud movement speed
    pub fn set_cloud_speed(&mut self, speed: Vec2) {
        self.uniforms.cloud_speed = [speed.x, speed.y];
        self.animation_params.speed = speed;
    }

    /// Set cloud scale
    pub fn set_cloud_scale(&mut self, scale: f32) {
        self.uniforms.cloud_scale = scale.max(0.1);
    }

    /// Set cloud density
    pub fn set_cloud_density(&mut self, density: f32) {
        self.uniforms.cloud_density = density.clamp(0.0, 1.0);
    }

    /// Set cloud coverage
    pub fn set_cloud_coverage(&mut self, coverage: f32) {
        self.uniforms.cloud_coverage = coverage.clamp(0.0, 1.0);
    }

    /// Set shadow intensity
    pub fn set_shadow_intensity(&mut self, intensity: f32) {
        self.uniforms.shadow_intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set shadow softness
    pub fn set_shadow_softness(&mut self, softness: f32) {
        self.uniforms.shadow_softness = softness.clamp(0.0, 1.0);
    }

    /// Set wind parameters
    pub fn set_wind(&mut self, direction: f32, strength: f32) {
        self.uniforms.wind_direction = direction;
        self.animation_params.wind_direction = direction;
        self.animation_params.wind_strength = strength.max(0.0);
    }

    /// Set noise parameters
    pub fn set_noise_params(&mut self, frequency: f32, amplitude: f32) {
        self.uniforms.noise_frequency = frequency.max(0.1);
        self.uniforms.noise_amplitude = amplitude.max(0.0);
    }

    /// Set debug mode
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.uniforms.debug_mode = mode;
    }

    /// Enable/disable clouds-only view
    pub fn set_show_clouds_only(&mut self, show: bool) {
        self.uniforms.show_clouds_only = if show { 1 } else { 0 };
    }

    /// Update animation time
    pub fn update(&mut self, delta_time: f32) {
        self.current_time += delta_time;
        self.uniforms.time = self.current_time;

        // Apply wind effects to cloud speed
        let wind_effect = Vec2::new(
            self.animation_params.wind_direction.cos(),
            self.animation_params.wind_direction.sin(),
        ) * self.animation_params.wind_strength
            * delta_time
            * 0.01;

        let total_speed = self.animation_params.speed + wind_effect;
        self.uniforms.cloud_speed = [total_speed.x, total_speed.y];
    }

    /// Upload uniform data to GPU
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Create bind group for cloud shadow resources
    pub fn create_bind_group(&mut self, device: &Device) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_shadow_bind_group"),
            layout: &self.compute_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.shadow_storage_view),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }

    /// Generate cloud shadow texture
    pub fn generate_shadows(&self, encoder: &mut CommandEncoder) {
        let Some(ref bind_group) = self.bind_group else {
            return; // No bind group created
        };

        let texture_size = self.quality.texture_size();
        let workgroup_count_x = (texture_size + 7) / 8;
        let workgroup_count_y = (texture_size + 7) / 8;

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloud_shadow_generation_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    /// Get cloud shadow texture
    pub fn shadow_texture(&self) -> &Texture {
        &self.shadow_texture
    }

    /// Get cloud shadow texture view
    pub fn shadow_view(&self) -> &TextureView {
        &self.shadow_view
    }

    /// Get cloud shadow sampler
    pub fn shadow_sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// Resize cloud shadow texture
    pub fn resize(&mut self, device: &Device, new_quality: CloudShadowQuality) {
        if new_quality as u8 == self.quality as u8 {
            return; // No change needed
        }

        self.quality = new_quality;
        let texture_size = new_quality.texture_size();

        // Update uniforms
        self.uniforms.texture_size = [texture_size as f32, texture_size as f32];
        self.uniforms.inv_texture_size = [1.0 / texture_size as f32, 1.0 / texture_size as f32];
        self.uniforms.noise_octaves = new_quality.noise_octaves();

        // Recreate shadow texture
        self.shadow_texture = device.create_texture(&TextureDescriptor {
            label: Some("cloud_shadow_texture"),
            size: Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Recreate views
        self.shadow_view = self
            .shadow_texture
            .create_view(&TextureViewDescriptor::default());
        self.shadow_storage_view = self
            .shadow_texture
            .create_view(&TextureViewDescriptor::default());

        // Clear bind group to force recreation
        self.bind_group = None;
    }

    /// Get current animation parameters
    pub fn animation_params(&self) -> CloudAnimationParams {
        self.animation_params
    }

    /// Set animation parameters
    pub fn set_animation_params(&mut self, params: CloudAnimationParams) {
        self.animation_params = params;
        self.uniforms.cloud_speed = [params.speed.x, params.speed.y];
        self.uniforms.wind_direction = params.wind_direction;
    }

    /// Get current cloud parameters for external use
    pub fn get_cloud_params(&self) -> (f32, f32, f32, f32) {
        (
            self.uniforms.cloud_density,
            self.uniforms.cloud_coverage,
            self.uniforms.shadow_intensity,
            self.uniforms.shadow_softness,
        )
    }

    /// Get WGSL shader source
    pub fn shader_source() -> &'static str {
        include_str!("../shaders/cloud_shadows.wgsl")
    }
}

/// Utility functions for cloud shadow calculations
pub mod utils {
    use super::*;

    /// Calculate optimal cloud speed based on wind parameters
    pub fn calculate_cloud_speed(wind_direction: f32, wind_strength: f32, base_speed: f32) -> Vec2 {
        Vec2::new(
            wind_direction.cos() * wind_strength * base_speed,
            wind_direction.sin() * wind_strength * base_speed,
        )
    }

    /// Calculate cloud density based on coverage and base density
    pub fn calculate_effective_density(base_density: f32, coverage: f32) -> f32 {
        (base_density * coverage).clamp(0.0, 1.0)
    }

    /// Calculate shadow intensity based on sun elevation
    pub fn calculate_shadow_intensity(base_intensity: f32, sun_elevation: f32) -> f32 {
        // Shadows are stronger when sun is higher
        let elevation_factor = (sun_elevation.to_radians().sin()).max(0.0);
        (base_intensity * elevation_factor).clamp(0.0, 1.0)
    }

    /// Create cloud animation preset
    pub fn create_animation_preset(preset_name: &str) -> CloudAnimationParams {
        match preset_name {
            "calm" => CloudAnimationParams {
                speed: Vec2::new(0.005, 0.003),
                wind_direction: 0.0,
                wind_strength: 0.2,
                turbulence: 0.05,
            },
            "windy" => CloudAnimationParams {
                speed: Vec2::new(0.02, 0.01),
                wind_direction: 45.0_f32.to_radians(),
                wind_strength: 1.5,
                turbulence: 0.2,
            },
            "stormy" => CloudAnimationParams {
                speed: Vec2::new(0.05, 0.03),
                wind_direction: 180.0_f32.to_radians(),
                wind_strength: 3.0,
                turbulence: 0.5,
            },
            _ => CloudAnimationParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_shadow_uniforms_size() {
        // Ensure uniforms match WGSL layout
        assert_eq!(std::mem::size_of::<CloudShadowUniforms>(), 72); // Expected size for alignment
    }

    #[test]
    fn test_quality_settings() {
        assert_eq!(CloudShadowQuality::Low.texture_size(), 256);
        assert_eq!(CloudShadowQuality::Medium.texture_size(), 512);
        assert_eq!(CloudShadowQuality::High.texture_size(), 1024);
        assert_eq!(CloudShadowQuality::Ultra.texture_size(), 2048);

        assert_eq!(CloudShadowQuality::Low.noise_octaves(), 3);
        assert_eq!(CloudShadowQuality::Medium.noise_octaves(), 4);
        assert_eq!(CloudShadowQuality::High.noise_octaves(), 5);
        assert_eq!(CloudShadowQuality::Ultra.noise_octaves(), 6);
    }

    #[test]
    fn test_cloud_speed_calculation() {
        let speed = utils::calculate_cloud_speed(0.0, 1.0, 0.02);
        assert!((speed.x - 0.02).abs() < 0.001);
        assert!(speed.y.abs() < 0.001);

        let speed = utils::calculate_cloud_speed(90.0_f32.to_radians(), 1.0, 0.02);
        assert!(speed.x.abs() < 0.001);
        assert!((speed.y - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_animation_presets() {
        let calm = utils::create_animation_preset("calm");
        assert!(calm.speed.length() < 0.01);
        assert!(calm.wind_strength < 0.5);

        let stormy = utils::create_animation_preset("stormy");
        assert!(stormy.speed.length() > 0.05);
        assert!(stormy.wind_strength > 2.0);
    }
}
