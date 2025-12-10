// B14: Rect Area Lights (LTC) - Linearly Transformed Cosines Implementation
// Provides physically accurate real-time rectangular area lighting using LTC approximation

use std::sync::Arc;

// Re-export types and LUT helpers
pub use super::ltc_lut::{compute_ltc_matrix, create_ltc_matrix_texture, create_ltc_scale_texture};
pub use super::ltc_types::{LTCUniforms, RectAreaLight, LTC_LUT_FORMAT, LTC_LUT_SIZE};

/// LTC rectangular area light renderer
pub struct LTCRectAreaLightRenderer {
    /// GPU device reference
    device: Arc<wgpu::Device>,
    /// Array of rect area lights
    lights: Vec<RectAreaLight>,
    /// Maximum supported lights
    max_lights: usize,
    /// GPU buffer for light data
    light_buffer: Option<wgpu::Buffer>,
    /// Uniform data buffer
    uniform_buffer: wgpu::Buffer,
    /// Current uniform data
    uniforms: LTCUniforms,
    /// LTC lookup texture (matrix data)
    ltc_matrix_texture: wgpu::Texture,
    /// LTC scale texture (amplitude/fresnel data)
    ltc_scale_texture: wgpu::Texture,
    /// Sampler for LTC lookup
    ltc_sampler: wgpu::Sampler,
    /// Bind group for LTC resources
    bind_group: Option<wgpu::BindGroup>,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl LTCRectAreaLightRenderer {
    /// Create a new LTC rect area light renderer
    pub fn new(device: Arc<wgpu::Device>, max_lights: usize) -> Result<Self, String> {
        let uniforms = LTCUniforms::default();

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LTC Uniforms"),
            size: std::mem::size_of::<LTCUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create LTC lookup textures
        let ltc_matrix_texture = create_ltc_matrix_texture(&device)?;
        let ltc_scale_texture = create_ltc_scale_texture(&device)?;

        // Create sampler for LTC lookup
        let ltc_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("LTC Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LTC Rect Area Lights"),
            entries: &[
                // Binding 0: Light data storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: LTC uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: LTC matrix texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: LTC scale texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: LTC sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Ok(Self {
            device,
            lights: Vec::new(),
            max_lights,
            light_buffer: None,
            uniform_buffer,
            uniforms,
            ltc_matrix_texture,
            ltc_scale_texture,
            ltc_sampler,
            bind_group: None,
            bind_group_layout,
        })
    }

    /// Add a rectangular area light
    pub fn add_light(&mut self, mut light: RectAreaLight) -> Result<usize, String> {
        if self.lights.len() >= self.max_lights {
            return Err(format!(
                "Maximum number of lights ({}) exceeded",
                self.max_lights
            ));
        }

        light.validate()?;
        light.update_normal();
        light.update_power();

        self.lights.push(light);
        self.uniforms.light_count = self.lights.len() as u32;

        Ok(self.lights.len() - 1)
    }

    /// Remove light by index
    pub fn remove_light(&mut self, index: usize) -> Result<(), String> {
        if index >= self.lights.len() {
            return Err("Light index out of bounds".to_string());
        }

        self.lights.remove(index);
        self.uniforms.light_count = self.lights.len() as u32;
        Ok(())
    }

    /// Update light by index
    pub fn update_light(&mut self, index: usize, mut light: RectAreaLight) -> Result<(), String> {
        if index >= self.lights.len() {
            return Err("Light index out of bounds".to_string());
        }

        light.validate()?;
        light.update_normal();
        light.update_power();

        self.lights[index] = light;
        Ok(())
    }

    /// Get light count
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Set global intensity multiplier
    pub fn set_global_intensity(&mut self, intensity: f32) {
        self.uniforms.global_intensity = intensity.max(0.0);
    }

    /// Enable or disable LTC approximation
    pub fn set_ltc_enabled(&mut self, enabled: bool) {
        self.uniforms.enable_ltc = if enabled { 1.0 } else { 0.0 };
    }

    /// Update GPU resources with current light data
    pub fn update_gpu_resources(&mut self, queue: &wgpu::Queue) -> Result<(), String> {
        // Update light buffer
        let buffer_size = (self.max_lights * std::mem::size_of::<RectAreaLight>()) as u64;

        if self.light_buffer.is_none() || buffer_size > 0 {
            self.light_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("LTC Rect Area Lights"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        if let Some(buffer) = &self.light_buffer {
            // Prepare light data with padding
            let mut buffer_data = vec![RectAreaLight::default(); self.max_lights];
            for (i, light) in self.lights.iter().enumerate() {
                if i < self.max_lights {
                    buffer_data[i] = *light;
                }
            }

            // Upload light data
            let data_bytes = bytemuck::cast_slice(&buffer_data);
            queue.write_buffer(buffer, 0, data_bytes);
        }

        // Update uniform buffer
        let binding = [self.uniforms];
        let uniform_bytes = bytemuck::cast_slice(&binding);
        queue.write_buffer(&self.uniform_buffer, 0, uniform_bytes);

        // Update bind group
        self.update_bind_group();

        Ok(())
    }

    /// Update the bind group with current resources
    fn update_bind_group(&mut self) {
        if let Some(light_buffer) = &self.light_buffer {
            let matrix_view = self
                .ltc_matrix_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let scale_view = self
                .ltc_scale_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LTC Rect Area Lights Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&matrix_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&scale_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.ltc_sampler),
                    },
                ],
            }));
        }
    }

    /// Get bind group layout for integration with other systems
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get bind group for rendering
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get current uniforms
    pub fn uniforms(&self) -> &LTCUniforms {
        &self.uniforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_area_light_creation() {
        let light = RectAreaLight::quad(
            Vec3::new(0.0, 5.0, 0.0),
            4.0,
            2.0,
            Vec3::new(1.0, 0.8, 0.6),
            15.0,
        );

        assert_eq!(light.width, 4.0);
        assert_eq!(light.height, 2.0);
        assert!(light.power > 0.0);
        assert!(light.validate().is_ok());
    }

    #[test]
    fn test_light_validation() {
        let mut light = RectAreaLight::default();
        assert!(light.validate().is_ok());

        light.width = -1.0;
        assert!(light.validate().is_err());

        light.width = 2.0;
        light.intensity = -1.0;
        assert!(light.validate().is_err());
    }

    #[test]
    fn test_ltc_matrix_generation() {
        let matrix = LTCRectAreaLightRenderer::compute_ltc_matrix(0.5, 0.7854); // 45 degrees

        // Matrix should be valid (not NaN or infinite)
        assert!(!matrix.x_axis.x.is_nan());
        assert!(!matrix.y_axis.y.is_nan());
        assert!(!matrix.z_axis.z.is_nan());
        assert!(matrix.x_axis.x.is_finite());
        assert!(matrix.y_axis.y.is_finite());
        assert!(matrix.z_axis.z.is_finite());
    }
}
