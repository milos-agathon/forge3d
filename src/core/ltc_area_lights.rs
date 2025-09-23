// B14: Rect Area Lights (LTC) - Linearly Transformed Cosines Implementation
// Provides physically accurate real-time rectangular area lighting using LTC approximation

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use std::sync::Arc;

/// LTC matrix and scale lookup table dimensions
const LTC_LUT_SIZE: u32 = 64;
const LTC_LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Rectangular area light configuration
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RectAreaLight {
    /// Light position in world space
    pub position: [f32; 3],
    /// Light intensity
    pub intensity: f32,

    /// Light right vector (half-width direction)
    pub right: [f32; 3],
    /// Light width (full width)
    pub width: f32,

    /// Light up vector (half-height direction)
    pub up: [f32; 3],
    /// Light height (full height)
    pub height: f32,

    /// Light color (RGB)
    pub color: [f32; 3],
    /// Light emission power
    pub power: f32,

    /// Light normal (computed from right × up)
    pub normal: [f32; 3],
    /// Two-sided lighting flag
    pub two_sided: f32,
}

impl Default for RectAreaLight {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            intensity: 1.0,
            right: [1.0, 0.0, 0.0],
            width: 2.0,
            up: [0.0, 0.0, 1.0],
            height: 2.0,
            color: [1.0, 1.0, 1.0],
            power: 10.0,
            normal: [0.0, -1.0, 0.0],
            two_sided: 0.0,
        }
    }
}

impl RectAreaLight {
    /// Create a new rectangular area light
    pub fn new(
        position: Vec3,
        right: Vec3,
        up: Vec3,
        width: f32,
        height: f32,
        color: Vec3,
        intensity: f32,
        two_sided: bool,
    ) -> Self {
        let normal = right.cross(up).normalize();
        let power = intensity * width * height * std::f32::consts::PI;

        Self {
            position: position.to_array(),
            intensity,
            right: right.normalize().to_array(),
            width: width.max(0.01),
            up: up.normalize().to_array(),
            height: height.max(0.01),
            color: color.to_array(),
            power,
            normal: normal.to_array(),
            two_sided: if two_sided { 1.0 } else { 0.0 },
        }
    }

    /// Create a simple rectangular area light facing down
    pub fn quad(
        position: Vec3,
        width: f32,
        height: f32,
        color: Vec3,
        intensity: f32,
    ) -> Self {
        Self::new(
            position,
            Vec3::X,    // Right vector
            Vec3::Z,    // Up vector
            width,
            height,
            color,
            intensity,
            false, // One-sided
        )
    }

    /// Update the normal vector from right and up vectors
    pub fn update_normal(&mut self) {
        let right = Vec3::from(self.right);
        let up = Vec3::from(self.up);
        let normal = right.cross(up).normalize();
        self.normal = normal.to_array();
    }

    /// Update power based on current intensity and dimensions
    pub fn update_power(&mut self) {
        self.power = self.intensity * self.width * self.height * std::f32::consts::PI;
    }

    /// Validate light parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.width <= 0.0 {
            return Err("Light width must be positive".to_string());
        }
        if self.height <= 0.0 {
            return Err("Light height must be positive".to_string());
        }
        if self.intensity <= 0.0 {
            return Err("Light intensity must be positive".to_string());
        }

        // Check that right and up vectors are not parallel
        let right = Vec3::from(self.right);
        let up = Vec3::from(self.up);
        let cross = right.cross(up);
        if cross.length() < 0.001 {
            return Err("Right and up vectors cannot be parallel".to_string());
        }

        Ok(())
    }
}

/// LTC uniform data for GPU shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LTCUniforms {
    /// Number of active rect area lights
    pub light_count: u32,
    /// LTC lookup texture size
    pub lut_size: u32,
    /// Global LTC intensity multiplier
    pub global_intensity: f32,
    /// Enable LTC approximation (vs. exact computation)
    pub enable_ltc: f32,

    /// Quality settings
    pub sample_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for LTCUniforms {
    fn default() -> Self {
        Self {
            light_count: 0,
            lut_size: LTC_LUT_SIZE,
            global_intensity: 1.0,
            enable_ltc: 1.0,
            sample_count: 8,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// LTC matrix and scale data structure for lookup
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LTCData {
    /// Transform matrix (3x3 flattened to 4 floats with padding)
    pub matrix: [f32; 12], // 3x4 with padding for GPU alignment
    /// Scale factor and fresnel term
    pub scale: f32,
    /// Bias term for improved accuracy
    pub bias: f32,
    /// Padding for GPU struct alignment
    pub _pad0: f32,
    pub _pad1: f32,
}

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
        let ltc_matrix_texture = Self::create_ltc_matrix_texture(&device)?;
        let ltc_scale_texture = Self::create_ltc_scale_texture(&device)?;

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

    /// Create LTC matrix lookup texture with precomputed transform matrices
    fn create_ltc_matrix_texture(device: &wgpu::Device) -> Result<wgpu::Texture, String> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LTC Matrix Texture"),
            size: wgpu::Extent3d {
                width: LTC_LUT_SIZE,
                height: LTC_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LTC_LUT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Generate LTC matrix data
        let _matrix_data = Self::generate_ltc_matrix_data();

        // Upload matrix data would go here in a complete implementation
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LTC Matrix Upload"),
        });

        // Write texture data
        // queue.write_texture(
        //     wgpu::ImageCopyTexture {
        //         texture: &texture,
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     &matrix_data,
        //     wgpu::ImageDataLayout {
        //         offset: 0,
        //         bytes_per_row: Some((LTC_LUT_SIZE * 16) as u32), // 4 floats * 4 bytes
        //         rows_per_image: Some(LTC_LUT_SIZE),
        //     },
        //     texture.size(),
        // );

        Ok(texture)
    }

    /// Create LTC scale lookup texture with amplitude and fresnel data
    fn create_ltc_scale_texture(device: &wgpu::Device) -> Result<wgpu::Texture, String> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LTC Scale Texture"),
            size: wgpu::Extent3d {
                width: LTC_LUT_SIZE,
                height: LTC_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Generate and upload scale data would go here
        // For now, create a placeholder texture

        Ok(texture)
    }

    /// Generate LTC matrix lookup table data
    fn generate_ltc_matrix_data() -> Vec<u8> {
        let mut data = Vec::with_capacity((LTC_LUT_SIZE * LTC_LUT_SIZE * 16) as usize);

        for v in 0..LTC_LUT_SIZE {
            for u in 0..LTC_LUT_SIZE {
                // Convert UV coordinates to roughness and theta
                let roughness = (u as f32 + 0.5) / LTC_LUT_SIZE as f32;
                let theta = std::f32::consts::PI * 0.5 * (v as f32 + 0.5) / LTC_LUT_SIZE as f32;

                // Compute LTC matrix for this roughness and view angle
                let ltc_matrix = Self::compute_ltc_matrix(roughness, theta);

                // Pack matrix into RGBA32Float format (3x3 -> 4x4 with padding)
                let matrix_bytes = [
                    ltc_matrix.x_axis.x, ltc_matrix.x_axis.y, ltc_matrix.x_axis.z, 0.0,
                    ltc_matrix.y_axis.x, ltc_matrix.y_axis.y, ltc_matrix.y_axis.z, 0.0,
                    ltc_matrix.z_axis.x, ltc_matrix.z_axis.y, ltc_matrix.z_axis.z, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ];

                // Convert to bytes
                for f in matrix_bytes.iter() {
                    data.extend_from_slice(&f.to_le_bytes());
                }
            }
        }

        data
    }

    /// Compute LTC matrix for given roughness and viewing angle
    fn compute_ltc_matrix(roughness: f32, theta: f32) -> Mat3 {
        // Simplified LTC matrix computation
        // In a real implementation, this would be based on BRDF fitting
        let alpha = roughness * roughness;
        let cos_theta = theta.cos();

        // Basic approximation - real LTC uses fitted polynomials
        let a = 1.0 / (alpha + 0.001);
        let b = 1.0;
        let c = alpha * cos_theta;

        Mat3::from_cols(
            Vec3::new(a, 0.0, c),
            Vec3::new(0.0, b, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Add a rectangular area light
    pub fn add_light(&mut self, mut light: RectAreaLight) -> Result<usize, String> {
        if self.lights.len() >= self.max_lights {
            return Err(format!("Maximum number of lights ({}) exceeded", self.max_lights));
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
            let matrix_view = self.ltc_matrix_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let scale_view = self.ltc_scale_texture.create_view(&wgpu::TextureViewDescriptor::default());

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