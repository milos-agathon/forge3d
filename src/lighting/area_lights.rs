//! Soft area lights with parametric penumbra control
//!
//! Implements A20 requirements for area lights with radius-controlled
//! penumbra softness and multi-light support with energy conservation.

use glam::{Vec3, Vec4, Mat4};
use wgpu::*;

/// Area light types supported by the system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AreaLightType {
    /// Rectangular area light
    Rectangle,
    /// Circular/disc area light
    Disc,
    /// Spherical area light
    Sphere,
    /// Cylindrical area light
    Cylinder,
}

/// Area light configuration
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AreaLight {
    /// Light position in world space
    pub position: [f32; 3],
    /// Light type (0=Rectangle, 1=Disc, 2=Sphere, 3=Cylinder)
    pub light_type: u32,

    /// Light direction/normal
    pub direction: [f32; 3],
    /// Light radius for penumbra control
    pub radius: f32,

    /// Light color (RGB)
    pub color: [f32; 3],
    /// Light intensity
    pub intensity: f32,

    /// Light size parameters (width, height, depth, unused)
    pub size: [f32; 4],

    /// Penumbra softness factor (0.0 = hard, 1.0 = very soft)
    pub softness: f32,
    /// Energy normalization factor
    pub energy_factor: f32,
    /// Shadow bias
    pub shadow_bias: f32,
    /// Padding for alignment
    pub _padding: f32,
}

impl Default for AreaLight {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            light_type: AreaLightType::Disc as u32,
            direction: [0.0, -1.0, 0.0],
            radius: 1.0,
            color: [1.0, 1.0, 1.0],
            intensity: 10.0,
            size: [2.0, 2.0, 0.0, 0.0],
            softness: 0.5,
            energy_factor: 1.0,
            shadow_bias: 0.001,
            _padding: 0.0,
        }
    }
}

impl AreaLight {
    /// Create new rectangular area light
    pub fn rectangle(position: Vec3, direction: Vec3, width: f32, height: f32,
                    color: Vec3, intensity: f32, radius: f32) -> Self {
        Self {
            position: position.to_array(),
            light_type: AreaLightType::Rectangle as u32,
            direction: direction.normalize().to_array(),
            radius,
            color: color.to_array(),
            intensity,
            size: [width, height, 0.0, 0.0],
            energy_factor: Self::compute_energy_factor(AreaLightType::Rectangle, radius, width * height),
            ..Default::default()
        }
    }

    /// Create new disc area light
    pub fn disc(position: Vec3, direction: Vec3, radius: f32,
               color: Vec3, intensity: f32, penumbra_radius: f32) -> Self {
        Self {
            position: position.to_array(),
            light_type: AreaLightType::Disc as u32,
            direction: direction.normalize().to_array(),
            radius: penumbra_radius,
            color: color.to_array(),
            intensity,
            size: [radius, radius, 0.0, 0.0],
            energy_factor: Self::compute_energy_factor(AreaLightType::Disc, penumbra_radius,
                                                      std::f32::consts::PI * radius * radius),
            ..Default::default()
        }
    }

    /// Create new spherical area light
    pub fn sphere(position: Vec3, radius: f32, color: Vec3, intensity: f32,
                 penumbra_radius: f32) -> Self {
        Self {
            position: position.to_array(),
            light_type: AreaLightType::Sphere as u32,
            direction: [0.0, 0.0, 0.0], // Omnidirectional
            radius: penumbra_radius,
            color: color.to_array(),
            intensity,
            size: [radius, radius, radius, 0.0],
            energy_factor: Self::compute_energy_factor(AreaLightType::Sphere, penumbra_radius,
                                                      4.0 * std::f32::consts::PI * radius * radius),
            ..Default::default()
        }
    }

    /// Compute energy normalization factor
    fn compute_energy_factor(light_type: AreaLightType, penumbra_radius: f32, base_area: f32) -> f32 {
        // Energy should remain approximately constant as penumbra radius changes
        // Larger penumbra = more spread = need higher normalization
        let penumbra_factor = 1.0 + (penumbra_radius * 0.1);

        match light_type {
            AreaLightType::Rectangle => 1.0 / (base_area * penumbra_factor),
            AreaLightType::Disc => 1.0 / (base_area * penumbra_factor),
            AreaLightType::Sphere => 1.0 / (base_area * penumbra_factor),
            AreaLightType::Cylinder => 1.0 / (base_area * penumbra_factor),
        }
    }

    /// Set penumbra softness (0.0 = hard shadows, 1.0 = very soft)
    pub fn set_softness(&mut self, softness: f32) {
        self.softness = softness.clamp(0.0, 1.0);
    }

    /// Update energy factor based on current parameters
    pub fn update_energy_factor(&mut self) {
        let area = match AreaLightType::try_from(self.light_type).unwrap_or(AreaLightType::Disc) {
            AreaLightType::Rectangle => self.size[0] * self.size[1],
            AreaLightType::Disc => std::f32::consts::PI * self.size[0] * self.size[0],
            AreaLightType::Sphere => 4.0 * std::f32::consts::PI * self.size[0] * self.size[0],
            AreaLightType::Cylinder => 2.0 * std::f32::consts::PI * self.size[0] * self.size[1],
        };

        self.energy_factor = Self::compute_energy_factor(
            AreaLightType::try_from(self.light_type).unwrap_or(AreaLightType::Disc),
            self.radius,
            area
        );
    }

    /// Validate light parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.radius <= 0.0 {
            return Err("Light radius must be positive".to_string());
        }
        if self.intensity <= 0.0 {
            return Err("Light intensity must be positive".to_string());
        }
        if self.size[0] <= 0.0 || self.size[1] <= 0.0 {
            return Err("Light size dimensions must be positive".to_string());
        }
        if self.softness < 0.0 || self.softness > 1.0 {
            return Err("Softness must be in range [0.0, 1.0]".to_string());
        }

        Ok(())
    }
}

impl TryFrom<u32> for AreaLightType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(AreaLightType::Rectangle),
            1 => Ok(AreaLightType::Disc),
            2 => Ok(AreaLightType::Sphere),
            3 => Ok(AreaLightType::Cylinder),
            _ => Err(()),
        }
    }
}

/// Area light manager for multi-light scenes
pub struct AreaLightManager {
    /// Array of area lights
    pub lights: Vec<AreaLight>,
    /// GPU buffer for light data
    light_buffer: Option<Buffer>,
    /// Maximum supported lights
    max_lights: usize,
    /// Device reference
    device: std::sync::Arc<Device>,
}

impl AreaLightManager {
    /// Create new area light manager
    pub fn new(device: std::sync::Arc<Device>, max_lights: usize) -> Self {
        Self {
            lights: Vec::new(),
            light_buffer: None,
            max_lights,
            device,
        }
    }

    /// Add area light to the scene
    pub fn add_light(&mut self, mut light: AreaLight) -> Result<usize, String> {
        if self.lights.len() >= self.max_lights {
            return Err(format!("Maximum number of lights ({}) exceeded", self.max_lights));
        }

        light.validate()?;
        light.update_energy_factor();

        self.lights.push(light);
        Ok(self.lights.len() - 1)
    }

    /// Remove light by index
    pub fn remove_light(&mut self, index: usize) -> Result<(), String> {
        if index >= self.lights.len() {
            return Err("Light index out of bounds".to_string());
        }

        self.lights.remove(index);
        Ok(())
    }

    /// Update light parameters
    pub fn update_light(&mut self, index: usize, light: AreaLight) -> Result<(), String> {
        if index >= self.lights.len() {
            return Err("Light index out of bounds".to_string());
        }

        light.validate()?;
        self.lights[index] = light;
        self.lights[index].update_energy_factor();
        Ok(())
    }

    /// Get light count
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Update GPU buffer with current light data
    pub fn update_gpu_buffer(&mut self, queue: &Queue) -> Result<&Buffer, String> {
        let buffer_size = (self.max_lights * std::mem::size_of::<AreaLight>()) as u64;

        // Create buffer if needed
        if self.light_buffer.is_none() {
            self.light_buffer = Some(self.device.create_buffer(&BufferDescriptor {
                label: Some("Area Lights Buffer"),
                size: buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Prepare data for upload
        let mut buffer_data = vec![AreaLight::default(); self.max_lights];
        for (i, light) in self.lights.iter().enumerate() {
            buffer_data[i] = *light;
        }

        // Upload to GPU
        let data_bytes = bytemuck::cast_slice(&buffer_data);
        queue.write_buffer(self.light_buffer.as_ref().unwrap(), 0, data_bytes);

        Ok(self.light_buffer.as_ref().unwrap())
    }

    /// Get bind group layout for area lights
    pub fn get_bind_group_layout(&self) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor {
            label: Some("Area Lights Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        }
    }

    /// Calculate total scene energy from all lights
    pub fn calculate_total_energy(&self) -> f32 {
        self.lights.iter()
            .map(|light| light.intensity * light.energy_factor)
            .sum()
    }

    /// Normalize all light intensities to maintain total energy
    pub fn normalize_energy(&mut self, target_total: f32) {
        let current_total = self.calculate_total_energy();
        if current_total > 0.0 {
            let scale_factor = target_total / current_total;
            for light in &mut self.lights {
                light.intensity *= scale_factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area_light_creation() {
        let light = AreaLight::disc(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0,
            Vec3::new(1.0, 0.8, 0.6),
            15.0,
            1.5
        );

        assert_eq!(light.light_type, AreaLightType::Disc as u32);
        assert_eq!(light.radius, 1.5);
        assert!(light.energy_factor > 0.0);
    }

    #[test]
    fn test_light_validation() {
        let mut light = AreaLight::default();
        assert!(light.validate().is_ok());

        light.radius = -1.0;
        assert!(light.validate().is_err());

        light.radius = 1.0;
        light.intensity = -5.0;
        assert!(light.validate().is_err());
    }

    #[test]
    fn test_energy_factor_calculation() {
        let light1 = AreaLight::disc(Vec3::ZERO, Vec3::Y, 1.0, Vec3::ONE, 10.0, 0.5);
        let light2 = AreaLight::disc(Vec3::ZERO, Vec3::Y, 1.0, Vec3::ONE, 10.0, 2.0);

        // Larger penumbra should have different energy factor
        assert_ne!(light1.energy_factor, light2.energy_factor);
    }

}
