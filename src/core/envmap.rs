//! Environment mapping and image-based lighting (IBL)
//! 
//! Provides CPU-side environment map loading and GPU pipeline setup for
//! realistic environment-based lighting using cubemap textures and roughness-based 
//! mip sampling for physically-based rendering.

use glam::{Vec3, Mat3, Mat4};
use std::f32::consts::PI;
use wgpu::{
    Device, Queue, Texture, TextureDescriptor, TextureUsages, TextureDimension, TextureFormat,
    Extent3d, TextureViewDescriptor, TextureViewDimension, ImageCopyTexture, Origin3d, 
    ImageDataLayout, TextureAspect,
};

/// Environment map data structure
#[derive(Debug, Clone)]
pub struct EnvironmentMap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>, // HDR RGB data
    pub mip_levels: u32,
}

/// Environment mapping configuration
#[derive(Debug, Clone)]
pub struct EnvMapConfig {
    pub roughness_levels: u32,
    pub irradiance_size: u32,
    pub specular_size: u32,
    pub sample_count: u32,
}

impl Default for EnvMapConfig {
    fn default() -> Self {
        Self {
            roughness_levels: 8,
            irradiance_size: 32,
            specular_size: 128, 
            sample_count: 1024,
        }
    }
}

impl EnvironmentMap {
    /// Create a new environment map from HDR data
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self, String> {
        let expected_size = (width * height * 3) as usize;
        if data.len() != expected_size {
            return Err(format!(
                "Data size mismatch: expected {} floats, got {}",
                expected_size, data.len()
            ));
        }
        
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;
        
        Ok(Self {
            width,
            height,
            data,
            mip_levels,
        })
    }
    
    /// Create a synthetic environment map for testing
    pub fn create_test_envmap(size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size * 3) as usize);
        
        for y in 0..size {
            for x in 0..size {
                // Convert to spherical coordinates
                let u = x as f32 / size as f32;
                let v = y as f32 / size as f32;
                
                let phi = u * 2.0 * PI; // Longitude
                let theta = v * PI;     // Latitude
                
                // Create gradient-based environment
                let r = (theta / PI).powf(0.5);
                let g = ((phi + theta) / (3.0 * PI)).sin().abs();
                let b = (1.0 - r * g).max(0.1);
                
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }
        
        Self::new(size, size, data).unwrap()
    }
    
    /// Sample environment map using bilinear interpolation
    pub fn sample(&self, direction: Vec3) -> Vec3 {
        // Convert direction to spherical coordinates
        let phi = direction.z.atan2(direction.x);
        let theta = direction.y.acos();
        
        // Convert to UV coordinates  
        let u = (phi / (2.0 * PI) + 0.5) % 1.0;
        let v = theta / PI;
        
        // Convert to pixel coordinates
        let x = (u * self.width as f32).fract();
        let y = (v * self.height as f32).fract();
        
        let px = (x * (self.width - 1) as f32) as u32;
        let py = (y * (self.height - 1) as f32) as u32;
        
        // Sample RGB
        let idx = ((py * self.width + px) * 3) as usize;
        if idx + 2 < self.data.len() {
            Vec3::new(self.data[idx], self.data[idx + 1], self.data[idx + 2])
        } else {
            Vec3::new(0.5, 0.7, 1.0) // Default sky color
        }
    }
    
    /// Generate irradiance map for diffuse lighting
    pub fn generate_irradiance_map(&self, output_size: u32, sample_count: u32) -> EnvironmentMap {
        let mut irradiance_data = Vec::with_capacity((output_size * output_size * 3) as usize);
        
        for y in 0..output_size {
            for x in 0..output_size {
                // Convert pixel to direction
                let u = x as f32 / output_size as f32;
                let v = y as f32 / output_size as f32;
                
                let phi = u * 2.0 * PI;
                let theta = v * PI;
                
                let direction = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(), 
                    theta.sin() * phi.sin(),
                );
                
                // Compute irradiance by integrating hemisphere
                let irradiance = self.compute_irradiance(direction, sample_count);
                
                irradiance_data.push(irradiance.x);
                irradiance_data.push(irradiance.y);
                irradiance_data.push(irradiance.z);
            }
        }
        
        EnvironmentMap::new(output_size, output_size, irradiance_data).unwrap()
    }
    
    /// Compute irradiance for a given normal direction
    fn compute_irradiance(&self, normal: Vec3, sample_count: u32) -> Vec3 {
        let mut irradiance = Vec3::ZERO;
        let mut weight = 0.0;
        
        // Monte Carlo sampling over hemisphere
        for i in 0..sample_count {
            let xi = generate_hammersley_sample(i, sample_count);
            let sample_dir = sample_cosine_hemisphere(xi.x, xi.y);
            
            // Transform to world space aligned with normal
            let tangent = if normal.y.abs() < 0.9 {
                Vec3::Y.cross(normal).normalize()
            } else {
                Vec3::X.cross(normal).normalize()
            };
            let bitangent = normal.cross(tangent);
            
            let world_sample = sample_dir.x * tangent + 
                              sample_dir.y * bitangent + 
                              sample_dir.z * normal;
            
            // Sample environment map
            let radiance = self.sample(world_sample);
            let cos_theta = world_sample.dot(normal).max(0.0);
            
            irradiance += radiance * cos_theta;
            weight += cos_theta;
        }
        
        if weight > 0.0 {
            irradiance / weight * PI
        } else {
            Vec3::new(0.5, 0.7, 1.0)
        }
    }
    
    /// Generate prefiltered environment map for specular reflections
    pub fn generate_prefiltered_map(&self, output_size: u32, roughness: f32, sample_count: u32) -> EnvironmentMap {
        let mut filtered_data = Vec::with_capacity((output_size * output_size * 3) as usize);
        
        for y in 0..output_size {
            for x in 0..output_size {
                // Convert pixel to reflection direction
                let u = x as f32 / output_size as f32;
                let v = y as f32 / output_size as f32;
                
                let phi = u * 2.0 * PI;
                let theta = v * PI;
                
                let reflection_dir = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                
                // Prefilter for given roughness
                let filtered_color = self.prefilter_specular(reflection_dir, roughness, sample_count);
                
                filtered_data.push(filtered_color.x);
                filtered_data.push(filtered_color.y);
                filtered_data.push(filtered_color.z);
            }
        }
        
        EnvironmentMap::new(output_size, output_size, filtered_data).unwrap()
    }
    
    /// Prefilter environment map for specular reflections with given roughness
    fn prefilter_specular(&self, reflection_dir: Vec3, roughness: f32, sample_count: u32) -> Vec3 {
        let mut filtered_color = Vec3::ZERO;
        let mut total_weight = 0.0;
        
        // Importance sampling using GGX distribution
        for i in 0..sample_count {
            let xi = generate_hammersley_sample(i, sample_count);
            let half_vector = sample_ggx_hemisphere(xi.x, xi.y, roughness);
            
            // Transform to world space
            let tangent = if reflection_dir.y.abs() < 0.9 {
                Vec3::Y.cross(reflection_dir).normalize()
            } else {
                Vec3::X.cross(reflection_dir).normalize()
            };
            let bitangent = reflection_dir.cross(tangent);
            
            let world_half = half_vector.x * tangent + 
                            half_vector.y * bitangent + 
                            half_vector.z * reflection_dir;
            
            let sample_dir = 2.0 * reflection_dir.dot(world_half) * world_half - reflection_dir;
            
            let cos_theta = sample_dir.dot(reflection_dir).max(0.0);
            if cos_theta > 0.0 {
                let radiance = self.sample(sample_dir);
                filtered_color += radiance * cos_theta;
                total_weight += cos_theta;
            }
        }
        
        if total_weight > 0.0 {
            filtered_color / total_weight
        } else {
            self.sample(reflection_dir)
        }
    }
    
    /// Upload environment map to GPU texture
    pub fn upload_to_gpu(&self, device: &Device, queue: &Queue) -> Texture {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("environment_map"),
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1, // Single mip level for now
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float, // HDR format
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Convert RGB to RGBA
        let mut rgba_data = Vec::with_capacity(self.data.len() * 4 / 3);
        for chunk in self.data.chunks(3) {
            rgba_data.push(chunk[0]);
            rgba_data.push(chunk[1]); 
            rgba_data.push(chunk[2]);
            rgba_data.push(1.0); // Alpha
        }
        
        // Upload texture data
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba_data),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4 * 4), // 4 components * 4 bytes per f32
                rows_per_image: Some(self.height),
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        
        texture
    }
}

/// Generate Hammersley sequence sample
fn generate_hammersley_sample(i: u32, n: u32) -> (f32, f32) {
    let u1 = i as f32 / n as f32;
    let u2 = radical_inverse_vdc(i);
    (u1, u2)
}

/// Radical inverse for Hammersley sequence  
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    
    (bits as f32) * 2.3283064365386963e-10 // / 0x100000000
}

/// Sample cosine-weighted hemisphere
fn sample_cosine_hemisphere(u1: f32, u2: f32) -> Vec3 {
    let cos_theta = (1.0 - u2).sqrt();
    let sin_theta = u2.sqrt();
    let phi = 2.0 * PI * u1;
    
    Vec3::new(phi.cos() * sin_theta, cos_theta, phi.sin() * sin_theta)
}

/// Sample GGX distribution on hemisphere
fn sample_ggx_hemisphere(u1: f32, u2: f32, roughness: f32) -> Vec3 {
    let alpha = roughness * roughness;
    let cos_theta = ((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = 2.0 * PI * u1;
    
    Vec3::new(phi.cos() * sin_theta, cos_theta, phi.sin() * sin_theta)
}

/// Compute environment lighting for a given set of roughness values
pub fn compute_roughness_luminance_series(envmap: &EnvironmentMap, roughness_values: &[f32]) -> Vec<f32> {
    let sample_count = 256;
    let mut luminances = Vec::new();
    
    // Sample in a consistent direction for comparison
    let sample_direction = Vec3::new(0.0, 1.0, 0.0); // Upward direction
    
    for &roughness in roughness_values {
        let filtered = envmap.prefilter_specular(sample_direction, roughness, sample_count);
        
        // Compute luminance (Y component in XYZ color space)
        let luminance = 0.299 * filtered.x + 0.587 * filtered.y + 0.114 * filtered.z;
        luminances.push(luminance);
    }
    
    luminances
}