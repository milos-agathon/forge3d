// src/lighting/ibl_cache.rs
// IBL resource cache for BRDF LUT, irradiance, and prefiltered specular maps
// P0: Simple cache with fixed-size textures (<=64 MiB budget)

use std::sync::Arc;
use wgpu::*;

/// IBL resource cache (P0)
/// Manages BRDF LUT, irradiance cubemap, and prefiltered specular cubemap
/// Total budget: <=64 MiB
/// IBL resource cache for BRDF/irradiance/specular maps.
/// Some fields are not accessed in all builds; silence until the full IBL path is wired.
#[allow(dead_code)]
pub struct IblResourceCache {
    device: Arc<Device>,
    queue: Arc<Queue>,

    /// BRDF integration LUT (512x512 RG16F) ~2 MiB
    pub brdf_lut: Option<Texture>,
    pub brdf_lut_view: Option<TextureView>,

    /// Irradiance cubemap (32x32x6 RGBA16F) ~0.1 MiB
    pub irradiance_map: Option<Texture>,
    pub irradiance_map_view: Option<TextureView>,

    /// Prefiltered specular cubemap (128x128x6 RGBA16F with mips) ~6 MiB
    pub specular_map: Option<Texture>,
    pub specular_map_view: Option<TextureView>,

    /// Sampler for IBL textures
    pub sampler: Sampler,
}

impl IblResourceCache {
    /// Create new IBL resource cache
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        // Create sampler for IBL lookups
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("IBL Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        Self {
            device,
            queue,
            brdf_lut: None,
            brdf_lut_view: None,
            irradiance_map: None,
            irradiance_map_view: None,
            specular_map: None,
            specular_map_view: None,
            sampler,
        }
    }

    /// Initialize BRDF LUT texture (512x512 RG16F)
    pub fn init_brdf_lut(&mut self) {
        let size = 512;
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("BRDF LUT"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor::default());

        // TODO: Generate BRDF LUT via compute shader
        // For P0, we'll upload a pre-computed LUT or generate on first use

        self.brdf_lut = Some(texture);
        self.brdf_lut_view = Some(view);
    }

    /// Initialize irradiance cubemap (32x32x6 RGBA16F)
    pub fn init_irradiance_map(&mut self) {
        let size = 32;
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Irradiance Cubemap"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6, // Cubemap
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        // TODO: Generate irradiance from environment map via compute/render

        self.irradiance_map = Some(texture);
        self.irradiance_map_view = Some(view);
    }

    /// Initialize prefiltered specular cubemap (128x128x6 RGBA16F with mips)
    pub fn init_specular_map(&mut self) {
        let size = 128;
        let mip_count = 6; // 128 -> 64 -> 32 -> 16 -> 8 -> 4

        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Prefiltered Specular Cubemap"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6, // Cubemap
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        // TODO: Generate prefiltered specular mips via compute/render

        self.specular_map = Some(texture);
        self.specular_map_view = Some(view);
    }

    /// Initialize all IBL resources
    pub fn init_all(&mut self) {
        self.init_brdf_lut();
        self.init_irradiance_map();
        self.init_specular_map();
    }

    /// Calculate total memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        let mut total = 0u64;

        // BRDF LUT: 512x512 RG16F = 512*512*2*2 = ~2 MiB
        if self.brdf_lut.is_some() {
            total += 512 * 512 * 2 * 2;
        }

        // Irradiance: 32x32x6 RGBA16F = 32*32*6*4*2 = ~98 KiB
        if self.irradiance_map.is_some() {
            total += 32 * 32 * 6 * 4 * 2;
        }

        // Specular: 128x128x6 RGBA16F with 6 mip levels
        // Mip 0: 128*128*6*4*2 = 786 KiB
        // Mip 1: 64*64*6*4*2 = 196 KiB
        // Mip 2: 32*32*6*4*2 = 49 KiB
        // Mip 3: 16*16*6*4*2 = 12 KiB
        // Mip 4: 8*8*6*4*2 = 3 KiB
        // Mip 5: 4*4*6*4*2 = 768 bytes
        // Total: ~1.05 MiB
        if self.specular_map.is_some() {
            let mut size = 128u64;
            for _ in 0..6 {
                total += size * size * 6 * 4 * 2;
                size /= 2;
            }
        }

        total
    }

    /// Check if all resources are initialized
    pub fn is_ready(&self) -> bool {
        self.brdf_lut.is_some() && self.irradiance_map.is_some() && self.specular_map.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibl_memory_budget() {
        // Verify P0 budget constraint: <=64 MiB
        // Our design:
        // - BRDF LUT: ~2 MiB
        // - Irradiance: ~0.1 MiB
        // - Specular: ~1 MiB
        // Total: ~3 MiB (well under budget)

        let brdf_lut = 512 * 512 * 2 * 2;
        let irradiance = 32 * 32 * 6 * 4 * 2;
        let mut specular = 0u64;
        let mut size = 128u64;
        for _ in 0..6 {
            specular += size * size * 6 * 4 * 2;
            size /= 2;
        }

        let total = brdf_lut + irradiance + specular;
        let total_mib = total as f64 / (1024.0 * 1024.0);

        println!(
            "BRDF LUT: {} bytes ({:.2} MiB)",
            brdf_lut,
            brdf_lut as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Irradiance: {} bytes ({:.2} MiB)",
            irradiance,
            irradiance as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Specular: {} bytes ({:.2} MiB)",
            specular,
            specular as f64 / (1024.0 * 1024.0)
        );
        println!("Total: {} bytes ({:.2} MiB)", total, total_mib);

        assert!(total_mib < 64.0, "IBL cache exceeds 64 MiB budget");
    }
}
