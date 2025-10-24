// src/util/memory_budget.rs
//! Memory budget tracker for GPU resources
//!
//! Implements the 512 MiB host-visible memory cap by estimating texture
//! and buffer sizes, selecting appropriate quality tiers, and logging usage.
//!
//! RELEVANT FILES: src/material_set.rs, src/ibl_wrapper.rs, src/core/ibl.rs

use log::{info, warn};

/// Memory budget cap in bytes (512 MiB)
pub const MEMORY_BUDGET_CAP: u64 = 512 * 1024 * 1024;

/// Conservative estimate threshold for auto-tier selection (384 MiB)
pub const MEMORY_BUDGET_CONSERVATIVE: u64 = 384 * 1024 * 1024;

/// Quality tier for textures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureQualityTier {
    Low,
    Medium,
    High,
    Ultra,
}

impl TextureQualityTier {
    /// Maximum texture dimension for this tier
    pub fn max_texture_dimension(self) -> u32 {
        match self {
            Self::Low => 512,
            Self::Medium => 1024,
            Self::High => 2048,
            Self::Ultra => 4096,
        }
    }

    /// From string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "ultra" => Some(Self::Ultra),
            _ => None,
        }
    }
}

/// Memory usage breakdown for logging and validation
#[derive(Debug, Clone, Default)]
pub struct MemoryUsageReport {
    pub material_textures: u64,
    pub ibl_irradiance: u64,
    pub ibl_specular: u64,
    pub ibl_brdf: u64,
    pub other: u64,
}

impl MemoryUsageReport {
    pub fn total(&self) -> u64 {
        self.material_textures
            + self.ibl_irradiance
            + self.ibl_specular
            + self.ibl_brdf
            + self.other
    }

    pub fn total_mb(&self) -> f32 {
        self.total() as f32 / (1024.0 * 1024.0)
    }

    pub fn log_summary(&self, tier_name: &str) {
        let total = self.total();
        let total_mb = self.total_mb();

        if total > MEMORY_BUDGET_CAP {
            warn!(
                "Memory usage EXCEEDS budget: {:.2} MiB / 512 MiB (tier: {})",
                total_mb, tier_name
            );
        } else {
            info!(
                "Memory usage: {:.2} MiB / 512 MiB (tier: {})",
                total_mb, tier_name
            );
        }

        info!(
            "  Material textures: {:.2} MiB",
            self.material_textures as f32 / (1024.0 * 1024.0)
        );
        info!(
            "  IBL irradiance:    {:.2} MiB",
            self.ibl_irradiance as f32 / (1024.0 * 1024.0)
        );
        info!(
            "  IBL specular:      {:.2} MiB",
            self.ibl_specular as f32 / (1024.0 * 1024.0)
        );
        info!(
            "  IBL BRDF LUT:      {:.2} MiB",
            self.ibl_brdf as f32 / (1024.0 * 1024.0)
        );
        if self.other > 0 {
            info!(
                "  Other:             {:.2} MiB",
                self.other as f32 / (1024.0 * 1024.0)
            );
        }
    }
}

/// Estimate memory for RGBA8UnormSrgb texture
pub fn estimate_rgba8_texture(width: u32, height: u32, layers: u32) -> u64 {
    (width as u64) * (height as u64) * (layers as u64) * 4
}

/// Estimate memory for RGBA16Float texture
pub fn estimate_rgba16_texture(width: u32, height: u32, layers: u32) -> u64 {
    (width as u64) * (height as u64) * (layers as u64) * 8
}

/// Estimate memory for cubemap with mipmaps
pub fn estimate_cubemap_with_mips(face_size: u32, mip_levels: u32, bytes_per_pixel: u32) -> u64 {
    let mut total = 0u64;
    for mip in 0..mip_levels {
        let mip_size = face_size >> mip;
        if mip_size == 0 {
            break;
        }
        total += (mip_size as u64) * (mip_size as u64) * 6 * (bytes_per_pixel as u64);
    }
    total
}

/// Estimate memory for IBL resources based on quality
pub fn estimate_ibl_memory(quality: crate::core::ibl::IBLQuality) -> (u64, u64, u64) {
    let irradiance_size = quality.irradiance_size();
    let specular_size = quality.specular_size();
    let brdf_size = quality.brdf_size();
    let mip_levels = quality.specular_mip_levels();

    // IBL uses RGBA16Float (8 bytes per pixel)
    let irradiance_bytes = estimate_cubemap_with_mips(irradiance_size, 1, 8);
    let specular_bytes = estimate_cubemap_with_mips(specular_size, mip_levels, 8);
    let brdf_bytes = estimate_rgba16_texture(brdf_size, brdf_size, 1);

    (irradiance_bytes, specular_bytes, brdf_bytes)
}

/// Select appropriate tier based on estimated usage
pub fn select_tier_for_budget(
    material_texture_size: u64,
    ibl_quality: crate::core::ibl::IBLQuality,
) -> TextureQualityTier {
    let (ibl_irr, ibl_spec, ibl_brdf) = estimate_ibl_memory(ibl_quality);
    let total = material_texture_size + ibl_irr + ibl_spec + ibl_brdf;

    if total > MEMORY_BUDGET_CONSERVATIVE {
        warn!(
            "Estimated memory ({:.2} MiB) exceeds conservative budget, selecting Low tier",
            total as f32 / (1024.0 * 1024.0)
        );
        TextureQualityTier::Low
    } else if total > (MEMORY_BUDGET_CAP * 3 / 4) {
        info!(
            "Estimated memory ({:.2} MiB) is high, selecting Medium tier",
            total as f32 / (1024.0 * 1024.0)
        );
        TextureQualityTier::Medium
    } else if total > (MEMORY_BUDGET_CAP / 2) {
        TextureQualityTier::High
    } else {
        TextureQualityTier::Ultra
    }
}

/// Determine if GPU is likely an integrated GPU based on wgpu AdapterInfo
pub fn is_likely_igpu(info: &wgpu::AdapterInfo) -> bool {
    use wgpu::DeviceType;

    match info.device_type {
        DeviceType::IntegratedGpu => true,
        DeviceType::DiscreteGpu => false,
        DeviceType::VirtualGpu => true, // Treat virtual as integrated for safety
        DeviceType::Cpu => true,        // CPU renderer needs conservative limits
        DeviceType::Other => {
            // Heuristic: check for known iGPU identifiers in name
            let name_lower = info.name.to_lowercase();
            name_lower.contains("intel") && name_lower.contains("uhd")
                || name_lower.contains("intel") && name_lower.contains("iris")
                || name_lower.contains("amd")
                    && name_lower.contains("radeon")
                    && name_lower.contains("graphics")
                || name_lower.contains("apple") && name_lower.contains("m1")
                || name_lower.contains("apple") && name_lower.contains("m2")
        }
    }
}

/// Auto-select IBL quality based on GPU type and memory budget
pub fn auto_select_ibl_quality(adapter_info: &wgpu::AdapterInfo) -> crate::core::ibl::IBLQuality {
    if is_likely_igpu(adapter_info) {
        info!(
            "Detected integrated GPU ({}), selecting Low IBL quality tier",
            adapter_info.name
        );
        crate::core::ibl::IBLQuality::Low
    } else {
        info!(
            "Detected discrete GPU ({}), selecting High IBL quality tier",
            adapter_info.name
        );
        crate::core::ibl::IBLQuality::High
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimates() {
        // Test RGBA8 texture estimate
        let rgba8_512 = estimate_rgba8_texture(512, 512, 4);
        assert_eq!(rgba8_512, 512 * 512 * 4 * 4);

        // Test cubemap with mips
        let cubemap_128_5mips = estimate_cubemap_with_mips(128, 5, 8);
        // 128^2 + 64^2 + 32^2 + 16^2 + 8^2 = 16384 + 4096 + 1024 + 256 + 64 = 21824 per face
        // 21824 * 6 faces * 8 bytes = 1,046,016 bytes
        assert_eq!(cubemap_128_5mips, 1_046_016);
    }

    #[test]
    fn test_tier_selection() {
        // Small usage should select high tier
        let tier = select_tier_for_budget(10_000_000, crate::core::ibl::IBLQuality::Low);
        assert!(matches!(
            tier,
            TextureQualityTier::High | TextureQualityTier::Ultra
        ));

        // Large usage should select low tier
        let tier = select_tier_for_budget(400_000_000, crate::core::ibl::IBLQuality::Ultra);
        assert_eq!(tier, TextureQualityTier::Low);
    }

    #[test]
    fn test_igpu_detection() {
        let info = wgpu::AdapterInfo {
            name: "Intel(R) UHD Graphics 620".to_string(),
            vendor: 0x8086,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert!(is_likely_igpu(&info));

        let info_discrete = wgpu::AdapterInfo {
            name: "NVIDIA GeForce RTX 3080".to_string(),
            vendor: 0x10de,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert!(!is_likely_igpu(&info_discrete));
    }
}
