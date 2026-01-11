//! Sampler mode matrix and policy utilities
//!
//! Provides systematic creation of sampler configurations for all combinations
//! of address modes, filters, and mipmap filters.

use super::error::{RenderError, RenderResult};
use super::gpu::ctx;
use std::collections::HashMap;

/// Address mode options for texture sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Clamp texture coordinates to edge
    Clamp,
    /// Repeat texture coordinates
    Repeat,
    /// Mirror texture coordinates
    Mirror,
}

impl AddressMode {
    pub fn to_wgpu(self) -> wgpu::AddressMode {
        match self {
            AddressMode::Clamp => wgpu::AddressMode::ClampToEdge,
            AddressMode::Repeat => wgpu::AddressMode::Repeat,
            AddressMode::Mirror => wgpu::AddressMode::MirrorRepeat,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            AddressMode::Clamp => "clamp",
            AddressMode::Repeat => "repeat",
            AddressMode::Mirror => "mirror",
        }
    }

    pub fn all() -> [AddressMode; 3] {
        [AddressMode::Clamp, AddressMode::Repeat, AddressMode::Mirror]
    }
}

/// Filter mode options for texture sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// Nearest neighbor filtering (point sampling)
    Nearest,
    /// Linear filtering (bilinear interpolation)
    Linear,
}

impl FilterMode {
    pub fn to_wgpu(self) -> wgpu::FilterMode {
        match self {
            FilterMode::Nearest => wgpu::FilterMode::Nearest,
            FilterMode::Linear => wgpu::FilterMode::Linear,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FilterMode::Nearest => "nearest",
            FilterMode::Linear => "linear",
        }
    }

    pub fn all() -> [FilterMode; 2] {
        [FilterMode::Nearest, FilterMode::Linear]
    }
}

/// Complete sampler configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// Address mode for U coordinate
    pub address_mode_u: AddressMode,
    /// Address mode for V coordinate  
    pub address_mode_v: AddressMode,
    /// Address mode for W coordinate
    pub address_mode_w: AddressMode,
    /// Magnification filter
    pub mag_filter: FilterMode,
    /// Minification filter
    pub min_filter: FilterMode,
    /// Mipmap filter
    pub mipmap_filter: FilterMode,
    /// Minimum LOD level
    pub lod_min: f32,
    /// Maximum LOD level
    pub lod_max: f32,
    /// Anisotropic filtering level (1 = disabled, 2-16 = enabled)
    pub anisotropy_clamp: u16,
}

impl SamplerConfig {
    /// Create a new sampler configuration
    pub fn new(
        address_mode: AddressMode,
        mag_filter: FilterMode,
        min_filter: FilterMode,
        mipmap_filter: FilterMode,
    ) -> Self {
        Self {
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min: 0.0,
            lod_max: f32::MAX,
            anisotropy_clamp: 1,
        }
    }

    /// Create configuration with individual address modes per axis
    pub fn new_per_axis(
        address_mode_u: AddressMode,
        address_mode_v: AddressMode,
        address_mode_w: AddressMode,
        mag_filter: FilterMode,
        min_filter: FilterMode,
        mipmap_filter: FilterMode,
    ) -> Self {
        Self {
            address_mode_u,
            address_mode_v,
            address_mode_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min: 0.0,
            lod_max: f32::MAX,
            anisotropy_clamp: 1,
        }
    }

    /// Set LOD range
    pub fn with_lod_range(mut self, min: f32, max: f32) -> Self {
        self.lod_min = min;
        self.lod_max = max;
        self
    }

    /// Convert to wgpu sampler descriptor
    pub fn to_wgpu_descriptor(&self) -> wgpu::SamplerDescriptor<'static> {
        wgpu::SamplerDescriptor {
            label: Some("sampler"),
            address_mode_u: self.address_mode_u.to_wgpu(),
            address_mode_v: self.address_mode_v.to_wgpu(),
            address_mode_w: self.address_mode_w.to_wgpu(),
            mag_filter: self.mag_filter.to_wgpu(),
            min_filter: self.min_filter.to_wgpu(),
            mipmap_filter: self.mipmap_filter.to_wgpu(),
            lod_min_clamp: self.lod_min,
            lod_max_clamp: self.lod_max,
            compare: None,
            anisotropy_clamp: self.anisotropy_clamp,
            border_color: None,
        }
    }

    /// Set anisotropic filtering level (1-16)
    pub fn with_anisotropy(mut self, level: u16) -> Self {
        self.anisotropy_clamp = level.clamp(1, 16);
        self
    }

    /// Create wgpu sampler
    pub fn create_sampler(&self) -> wgpu::Sampler {
        let g = ctx();
        g.device.create_sampler(&self.to_wgpu_descriptor())
    }

    /// Generate a descriptive name for this configuration
    pub fn name(&self) -> String {
        format!(
            "{}_{}_{}_{}",
            self.address_mode_u.name(),
            self.mag_filter.name(),
            self.min_filter.name(),
            self.mipmap_filter.name()
        )
    }

    /// Generate a short descriptive name
    pub fn short_name(&self) -> String {
        format!(
            "{}_{}_{}",
            self.address_mode_u.name().chars().next().unwrap(),
            self.mag_filter.name().chars().next().unwrap(),
            self.mipmap_filter.name().chars().next().unwrap()
        )
    }
}

/// Sampler mode matrix - systematic generation of all sampler combinations
#[derive(Debug)]
pub struct SamplerModeMatrix {
    /// All generated configurations
    pub configs: Vec<SamplerConfig>,
    /// Lookup by name
    pub by_name: HashMap<String, usize>,
}

impl SamplerModeMatrix {
    /// Generate the complete matrix of sampler modes
    /// 3 address modes x 2 mag filters x 2 min filters x 2 mip filters = 24 combinations
    pub fn generate_full() -> Self {
        let mut configs = Vec::new();
        let mut by_name = HashMap::new();

        for address in AddressMode::all() {
            for mag_filter in FilterMode::all() {
                for min_filter in FilterMode::all() {
                    for mip_filter in FilterMode::all() {
                        let config =
                            SamplerConfig::new(address, mag_filter, min_filter, mip_filter);
                        let name = config.name();
                        by_name.insert(name, configs.len());
                        configs.push(config);
                    }
                }
            }
        }

        Self { configs, by_name }
    }

    /// Generate a reduced matrix with commonly used combinations
    /// 3 address modes x 2 filters x 2 mip filters = 12 combinations
    /// (mag_filter = min_filter for simplicity)
    pub fn generate_reduced() -> Self {
        let mut configs = Vec::new();
        let mut by_name = HashMap::new();

        for address in AddressMode::all() {
            for filter in FilterMode::all() {
                for mip_filter in FilterMode::all() {
                    let config = SamplerConfig::new(address, filter, filter, mip_filter);
                    let name = config.name();
                    by_name.insert(name, configs.len());
                    configs.push(config);
                }
            }
        }

        Self { configs, by_name }
    }

    /// Get configuration by name
    pub fn get_config(&self, name: &str) -> Option<&SamplerConfig> {
        self.by_name.get(name).map(|&idx| &self.configs[idx])
    }

    /// Get configuration by index
    pub fn get_config_by_index(&self, index: usize) -> Option<&SamplerConfig> {
        self.configs.get(index)
    }

    /// Number of configurations in the matrix
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Check if matrix is empty
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// List all configuration names
    pub fn list_names(&self) -> Vec<String> {
        self.configs.iter().map(|c| c.name()).collect()
    }

    /// Create samplers for all configurations
    pub fn create_all_samplers(&self) -> RenderResult<Vec<wgpu::Sampler>> {
        self.configs
            .iter()
            .map(|config| Ok(config.create_sampler()))
            .collect()
    }
}

/// Policy-based sampler creation for common use cases
pub struct SamplerPolicy;

impl SamplerPolicy {
    /// Create a sampler suitable for UI textures
    pub fn ui() -> SamplerConfig {
        SamplerConfig::new(
            AddressMode::Clamp,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        )
    }

    /// Create a sampler suitable for tiled textures
    pub fn tiled() -> SamplerConfig {
        SamplerConfig::new(
            AddressMode::Repeat,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Linear,
        )
    }

    /// Create a sampler suitable for skybox textures
    pub fn skybox() -> SamplerConfig {
        SamplerConfig::new(
            AddressMode::Clamp,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Linear,
        )
    }

    /// Create a sampler suitable for pixel art (no filtering)
    pub fn pixel_art() -> SamplerConfig {
        SamplerConfig::new(
            AddressMode::Clamp,
            FilterMode::Nearest,
            FilterMode::Nearest,
            FilterMode::Nearest,
        )
    }

    /// Create a sampler for terrain height maps
    pub fn heightmap() -> SamplerConfig {
        SamplerConfig::new(
            AddressMode::Clamp,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        )
    }

    /// Create a sampler for normal maps
    pub fn normal_map() -> SamplerConfig {
        SamplerConfig::new(
            AddressMode::Repeat,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Linear,
        )
    }
}

/// Convenience functions for creating common sampler configurations
impl SamplerConfig {
    /// Parse sampler configuration from string description
    /// Format: "address_mag_min_mip" (e.g., "clamp_linear_linear_nearest")
    pub fn parse(desc: &str) -> RenderResult<Self> {
        let parts: Vec<&str> = desc.split('_').collect();

        if parts.len() != 4 {
            return Err(RenderError::upload(format!(
                "Invalid sampler description '{}', expected format: address_mag_min_mip",
                desc
            )));
        }

        let address = match parts[0] {
            "clamp" => AddressMode::Clamp,
            "repeat" => AddressMode::Repeat,
            "mirror" => AddressMode::Mirror,
            _ => {
                return Err(RenderError::upload(format!(
                    "Invalid address mode '{}'",
                    parts[0]
                )))
            }
        };

        let mag_filter = match parts[1] {
            "nearest" => FilterMode::Nearest,
            "linear" => FilterMode::Linear,
            _ => {
                return Err(RenderError::upload(format!(
                    "Invalid mag filter '{}'",
                    parts[1]
                )))
            }
        };

        let min_filter = match parts[2] {
            "nearest" => FilterMode::Nearest,
            "linear" => FilterMode::Linear,
            _ => {
                return Err(RenderError::upload(format!(
                    "Invalid min filter '{}'",
                    parts[2]
                )))
            }
        };

        let mip_filter = match parts[3] {
            "nearest" => FilterMode::Nearest,
            "linear" => FilterMode::Linear,
            _ => {
                return Err(RenderError::upload(format!(
                    "Invalid mip filter '{}'",
                    parts[3]
                )))
            }
        };

        Ok(SamplerConfig::new(
            address, mag_filter, min_filter, mip_filter,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_config_creation() {
        let config = SamplerConfig::new(
            AddressMode::Clamp,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        );

        assert_eq!(config.address_mode_u, AddressMode::Clamp);
        assert_eq!(config.mag_filter, FilterMode::Linear);
        assert_eq!(config.min_filter, FilterMode::Linear);
        assert_eq!(config.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_sampler_matrix_generation() {
        let matrix = SamplerModeMatrix::generate_reduced();

        // Should have 3 address modes x 2 filters x 2 mip filters = 12 combinations
        assert_eq!(matrix.len(), 12);

        // All configs should have unique names
        let names: std::collections::HashSet<_> = matrix.configs.iter().map(|c| c.name()).collect();
        assert_eq!(names.len(), matrix.len());
    }

    #[test]
    fn test_sampler_matrix_full() {
        let matrix = SamplerModeMatrix::generate_full();

        // Should have 3 × 2 × 2 × 2 = 24 combinations
        assert_eq!(matrix.len(), 24);
    }

    #[test]
    fn test_sampler_lookup() {
        let matrix = SamplerModeMatrix::generate_reduced();

        let clamp_linear = matrix.get_config("clamp_linear_linear_nearest");
        assert!(clamp_linear.is_some());

        let config = clamp_linear.unwrap();
        assert_eq!(config.address_mode_u, AddressMode::Clamp);
        assert_eq!(config.mag_filter, FilterMode::Linear);
        assert_eq!(config.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_sampler_policies() {
        let ui = SamplerPolicy::ui();
        assert_eq!(ui.address_mode_u, AddressMode::Clamp);
        assert_eq!(ui.mag_filter, FilterMode::Linear);

        let pixel_art = SamplerPolicy::pixel_art();
        assert_eq!(pixel_art.mag_filter, FilterMode::Nearest);
        assert_eq!(pixel_art.min_filter, FilterMode::Nearest);
        assert_eq!(pixel_art.mipmap_filter, FilterMode::Nearest);

        let tiled = SamplerPolicy::tiled();
        assert_eq!(tiled.address_mode_u, AddressMode::Repeat);
    }

    #[test]
    fn test_config_parsing() {
        let config = SamplerConfig::parse("clamp_linear_linear_nearest").unwrap();
        assert_eq!(config.address_mode_u, AddressMode::Clamp);
        assert_eq!(config.mag_filter, FilterMode::Linear);
        assert_eq!(config.min_filter, FilterMode::Linear);
        assert_eq!(config.mipmap_filter, FilterMode::Nearest);

        // Test error cases
        assert!(SamplerConfig::parse("invalid").is_err());
        assert!(SamplerConfig::parse("clamp_invalid_linear_nearest").is_err());
    }

    #[test]
    fn test_config_naming() {
        let config = SamplerConfig::new(
            AddressMode::Clamp,
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        );

        assert_eq!(config.name(), "clamp_linear_linear_nearest");
        assert_eq!(config.short_name(), "c_l_n");
    }
}
