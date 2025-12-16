// src/material_set.rs
//! Material set for terrain rendering with triplanar mapping support

use pyo3::prelude::*;
use std::path::Path;

#[cfg(feature = "extension-module")]
use anyhow::Result;
#[cfg(feature = "extension-module")]
use image::{imageops::FilterType, DynamicImage, GenericImageView};
#[cfg(feature = "extension-module")]
use log::{info, warn};
#[cfg(feature = "extension-module")]
use once_cell::sync::OnceCell;
#[cfg(feature = "extension-module")]
use std::sync::Arc;
#[cfg(feature = "extension-module")]
use wgpu::{
    AddressMode, Extent3d, FilterMode, ImageDataLayout, Origin3d, SamplerDescriptor,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension, COPY_BYTES_PER_ROW_ALIGNMENT,
};

/// Material set for terrain rendering
#[pyclass(module = "forge3d._forge3d", name = "MaterialSet")]
pub struct MaterialSet {
    pub(crate) materials: Vec<crate::core::material::PbrMaterial>,
    pub(crate) triplanar_scale: f32,
    pub(crate) normal_strength: f32,
    pub(crate) blend_sharpness: f32,
    #[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
    pub(crate) texture_paths: Vec<Option<String>>,
    #[cfg(feature = "extension-module")]
    gpu_cache: OnceCell<Arc<GpuMaterialSet>>,
}

#[pymethods]
impl MaterialSet {
    /// Create default terrain material set
    ///
    /// Args:
    ///     triplanar_scale: Texture scaling for triplanar mapping (default: 6.0)
    ///     normal_strength: Normal map strength multiplier (default: 1.0)
    ///     blend_sharpness: Triplanar blend sharpness (default: 4.0)
    ///
    /// Returns:
    ///     MaterialSet with rock, grass, and snow materials
    #[staticmethod]
    #[pyo3(signature = (triplanar_scale=6.0, normal_strength=1.0, blend_sharpness=4.0))]
    pub fn terrain_default(
        triplanar_scale: f32,
        normal_strength: f32,
        blend_sharpness: f32,
    ) -> PyResult<Self> {
        // Validate inputs
        if triplanar_scale <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "triplanar_scale must be > 0",
            ));
        }
        if normal_strength < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "normal_strength must be >= 0",
            ));
        }
        if blend_sharpness <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "blend_sharpness must be > 0",
            ));
        }

        let mut materials = Vec::new();
        let mut texture_paths = Vec::new();

        // Material 0: Rock (steep slopes, low altitude) - rough weathered surface
        // Moderate-high roughness reduces specular aliasing on cliff faces
        // Real rock varies widely; 0.5 is typical for weathered granite/sandstone
        materials.push(
            crate::core::material::PbrMaterial::dielectric(
                glam::Vec3::new(0.28, 0.26, 0.24), // neutral gray rock
                0.50,                              // roughness - weathered rock
            )
            .with_normal_scale(normal_strength * 1.5), // Enhanced normal detail
        );
        texture_paths.push(resolve_default_texture("rock_albedo.png"));

        // Material 1: Grass (mid altitude, flat areas) - matte organic surface
        // High roughness = diffuse response, distinct from rock
        materials.push(
            crate::core::material::PbrMaterial::dielectric(
                glam::Vec3::new(0.18, 0.38, 0.10), // saturated green
                0.85,                              // roughness - very matte (grass is diffuse)
            )
            .with_normal_scale(normal_strength * 0.8), // Softer normals for grass
        );
        texture_paths.push(resolve_default_texture("grass_albedo.png"));

        // Material 2: Dirt/soil (transition zones) - medium rough with wetness variation
        materials.push(
            crate::core::material::PbrMaterial::dielectric(
                glam::Vec3::new(0.35, 0.25, 0.15), // warm brown dirt
                0.50,                              // roughness - between rock and grass
            )
            .with_normal_scale(normal_strength * 1.2), // Visible soil texture
        );
        texture_paths.push(resolve_default_texture("dirt_albedo.png"));

        // Material 3: Snow (high altitude peaks) - powder/packed snow surface
        // Moderate roughness to reduce specular aliasing while keeping soft highlights
        // Real snow ranges from 0.2 (fresh powder) to 0.5 (packed/wet); 0.25 is a good balance
        materials.push(
            crate::core::material::PbrMaterial::dielectric(
                glam::Vec3::new(0.95, 0.97, 1.0), // bright white-blue snow
                0.25,                             // roughness - powder snow, still bright
            )
            .with_normal_scale(normal_strength * 0.3), // Subtle snow surface variation
        );
        texture_paths.push(resolve_default_texture("snow_albedo.png"));

        Ok(Self {
            materials,
            triplanar_scale,
            normal_strength,
            blend_sharpness,
            texture_paths,
            #[cfg(feature = "extension-module")]
            gpu_cache: OnceCell::new(),
        })
    }

    /// Create a material set with a single custom material
    ///
    /// Args:
    ///     base_color: Base color (RGB) as tuple (r, g, b) in [0, 1]
    ///     metallic: Metallic factor [0, 1]
    ///     roughness: Roughness factor [0.04, 1.0]
    ///     triplanar_scale: Texture scaling for triplanar mapping (default: 1.0)
    ///     normal_strength: Normal map strength multiplier (default: 0.0)
    ///     blend_sharpness: Triplanar blend sharpness (default: 1.0)
    ///
    /// Returns:
    ///     MaterialSet with a single material with the specified properties
    #[staticmethod]
    #[pyo3(signature = (base_color, metallic, roughness, triplanar_scale=1.0, normal_strength=0.0, blend_sharpness=1.0))]
    pub fn custom(
        base_color: (f32, f32, f32),
        metallic: f32,
        roughness: f32,
        triplanar_scale: f32,
        normal_strength: f32,
        blend_sharpness: f32,
    ) -> PyResult<Self> {
        // Validate inputs
        if triplanar_scale <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "triplanar_scale must be > 0",
            ));
        }
        if normal_strength < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "normal_strength must be >= 0",
            ));
        }
        if blend_sharpness <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "blend_sharpness must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&metallic) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "metallic must be in [0, 1]",
            ));
        }
        if !(0.04..=1.0).contains(&roughness) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roughness must be in [0.04, 1.0]",
            ));
        }

        let mut materials = Vec::new();
        let mut texture_paths = Vec::new();

        // Create a single material with the specified properties
        let color = glam::Vec3::new(base_color.0, base_color.1, base_color.2);
        let material = if metallic > 0.5 {
            crate::core::material::PbrMaterial::metallic(color, roughness)
                .with_normal_scale(normal_strength)
        } else {
            crate::core::material::PbrMaterial::dielectric(color, roughness)
                .with_normal_scale(normal_strength)
        };
        materials.push(material);
        texture_paths.push(None); // No texture for custom material

        Ok(Self {
            materials,
            triplanar_scale,
            normal_strength,
            blend_sharpness,
            texture_paths,
            #[cfg(feature = "extension-module")]
            gpu_cache: OnceCell::new(),
        })
    }

    /// Get number of materials
    #[getter]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Get triplanar scale
    #[getter]
    pub fn triplanar_scale(&self) -> f32 {
        self.triplanar_scale
    }

    /// Get normal strength
    #[getter]
    pub fn normal_strength(&self) -> f32 {
        self.normal_strength
    }

    /// Get blend sharpness
    #[getter]
    pub fn blend_sharpness(&self) -> f32 {
        self.blend_sharpness
    }

    /// Python repr
    fn __repr__(&self) -> String {
        format!(
            "MaterialSet(materials={}, triplanar_scale={:.1}, normal_strength={:.1}, blend_sharpness={:.1})",
            self.materials.len(),
            self.triplanar_scale,
            self.normal_strength,
            self.blend_sharpness
        )
    }
}

impl Clone for MaterialSet {
    fn clone(&self) -> Self {
        Self {
            materials: self.materials.clone(),
            triplanar_scale: self.triplanar_scale,
            normal_strength: self.normal_strength,
            blend_sharpness: self.blend_sharpness,
            texture_paths: self.texture_paths.clone(),
            #[cfg(feature = "extension-module")]
            gpu_cache: OnceCell::new(),
        }
    }
}

impl MaterialSet {
    /// Get reference to materials
    pub fn materials(&self) -> &[crate::core::material::PbrMaterial] {
        &self.materials
    }

    /// Get material at index
    pub fn get_material(&self, index: usize) -> Option<&crate::core::material::PbrMaterial> {
        self.materials.get(index)
    }

    #[cfg(feature = "extension-module")]
    #[allow(dead_code)]
    pub(crate) fn texture_paths(&self) -> &[Option<String>] {
        &self.texture_paths
    }
}

#[cfg(feature = "extension-module")]
impl MaterialSet {
    pub(crate) fn gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Arc<GpuMaterialSet>> {
        self.gpu_cache
            .get_or_try_init(|| {
                let gpu = GpuMaterialSet::new(device, queue, &self.materials, &self.texture_paths)?;
                Ok(Arc::new(gpu))
            })
            .map(|arc| Arc::clone(arc))
    }
}

fn resolve_default_texture(file_name: &str) -> Option<String> {
    let asset_root = std::env::var("FORGE3D_MATERIAL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("assets/materials"));
    let candidate = asset_root.join(file_name);
    if Path::new(&candidate).exists() {
        Some(candidate.to_string_lossy().to_string())
    } else {
        None
    }
}

#[cfg(feature = "extension-module")]
pub(crate) const MAX_LAYERS: usize = 4;

#[cfg(feature = "extension-module")]
pub(crate) struct GpuMaterialSet {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
    pub(crate) layer_count: u32,
    pub(crate) layer_centers: [f32; MAX_LAYERS],
}

#[cfg(feature = "extension-module")]
impl GpuMaterialSet {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[crate::core::material::PbrMaterial],
        texture_paths: &[Option<String>],
    ) -> Result<Self> {
        let mut layer_count = materials.len();
        if layer_count == 0 {
            layer_count = 1;
        }
        if layer_count > MAX_LAYERS {
            warn!(
                "MaterialSet has {} materials; only the first {} will be used for terrain rendering",
                layer_count,
                MAX_LAYERS
            );
            layer_count = MAX_LAYERS;
        }

        let mut images: Vec<Option<DynamicImage>> = Vec::with_capacity(layer_count);
        for idx in 0..layer_count {
            let path_opt = texture_paths.get(idx).and_then(|p| p.as_ref());
            let image = match path_opt {
                Some(path) => match image::open(path) {
                    Ok(img) => Some(img),
                    Err(err) => {
                        warn!(
                            "Failed to load terrain material texture '{}': {}",
                            path, err
                        );
                        None
                    }
                },
                None => None,
            };
            images.push(image);
        }

        let mut canonical_size: Option<(u32, u32)> = None;
        for img in images.iter() {
            if let Some(image) = img {
                canonical_size = Some(image.dimensions());
                break;
            }
        }
        let (mut target_width, mut target_height) = canonical_size.unwrap_or((512, 512));

        let (max_dimension, mut selected_tier) = Self::calculate_max_texture_dimension(layer_count);
        let original_size = (target_width, target_height);
        if target_width > max_dimension || target_height > max_dimension {
            target_width = target_width.min(max_dimension);
            target_height = target_height.min(max_dimension);
            warn!(
                "Material textures exceed memory budget: {}x{} -> {}x{} (max dim: {}, tier {:?})",
                original_size.0,
                original_size.1,
                target_width,
                target_height,
                max_dimension,
                selected_tier
            );
        }

        let mut resolved_width = target_width.max(1);
        let mut resolved_height = target_height.max(1);
        let layer_count_u32 = layer_count as u32;
        // ALWAYS generate mipmaps for material textures to prevent aliasing/sparkle
        let mut generate_mips = true;
        let mut mip_level_count = compute_mip_level_count(resolved_width, resolved_height);
        let mut final_bytes;

        loop {
            let estimated = if generate_mips {
                estimate_rgba8_mip_chain(resolved_width, resolved_height, layer_count_u32)
            } else {
                crate::util::memory_budget::estimate_rgba8_texture(
                    resolved_width,
                    resolved_height,
                    layer_count_u32,
                )
            };

            if estimated <= crate::util::memory_budget::MEMORY_BUDGET_CONSERVATIVE
                || (resolved_width <= 256 && resolved_height <= 256)
            {
                final_bytes = estimated;
                break;
            }

            let prev = (resolved_width, resolved_height);
            resolved_width = (resolved_width / 2).max(256);
            resolved_height = (resolved_height / 2).max(256);
            if prev == (resolved_width, resolved_height) {
                final_bytes = estimated;
                break;
            }

            selected_tier = downgrade_tier(selected_tier);
            generate_mips = true;
            mip_level_count = compute_mip_level_count(resolved_width, resolved_height);
        }

        if final_bytes > crate::util::memory_budget::MEMORY_BUDGET_CAP
            && (resolved_width > 256 || resolved_height > 256)
        {
            warn!(
                "Material textures still exceed budget ({:.2} MiB); forcing Low tier fallback.",
                final_bytes as f32 / (1024.0 * 1024.0)
            );
            resolved_width = 256;
            resolved_height = 256;
            selected_tier = crate::util::memory_budget::TextureQualityTier::Low;
            generate_mips = true;
            mip_level_count = compute_mip_level_count(resolved_width, resolved_height);
        }

        if generate_mips && mip_level_count == 1 {
            mip_level_count = compute_mip_level_count(resolved_width, resolved_height);
        }

        if (resolved_width, resolved_height) != original_size {
            info!(
                "Material textures resolved to {}x{} (tier {:?}) from {}x{}",
                resolved_width, resolved_height, selected_tier, original_size.0, original_size.1
            );
        }

        // Recalculate final bytes after all adjustments
        final_bytes = if generate_mips {
            estimate_rgba8_mip_chain(resolved_width, resolved_height, layer_count_u32)
        } else {
            crate::util::memory_budget::estimate_rgba8_texture(
                resolved_width,
                resolved_height,
                layer_count_u32,
            )
        };

        let mut usage_report = crate::util::memory_budget::MemoryUsageReport::default();
        usage_report.material_textures = final_bytes;
        usage_report.log_summary(&format!("Materials::{:?}", selected_tier));

        target_width = resolved_width;
        target_height = resolved_height;

        let mut layer_pixels: Vec<Vec<Vec<u8>>> = Vec::with_capacity(layer_count);
        for idx in 0..layer_count {
            let image_opt = images.get_mut(idx).and_then(|slot| slot.take());
            let mip_chain = prepare_layer_mips(
                image_opt,
                &materials[idx],
                target_width,
                target_height,
                mip_level_count,
            );
            layer_pixels.push(mip_chain);
        }

        let texture = device.create_texture(&TextureDescriptor {
            label: Some("terrain.materials.albedo"),
            size: Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: layer_count_u32.max(1),
            },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer_idx, mip_chain) in layer_pixels.iter().enumerate() {
            let mut mip_width = target_width;
            let mut mip_height = target_height;
            for (mip_level, pixels) in mip_chain.iter().enumerate() {
                let width = mip_width.max(1);
                let height = mip_height.max(1);
                let (padded, padded_bpr) = pad_rgba_rows(width, height, pixels);
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: mip_level as u32,
                        origin: Origin3d {
                            x: 0,
                            y: 0,
                            z: layer_idx as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &padded,
                    ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(height),
                    },
                    Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                if mip_width > 1 {
                    mip_width /= 2;
                }
                if mip_height > 1 {
                    mip_height /= 2;
                }
            }
        }

        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("terrain.materials.albedo.view"),
            format: Some(TextureFormat::Rgba8UnormSrgb),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain.materials.albedo.sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            anisotropy_clamp: 16, // Max anisotropic filtering to reduce triplanar aliasing
            ..Default::default()
        });

        let mut layer_centers = [0.0f32; MAX_LAYERS];
        if layer_count == 1 {
            layer_centers[0] = 0.0;
        } else {
            let denom = (layer_count as f32 - 1.0).max(1.0);
            for idx in 0..layer_count {
                layer_centers[idx] = idx as f32 / denom;
            }
        }

        Ok(Self {
            texture,
            view,
            sampler,
            layer_count: layer_count_u32,
            layer_centers,
        })
    }

    pub(crate) fn layer_centers(&self) -> [f32; MAX_LAYERS] {
        self.layer_centers
    }

    /// Calculate maximum texture dimension based on layer count and memory budget
    fn calculate_max_texture_dimension(
        layer_count: usize,
    ) -> (u32, crate::util::memory_budget::TextureQualityTier) {
        use crate::util::memory_budget::{self, TextureQualityTier};

        let layer_count_u32 = layer_count.max(1) as u32;
        let tiers = [
            (TextureQualityTier::Ultra, 4096),
            (TextureQualityTier::High, 2048),
            (TextureQualityTier::Medium, 1024),
            (TextureQualityTier::Low, 512),
        ];

        for (tier, dim) in tiers {
            let estimated_bytes = memory_budget::estimate_rgba8_texture(dim, dim, layer_count_u32);
            let reserved_for_ibl = 25 * 1024 * 1024;
            if estimated_bytes + reserved_for_ibl < memory_budget::MEMORY_BUDGET_CONSERVATIVE {
                return (dim, tier);
            }
        }

        warn!(
            "All material texture tiers exceed budget, using minimum 256x256 for {} layers",
            layer_count
        );
        (256, TextureQualityTier::Low)
    }
}

#[cfg(feature = "extension-module")]
fn prepare_layer_mips(
    image: Option<DynamicImage>,
    material: &crate::core::material::PbrMaterial,
    width: u32,
    height: u32,
    mip_level_count: u32,
) -> Vec<Vec<u8>> {
    let mut mips = Vec::with_capacity(mip_level_count as usize);
    let mut current_width = width.max(1);
    let mut current_height = height.max(1);

    match image {
        Some(img) => {
            let mut level_image = if img.dimensions() == (width, height) {
                img
            } else {
                img.resize_exact(width, height, FilterType::Lanczos3)
            };

            for level in 0..mip_level_count {
                mips.push(level_image.to_rgba8().into_raw());
                if level + 1 < mip_level_count {
                    current_width = (current_width / 2).max(1);
                    current_height = (current_height / 2).max(1);
                    level_image = level_image.resize_exact(
                        current_width,
                        current_height,
                        FilterType::Lanczos3,
                    );
                }
            }
        }
        None => {
            let color = material.base_color;
            let rgba = [
                (color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                255u8,
            ];

            for level in 0..mip_level_count {
                let mut data = vec![0u8; (current_width as usize) * (current_height as usize) * 4];
                for chunk in data.chunks_exact_mut(4) {
                    chunk.copy_from_slice(&rgba);
                }
                mips.push(data);
                if level + 1 < mip_level_count {
                    current_width = (current_width / 2).max(1);
                    current_height = (current_height / 2).max(1);
                }
            }
        }
    }

    mips
}

#[cfg(feature = "extension-module")]
fn pad_rgba_rows(width: u32, height: u32, pixels: &[u8]) -> (Vec<u8>, u32) {
    let row_bytes = (width as usize) * 4;
    let align = COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_row_bytes = ((row_bytes + align - 1) / align) * align;

    if padded_row_bytes == row_bytes {
        return (pixels.to_vec(), row_bytes as u32);
    }

    let mut padded = vec![0u8; padded_row_bytes * height as usize];
    for row in 0..height as usize {
        let src = row * row_bytes;
        let dst = row * padded_row_bytes;
        padded[dst..dst + row_bytes].copy_from_slice(&pixels[src..src + row_bytes]);
    }

    (padded, padded_row_bytes as u32)
}

#[cfg(feature = "extension-module")]
fn estimate_rgba8_mip_chain(width: u32, height: u32, layers: u32) -> u64 {
    let mut total = 0u64;
    let mut w = width.max(1);
    let mut h = height.max(1);

    loop {
        total += (w as u64) * (h as u64) * (layers as u64) * 4;
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }

    total
}

#[cfg(feature = "extension-module")]
fn compute_mip_level_count(width: u32, height: u32) -> u32 {
    let mut levels = 1u32;
    let mut w = width.max(1);
    let mut h = height.max(1);

    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        levels += 1;
    }

    levels
}

#[cfg(feature = "extension-module")]
fn downgrade_tier(
    tier: crate::util::memory_budget::TextureQualityTier,
) -> crate::util::memory_budget::TextureQualityTier {
    use crate::util::memory_budget::TextureQualityTier::{High, Low, Medium, Ultra};
    match tier {
        Ultra => High,
        High => Medium,
        Medium => Low,
        Low => Low,
    }
}
