#[cfg(feature = "extension-module")]
use anyhow::Result;
#[cfg(feature = "extension-module")]
use image::{DynamicImage, GenericImageView};
#[cfg(feature = "extension-module")]
use log::{info, warn};
#[cfg(feature = "extension-module")]
use std::sync::Arc;
#[cfg(feature = "extension-module")]
use wgpu::{
    AddressMode, Extent3d, FilterMode, ImageDataLayout, Origin3d, SamplerDescriptor,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

use super::gpu_helpers::{
    compute_mip_level_count, downgrade_tier, estimate_rgba8_mip_chain, pad_rgba_rows,
    prepare_layer_mips,
};
use super::MaterialSet;

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
            .map(Arc::clone)
    }
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
                layer_count, MAX_LAYERS
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

        let canonical_size = images
            .iter()
            .find_map(|img| img.as_ref().map(GenericImageView::dimensions));
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
        let mut mip_level_count = compute_mip_level_count(resolved_width, resolved_height);
        let mut final_bytes;

        loop {
            let estimated =
                estimate_rgba8_mip_chain(resolved_width, resolved_height, layer_count_u32);
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
            mip_level_count = compute_mip_level_count(resolved_width, resolved_height);
        }

        if (resolved_width, resolved_height) != original_size {
            info!(
                "Material textures resolved to {}x{} (tier {:?}) from {}x{}",
                resolved_width, resolved_height, selected_tier, original_size.0, original_size.1
            );
        }

        final_bytes = estimate_rgba8_mip_chain(resolved_width, resolved_height, layer_count_u32);
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
            anisotropy_clamp: 16,
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
