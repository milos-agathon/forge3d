#[cfg(feature = "extension-module")]
use image::{imageops::FilterType, DynamicImage, GenericImageView};

#[cfg(feature = "extension-module")]
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

#[cfg(feature = "extension-module")]
pub(super) fn prepare_layer_mips(
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
pub(super) fn pad_rgba_rows(width: u32, height: u32, pixels: &[u8]) -> (Vec<u8>, u32) {
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
pub(super) fn estimate_rgba8_mip_chain(width: u32, height: u32, layers: u32) -> u64 {
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
pub(super) fn compute_mip_level_count(width: u32, height: u32) -> u32 {
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
pub(super) fn downgrade_tier(
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
