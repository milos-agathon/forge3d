use wgpu::{Device, Extent3d, TextureDimension, TextureFormat};

use crate::core::texture_format::global_format_registry;
use crate::core::texture_layout::mip_extent;

use super::compression::{calculate_mip_levels, compress_rgba_to_format};
use super::{CompressedImage, CompressionOptions};

impl CompressedImage {
    /// Create from raw image data
    pub fn from_rgba_data(
        data: &[u8],
        width: u32,
        height: u32,
        device: &Device,
        options: &CompressionOptions,
    ) -> Result<Self, String> {
        let start_time = std::time::Instant::now();

        if data.len() != (width * height * 4) as usize {
            return Err(format!(
                "Data size {} doesn't match dimensions {}x{}x4",
                data.len(),
                width,
                height
            ));
        }

        let format = options.target_format.unwrap_or_else(|| {
            global_format_registry()
                .select_best_compressed_format(
                    options.use_case,
                    &device.features(),
                    options.quality,
                )
                .unwrap_or(TextureFormat::Bc7RgbaUnorm)
        });
        let format_info = global_format_registry()
            .get_format_info(format)
            .ok_or_else(|| format!("Unsupported format: {:?}", format))?;

        if !global_format_registry().is_format_supported(format, &device.features()) {
            return Err(format!("Format {:?} not supported by device", format));
        }

        // Compress every level the image claims: a base-only payload with
        // `mip_levels > 1` would fail upload with a partial-upload error.
        let level_count = if options.generate_mipmaps {
            calculate_mip_levels(width, height)
        } else {
            1
        };
        let base_extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let compressed_data = if level_count == 1 {
            compress_rgba_to_format(data, width, height, format)?
        } else {
            let mut packed = Vec::new();
            for (mip, rgba) in
                generate_rgba_mip_chain(data, width, height, level_count).iter().enumerate()
            {
                let extent = mip_extent(base_extent, TextureDimension::D2, mip as u32);
                packed.extend_from_slice(&compress_rgba_to_format(
                    rgba,
                    extent.width,
                    extent.height,
                    format,
                )?);
            }
            packed
        };
        let _compression_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(Self {
            data: compressed_data,
            width,
            height,
            mip_levels: level_count,
            format,
            is_srgb: format_info.is_srgb,
            source_format: "RGBA8".to_string(),
        })
    }
}

/// Generate a box-filtered RGBA8 mip chain of `level_count` levels.
///
/// Level extents follow the WebGPU mip rule (`dim >> mip`, floored at 1), so
/// each level's length matches `mip_extent`'s footprint exactly.
pub(super) fn generate_rgba_mip_chain(
    data: &[u8],
    width: u32,
    height: u32,
    level_count: u32,
) -> Vec<Vec<u8>> {
    let base = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let mut levels: Vec<Vec<u8>> = Vec::with_capacity(level_count as usize);
    levels.push(data.to_vec());
    for mip in 1..level_count {
        let src = mip_extent(base, TextureDimension::D2, mip - 1);
        let dst = mip_extent(base, TextureDimension::D2, mip);
        let next = downsample_rgba8(
            levels.last().expect("mip chain starts with the base level"),
            src.width,
            src.height,
            dst.width,
            dst.height,
        );
        levels.push(next);
    }
    levels
}

/// Area-average RGBA8 pixels down to a smaller extent (box filter).
fn downsample_rgba8(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(dst_w as usize * dst_h as usize * 4);
    for y in 0..dst_h {
        let y0 = (y * src_h) / dst_h;
        let y1 = (((y + 1) * src_h) / dst_h).max(y0 + 1).min(src_h);
        for x in 0..dst_w {
            let x0 = (x * src_w) / dst_w;
            let x1 = (((x + 1) * src_w) / dst_w).max(x0 + 1).min(src_w);
            let mut sums = [0u32; 4];
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let offset = ((sy * src_w + sx) * 4) as usize;
                    for channel in 0..4 {
                        sums[channel] += u32::from(src[offset + channel]);
                    }
                    count += 1;
                }
            }
            for sum in sums {
                out.push((sum / count) as u8);
            }
        }
    }
    out
}
