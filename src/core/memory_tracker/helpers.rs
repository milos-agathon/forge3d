use wgpu::{BufferUsages, TextureFormat};

/// Calculate texture size in bytes based on dimensions and format.
pub fn calculate_texture_size(width: u32, height: u32, format: TextureFormat) -> u64 {
    let bytes_per_pixel = match format {
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint => 1,

        TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint
        | TextureFormat::R16Uint
        | TextureFormat::R16Sint
        | TextureFormat::R16Float
        | TextureFormat::Depth16Unorm => 2,

        TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rgb10a2Uint
        | TextureFormat::Rg11b10Float
        | TextureFormat::Rg16Uint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rg16Float
        | TextureFormat::R32Uint
        | TextureFormat::R32Sint
        | TextureFormat::R32Float
        | TextureFormat::Depth32Float
        | TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8 => 4,

        TextureFormat::Rgba16Uint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba16Float
        | TextureFormat::Rg32Uint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rg32Float
        | TextureFormat::Depth32FloatStencil8 => 8,

        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => 16,

        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::Bc2RgbaUnorm | TextureFormat::Bc2RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::Etc2Rgb8A1Unorm | TextureFormat::Etc2Rgb8A1UnormSrgb => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::Astc { .. } => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        _ => 4,
    };

    (width as u64) * (height as u64) * bytes_per_pixel
}

pub fn calculate_compressed_texture_size(
    width: u32,
    height: u32,
    bytes_per_block: u64,
    block_size: u32,
) -> u64 {
    let blocks_x = width.div_ceil(block_size);
    let blocks_y = height.div_ceil(block_size);
    (blocks_x as u64) * (blocks_y as u64) * bytes_per_block
}

pub fn is_host_visible_usage(usage: BufferUsages) -> bool {
    usage.contains(BufferUsages::MAP_READ) || usage.contains(BufferUsages::MAP_WRITE)
}
