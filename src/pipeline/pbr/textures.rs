use super::*;

/// PBR texture set for a material
#[derive(Debug)]
pub struct PbrTextures {
    /// Base color (albedo) texture - RGBA8
    pub base_color: Option<Texture>,

    /// Metallic-roughness texture - RG format (B=metallic, G=roughness)
    pub metallic_roughness: Option<Texture>,

    /// Normal map texture - RGB format (tangent space)
    pub normal: Option<Texture>,

    /// Ambient occlusion texture - R format
    pub occlusion: Option<Texture>,

    /// Emissive texture - RGB format
    pub emissive: Option<Texture>,
}

/// Create texture from raw data
pub(super) fn create_texture_from_data(
    device: &Device,

    queue: &Queue,
    label: &str,
    data: &[u8],
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Texture {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Calculate bytes per pixel based on format
    let bytes_per_pixel = match format {
        TextureFormat::R8Unorm => 1,
        TextureFormat::Rg8Unorm => 2,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => 4,
        _ => 4, // Default to 4 bytes
    };

    let bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((bytes_per_row + alignment - 1) / alignment) * alignment
    };

    // Create padded data if necessary
    if bytes_per_row == padded_bytes_per_row {
        // No padding needed
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        // Need to pad rows
        let mut padded_data = vec![0u8; (padded_bytes_per_row * height) as usize];
        for y in 0..height {
            let src_offset = (y * bytes_per_row) as usize;
            let dst_offset = (y * padded_bytes_per_row) as usize;
            let src_range = src_offset..(src_offset + bytes_per_row as usize);
            let dst_range = dst_offset..(dst_offset + bytes_per_row as usize);

            if src_range.end <= data.len() && dst_range.end <= padded_data.len() {
                padded_data[dst_range].copy_from_slice(&data[src_range]);
            }
        }

        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    texture
}

/// Create default 1x1 texture with specific color
pub(super) fn create_default_texture(
    device: &Device,
    queue: &Queue,
    label: &str,
    color: [u8; 4],
) -> Texture {
    create_texture_from_data(
        device,
        queue,
        label,
        &color,
        1,
        1,
        TextureFormat::Rgba8Unorm,
    )
}

/// Create PBR material sampler
pub fn create_pbr_sampler(device: &Device) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some("pbr_material_sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    })
}
