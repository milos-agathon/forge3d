use super::*;

// ---------- Colormaps ----------

pub struct ColormapLUT {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub format: wgpu::TextureFormat,
}

impl ColormapLUT {
    pub fn new_single_palette(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if data.len() != 256 * 4 {
            return Err(format!(
                "Invalid colormap data length: expected 1024 bytes, got {}",
                data.len()
            )
            .into());
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap1d-lut"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(256 * 4).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(1).unwrap().into()),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap1d-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
            format: wgpu::TextureFormat::Rgba8Unorm,
        })
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        which: ColormapType,
    ) -> Result<(Self, &'static str), Box<dyn std::error::Error>> {
        let name = match which {
            ColormapType::Viridis => "viridis",
            ColormapType::Magma => "magma",
            ColormapType::Terrain => "terrain",
        };

        // R2a: Runtime format selection
        let force_unorm = std::env::var_os("VF_FORCE_LUT_UNORM").is_some();
        let srgb_ok = adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let use_srgb = !force_unorm && srgb_ok;

        let (format, format_name, palette) = if use_srgb {
            // Use sRGB format with PNG bytes as-is
            let palette = decode_png_rgba8(name)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            (
                wgpu::TextureFormat::Rgba8UnormSrgb,
                "Rgba8UnormSrgb",
                palette,
            )
        } else {
            // Use UNORM format with CPU-linearized bytes
            let srgb_palette = decode_png_rgba8(name)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            let palette = to_linear_u8_rgba(&srgb_palette);
            (wgpu::TextureFormat::Rgba8Unorm, "Rgba8Unorm", palette)
        };

        // 256x1 RGBA8
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap-lut"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &palette,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(256 * 4).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(1).unwrap().into()),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Ok((
            Self {
                texture: tex,
                view,
                sampler,
                format,
            },
            format_name,
        ))
    }

    /// Create a multi-palette LUT supporting runtime palette selection
    /// Creates a 256xN texture where N is the number of palettes
    pub fn new_multi_palette(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        palette_names: &[&str],
    ) -> Result<(Self, &'static str), Box<dyn std::error::Error>> {
        if palette_names.is_empty() {
            return Err("At least one palette must be specified".into());
        }

        // R2a: Runtime format selection
        let force_unorm = std::env::var_os("VF_FORCE_LUT_UNORM").is_some();
        let srgb_ok = adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let use_srgb = !force_unorm && srgb_ok;

        let height = palette_names.len() as u32;
        let format = if use_srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let format_name = if use_srgb {
            "Rgba8UnormSrgb"
        } else {
            "Rgba8Unorm"
        };

        // Create 256xN texture for N palettes
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap-lut-multi"),
            size: wgpu::Extent3d {
                width: 256,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create combined palette data
        let mut combined_data = Vec::with_capacity(256 * height as usize * 4);

        for &palette_name in palette_names {
            let _palette_type = map_name_to_type(palette_name)?;

            let palette_data = if use_srgb {
                // Use sRGB format with PNG bytes as-is
                decode_png_rgba8(palette_name).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })?
            } else {
                // Use UNORM format with CPU-linearized bytes
                let srgb_palette = decode_png_rgba8(palette_name).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })?;
                to_linear_u8_rgba(&srgb_palette)
            };

            if palette_data.len() != 256 * 4 {
                return Err(format!(
                    "Invalid palette size for {}: expected 1024 bytes, got {}",
                    palette_name,
                    palette_data.len()
                )
                .into());
            }

            combined_data.extend_from_slice(&palette_data);
        }

        // Upload all palette rows at once
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &combined_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(256 * 4).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
            },
            wgpu::Extent3d {
                width: 256,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap-lut-multi-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok((
            Self {
                texture: tex,
                view,
                sampler,
                format,
            },
            format_name,
        ))
    }

    /// Get the number of palette rows in this LUT
    pub fn palette_count(&self) -> u32 {
        self.texture.height()
    }
}
