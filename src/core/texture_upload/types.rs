/// Configuration for HDR texture creation
#[derive(Debug, Clone)]
pub struct HdrTextureConfig {
    pub label: Option<String>,
    pub width: u32,
    pub height: u32,
    pub format: HdrFormat,
    pub usage: wgpu::TextureUsages,
    pub generate_mipmaps: bool,
}

/// HDR texture format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrFormat {
    Rgba16Float,
    Rgba32Float,
}

impl HdrFormat {
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            HdrFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            HdrFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            HdrFormat::Rgba16Float => 8,
            HdrFormat::Rgba32Float => 16,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            HdrFormat::Rgba16Float => "Rgba16Float",
            HdrFormat::Rgba32Float => "Rgba32Float",
        }
    }
}

impl Default for HdrTextureConfig {
    fn default() -> Self {
        Self {
            label: None,
            width: 1,
            height: 1,
            format: HdrFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            generate_mipmaps: false,
        }
    }
}

/// Created HDR texture with associated resources
pub struct HdrTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: HdrFormat,
    pub width: u32,
    pub height: u32,
}

impl HdrTexture {
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    pub fn size_bytes(&self) -> usize {
        self.pixel_count() * self.format.bytes_per_pixel()
    }

    pub fn create_sampler(&self, linear_filtering: bool) -> wgpu::Sampler {
        let g = crate::core::gpu::ctx();
        let filter = if linear_filtering {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        };

        g.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr-texture-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        })
    }
}
