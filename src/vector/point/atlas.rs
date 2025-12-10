use crate::core::error::RenderError;

/// H21: Texture atlas configuration
#[derive(Debug)]
pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,     // Size of each tile in pixels
    pub tiles_per_row: u32, // Number of tiles per row
}

impl TextureAtlas {
    /// Create a new texture atlas
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        tile_size: u32,
        data: &[u8],
    ) -> Result<Self, RenderError> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Point.TextureAtlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
                bytes_per_row: Some(width * 4), // RGBA8
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vf.Vector.Point.AtlasSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let tiles_per_row = width / tile_size;

        Ok(Self {
            texture,
            view,
            sampler,
            width,
            height,
            tile_size,
            tiles_per_row,
        })
    }

    /// Get UV coordinates for a tile index
    pub fn get_tile_uv(&self, tile_index: u32) -> ([f32; 2], [f32; 2]) {
        let tile_x = tile_index % self.tiles_per_row;
        let tile_y = tile_index / self.tiles_per_row;

        let u_size = self.tile_size as f32 / self.width as f32;
        let v_size = self.tile_size as f32 / self.height as f32;

        let u_min = tile_x as f32 * u_size;
        let v_min = tile_y as f32 * v_size;
        let u_max = u_min + u_size;
        let v_max = v_min + v_size;

        ([u_min, v_min], [u_max, v_max])
    }
}
