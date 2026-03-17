use super::*;
use wgpu::{ImageCopyTexture, ImageDataLayout, TextureViewDescriptor, TextureViewDimension};

impl CloudRenderer {
    pub(super) fn recreate_noise_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
    ) -> Result<(), String> {
        let resolution = self.desired_noise_resolution();
        let (data, padded_row) = Self::build_noise_data(resolution);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_noise_texture"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
        );
        self.noise_texture = Some(texture);
        self.noise_view = self
            .noise_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.noise_resolution = resolution;
        self.bind_group_textures = None;
        Ok(())
    }

    pub(super) fn recreate_shape_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
    ) -> Result<(), String> {
        let size = 256;
        let (data, padded_row) = Self::build_shape_data(size);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_shape_texture"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        self.shape_texture = Some(texture);
        self.shape_view = self
            .shape_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.bind_group_textures = None;
        Ok(())
    }

    pub(super) fn recreate_default_ibl(
        &mut self,
        device: &Device,
        queue: &Queue,
    ) -> Result<(), String> {
        let padded_row = Self::align_to(4, 256);
        let mut irradiance_data = vec![0u8; padded_row as usize * 6];
        let mut prefilter_data = vec![0u8; padded_row as usize * 6];
        let irradiance_colors = [
            [0.62, 0.72, 0.92],
            [0.58, 0.68, 0.88],
            [0.70, 0.80, 0.95],
            [0.64, 0.74, 0.90],
            [0.55, 0.65, 0.85],
            [0.60, 0.70, 0.89],
        ];
        let prefilter_colors = [
            [0.80, 0.82, 0.86],
            [0.78, 0.80, 0.84],
            [0.82, 0.84, 0.88],
            [0.76, 0.78, 0.82],
            [0.74, 0.76, 0.80],
            [0.83, 0.85, 0.89],
        ];

        for (layer, color) in irradiance_colors.iter().enumerate() {
            let offset = layer * padded_row as usize;
            irradiance_data[offset] = Self::float_to_u8(color[0]);
            irradiance_data[offset + 1] = Self::float_to_u8(color[1]);
            irradiance_data[offset + 2] = Self::float_to_u8(color[2]);
            irradiance_data[offset + 3] = 255;
        }
        for (layer, color) in prefilter_colors.iter().enumerate() {
            let offset = layer * padded_row as usize;
            prefilter_data[offset] = Self::float_to_u8(color[0]);
            prefilter_data[offset + 1] = Self::float_to_u8(color[1]);
            prefilter_data[offset + 2] = Self::float_to_u8(color[2]);
            prefilter_data[offset + 3] = 255;
        }

        let irradiance = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_ibl_irradiance"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            ImageCopyTexture {
                texture: &irradiance,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &irradiance_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
        );
        let irradiance_view = irradiance.create_view(&TextureViewDescriptor {
            label: Some("cloud_ibl_irradiance_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        let prefilter = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_ibl_prefilter"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            ImageCopyTexture {
                texture: &prefilter,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &prefilter_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
        );
        let prefilter_view = prefilter.create_view(&TextureViewDescriptor {
            label: Some("cloud_ibl_prefilter_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        self.ibl_irradiance_texture = Some(irradiance);
        self.ibl_irradiance_view = Some(irradiance_view);
        self.ibl_prefilter_texture = Some(prefilter);
        self.ibl_prefilter_view = Some(prefilter_view);
        self.bind_group_ibl = None;
        Ok(())
    }
}
