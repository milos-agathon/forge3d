use super::*;

impl TerrainScene {
    pub(super) fn ensure_reflection_texture_size(&self, width: u32, height: u32) -> Result<bool> {
        let target_width = (width / 2).max(1);
        let target_height = (height / 2).max(1);

        let mut size = self
            .water_reflection_size
            .lock()
            .map_err(|_| anyhow!("water_reflection_size mutex poisoned"))?;

        if size.0 == target_width && size.1 == target_height {
            return Ok(false);
        }

        log::info!(
            target: "terrain.water_reflection",
            "P4: Recreating reflection textures: {}x{} -> {}x{} (half of {}x{})",
            size.0, size.1, target_width, target_height, width, height
        );

        let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.water_reflection.texture"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let new_depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.water_reflection.depth"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let new_depth_view = new_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut tex = self
            .water_reflection_texture
            .lock()
            .map_err(|_| anyhow!("water_reflection_texture mutex poisoned"))?;
        let mut view = self
            .water_reflection_view
            .lock()
            .map_err(|_| anyhow!("water_reflection_view mutex poisoned"))?;
        let mut depth_tex = self
            .water_reflection_depth_texture
            .lock()
            .map_err(|_| anyhow!("water_reflection_depth_texture mutex poisoned"))?;
        let mut depth_view = self
            .water_reflection_depth_view
            .lock()
            .map_err(|_| anyhow!("water_reflection_depth_view mutex poisoned"))?;

        *tex = new_texture;
        *view = new_view;
        *depth_tex = new_depth_texture;
        *depth_view = new_depth_view;
        *size = (target_width, target_height);

        Ok(true)
    }

    pub(super) fn ensure_height_ao_texture_size(
        &self,
        width: u32,
        height: u32,
        resolution_scale: f32,
    ) -> Result<bool> {
        let target_width = ((width as f32 * resolution_scale) as u32).max(1);
        let target_height = ((height as f32 * resolution_scale) as u32).max(1);

        let mut size = self
            .height_ao_size
            .lock()
            .map_err(|_| anyhow!("height_ao_size mutex poisoned"))?;

        if size.0 == target_width && size.1 == target_height {
            return Ok(false);
        }

        log::info!(
            target: "terrain.height_ao",
            "Recreating height AO texture: {}x{} -> {}x{} (scale={:.2} of {}x{})",
            size.0, size.1, target_width, target_height, resolution_scale, width, height
        );

        let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.height_ao.texture"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let new_storage_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.height_ao.storage_view"),
            ..Default::default()
        });
        let new_sample_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.height_ao.sample_view"),
            ..Default::default()
        });

        let mut tex = self
            .height_ao_texture
            .lock()
            .map_err(|_| anyhow!("height_ao_texture mutex poisoned"))?;
        let mut storage_view = self
            .height_ao_storage_view
            .lock()
            .map_err(|_| anyhow!("height_ao_storage_view mutex poisoned"))?;
        let mut sample_view = self
            .height_ao_sample_view
            .lock()
            .map_err(|_| anyhow!("height_ao_sample_view mutex poisoned"))?;

        *tex = Some(new_texture);
        *storage_view = Some(new_storage_view);
        *sample_view = Some(new_sample_view);
        *size = (target_width, target_height);

        Ok(true)
    }

    pub(super) fn ensure_sun_vis_texture_size(
        &self,
        width: u32,
        height: u32,
        resolution_scale: f32,
    ) -> Result<bool> {
        let target_width = ((width as f32 * resolution_scale) as u32).max(1);
        let target_height = ((height as f32 * resolution_scale) as u32).max(1);

        let mut size = self
            .sun_vis_size
            .lock()
            .map_err(|_| anyhow!("sun_vis_size mutex poisoned"))?;

        if size.0 == target_width && size.1 == target_height {
            return Ok(false);
        }

        log::info!(
            target: "terrain.sun_vis",
            "Recreating sun visibility texture: {}x{} -> {}x{} (scale={:.2} of {}x{})",
            size.0, size.1, target_width, target_height, resolution_scale, width, height
        );

        let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.sun_vis.texture"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let new_storage_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.sun_vis.storage_view"),
            ..Default::default()
        });
        let new_sample_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.sun_vis.sample_view"),
            ..Default::default()
        });

        let mut tex = self
            .sun_vis_texture
            .lock()
            .map_err(|_| anyhow!("sun_vis_texture mutex poisoned"))?;
        let mut storage_view = self
            .sun_vis_storage_view
            .lock()
            .map_err(|_| anyhow!("sun_vis_storage_view mutex poisoned"))?;
        let mut sample_view = self
            .sun_vis_sample_view
            .lock()
            .map_err(|_| anyhow!("sun_vis_sample_view mutex poisoned"))?;

        *tex = Some(new_texture);
        *storage_view = Some(new_storage_view);
        *sample_view = Some(new_sample_view);
        *size = (target_width, target_height);

        Ok(true)
    }

    pub(super) fn ensure_accumulation_texture_size(&self, width: u32, height: u32) -> Result<bool> {
        let mut size = self
            .accumulation_size
            .lock()
            .map_err(|_| anyhow!("accumulation_size mutex poisoned"))?;

        if size.0 == width && size.1 == height {
            return Ok(false);
        }

        log::info!(
            target: "terrain.accumulation",
            "M1: Recreating accumulation texture: {}x{} -> {}x{}",
            size.0, size.1, width, height
        );

        let new_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.accumulation.texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.accumulation.view"),
            ..Default::default()
        });

        let mut tex = self
            .accumulation_texture
            .lock()
            .map_err(|_| anyhow!("accumulation_texture mutex poisoned"))?;
        let mut view = self
            .accumulation_view
            .lock()
            .map_err(|_| anyhow!("accumulation_view mutex poisoned"))?;

        *tex = Some(new_texture);
        *view = Some(new_view);
        *size = (width, height);

        Ok(true)
    }

    pub(super) fn clear_accumulation_texture(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _width: u32,
        _height: u32,
    ) -> Result<()> {
        let view_guard = self
            .accumulation_view
            .lock()
            .map_err(|_| anyhow!("accumulation_view mutex poisoned"))?;

        if let Some(view) = view_guard.as_ref() {
            let _ = view;
        }

        Ok(())
    }
}
