use super::*;

pub(super) struct BaseInitResources {
    pub(super) sampler_linear: wgpu::Sampler,
    pub(super) height_curve_lut_sampler: wgpu::Sampler,
    pub(super) height_curve_identity_texture: wgpu::Texture,
    pub(super) height_curve_identity_view: wgpu::TextureView,
    pub(super) water_mask_fallback_texture: wgpu::Texture,
    pub(super) water_mask_fallback_view: wgpu::TextureView,
    pub(super) detail_normal_fallback_view: wgpu::TextureView,
    pub(super) detail_normal_sampler: wgpu::Sampler,
}

pub(super) struct AccumulationInitResources {
    pub(super) accumulation_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) accumulation_pipeline: wgpu::ComputePipeline,
    pub(super) accumulation_params_buffer: wgpu::Buffer,
}

pub(super) fn create_base_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<BaseInitResources> {
    let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.sampler.nearest"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let height_curve_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.height_curve.lut_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let identity_lut_data: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
    let (height_curve_identity_texture, height_curve_identity_view) =
        TerrainScene::upload_height_curve_lut_internal(device, queue, &identity_lut_data)?;

    let water_mask_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_mask_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &water_mask_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let water_mask_fallback_view =
        water_mask_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let detail_normal_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.detail_normal_fallback"),
        size: wgpu::Extent3d {
            width: 1,
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
            texture: &detail_normal_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[128u8, 128u8, 255u8, 255u8],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let detail_normal_fallback_view =
        detail_normal_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let detail_normal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.detail_normal.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Ok(BaseInitResources {
        sampler_linear,
        height_curve_lut_sampler,
        height_curve_identity_texture,
        height_curve_identity_view,
        water_mask_fallback_texture,
        water_mask_fallback_view,
        detail_normal_fallback_view,
        detail_normal_sampler,
    })
}

pub(super) fn create_accumulation_init_resources(
    device: &wgpu::Device,
) -> AccumulationInitResources {
    let accumulation_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.accumulation.bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

    let accumulation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain.accumulation.shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../../shaders/accumulation_blend.wgsl"
        ))),
    });
    let accumulation_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.accumulation.pipeline_layout"),
            bind_group_layouts: &[&accumulation_bind_group_layout],
            push_constant_ranges: &[],
        });
    let accumulation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain.accumulation.pipeline"),
        layout: Some(&accumulation_pipeline_layout),
        module: &accumulation_shader,
        entry_point: "accumulate",
    });
    let accumulation_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.accumulation.params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    AccumulationInitResources {
        accumulation_bind_group_layout,
        accumulation_pipeline,
        accumulation_params_buffer,
    }
}

impl TerrainScene {
    pub fn light_debug_info(&self) -> PyResult<String> {
        let light_buffer = self
            .light_buffer
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock light buffer: {}", e)))?;
        Ok(light_buffer.debug_info())
    }

    pub fn set_ao_debug_view(&mut self, view: Option<wgpu::TextureView>) {
        self.ao_debug_view = view;
    }

    pub fn compute_coarse_ao_from_heightmap(
        &mut self,
        width: u32,
        height: u32,
        heightmap_data: &[f32],
    ) -> Result<()> {
        let mut ao_data = vec![1.0f32; (width * height) as usize];
        let sample_radius = 8i32;
        let height_scale = 10.0f32;

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let idx = (y as u32 * width + x as u32) as usize;
                let center_h = heightmap_data[idx];

                let mut occlusion = 0.0f32;
                let mut sample_count = 0;

                for dy in -sample_radius..=sample_radius {
                    for dx in -sample_radius..=sample_radius {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nidx = (ny as u32 * width + nx as u32) as usize;
                            let neighbor_h = heightmap_data[nidx];
                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let h_diff = (neighbor_h - center_h) * height_scale;
                            if h_diff > 0.0 {
                                let angle = (h_diff / dist).atan();
                                occlusion += (angle / std::f32::consts::FRAC_PI_2).min(1.0);
                            }
                            sample_count += 1;
                        }
                    }
                }

                if sample_count > 0 {
                    let avg_occlusion = occlusion / sample_count as f32;
                    ao_data[idx] = (1.0 - avg_occlusion.min(0.9)).max(0.01);
                }
            }
        }

        let coarse_ao_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.coarse_ao"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &coarse_ao_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&ao_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let coarse_ao_view = coarse_ao_texture.create_view(&wgpu::TextureViewDescriptor::default());
        log::info!(
            target: "terrain.ao",
            "P5: Computed coarse horizon AO from heightmap ({}x{})",
            width, height
        );

        self.coarse_ao_texture = Some(coarse_ao_texture);
        self.coarse_ao_view = Some(coarse_ao_view);
        Ok(())
    }

    pub fn coarse_ao_view(&self) -> Option<&wgpu::TextureView> {
        self.coarse_ao_view.as_ref()
    }

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
