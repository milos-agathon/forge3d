use super::*;
use crate::terrain::render_params;

pub(super) struct HeightfieldInitResources {
    pub(super) ao_debug_sampler: wgpu::Sampler,
    pub(super) ao_debug_fallback_texture: wgpu::Texture,
    pub(super) ao_debug_fallback_view: wgpu::TextureView,
    pub(super) height_ao_fallback_view: wgpu::TextureView,
    pub(super) height_ao_sampler: wgpu::Sampler,
    pub(super) height_ao_compute_pipeline: wgpu::ComputePipeline,
    pub(super) height_ao_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) height_ao_uniform_buffer: wgpu::Buffer,
    pub(super) sun_vis_fallback_view: wgpu::TextureView,
    pub(super) sun_vis_sampler: wgpu::Sampler,
    pub(super) sun_vis_compute_pipeline: wgpu::ComputePipeline,
    pub(super) sun_vis_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sun_vis_uniform_buffer: wgpu::Buffer,
}

pub(super) fn create_heightfield_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> HeightfieldInitResources {
    let ao_debug_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.ao_debug.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let height_ao_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.height_ao_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &height_ao_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&[1.0f32]),
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
    let height_ao_fallback_view =
        height_ao_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let height_ao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.height_ao.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let sun_vis_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.sun_vis_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &sun_vis_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&[1.0f32]),
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
    let sun_vis_fallback_view =
        sun_vis_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sun_vis_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.sun_vis.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let height_ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("heightfield_ao.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/heightfield_ao.wgsl").into()),
    });
    let height_ao_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("height_ao.bind_group_layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let height_ao_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("height_ao.pipeline_layout"),
            bind_group_layouts: &[&height_ao_bind_group_layout],
            push_constant_ranges: &[],
        });
    let height_ao_compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("height_ao.compute_pipeline"),
            layout: Some(&height_ao_pipeline_layout),
            module: &height_ao_shader,
            entry_point: "main",
        });
    let height_ao_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("height_ao.uniform_buffer"),
        contents: bytemuck::bytes_of(&HeightAoUniforms {
            params0: [6.0, 16.0, 200.0, 1.0],
            params1: [1.0, 1.0, 1.0, 0.0],
            params2: [1.0, 1.0, 1.0, 1.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let sun_vis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("heightfield_sun_vis.wgsl"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/heightfield_sun_vis.wgsl").into(),
        ),
    });
    let sun_vis_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun_vis.bind_group_layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let sun_vis_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sun_vis.pipeline_layout"),
        bind_group_layouts: &[&sun_vis_bind_group_layout],
        push_constant_ranges: &[],
    });
    let sun_vis_compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sun_vis.compute_pipeline"),
            layout: Some(&sun_vis_pipeline_layout),
            module: &sun_vis_shader,
            entry_point: "main",
        });
    let sun_vis_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sun_vis.uniform_buffer"),
        contents: bytemuck::bytes_of(&SunVisUniforms {
            params0: [4.0, 24.0, 400.0, 1.0],
            params1: [1.0, 1.0, 1.0, 0.0],
            params2: [1.0, 1.0, 1.0, 1.0],
            params3: [0.0, 1.0, 0.0, 0.01],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let ao_debug_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.ao_debug_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &ao_debug_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&[1.0f32]),
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
    let ao_debug_fallback_view =
        ao_debug_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    HeightfieldInitResources {
        ao_debug_sampler,
        ao_debug_fallback_texture,
        ao_debug_fallback_view,
        height_ao_fallback_view,
        height_ao_sampler,
        height_ao_compute_pipeline,
        height_ao_bind_group_layout,
        height_ao_uniform_buffer,
        sun_vis_fallback_view,
        sun_vis_sampler,
        sun_vis_compute_pipeline,
        sun_vis_bind_group_layout,
        sun_vis_uniform_buffer,
    }
}

impl TerrainScene {
    pub(super) fn compute_height_ao_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        heightmap_view: &wgpu::TextureView,
        internal_width: u32,
        internal_height: u32,
        width: u32,
        height: u32,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
    ) -> Result<bool> {
        if !decoded.height_ao.enabled {
            return Ok(false);
        }

        let ao_resolution_scale = decoded.height_ao.resolution_scale.clamp(0.1, 1.0);
        self.ensure_height_ao_texture_size(internal_width, internal_height, ao_resolution_scale)?;

        let ao_size = self
            .height_ao_size
            .lock()
            .map_err(|_| anyhow!("height_ao_size mutex poisoned"))?;
        let (ao_width, ao_height) = *ao_size;
        drop(ao_size);

        let ao_uniforms = HeightAoUniforms {
            params0: [
                decoded.height_ao.directions as f32,
                decoded.height_ao.steps as f32,
                decoded.height_ao.max_distance,
                decoded.height_ao.strength,
            ],
            params1: [
                params.terrain_span / width as f32,
                params.terrain_span / height as f32,
                params.z_scale,
                decoded.clamp.height_range.0,
            ],
            params2: [
                ao_width as f32,
                ao_height as f32,
                width as f32,
                height as f32,
            ],
        };
        self.queue.write_buffer(
            &self.height_ao_uniform_buffer,
            0,
            bytemuck::bytes_of(&ao_uniforms),
        );

        let storage_view_guard = self
            .height_ao_storage_view
            .lock()
            .map_err(|_| anyhow!("height_ao_storage_view mutex poisoned"))?;
        let storage_view = storage_view_guard
            .as_ref()
            .ok_or_else(|| anyhow!("height_ao_storage_view not initialized"))?;

        let ao_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height_ao.bind_group"),
            layout: &self.height_ao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.height_ao_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(storage_view),
                },
            ],
        });
        drop(storage_view_guard);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("height_ao.compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.height_ao_compute_pipeline);
            compute_pass.set_bind_group(0, &ao_bind_group, &[]);
            compute_pass.dispatch_workgroups((ao_width + 7) / 8, (ao_height + 7) / 8, 1);
        }

        Ok(true)
    }

    pub(super) fn compute_sun_visibility_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        heightmap_view: &wgpu::TextureView,
        internal_width: u32,
        internal_height: u32,
        width: u32,
        height: u32,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
    ) -> Result<bool> {
        if !decoded.sun_visibility.enabled {
            return Ok(false);
        }

        let sv_resolution_scale = decoded.sun_visibility.resolution_scale.clamp(0.1, 1.0);
        self.ensure_sun_vis_texture_size(internal_width, internal_height, sv_resolution_scale)?;

        let sv_size = self
            .sun_vis_size
            .lock()
            .map_err(|_| anyhow!("sun_vis_size mutex poisoned"))?;
        let (sv_width, sv_height) = *sv_size;
        drop(sv_size);

        let sun_dir = glam::Vec3::new(
            -decoded.light.direction[0],
            -decoded.light.direction[1],
            -decoded.light.direction[2],
        )
        .normalize();

        let sv_uniforms = SunVisUniforms {
            params0: [
                decoded.sun_visibility.samples as f32,
                decoded.sun_visibility.steps as f32,
                decoded.sun_visibility.max_distance,
                decoded.sun_visibility.softness,
            ],
            params1: [
                params.terrain_span / width as f32,
                params.terrain_span / height as f32,
                params.z_scale,
                decoded.clamp.height_range.0,
            ],
            params2: [
                sv_width as f32,
                sv_height as f32,
                width as f32,
                height as f32,
            ],
            params3: [sun_dir.x, sun_dir.y, sun_dir.z, decoded.sun_visibility.bias],
        };
        self.queue.write_buffer(
            &self.sun_vis_uniform_buffer,
            0,
            bytemuck::bytes_of(&sv_uniforms),
        );

        let storage_view_guard = self
            .sun_vis_storage_view
            .lock()
            .map_err(|_| anyhow!("sun_vis_storage_view mutex poisoned"))?;
        let storage_view = storage_view_guard
            .as_ref()
            .ok_or_else(|| anyhow!("sun_vis_storage_view not initialized"))?;

        let sv_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sun_vis.bind_group"),
            layout: &self.sun_vis_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sun_vis_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(storage_view),
                },
            ],
        });
        drop(storage_view_guard);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sun_vis.compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.sun_vis_compute_pipeline);
            compute_pass.set_bind_group(0, &sv_bind_group, &[]);
            compute_pass.dispatch_workgroups((sv_width + 7) / 8, (sv_height + 7) / 8, 1);
        }

        Ok(true)
    }
}
