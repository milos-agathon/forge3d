use super::*;

impl PointRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let shader = super::pipelines::create_shader(device);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Point.Uniform"),
            size: std::mem::size_of::<PointUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Point.BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Point.DefaultTexture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Point.BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&default_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&default_sampler),
                },
            ],
        });

        let render_pipeline = super::pipelines::create_render_pipeline(
            device,
            &shader,
            target_format,
            &bind_group_layout,
        );
        let oit_pipeline =
            super::pipelines::create_oit_pipeline(device, &shader, &bind_group_layout);

        let pick_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vf.Vector.Point.PickBindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pick_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Point.PickUniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Point.PickBindGroup"),
            layout: &pick_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pick_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let pick_pipeline =
            super::pipelines::create_pick_pipeline(device, &shader, &pick_bind_group_layout);

        Ok(Self {
            render_pipeline,
            instance_buffer: None,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            pick_pipeline,
            pick_uniform_buffer,
            pick_bind_group,
            oit_pipeline,
            instance_capacity: 0,
            debug_flags: DebugFlags::default(),
            texture_atlas: None,
            enable_clip_w_scaling: false,
            depth_range: (0.1, 1000.0),
            shape_mode: get_global_config().0,
            lod_threshold: get_global_config().1,
        })
    }
}
