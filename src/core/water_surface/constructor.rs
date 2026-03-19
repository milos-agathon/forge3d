use super::*;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    Extent3d, FilterMode, FragmentState, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages,
    TextureViewDescriptor, TextureViewDimension, VertexBufferLayout, VertexState, VertexStepMode,
};

impl WaterSurfaceRenderer {
    pub fn new(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> Self {
        let params = WaterSurfaceParams::default();
        let uniforms = WaterSurfaceUniforms::default();

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("water_surface_uniform_buffer"),
            size: std::mem::size_of::<WaterSurfaceUniforms>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_surface_bind_group_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_surface_bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let mask_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("water_mask_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        let mask_size = (1u32, 1u32);
        let mask_texture = device.create_texture(&TextureDescriptor {
            label: Some("water_mask_texture"),
            size: Extent3d {
                width: mask_size.0,
                height: mask_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mask_view = mask_texture.create_view(&TextureViewDescriptor::default());

        let mask_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_surface_mask_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let mask_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_surface_mask_bind_group"),
            layout: &mask_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&mask_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("water_surface_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/water_surface.wgsl"
            ))),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("water_surface_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout, &mask_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 8]>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
        };
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water_surface_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let (vertex_buffer, index_buffer, index_count) =
            Self::create_water_surface_geometry(device, params.size);

        Self {
            uniforms,
            params,
            uniform_buffer,
            water_pipeline,
            bind_group_layout,
            bind_group,
            mask_bind_group_layout,
            mask_bind_group,
            mask_texture,
            mask_view,
            mask_sampler,
            mask_size,
            vertex_buffer,
            index_buffer,
            index_count,
            animation_time: 0.0,
            enabled: true,
        }
    }

    fn create_water_surface_geometry(device: &Device, size: f32) -> (Buffer, Buffer, u32) {
        let half_size = size * 0.5;
        let subdivisions = 32;
        let step = size / subdivisions as f32;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for y in 0..=subdivisions {
            for x in 0..=subdivisions {
                let world_x = -half_size + x as f32 * step;
                let world_z = -half_size + y as f32 * step;
                let u = x as f32 / subdivisions as f32;
                let v = y as f32 / subdivisions as f32;
                vertices.extend_from_slice(&[world_x, 0.0, world_z, u, v, 0.0, 1.0, 0.0]);
            }
        }

        for y in 0..subdivisions {
            for x in 0..subdivisions {
                let i0 = y * (subdivisions + 1) + x;
                let i1 = i0 + 1;
                let i2 = i0 + (subdivisions + 1);
                let i3 = i2 + 1;
                indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
            }
        }

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("water_surface_vertex_buffer"),
            size: (vertices.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("water_surface_index_buffer"),
            size: (indices.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        index_buffer.unmap();

        (vertex_buffer, index_buffer, indices.len() as u32)
    }
}
