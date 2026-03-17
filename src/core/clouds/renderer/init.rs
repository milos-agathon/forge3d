use super::*;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    FragmentState, FrontFace, MultisampleState, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureSampleType, TextureViewDimension,
    VertexBufferLayout, VertexState, VertexStepMode,
};

impl CloudRenderer {
    pub fn new(device: &Device, color_format: TextureFormat, sample_count: u32) -> Self {
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_uniform_buffer"),
            size: std::mem::size_of::<CloudUniforms>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout_uniforms =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("cloud_bind_group_layout_uniforms"),
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
        let bind_group_layout_textures =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("cloud_bind_group_layout_textures"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let bind_group_layout_ibl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_bind_group_layout_ibl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let (vertex_buffer, index_buffer, index_count) = Self::create_cloud_quad_geometry(device);

        let cloud_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_density_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shape_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_shape_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let ibl_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_ibl_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_uniforms = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_bind_group_uniforms"),
            layout: &bind_group_layout_uniforms,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cloud_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../../shaders/clouds.wgsl"))),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_pipeline_layout"),
            bind_group_layouts: &[
                &bind_group_layout_uniforms,
                &bind_group_layout_textures,
                &bind_group_layout_ibl,
            ],
            push_constant_ranges: &[],
        });

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 8]>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
        };
        let cloud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: sample_count.max(1),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let uniforms = CloudUniforms::default();
        let mut params = CloudParams::default();
        params.quality = match sample_count {
            0 | 1 | 2 => CloudQuality::Medium,
            4 => CloudQuality::High,
            _ => CloudQuality::High,
        };

        let mut renderer = Self {
            uniforms,
            params,
            uniform_buffer,
            cloud_pipeline,
            compute_pipeline: None,
            vertex_buffer,
            index_buffer,
            index_count,
            bind_group_layout_uniforms,
            bind_group_layout_textures,
            bind_group_layout_ibl,
            bind_group_uniforms,
            bind_group_textures: None,
            bind_group_ibl: None,
            noise_texture: None,
            noise_view: None,
            shape_texture: None,
            shape_view: None,
            ibl_irradiance_texture: None,
            ibl_irradiance_view: None,
            ibl_prefilter_texture: None,
            ibl_prefilter_view: None,
            cloud_sampler,
            shape_sampler,
            ibl_sampler,
            noise_resolution: 0,
            time: 0.0,
            enabled: true,
        };
        renderer.update_uniforms();
        renderer
    }
}
