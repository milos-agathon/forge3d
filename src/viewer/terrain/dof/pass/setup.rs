use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::DofPass;
use crate::viewer::terrain::dof::shader::DOF_SHADER;
use crate::viewer::terrain::dof::DofUniforms;

impl DofPass {
    pub fn new(device: Arc<wgpu::Device>, surface_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof.bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof.shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_SHADER.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("dof.color_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof.uniforms_h"),
            contents: bytemuck::cast_slice(&[DofUniforms {
                screen_dims: [1.0, 1.0, 1.0, 1.0],
                dof_params: [500.0, 5.6, 50.0, 16.0],
                dof_params2: [1.0, 10000.0, 0.0, 8.0],
                camera_params: [24.0, 500.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof.uniforms_v"),
            contents: bytemuck::cast_slice(&[DofUniforms {
                screen_dims: [1.0, 1.0, 1.0, 1.0],
                dof_params: [500.0, 5.6, 50.0, 16.0],
                dof_params2: [1.0, 10000.0, 1.0, 8.0],
                camera_params: [24.0, 500.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer_h,
            uniform_buffer_v,
            input_texture: None,
            input_view: None,
            intermediate_texture: None,
            intermediate_view: None,
            current_size: (0, 0),
        }
    }

    pub(super) fn ensure_textures(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) {
        if self.current_size != (width, height) || self.input_texture.is_none() {
            let input_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dof.input"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.input_view = Some(input_tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.input_texture = Some(input_tex);

            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dof.intermediate"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.intermediate_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.intermediate_texture = Some(tex);
            self.current_size = (width, height);
        }
    }

    pub fn get_input_view(
        &mut self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> &wgpu::TextureView {
        self.ensure_textures(width, height, format);
        self.input_view.as_ref().unwrap()
    }
}
