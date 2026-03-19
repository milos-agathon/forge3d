use super::uniforms::{BloomBlurUniforms, BloomBrightPassUniforms, BloomCompositeUniforms};
use super::TerrainBloomProcessor;
use anyhow::Result;
use std::borrow::Cow;

impl TerrainBloomProcessor {
    /// Create a new bloom processor
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let brightpass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bloom.brightpass_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bloom.blur_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bloom.composite_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let brightpass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.brightpass_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/bloom_brightpass.wgsl"
            ))),
        });
        let blur_h_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.blur_h_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/bloom_blur_h.wgsl"
            ))),
        });
        let blur_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.blur_v_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/bloom_blur_v.wgsl"
            ))),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.composite_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/bloom_composite.wgsl"
            ))),
        });

        let brightpass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain.bloom.brightpass_pipeline_layout"),
                bind_group_layouts: &[&brightpass_layout],
                push_constant_ranges: &[],
            });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.bloom.blur_pipeline_layout"),
            bind_group_layouts: &[&blur_layout],
            push_constant_ranges: &[],
        });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain.bloom.composite_pipeline_layout"),
                bind_group_layouts: &[&composite_layout],
                push_constant_ranges: &[],
            });

        let brightpass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("terrain.bloom.brightpass_pipeline"),
                layout: Some(&brightpass_pipeline_layout),
                module: &brightpass_shader,
                entry_point: "main",
            });
        let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain.bloom.blur_h_pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_h_shader,
            entry_point: "main",
        });
        let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain.bloom.blur_v_pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_v_shader,
            entry_point: "main",
        });
        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain.bloom.composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            module: &composite_shader,
            entry_point: "main",
        });

        let brightpass_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.bloom.brightpass_uniforms"),
            size: std::mem::size_of::<BloomBrightPassUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.bloom.blur_uniforms"),
            size: std::mem::size_of::<BloomBlurUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.bloom.composite_uniforms"),
            size: std::mem::size_of::<BloomCompositeUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            brightpass_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            composite_pipeline,
            brightpass_layout,
            blur_layout,
            composite_layout,
            brightpass_uniform_buffer,
            blur_uniform_buffer,
            composite_uniform_buffer,
            bright_texture: None,
            bright_view: None,
            blur_temp_texture: None,
            blur_temp_view: None,
            blur_result_texture: None,
            blur_result_view: None,
            current_size: (0, 0),
        })
    }

    /// Ensure intermediate textures match the required size
    pub(super) fn ensure_textures(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.current_size == (width, height) && self.bright_texture.is_some() {
            return;
        }

        log::info!(
            target: "terrain.bloom",
            "M2: Creating bloom intermediate textures: {}x{}",
            width,
            height
        );

        let bright_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.bloom.bright_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bright_view = bright_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blur_temp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.bloom.blur_temp_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let blur_temp_view = blur_temp_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blur_result_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.bloom.blur_result_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let blur_result_view =
            blur_result_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.bright_texture = Some(bright_texture);
        self.bright_view = Some(bright_view);
        self.blur_temp_texture = Some(blur_temp_texture);
        self.blur_temp_view = Some(blur_temp_view);
        self.blur_result_texture = Some(blur_result_texture);
        self.blur_result_view = Some(blur_result_view);
        self.current_size = (width, height);
    }
}
