use super::*;

impl IBLRenderer {
    pub fn new(device: &wgpu::Device, quality: IBLQuality) -> Self {
        let shader_equirect = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl.precompute.shader.equirect"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/ibl_equirect.wgsl").into(),
            ),
        });
        let shader_prefilter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl.precompute.shader.prefilter"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/ibl_prefilter.wgsl").into(),
            ),
        });
        let shader_brdf = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl.precompute.shader.brdf"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ibl_brdf.wgsl").into()),
        });

        let equirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.precompute.equirect.layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let convolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.precompute.convolve.layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let brdf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.precompute.brdf.layout"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pbr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.runtime.pbr.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let equirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl.precompute.pipeline.equirect"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ibl.precompute.layout.equirect"),
                    bind_group_layouts: &[&equirect_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader_equirect,
            entry_point: "cs_equirect_to_cubemap",
        });

        let irradiance_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ibl.precompute.pipeline.irradiance"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ibl.precompute.layout.irradiance"),
                        bind_group_layouts: &[&convolve_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader_prefilter,
                entry_point: "cs_irradiance_convolve",
            });

        let specular_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl.precompute.pipeline.specular"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ibl.precompute.layout.specular"),
                    bind_group_layouts: &[&convolve_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader_prefilter,
            entry_point: "cs_specular_prefilter",
        });

        let brdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl.precompute.pipeline.brdf"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ibl.precompute.layout.brdf"),
                    bind_group_layouts: &[&brdf_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader_brdf,
            entry_point: "cs_brdf_lut",
        });

        let base_resolution = quality.base_environment_size();
        let uniforms = PrefilterUniforms::new(base_resolution, quality);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibl.prefilter.uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl.runtime.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 16.0,
            ..Default::default()
        });

        // Separate sampler for equirectangular sampling: Repeat on U to avoid horizontal seam
        let equirect_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl.precompute.equirect.sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 16.0,
            ..Default::default()
        });

        Self {
            quality,
            base_resolution,
            equirect_layout,
            convolve_layout,
            brdf_layout,
            pbr_layout,
            equirect_pipeline,
            irradiance_pipeline,
            specular_pipeline,
            brdf_pipeline,
            uniforms,
            uniform_buffer,
            environment_equirect: None,
            environment_cubemap: None,
            environment_view: None,
            irradiance_map: None,
            irradiance_view: None,
            specular_map: None,
            specular_view: None,
            brdf_lut: None,
            brdf_view: None,
            specular_size_override: None,
            irradiance_size_override: None,
            brdf_size_override: None,
            env_sampler,
            equirect_sampler,
            cache: None,
            pbr_bind_group: None,
            is_initialized: false,
        }
    }
}
