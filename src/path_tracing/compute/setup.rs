use super::*;

pub(super) fn create_dispatch_resources(
    width: u32,
    height: u32,
    spheres: &[Sphere],
    uniforms: Uniforms,
) -> DispatchResources {
    let g = ctx();
    let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pt_kernel"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_kernel.wgsl").into()),
    });

    let bgl0 = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt-bgl0-uniforms"),
            entries: &[uniform_buffer_entry(0)],
        });
    let bgl1 = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt-bgl1-scene"),
            entries: &[
                storage_buffer_entry(0, true),
                storage_buffer_entry(1, true),
                storage_buffer_entry(2, true),
                storage_buffer_entry(3, true),
            ],
        });
    let bgl2 = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt-bgl2-accum"),
            entries: &[storage_buffer_entry(0, false)],
        });
    let bgl3 = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt-bgl3-out"),
            entries: &[storage_texture_entry(0, wgpu::TextureFormat::Rgba16Float)],
        });
    let bgl4 = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt-bgl4-aovs"),
            entries: &[
                storage_texture_entry(0, wgpu::TextureFormat::Rgba16Float),
                storage_texture_entry(1, wgpu::TextureFormat::Rgba16Float),
                storage_texture_entry(2, wgpu::TextureFormat::R32Float),
                storage_texture_entry(3, wgpu::TextureFormat::Rgba16Float),
                storage_texture_entry(4, wgpu::TextureFormat::Rgba16Float),
                storage_texture_entry(5, wgpu::TextureFormat::Rgba16Float),
                storage_texture_entry(6, wgpu::TextureFormat::R8Unorm),
            ],
        });

    let pipeline_layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt-pipeline-layout"),
            bind_group_layouts: &[&bgl0, &bgl1, &bgl2, &bgl3, &bgl4],
            push_constant_ranges: &[],
        });
    let pipeline = g
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pt-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

    let ubo = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pt-ubo"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let scene_buf = g
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pt-scene"),
            contents: bytemuck::cast_slice(spheres),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let (mesh_vertices, mesh_indices, mesh_bvh) = create_empty_mesh_buffers(&g.device);
    let accum_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pt-accum"),
        size: (width as u64) * (height as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pt-out-tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pt-bg0"),
        layout: &bgl0,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ubo.as_entire_binding(),
        }],
    });
    let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pt-bg1"),
        layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mesh_vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: mesh_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: mesh_bvh.as_entire_binding(),
            },
        ],
    });
    let bg2 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pt-bg2"),
        layout: &bgl2,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: accum_buf.as_entire_binding(),
        }],
    });
    let bg3 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pt-bg3"),
        layout: &bgl3,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }],
    });

    let aovs_all = [
        AovKind::Albedo,
        AovKind::Normal,
        AovKind::Depth,
        AovKind::Direct,
        AovKind::Indirect,
        AovKind::Emission,
        AovKind::Visibility,
    ];
    let aov_frames = AovFrames::new(&g.device, width, height, &aovs_all);
    let aov_views: Vec<wgpu::TextureView> = aovs_all
        .iter()
        .map(|kind| {
            aov_frames
                .get_texture(*kind)
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default())
        })
        .collect();
    let bg4 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pt-bg4"),
        layout: &bgl4,
        entries: &[
            texture_view_entry(0, &aov_views[0]),
            texture_view_entry(1, &aov_views[1]),
            texture_view_entry(2, &aov_views[2]),
            texture_view_entry(3, &aov_views[3]),
            texture_view_entry(4, &aov_views[4]),
            texture_view_entry(5, &aov_views[5]),
            texture_view_entry(6, &aov_views[6]),
        ],
    });

    DispatchResources {
        pipeline,
        bg0,
        bg1,
        bg2,
        bg3,
        bg4,
        out_tex,
        aov_frames,
    }
}

fn uniform_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_texture_entry(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn texture_view_entry<'a>(
    binding: u32,
    view: &'a wgpu::TextureView,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}
