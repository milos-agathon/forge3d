// tests/test_p6_godrays.rs
// P6 Phase 3: God-rays shadow integration test
//
// This test drives the cs_volumetric compute shader with a synthetic
// shadow map to verify that volumetric scattering responds to the
// shadow map bindings (shadow_map, shadow_sampler, shadow_matrix).
// It compares average fog RGB between a fully lit case and a shadowed
// case and asserts that shadows reduce sun-driven in-scattering.

use wgpu::util::DeviceExt;
use wgpu::{DeviceDescriptor, Instance, InstanceDescriptor, RequestAdapterOptions, TextureFormat, TextureViewDimension};

fn create_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let limits = wgpu::Limits::downlevel_defaults();
    let desc = DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        label: Some("p6_godrays_test_device"),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc, None)).ok()?;
    Some((device, queue))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumetricParams {
    density: f32,
    height_falloff: f32,
    phase_g: f32,
    max_steps: u32,
    start_distance: f32,
    max_distance: f32,
    scattering_color: [f32; 3],
    absorption: f32,
    sun_direction: [f32; 3],
    sun_intensity: f32,
    ambient_color: [f32; 3],
    temporal_alpha: f32,
    use_shadows: u32,
    jitter_strength: f32,
    frame_index: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    eye_position: [f32; 3],
    near: f32,
    far: f32,
    _pad: [f32; 3],
}

fn ident4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn run_fog_frame_with_shadow(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: (u32, u32),
    params: VolumetricParams,
    shadow_depth_value: f32,
) -> wgpu::Texture {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog.godrays.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/shaders/volumetric.wgsl").into()),
    });

    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog.godrays.bgl0"),
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
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog.godrays.bgl1"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
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

    let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog.godrays.bgl2"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
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
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fog.godrays.pl"),
        bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fog.godrays.pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "cs_volumetric",
    });

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.godrays.depth"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut depth_bytes = vec![0u8; (size.0 * size.1 * 2) as usize];
    for y in 0..size.1 as usize {
        for x in 0..size.0 as usize {
            let idx = (y * size.0 as usize + x) * 2;
            depth_bytes[idx] = 0x00;
            depth_bytes[idx + 1] = 0x38; // f16(0.5)
        }
    }
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &depth_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &depth_bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(size.0 * 2),
            rows_per_image: Some(size.1),
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
    let depth_sam = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog.godrays.depth.sam"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.godrays.shadow"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let shadow_bytes = shadow_depth_value.to_le_bytes();
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &shadow_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        &shadow_bytes,
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

    let shadow_sam = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog.godrays.shadow.sam"),
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let shadow_mat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fog.godrays.shadow.mat"),
        contents: bytemuck::cast_slice(&ident4()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fog.godrays.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let cam = CameraUniforms {
        view: ident4(),
        proj: ident4(),
        inv_view: ident4(),
        inv_proj: ident4(),
        view_proj: ident4(),
        eye_position: [0.0, 0.0, 0.0],
        near: 0.1,
        far: 1000.0,
        _pad: [0.0; 3],
    };
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fog.godrays.cam"),
        contents: bytemuck::bytes_of(&cam),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.godrays.out"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let out_view = output.create_view(&wgpu::TextureViewDescriptor::default());

    let hist = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.godrays.history"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let hist_view = hist.create_view(&wgpu::TextureViewDescriptor::default());
    let hist_sam = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog.godrays.hist.sam"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog.godrays.bg0"),
        layout: &bgl0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cam_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&depth_sam),
            },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog.godrays.bg1"),
        layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&shadow_sam),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: shadow_mat.as_entire_binding(),
            },
        ],
    });

    let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog.godrays.bg2"),
        layout: &bgl2,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&out_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&hist_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&hist_sam),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fog.godrays.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fog.godrays.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.dispatch_workgroups((size.0 + 7) / 8, (size.1 + 7) / 8, 1);
    }
    queue.submit(Some(encoder.finish()));

    output
}

#[test]
fn p6_godrays_shadow_reduces_sun_scattering() {
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping god-rays test: no adapter");
        return;
    };

    let size = (32u32, 32u32);

    let base_params = VolumetricParams {
        density: 0.03,
        height_falloff: 0.1,
        phase_g: 0.7,
        max_steps: 32,
        start_distance: 0.1,
        max_distance: 50.0,
        scattering_color: [1.0, 1.0, 1.0],
        absorption: 1.0,
        sun_direction: [0.0, 1.0, 0.0],
        sun_intensity: 5.0,
        ambient_color: [0.1, 0.1, 0.1],
        temporal_alpha: 0.0,
        use_shadows: 1,
        jitter_strength: 0.8,
        frame_index: 0,
        _pad0: 0,
    };

    // Fully lit case: shadow depth behind geometry (no occlusion).
    let lit_tex = run_fog_frame_with_shadow(&device, &queue, size, base_params, 1.0);
    let lit_bytes = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &lit_tex,
        size,
        TextureFormat::Rgba16Float,
    )
    .expect("readback lit");

    // Shadowed case: depth closer than the sample, causing occlusion.
    let shadowed_tex = run_fog_frame_with_shadow(&device, &queue, size, base_params, 0.25);
    let shadowed_bytes = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &shadowed_tex,
        size,
        TextureFormat::Rgba16Float,
    )
    .expect("readback shadowed");

    // Compute average RGB (in u16 domain) for both cases.
    let mut sum_lit: u64 = 0;
    let mut sum_shadowed: u64 = 0;
    let mut count: u64 = 0;

    for (lit_px, sh_px) in lit_bytes.chunks_exact(8).zip(shadowed_bytes.chunks_exact(8)) {
        // R,G,B are first 6 bytes (3 x f16).
        let r_l = u16::from_le_bytes([lit_px[0], lit_px[1]]) as u64;
        let g_l = u16::from_le_bytes([lit_px[2], lit_px[3]]) as u64;
        let b_l = u16::from_le_bytes([lit_px[4], lit_px[5]]) as u64;
        let r_s = u16::from_le_bytes([sh_px[0], sh_px[1]]) as u64;
        let g_s = u16::from_le_bytes([sh_px[2], sh_px[3]]) as u64;
        let b_s = u16::from_le_bytes([sh_px[4], sh_px[5]]) as u64;

        sum_lit += r_l + g_l + b_l;
        sum_shadowed += r_s + g_s + b_s;
        count += 1;
    }

    assert!(count > 0);

    // Expect less sun-driven scattering when the shadow map occludes the light.
    assert!(
        sum_lit > sum_shadowed,
        "Lit case should have higher average RGB than shadowed case (sum_lit={} sum_shadowed={})",
        sum_lit,
        sum_shadowed
    );
}
