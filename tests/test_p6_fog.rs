// tests/test_p6_fog.rs
// Milestone 3 (P6-08): Volumetric fog acceptance (basic)
// Validates that fog alpha increases with density and temporal smoothing reduces inter-frame variance.

use wgpu::util::DeviceExt;
use wgpu::{
    DeviceDescriptor, Instance, InstanceDescriptor, RequestAdapterOptions, TextureFormat,
    TextureViewDimension,
};

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
        label: Some("p6_fog_test_device"),
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

fn run_fog_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: (u32, u32),
    params: VolumetricParams,
    history_view: &wgpu::TextureView,
) -> (wgpu::Texture, wgpu::TextureView) {
    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog.test.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/shaders/volumetric.wgsl").into()),
    });
    // BGL 0
    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog.test.bgl0"),
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
    // BGL 1 (shadows)
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog.test.bgl1"),
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
    // BGL 2 (output/history)
    let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog.test.bgl2"),
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
        label: Some("fog.test.pl"),
        bind_group_layouts: &[&bgl0, &bgl1, &bgl2],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fog.test.pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: "cs_volumetric",
    });

    // Depth texture (constant depth 0.5)
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.test.depth"),
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
    // Fill depth with mid value 0.5
    let mut depth_bytes = vec![0u8; (size.0 * size.1 * 2) as usize];
    // 0.5 as f16: 0x3800; but to avoid half conversion, just write zeros; near zero means background; we need >0 to march
    // We'll write ~0.5 in f16 by using a 16-bit pattern 0x3800
    for y in 0..size.1 as usize {
        for x in 0..size.0 as usize {
            let idx = (y * size.0 as usize + x) * 2;
            depth_bytes[idx] = 0x00;
            depth_bytes[idx + 1] = 0x38;
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
        label: Some("fog.test.depth.sam"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Shadow map (1x1 depth texture) + sampler + matrix
    let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.test.shadow"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let shadow_sam = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog.test.shadow.sam"),
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let shadow_mat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fog.test.shadow.mat"),
        contents: bytemuck::cast_slice(&ident4()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Buffers & textures
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fog.test.params"),
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
        label: Some("fog.test.cam"),
        contents: bytemuck::bytes_of(&cam),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.test.out"),
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
    let hist_sam = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog.test.hist.sam"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog.test.bg0"),
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
        label: Some("fog.test.bg1"),
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
        label: Some("fog.test.bg2"),
        layout: &bgl2,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&out_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(history_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&hist_sam),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fog.test.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fog.test.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.dispatch_workgroups((size.0 + 7) / 8, (size.1 + 7) / 8, 1);
    }
    queue.submit(Some(encoder.finish()));

    (output, out_view)
}

#[test]
fn fog_alpha_increases_with_density_and_temporal_smoothing() {
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping fog test: no adapter");
        return;
    };
    let size = (32u32, 32u32);

    // History starts as zero texture
    let hist = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog.test.history"),
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

    // params with no fog
    let base_params = VolumetricParams {
        density: 0.0,
        height_falloff: 0.1,
        phase_g: 0.0,
        max_steps: 32,
        start_distance: 0.1,
        max_distance: 100.0,
        scattering_color: [1.0, 1.0, 1.0],
        absorption: 1.0,
        sun_direction: [0.0, 1.0, 0.0],
        sun_intensity: 5.0,
        ambient_color: [0.2, 0.25, 0.3],
        temporal_alpha: 0.0,
        use_shadows: 0,
        jitter_strength: 0.8,
        frame_index: 0,
        _pad0: 0,
    };

    // Frame with zero density
    let (out0, _view0) = run_fog_frame(&device, &queue, size, base_params, &hist_view);
    let data0 = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &out0,
        size,
        TextureFormat::Rgba16Float,
    )
    .expect("readback0");

    // Frame with higher density
    let mut p1 = base_params;
    p1.density = 0.05;
    p1.frame_index = 1;
    p1.temporal_alpha = 0.0;
    let (out1, _view1) = run_fog_frame(&device, &queue, size, p1, &hist_view);
    let data1 = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &out1,
        size,
        TextureFormat::Rgba16Float,
    )
    .expect("readback1");

    // Compare average alpha (last two bytes represent f16 alpha per pixel); For simplicity use u16 reinterpret (approx)
    let mut sum0 = 0u64;
    let mut sum1 = 0u64;
    let mut count = 0u64;
    for px in data0.chunks_exact(8) {
        // RGBA16F => 8 bytes per pixel
        let a16 = u16::from_le_bytes([px[6], px[7]]);
        sum0 += a16 as u64;
        count += 1;
    }
    for px in data1.chunks_exact(8) {
        let a16 = u16::from_le_bytes([px[6], px[7]]);
        sum1 += a16 as u64;
    }
    assert!(sum1 > sum0, "Fog alpha should increase with density");

    // Temporal smoothing: run 2 frames with jitter; compare L1 difference between frames with and without temporal
    let mut p2 = p1;
    p2.frame_index = 2;
    p2.temporal_alpha = 0.0;
    let (out2, _v2) = run_fog_frame(&device, &queue, size, p2, &hist_view);
    let img2 = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &out2,
        size,
        TextureFormat::Rgba16Float,
    )
    .unwrap();

    let mut p3 = p1;
    p3.frame_index = 2;
    p3.temporal_alpha = 0.5;
    let (out3, _v3) = run_fog_frame(&device, &queue, size, p3, &hist_view);
    let img3 = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &out3,
        size,
        TextureFormat::Rgba16Float,
    )
    .unwrap();

    // Compute simple per-pixel L1 sum between frame1 and others; expect temporal (img3) closer to frame1
    let mut l1_no_temporal = 0u64;
    let mut l1_temporal = 0u64;
    for (a, b) in data1.chunks_exact(8).zip(img2.chunks_exact(8)) {
        // compare alpha only for simplicity
        let a16 = u16::from_le_bytes([a[6], a[7]]) as i64;
        let b16 = u16::from_le_bytes([b[6], b[7]]) as i64;
        l1_no_temporal += a16.abs_diff(b16) as u64;
    }
    for (a, b) in data1.chunks_exact(8).zip(img3.chunks_exact(8)) {
        let a16 = u16::from_le_bytes([a[6], a[7]]) as i64;
        let b16 = u16::from_le_bytes([b[6], b[7]]) as i64;
        l1_temporal += a16.abs_diff(b16) as u64;
    }
    assert!(
        l1_temporal < l1_no_temporal,
        "Temporal smoothing should reduce inter-frame difference"
    );
}
