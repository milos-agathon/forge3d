// tests/test_p6_sky.rs
// Milestone 3 (P6-08): Sky acceptance tests (GPU compute)
// Validates that the sky shader renders with plausible chromatic shift near horizon at low sun elevation.

use wgpu::{DeviceDescriptor, Instance, InstanceDescriptor, RequestAdapterOptions, TextureFormat};

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
        label: Some("p6_sky_test_device"),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc, None)).ok()?;
    Some((device, queue))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyParams {
    sun_direction: [f32; 3],
    turbidity: f32,
    ground_albedo: f32,
    model: u32,
    sun_intensity: f32,
    exposure: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    eye_position: [f32; 3],
    _pad0: f32,
}

#[test]
fn sky_midday_vs_low_sun_chromatic_shift() {
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping sky test: no adapter");
        return;
    };

    let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky.test.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/shaders/sky.wgsl").into()),
    });
    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky.test.bgl0"),
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
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky.test.bgl1"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky.test.pl"),
        bind_group_layouts: &[&bgl0, &bgl1],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sky.test.pipeline"),
        layout: Some(&pl),
        module: &sky_shader,
        entry_point: "cs_render_sky",
    });

    let size = (64u32, 64u32);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sky.test.out"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Camera uniforms (identity with plausible inv)
    let ident = glam::Mat4::IDENTITY.to_cols_array_2d();
    let eye = [0.0f32, 0.0, 0.0];
    let cam = CameraUniforms {
        view: ident,
        proj: ident,
        inv_view: ident,
        inv_proj: ident,
        eye_position: eye,
        _pad0: 0.0,
    };
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sky.test.cam"),
        contents: bytemuck::bytes_of(&cam),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Helper to render with given sun and get pixel data
    let mut render_with = |sun_dir: [f32; 3], exposure: f32| -> Vec<u8> {
        let params = SkyParams {
            sun_direction: sun_dir,
            turbidity: 3.0,
            ground_albedo: 0.2,
            model: 1,
            sun_intensity: 20.0,
            exposure,
            _pad: [0.0; 2],
        };
        let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky.test.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky.test.bg0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky.test.bg1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sky.test.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky.test.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups((size.0 + 7) / 8, (size.1 + 7) / 8, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        // Read back
        let data = crate::renderer::readback::read_texture_tight(
            &device,
            &queue,
            &tex,
            size,
            TextureFormat::Rgba8Unorm,
        )
        .expect("readback");
        data
    };

    // Midday sun (high)
    let img_day = render_with([0.0, 1.0, 0.0], 1.0);
    // Low sun near horizon
    let img_low = render_with([0.0, 0.2, 0.98], 1.0);

    // Compare average (R-B) near horizon band (bottom 8 rows)
    let stride = (size.0 * 4) as usize;
    let horizon_rows = 56..64; // bottom band
    let mut sum_day = 0f64;
    let mut sum_low = 0f64;
    let mut count = 0f64;
    for y in horizon_rows.clone() {
        let row = y as usize;
        for x in 0..size.0 as usize {
            let i = row * stride + x * 4;
            let r_day = img_day[i] as f64;
            let b_day = img_day[i + 2] as f64;
            let r_low = img_low[i] as f64;
            let b_low = img_low[i + 2] as f64;
            sum_day += r_day - b_day;
            sum_low += r_low - b_low;
            count += 1.0;
        }
    }
    let avg_day = sum_day / count;
    let avg_low = sum_low / count;

    // Expect low sun to be warmer near horizon than midday
    assert!(
        avg_low > avg_day,
        "expected warmer horizon at low sun: low={avg_low:.2} > day={avg_day:.2}"
    );
}

#[test]
fn sky_sun_elevation_chromatic_trend() {
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping sky test: no adapter");
        return;
    };

    let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky.test.shader.trend"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/shaders/sky.wgsl").into()),
    });
    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky.test.trend.bgl0"),
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
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky.test.trend.bgl1"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky.test.trend.pl"),
        bind_group_layouts: &[&bgl0, &bgl1],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sky.test.trend.pipeline"),
        layout: Some(&pl),
        module: &sky_shader,
        entry_point: "cs_render_sky",
    });

    let size = (64u32, 64u32);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sky.test.trend.out"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Camera uniforms (identity with plausible inv)
    let ident = glam::Mat4::IDENTITY.to_cols_array_2d();
    let eye = [0.0f32, 0.0, 0.0];
    let cam = CameraUniforms {
        view: ident,
        proj: ident,
        inv_view: ident,
        inv_proj: ident,
        eye_position: eye,
        _pad0: 0.0,
    };
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sky.test.trend.cam"),
        contents: bytemuck::bytes_of(&cam),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Helper to render with given sun direction and return average (R-B) in horizon band
    let mut horizon_warmth = |sun_dir: [f32; 3]| -> f64 {
        let params = SkyParams {
            sun_direction: sun_dir,
            turbidity: 3.0,
            ground_albedo: 0.2,
            model: 1,
            sun_intensity: 20.0,
            exposure: 1.0,
            _pad: [0.0; 2],
        };
        let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky.test.trend.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky.test.trend.bg0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky.test.trend.bg1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sky.test.trend.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky.test.trend.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups((size.0 + 7) / 8, (size.1 + 7) / 8, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let data = crate::renderer::readback::read_texture_tight(
            &device,
            &queue,
            &tex,
            size,
            TextureFormat::Rgba8Unorm,
        )
        .expect("readback.trend");

        // Compute average (R-B) near horizon band (bottom 8 rows)
        let stride = (size.0 * 4) as usize;
        let horizon_rows = 56..64; // bottom band
        let mut sum = 0f64;
        let mut count = 0f64;
        for y in horizon_rows.clone() {
            let row = y as usize;
            for x in 0..size.0 as usize {
                let i = row * stride + x * 4;
                let r = data[i] as f64;
                let b = data[i + 2] as f64;
                sum += r - b;
                count += 1.0;
            }
        }
        sum / count.max(1.0)
    };

    // Sample several sun elevations from midday to near-horizon
    let sun_dirs = [
        [0.0, 1.0, 0.0],   // high sun (midday)
        [0.0, 0.6, 0.8],   // mid elevation
        [0.0, 0.3, 0.95],  // lower
        [0.0, 0.2, 0.98],  // near horizon
    ];

    let mut warmth_values = Vec::new();
    for dir in sun_dirs {
        warmth_values.push(horizon_warmth(dir));
    }

    // Expect horizon warmth to increase monotonically as sun lowers
    for pair in warmth_values.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        assert!(
            next >= prev,
            "expected monotonic warming trend as sun lowers: next={next:.2} >= prev={prev:.2}",
        );
    }
}
