#![cfg(feature = "enable-globe")]

use super::*;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedTexture,
};
use crate::core::shader_registry::create_compute_pipeline_scoped;

fn r32_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    values: &[f32],
) -> (TrackedTexture, wgpu::TextureView) {
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )
    .unwrap();
    let mut padded = vec![0u8; 256 * height as usize];
    for row in 0..height as usize {
        let source =
            bytemuck::cast_slice(&values[row * width as usize..(row + 1) * width as usize]);
        padded[row * 256..row * 256 + source.len()].copy_from_slice(source);
    }
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn production_globe_center_uv(lon: f64, lat: f64) -> [f32; 2] {
    use crate::terrain::clipmap::globe::GlobeFrame;
    use crate::terrain::clipmap::{ClipmapConfig, ClipmapLevel};
    let seed = GlobeFrame::globe(
        GlobeFrame::WGS84_MEAN_RADIUS_M,
        glam::DVec3::X * GlobeFrame::WGS84_MEAN_RADIUS_M,
    )
    .unwrap();
    let center = seed.lonlat_alt_to_ecef(lon, lat, 0.0).unwrap();
    let frame = GlobeFrame::globe(
        GlobeFrame::WGS84_MEAN_RADIUS_M,
        center + center.normalize() * 1_000.0,
    )
    .unwrap();
    let config = ClipmapConfig {
        ring_count: 0,
        ring_resolution: 2,
        center_resolution: 2,
        ..ClipmapConfig::default()
    };
    let mut level = ClipmapLevel::new_globe(config, center, frame, 0.004).unwrap();
    level.generate().unwrap().vertices[4].uv
}

#[test]
fn production_shader_samples_regional_overview_at_two_global_positions() {
    let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
        eprintln!("SKIP: no GPU adapter for regional overview probe");
        return;
    };
    let overview = OverviewUvTransform::from_lonlat_bounds((-180.0, 0.0, 0.0, 90.0)).unwrap();
    let initial =
        SerializedPageTable::from_entries_with_overview(&[], 4, 2, 1, (1, 1), 1, overview).unwrap();
    let page_table = tracked_create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("orbis.overview-probe.page-table"),
            contents: &initial.bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )
    .unwrap();
    let (_overview_texture, overview_view) = r32_texture(
        &device,
        &queue,
        "orbis.overview-probe.overview",
        2,
        2,
        &[0.0, 2.0, 20.0, 22.0],
    );
    let (_atlas_texture, atlas_view) =
        r32_texture(&device, &queue, "orbis.overview-probe.atlas", 1, 1, &[99.0]);
    let (_coverage_texture, coverage_view) = r32_texture(
        &device,
        &queue,
        "orbis.overview-probe.coverage",
        1,
        1,
        &[1.0],
    );
    let probes = [
        production_globe_center_uv(-90.0, 45.0),
        production_globe_center_uv(-135.0, 67.5),
        production_globe_center_uv(90.0, -45.0),
    ];
    let probe_buffer = tracked_create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("orbis.overview-probe.uvs"),
            contents: bytemuck::cast_slice(&probes),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )
    .unwrap();
    let output = tracked_create_buffer(
        &device,
        &wgpu::BufferDescriptor {
            label: Some("orbis.overview-probe.output"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )
    .unwrap();
    let readback = tracked_create_buffer(
        &device,
        &wgpu::BufferDescriptor {
            label: Some("orbis.overview-probe.readback"),
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )
    .unwrap();
    let shader = format!(
        "{}\n@group(0) @binding(23) var<storage, read> orbis_probe_uvs: array<vec2<f32>>;\n\
         @group(0) @binding(24) var<storage, read_write> orbis_probe_out: array<f32>;\n\
         @compute @workgroup_size(1) fn cs_orbis_overview_probe(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
           if (id.x < 3u) {{ orbis_probe_out[id.x] = sample_height_bilinear_level(orbis_probe_uvs[id.x], 0.0); }}\n\
         }}",
        crate::shader_sources::terrain()
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("orbis.overview-probe.shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let storage_ro = |min_size| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: std::num::NonZeroU64::new(min_size),
    };
    // The shared height sampler reads `u_terrain.spacing_h_exag.w` (the ORBIS
    // detail blend); zeroed terrain uniforms select the plain page walk.
    let terrain_uniforms = tracked_create_buffer(
        &device,
        &wgpu::BufferDescriptor {
            label: Some("orbis.overview-probe.terrain-uniforms"),
            size: 176,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        },
    )
    .unwrap();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orbis.overview-probe.layout"),
        entries: &[
            layout_entry(
                0,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(176),
                },
            ),
            layout_entry(
                1,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(20, storage_ro(96)),
            layout_entry(
                21,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                22,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(23, storage_ro(24)),
            layout_entry(
                24,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(12),
                },
            ),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("orbis.overview-probe.pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = create_compute_pipeline_scoped(
        &device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("orbis.overview-probe.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "cs_orbis_overview_probe",
        },
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("orbis.overview-probe.bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&overview_view),
            },
            wgpu::BindGroupEntry {
                binding: 20,
                resource: page_table.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 21,
                resource: wgpu::BindingResource::TextureView(&atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 22,
                resource: wgpu::BindingResource::TextureView(&coverage_view),
            },
            wgpu::BindGroupEntry {
                binding: 23,
                resource: probe_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 24,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("orbis.overview-probe.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("orbis.overview-probe.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(3, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 12);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver.receive()).unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let actual = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    assert!((actual[0] - 11.0).abs() < 1.0e-5, "center {actual:?}");
    assert!((actual[1] - 5.5).abs() < 1.0e-5, "quarter {actual:?}");
    assert_eq!(
        actual[2], 0.0,
        "outside overview must not smear edge texels"
    );

    // A real hashed LOD-2 page must override only its global quadrant; the
    // second in-bounds point remains a correctly transformed overview miss.
    let populated = SerializedPageTable::from_entries_with_overview(
        &[(TileId::new(2, 1, 1), (0, 0))],
        4,
        2,
        1,
        (1, 1),
        1,
        overview,
    )
    .unwrap();
    queue.write_buffer(&page_table, 0, &populated.bytes());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("orbis.overview-probe.resident-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("orbis.overview-probe.resident-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(3, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 12);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver.receive()).unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let resident_actual = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    assert_eq!(
        resident_actual[0], 99.0,
        "resident hash/atlas {resident_actual:?}"
    );
    assert!(
        (resident_actual[1] - 5.5).abs() < 1.0e-5,
        "overview miss {resident_actual:?}"
    );
    assert_eq!(resident_actual[2], 0.0, "outside {resident_actual:?}");
}
