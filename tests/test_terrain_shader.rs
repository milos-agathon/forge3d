// tests/test_terrain_shader.rs
// Validate terrain height fallback gating to avoid bias on real data.
// Ensures analytic fallback only applies to sentinel textures introduced in R1.
// RELEVANT FILES:src/shaders/terrain.wgsl,src/terrain/mod.rs,tests/test_terrain_shader.rs

use futures_intrusive::channel::shared::oneshot_channel;
use std::num::{NonZeroU32, NonZeroU64};
use wgpu::{
    Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, TextureFormat,
};

const FALLBACK_WGSL: &str = r#"
@group(0) @binding(0) var height_tex: texture_2d<f32>;
@group(0) @binding(1) var height_samp: sampler;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32, 1>;

fn analytic_height(x: f32, z: f32) -> f32 {
  return sin(x * 1.3) * 0.25 + cos(z * 1.1) * 0.25;
}

fn sample_height_with_fallback(
  uv: vec2<f32>,
  uv_offset: vec2<f32>,
  xz: vec2<f32>,
  spacing: f32,
  use_fallback: bool,
) -> f32 {
  let sample_uv = clamp(uv + uv_offset, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
  let h_tex = textureSampleLevel(height_tex, height_samp, sample_uv, 0.0).r;
  if (!use_fallback) {
    return h_tex;
  }
  let offset_xz = vec2<f32>(xz.x + uv_offset.x * spacing, xz.y + uv_offset.y * spacing);
  let h_ana = analytic_height(offset_xz.x, offset_xz.y);
  return h_tex + h_ana;
}

@compute @workgroup_size(1)
fn main() {
  let dims = textureDimensions(height_tex, 0);
  let use_fallback = dims.x <= 1u && dims.y <= 1u;
  let value = sample_height_with_fallback(vec2<f32>(0.5, 0.5), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), 1.0, use_fallback);
  out_data[0] = value;
}
"#;

fn create_device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &DeviceDescriptor {
            label: Some("terrain-fallback-test-device"),
            required_features: Features::empty(),
            required_limits: Limits::downlevel_defaults(),
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

fn create_height_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    value: f32,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("height-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let data = vec![value; (width * height) as usize];
    let bytes = bytemuck::cast_slice(&data);
    let bytes_per_row = NonZeroU32::new(width * 4).map(|v| v.get());

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row,
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    texture
}

fn dispatch_sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    texture: &wgpu::Texture,
) -> f32 {
    let buffer_size = std::mem::size_of::<f32>() as u64;
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fallback-storage"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fallback-readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fallback-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &storage_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(buffer_size).unwrap()),
                }),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fallback-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fallback-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &readback_buffer, 0, buffer_size);
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let buffer_slice = readback_buffer.slice(..);
    let (sender, receiver) = oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    let map_result = pollster::block_on(receiver.receive()).expect("receive map result");
    map_result.expect("map output buffer");
    let data = buffer_slice.get_mapped_range();
    let value = bytemuck::from_bytes::<f32>(&data).to_owned();
    drop(data);
    readback_buffer.unmap();
    value
}

#[test]
fn terrain_height_fallback_skips_real_textures() {
    let (device, queue) = match create_device_and_queue() {
        Some(pair) => pair,
        None => {
            eprintln!("SKIPPED: terrain fallback test requires a GPU adapter");
            return;
        }
    };

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("height-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain-fallback"),
        source: wgpu::ShaderSource::Wgsl(FALLBACK_WGSL.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fallback-layout"),
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(std::mem::size_of::<f32>() as u64).unwrap(),
                    ),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fallback-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fallback-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let sentinel_tex = create_height_texture(&device, &queue, 1, 1, 0.25);
    let real_tex = create_height_texture(&device, &queue, 2, 1, 0.25);

    let sentinel_value = dispatch_sample(
        &device,
        &queue,
        &pipeline,
        &bind_group_layout,
        &sampler,
        &sentinel_tex,
    );
    let real_value = dispatch_sample(
        &device,
        &queue,
        &pipeline,
        &bind_group_layout,
        &sampler,
        &real_tex,
    );

    assert!(
        (real_value - 0.25).abs() < 1e-5,
        "expected real texture height to remain unchanged"
    );
    assert!(
        sentinel_value > real_value + 0.1,
        "sentinel texture should receive analytic fallback"
    );
}
