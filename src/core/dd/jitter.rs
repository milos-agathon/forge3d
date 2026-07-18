use super::jitter_model::{
    build_model, jitter_evidence, reduce_measurements, DdJitterReport, DEFAULT_FRAMES,
};
use super::jitter_pipeline::create_pipelines;
use super::selftest;
use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture,
};
use sha2::{Digest, Sha256};

pub(super) const LABEL: &str = "dupla.dd_jitter";

fn read_measurements(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    count: u32,
) -> RenderResult<Vec<[f32; 2]>> {
    let slice = buffer.slice(..count as u64 * 8);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|error| RenderError::readback(format!("jitter callback failed: {error}")))?
        .map_err(|error| RenderError::readback(format!("jitter map failed: {error}")))?;
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, [f32; 2]>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn hash_render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> RenderResult<String> {
    let bytes = crate::renderer::readback::read_texture_tight(
        device,
        queue,
        texture,
        (width, height),
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .map_err(|error| RenderError::readback(format!("jitter render readback failed: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn jitter_demo(frames: u32) -> RenderResult<DdJitterReport> {
    if frames == 0 || frames > DEFAULT_FRAMES {
        return Err(RenderError::render(
            "DD jitter demo frames must be in 1..=1000",
        ));
    }
    let capability = selftest()?;
    let context = crate::core::gpu::try_ctx()?;
    let device = &context.device;
    let queue = &context.queue;
    let capture = crate::core::certificate::begin_render_capture("dupla.dd_jitter");

    let model = build_model(frames)?;

    let positions = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("dupla.jitter.positions"),
            contents: bytemuck::cast_slice(&model.points),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    let camera_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("dupla.jitter.cameras"),
            contents: bytemuck::cast_slice(&model.cameras),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    let globals_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("dupla.jitter.globals"),
            contents: bytemuck::bytes_of(&model.globals),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let result_size = frames as u64 * 8;
    let output = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("dupla.jitter.output"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    let readback = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("dupla.jitter.readback"),
            size: result_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    if std::env::var_os("FORGE3D_DD_FORCE_JITTER_FAIL").is_some() {
        return Err(RenderError::render("forced DD jitter failure"));
    }

    let pipelines = create_pipelines(device, capability.two_prod_variant)?;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dupla.dd_jitter.bind_group"),
        layout: &pipelines.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: positions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: globals_buffer.as_entire_binding(),
            },
        ],
    });
    let target = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("dupla.dd_jitter.target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )?;
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dupla.dd_jitter.encoder"),
    });
    let mut timing = crate::core::gpu_timing::OneShotTiming::for_current_device();
    let timing_scope = timing.begin(&mut encoder, "dupla.dd_jitter");
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dupla.dd_jitter.measure"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.compute);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(frames.div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, result_size);
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dupla.dd_jitter.render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.render);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    timing.end(&mut encoder, timing_scope, 2);
    timing.resolve(&mut encoder);
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    crate::core::shader_registry::record_shader_use(LABEL);
    if !timing.record_into_certificate() {
        crate::core::certificate::record_pass("dupla.dd_jitter", 0.0, 2);
    }
    let measured = read_measurements(device, &readback, frames)?;
    let dd_hash_a = hash_render(device, queue, &target, 64, 64)?;
    let mut second = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dupla.dd_jitter.second_render"),
    });
    {
        let mut pass = second.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dupla.dd_jitter.second_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.render);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(second.finish()));
    device.poll(wgpu::Maintain::Wait);
    let dd_hash_b = hash_render(device, queue, &target, 64, 64)?;

    let metrics = reduce_measurements(&measured, &model)?;
    super::jitter_model::validate_acceptance(&metrics, frames, (&dd_hash_a, &dd_hash_b))?;
    let shader_hash = crate::core::shader_registry::shader_hashes_snapshot()
        .get(LABEL)
        .cloned()
        .unwrap_or_default();
    crate::core::certificate::record_jitter_evidence(jitter_evidence(
        &capability,
        frames,
        &metrics,
        shader_hash,
        (&dd_hash_a, &dd_hash_b),
    ));
    capture.finish();
    let certificate_json = crate::core::certificate::execution_report_json()?;
    Ok(DdJitterReport {
        dd_errors_px: metrics.dd_errors_px,
        f32_errors_px: metrics.f32_errors_px,
        dd_max_error_px: metrics.dd_max_error_px,
        f32_max_error_px: metrics.f32_max_error_px,
        raw_over_one_px: metrics.raw_over_one_px,
        dd_hash_a,
        dd_hash_b,
        backend: capability.backend,
        shader_label: LABEL.to_string(),
        certificate_json,
    })
}
