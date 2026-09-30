//! TERRA-DETERMINATA arithmetic canary.
//!
//! `run_det_probe` dispatches `src/shaders/det_probe.wgsl` (assembled with the
//! determinism include) on the active adapter and reads back a fixed 256-byte
//! result buffer. The host hashes those bytes as `probe_sha256`. Because the
//! identical WGSL also runs in browser WebGPU
//! (`tools/determinism_browser/det_probe.js`), the probe hash is a real,
//! executed cross-implementation arithmetic-identity leg — much cheaper to run
//! in a browser than a full terrain frame, and it localizes which helper
//! diverged when two backends disagree.

use sha2::{Digest, Sha256};

use crate::core::error::{RenderError, RenderResult};

/// Number of f32 lanes the probe writes (`array<vec4<f32>, 16>`).
pub const DET_PROBE_OUTPUT_BYTES: u64 = 16 * 16;

/// Assembled probe shader source (determinism include + canary). Public so the
/// browser harness can be kept byte-identical with the native leg.
pub fn det_probe_shader_source() -> String {
    crate::shader_sources::assemble_parts(&crate::shader_sources::det_probe_parts())
}

/// Assembled raster-canary shader source (determinism include + raster pass).
pub fn det_raster_shader_source() -> String {
    crate::shader_sources::assemble_parts(&crate::shader_sources::det_raster_parts())
}

/// Raster canary output extent (square).
pub const DET_RASTER_SIZE: u32 = 64;
/// Bytes per row: 64 px * 16 B (rgba32float). Aligned to
/// `COPY_BYTES_PER_ROW_ALIGNMENT` (256). Raw f32 lanes — an 8-bit unorm
/// target would quantize away the one-ULP divergences this canary exists to
/// catch.
pub const DET_RASTER_ROW_BYTES: u64 = DET_RASTER_SIZE as u64 * 16;
/// Total readback size: 64 rows of 1024 bytes.
pub const DET_RASTER_READBACK_BYTES: u64 = DET_RASTER_ROW_BYTES * DET_RASTER_SIZE as u64;

/// The committed arithmetic-canary golden: every deterministic-mode context
/// must reproduce this `probe_sha256` or refuse to run. Measured on real
/// hardware (NVIDIA RTX 3070, Vulkan, driver 595.95 — see
/// `tools/determinism_browser/` for the browser leg running the identical
/// WGSL); update only deliberately, with the adapter evidence.
pub const EXPECTED_DET_PROBE_SHA256: &str =
    "f94de1bffbf3383447e63c8db2fa6f9c891af406303303ec0cc9c2fc9d729aeb";

pub const EXPECTED_DET_RASTER_SHA256: &str =
    "298a84780b271bfa0633e50507bc1209ed17fbed4a68c1afb5a7d38f0b23875a";

/// Run the canary on the active GPU context. Returns `(raw_bytes, sha256_hex)`.
///
/// Requires an initialized GPU context (`try_ctx()`); callers are expected to
/// run under deterministic mode with the backend already pinned.
pub fn run_det_probe() -> RenderResult<(Vec<u8>, String)> {
    let ctx = crate::core::gpu::try_ctx()?;
    run_det_probe_on(ctx.device.as_ref(), ctx.queue.as_ref())
}

/// `run_det_probe` on an explicit device/queue, so the deterministic-mode
/// context gate can run the canary before `GpuContext` is committed.
pub(crate) fn run_det_probe_on(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> RenderResult<(Vec<u8>, String)> {
    let shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "det_probe.shader",
        &det_probe_shader_source(),
    );

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("det_probe.bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("det_probe.pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("det_probe.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        },
    )
    .map_err(|error| {
        RenderError::render(format!(
            "det_probe compute pipeline creation failed: {error}"
        ))
    })?;

    let output_buffer = crate::core::resource_tracker::tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("det_probe.output"),
            size: DET_PROBE_OUTPUT_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    let readback_buffer = crate::core::resource_tracker::tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("det_probe.readback"),
            size: DET_PROBE_OUTPUT_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("det_probe.bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("det_probe.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("det_probe.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(4, 4, 1);
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &readback_buffer,
        0,
        DET_PROBE_OUTPUT_BYTES,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| RenderError::readback(format!("det_probe map channel: {e}")))?
        .map_err(|e| RenderError::readback(format!("det_probe map: {e:?}")))?;

    let bytes = {
        let data = slice.get_mapped_range();
        data.to_vec()
    };
    readback_buffer.unmap();

    let digest = Sha256::digest(&bytes);
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok((bytes, hex))
}

/// Run the raster canary: a real fullscreen-triangle render pass into a
/// 64x64 rgba32float target, then readback. Returns `(raw_bytes, sha256_hex)`.
/// The bytes are raw f32 lanes (1024-byte rows, no row padding needed since
/// 1024 is already a multiple of 256), so the hash is comparable across
/// implementations and sensitive to single-ULP differences.
pub fn run_det_raster() -> RenderResult<(Vec<u8>, String)> {
    let ctx = crate::core::gpu::try_ctx()?;
    run_det_raster_on(ctx.device.as_ref(), ctx.queue.as_ref())
}

/// `run_det_raster` on an explicit device/queue (see [`run_det_probe_on`]).
pub(crate) fn run_det_raster_on(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> RenderResult<(Vec<u8>, String)> {
    let shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "det_raster.shader",
        &det_raster_shader_source(),
    );

    let size = DET_RASTER_SIZE;
    let target = crate::core::resource_tracker::tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("det_raster.target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )?;
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("det_raster.pipeline_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some("det_raster.pipeline"),
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
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        },
    );

    let readback_buffer = crate::core::resource_tracker::tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("det_raster.readback"),
            size: DET_RASTER_READBACK_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("det_raster.encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("det_raster.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(DET_RASTER_ROW_BYTES as u32),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| RenderError::readback(format!("det_raster map channel: {e}")))?
        .map_err(|e| RenderError::readback(format!("det_raster map: {e:?}")))?;

    let bytes = {
        let data = slice.get_mapped_range();
        data.to_vec()
    };
    readback_buffer.unmap();

    let digest = Sha256::digest(&bytes);
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok((bytes, hex))
}

#[cfg(test)]
mod tests {
    use super::{EXPECTED_DET_PROBE_SHA256, EXPECTED_DET_RASTER_SHA256};

    #[test]
    fn verify_runtime_canary_hashes_match_committed_sidecars() {
        assert_eq!(
            EXPECTED_DET_PROBE_SHA256,
            include_str!("../../tests/goldens/determinism/terra_determinata_v1.probe.sha256")
                .trim()
        );
        assert_eq!(
            EXPECTED_DET_RASTER_SHA256,
            include_str!("../../tests/goldens/determinism/terra_determinata_v1.raster.sha256")
                .trim()
        );
    }
}
