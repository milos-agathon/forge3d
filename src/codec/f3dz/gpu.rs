//! Fail-closed F3DZ decode on wgpu.
//!
//! The CPU validates the container, CRCs, page contracts, and adaptive rANS
//! tables, but deliberately does not entropy-decode a single symbol. Compressed
//! lanes are copied to storage buffers and the WGSL kernel performs rANS,
//! predictor reconstruction, and final binary32 conversion. Dispatches are
//! bounded to [`MAX_BATCH_PAGES`] so staging stays independent of DEM size.

use super::format::{ContainerHeader, ValidatedPage};
use super::rans::{RansEncoded, RANS_L, SCALE};
use super::{F3dzError, F3dzResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture, TrackedBuffer,
};
use futures_intrusive::channel::shared::oneshot_channel;
use std::num::NonZeroU64;
use std::time::{Duration, Instant};
use wgpu::util::BufferInitDescriptor;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Extent3d, PipelineLayoutDescriptor, Queue, ShaderStages, StorageTextureAccess, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

/// A 512-page batch exposes enough independent entropy streams to saturate
/// discrete GPUs while bounding worst-case staging below the enforced 512 MiB
/// host-visible budget (64x64 samples, two five-byte token layers per page).
pub const MAX_BATCH_PAGES: usize = 512;
const DESC_STRIDE: usize = 40;
const TABLE_STRIDE: usize = 4096 + 256 + 256;

#[derive(Debug)]
pub struct GpuDecodeResult {
    pub width: u32,
    pub height: u32,
    pub epsilon: f32,
    pub base_quality: bool,
    pub adapter: String,
    pub elapsed_seconds: f64,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GpuDecodeBenchmark {
    pub adapter: String,
    pub iterations: u32,
    pub pixels_per_iteration: u64,
    pub fast_path: bool,
    pub direct_tokens: bool,
    pub elapsed_seconds: f64,
    pub wall_seconds: f64,
    pub timing_source: &'static str,
    pub gigapixels_per_second: f64,
}

pub struct F3dzGpuDecoder {
    bind_group_layout: BindGroupLayout,
    buffer_pipeline: ComputePipeline,
    atlas_pipeline: ComputePipeline,
    fast_buffer_pipeline: ComputePipeline,
    fast_atlas_pipeline: ComputePipeline,
}

impl F3dzGpuDecoder {
    pub fn new(device: &Device) -> F3dzResult<Self> {
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 8 {
            return Err(F3dzError::GpuUnavailable(format!(
                "f3dz requires 8 storage buffers per compute stage, adapter exposes {}",
                limits.max_storage_buffers_per_shader_stage
            )));
        }
        if limits.max_compute_workgroup_storage_size < 4096 * 4 {
            return Err(F3dzError::GpuUnavailable(format!(
                "f3dz requires 16384 bytes of workgroup storage, adapter exposes {}",
                limits.max_compute_workgroup_storage_size
            )));
        }
        let buffer_entry = |binding, read_only| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(4),
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("f3dz.decode.bind-group-layout"),
            entries: &[
                buffer_entry(0, true),
                buffer_entry(1, true),
                buffer_entry(2, true),
                buffer_entry(3, false),
                buffer_entry(4, false),
                buffer_entry(5, false),
                buffer_entry(6, false),
                buffer_entry(7, false),
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("f3dz.decode.pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "f3dz.decode.shader",
            include_str!("../../shaders/f3dz_decode.wgsl"),
        );
        let buffer_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("f3dz.decode.to-buffer"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "decode_to_buffer",
            },
        )
        .map_err(gpu_error)?;
        let atlas_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("f3dz.decode.to-atlas"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "decode_to_atlas",
            },
        )
        .map_err(gpu_error)?;
        let fast_buffer_pipeline =
            crate::core::shader_registry::try_create_compute_pipeline_scoped(
                device,
                &ComputePipelineDescriptor {
                    label: Some("f3dz.decode.prefix-to-buffer"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: "decode_to_buffer_fast",
                },
            )
            .map_err(gpu_error)?;
        let fast_atlas_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("f3dz.decode.prefix-to-atlas"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "decode_to_atlas_fast",
            },
        )
        .map_err(gpu_error)?;
        Ok(Self {
            bind_group_layout,
            buffer_pipeline,
            atlas_pipeline,
            fast_buffer_pipeline,
            fast_atlas_pipeline,
        })
    }

    /// Decode a complete stream into a GPU buffer, then read it back. This is
    /// the identity/benchmark path; streaming should use [`decode_into_atlas`].
    pub fn decode_to_vec(
        &self,
        device: &Device,
        queue: &Queue,
        data: &[u8],
    ) -> F3dzResult<(ContainerHeader, Vec<f32>)> {
        let prepared = prepare_stream(data)?;
        let output_len = checked_output_len(&prepared.header)?;
        let output = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("f3dz.decode.output"),
                size: byte_size(output_len)?,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .map_err(gpu_error)?;
        let dummy_atlas = tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("f3dz.decode.dummy-atlas"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        )
        .map_err(gpu_error)?;
        let dummy_view = dummy_atlas.create_view(&TextureViewDescriptor::default());
        for batch in &prepared.batches {
            let pipeline = if batch.gpu_fast {
                &self.fast_buffer_pipeline
            } else {
                &self.buffer_pipeline
            };
            self.dispatch_batch(device, queue, batch, &output, &dummy_view, pipeline)?;
        }
        let bytes = read_buffer(device, queue, &output, byte_size(output_len)?)?;
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
            .collect();
        if prepared.header.base_only() {
            record_base_quality();
        }
        crate::core::certificate::record_f3dz_pages(
            prepared.header.epsilon,
            prepared.header.page_count,
            prepared.header.base_only(),
        );
        Ok((prepared.header, values))
    }

    /// Decode directly into a caller-owned R32Float atlas. The target texture
    /// must include `STORAGE_BINDING`; no CPU height array is materialized.
    pub fn decode_into_atlas(
        &self,
        device: &Device,
        queue: &Queue,
        data: &[u8],
        atlas: &Texture,
        origin: (u32, u32),
    ) -> F3dzResult<ContainerHeader> {
        let prepared = prepare_stream(data)?;
        let atlas_size = atlas.size();
        if atlas.format() != TextureFormat::R32Float
            || !atlas.usage().contains(TextureUsages::STORAGE_BINDING)
        {
            return Err(F3dzError::InvalidArgument(
                "f3dz GPU atlas target must be R32Float with STORAGE_BINDING".to_string(),
            ));
        }
        let end_x = origin
            .0
            .checked_add(prepared.header.width)
            .ok_or_else(|| F3dzError::InvalidArgument("atlas x range overflow".to_string()))?;
        let end_y = origin
            .1
            .checked_add(prepared.header.height)
            .ok_or_else(|| F3dzError::InvalidArgument("atlas y range overflow".to_string()))?;
        if end_x > atlas_size.width || end_y > atlas_size.height {
            return Err(F3dzError::InvalidArgument(format!(
                "f3dz grid {}x{} at ({},{}) exceeds atlas {}x{}",
                prepared.header.width,
                prepared.header.height,
                origin.0,
                origin.1,
                atlas_size.width,
                atlas_size.height
            )));
        }
        let atlas_view = atlas.create_view(&TextureViewDescriptor::default());
        let dummy_output = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("f3dz.decode.dummy-output"),
                size: 4,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )
        .map_err(gpu_error)?;
        for source_batch in &prepared.batches {
            let mut batch = source_batch.clone();
            for descriptor in batch.descriptors.chunks_exact_mut(DESC_STRIDE) {
                descriptor[5] = origin.0;
                descriptor[6] = origin.1;
            }
            let pipeline = if batch.gpu_fast {
                &self.fast_atlas_pipeline
            } else {
                &self.atlas_pipeline
            };
            self.dispatch_batch(device, queue, &batch, &dummy_output, &atlas_view, pipeline)?;
        }
        if prepared.header.base_only() {
            record_base_quality();
        }
        crate::core::certificate::record_f3dz_pages(
            prepared.header.epsilon,
            prepared.header.page_count,
            prepared.header.base_only(),
        );
        Ok(prepared.header)
    }

    fn dispatch_batch(
        &self,
        device: &Device,
        queue: &Queue,
        batch: &PreparedBatch,
        output: &TrackedBuffer,
        atlas_view: &wgpu::TextureView,
        pipeline: &ComputePipeline,
    ) -> F3dzResult<()> {
        let compressed = init_u32_buffer(
            device,
            "f3dz.decode.compressed",
            &batch.compressed,
            BufferUsages::STORAGE,
        )?;
        let descriptors = init_u32_buffer(
            device,
            "f3dz.decode.descriptors",
            &batch.descriptors,
            BufferUsages::STORAGE,
        )?;
        let tables = init_u32_buffer(
            device,
            "f3dz.decode.tables",
            &batch.tables,
            BufferUsages::STORAGE,
        )?;
        let byte_scratch = storage_buffer(device, "f3dz.decode.byte-scratch", batch.decoded_bytes)?;
        let q_scratch = storage_buffer(device, "f3dz.decode.q-scratch", batch.samples)?;
        let value_bits = storage_buffer(device, "f3dz.decode.value-bits", batch.samples)?;
        let status_zeros = vec![0u32; batch.page_count];
        let status = init_u32_buffer(
            device,
            "f3dz.decode.status",
            &status_zeros,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let bind_group = self.create_bind_group(
            device,
            &compressed,
            &descriptors,
            &tables,
            &byte_scratch,
            &q_scratch,
            &value_bits,
            output,
            &status,
            atlas_view,
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("f3dz.decode.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("f3dz.decode.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(batch.page_count as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        let bytes = read_buffer(device, queue, &status, byte_size(batch.page_count)?)?;
        for (page, raw) in bytes.chunks_exact(4).enumerate() {
            let code = u32::from_le_bytes(raw.try_into().unwrap());
            if code != 0 {
                return Err(F3dzError::CorruptPage {
                    page: batch.first_page + page,
                    reason: format!("GPU decoder status=0x{code:08x}"),
                });
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_bind_group(
        &self,
        device: &Device,
        compressed: &TrackedBuffer,
        descriptors: &TrackedBuffer,
        tables: &TrackedBuffer,
        byte_scratch: &TrackedBuffer,
        q_scratch: &TrackedBuffer,
        value_bits: &TrackedBuffer,
        output: &TrackedBuffer,
        status: &TrackedBuffer,
        atlas_view: &wgpu::TextureView,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("f3dz.decode.bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                binding(0, compressed),
                binding(1, descriptors),
                binding(2, tables),
                binding(3, byte_scratch),
                binding(4, q_scratch),
                binding(5, value_bits),
                binding(6, output),
                binding(7, status),
                BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
            ],
        })
    }
}

/// Convenience entry point using forge3d's negotiated global GPU context.
pub fn decode_dem_gpu(data: &[u8]) -> F3dzResult<GpuDecodeResult> {
    let context = crate::core::gpu::try_ctx().map_err(gpu_error)?;
    ensure_hardware_adapter(context.software_fallback, &context.adapter.get_info())?;
    let decoder = F3dzGpuDecoder::new(context.device.as_ref())?;
    let started = Instant::now();
    let (header, values) =
        decoder.decode_to_vec(context.device.as_ref(), context.queue.as_ref(), data)?;
    Ok(GpuDecodeResult {
        width: header.width,
        height: header.height,
        epsilon: header.epsilon,
        base_quality: header.base_only(),
        adapter: context.adapter.get_info().name,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        values,
    })
}

/// Hardware benchmark path. The stream is preflighted and uploaded once, then
/// the decode kernel is dispatched repeatedly against the same output buffer.
/// Setup, upload, and readback are deliberately outside the timed interval.
pub fn benchmark_decode_gpu(data: &[u8], iterations: u32) -> F3dzResult<GpuDecodeBenchmark> {
    if iterations == 0 {
        return Err(F3dzError::InvalidArgument(
            "GPU benchmark iterations must be positive".to_string(),
        ));
    }
    let context = crate::core::gpu::try_ctx().map_err(gpu_error)?;
    ensure_hardware_adapter(context.software_fallback, &context.adapter.get_info())?;
    let device = context.device.as_ref();
    let queue = context.queue.as_ref();
    if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return Err(F3dzError::GpuUnavailable(
            "F3DZ throughput benchmark requires TIMESTAMP_QUERY".to_string(),
        ));
    }
    let decoder = F3dzGpuDecoder::new(device)?;
    let prepared = prepare_stream(data)?;
    let output_len = checked_output_len(&prepared.header)?;
    let output = storage_buffer(device, "f3dz.bench.output", output_len)?;
    let dummy_atlas = tracked_create_texture(
        device,
        &TextureDescriptor {
            label: Some("f3dz.bench.dummy-atlas"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
    )
    .map_err(gpu_error)?;
    let atlas_view = dummy_atlas.create_view(&TextureViewDescriptor::default());
    let timestamp_query = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("f3dz.bench.timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let timestamp_resolve = tracked_create_buffer(
        device,
        &BufferDescriptor {
            label: Some("f3dz.bench.timestamp-resolve"),
            size: 2 * std::mem::size_of::<u64>() as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )
    .map_err(gpu_error)?;
    struct BenchBatch {
        _compressed: TrackedBuffer,
        _descriptors: TrackedBuffer,
        _tables: TrackedBuffer,
        _byte_scratch: TrackedBuffer,
        _q_scratch: TrackedBuffer,
        _value_bits: TrackedBuffer,
        status: TrackedBuffer,
        bind_group: BindGroup,
        first_page: usize,
        page_count: usize,
        gpu_fast: bool,
        gpu_direct: bool,
    }
    let mut resources = Vec::with_capacity(prepared.batches.len());
    for batch in &prepared.batches {
        let compressed = init_u32_buffer(
            device,
            "f3dz.bench.compressed",
            &batch.compressed,
            BufferUsages::STORAGE,
        )?;
        let descriptors = init_u32_buffer(
            device,
            "f3dz.bench.descriptors",
            &batch.descriptors,
            BufferUsages::STORAGE,
        )?;
        let tables = init_u32_buffer(
            device,
            "f3dz.bench.tables",
            &batch.tables,
            BufferUsages::STORAGE,
        )?;
        let byte_scratch = storage_buffer(device, "f3dz.bench.byte-scratch", batch.decoded_bytes)?;
        let q_scratch = storage_buffer(device, "f3dz.bench.q-scratch", batch.samples)?;
        let value_bits = storage_buffer(device, "f3dz.bench.value-bits", batch.samples)?;
        let status = init_u32_buffer(
            device,
            "f3dz.bench.status",
            &vec![0u32; batch.page_count],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let bind_group = decoder.create_bind_group(
            device,
            &compressed,
            &descriptors,
            &tables,
            &byte_scratch,
            &q_scratch,
            &value_bits,
            &output,
            &status,
            &atlas_view,
        );
        resources.push(BenchBatch {
            _compressed: compressed,
            _descriptors: descriptors,
            _tables: tables,
            _byte_scratch: byte_scratch,
            _q_scratch: q_scratch,
            _value_bits: value_bits,
            status,
            bind_group,
            first_page: batch.first_page,
            page_count: batch.page_count,
            gpu_fast: batch.gpu_fast,
            gpu_direct: batch.gpu_direct,
        });
    }
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("f3dz.bench.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("f3dz.bench.pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &timestamp_query,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        for batch in &resources {
            let pipeline = if batch.gpu_fast {
                &decoder.fast_buffer_pipeline
            } else {
                &decoder.buffer_pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &batch.bind_group, &[]);
            for _ in 0..iterations {
                pass.dispatch_workgroups(batch.page_count as u32, 1, 1);
            }
        }
    }
    encoder.resolve_query_set(&timestamp_query, 0..2, &timestamp_resolve, 0);
    let started = Instant::now();
    queue.submit(Some(encoder.finish()));
    wait_for_queue(device, queue, Duration::from_secs(60))?;
    let wall_seconds = started.elapsed().as_secs_f64();
    let timestamp_bytes = read_buffer(
        device,
        queue,
        &timestamp_resolve,
        2 * std::mem::size_of::<u64>() as u64,
    )?;
    let begin = u64::from_le_bytes(timestamp_bytes[0..8].try_into().unwrap());
    let end = u64::from_le_bytes(timestamp_bytes[8..16].try_into().unwrap());
    if begin == 0 || end < begin {
        return Err(F3dzError::GpuUnavailable(format!(
            "invalid F3DZ GPU timestamps: begin={begin} end={end}"
        )));
    }
    let elapsed_seconds = (end - begin) as f64 * f64::from(queue.get_timestamp_period()) / 1.0e9;
    for batch in &resources {
        let status_bytes = read_buffer(device, queue, &batch.status, byte_size(batch.page_count)?)?;
        for (page, raw) in status_bytes.chunks_exact(4).enumerate() {
            let code = u32::from_le_bytes(raw.try_into().unwrap());
            if code != 0 {
                return Err(F3dzError::CorruptPage {
                    page: batch.first_page + page,
                    reason: format!("GPU benchmark decoder status=0x{code:08x}"),
                });
            }
        }
    }
    let pixels_per_iteration = output_len as u64;
    let gigapixels_per_second =
        pixels_per_iteration as f64 * f64::from(iterations) / elapsed_seconds / 1.0e9;
    Ok(GpuDecodeBenchmark {
        adapter: context.adapter.get_info().name,
        iterations,
        pixels_per_iteration,
        fast_path: resources.iter().all(|batch| batch.gpu_fast),
        direct_tokens: resources.iter().all(|batch| batch.gpu_direct),
        elapsed_seconds,
        wall_seconds,
        timing_source: "gpu_timestamp",
        gigapixels_per_second,
    })
}

/// Validate all container/page metadata, CRCs, and rANS framing without
/// entropy-decoding. Streaming uses this before it reserves an atlas slot.
pub fn validate_stream(data: &[u8]) -> F3dzResult<ContainerHeader> {
    prepare_stream(data).map(|prepared| prepared.header)
}

fn ensure_hardware_adapter(software_fallback: bool, adapter: &wgpu::AdapterInfo) -> F3dzResult<()> {
    if software_fallback || adapter.device_type == wgpu::DeviceType::Cpu {
        return Err(F3dzError::GpuUnavailable(format!(
            "F3DZ GPU decode refuses software/CPU adapter '{}' ({:?}); \
             use a physical GPU for codec identity and throughput evidence",
            adapter.name, adapter.backend
        )));
    }
    Ok(())
}

fn binding(binding: u32, buffer: &TrackedBuffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn init_u32_buffer(
    device: &Device,
    label: &'static str,
    values: &[u32],
    usage: BufferUsages,
) -> F3dzResult<TrackedBuffer> {
    let fallback = [0u32];
    let values = if values.is_empty() { &fallback } else { values };
    tracked_create_buffer_init(
        device,
        &BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(values),
            usage,
        },
    )
    .map_err(gpu_error)
}

fn storage_buffer(device: &Device, label: &'static str, words: usize) -> F3dzResult<TrackedBuffer> {
    tracked_create_buffer(
        device,
        &BufferDescriptor {
            label: Some(label),
            size: byte_size(words.max(1))?,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        },
    )
    .map_err(gpu_error)
}

fn read_buffer(
    device: &Device,
    queue: &Queue,
    source: &TrackedBuffer,
    size: u64,
) -> F3dzResult<Vec<u8>> {
    let readback = tracked_create_buffer(
        device,
        &BufferDescriptor {
            label: Some("f3dz.decode.readback"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )
    .map_err(gpu_error)?;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("f3dz.decode.readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..size);
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    crate::core::gpu::wait_for_device_idle(&device);
    pollster::block_on(receiver.receive())
        .ok_or_else(|| F3dzError::GpuUnavailable("GPU readback was cancelled".to_string()))?
        .map_err(|error| F3dzError::GpuUnavailable(format!("GPU readback failed: {error}")))?;
    let bytes = slice.get_mapped_range().to_vec();
    readback.unmap();
    Ok(bytes)
}

fn wait_for_queue(device: &Device, queue: &Queue, timeout: Duration) -> F3dzResult<()> {
    let (sender, receiver) = std::sync::mpsc::channel();
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });
    let started = Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        match receiver.try_recv() {
            Ok(()) => return Ok(()),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(F3dzError::GpuUnavailable(
                    "F3DZ benchmark completion callback disconnected".to_string(),
                ));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        if started.elapsed() >= timeout {
            return Err(F3dzError::GpuUnavailable(format!(
                "F3DZ benchmark exceeded {:.0}-second GPU timeout",
                timeout.as_secs_f64()
            )));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Clone)]
struct PreparedBatch {
    first_page: usize,
    page_count: usize,
    compressed: Vec<u32>,
    descriptors: Vec<u32>,
    tables: Vec<u32>,
    decoded_bytes: usize,
    samples: usize,
    gpu_fast: bool,
    gpu_direct: bool,
}

struct PreparedStream {
    header: ContainerHeader,
    batches: Vec<PreparedBatch>,
}

fn prepare_stream(data: &[u8]) -> F3dzResult<PreparedStream> {
    let validated = super::format::validate_stream(data)?;
    let header = validated.header;
    let mut batches = Vec::new();
    for (batch_index, chunk) in validated.pages.chunks(MAX_BATCH_PAGES).enumerate() {
        let first_page = batch_index * MAX_BATCH_PAGES;
        let mut batch = PreparedBatch {
            first_page,
            page_count: chunk.len(),
            compressed: Vec::new(),
            descriptors: vec![0; chunk.len() * DESC_STRIDE],
            tables: Vec::new(),
            decoded_bytes: 0,
            samples: 0,
            gpu_fast: true,
            gpu_direct: true,
        };
        for (local_page, page) in chunk.iter().enumerate() {
            prepare_page(
                &header,
                page,
                first_page + local_page,
                local_page,
                &mut batch,
            )?;
        }
        batches.push(batch);
    }
    Ok(PreparedStream { header, batches })
}

fn prepare_page(
    header: &ContainerHeader,
    page: &ValidatedPage<'_>,
    absolute_page: usize,
    local_page: usize,
    batch: &mut PreparedBatch,
) -> F3dzResult<()> {
    let entry = &page.entry;
    let sample_count = entry.sample_count as usize;
    let plane = [page.plane.x_slope, page.plane.y_slope, page.plane.intercept];
    validate_plane(plane, entry.width, entry.height).map_err(|reason| F3dzError::CorruptPage {
        page: absolute_page,
        reason,
    })?;
    let base = parse_layer(page.base_layer, sample_count, absolute_page)?;
    let enhancement = if page.enhancement_layer.is_empty() {
        None
    } else {
        Some(parse_layer(
            page.enhancement_layer,
            sample_count,
            absolute_page,
        )?)
    };

    let mut descriptor = [0u32; DESC_STRIDE];
    descriptor[0] = u32::from(entry.width);
    descriptor[1] = u32::from(entry.height);
    descriptor[2] = header.width;
    descriptor[3] = entry.page_x * u32::from(header.tile_size);
    descriptor[4] = entry.page_y * u32::from(header.tile_size);
    descriptor[7] = u32::from(enhancement.is_none());
    descriptor[8] = u32::from(entry.predictor_id);
    descriptor[9] = u32::from(page.enhancement_predictor);
    descriptor[10] = page.fine_step.to_bits();
    descriptor[11] = page.base_step.to_bits();
    descriptor[12] = plane[0] as u32;
    descriptor[13] = plane[1] as u32;
    descriptor[14] = plane[2] as u32;
    descriptor[33] = batch.samples as u32;
    descriptor[34] = page.base_scale;
    descriptor[35] = batch.samples as u32;
    descriptor[36] = page.enhancement_scale;
    descriptor[37] = if entry.gpu_direct() {
        2
    } else {
        u32::from(entry.gpu_fast())
    };
    descriptor[38] = sample_count as u32;
    descriptor[39] = entry.nan_count;
    pack_layer(&base, 15, &mut descriptor, batch)?;
    if let Some(enhancement) = enhancement {
        pack_layer(&enhancement, 24, &mut descriptor, batch)?;
    }
    batch.descriptors[local_page * DESC_STRIDE..(local_page + 1) * DESC_STRIDE]
        .copy_from_slice(&descriptor);
    batch.samples = batch
        .samples
        .checked_add(sample_count)
        .ok_or_else(|| F3dzError::InvalidHeader("batch sample count overflow".into()))?;
    batch.gpu_fast &= entry.gpu_fast();
    batch.gpu_direct &= entry.gpu_direct();
    Ok(())
}

struct PreparedLayer {
    rans: RansEncoded,
}

fn parse_layer(data: &[u8], sample_count: usize, page: usize) -> F3dzResult<PreparedLayer> {
    let (rans, consumed) =
        RansEncoded::from_bytes(data).map_err(|error| F3dzError::CorruptPage {
            page,
            reason: error.to_string(),
        })?;
    let maximum = sample_count
        .checked_mul(5)
        .ok_or_else(|| F3dzError::InvalidHeader("escape byte count overflow".into()))?;
    if consumed != data.len()
        || !(1..=maximum).contains(&(rans.decoded_len as usize))
        || rans.states.iter().any(|&state| state < RANS_L)
    {
        return corrupt(page, "invalid canonical rANS layer".to_string());
    }
    Ok(PreparedLayer { rans })
}

fn pack_layer(
    layer: &PreparedLayer,
    field: usize,
    descriptor: &mut [u32],
    batch: &mut PreparedBatch,
) -> F3dzResult<()> {
    let lane0_offset = batch.compressed.len();
    batch
        .compressed
        .extend(layer.rans.lanes[0].iter().map(|&byte| u32::from(byte)));
    let lane1_offset = batch.compressed.len();
    batch
        .compressed
        .extend(layer.rans.lanes[1].iter().map(|&byte| u32::from(byte)));
    let table_offset = batch.tables.len();
    let mut cumulative = 0u32;
    let mut lookup = vec![0u32; SCALE as usize];
    for (symbol, &frequency) in layer.rans.frequencies.iter().enumerate() {
        let end = cumulative + u32::from(frequency);
        lookup[cumulative as usize..end as usize].fill(symbol as u32);
        cumulative = end;
    }
    batch.tables.extend_from_slice(&lookup);
    batch
        .tables
        .extend(layer.rans.frequencies.iter().map(|&value| u32::from(value)));
    let mut start = 0u32;
    for &frequency in &layer.rans.frequencies {
        batch.tables.push(start);
        start += u32::from(frequency);
    }
    debug_assert_eq!(batch.tables.len() - table_offset, TABLE_STRIDE);
    descriptor[field] = layer.rans.decoded_len;
    descriptor[field + 1] = layer.rans.states[0];
    descriptor[field + 2] = layer.rans.states[1];
    descriptor[field + 3] = lane0_offset as u32;
    descriptor[field + 4] = layer.rans.lanes[0].len() as u32;
    descriptor[field + 5] = lane1_offset as u32;
    descriptor[field + 6] = layer.rans.lanes[1].len() as u32;
    descriptor[field + 7] = table_offset as u32;
    descriptor[field + 8] = batch.decoded_bytes as u32;
    batch.decoded_bytes = batch
        .decoded_bytes
        .checked_add(layer.rans.decoded_len as usize)
        .ok_or_else(|| F3dzError::InvalidHeader("batch byte scratch overflow".into()))?;
    Ok(())
}

fn validate_plane(plane: [i32; 3], width: u16, height: u16) -> Result<(), String> {
    let xs = [0i64, i64::from(width.saturating_sub(1))];
    let ys = [0i64, i64::from(height.saturating_sub(1))];
    for x in xs {
        for y in ys {
            let x_term = i64::from(plane[0]) * x;
            let xy = x_term + i64::from(plane[1]) * y;
            let result = xy + i64::from(plane[2]);
            if i32::try_from(x_term).is_err()
                || i32::try_from(xy).is_err()
                || i32::try_from(result).is_err()
            {
                return Err("plane predictor arithmetic exceeds GPU i32".to_string());
            }
        }
    }
    Ok(())
}

fn checked_output_len(header: &ContainerHeader) -> F3dzResult<usize> {
    (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or_else(|| F3dzError::InvalidHeader("DEM element count overflow".to_string()))
}

fn byte_size(words: usize) -> F3dzResult<u64> {
    u64::try_from(
        words
            .checked_mul(4)
            .ok_or_else(|| F3dzError::InvalidHeader("GPU buffer size overflow".to_string()))?,
    )
    .map_err(|_| F3dzError::InvalidHeader("GPU buffer exceeds u64".to_string()))
}

fn record_base_quality() {
    crate::core::degradation::record_degradation(
        "base_quality",
        "f3dz_unrefined_pages",
        "terrain heights were decoded from the progressive base layer at a declared 4*epsilon bound",
    );
}

fn corrupt<T>(page: usize, reason: String) -> F3dzResult<T> {
    Err(F3dzError::CorruptPage { page, reason })
}

fn gpu_error(error: impl std::fmt::Display) -> F3dzError {
    F3dzError::GpuUnavailable(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::f3dz::{base_only_stream, decode_dem, encode_dem, EncodeOptions};

    fn terrain(width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|index| {
                let x = (index % width) as f32;
                let y = (index / width) as f32;
                if index == width + 2 {
                    f32::from_bits(0x7fc0_0042)
                } else {
                    314.0 + x * 0.31 - y * 0.17 + ((x * y) % 11.0) * 0.07
                }
            })
            .collect()
    }

    #[test]
    fn decode_shader_parses_and_validates() {
        let source = include_str!("../../shaders/f3dz_decode.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn decode_shader_synchronizes_storage_between_codec_stages() {
        let source = include_str!("../../shaders/f3dz_decode.wgsl");
        assert!(
            source.contains("fn decode_rans_lane(page: u32, layer: u32, lane: u32)")
                && source.contains("var position = 0u;")
                && source
                    .contains("for (var index = lane; index < decoded_len; index = index + 2u)")
                && source.contains("if (lid < 2u)"),
            "the two rANS lanes must decode concurrently with scalar cursors"
        );
        assert!(
            source.contains(
                "for (var diagonal = 0u; diagonal < MAX_PAGE_DIAGONALS; diagonal = diagonal + 1u)"
            ) && source.contains("reconstruct_base(page, lid, atomicLoad(&status[page]) == 0u);")
                && !source[source.find("fn decode_page").unwrap()
                    ..source.find("@compute @workgroup_size").unwrap()]
                    .contains("return;"),
            "all codec-stage barriers must remain in uniform control flow for FXC/D3D11"
        );
        assert!(
            source.contains("fn reconstruct_base_fast(page: u32, lid: u32)")
                && source.contains("decode_page_fast(group_id.x, local_id.x, false);")
                && source.contains("decode_page_fast(group_id.x, local_id.x, true);")
                && source.contains("value = add_checked(work_q[index - 1u], value, page);")
                && source.contains(
                    "work_q[index] = add_checked(work_q[index - width], work_q[index], page);"
                ),
            "the proven Lorenzo pipeline must keep checked horizontal/vertical prefix scans"
        );
        let base_publish = source
            .find("q_scratch[base_q_offset + index] = work_q[index];")
            .unwrap();
        let enhancement = source.find("fn reconstruct_enhancement").unwrap();
        assert!(
            source[base_publish..enhancement].contains("storageBarrier();"),
            "q_scratch must be storage-synchronized before refinement consumes it"
        );
        let decode_page = &source[source.find("fn decode_page").unwrap()..];
        for (decode, layer) in [
            (
                "decode_rans_lane(page, 0u, lid);",
                "parse_tokens(page, 0u);",
            ),
            (
                "decode_rans_lane(page, 1u, lid);",
                "parse_tokens(page, 1u);",
            ),
        ] {
            let entropy_publish = decode_page.find(decode).unwrap();
            let publish = decode_page.find(layer).unwrap();
            let entropy_to_tokens = &decode_page[entropy_publish + decode.len()..publish];
            let storage = entropy_to_tokens.find("storageBarrier();").unwrap();
            let workgroup = entropy_to_tokens.find("workgroupBarrier();").unwrap();
            assert!(
                storage < workgroup,
                "{decode} must publish byte_scratch before {layer}"
            );
            let suffix = &decode_page[publish + layer.len()..];
            let storage = suffix.find("storageBarrier();").unwrap();
            let workgroup = suffix.find("workgroupBarrier();").unwrap();
            assert!(
                storage < workgroup,
                "{layer} must publish storage before releasing the workgroup"
            );
        }
        assert!(
            source.contains("fn parse_tokens_direct(page: u32, layer: u32, lid: u32)")
                && source
                    .contains("for (var index = lid; index < sample_count; index = index + 64u)")
                && source.contains("parse_tokens_direct(page, 0u, lid);")
                && source.contains("parse_tokens_direct(page, 1u, lid);"),
            "direct-token pages must parse samples across the full workgroup"
        );
    }

    #[test]
    fn gpu_matches_cpu_bit_for_bit_for_refined_and_base_streams() {
        let source = terrain(67, 65);
        let finite_source: Vec<f32> = source
            .iter()
            .map(|&value| if value.is_nan() { 314.0 } else { value })
            .collect();
        let mut options = EncodeOptions::new(0.1);
        options.tile_size = 64;
        let refined = encode_dem(&source, 67, 65, &options).unwrap();
        let base = base_only_stream(&refined).unwrap();
        let mut fast_options = options.clone();
        fast_options.epsilon = 0.125;
        let fast_refined = encode_dem(&finite_source, 67, 65, &fast_options).unwrap();
        let fast_base = base_only_stream(&fast_refined).unwrap();
        assert!(prepare_stream(&fast_refined)
            .unwrap()
            .batches
            .iter()
            .all(|batch| batch.gpu_fast));
        let context = match crate::core::gpu::try_ctx() {
            Ok(context) => context,
            Err(error) => {
                assert_ne!(
                    std::env::var("FORGE3D_REQUIRE_F3DZ_GPU").as_deref(),
                    Ok("1"),
                    "physical F3DZ GPU was required, but context creation failed: {error}"
                );
                eprintln!("GPU identity test skipped: {error}");
                return;
            }
        };
        let adapter_info = context.adapter.get_info();
        if let Err(error) = ensure_hardware_adapter(context.software_fallback, &adapter_info) {
            assert_ne!(
                std::env::var("FORGE3D_REQUIRE_F3DZ_GPU").as_deref(),
                Ok("1"),
                "physical F3DZ GPU was required, but wgpu selected software adapter {adapter_info:?}"
            );
            eprintln!(
                "GPU identity test skipped on software adapter {adapter_info:?}: {error}; \
                 physical adapters remain mandatory in the zero-skip F3DZ CI lane",
            );
            return;
        }
        let decoder = F3dzGpuDecoder::new(context.device.as_ref()).unwrap();
        for stream in [&refined, &base, &fast_refined, &fast_base] {
            let cpu = decode_dem(stream, None).unwrap();
            let (_, gpu) = decoder
                .decode_to_vec(context.device.as_ref(), context.queue.as_ref(), stream)
                .unwrap();
            assert_eq!(gpu.len(), cpu.values.len());
            for (index, (gpu, cpu)) in gpu.iter().zip(&cpu.values).enumerate() {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "CPU/GPU bit mismatch at sample {index}"
                );
            }
        }
    }

    #[test]
    fn software_adapters_are_rejected_before_codec_dispatch() {
        let software = wgpu::AdapterInfo {
            name: "Microsoft Basic Render Driver".to_string(),
            vendor: 0x1414,
            device: 0,
            device_type: wgpu::DeviceType::Cpu,
            driver: "WARP".to_string(),
            driver_info: String::new(),
            backend: wgpu::Backend::Dx12,
        };
        let hardware = wgpu::AdapterInfo {
            name: "NVIDIA GeForce RTX 3070".to_string(),
            vendor: 0x10de,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "NVIDIA".to_string(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert!(matches!(
            ensure_hardware_adapter(true, &software),
            Err(F3dzError::GpuUnavailable(message))
                if message.contains("refuses software/CPU adapter")
        ));
        assert!(ensure_hardware_adapter(false, &software).is_err());
        assert!(ensure_hardware_adapter(true, &hardware).is_err());
        assert!(ensure_hardware_adapter(false, &hardware).is_ok());
    }

    #[test]
    fn gpu_preflight_rejects_duplicate_page_coordinates() {
        let source = terrain(128, 64);
        let stream = encode_dem(&source, 128, 64, &EncodeOptions::new(0.1)).unwrap();
        let mut corrupt = stream;
        // Empty datum makes the second 64-byte index entry start at byte 128.
        corrupt[128..132].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            validate_stream(&corrupt),
            Err(F3dzError::CorruptPage { page: 1, reason })
                if reason.contains("canonical row-major")
        ));
    }

    #[test]
    fn gpu_preparation_does_not_entropy_decode() {
        let stream = encode_dem(&terrain(8, 8), 8, 8, &EncodeOptions::new(0.1)).unwrap();
        let (header, entries) = super::super::format::parse_prefix(&stream).unwrap();
        let entry = &entries[0];
        let start = entry.payload_offset as usize;
        let end = start + entry.payload_len as usize;
        let layer_start = start + super::super::format::PAGE_HEADER_LEN;
        let (mut layer, consumed) = RansEncoded::from_bytes(&stream[layer_start..end]).unwrap();
        let mut corrupt = stream;
        // A normalized initial state passes rANS framing but cannot decode the
        // original canonical symbol stream. Preparation must leave it to WGSL.
        layer.states[0] = u32::MAX;
        let bytes = layer.to_bytes().unwrap();
        assert_eq!(bytes.len(), consumed);
        corrupt[layer_start..layer_start + consumed].copy_from_slice(&bytes);
        let checksum = super::super::format::crc32(&corrupt[start..end]);
        let crc_offset = header.index_offset as usize + 32;
        corrupt[crc_offset..crc_offset + 4].copy_from_slice(&checksum.to_le_bytes());
        assert!(super::super::format::validate_stream(&corrupt).is_ok());
        assert!(validate_stream(&corrupt).is_ok());
        assert!(decode_dem(&corrupt, None).is_err());
    }
}
