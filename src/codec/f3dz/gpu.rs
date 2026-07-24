//! Fail-closed F3DZ decode on wgpu.
//!
//! The CPU validates the container, CRCs, page contracts, and adaptive rANS
//! tables, but deliberately does not entropy-decode a single symbol. Compressed
//! lanes are copied to storage buffers and the WGSL kernel performs rANS,
//! predictor reconstruction, and final binary32 conversion. Dispatches are
//! bounded to [`MAX_BATCH_PAGES`] so staging stays independent of DEM size.

use super::encode::PAGE_HEADER_LEN;
use super::format::{
    crc32, parse_prefix, ContainerHeader, PageIndexEntry, PAGE_FLAG_BASE_ONLY,
    PAGE_FLAG_PROGRESSIVE, PAGE_MAGIC, PREDICTOR_LORENZO, PREDICTOR_ORDER_ZERO, PREDICTOR_PLANE,
    PREDICTOR_PREVIOUS_LOD, VERSION,
};
use super::rans::{RansEncoded, RANS_L, SCALE};
use super::{F3dzError, F3dzResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture, TrackedBuffer,
};
use futures_intrusive::channel::shared::oneshot_channel;
use std::num::NonZeroU64;
use std::time::Instant;
use wgpu::util::BufferInitDescriptor;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Extent3d, PipelineLayoutDescriptor, Queue, ShaderStages, StorageTextureAccess, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

pub const MAX_BATCH_PAGES: usize = 64;
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

pub struct F3dzGpuDecoder {
    bind_group_layout: BindGroupLayout,
    buffer_pipeline: ComputePipeline,
    atlas_pipeline: ComputePipeline,
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
        let buffer_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("f3dz.decode.to-buffer"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "decode_to_buffer",
            },
        );
        let atlas_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("f3dz.decode.to-atlas"),
                layout: Some(&layout),
                module: &shader,
                entry_point: "decode_to_atlas",
            },
        );
        Ok(Self {
            bind_group_layout,
            buffer_pipeline,
            atlas_pipeline,
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
            self.dispatch_batch(
                device,
                queue,
                batch,
                &output,
                &dummy_view,
                &self.buffer_pipeline,
            )?;
        }
        let bytes = read_buffer(device, queue, &output, byte_size(output_len)?)?;
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
            .collect();
        if prepared.header.base_only() {
            record_base_quality();
        }
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
            self.dispatch_batch(
                device,
                queue,
                &batch,
                &dummy_output,
                &atlas_view,
                &self.atlas_pipeline,
            )?;
        }
        if prepared.header.base_only() {
            record_base_quality();
        }
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

/// Validate all container/page metadata, CRCs, and rANS framing without
/// entropy-decoding. Streaming uses this before it reserves an atlas slot.
pub fn validate_stream(data: &[u8]) -> F3dzResult<ContainerHeader> {
    prepare_stream(data).map(|prepared| prepared.header)
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
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver.receive())
        .ok_or_else(|| F3dzError::GpuUnavailable("GPU readback was cancelled".to_string()))?
        .map_err(|error| F3dzError::GpuUnavailable(format!("GPU readback failed: {error}")))?;
    let bytes = slice.get_mapped_range().to_vec();
    readback.unmap();
    Ok(bytes)
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
}

struct PreparedStream {
    header: ContainerHeader,
    batches: Vec<PreparedBatch>,
}

fn prepare_stream(data: &[u8]) -> F3dzResult<PreparedStream> {
    let (header, entries) = parse_prefix(data)?;
    let mut expected_offset = header.payload_offset;
    let mut batches = Vec::new();
    for (batch_index, chunk) in entries.chunks(MAX_BATCH_PAGES).enumerate() {
        let first_page = batch_index * MAX_BATCH_PAGES;
        let mut batch = PreparedBatch {
            first_page,
            page_count: chunk.len(),
            compressed: Vec::new(),
            descriptors: vec![0; chunk.len() * DESC_STRIDE],
            tables: Vec::new(),
            decoded_bytes: 0,
            samples: 0,
        };
        for (local_page, entry) in chunk.iter().enumerate() {
            let absolute_page = first_page + local_page;
            let pages_x = header.width.div_ceil(u32::from(header.tile_size));
            if entry.page_x != absolute_page as u32 % pages_x
                || entry.page_y != absolute_page as u32 / pages_x
            {
                return corrupt(
                    absolute_page,
                    "page index entries must appear once in canonical row-major order".to_string(),
                );
            }
            if entry.payload_offset != expected_offset {
                return corrupt(
                    absolute_page,
                    format!(
                        "payload offset is not canonical: expected={expected_offset} actual={}",
                        entry.payload_offset
                    ),
                );
            }
            let start = usize::try_from(entry.payload_offset)
                .map_err(|_| F3dzError::InvalidHeader("payload offset exceeds usize".into()))?;
            let end = start
                .checked_add(entry.payload_len as usize)
                .ok_or_else(|| F3dzError::InvalidHeader("payload range overflow".into()))?;
            if data.len() < end {
                return Err(F3dzError::Truncated {
                    needed: end,
                    available: data.len(),
                });
            }
            let payload = &data[start..end];
            if crc32(payload) != entry.crc32 {
                return corrupt(absolute_page, "CRC mismatch".to_string());
            }
            prepare_page(
                &header,
                entry,
                payload,
                absolute_page,
                local_page,
                &mut batch,
            )?;
            expected_offset = expected_offset
                .checked_add(u64::from(entry.payload_len))
                .ok_or_else(|| F3dzError::InvalidHeader("payload offset overflow".into()))?;
        }
        batches.push(batch);
    }
    if expected_offset != data.len() as u64 {
        return Err(F3dzError::InvalidHeader(format!(
            "container has trailing or missing payload bytes: indexed_end={expected_offset} actual={}",
            data.len()
        )));
    }
    Ok(PreparedStream { header, batches })
}

#[allow(clippy::too_many_arguments)]
fn prepare_page(
    header: &ContainerHeader,
    entry: &PageIndexEntry,
    payload: &[u8],
    absolute_page: usize,
    local_page: usize,
    batch: &mut PreparedBatch,
) -> F3dzResult<()> {
    if payload.len() < PAGE_HEADER_LEN
        || payload[0..4] != PAGE_MAGIC
        || get_u16(payload, 4) != VERSION
    {
        return corrupt(absolute_page, "invalid page header".to_string());
    }
    let expected_flags = if header.progressive() {
        u16::from(PAGE_FLAG_PROGRESSIVE)
            | if header.base_only() {
                u16::from(PAGE_FLAG_BASE_ONLY)
            } else {
                0
            }
    } else {
        0
    };
    if get_u16(payload, 6) != expected_flags
        || get_u16(payload, 10) != 0
        || get_u32(payload, 44) != 0
    {
        return corrupt(
            absolute_page,
            "page flags/reserved fields disagree with container".to_string(),
        );
    }
    let predictor = payload[8];
    let enhancement_predictor = payload[9];
    if !matches!(
        predictor,
        PREDICTOR_LORENZO | PREDICTOR_PLANE | PREDICTOR_ORDER_ZERO
    ) || (header.progressive() && enhancement_predictor != PREDICTOR_PREVIOUS_LOD)
        || (!header.progressive() && enhancement_predictor != predictor)
        || predictor != entry.predictor_id
    {
        return corrupt(absolute_page, "invalid predictor contract".to_string());
    }
    let sample_count = usize::from(entry.width)
        .checked_mul(usize::from(entry.height))
        .ok_or_else(|| F3dzError::InvalidHeader("page sample count overflow".into()))?;
    if get_u32(payload, 32) as usize != sample_count || entry.sample_count as usize != sample_count
    {
        return corrupt(absolute_page, "page sample count mismatch".to_string());
    }
    let fine_step = f32::from_bits(get_u32(payload, 36));
    let base_step = f32::from_bits(get_u32(payload, 40));
    let expected_fine = header.epsilon * 2.0;
    let expected_base = if header.progressive() {
        header.epsilon * 8.0
    } else {
        expected_fine
    };
    if fine_step.to_bits() != expected_fine.to_bits()
        || base_step.to_bits() != expected_base.to_bits()
    {
        return corrupt(
            absolute_page,
            "quantization step disagrees with epsilon".to_string(),
        );
    }
    let plane = [
        get_i32(payload, 12),
        get_i32(payload, 16),
        get_i32(payload, 20),
    ];
    if predictor != PREDICTOR_PLANE && plane != [0; 3] {
        return corrupt(
            absolute_page,
            "non-plane predictor carries plane coefficients".to_string(),
        );
    }
    validate_plane(plane, entry.width, entry.height).map_err(|reason| F3dzError::CorruptPage {
        page: absolute_page,
        reason,
    })?;
    let base_len = get_u32(payload, 24) as usize;
    let enhancement_len = get_u32(payload, 28) as usize;
    let base_end = PAGE_HEADER_LEN
        .checked_add(base_len)
        .ok_or_else(|| F3dzError::InvalidHeader("base layer overflow".into()))?;
    let payload_end = base_end
        .checked_add(enhancement_len)
        .ok_or_else(|| F3dzError::InvalidHeader("enhancement layer overflow".into()))?;
    if base_end != entry.base_layer_len as usize || payload_end != payload.len() {
        return corrupt(absolute_page, "page layer lengths mismatch".to_string());
    }
    if (header.base_only() || !header.progressive()) && enhancement_len != 0 {
        return corrupt(
            absolute_page,
            "quality mode forbids enhancement bytes".to_string(),
        );
    }
    if header.progressive() && !header.base_only() && enhancement_len == 0 {
        return corrupt(
            absolute_page,
            "refined page is missing enhancement bytes".to_string(),
        );
    }
    let base = parse_layer(
        &payload[PAGE_HEADER_LEN..base_end],
        sample_count,
        absolute_page,
    )?;
    let enhancement = if enhancement_len == 0 {
        None
    } else {
        Some(parse_layer(
            &payload[base_end..payload_end],
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
    descriptor[8] = u32::from(predictor);
    descriptor[9] = u32::from(enhancement_predictor);
    descriptor[10] = fine_step.to_bits();
    descriptor[11] = base_step.to_bits();
    descriptor[12] = plane[0] as u32;
    descriptor[13] = plane[1] as u32;
    descriptor[14] = plane[2] as u32;
    descriptor[33] = batch.samples as u32;
    descriptor[35] = batch.samples as u32;
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
    let minimum = sample_count
        .checked_mul(4)
        .ok_or_else(|| F3dzError::InvalidHeader("token byte count overflow".into()))?;
    let maximum = minimum
        .checked_mul(2)
        .ok_or_else(|| F3dzError::InvalidHeader("escape byte count overflow".into()))?;
    if consumed != data.len()
        || !(minimum..=maximum).contains(&(rans.decoded_len as usize))
        || !rans.decoded_len.is_multiple_of(4)
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

fn get_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn get_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn get_i32(data: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
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
    fn gpu_matches_cpu_bit_for_bit_for_refined_and_base_streams() {
        let source = terrain(67, 65);
        let mut options = EncodeOptions::new(0.1);
        options.tile_size = 64;
        let refined = encode_dem(&source, 67, 65, &options).unwrap();
        let base = base_only_stream(&refined).unwrap();
        let context = match crate::core::gpu::try_ctx() {
            Ok(context) => context,
            Err(error) => {
                eprintln!("GPU identity test skipped: {error}");
                return;
            }
        };
        let decoder = F3dzGpuDecoder::new(context.device.as_ref()).unwrap();
        for stream in [&refined, &base] {
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
}
