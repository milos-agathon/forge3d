use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};

pub(crate) struct TimestampResources {
    query_set: Option<wgpu::QuerySet>,
    buffer: Option<TrackedBuffer>,
    readback: Option<TrackedBuffer>,
}

impl TimestampResources {
    pub(crate) fn new(device: &wgpu::Device) -> RenderResult<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Ok(Self {
                query_set: None,
                buffer: None,
                readback: None,
            });
        }

        Ok(Self {
            query_set: Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("brdf_tile.timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })),
            buffer: Some(tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("brdf_tile.timestamp_buffer"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )?),
            readback: Some(tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("brdf_tile.timestamp_readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )?),
        })
    }

    pub(crate) fn write_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(query_set) = &self.query_set {
            encoder.write_timestamp(query_set, 0);
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.query_set.is_some()
    }

    pub(crate) fn timestamp_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.query_set
            .as_ref()
            .map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            })
    }

    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(query_set), Some(buffer), Some(readback)) =
            (&self.query_set, &self.buffer, &self.readback)
        {
            encoder.resolve_query_set(query_set, 0..2, buffer, 0);
            encoder.copy_buffer_to_buffer(buffer, 0, readback, 0, 16);
        }
    }

    /// Read the resolved begin/end pass timestamps and return the GPU duration
    /// in milliseconds. `None` when `TIMESTAMP_QUERY` is not granted, mapping
    /// fails, or the values are implausible (zero/backwards) — callers then
    /// record the pass with `gpu_ms == 0` rather than fabricating a timing.
    /// Must be called after the encoder carrying `resolve` was submitted.
    pub(crate) fn read_gpu_ms(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<f64> {
        let readback = self.readback.as_ref()?;
        let slice = readback.slice(0..16);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        match pollster::block_on(receiver.receive()) {
            Some(Ok(())) => {}
            _ => return None,
        }
        let (begin, end) = {
            let data = slice.get_mapped_range();
            (
                u64::from_le_bytes(data[0..8].try_into().ok()?),
                u64::from_le_bytes(data[8..16].try_into().ok()?),
            )
        };
        readback.unmap();
        if begin == 0 || end < begin {
            return None;
        }
        let period = queue.get_timestamp_period() as f64;
        Some(((end - begin) as f64 * period) / 1_000_000.0)
    }
}
