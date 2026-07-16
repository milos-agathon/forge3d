use super::*;

struct LiveVectorPassTiming {
    device: Arc<wgpu::Device>,
    query_set: wgpu::QuerySet,
    resolve_buffer: crate::core::resource_tracker::TrackedBuffer,
    readback_buffer: crate::core::resource_tracker::TrackedBuffer,
    timestamp_period: f64,
}

/// Single-render timestamp queries using render-pass boundary writes.
///
/// Metal intermittently discarded OIT attachments when encoder-level
/// `write_timestamp` calls bracketed the passes. `RenderPassTimestampWrites`
/// expresses the same live GPU measurement at native pass boundaries without
/// perturbing the attachment lifecycle.
pub(super) struct VectorPassTiming {
    live: Option<LiveVectorPassTiming>,
    labels: Vec<(String, u32)>,
}

impl VectorPassTiming {
    pub(super) fn new(
        device: Arc<wgpu::Device>,
        queue: &wgpu::Queue,
        max_passes: u32,
    ) -> PyResult<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Ok(Self {
                live: None,
                labels: Vec::new(),
            });
        }
        let query_count = max_passes.max(1) * 2;
        let size = u64::from(query_count) * std::mem::size_of::<u64>() as u64;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("vf.Vector.PassTiming.Queries"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buffer = crate::core::resource_tracker::tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.PassTiming.Resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .map_err(vector_runtime_err)?;
        let readback_buffer = crate::core::resource_tracker::tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("vf.Vector.PassTiming.Readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .map_err(vector_runtime_err)?;
        Ok(Self {
            live: Some(LiveVectorPassTiming {
                device,
                query_set,
                resolve_buffer,
                readback_buffer,
                timestamp_period: f64::from(queue.get_timestamp_period()),
            }),
            labels: Vec::new(),
        })
    }

    pub(super) fn reserve(&mut self, label: &str, draw_calls: u32) -> Option<(u32, u32)> {
        self.live.as_ref()?;
        let begin = self.labels.len() as u32 * 2;
        self.labels.push((label.to_string(), draw_calls));
        Some((begin, begin + 1))
    }

    pub(super) fn render_pass_writes(
        &self,
        indices: Option<(u32, u32)>,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (begin, end) = indices?;
        let live = self.live.as_ref()?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &live.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(live) = self.live.as_ref() else {
            return;
        };
        let count = self.labels.len() as u32 * 2;
        if count == 0 {
            return;
        }
        let size = u64::from(count) * std::mem::size_of::<u64>() as u64;
        encoder.resolve_query_set(&live.query_set, 0..count, &live.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&live.resolve_buffer, 0, &live.readback_buffer, 0, size);
    }

    pub(super) fn record_into_certificate(self) -> bool {
        let Some(live) = self.live else {
            return false;
        };
        let count = self.labels.len() as u32 * 2;
        if count == 0 {
            return false;
        }
        let size = u64::from(count) * std::mem::size_of::<u64>() as u64;
        let slice = live.readback_buffer.slice(0..size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        live.device.poll(wgpu::Maintain::Wait);
        let Some(Ok(())) = pollster::block_on(receiver.receive()) else {
            return false;
        };
        let data = slice.get_mapped_range();
        let timestamps = bytemuck::cast_slice::<u8, u64>(&data);
        for (index, (label, draw_calls)) in self.labels.iter().enumerate() {
            let begin = timestamps[index * 2];
            let end = timestamps[index * 2 + 1];
            let gpu_ms = if begin != 0 && end >= begin {
                (end - begin) as f64 * live.timestamp_period / 1_000_000.0
            } else {
                0.0
            };
            crate::core::certificate::record_pass(label, gpu_ms, *draw_calls);
        }
        drop(data);
        live.readback_buffer.unmap();
        true
    }
}
