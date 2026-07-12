use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use wgpu::{BufferUsages, CommandEncoder, Device, Queue};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueueHeader {
    pub in_count: u32,
    pub out_count: u32,
    pub capacity: u32,
    pub _pad: u32,
}

impl QueueHeader {
    pub fn new(capacity: u32) -> Self {
        Self {
            in_count: 0,
            out_count: 0,
            capacity,
            _pad: 0,
        }
    }

    pub fn active_count(&self) -> u32 {
        self.in_count.saturating_sub(self.out_count)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ray {
    pub o: [f32; 3],
    pub tmin: f32,
    pub d: [f32; 3],
    pub tmax: f32,
    pub throughput: [f32; 3],
    pub pdf: f32,
    pub pixel: u32,
    pub depth: u32,
    pub rng_hi: u32,
    pub rng_lo: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Hit {
    pub p: [f32; 3],
    pub t: f32,
    pub n: [f32; 3],
    pub wo: [f32; 3],
    pub _pad_wo: f32,
    pub mat: u32,
    pub throughput: [f32; 3],
    pub pdf: f32,
    pub pixel: u32,
    pub depth: u32,
    pub rng_hi: u32,
    pub rng_lo: u32,
    pub tangent: [f32; 3],
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterRay {
    pub o: [f32; 3],
    pub tmin: f32,
    pub d: [f32; 3],
    pub tmax: f32,
    pub throughput: [f32; 3],
    pub pdf: f32,
    pub pixel: u32,
    pub depth: u32,
    pub rng_hi: u32,
    pub rng_lo: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowRay {
    pub o: [f32; 3],
    pub tmin: f32,
    pub d: [f32; 3],
    pub tmax: f32,
    pub contrib: [f32; 3],
    pub _pad0: f32,
    pub pixel: u32,
    pub _pad1: [u32; 3],
}

pub struct QueueBuffers {
    pub capacity: u32,
    pub ray_queue_header: TrackedBuffer,
    pub hit_queue_header: TrackedBuffer,
    pub scatter_queue_header: TrackedBuffer,
    pub miss_queue_header: TrackedBuffer,
    pub ray_queue: TrackedBuffer,
    pub hit_queue: TrackedBuffer,
    pub scatter_queue: TrackedBuffer,
    pub miss_queue: TrackedBuffer,
    pub shadow_queue_header: TrackedBuffer,
    pub shadow_queue: TrackedBuffer,
}

impl QueueBuffers {
    pub fn new(device: &Device, capacity: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let header_size = std::mem::size_of::<QueueHeader>() as u64;
        let ray_queue_header = create_header_buffer(device, "ray-queue-header", header_size)?;
        let hit_queue_header = create_header_buffer(device, "hit-queue-header", header_size)?;
        let scatter_queue_header =
            create_header_buffer(device, "scatter-queue-header", header_size)?;
        let miss_queue_header = create_header_buffer(device, "miss-queue-header", header_size)?;
        let shadow_queue_header = create_header_buffer(device, "shadow-queue-header", header_size)?;

        let ray_size = (std::mem::size_of::<Ray>() * capacity as usize) as u64;
        let hit_size = (std::mem::size_of::<Hit>() * capacity as usize) as u64;
        let scatter_size = (std::mem::size_of::<ScatterRay>() * capacity as usize) as u64;
        let shadow_size = (std::mem::size_of::<ShadowRay>() * capacity as usize) as u64;

        Ok(Self {
            capacity,
            ray_queue_header,
            hit_queue_header,
            scatter_queue_header,
            miss_queue_header,
            ray_queue: create_data_buffer(device, "ray-queue", ray_size)?,
            hit_queue: create_data_buffer(device, "hit-queue", hit_size)?,
            scatter_queue: create_data_buffer(device, "scatter-queue", scatter_size)?,
            miss_queue: create_data_buffer(device, "miss-queue", ray_size)?,
            shadow_queue_header,
            shadow_queue: create_data_buffer(device, "shadow-queue", shadow_size)?,
        })
    }

    pub fn reset_counters(&self, queue: &Queue, _encoder: &mut CommandEncoder) {
        let zero_header = QueueHeader::new(self.capacity);
        let header_arr = [zero_header];
        let header_data = bytemuck::cast_slice(&header_arr);

        queue.write_buffer(&self.ray_queue_header, 0, header_data);
        queue.write_buffer(&self.hit_queue_header, 0, header_data);
        queue.write_buffer(&self.scatter_queue_header, 0, header_data);
        queue.write_buffer(&self.miss_queue_header, 0, header_data);
        queue.write_buffer(&self.shadow_queue_header, 0, header_data);
    }

    pub fn get_active_ray_count(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(self
            .read_ray_queue_header(device, queue, encoder)?
            .active_count())
    }

    /// Read back the full ray-queue header (submits all pending encoder work
    /// and stalls until the copy completes; `encoder` is replaced with a fresh
    /// one). The caller owns interpreting `in_count`/`out_count`: after an
    /// iteration the persistent-thread pop pattern leaves `out_count`
    /// over-incremented past `in_count` (every launched thread's final failed
    /// pop still bumps the counter), so `active_count()` is only meaningful
    /// straight after raygen — mid-frame accounting must be reconstructed on
    /// the CPU (see `WavefrontScheduler::render_frame_simple`).
    pub fn read_ray_queue_header(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
    ) -> Result<QueueHeader, Box<dyn std::error::Error>> {
        let header_size = std::mem::size_of::<QueueHeader>() as u64;
        // `tracked_create_buffer` records this host-visible readback allocation
        // in the global ledger and releases it when `staging` drops below.
        let staging = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("ray-queue-header-readback"),
                size: header_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;

        encoder.copy_buffer_to_buffer(&self.ray_queue_header, 0, &staging, 0, header_size);
        let next_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wavefront-after-active-ray-count"),
        });
        let pending = std::mem::replace(encoder, next_encoder);
        queue.submit(Some(pending.finish()));

        let read_result = (|| -> Result<QueueHeader, Box<dyn std::error::Error>> {
            let slice = staging.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).ok();
            });
            device.poll(wgpu::Maintain::Wait);
            let map_result = pollster::block_on(receiver.receive())
                .ok_or("ray queue header readback callback dropped")?;
            map_result.map_err(|e| format!("ray queue header readback failed: {e:?}"))?;
            let data = slice.get_mapped_range();
            let header = *bytemuck::from_bytes::<QueueHeader>(&data[..header_size as usize]);
            drop(data);
            staging.unmap();
            Ok(header)
        })();

        read_result
    }

    /// Prepare the queues for one wavefront iteration.
    ///
    /// The ray queue is append-only within a frame: raygen and the scatter
    /// stage push at monotonically increasing `in_count` indices, while the
    /// persistent-thread pop pattern over-increments `out_count` once a wave
    /// drains. Rewriting `out_count` to the exact number of rays consumed in
    /// completed iterations makes both the active count and the next wave's
    /// pop indices correct.
    ///
    /// The hit/scatter/shadow/miss queues are strictly intra-iteration
    /// (produced and consumed inside a single wave), so their headers are
    /// reset to empty each iteration; this also keeps them from accumulating
    /// the same out-count corruption and from overflowing across bounces.
    ///
    /// Must be called after all previously submitted GPU work has completed
    /// (i.e. after `read_ray_queue_header`, which stalls) and before the
    /// iteration's dispatches are submitted.
    pub fn begin_iteration(&self, queue: &Queue, consumed_rays: u32) {
        // QueueHeader layout: in_count @0, out_count @4, capacity @8.
        queue.write_buffer(
            &self.ray_queue_header,
            4,
            bytemuck::bytes_of(&consumed_rays),
        );
        let empty: [u32; 2] = [0, 0];
        for header in [
            &self.hit_queue_header,
            &self.scatter_queue_header,
            &self.shadow_queue_header,
            &self.miss_queue_header,
        ] {
            queue.write_buffer(header, 0, bytemuck::cast_slice(&empty));
        }
    }
}

fn create_header_buffer(
    device: &Device,
    label: &str,
    size: u64,
) -> Result<TrackedBuffer, RenderError> {
    tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )
}

fn create_data_buffer(
    device: &Device,
    label: &str,
    size: u64,
) -> Result<TrackedBuffer, RenderError> {
    tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_queue() -> Option<(Device, Queue)> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
    }

    #[test]
    fn active_ray_count_reads_queue_header_state() {
        let Some((device, queue)) = device_queue() else {
            return;
        };
        let queues = QueueBuffers::new(&device, 16).expect("queue buffers");
        let header = QueueHeader {
            in_count: 7,
            out_count: 2,
            capacity: 16,
            _pad: 0,
        };
        queue.write_buffer(&queues.ray_queue_header, 0, bytemuck::bytes_of(&header));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("active-ray-count-test"),
        });

        let got = queues
            .get_active_ray_count(&device, &queue, &mut encoder)
            .expect("active ray count");

        assert_eq!(got, 5);
    }
}
