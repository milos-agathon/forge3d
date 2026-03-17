use wgpu::{Buffer, BufferUsages, CommandEncoder, Device, Queue};

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
    pub ray_queue_header: Buffer,
    pub hit_queue_header: Buffer,
    pub scatter_queue_header: Buffer,
    pub miss_queue_header: Buffer,
    pub ray_queue: Buffer,
    pub hit_queue: Buffer,
    pub scatter_queue: Buffer,
    pub miss_queue: Buffer,
    pub shadow_queue_header: Buffer,
    pub shadow_queue: Buffer,
    pub ray_queue_compacted: Buffer,
    pub ray_flags: Buffer,
    pub prefix_sums: Buffer,
}

impl QueueBuffers {
    pub fn new(device: &Device, capacity: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let header_size = std::mem::size_of::<QueueHeader>() as u64;
        let ray_queue_header = create_header_buffer(device, "ray-queue-header", header_size);
        let hit_queue_header = create_header_buffer(device, "hit-queue-header", header_size);
        let scatter_queue_header =
            create_header_buffer(device, "scatter-queue-header", header_size);
        let miss_queue_header = create_header_buffer(device, "miss-queue-header", header_size);
        let shadow_queue_header = create_header_buffer(device, "shadow-queue-header", header_size);

        let ray_size = (std::mem::size_of::<Ray>() * capacity as usize) as u64;
        let hit_size = (std::mem::size_of::<Hit>() * capacity as usize) as u64;
        let scatter_size = (std::mem::size_of::<ScatterRay>() * capacity as usize) as u64;
        let shadow_size = (std::mem::size_of::<ShadowRay>() * capacity as usize) as u64;
        let flags_size = (std::mem::size_of::<u32>() * capacity as usize) as u64;

        Ok(Self {
            capacity,
            ray_queue_header,
            hit_queue_header,
            scatter_queue_header,
            miss_queue_header,
            ray_queue: create_data_buffer(device, "ray-queue", ray_size),
            hit_queue: create_data_buffer(device, "hit-queue", hit_size),
            scatter_queue: create_data_buffer(device, "scatter-queue", scatter_size),
            miss_queue: create_data_buffer(device, "miss-queue", ray_size),
            shadow_queue_header,
            shadow_queue: create_data_buffer(device, "shadow-queue", shadow_size),
            ray_queue_compacted: create_data_buffer(device, "ray-queue-compacted", ray_size),
            ray_flags: create_data_buffer(device, "ray-flags", flags_size),
            prefix_sums: create_data_buffer(device, "prefix-sums", flags_size),
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
        _device: &Device,
        _queue: &Queue,
        _encoder: &mut CommandEncoder,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(0)
    }
}

fn create_header_buffer(device: &Device, label: &str, size: u64) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_data_buffer(device: &Device, label: &str, size: u64) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
