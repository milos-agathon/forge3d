// src/path_tracing/wavefront/queues.rs
// Queue management structures for wavefront path tracing
// Handles GPU buffers for rays, hits, scatter rays, and miss rays

use wgpu::{Device, Queue, CommandEncoder, Buffer, BufferUsages, BindGroup, BindGroupLayout};
use std::sync::Arc;

/// GPU buffer structures for queue headers
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueueHeader {
    pub in_count: u32,     // number of items pushed
    pub out_count: u32,    // number of items popped
    pub capacity: u32,     // maximum capacity
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

/// Ray structure matching WGSL layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ray {
    pub o: [f32; 3],           // origin
    pub tmin: f32,             // minimum ray parameter
    pub d: [f32; 3],           // direction
    pub tmax: f32,             // maximum ray parameter
    pub throughput: [f32; 3],  // path throughput
    pub pdf: f32,              // path pdf
    pub pixel: u32,            // pixel index
    pub depth: u32,            // bounce depth
    pub rng_hi: u32,           // RNG state high
    pub rng_lo: u32,           // RNG state low
}

/// Hit record structure matching WGSL layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Hit {
    pub p: [f32; 3],           // hit position
    pub t: f32,                // ray parameter
    pub n: [f32; 3],           // surface normal
    pub mat: u32,              // material index
    pub throughput: [f32; 3],  // inherited throughput
    pub pdf: f32,              // inherited pdf
    pub pixel: u32,            // pixel index
    pub depth: u32,            // bounce depth
    pub rng_hi: u32,           // RNG state high
    pub rng_lo: u32,           // RNG state low
}

/// Scatter ray structure matching WGSL layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterRay {
    pub o: [f32; 3],           // origin
    pub tmin: f32,             // minimum ray parameter
    pub d: [f32; 3],           // direction
    pub tmax: f32,             // maximum ray parameter
    pub throughput: [f32; 3],  // updated throughput
    pub pdf: f32,              // updated pdf
    pub pixel: u32,            // pixel index
    pub depth: u32,            // bounce depth + 1
    pub rng_hi: u32,           // updated RNG state high
    pub rng_lo: u32,           // updated RNG state low
}

/// All GPU buffers for wavefront queues
pub struct QueueBuffers {
    pub capacity: u32,
    
    // Queue headers (atomic counters)
    pub ray_queue_header: Buffer,
    pub hit_queue_header: Buffer,
    pub scatter_queue_header: Buffer,
    pub miss_queue_header: Buffer,
    
    // Queue data buffers
    pub ray_queue: Buffer,
    pub hit_queue: Buffer,
    pub scatter_queue: Buffer,
    pub miss_queue: Buffer,
    
    // Compaction buffers
    pub ray_queue_compacted: Buffer,
    pub ray_flags: Buffer,
    pub prefix_sums: Buffer,
}

impl QueueBuffers {
    /// Create all queue buffers with given capacity
    pub fn new(device: &Device, capacity: u32) -> Result<Self, Box<dyn std::error::Error>> {
        // Create queue headers
        let header_size = std::mem::size_of::<QueueHeader>() as u64;
        let ray_queue_header = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-queue-header"),
            size: header_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let hit_queue_header = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit-queue-header"),
            size: header_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let scatter_queue_header = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-queue-header"),
            size: header_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let miss_queue_header = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("miss-queue-header"),
            size: header_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create queue data buffers
        let ray_size = (std::mem::size_of::<Ray>() * capacity as usize) as u64;
        let ray_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-queue"),
            size: ray_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let hit_size = (std::mem::size_of::<Hit>() * capacity as usize) as u64;
        let hit_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit-queue"),
            size: hit_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let scatter_size = (std::mem::size_of::<ScatterRay>() * capacity as usize) as u64;
        let scatter_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter-queue"),
            size: scatter_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let miss_queue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("miss-queue"),
            size: ray_size, // Same as ray queue
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create compaction buffers
        let ray_queue_compacted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-queue-compacted"),
            size: ray_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let flags_size = (std::mem::size_of::<u32>() * capacity as usize) as u64;
        let ray_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray-flags"),
            size: flags_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let prefix_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix-sums"),
            size: flags_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Ok(Self {
            capacity,
            ray_queue_header,
            hit_queue_header,
            scatter_queue_header,
            miss_queue_header,
            ray_queue,
            hit_queue,
            scatter_queue,
            miss_queue,
            ray_queue_compacted,
            ray_flags,
            prefix_sums,
        })
    }
    
    /// Reset all queue counters for new frame
    pub fn reset_counters(&self, queue: &Queue, encoder: &mut CommandEncoder) {
        let zero_header = QueueHeader::new(self.capacity);
        // Avoid borrowing a temporary slice; bind the array first to satisfy borrow checker
        let header_arr = [zero_header];
        let header_data = bytemuck::cast_slice(&header_arr);
        
        // Reset all queue headers
        queue.write_buffer(&self.ray_queue_header, 0, header_data);
        queue.write_buffer(&self.hit_queue_header, 0, header_data);
        queue.write_buffer(&self.scatter_queue_header, 0, header_data);
        queue.write_buffer(&self.miss_queue_header, 0, header_data);
    }
    
    /// Get active ray count (requires GPU readback)
    pub fn get_active_ray_count(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        // For now, return estimate based on capacity
        // In full implementation, would do async readback
        Ok(0) // Placeholder - triggers loop termination
    }
    
    /// Create bind group for raygen stage
    pub fn create_raygen_bind_group(&self, device: &Device) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raygen-queue-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raygen-queue-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ray_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.ray_queue.as_entire_binding(),
                },
            ],
        });
        
        Ok(bind_group)
    }
    
    /// Create bind group for intersect stage
    pub fn create_intersect_bind_group(&self, device: &Device) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("intersect-queue-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("intersect-queue-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ray_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.ray_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.hit_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.hit_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.miss_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.miss_queue.as_entire_binding(),
                },
            ],
        });
        
        Ok(bind_group)
    }
    
    /// Create bind group for shade stage
    pub fn create_shade_bind_group(&self, device: &Device) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shade-queue-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade-queue-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.hit_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.hit_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.scatter_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scatter_queue.as_entire_binding(),
                },
            ],
        });
        
        Ok(bind_group)
    }
    
    /// Create bind group for scatter stage
    pub fn create_scatter_bind_group(&self, device: &Device) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter-queue-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter-queue-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scatter_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scatter_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ray_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ray_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.miss_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.miss_queue.as_entire_binding(),
                },
            ],
        });
        
        Ok(bind_group)
    }
    
    /// Create bind group for compact stage
    pub fn create_compact_bind_group(&self, device: &Device) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact-queue-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact-queue-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ray_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.ray_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ray_queue_compacted.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ray_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.prefix_sums.as_entire_binding(),
                },
            ],
        });
        
        Ok(bind_group)
    }
}
