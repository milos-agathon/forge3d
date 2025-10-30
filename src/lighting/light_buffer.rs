// src/lighting/light_buffer.rs
// P1: Light buffer management with triple-buffering for multi-light support
// SSBO storage buffer layout (std430) for efficient GPU upload
use wgpu::{Device, Queue, Buffer, BufferUsages};
use crate::lighting::types::Light;

/// Maximum number of lights supported (P1 default)
pub const MAX_LIGHTS: usize = 16;

/// Light buffer manager with triple-buffering for TAA-friendly updates
///
/// Memory budget:
/// - Light struct: 80 bytes
/// - 16 lights × 80 bytes = 1280 bytes per buffer
/// - 3 buffers (triple-buffered) = 3840 bytes = 3.75 KB
/// - Plus metadata buffer (16 bytes × 3) = 48 bytes
/// - Total: ~4 KB (negligible)
pub struct LightBuffer {
    /// Storage buffers for light array (triple-buffered)
    buffers: [Buffer; 3],
    /// Uniform buffer for light count (triple-buffered)
    count_buffers: [Buffer; 3],
    /// Uniform buffer placeholder for environment lighting parameters
    environment_stub: Buffer,
    /// Current frame index (0, 1, 2)
    frame_index: usize,
    /// Monotonic frame counter used for quasi-random sampling offsets
    frame_counter: u64,
    /// Cached 2D R2 sequence seed for the current frame
    sequence_seed: [f32; 2],
    /// Current number of active lights
    light_count: u32,
    /// Bind group for current frame
    bind_group: Option<wgpu::BindGroup>,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl LightBuffer {
    /// Create a new light buffer manager
    pub fn new(device: &Device) -> Self {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Buffer Bind Group Layout"),
            entries: &[
                // Binding 0: Light array (SSBO, read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Light count (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create triple-buffered storage buffers
        let buffers = [
            Self::create_light_buffer(device, 0),
            Self::create_light_buffer(device, 1),
            Self::create_light_buffer(device, 2),
        ];

        // Create triple-buffered count buffers
        let count_buffers = [
            Self::create_count_buffer(device, 0),
            Self::create_count_buffer(device, 1),
            Self::create_count_buffer(device, 2),
        ];
        let environment_stub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Environment Stub Buffer"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffers,
            count_buffers,
            environment_stub,
            frame_index: 0,
            frame_counter: 0,
            sequence_seed: r2_sample(0),
            light_count: 0,
            bind_group: None,
            bind_group_layout,
        }
    }

    /// Create a single light storage buffer
    fn create_light_buffer(device: &Device, index: usize) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Light Storage Buffer {}", index)),
            size: (MAX_LIGHTS * std::mem::size_of::<Light>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a single count uniform buffer
    fn create_count_buffer(device: &Device, index: usize) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Light Count Buffer {}", index)),
            size: 16,  // Single u32 with padding to 16 bytes (uniform buffer alignment)
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Update lights for the current frame
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer upload
    /// * `lights` - Slice of lights to upload (max MAX_LIGHTS)
    ///
    /// # Returns
    /// Result indicating success or error if too many lights
    pub fn update(&mut self, device: &Device, queue: &Queue, lights: &[Light]) -> Result<(), String> {
        if lights.len() > MAX_LIGHTS {
            return Err(format!(
                "Too many lights: {} (max {})",
                lights.len(),
                MAX_LIGHTS
            ));
        }

        // Get current buffer
        let buffer = &self.buffers[self.frame_index];
        let count_buffer = &self.count_buffers[self.frame_index];

        // Upload light data
        if !lights.is_empty() {
            let light_bytes = bytemuck::cast_slice(lights);
            queue.write_buffer(buffer, 0, light_bytes);
        }

        // Upload metadata: light count, frame counter (lower 32 bits), and R2 seed encoded as bits
        let seed = self.sequence_seed;
        let count_data = [
            lights.len() as u32,
            (self.frame_counter & 0xFFFF_FFFF) as u32,
            seed[0].to_bits(),
            seed[1].to_bits(),
        ];
        queue.write_buffer(count_buffer, 0, bytemuck::cast_slice(&count_data));
        queue.write_buffer(&self.environment_stub, 0, &[0u8; 16]);

        self.light_count = lights.len() as u32;

        // Recreate bind group for current frame
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Light Bind Group Frame {}", self.frame_index)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.environment_stub.as_entire_binding(),
                },
            ],
        }));

        Ok(())
    }

    /// Advance to next frame (call once per frame)
    pub fn next_frame(&mut self) {
        self.frame_index = (self.frame_index + 1) % 3;
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.sequence_seed = r2_sample(self.frame_counter);
    }

    /// Get bind group for current frame
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    /// Get bind group layout
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get current light count
    pub fn light_count(&self) -> u32 {
        self.light_count
    }

    /// Calculate memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        let light_buffer_size = (MAX_LIGHTS * std::mem::size_of::<Light>()) as u64;
        let count_buffer_size = 16u64;

        // 3 buffers × (light_buffer + count_buffer)
        3 * (light_buffer_size + count_buffer_size) + 16
    }

    /// Calculate memory usage in megabytes
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Return the current frame's R2 sequence seed for TAA-friendly light sampling
    pub fn sequence_seed(&self) -> [f32; 2] {
        self.sequence_seed
    }

    /// Expose the monotonic frame counter (useful for debugging)
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_buffer_memory() {
        // Light struct is 80 bytes (verified in types.rs)
        let light_size = std::mem::size_of::<Light>();
        assert_eq!(light_size, 80);

        // Memory calculation
        let light_buffer_size = MAX_LIGHTS * light_size;  // 16 * 80 = 1280 bytes
        let count_buffer_size = 16;
        let total_per_buffer = light_buffer_size + count_buffer_size;  // 1296 bytes
        let total = 3 * total_per_buffer;  // 3888 bytes

        assert_eq!(total, 3888);

        let mb = total as f64 / (1024.0 * 1024.0);
        assert!(mb < 0.01);  // Less than 10 KB = ~0.0037 MiB
    }

    #[test]
    fn test_max_lights_constant() {
        // Verify MAX_LIGHTS fits in memory budget
        // At 80 bytes per light, 16 lights = 1.28 KB per buffer
        // Triple-buffered: 3.84 KB total
        // This is well within the 512 MiB host-visible budget
        assert_eq!(MAX_LIGHTS, 16);

        let total_bytes = 3 * MAX_LIGHTS * std::mem::size_of::<Light>();
        assert!(total_bytes < 512 * 1024 * 1024);  // < 512 MiB
    }

    #[test]
    fn test_r2_sequence_variation() {
        let first = super::r2_sample(0);
        let second = super::r2_sample(1);
        assert_ne!(first, second);
        assert!(first[0] >= 0.0 && first[0] <= 1.0);
        assert!(second[1] >= 0.0 && second[1] <= 1.0);
    }
}

fn r2_sample(index: u64) -> [f32; 2] {
    const PHI: f64 = 1.324_717_957_244_746;
    const A1: f64 = 1.0 / PHI;
    const A2: f64 = 1.0 / (PHI * PHI);
    let idx = index as f64;
    [
        frac(0.5 + A1 * idx) as f32,
        frac(0.5 + A2 * idx) as f32,
    ]
}

fn frac(x: f64) -> f64 {
    x - x.floor()
}
