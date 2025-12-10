use super::types::{LightBuffer, MAX_LIGHTS};
use crate::lighting::types::Light;
use wgpu::{Device, Buffer, BufferUsages};
use super::r2::r2_sample;

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
        // P1-05: Environment params stub (zeros for now, full IBL in P4)
        let environment_stub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Environment Stub Buffer"),
            size: 16, // vec4<f32>
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize count buffer with zero lights for default bind group
        let seed = r2_sample(0);
        let _count_data = [
            0u32, // light_count = 0
            0u32, // frame_index = 0
            seed[0].to_bits(),
            seed[1].to_bits(),
        ];
        // Initialize the first count buffer with zero lights
        // Note: We'll use a queue to write, but for initialization we can use write_buffer
        // However, we need a queue. For now, create the default bind group after initialization.
        // Actually, we can create it here since buffers are already created.

        // Create default bind group with zero lights (neutral/empty state)
        let default_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group Default (Zero Lights)"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: environment_stub.as_entire_binding(),
                },
            ],
        });

        Self {
            buffers,
            count_buffers,
            environment_stub,
            frame_index: 0,
            frame_counter: 0,
            sequence_seed: r2_sample(0),
            light_count: 0,
            bind_group: Some(default_bind_group),
            bind_group_layout,
            last_uploaded_lights: Vec::new(),
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
            size: 16, // Single u32 with padding to 16 bytes (uniform buffer alignment)
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
