use wgpu::{BindGroup, Device};

use super::QueueBuffers;

impl QueueBuffers {
    pub fn create_raygen_bind_group(
        &self,
        device: &Device,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raygen-queue-layout"),
            entries: &storage_entries(&[0, 1]),
        });

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }))
    }

    pub fn create_shadow_bind_group(
        &self,
        device: &Device,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-queue-layout"),
            entries: &storage_entries(&[0, 1]),
        });

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-queue-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.shadow_queue_header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.shadow_queue.as_entire_binding(),
                },
            ],
        }))
    }
}

fn storage_entries(bindings: &[u32]) -> Vec<wgpu::BindGroupLayoutEntry> {
    bindings
        .iter()
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding: *binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
