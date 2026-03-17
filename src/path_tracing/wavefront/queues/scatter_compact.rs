use wgpu::{BindGroup, Device};

use super::QueueBuffers;

impl QueueBuffers {
    pub fn create_scatter_bind_group(
        &self,
        device: &Device,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter-queue-layout"),
            entries: &storage_entries(&[0, 1, 2, 3, 4, 5]),
        });

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter-queue-bind-group"),
            layout: &layout,
            entries: &[
                bind_entry(0, &self.scatter_queue_header),
                bind_entry(1, &self.scatter_queue),
                bind_entry(2, &self.ray_queue_header),
                bind_entry(3, &self.ray_queue),
                bind_entry(4, &self.miss_queue_header),
                bind_entry(5, &self.miss_queue),
            ],
        }))
    }

    pub fn create_compact_bind_group(
        &self,
        device: &Device,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact-queue-layout"),
            entries: &storage_entries(&[0, 1, 2, 3, 4]),
        });

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact-queue-bind-group"),
            layout: &layout,
            entries: &[
                bind_entry(0, &self.ray_queue_header),
                bind_entry(1, &self.ray_queue),
                bind_entry(2, &self.ray_queue_compacted),
                bind_entry(3, &self.ray_flags),
                bind_entry(4, &self.prefix_sums),
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

fn bind_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
