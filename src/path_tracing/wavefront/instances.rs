use super::*;

impl WavefrontScheduler {
    pub fn upload_instances(&mut self, transforms: &[Mat4]) {
        if transforms.is_empty() {
            return;
        }
        let mut inst: Vec<crate::accel::instancing::InstanceData> =
            Vec::with_capacity(transforms.len());
        for m in transforms {
            let inv = m.inverse();
            inst.push(crate::accel::instancing::InstanceData {
                transform: m.to_cols_array(),
                inv_transform: inv.to_cols_array(),
                blas_index: 0,
                material_id: 0,
                _padding: [0; 2],
            });
        }
        self.upload_instances_data(&inst);
    }

    pub fn upload_instances_with_meta(&mut self, items: &[(Mat4, u32, u32)]) {
        if items.is_empty() {
            return;
        }
        let mut inst: Vec<crate::accel::instancing::InstanceData> = Vec::with_capacity(items.len());
        for (m, blas_index, material_id) in items.iter().copied() {
            let inv = m.inverse();
            inst.push(crate::accel::instancing::InstanceData {
                transform: m.to_cols_array(),
                inv_transform: inv.to_cols_array(),
                blas_index,
                material_id,
                _padding: [0; 2],
            });
        }
        self.upload_instances_data(&inst);
    }

    pub fn upload_instances_data(&mut self, instances: &[crate::accel::instancing::InstanceData]) {
        if instances.is_empty() {
            return;
        }
        self.instances_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("instances-buffer"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
    }
}
