use super::{
    CullableInstance, CullingStats, CullingUniforms, IndirectDrawCommand, IndirectRenderer,
};
use crate::core::error::RenderError;
use crate::core::gpu_timing::GpuTimingManager;
use crate::vector::batch::Frustum;
use glam::{Mat4, Vec3};

impl IndirectRenderer {
    pub fn cull_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        camera_pos: Vec3,
        frustum: &Frustum,
        instance_count: u32,
    ) -> Result<(), RenderError> {
        self.cull_gpu_with_timing(
            device,
            queue,
            view_proj,
            camera_pos,
            frustum,
            instance_count,
            None,
        )
    }

    pub fn cull_gpu_with_timing(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        camera_pos: Vec3,
        frustum: &Frustum,
        instance_count: u32,
        mut timing_manager: Option<&mut GpuTimingManager>,
    ) -> Result<(), RenderError> {
        let mut frustum_planes = [[0.0f32; 4]; 6];
        for (i, plane) in frustum.planes.iter().enumerate() {
            frustum_planes[i] = [plane.x, plane.y, plane.z, plane.w];
        }
        frustum_planes[4] = [0.0, 0.0, 1.0, 1000.0];
        frustum_planes[5] = [0.0, 0.0, -1.0, 0.1];

        let uniforms = CullingUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            frustum_plane_0: frustum_planes[0],
            frustum_plane_1: frustum_planes[1],
            frustum_plane_2: frustum_planes[2],
            frustum_plane_3: frustum_planes[3],
            frustum_plane_4: frustum_planes[4],
            frustum_plane_5: frustum_planes[5],
            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z],
            _pad0: 0.0,
            cull_distance: 1000.0,
            enable_frustum_cull: 1,
            enable_distance_cull: 1,
            enable_occlusion_cull: 0,
        };

        queue.write_buffer(
            &self.culling_uniforms_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        let zero_counters: [u32; 4] = [0; 4];
        queue.write_buffer(
            &self.counter_buffer,
            0,
            bytemuck::cast_slice(&zero_counters),
        );

        let bind_group_layout = device.create_bind_group_layout(&self.culling_bind_group_layout);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Indirect.CullingBindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.culling_uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.draw_commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.counter_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Indirect.CullingDispatch"),
        });

        let timing_scope = if let Some(timer) = timing_manager.as_mut() {
            Some(timer.begin_scope(&mut encoder, "vector_indirect_culling"))
        } else {
            None
        };

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf.Vector.Indirect.CullingPass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.culling_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 64;
            let workgroups = (instance_count + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.counter_buffer, 0, &self.readback_buffer, 0, 16);

        if let (Some(timer), Some(scope_id)) = (timing_manager, timing_scope) {
            timer.end_scope(&mut encoder, scope_id);
        }

        queue.submit(Some(encoder.finish()));

        Ok(())
    }

    pub fn cull_cpu(
        &self,
        instances: &[CullableInstance],
        frustum: &Frustum,
        camera_pos: Vec3,
        max_distance: f32,
    ) -> Vec<IndirectDrawCommand> {
        let mut visible_commands = Vec::new();

        for (i, instance) in instances.iter().enumerate() {
            let transform = Mat4::from_cols_array_2d(&instance.transform);
            let aabb_min = Vec3::from(instance.aabb_min);
            let aabb_max = Vec3::from(instance.aabb_max);
            let world_center = transform.transform_point3((aabb_min + aabb_max) * 0.5);

            let distance = (world_center - camera_pos).length();
            if distance > max_distance {
                continue;
            }

            let radius = (aabb_max - aabb_min).length() * 0.5;
            let mut inside_frustum = true;

            for plane in &frustum.planes {
                let distance_to_plane = plane.dot(world_center.extend(1.0));
                if distance_to_plane < -radius {
                    inside_frustum = false;
                    break;
                }
            }

            if inside_frustum {
                visible_commands.push(IndirectDrawCommand {
                    vertex_count: self.get_vertex_count_for_type(instance.primitive_type),
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: i as u32,
                });
            }
        }

        visible_commands
    }

    pub fn read_culling_stats(&self, device: &wgpu::Device) -> Result<CullingStats, RenderError> {
        device.poll(wgpu::Maintain::Wait);

        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().map_err(|e| {
            RenderError::Readback(format!("Failed to map culling stats buffer: {:?}", e))
        })?;

        let data = buffer_slice.get_mapped_range();
        let counters: &[u32; 4] = bytemuck::from_bytes(&data[..16]);

        let stats = CullingStats {
            total_objects: counters[0],
            visible_objects: counters[1],
            frustum_culled: counters[2],
            distance_culled: counters[3],
            occlusion_culled: 0,
            gpu_time_ms: 0.0,
        };

        drop(data);
        self.readback_buffer.unmap();

        Ok(stats)
    }

    pub(super) fn get_vertex_count_for_type(&self, primitive_type: u32) -> u32 {
        match primitive_type {
            0 => 3,
            1 => 4,
            2 => 1,
            3 => 2,
            _ => 3,
        }
    }

    pub fn set_cpu_culling(&mut self, enabled: bool) {
        self.cpu_culling_enabled = enabled;
    }

    pub fn is_cpu_culling_enabled(&self) -> bool {
        self.cpu_culling_enabled
    }
}
