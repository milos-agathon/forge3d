//! H17,H19: Indirect drawing and GPU culling
//! CPU and GPU paths for efficient large-scale vector rendering

use crate::core::gpu_timing::GpuTimingManager;
use crate::core::error::RenderError;
use crate::vector::batch::{Frustum, PrimitiveType, AABB};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Indirect draw command for GPU execution
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndirectDrawCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Indexed indirect draw command
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndirectDrawIndexedCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

/// Object instance data for culling
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CullableInstance {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub transform: [[f32; 4]; 4], // World transform matrix
    pub primitive_type: u32,      // PrimitiveType as u32
    pub draw_command_index: u32,  // Index into draw command buffer
}

/// GPU culling uniforms
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CullingUniforms {
    view_proj: [[f32; 4]; 4],      // View-projection matrix
    frustum_planes: [[f32; 4]; 6], // Frustum planes (ax + by + cz + d = 0)
    camera_position: [f32; 3],     // Camera world position
    _pad0: f32,
    cull_distance: f32,       // Maximum culling distance
    enable_frustum_cull: u32, // Boolean flags
    enable_distance_cull: u32,
    enable_occlusion_cull: u32,
}

/// Indirect drawing and GPU culling manager
pub struct IndirectRenderer {
    // Draw command buffers
    draw_commands_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    draw_commands_capacity: usize,

    // Instance data for culling
    instances_buffer: wgpu::Buffer,
    instances_capacity: usize,

    // Culling compute pipeline
    culling_pipeline: wgpu::ComputePipeline,
    culling_bind_group_layout: wgpu::BindGroupLayoutDescriptor<'static>,

    // Uniforms
    culling_uniforms_buffer: wgpu::Buffer,

    // Counters and results
    counter_buffer: wgpu::Buffer,  // Draw count results
    readback_buffer: wgpu::Buffer, // CPU readback for statistics

    // CPU fallback path
    cpu_culling_enabled: bool,
}

/// Culling statistics
#[derive(Debug, Clone, Default)]
pub struct CullingStats {
    pub total_objects: u32,
    pub visible_objects: u32,
    pub frustum_culled: u32,
    pub distance_culled: u32,
    pub occlusion_culled: u32,
    pub gpu_time_ms: f32,
}

impl IndirectRenderer {
    pub fn new(device: &wgpu::Device) -> Result<Self, RenderError> {
        let initial_capacity = 4096;

        // Create draw commands buffer
        let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.DrawCommands"),
            size: (initial_capacity * std::mem::size_of::<IndirectDrawCommand>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create instances buffer
        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.Instances"),
            size: (initial_capacity * std::mem::size_of::<CullableInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create culling uniforms buffer
        let culling_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.CullingUniforms"),
            size: std::mem::size_of::<CullingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create counter buffer for results
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.Counters"),
            size: 16, // 4 x u32 counters
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create readback buffer
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.Readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create compute shader for culling
        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf.Vector.Indirect.CullingCompute"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/culling_compute.wgsl"
            ))),
        });

        // Create bind group layout for culling
        let culling_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Indirect.CullingBindGroupLayout"),
            entries: &[
                // Culling uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input instances
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output draw commands
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
                // Counters
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
        };

        let bind_group_layout = device.create_bind_group_layout(&culling_bind_group_layout);

        // Create compute pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Indirect.CullingPipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create culling compute pipeline
        let culling_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vf.Vector.Indirect.CullingPipeline"),
            layout: Some(&pipeline_layout),
            module: &culling_shader,
            entry_point: "cs_main",
        });

        Ok(Self {
            draw_commands_buffer,
            draw_commands_capacity: initial_capacity,
            instances_buffer,
            instances_capacity: initial_capacity,
            culling_pipeline,
            culling_bind_group_layout,
            culling_uniforms_buffer,
            counter_buffer,
            readback_buffer,
            cpu_culling_enabled: true, // Enable CPU fallback by default
        })
    }

    /// Upload instance data for culling
    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        instances: &[CullableInstance],
    ) -> Result<(), RenderError> {
        if instances.is_empty() {
            return Ok(());
        }

        // Reallocate buffer if needed
        if instances.len() > self.instances_capacity {
            let new_capacity = (instances.len() * 2).max(1024);
            self.instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf.Vector.Indirect.Instances"),
                size: (new_capacity * std::mem::size_of::<CullableInstance>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances_capacity = new_capacity;
        }

        // Upload instance data
        let instance_data = bytemuck::cast_slice(instances);
        let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf.Vector.Indirect.InstancesStaging"),
            contents: instance_data,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Indirect.InstancesUpload"),
        });

        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &self.instances_buffer,
            0,
            instance_data.len() as u64,
        );

        // Submit upload command (Note: In production, this would be batched)

        Ok(())
    }

    /// Perform GPU culling and generate indirect draw commands
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

    /// Perform GPU culling with optional timing
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
        // Create culling uniforms
        let mut frustum_planes = [[0.0f32; 4]; 6];
        // Copy available planes (2D frustum has 4 planes, 3D needs 6)
        for (i, plane) in frustum.planes.iter().enumerate() {
            frustum_planes[i] = [plane.x, plane.y, plane.z, plane.w];
        }
        // Add default near/far planes for 2D->3D conversion
        frustum_planes[4] = [0.0, 0.0, 1.0, 1000.0]; // Near plane
        frustum_planes[5] = [0.0, 0.0, -1.0, 0.1]; // Far plane

        let uniforms = CullingUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            frustum_planes,
            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z],
            _pad0: 0.0,
            cull_distance: 1000.0, // Max culling distance
            enable_frustum_cull: 1,
            enable_distance_cull: 1,
            enable_occlusion_cull: 0, // Disabled for now
        };

        // Update uniforms buffer
        queue.write_buffer(
            &self.culling_uniforms_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Clear counters
        let zero_counters: [u32; 4] = [0; 4];
        queue.write_buffer(
            &self.counter_buffer,
            0,
            bytemuck::cast_slice(&zero_counters),
        );

        // Create bind group
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

        // Dispatch culling compute shader
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

            // Dispatch with 64 threads per workgroup
            let workgroup_size = 64;
            let workgroups = (instance_count + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy counters for readback
        encoder.copy_buffer_to_buffer(&self.counter_buffer, 0, &self.readback_buffer, 0, 16);

        // End GPU timing scope
        if let (Some(timer), Some(scope_id)) = (timing_manager, timing_scope) {
            timer.end_scope(&mut encoder, scope_id);
        }

        queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Perform CPU culling (fallback path)
    pub fn cull_cpu(
        &self,
        instances: &[CullableInstance],
        frustum: &Frustum,
        camera_pos: Vec3,
        max_distance: f32,
    ) -> Vec<IndirectDrawCommand> {
        let mut visible_commands = Vec::new();

        for (i, instance) in instances.iter().enumerate() {
            // Transform AABB to world space
            let transform = Mat4::from_cols_array_2d(&instance.transform);
            let aabb_min = Vec3::from(instance.aabb_min);
            let aabb_max = Vec3::from(instance.aabb_max);

            // AABB center in world space
            let world_center = transform.transform_point3((aabb_min + aabb_max) * 0.5);

            // Distance culling
            let distance = (world_center - camera_pos).length();
            if distance > max_distance {
                continue;
            }

            // Frustum culling (simplified sphere test)
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
                // Object is visible, add draw command
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

    /// Read back culling statistics from GPU
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
            occlusion_culled: 0, // Not implemented yet
            gpu_time_ms: 0.0,    // Would require timestamps
        };

        drop(data);
        self.readback_buffer.unmap();

        Ok(stats)
    }

    /// Execute indirect draw commands
    pub fn draw_indirect<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        draw_count: u32,
    ) {
        if draw_count > 0 {
            render_pass.draw_indirect(&self.draw_commands_buffer, 0);
        }
    }

    /// Get vertex count for primitive type
    fn get_vertex_count_for_type(&self, primitive_type: u32) -> u32 {
        match primitive_type {
            0 => 3, // Triangle
            1 => 4, // Quad/Rectangle
            2 => 1, // Point
            3 => 2, // Line segment
            _ => 3, // Default to triangle
        }
    }

    /// Enable or disable CPU culling fallback
    pub fn set_cpu_culling(&mut self, enabled: bool) {
        self.cpu_culling_enabled = enabled;
    }

    /// Check if CPU culling is enabled
    pub fn is_cpu_culling_enabled(&self) -> bool {
        self.cpu_culling_enabled
    }
}

/// Helper function to create cullable instance from AABB and transform
pub fn create_cullable_instance(
    aabb: &AABB,
    transform: Mat4,
    primitive_type: PrimitiveType,
    draw_command_index: u32,
) -> CullableInstance {
    CullableInstance {
        aabb_min: [aabb.min.x, aabb.min.y, 0.0],
        aabb_max: [aabb.max.x, aabb.max.y, 1.0],
        transform: transform.to_cols_array_2d(),
        primitive_type: primitive_type as u32,
        draw_command_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_indirect_draw_command_size() {
        // Ensure draw command structure matches GPU expectations
        assert_eq!(std::mem::size_of::<IndirectDrawCommand>(), 16);
        assert_eq!(std::mem::size_of::<IndirectDrawIndexedCommand>(), 20);
    }

    #[test]
    fn test_cullable_instance_creation() {
        let aabb = AABB {
            min: Vec2::new(-1.0, -1.0),
            max: Vec2::new(1.0, 1.0),
        };
        let transform = Mat4::IDENTITY;

        let instance = create_cullable_instance(&aabb, transform, PrimitiveType::Triangle, 0);

        assert_eq!(instance.aabb_min, [-1.0, -1.0, 0.0]);
        assert_eq!(instance.aabb_max, [1.0, 1.0, 1.0]);
        assert_eq!(instance.primitive_type, PrimitiveType::Triangle as u32);
    }

    #[test]
    fn test_vertex_count_for_types() {
        let renderer = IndirectRenderer::new(&crate::core::gpu::create_device_for_test()).unwrap();

        assert_eq!(renderer.get_vertex_count_for_type(0), 3); // Triangle
        assert_eq!(renderer.get_vertex_count_for_type(1), 4); // Quad
        assert_eq!(renderer.get_vertex_count_for_type(2), 1); // Point
        assert_eq!(renderer.get_vertex_count_for_type(3), 2); // Line
    }

    #[test]
    fn test_cpu_culling_distance() {
        let renderer = IndirectRenderer::new(&crate::core::gpu::create_device_for_test()).unwrap();

        let instance = CullableInstance {
            aabb_min: [-1.0, -1.0, -1.0],
            aabb_max: [1.0, 1.0, 1.0],
            transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -100.0)).to_cols_array_2d(),
            primitive_type: PrimitiveType::Triangle as u32,
            draw_command_index: 0,
        };

        let frustum = Frustum::from_view_proj(&Mat4::IDENTITY);
        let camera_pos = Vec3::ZERO;
        let max_distance = 50.0;

        let visible = renderer.cull_cpu(&[instance], &frustum, camera_pos, max_distance);
        assert_eq!(visible.len(), 0); // Should be culled by distance
    }
}
