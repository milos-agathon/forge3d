use super::{types::CullingUniforms, CullableInstance, IndirectDrawCommand, IndirectRenderer};
use crate::core::error::RenderError;
use wgpu::util::DeviceExt;

impl IndirectRenderer {
    pub fn new(device: &wgpu::Device) -> Result<Self, RenderError> {
        let initial_capacity = 4096;

        let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.DrawCommands"),
            size: (initial_capacity * std::mem::size_of::<IndirectDrawCommand>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.Instances"),
            size: (initial_capacity * std::mem::size_of::<CullableInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let culling_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.CullingUniforms"),
            size: std::mem::size_of::<CullingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.Counters"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Indirect.Readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf.Vector.Indirect.CullingCompute"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../shaders/culling_compute.wgsl"
            ))),
        });

        let culling_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Indirect.CullingBindGroupLayout"),
            entries: &[
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
        };

        let bind_group_layout = device.create_bind_group_layout(&culling_bind_group_layout);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Indirect.CullingPipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

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
            cpu_culling_enabled: true,
        })
    }

    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        instances: &[CullableInstance],
    ) -> Result<(), RenderError> {
        if instances.is_empty() {
            return Ok(());
        }

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

        Ok(())
    }
}
