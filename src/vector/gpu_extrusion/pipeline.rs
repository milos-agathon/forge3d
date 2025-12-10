use super::buffers::{pack_tessellations, PolygonBuffers};
use super::types::GpuExtrusionOutput;
use crate::vector::extrusion::{tessellate_polygon, TessellatedPolygon};
use glam::Vec2;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub struct GpuExtrusion {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuExtrusion {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf.Vector.Extrusion.Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../shaders/extrusion.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Extrusion.BindGroupLayout"),
            entries: &[
                // Metadata
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Base vertices
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
                // Base indices
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ring vertices + UVs
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output positions
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
                // Output indices
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
                // Output normals
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output UVs
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniform height parameter
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Extrusion.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vf.Vector.Extrusion.Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn extrude(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        polygons: &[Vec<Vec2>],
        height: f32,
    ) -> Result<GpuExtrusionOutput, String> {
        if polygons.is_empty() {
            return Err("no polygons provided".to_string());
        }

        let tessellated: Vec<TessellatedPolygon> = polygons
            .iter()
            .enumerate()
            .map(|(idx, polygon)| {
                tessellate_polygon(polygon).ok_or_else(|| {
                    format!(
                        "polygon {} failed tessellation (need >=3 valid vertices)",
                        idx
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        if tessellated.is_empty() {
            return Err("no valid polygons to extrude".to_string());
        }

        let packed = pack_tessellations(&tessellated)?;
        let PolygonBuffers {
            metas,
            base_vertices,
            base_indices,
            ring_vertices,
            vertex_count,
            index_count,
        } = packed;

        if vertex_count == 0 || index_count == 0 {
            return Err("tessellated polygon produced empty mesh".to_string());
        }

        let meta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf.Vector.Extrusion.Meta"),
            contents: bytemuck::cast_slice(&metas),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let base_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf.Vector.Extrusion.BaseVertices"),
            contents: bytemuck::cast_slice(&base_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let base_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf.Vector.Extrusion.BaseIndices"),
            contents: bytemuck::cast_slice(&base_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let ring_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf.Vector.Extrusion.RingVertices"),
            contents: bytemuck::cast_slice(&ring_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Extrusion.Output.Positions"),
            size: (vertex_count as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Extrusion.Output.Indices"),
            size: (index_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let normals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Extrusion.Output.Normals"),
            size: (vertex_count as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uvs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Extrusion.Output.UVs"),
            size: (vertex_count as u64) * 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let height_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf.Vector.Extrusion.Height"),
            contents: bytemuck::cast_slice(&[height]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Extrusion.BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: base_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: base_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ring_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: normals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: uvs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: height_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Extrusion.Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf.Vector.Extrusion.Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = ((metas.len() as u32) + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count.max(1), 1, 1);
        }

        queue.submit(Some(encoder.finish()));

        Ok(GpuExtrusionOutput {
            positions: positions_buffer,
            indices: indices_buffer,
            normals: normals_buffer,
            uvs: uvs_buffer,
            vertex_count,
            index_count,
        })
    }
}
