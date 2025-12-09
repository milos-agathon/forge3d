// src/vector/gpu_extrusion.rs
// GPU polygon extrusion compute pipeline producing prism meshes
// Exists to accelerate F1 extrusion workloads and stay in sync with CPU reference implementation
// RELEVANT FILES: src/vector/extrusion.rs, shaders/extrusion.wgsl, src/vector/api.rs, docs/api/polygon_extrusion.md

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use wgpu::util::DeviceExt;

use crate::vector::extrusion::{tessellate_polygon, TessellatedPolygon};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolygonMeta {
    base_vertex_offset: u32,
    base_vertex_count: u32,
    base_index_offset: u32,
    base_index_count: u32,
    ring_offset: u32,
    ring_count: u32,
    output_vertex_offset: u32,
    output_index_offset: u32,
    bbox_min: [f32; 2],
    bbox_scale: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RingVertexPacked {
    position: [f32; 2],
    u_coord: f32,
    _pad: f32,
}

/// Buffers returned from a GPU extrusion dispatch.
pub struct GpuExtrusionOutput {
    pub positions: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub uvs: wgpu::Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
}

pub struct GpuExtrusion {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuExtrusion {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf.Vector.Extrusion.Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/extrusion.wgsl"
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

struct PolygonBuffers {
    metas: Vec<PolygonMeta>,
    base_vertices: Vec<[f32; 2]>,
    base_indices: Vec<u32>,
    ring_vertices: Vec<RingVertexPacked>,
    vertex_count: u32,
    index_count: u32,
}

fn pack_tessellations(tessellated: &[TessellatedPolygon]) -> Result<PolygonBuffers, String> {
    let mut metas = Vec::with_capacity(tessellated.len());
    let mut base_vertices = Vec::new();
    let mut base_indices = Vec::new();
    let mut ring_vertices = Vec::new();

    let mut base_vertex_offset: u32 = 0;
    let mut base_index_offset: u32 = 0;
    let mut ring_offset: u32 = 0;
    let mut output_vertex_offset: u32 = 0;
    let mut output_index_offset: u32 = 0;

    for tess in tessellated {
        let base_v = u32::try_from(tess.base_vertices.len())
            .map_err(|_| "polygon has too many base vertices (u32 overflow)".to_string())?;
        let base_i = u32::try_from(tess.base_indices.len())
            .map_err(|_| "polygon has too many indices (u32 overflow)".to_string())?;
        let ring_count = u32::try_from(tess.ring.len())
            .map_err(|_| "polygon ring too large (u32 overflow)".to_string())?;

        let side_vertex_count = ring_count
            .checked_mul(4)
            .ok_or_else(|| "side vertex count overflow".to_string())?;
        let side_index_count = ring_count
            .checked_mul(6)
            .ok_or_else(|| "side index count overflow".to_string())?;

        base_vertices.extend(tess.base_vertices.iter().map(|v| [v.x, v.y]));
        base_indices.extend(&tess.base_indices);
        ring_vertices.extend(
            tess.ring
                .iter()
                .zip(&tess.ring_u)
                .map(|(pos, u)| RingVertexPacked {
                    position: [pos.x, pos.y],
                    u_coord: *u,
                    _pad: 0.0,
                }),
        );

        metas.push(PolygonMeta {
            base_vertex_offset,
            base_vertex_count: base_v,
            base_index_offset,
            base_index_count: base_i,
            ring_offset,
            ring_count,
            output_vertex_offset,
            output_index_offset,
            bbox_min: [tess.bbox_min.x, tess.bbox_min.y],
            bbox_scale: compute_bbox_scale(tess.bbox_size),
        });

        base_vertex_offset = base_vertex_offset
            .checked_add(base_v)
            .ok_or_else(|| "base vertex offset overflow".to_string())?;
        base_index_offset = base_index_offset
            .checked_add(base_i)
            .ok_or_else(|| "base index offset overflow".to_string())?;
        ring_offset = ring_offset
            .checked_add(ring_count)
            .ok_or_else(|| "ring offset overflow".to_string())?;
        output_vertex_offset = output_vertex_offset
            .checked_add(base_v * 2)
            .and_then(|val| val.checked_add(side_vertex_count))
            .ok_or_else(|| "vertex offset overflow".to_string())?;
        output_index_offset = output_index_offset
            .checked_add(base_i * 2)
            .and_then(|val| val.checked_add(side_index_count))
            .ok_or_else(|| "index offset overflow".to_string())?;
    }

    Ok(PolygonBuffers {
        metas,
        base_vertices,
        base_indices,
        ring_vertices,
        vertex_count: output_vertex_offset,
        index_count: output_index_offset,
    })
}

fn compute_bbox_scale(size: Vec2) -> [f32; 2] {
    let x = if size.x.abs() > crate::vector::extrusion::EPSILON {
        1.0 / size.x
    } else {
        0.0
    };
    let y = if size.y.abs() > crate::vector::extrusion::EPSILON {
        1.0 / size.y
    } else {
        0.0
    };
    [x, y]
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::extrusion::extrude_polygon;
    use futures_intrusive::channel::shared::oneshot_channel;
    use glam::Vec2;

    #[test]
    fn gpu_matches_cpu_for_square() {
        let polygon = vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let height = 2.0;
        let (cpu_positions, cpu_indices, cpu_normals, cpu_uvs) = extrude_polygon(&polygon, height);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = match pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Some(adapter) => adapter,
            None => return,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .unwrap();

        let gpu = GpuExtrusion::new(&device);
        let output = gpu
            .extrude(&device, &queue, &[polygon.clone()], height)
            .expect("gpu extrusion failed");

        let vertex_count = output.vertex_count as usize;
        let index_count = output.index_count as usize;

        let position_slice = output.positions.slice(0..(vertex_count * 16) as u64);
        let normal_slice = output.normals.slice(0..(vertex_count * 16) as u64);
        let uv_slice = output.uvs.slice(0..(vertex_count * 8) as u64);
        let index_slice = output.indices.slice(0..(index_count * 4) as u64);

        let (pos_sender, pos_receiver) = oneshot_channel();
        position_slice.map_async(wgpu::MapMode::Read, move |result| {
            pos_sender.send(result).ok();
        });
        let (norm_sender, norm_receiver) = oneshot_channel();
        normal_slice.map_async(wgpu::MapMode::Read, move |result| {
            norm_sender.send(result).ok();
        });
        let (uv_sender, uv_receiver) = oneshot_channel();
        uv_slice.map_async(wgpu::MapMode::Read, move |result| {
            uv_sender.send(result).ok();
        });
        let (index_sender, index_receiver) = oneshot_channel();
        index_slice.map_async(wgpu::MapMode::Read, move |result| {
            index_sender.send(result).ok();
        });

        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(pos_receiver.receive()).unwrap().unwrap();
        pollster::block_on(norm_receiver.receive())
            .unwrap()
            .unwrap();
        pollster::block_on(uv_receiver.receive()).unwrap().unwrap();
        pollster::block_on(index_receiver.receive())
            .unwrap()
            .unwrap();

        let position_view = position_slice.get_mapped_range();
        let normal_view = normal_slice.get_mapped_range();
        let uv_view = uv_slice.get_mapped_range();
        let index_view = index_slice.get_mapped_range();

        let mut gpu_positions = Vec::with_capacity(vertex_count * 3);
        for chunk in bytemuck::cast_slice::<u8, f32>(&position_view).chunks_exact(4) {
            gpu_positions.extend_from_slice(&chunk[..3]);
        }
        let mut gpu_normals = Vec::with_capacity(vertex_count * 3);
        for chunk in bytemuck::cast_slice::<u8, f32>(&normal_view).chunks_exact(4) {
            gpu_normals.extend_from_slice(&chunk[..3]);
        }
        let gpu_uvs = bytemuck::cast_slice::<u8, f32>(&uv_view).to_vec();
        let gpu_indices = bytemuck::cast_slice::<u8, u32>(&index_view).to_vec();

        drop(position_view);
        drop(normal_view);
        drop(uv_view);
        drop(index_view);
        output.positions.unmap();
        output.normals.unmap();
        output.uvs.unmap();
        output.indices.unmap();

        let cpu_positions_flat: Vec<f32> =
            cpu_positions.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let cpu_normals_flat: Vec<f32> = cpu_normals.iter().flat_map(|n| [n.x, n.y, n.z]).collect();
        let cpu_uvs_flat: Vec<f32> = cpu_uvs.iter().flat_map(|uv| [uv.x, uv.y]).collect();

        assert_eq!(gpu_indices, cpu_indices);
        assert_eq!(gpu_positions.len(), cpu_positions_flat.len());
        assert_eq!(gpu_normals.len(), cpu_normals_flat.len());
        assert_eq!(gpu_uvs.len(), cpu_uvs_flat.len());

        for (gpu, cpu) in gpu_positions.iter().zip(cpu_positions_flat.iter()) {
            assert!((gpu - cpu).abs() < 1e-4);
        }
        for (gpu, cpu) in gpu_normals.iter().zip(cpu_normals_flat.iter()) {
            assert!((gpu - cpu).abs() < 1e-4);
        }
        for (gpu, cpu) in gpu_uvs.iter().zip(cpu_uvs_flat.iter()) {
            assert!((gpu - cpu).abs() < 1e-4);
        }
    }
}
