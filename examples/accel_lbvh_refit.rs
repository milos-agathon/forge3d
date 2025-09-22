// examples/accel_lbvh_refit.rs
// Minimal GPU LBVH build + refit demo: builds a tiny BVH on GPU, then perturbs a triangle
// and refits AABBs. Prints world AABB before/after for test assertions.

use anyhow::Result;
use std::sync::Arc;
use wgpu::{Instance, InstanceDescriptor, RequestAdapterOptions};

use forge3d::accel::types::{BuildOptions, BvhNode, Triangle};
use forge3d::accel::{lbvh_gpu::GpuBvhBuilder, BvhBackend};

fn fmt_aabb(min: [f32; 3], max: [f32; 3]) -> String {
    format!(
        "min=[{:.3},{:.3},{:.3}], max=[{:.3},{:.3},{:.3}]",
        min[0], min[1], min[2], max[0], max[1], max[2]
    )
}

fn main() -> Result<()> {
    env_logger::init();

    // Create wgpu device/queue
    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Tiny triangle scene: two triangles making a square
    let mut tris = vec![
        Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        Triangle::new([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
    ];

    let mut builder = GpuBvhBuilder::new(device.clone(), queue.clone())?;
    let opts = BuildOptions::default();

    let mut handle = builder.build(&tris, &opts)?;
    println!(
        "AABB before: {}",
        fmt_aabb(handle.world_aabb.min, handle.world_aabb.max)
    );

    // Dump nodes/indices after build for tests
    if let BvhBackend::Gpu(gpu) = &handle.backend {
        // Read back nodes buffer
        let nodes_size = (handle.node_count as usize * std::mem::size_of::<BvhNode>()) as u64;
        let rb_nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb-nodes"),
            size: nodes_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Read back indices buffer
        let indices_size = (handle.triangle_count as usize * std::mem::size_of::<u32>()) as u64;
        let rb_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb-indices"),
            size: indices_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dump-enc"),
        });
        enc.copy_buffer_to_buffer(&gpu.nodes_buffer, 0, &rb_nodes, 0, nodes_size);
        enc.copy_buffer_to_buffer(&gpu.indices_buffer, 0, &rb_indices, 0, indices_size);
        queue.submit(Some(enc.finish()));
        let sl_n = rb_nodes.slice(..);
        let sl_i = rb_indices.slice(..);
        sl_n.map_async(wgpu::MapMode::Read, |_| {});
        sl_i.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data_nodes = sl_n.get_mapped_range();
        let data_indices = sl_i.get_mapped_range();
        std::fs::create_dir_all("out").ok();
        std::fs::write("out/lbvh_nodes.bin", &data_nodes)?;
        std::fs::write("out/lbvh_indices.bin", &data_indices)?;
        drop(data_nodes);
        drop(data_indices);
        rb_nodes.unmap();
        rb_indices.unmap();
        println!(
            "Dumped: out/lbvh_nodes.bin ({} bytes), out/lbvh_indices.bin ({} bytes)",
            nodes_size, indices_size
        );
    }

    // Perturb a vertex to expand world AABB
    tris[0].v2 = [2.0, 2.0, 0.0];

    builder.refit(&mut handle, &tris)?;
    println!(
        "AABB after: {}",
        fmt_aabb(handle.world_aabb.min, handle.world_aabb.max)
    );

    Ok(())
}
