use super::pipeline::GpuExtrusion;
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
    let adapter =
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
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

    let cpu_positions_flat: Vec<f32> = cpu_positions.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
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
