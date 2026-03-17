use crate::vector::extrusion::extrude_polygon;
use crate::vector::gpu_extrusion::GpuExtrusion;
use futures_intrusive::channel::shared::oneshot_channel;
use glam::Vec2;
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(text_signature = "(polygons, height)")]
pub fn extrude_polygon_gpu_py<'py>(
    py: Python<'py>,
    polygons: Vec<PyReadonlyArray2<'py, f32>>,
    height: f32,
) -> PyResult<(
    Py<PyArray1<f32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<f32>>,
    Py<PyArray1<f32>>,
)> {
    if polygons.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polygons must not be empty",
        ));
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Failed to find an appropriate adapter")
            })?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create device: {}", e))
            })?;

    let polygons_vec: Vec<Vec<Vec2>> = polygons
        .iter()
        .map(|polygon| {
            polygon
                .as_array()
                .outer_iter()
                .map(|row| Vec2::new(row[0], row[1]))
                .collect()
        })
        .collect();

    let gpu_extrusion = GpuExtrusion::new(&device);
    let output = gpu_extrusion
        .extrude(&device, &queue, &polygons_vec, height)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let vertex_count = output.vertex_count as usize;
    let index_count = output.index_count as usize;

    if vertex_count == 0 || index_count == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Extrusion produced empty mesh",
        ));
    }

    let vertex_bytes = vertex_count * 16;
    let normal_bytes = vertex_count * 16;
    let uv_bytes = vertex_count * 8;
    let index_bytes = index_count * 4;

    let positions = output.positions;
    let indices = output.indices;
    let normals = output.normals;
    let uvs = output.uvs;

    let position_slice = positions.slice(0..vertex_bytes as u64);
    let index_slice = indices.slice(0..index_bytes as u64);
    let normal_slice = normals.slice(0..normal_bytes as u64);
    let uv_slice = uvs.slice(0..uv_bytes as u64);

    let (pos_sender, pos_receiver) = oneshot_channel();
    position_slice.map_async(wgpu::MapMode::Read, move |result| {
        pos_sender.send(result).ok();
    });
    let (idx_sender, idx_receiver) = oneshot_channel();
    index_slice.map_async(wgpu::MapMode::Read, move |result| {
        idx_sender.send(result).ok();
    });
    let (norm_sender, norm_receiver) = oneshot_channel();
    normal_slice.map_async(wgpu::MapMode::Read, move |result| {
        norm_sender.send(result).ok();
    });
    let (uv_sender, uv_receiver) = oneshot_channel();
    uv_slice.map_async(wgpu::MapMode::Read, move |result| {
        uv_sender.send(result).ok();
    });

    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(pos_receiver.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Vertex map cancelled"))?
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to map vertices buffer: {}",
                e
            ))
        })?;
    pollster::block_on(idx_receiver.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index map cancelled"))?
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to map indices buffer: {}",
                e
            ))
        })?;
    pollster::block_on(norm_receiver.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Normal map cancelled"))?
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to map normals buffer: {}",
                e
            ))
        })?;
    pollster::block_on(uv_receiver.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("UV map cancelled"))?
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to map uvs buffer: {}", e))
        })?;

    let position_view = position_slice.get_mapped_range();
    let index_view = index_slice.get_mapped_range();
    let normal_view = normal_slice.get_mapped_range();
    let uv_view = uv_slice.get_mapped_range();

    let position_floats = bytemuck::cast_slice::<u8, f32>(&position_view);
    let normal_floats = bytemuck::cast_slice::<u8, f32>(&normal_view);
    let uv_floats = bytemuck::cast_slice::<u8, f32>(&uv_view);
    let index_values = bytemuck::cast_slice::<u8, u32>(&index_view);

    let mut vertices_flat = Vec::with_capacity(vertex_count * 3);
    for chunk in position_floats.chunks_exact(4) {
        vertices_flat.extend_from_slice(&chunk[..3]);
    }

    let mut normals_flat = Vec::with_capacity(vertex_count * 3);
    for chunk in normal_floats.chunks_exact(4) {
        normals_flat.extend_from_slice(&chunk[..3]);
    }

    let uvs_flat = uv_floats.to_vec();
    let indices_flat = index_values.to_vec();

    drop(position_view);
    drop(normal_view);
    drop(uv_view);
    drop(index_view);
    positions.unmap();
    normals.unmap();
    uvs.unmap();
    indices.unmap();

    let vertices_py = PyArray1::from_vec_bound(py, vertices_flat).into();
    let indices_py = PyArray1::from_vec_bound(py, indices_flat).into();
    let normals_py = PyArray1::from_vec_bound(py, normals_flat).into();
    let uvs_py = PyArray1::from_vec_bound(py, uvs_flat).into();

    Ok((vertices_py, indices_py, normals_py, uvs_py))
}

#[pyfunction]
#[pyo3(text_signature = "(polygon, height)")]
pub fn extrude_polygon_py<'py>(
    py: Python<'py>,
    polygon: PyReadonlyArray2<'py, f32>,
    height: f32,
) -> PyResult<(
    Py<PyArray1<f32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<f32>>,
    Py<PyArray1<f32>>,
)> {
    let polygon_vec: Vec<Vec2> = polygon
        .as_array()
        .outer_iter()
        .map(|row| Vec2::new(row[0], row[1]))
        .collect();

    let (vertices, indices, normals, uvs) = extrude_polygon(&polygon_vec, height);

    let vertices_py = vertices
        .iter()
        .flat_map(|v| [v.x, v.y, v.z])
        .collect::<Vec<f32>>()
        .to_pyarray_bound(py)
        .into();
    let indices_py = indices.to_pyarray_bound(py).into();
    let normals_py = normals
        .iter()
        .flat_map(|n| [n.x, n.y, n.z])
        .collect::<Vec<f32>>()
        .to_pyarray_bound(py)
        .into();
    let uvs_py = uvs
        .iter()
        .flat_map(|uv| [uv.x, uv.y])
        .collect::<Vec<f32>>()
        .to_pyarray_bound(py)
        .into();

    Ok((vertices_py, indices_py, normals_py, uvs_py))
}
