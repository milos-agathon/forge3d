//! Python boundary for the CPU BVH builder and GPU mesh upload.
//!
//! `forge3d.mesh.build_bvh_cpu` and `forge3d.mesh.upload_mesh` previously
//! returned hand-written estimates and never touched Rust or the GPU. These
//! bindings expose the real `accel::cpu_bvh` builder and the canonical
//! `path_tracing::mesh` atlas upload so the Python module reports and uploads
//! the mesh that was actually built.

use super::super::*;

use crate::accel::cpu_bvh::{
    build_bvh_cpu, BuildMethod, BuildOptions, BuildStats, BvhCPU, BvhNode, MeshCPU,
};
use crate::path_tracing::mesh::{build_mesh_atlas, MeshAtlas};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

/// Packed byte size of one CPU `BvhNode`. The struct is `#[repr(C)]` + `Pod`,
/// so its in-memory layout is the wire layout the Python layer round-trips.
const BVH_NODE_BYTES: usize = std::mem::size_of::<BvhNode>();

/// Extract a validated `MeshCPU` from the Python vertex/index arrays.
///
/// `make_mesh` already validates shapes, dtypes and index ranges; these checks
/// repeat them so the native call is safe on its own.
fn mesh_cpu_from_py(
    vertices: &PyReadonlyArray2<'_, f32>,
    indices: &PyReadonlyArray2<'_, u32>,
) -> PyResult<MeshCPU> {
    let v = vertices.as_array();
    let i = indices.as_array();
    if v.shape()[0] == 0 {
        return Err(PyValueError::new_err("vertices must not be empty"));
    }
    if i.shape()[0] == 0 {
        return Err(PyValueError::new_err(
            "indices must contain at least one triangle",
        ));
    }

    let mut verts = Vec::with_capacity(v.shape()[0]);
    for row in v.rows() {
        let point = [row[0], row[1], row[2]];
        if !point.iter().all(|value| value.is_finite()) {
            return Err(PyValueError::new_err(
                "vertices must contain only finite coordinates",
            ));
        }
        verts.push(point);
    }

    let vertex_count = verts.len();
    let mut triangles = Vec::with_capacity(i.shape()[0]);
    for row in i.rows() {
        let triangle = [row[0], row[1], row[2]];
        for &index in &triangle {
            if index as usize >= vertex_count {
                return Err(PyValueError::new_err(
                    "indices must reference vertices inside the vertices array",
                ));
            }
        }
        triangles.push(triangle);
    }

    Ok(MeshCPU::new(verts, triangles))
}

/// Rebuild a `BvhCPU` from the packed node bytes and triangle order emitted by
/// `mesh_build_bvh_cpu`, rejecting any structure that is not a single connected,
/// in-range tree over `mesh`.
fn bvh_cpu_from_py(
    nodes: &PyReadonlyArray1<'_, u8>,
    tri_indices: &PyReadonlyArray1<'_, u32>,
    mesh: &MeshCPU,
) -> PyResult<BvhCPU> {
    let node_bytes = nodes.as_slice()?;
    if node_bytes.is_empty() {
        return Err(PyValueError::new_err("nodes must not be empty"));
    }
    if node_bytes.len() % BVH_NODE_BYTES != 0 {
        return Err(PyValueError::new_err(format!(
            "nodes must be a multiple of {BVH_NODE_BYTES} bytes, got {}",
            node_bytes.len()
        )));
    }
    let node_count = node_bytes.len() / BVH_NODE_BYTES;

    let triangle_count = mesh.triangle_count();
    let tri = tri_indices.as_slice()?;
    if tri.len() as u32 != triangle_count {
        return Err(PyValueError::new_err(format!(
            "tri_indices must have {triangle_count} entries, got {}",
            tri.len()
        )));
    }
    for &index in tri {
        if index >= triangle_count {
            return Err(PyValueError::new_err(
                "tri_indices must be a permutation of the triangle indices",
            ));
        }
    }

    // `pod_read_unaligned` reads the exact `#[repr(C)]` fields, so this file
    // never restates the node byte offsets.
    let mut parsed = Vec::with_capacity(node_count);
    for chunk in node_bytes.chunks_exact(BVH_NODE_BYTES) {
        parsed.push(bytemuck::pod_read_unaligned::<BvhNode>(chunk));
    }

    let mut referenced = vec![false; node_count];
    for (index, node) in parsed.iter().enumerate() {
        if !node.aabb_min.iter().all(|value| value.is_finite())
            || !node.aabb_max.iter().all(|value| value.is_finite())
        {
            return Err(PyValueError::new_err(format!(
                "node {index} has a non-finite bounding box"
            )));
        }
        if node.is_leaf() {
            let end = u64::from(node.left) + u64::from(node.right);
            if end > u64::from(triangle_count) {
                return Err(PyValueError::new_err(format!(
                    "leaf node {index} covers triangles {}..{} outside {triangle_count}",
                    node.left, end
                )));
            }
        } else {
            for child in [node.left, node.right] {
                if child as usize >= node_count {
                    return Err(PyValueError::new_err(format!(
                        "internal node {index} references child {child} outside {node_count}"
                    )));
                }
                referenced[child as usize] = true;
            }
        }
    }

    let roots: Vec<usize> = (0..node_count).filter(|&index| !referenced[index]).collect();
    if roots.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "BVH must have exactly one root node, found {}",
            roots.len()
        )));
    }

    let mut visited = vec![false; node_count];
    let mut stack = vec![roots[0]];
    visited[roots[0]] = true;
    while let Some(index) = stack.pop() {
        let node = &parsed[index];
        if node.is_leaf() {
            continue;
        }
        for child in [node.left, node.right] {
            let child = child as usize;
            if visited[child] {
                return Err(PyValueError::new_err(
                    "BVH contains a cycle or a node with two parents",
                ));
            }
            visited[child] = true;
            stack.push(child);
        }
    }
    if let Some(missing) = visited.iter().position(|&seen| !seen) {
        return Err(PyValueError::new_err(format!(
            "node {missing} is not reachable from the root"
        )));
    }

    // This BVH is only an upload input: `build_mesh_atlas` reads `nodes`,
    // `tri_indices` and `node_count`, never the build statistics. Fill the two
    // counts the atlas needs and leave the rest default rather than add f32
    // narrowing sites the world-coordinate gate would have to review.
    let build_stats = BuildStats {
        triangle_count,
        node_count: node_count as u32,
        ..Default::default()
    };

    let world_aabb = parsed[roots[0]].aabb();

    Ok(BvhCPU {
        nodes: parsed,
        tri_indices: tri.to_vec(),
        world_aabb,
        build_stats,
    })
}

/// Total bytes held by the four tracked atlas buffers.
fn atlas_bytes(atlas: &MeshAtlas) -> u64 {
    atlas.vertex_buffer.size()
        + atlas.index_buffer.size()
        + atlas.bvh_buffer.size()
        + atlas.descs_buffer.size()
}

/// Handle to a mesh and BVH uploaded into tracked GPU buffers.
#[pyclass(module = "forge3d._forge3d", name = "GpuMeshHandle")]
pub struct GpuMeshHandle {
    atlas: MeshAtlas,
    vertex_count: u32,
    triangle_count: u32,
    node_count: u32,
    world_aabb_min: [f32; 3],
    world_aabb_max: [f32; 3],
}

#[pymethods]
impl GpuMeshHandle {
    /// Bytes of tracked GPU memory held by the mesh, index, BVH and descriptor buffers.
    #[getter]
    fn gpu_memory_bytes(&self) -> u64 {
        atlas_bytes(&self.atlas)
    }

    #[getter]
    fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    #[getter]
    fn triangle_count(&self) -> u32 {
        self.triangle_count
    }

    #[getter]
    fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Number of BLAS descriptors in the uploaded atlas (one per mesh).
    #[getter]
    fn blas_desc_count(&self) -> u32 {
        self.atlas.desc_count
    }

    #[getter]
    fn world_aabb_min(&self) -> (f32, f32, f32) {
        (
            self.world_aabb_min[0],
            self.world_aabb_min[1],
            self.world_aabb_min[2],
        )
    }

    #[getter]
    fn world_aabb_max(&self) -> (f32, f32, f32) {
        (
            self.world_aabb_max[0],
            self.world_aabb_max[1],
            self.world_aabb_max[2],
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuMeshHandle(vertices={}, triangles={}, nodes={}, gpu_memory_bytes={})",
            self.vertex_count,
            self.triangle_count,
            self.node_count,
            self.gpu_memory_bytes()
        )
    }
}

/// Build a median-split CPU BVH for a triangle mesh.
///
/// Returns the real node count, depth, triangle order and packed node bytes from
/// `accel::cpu_bvh::build_bvh_cpu`.
#[pyfunction]
#[pyo3(signature = (vertices, indices, method = "median", max_leaf_size = 4))]
pub(crate) fn mesh_build_bvh_cpu(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f32>,
    indices: PyReadonlyArray2<'_, u32>,
    method: &str,
    max_leaf_size: u32,
) -> PyResult<PyObject> {
    if !method.eq_ignore_ascii_case("median") {
        return Err(PyValueError::new_err(format!(
            "unsupported BVH method '{method}'; only 'median' is supported"
        )));
    }
    if max_leaf_size == 0 {
        return Err(PyValueError::new_err("max_leaf_size must be at least 1"));
    }

    let mesh = mesh_cpu_from_py(&vertices, &indices)?;
    let options = BuildOptions {
        max_leaf_size,
        method: BuildMethod::MedianSplit,
    };
    let bvh = py
        .allow_threads(move || build_bvh_cpu(&mesh, &options))
        .map_err(|error| PyRuntimeError::new_err(format!("BVH build failed: {error}")))?;

    let node_bytes = bytemuck::cast_slice::<BvhNode, u8>(&bvh.nodes).to_vec();
    let stats = &bvh.build_stats;

    let dict = PyDict::new_bound(py);
    dict.set_item("method", "median")?;
    dict.set_item("triangle_count", stats.triangle_count)?;
    dict.set_item("node_count", bvh.node_count())?;
    dict.set_item("leaf_count", stats.leaf_count)?;
    dict.set_item("internal_count", stats.internal_count)?;
    dict.set_item("max_depth", stats.max_depth)?;
    dict.set_item("avg_leaf_size", stats.avg_leaf_size)?;
    dict.set_item("build_time_ms", stats.build_time_ms)?;
    dict.set_item("memory_usage_bytes", stats.memory_usage_bytes)?;
    dict.set_item(
        "world_aabb_min",
        vec![
            bvh.world_aabb.min[0],
            bvh.world_aabb.min[1],
            bvh.world_aabb.min[2],
        ],
    )?;
    dict.set_item(
        "world_aabb_max",
        vec![
            bvh.world_aabb.max[0],
            bvh.world_aabb.max[1],
            bvh.world_aabb.max[2],
        ],
    )?;
    dict.set_item("tri_indices", PyArray1::from_vec_bound(py, bvh.tri_indices.clone()))?;
    dict.set_item("nodes", PyArray1::<u8>::from_vec_bound(py, node_bytes))?;
    Ok(dict.into())
}

/// Upload a mesh and its BVH into tracked GPU buffers.
///
/// The BVH is rebuilt from the packed node bytes so a corrupt or mismatched tree
/// fails before any buffer is allocated.
#[pyfunction]
pub(crate) fn mesh_upload_gpu(
    vertices: PyReadonlyArray2<'_, f32>,
    indices: PyReadonlyArray2<'_, u32>,
    tri_indices: PyReadonlyArray1<'_, u32>,
    nodes: PyReadonlyArray1<'_, u8>,
) -> PyResult<GpuMeshHandle> {
    let mesh = mesh_cpu_from_py(&vertices, &indices)?;
    let bvh = bvh_cpu_from_py(&nodes, &tri_indices, &mesh)?;

    let vertex_count = mesh.vertex_count();
    let triangle_count = mesh.triangle_count();
    let node_count = bvh.node_count();
    let world_aabb_min = bvh.world_aabb.min;
    let world_aabb_max = bvh.world_aabb.max;

    let device = Arc::clone(&crate::core::gpu::try_ctx()?.device);
    let atlas = build_mesh_atlas(&device, &[(mesh, bvh)])
        .map_err(|error| PyRuntimeError::new_err(format!("mesh GPU upload failed: {error}")))?;

    Ok(GpuMeshHandle {
        atlas,
        vertex_count,
        triangle_count,
        node_count,
        world_aabb_min,
        world_aabb_max,
    })
}
