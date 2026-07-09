use super::types::{BlasDesc, GpuMesh, GpuVertex, MeshAtlas};
use crate::accel::cpu_bvh::{BvhCPU, BvhNode, MeshCPU};
use crate::core::resource_tracker::tracked_create_buffer_init;
use anyhow::Result;
use wgpu::{BufferUsages, Device, Queue};

/// BVH node padded to the WGSL `BvhNode` layout consumed by the wavefront
/// kernels (pt_intersect/pt_shadow): vec3 members force a 16-byte struct
/// alignment there, so the array stride is 48 bytes — not the 40 bytes of the
/// packed CPU `BvhNode`. Field offsets: aabb_min@0, left@12, aabb_max@16,
/// right@28, flags@32.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBvhNode48 {
    aabb_min: [f32; 3],
    left: u32,
    aabb_max: [f32; 3],
    right: u32,
    flags: u32,
    _pad: [u32; 3],
}

impl From<&BvhNode> for GpuBvhNode48 {
    fn from(n: &BvhNode) -> Self {
        Self {
            aabb_min: n.aabb_min,
            left: n.left,
            aabb_max: n.aabb_max,
            right: n.right,
            flags: n.flags,
            _pad: [0; 3],
        }
    }
}

/// Build a mesh atlas from multiple MeshCPU/BvhCPU pairs.
/// The atlas concatenates all vertices, indices (triangles), and BVH nodes into three buffers,
/// and creates a descriptor table describing offsets/counts for each BLAS.
pub fn build_mesh_atlas(device: &Device, items: &[(MeshCPU, BvhCPU)]) -> anyhow::Result<MeshAtlas> {
    if items.is_empty() {
        anyhow::bail!("build_mesh_atlas: items must be non-empty");
    }

    // Concatenate vertices, indices, nodes
    let mut all_vertices: Vec<GpuVertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new(); // triangle index stream (3 per triangle)
    let mut all_nodes: Vec<GpuBvhNode48> = Vec::new();
    let mut descs: Vec<BlasDesc> = Vec::with_capacity(items.len());

    let mut vtx_ofs: u32 = 0;
    let mut tri_ofs: u32 = 0; // measured in triangles, not u32s
    let mut node_ofs: u32 = 0;

    for (mesh, bvh) in items.iter() {
        // Vertices
        let start_vtx = vtx_ofs;
        let vtx_count = mesh.vertex_count();
        all_vertices.extend(mesh.vertices.iter().copied().map(GpuVertex::from));
        vtx_ofs += vtx_count;

        // Indices (triangles) in the exact order referenced by BVH leaves
        // BVH nodes refer to triangle ranges by index into bvh.tri_indices
        let start_tri = tri_ofs;
        let tri_count = mesh.triangle_count();
        for &tri_idx in &bvh.tri_indices {
            let tri = mesh.indices[tri_idx as usize];
            all_indices.extend_from_slice(&[tri[0], tri[1], tri[2]]);
        }
        tri_ofs += tri_count;

        // BVH nodes (padded to the 48-byte WGSL stride)
        let start_node = node_ofs;
        let node_count = bvh.node_count();
        all_nodes.extend(bvh.nodes.iter().map(GpuBvhNode48::from));
        node_ofs += node_count;

        // Descriptor entry
        descs.push(BlasDesc {
            node_offset: start_node,
            node_count,
            tri_offset: start_tri,
            tri_count,
            vtx_offset: start_vtx,
            vtx_count,
            _pad0: 0,
            _pad1: 0,
        });
    }

    // Create GPU buffers
    let vertex_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("atlas-vertex-buffer"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    let index_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("atlas-index-buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    let bvh_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("atlas-bvh-buffer"),
            contents: bytemuck::cast_slice(&all_nodes),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    let descs_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("atlas-blas-descs"),
            contents: bytemuck::cast_slice(&descs),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    Ok(MeshAtlas {
        vertex_buffer,
        index_buffer,
        bvh_buffer,
        descs_buffer,
        desc_count: descs.len() as u32,
    })
}

/// Upload mesh and BVH data to GPU buffers
/// This is the main integration point for getting CPU mesh data into GPU format
pub fn upload_mesh_and_bvh(
    device: &Device,
    _queue: &Queue,
    mesh: &MeshCPU,
    bvh: &BvhCPU,
) -> Result<GpuMesh> {
    // Validate input data
    if mesh.vertices.is_empty() {
        anyhow::bail!("Cannot upload empty mesh");
    }
    if mesh.indices.is_empty() {
        anyhow::bail!("Cannot upload mesh with no triangles");
    }
    if bvh.nodes.is_empty() {
        anyhow::bail!("Cannot upload empty BVH");
    }

    // Convert vertices to GPU format
    let gpu_vertices: Vec<GpuVertex> = mesh
        .vertices
        .iter()
        .map(|&pos| GpuVertex::from(pos))
        .collect();

    // Flatten triangle indices to u32 array
    let flat_indices: Vec<u32> = mesh
        .indices
        .iter()
        .flat_map(|tri| tri.iter().copied())
        .collect();

    // Create vertex buffer
    let vertex_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    // Create index buffer
    let index_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(&flat_indices),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    // Create BVH buffer
    let bvh_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("Mesh BVH Buffer"),
            contents: bytemuck::cast_slice(&bvh.nodes),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
    )?;

    log::info!(
        "Uploaded mesh to GPU: {} vertices, {} triangles, {} BVH nodes",
        gpu_vertices.len(),
        mesh.indices.len(),
        bvh.nodes.len()
    );
    log::info!(
        "GPU memory usage: vertices={} bytes, indices={} bytes, BVH={} bytes",
        vertex_buffer.size(),
        index_buffer.size(),
        bvh_buffer.size()
    );

    Ok(GpuMesh {
        vertex_buffer,
        index_buffer,
        bvh_buffer,
        vertex_count: gpu_vertices.len() as u32,
        triangle_count: mesh.triangle_count(),
        node_count: bvh.node_count(),
        world_aabb: bvh.world_aabb,
        build_stats: bvh.build_stats.clone(),
    })
}
