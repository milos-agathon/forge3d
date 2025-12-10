use super::types::GpuVertex;
use super::validation::{compute_mesh_stats, validate_mesh, MeshBuilder};
use crate::accel::cpu_bvh::MeshCPU;

#[test]
fn test_gpu_vertex_layout() {
    // Verify GPU vertex layout
    assert_eq!(std::mem::size_of::<GpuVertex>(), 16); // 4 floats
    assert_eq!(std::mem::align_of::<GpuVertex>(), 4);
}

#[test]
fn test_mesh_builder_triangle() {
    let mesh = MeshBuilder::triangle();
    assert_eq!(mesh.vertex_count(), 3);
    assert_eq!(mesh.triangle_count(), 1);
    validate_mesh(&mesh).unwrap();
}

#[test]
fn test_mesh_builder_cube() {
    let mesh = MeshBuilder::cube();
    assert_eq!(mesh.vertex_count(), 8);
    assert_eq!(mesh.triangle_count(), 12);
    validate_mesh(&mesh).unwrap();
}

#[test]
fn test_mesh_validation() {
    // Valid mesh
    let valid_mesh = MeshBuilder::triangle();
    assert!(validate_mesh(&valid_mesh).is_ok());

    // Empty mesh
    let empty_mesh = MeshCPU::new(vec![], vec![]);
    assert!(validate_mesh(&empty_mesh).is_err());

    // Invalid indices
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let indices = vec![[0, 1, 2]]; // Index 2 is out of bounds
    let invalid_mesh = MeshCPU::new(vertices, indices);
    assert!(validate_mesh(&invalid_mesh).is_err());
}

#[test]
fn test_mesh_stats() {
    let mesh = MeshBuilder::cube();
    let stats = compute_mesh_stats(&mesh);

    assert_eq!(stats.vertex_count, 8);
    assert_eq!(stats.triangle_count, 12);
    assert!(stats.world_aabb.is_valid());
    assert!(stats.average_triangle_area > 0.0);
    assert!(stats.memory_usage_bytes > 0);
}
