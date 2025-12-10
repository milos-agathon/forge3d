mod bind_groups;
mod types;
mod upload;
mod validation;

#[cfg(test)]
mod tests;

pub use bind_groups::{
    create_empty_mesh_buffers, create_mesh_bind_group, create_mesh_bind_group_layout,
};
pub use types::{BlasDesc, GpuMesh, GpuVertex, MeshAtlas, MeshStats};
pub use upload::{build_mesh_atlas, upload_mesh_and_bvh};
pub use validation::{compute_mesh_stats, validate_mesh, MeshBuilder};
