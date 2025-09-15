// src/shaders/svgf_reproject.wgsl
// WGSL compute pass stub for SVGF temporal reprojection using motion vectors
// Provides a placeholder kernel to document bindings and pipeline stages for future GPU implementation
// RELEVANT FILES:src/shaders/svgf_variance.wgsl,src/shaders/svgf_atrous.wgsl,src/denoise/svgf/pipelines.rs

// Bind group 0:
//  binding 0: history_color      (texture_storage_2d<rgba16float, read>)
//  binding 1: history_moments    (texture_storage_2d<rg16float,   read>)
//  binding 2: motion_vectors     (texture_2d<f16/f32>, sampled)
//  binding 3: current_depth      (texture_2d<f32>, sampled)
//  binding 4: reprojected_color  (texture_storage_2d<rgba16float, write>)
//  binding 5: reprojected_moment (texture_storage_2d<rg16float,   write>)
//  binding 6: params             (uniform)

struct Params {
  width: u32,
  height: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(6) var<uniform> params: Params;

// Placeholder: no-op write to keep pipeline shape valid for future wiring
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  // Intentionally left blank in stub
}

